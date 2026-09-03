"""Patch operations for ``SimMesh`` declarations.

SimMesh declarations are patched like studies: located by stable index or name
(:func:`~cadjoint.viewer.source_map.declarations.locate_mesh_statements`),
edited by span surgery on their constructor call.

The placement rule in :func:`add_mesh` is the one thing that is not obvious:
a new mesh goes after the last existing mesh, otherwise **before the first
study**, otherwise after the ``scene`` assignment — a study can only reference
meshes declared above it, so a mesh appended at the end would be unusable.

Deletion is refused while anything still points at the mesh, through either of
its two reference forms: the assignment variable, or its ``name`` as a
``mesh="..."`` string argument on a study.

``_MESH_FIELDS`` is the constructor's field order, used to find positionally
written arguments; ``method`` is written as a string literal and validated
against :class:`cadjoint.enums.MeshMethod`.  Add an operation here when it
edits a mesh declaration.
"""

from __future__ import annotations

import ast

from cadjoint.enums import MeshMethod, listed, values
from cadjoint.viewer.patch.edits import (
    _after_statement,
    _delete_statement,
    _ensure_import,
    _module_names,
    _name_references,
    _named_by_keyword,
    _rewrite_call_argument,
    _set_keyword_expression,
    _validate,
)
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.patch.format import _exact_number, _format_study_argument
from cadjoint.viewer.patch.resolvers import _located_mesh
from cadjoint.viewer.patch.scene import _domain_expression, _scene_assignment
from cadjoint.viewer.patch.studies import _checked_box_pair
from cadjoint.viewer.source_map import locate_mesh_statements, locate_study_statements
from cadjoint.viewer.source_map.nodes import _line_offsets, _node_span

# Constructor field order, for resolving positionally written arguments;
# `name` is excluded from editing.
_MESH_FIELDS = ("name", "resolution", "domain", "bounds", "size", "padding")
_MESH_NUMERIC_ARGUMENTS = ("resolution", "bounds", "size", "padding")
#: The accepted spellings of :class:`cadjoint.enums.MeshMethod`.
_MESH_METHODS = values(MeshMethod)


def add_mesh(source: str, name: str | None = None) -> str:
    """Declare a new simulation mesh at the end of the scene program.

    Writes a kind-less ``meshN = SimMesh(name=..., resolution=20)``
    declaration (importing ``SimMesh`` from ``cadjoint.fem`` beside it) —
    after the last existing mesh, otherwise before the first study (a study
    can only reference meshes declared above it), otherwise after the
    ``scene`` assignment.

    Args:
        source: The program text.
        name: Optional mesh display name; the generated variable name is
            used when omitted.

    Returns:
        The patched source.

    Raises:
        PatchError: On a duplicate name or a program without a ``scene``
            assignment to anchor the mesh.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise PatchError("Mesh `name` must be a non-empty string.")
    meshes = locate_mesh_statements(source) or []
    taken_names = {mesh.name for mesh in meshes if mesh.name is not None}
    if name is not None and name in taken_names:
        raise PatchError(f"A mesh named {name!r} already exists.")
    taken = _module_names(tree)
    index = 1
    while f"mesh{index}" in taken or f"mesh{index}" in taken_names:
        index += 1
    variable = f"mesh{index}"
    mesh_name = name if name is not None else variable

    studies = locate_study_statements(source) or []
    if meshes:
        insert = _after_statement(source, meshes[-1].statement)
    elif studies:
        insert = _line_offsets(source)[studies[0].statement.lineno - 1]
    else:
        anchor = _scene_assignment(tree)
        if anchor is None:
            raise PatchError(
                "Add a `scene = ...` assignment before declaring meshes from the viewer."
            )
        insert = _after_statement(source, anchor)
    statement = f"{variable} = SimMesh(name={mesh_name!r}, resolution=20)\n"
    patched = source[:insert] + statement + source[insert:]
    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.fem", "SimMesh", prefer_offset=insert
    )
    return _validate(patched)


def delete_mesh(source: str, mesh) -> str:
    """Remove one mesh declaration, refusing while anything references it.

    A mesh is referenced either through its assignment variable (used by a
    study's ``mesh=`` or anywhere else in the program) or through its name
    as a ``mesh="..."`` string argument.
    """
    located = _located_mesh(source, mesh)
    tree = ast.parse(source)
    if located.variable is not None:
        uses = _name_references(tree, located.variable, located.statement)
        if uses:
            raise PatchError(
                f"`{located.variable}` is used elsewhere in the program, so it cannot be "
                "deleted from the viewer. Remove those uses first."
            )
    if located.name is not None and _named_by_keyword(tree, "mesh", located.name):
        raise PatchError(
            f"Mesh {located.name!r} is referenced by a study, so it cannot be deleted "
            "from the viewer. Point the study at another mesh first."
        )
    return _validate(_delete_statement(source, located.statement))


def set_mesh_value(source: str, mesh, argument, value) -> str:
    """Edit one SimMesh argument in place (or add it when absent).

    Args:
        source: The program text.
        mesh: Mesh reference — payload index, name, or variable.
        argument: ``resolution``, ``bounds``, ``size``, ``padding`` (numeric,
            written with exact float ``repr``; ``resolution`` stays
            integral), ``domain`` (the variable name of a named scene
            object), or ``method`` (one of ``hex``/``tet4``/``tet10``,
            written as a string literal).
        value: The new number(s) or name.

    Returns:
        The patched source.

    Raises:
        PatchError: On an unknown argument, an invalid value, or a target
            that is not an editable literal.
    """
    located = _located_mesh(source, mesh)
    if argument == "domain":
        # A domain is a *name*, not a literal: the existing keyword's span is
        # replaced whatever expression it holds, the way a study's is.
        expression = _domain_expression(source, value, located.statement.lineno, "mesh")
        existing = next(
            (keyword.value for keyword in located.call.keywords if keyword.arg == "domain"),
            located.call.args[2] if len(located.call.args) > 2 else None,
        )
        if existing is None:
            return _set_keyword_expression(source, located.call, "domain", expression)
        offsets = _line_offsets(source)
        start, end = _node_span(source, offsets, existing)
        return _validate(source[:start] + expression + source[end:])
    if argument == "method":
        if value not in _MESH_METHODS:
            raise PatchError(f"Mesh `method` must be one of: {listed(MeshMethod)}.")
        # ``str`` first: a MeshMethod member reprs as the member, not the
        # literal the program should carry.
        return _set_keyword_expression(source, located.call, "method", repr(str(value)))
    if not isinstance(argument, str) or argument not in _MESH_NUMERIC_ARGUMENTS:
        allowed = ", ".join((*_MESH_NUMERIC_ARGUMENTS, "domain", "method"))
        raise PatchError(f"A mesh's editable arguments are: {allowed}.")
    if argument == "padding":
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
            raise PatchError("`padding` must be a non-negative number.")
        expression = _exact_number(value)
    else:
        _checked_box_pair(located.call, _MESH_FIELDS, argument, "mesh")
        expression = _format_study_argument(argument, value)
    return _rewrite_call_argument(source, located.call, _MESH_FIELDS, argument, expression, "mesh")
