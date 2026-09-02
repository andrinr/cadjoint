"""Patch operations for simulation studies and their boundary conditions.

Studies are first-class code citizens (:mod:`cadjoint.fem.study`): the viewer
edits them by patching their constructor source, exactly like constraints.
Everything a study panel can do — declare one, delete it, add or remove a
boundary condition, retune a coefficient, point it at a mesh or a domain —
is an edit to that one constructor call.

The tables at the top of the module are the contract:

- ``_STUDY_CLASSES``/``_STUDY_DEFAULTS`` decide what a new declaration looks
  like;
- ``_STUDY_BC_CLASSES`` and ``_STUDY_KIND_BC_TYPES`` decide which boundary
  conditions a thermal or elastic study will accept, so an incompatible BC is
  refused with a message naming the ones that fit;
- ``_STUDY_FIELDS`` is the constructor's field order, which is how a
  positionally written argument is found and rewritten.

Values typed in the viewer are written with exact float ``repr`` so they
round-trip, and node selections are rendered through
:func:`~cadjoint.viewer.patch.format._selection_source`, which refuses anything
the runtime could not rebuild.  Setting ``mesh`` also strips the study's own
``resolution``/``bounds``/``size``/``domain`` arguments: the contract keeps
meshing intent on the ``SimMesh``, and leaving both would be ambiguous.

Add an operation here when it edits a study declaration.  Mesh declarations
live in :mod:`cadjoint.viewer.patch.meshes`.
"""

from __future__ import annotations

import ast

from cadjoint.viewer.patch.edits import (
    _after_statement,
    _argument_span,
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
from cadjoint.viewer.patch.format import (
    _exact_value,
    _format_study_argument,
    _selection_source,
)
from cadjoint.viewer.patch.resolvers import _located_study, _located_study_bc
from cadjoint.viewer.patch.scene import _domain_expression, _scene_assignment
from cadjoint.viewer.source_map import (
    StudyStatement,
    locate_mesh_statements,
    locate_study_statements,
)
from cadjoint.viewer.source_map.nodes import (
    _called_name,
    _editable_value_node,
    _line_offsets,
    _node_span,
)

_STUDY_CLASSES = {"thermal": "ThermalStudy", "elastic": "ElasticStudy"}
_STUDY_DEFAULTS = {"thermal": "conductivity=1.0", "elastic": "youngs=200.0, poisson=0.3"}
_STUDY_BC_CLASSES = {
    "dirichlet": ("Dirichlet", "value"),
    "heat_flux": ("HeatFlux", "flux"),
    "fixed": ("Fixed", None),
    "traction": ("Traction", "vector"),
}
_STUDY_KIND_BC_TYPES = {
    "thermal": ("dirichlet", "heat_flux"),
    "elastic": ("fixed", "traction"),
}
_BC_CLASS_VALUE_KEYWORDS = {
    "Dirichlet": "value",
    "HeatFlux": "flux",
    "Fixed": None,
    "Traction": "vector",
}
# Constructor field order per kind, for resolving positionally written
# arguments; `name` and `bcs` are excluded from numeric-kwarg editing.
_STUDY_FIELDS = {
    "thermal": ("name", "resolution", "conductivity", "bcs", "source", "bounds", "size"),
    "elastic": ("name", "resolution", "youngs", "poisson", "bcs", "bounds", "size"),
}


def add_study(source: str, kind: str, name: str | None = None) -> str:
    """Declare a new simulation study at the end of the scene program.

    Appends a ``ThermalStudy``/``ElasticStudy`` constructor after the last
    existing study (or after the ``scene`` assignment when there is none) and
    imports the constructor from ``cadjoint.fem`` beside it, keeping every line
    above the insertion untouched.

    Args:
        source: The program text.
        kind: ``thermal`` or ``elastic``.
        name: Optional study display name; the generated variable name is
            used when omitted.

    Returns:
        The patched source.

    Raises:
        PatchError: On an unknown kind, a duplicate name, or a program
            without a ``scene`` assignment to anchor the study.
    """
    symbol = _STUDY_CLASSES.get(kind)
    if symbol is None:
        raise PatchError("Study `kind` must be `thermal` or `elastic`.")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    studies = locate_study_statements(source) or []
    taken_names = {study.name for study in studies if study.name is not None}
    if name is not None and name in taken_names:
        raise PatchError(f"A study named {name!r} already exists.")
    taken = _module_names(tree)
    index = 1
    while f"study{index}" in taken or f"study{index}" in taken_names:
        index += 1
    variable = f"study{index}"
    study_name = name if name is not None else variable

    if studies:
        anchor = studies[-1].statement
    else:
        anchor = _scene_assignment(tree)
        if anchor is None:
            raise PatchError(
                "Add a `scene = ...` assignment before declaring studies from the viewer."
            )
    statement = (
        f"{variable} = {symbol}(name={study_name!r}, resolution=20, "
        f"{_STUDY_DEFAULTS[kind]}, bcs=[])\n"
    )
    insert = _after_statement(source, anchor)
    patched = source[:insert] + statement + source[insert:]
    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.fem", symbol, prefer_offset=insert
    )
    return _validate(patched)


def delete_study(source: str, study) -> str:
    """Remove one study declaration, identified by payload index or name.

    Refuses on both ways a study can still be needed: the variable it is
    bound to being read somewhere else, and — the form an ``Optimization``
    actually uses — its literal name appearing as another declaration's
    ``study=`` argument. Deleting past either leaves a program that no
    longer runs, which is exactly what the compile after the patch would
    report and the user could not undo from the viewer.
    """
    located = _located_study(source, study)
    tree = ast.parse(source)
    if located.variable is not None:
        uses = _name_references(tree, located.variable, located.statement)
        if uses:
            raise PatchError(
                f"`{located.variable}` is used elsewhere in the program, so it cannot be "
                "deleted from the viewer. Remove those uses first."
            )
    if located.name is not None and _named_by_keyword(tree, "study", located.name):
        raise PatchError(
            f"Study {located.name!r} is referenced by an optimization, so it cannot be "
            "deleted from the viewer. Point the optimization at another study first."
        )
    return _validate(_delete_statement(source, located.statement))


def add_study_bc(source: str, study, bc_type: str, selection, value=None) -> str:
    """Append a boundary condition to a study's literal ``bcs`` list.

    Writes literal source such as
    ``Dirichlet(Nodes.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), value=300.0)``,
    with viewer-typed values formatted via exact float ``repr``.  The BC
    class and ``Nodes`` are imported from ``cadjoint.fem`` beside the study so
    every line above it stays untouched.

    Args:
        source: The program text.
        study: Study reference — payload index or name.
        bc_type: ``dirichlet``, ``heat_flux``, ``fixed``, or ``traction``.
        selection: Serializable node-selection description dict.
        value: Scalar for ``dirichlet``/``heat_flux``, 3-vector for
            ``traction``; ``fixed`` takes none.

    Returns:
        The patched source.

    Raises:
        PatchError: On an unknown or kind-incompatible BC type, an invalid
            selection or value, or a ``bcs`` argument that is not an
            editable literal list.
    """
    located = _located_study(source, study)
    if bc_type not in _STUDY_BC_CLASSES:
        allowed = ", ".join(_STUDY_BC_CLASSES)
        raise PatchError(f"`bc_type` must be one of: {allowed}.")
    if bc_type not in _STUDY_KIND_BC_TYPES[located.kind]:
        allowed = ", ".join(_STUDY_KIND_BC_TYPES[located.kind])
        raise PatchError(
            f"A {located.kind} study accepts {allowed} boundary conditions, not `{bc_type}`."
        )
    symbol, value_keyword = _STUDY_BC_CLASSES[bc_type]
    nodes_source = _selection_source(selection)
    if value_keyword is None:
        if value is not None:
            raise PatchError("A `fixed` boundary condition takes no value.")
        bc_source = f"{symbol}({nodes_source})"
    else:
        if bc_type == "traction":
            if not (isinstance(value, (list, tuple)) and len(value) == 3):
                raise PatchError("A `traction` boundary condition needs `value` as three numbers.")
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PatchError(f"A `{bc_type}` boundary condition needs a numeric `value`.")
        bc_source = f"{symbol}({nodes_source}, {value_keyword}={_exact_value(value)})"

    offsets = _line_offsets(source)
    anchor_line = located.statement.lineno
    if located.bcs is not None:
        anchor_line = min(anchor_line, located.bcs.lineno)
    import_offset = offsets[anchor_line - 1]

    if located.bcs is not None and located.bc_spans:
        _, end = located.bc_spans[-1]
        patched = source[:end] + f", {bc_source}" + source[end:]
    elif located.bcs is not None:
        _, end = located.bcs_span
        patched = source[: end - 1] + bc_source + source[end - 1 :]
    else:
        if any(keyword.arg == "bcs" for keyword in located.call.keywords):
            raise PatchError("The study's `bcs` argument is not an editable literal list.")
        patched = _set_keyword_expression(source, located.call, "bcs", f"[{bc_source}]")

    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.fem", symbol, prefer_offset=import_offset
    )
    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.fem", "Nodes", prefer_offset=import_offset
    )
    return _validate(patched)


def delete_study_bc(source: str, study, bc) -> str:
    """Remove one boundary condition from a study's literal ``bcs`` list."""
    located = _located_study(source, study)
    _located_study_bc(located, bc)
    spans = located.bc_spans
    start, end = spans[bc]
    if bc < len(spans) - 1:
        # Swallow the separator up to the next element.
        end = spans[bc + 1][0]
    elif bc > 0:
        # Last element: swallow the separator after the previous one.
        start = spans[bc - 1][1]
    return _validate(source[:start] + source[end:])


def _set_study_bc_value(source: str, located: StudyStatement, bc, value) -> str:
    """Rewrite the numeric payload of one boundary condition in place."""
    element, _ = _located_study_bc(located, bc)
    class_name = _called_name(element) or ""
    if class_name not in _BC_CLASS_VALUE_KEYWORDS or not isinstance(element, ast.Call):
        raise PatchError("This boundary condition is not an editable constructor call.")
    value_keyword = _BC_CLASS_VALUE_KEYWORDS[class_name]
    if value_keyword is None:
        raise PatchError("A `Fixed` boundary condition has no value to edit.")
    if class_name == "Traction":
        if not (isinstance(value, (list, tuple)) and len(value) == 3):
            raise PatchError("A `Traction` boundary condition needs `value` as three numbers.")
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PatchError(f"A `{class_name}` boundary condition needs a numeric `value`.")
    target = next(
        (keyword.value for keyword in element.keywords if keyword.arg == value_keyword),
        element.args[1] if len(element.args) > 1 else None,
    )
    if target is None:
        raise PatchError("The boundary condition has no value argument to rewrite.")
    tree = ast.parse(source)
    literal = _editable_value_node(target, tree)
    if literal is None:
        raise PatchError("The boundary-condition value is not an editable literal.")
    offsets = _line_offsets(source)
    start, end = _node_span(source, offsets, literal)
    return _validate(source[:start] + _exact_value(value) + source[end:])


def _set_study_mesh(source: str, located: StudyStatement, value) -> str:
    """Point a study at a declared SimMesh (removing conflicting keywords).

    The mesh is referenced by its declared ``name`` (written as a string
    literal) or by its assignment variable (written as a bare name).  The
    study contract keeps meshing intent on the SimMesh, so any
    ``resolution``/``bounds``/``size``/``domain`` arguments the study still
    carries are removed as part of the same edit.
    """
    if not isinstance(value, str) or not value.strip():
        raise PatchError("`mesh` needs the name of a declared SimMesh.")
    meshes = locate_mesh_statements(source) or []
    by_name = [mesh for mesh in meshes if mesh.name == value]
    by_variable = [mesh for mesh in meshes if mesh.variable == value]
    if len(by_name) == 1:
        expression = repr(value)
    elif not by_name and len(by_variable) == 1:
        expression = value
    else:
        declared = ", ".join(
            repr(mesh.name or mesh.variable or f"#{mesh.index}") for mesh in meshes
        )
        raise PatchError(
            f"No single SimMesh named {value!r}; the program declares: {declared or 'none'}."
        )

    offsets = _line_offsets(source)
    removals = [
        _argument_span(source, offsets, keyword)
        for keyword in located.call.keywords
        if keyword.arg in {"resolution", "bounds", "size", "domain"}
    ]
    if len(located.call.args) > 1:  # positional resolution
        removals.append(_argument_span(source, offsets, located.call.args[1]))

    def whole_line(start: int, end: int) -> tuple[int, int]:
        """Swallow the line when the removal leaves it blank (multiline calls)."""
        line_start = source.rfind("\n", 0, start) + 1
        if not source[line_start:start].strip() and source[end : end + 1] == "\n":
            return line_start, end + 1
        return start, end

    patched = source
    for start, end in sorted((whole_line(start, end) for start, end in removals), reverse=True):
        patched = patched[:start] + patched[end:]
    relocated = _located_study(patched, located.index)
    if relocated.mesh_span is not None:
        start, end = relocated.mesh_span
        return _validate(patched[:start] + expression + patched[end:])
    return _validate(_set_keyword_expression(patched, relocated.call, "mesh", expression))


def _set_study_domain(source: str, located: StudyStatement, value) -> str:
    """Point an implicit-mesh study at a named scene object as its domain."""
    if any(keyword.arg == "mesh" for keyword in located.call.keywords):
        raise PatchError(
            "This study solves on a SimMesh; set the mesh's `domain` instead (set_mesh_value)."
        )
    expression = _domain_expression(source, value, located.statement.lineno, "study")
    if located.domain_span is not None:
        start, end = located.domain_span
        return _validate(source[:start] + expression + source[end:])
    return _validate(_set_keyword_expression(source, located.call, "domain", expression))


def _set_study_argument(source: str, located: StudyStatement, argument, value) -> str:
    """Rewrite one study keyword in place (or add it when absent)."""
    if argument == "mesh":
        return _set_study_mesh(source, located, value)
    if argument == "domain":
        return _set_study_domain(source, located, value)
    fields = _STUDY_FIELDS[located.kind]
    if not isinstance(argument, str) or argument not in fields or argument in {"name", "bcs"}:
        allowed = ", ".join(field for field in fields if field not in {"name", "bcs"})
        raise PatchError(
            f"A {located.kind} study's editable arguments are: {allowed}, mesh, domain."
        )
    expression = _format_study_argument(argument, value)
    return _rewrite_call_argument(source, located.call, fields, argument, expression, "study")


def set_study_value(source: str, study, value, bc=None, argument=None) -> str:
    """Edit a BC's scalar/vector value or a study's keyword in place.

    Args:
        source: The program text.
        study: Study reference — payload index or name.
        value: The new number(s), written with exact float ``repr`` so typed
            values round-trip (``resolution`` stays integral); for
            ``argument="mesh"`` the name of a declared SimMesh; for
            ``argument="domain"`` the variable name of a named scene object.
        bc: Index of the boundary condition whose value to rewrite.
        argument: Study keyword to rewrite instead (``resolution``,
            ``conductivity``, ``source``, ``youngs``, ``poisson``,
            ``bounds``, ``size``, ``mesh``, ``domain``).

    Returns:
        The patched source.

    Raises:
        PatchError: When neither or both of ``bc``/``argument`` are given, or
            the target cannot be rewritten safely.
    """
    if (bc is None) == (argument is None):
        raise PatchError("set_study_value needs exactly one of `bc` or `argument`.")
    located = _located_study(source, study)
    if bc is not None:
        return _set_study_bc_value(source, located, bc, value)
    return _set_study_argument(source, located, argument, value)
