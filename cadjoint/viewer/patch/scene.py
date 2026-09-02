"""Read and extend the program's ``scene`` assignment.

Every solid the viewer creates has to end up in the scene, and every solid it
deletes has to come out of it, so the ``scene = ...`` statement is the one
piece of program structure that nearly all operations touch.  The rules that
live here:

- a program without a module-level ``scene`` assignment cannot have solids
  added from the viewer at all — the operations report that rather than
  inventing one;
- adding to the scene wraps the existing expression in ``Union(...)`` when it
  is not already one, importing ``Union`` as part of the same edit;
- ``domain=`` arguments name a scene object, and because the patcher never
  executes code, "this object exists" is checked structurally: the name must be
  assigned by a module-level statement above the declaration being edited.

Add code here when it answers a question about the scene expression or about
which names are in scope at a line.  Generic span surgery is one layer down in
:mod:`cadjoint.viewer.patch.edits`.
"""

from __future__ import annotations

import ast

from cadjoint.viewer.patch.edits import _ensure_import, _validate
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.source_map.nodes import _called_name, _line_offsets, _node_span


def _scene_assignment(tree: ast.Module) -> ast.Assign | None:
    """The module-level ``scene = ...`` statement, if there is one."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "scene" for target in node.targets
        ):
            return node
    return None


def _extend_scene_with(source: str, variable: str) -> str:
    """Add a named solid to the scene expression."""
    tree = ast.parse(source)
    assignment = _scene_assignment(tree)
    if assignment is None:
        raise PatchError("Add a `scene = ...` assignment before adding an extrusion.")
    offsets = _line_offsets(source)
    value = assignment.value
    if isinstance(value, ast.Call) and _called_name(value) == "Union":
        anchor = value.args[-1] if value.args else None
        if anchor is None:
            raise PatchError("The scene Union has no operands to extend.")
        _, end = _node_span(source, offsets, anchor)
        patched = source[:end] + f", {variable}" + source[end:]
    else:
        start, end = _node_span(source, offsets, value)
        patched = source[:start] + f"Union({source[start:end]}, {variable})" + source[end:]
        patched = _ensure_import(patched, ast.parse(patched), "cadjoint.sdf.boolean", "Union")
    return _validate(patched)


def _union_assignments(tree: ast.Module) -> list[ast.Assign]:
    """Every module-level ``name = Union(...)`` statement, the scene's included.

    A scene commonly builds sub-assemblies first — ``thermal_body = Union(sink,
    slug, ...)`` — and unions those into ``scene``; an object is deletable from
    any of them, not only from the final one.
    """
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and _called_name(node.value) == "Union"
    ]


def _all_union_operands(tree: ast.Module) -> list[ast.AST]:
    """Positional operands of every module-level ``Union(...)`` assignment."""
    operands: list[ast.AST] = []
    for assignment in _union_assignments(tree):
        operands.extend(assignment.value.args)  # type: ignore[union-attr]
    return operands


def _union_operands(scene: ast.Assign | None) -> list[ast.AST]:
    """Positional arguments of a ``scene = Union(...)`` assignment."""
    if scene is None or not isinstance(scene.value, ast.Call):
        return []
    if _called_name(scene.value) != "Union":
        return []
    return list(scene.value.args)


def _assigned_before(source: str, before_line: int) -> set[str]:
    """Names assigned by module-level statements above *before_line*."""
    tree = ast.parse(source)
    defined: set[str] = set()
    for statement in tree.body:
        if statement.lineno >= before_line:
            continue
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        defined.update(target.id for target in targets if isinstance(target, ast.Name))
    return defined


def _domain_expression(source: str, value, before_line: int, noun: str) -> str:
    """Validate a ``domain`` value: the name of an object assigned above.

    The patcher never executes code, so "exists among construction nodes"
    is checked structurally: the name must be assigned by a module-level
    statement before the edited declaration (execution order), which is
    where every named scene object lives.
    """
    if not isinstance(value, str) or not value.isidentifier():
        raise PatchError("`domain` needs the variable name of a named scene object.")
    if value not in _assigned_before(source, before_line):
        raise PatchError(
            f"`{value}` is not assigned before the {noun}; `domain` must name an "
            "existing construction object."
        )
    return value
