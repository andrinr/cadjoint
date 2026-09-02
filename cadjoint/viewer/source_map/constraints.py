"""Locate the constraint statements attached to a sketch profile.

A constraint is a bare top-level expression statement — ``FixedConstraint(
profile.vertices[0], ...)`` — so it has no name of its own.  Its identity in
the viewer is its *ordinal* among the profile's constraint statements in source
order, and this module is the single definition of that ordering: the payload
builder and the ``delete_constraint``/``set_constraint_value`` patch operations
both read it, so a chip's index always addresses the statement it displays.

Add code here when it concerns recognising a constraint call, mapping its
arguments back to vertex indices, or enumerating a profile's constraints.  The
runtime side (serializing the constraint objects the program actually built)
lives in :mod:`cadjoint.viewer.source_map.payload`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from cadjoint.viewer.source_map.calls import _vertices_argument
from cadjoint.viewer.source_map.nodes import (
    _called_name,
    _contains,
    _is_profile_call,
    _resolved_container,
    parse_module,
)

CONSTRAINT_CLASS_KINDS = {
    "FixedConstraint": "fixed",
    "DistanceConstraint": "distance",
    "HorizontalConstraint": "horizontal",
    "VerticalConstraint": "vertical",
    "CoincidentConstraint": "coincident",
    "EqualLengthConstraint": "equal_length",
    "ParallelEdgesConstraint": "parallel",
    "PerpendicularEdgesConstraint": "perpendicular",
    "ParallelConstraint": "parallel",
    "PerpendicularConstraint": "perpendicular",
}
"""Constraint class names mapped to their viewer payload ``kind``.

Kept as strings on purpose: the serializer must keep working while the
constraint classes themselves land independently in :mod:`cadjoint.constraints`.
"""


@dataclass(frozen=True)
class ConstraintStatement:
    """One top-level constraint statement attached to a profile's vertices."""

    kind: str
    vertices: tuple[int, ...]
    statement: ast.stmt
    call: ast.Call


def _profile_vertex_names(call: ast.Call, tree: ast.Module) -> dict[str, int]:
    """Map bare parameter names in the profile's vertex list to their indices."""
    container = _resolved_container(_vertices_argument(call), tree)
    if container is None:
        return {}
    return {
        element.id: index
        for index, element in enumerate(container.elts)
        if isinstance(element, ast.Name)
    }


def _constraint_vertex_index(node: ast.AST, profile: str, names: dict[str, int]) -> int | None:
    """Vertex index behind ``profile.vertices[i]`` or a bare vertex parameter name."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "vertices"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == profile
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
        and not isinstance(node.slice.value, bool)
    ):
        return node.slice.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    return None


def _constraint_call_vertices(
    call: ast.Call, profile: str, names: dict[str, int]
) -> tuple[str, tuple[int, ...]] | None:
    """``(kind, vertex indices)`` for a constraint call on this profile, if any."""
    name = _called_name(call)
    kind = CONSTRAINT_CLASS_KINDS.get(name or "")
    if kind is None:
        return None
    args = call.args

    def indices(nodes: list[ast.AST]) -> tuple[int, ...] | None:
        resolved = [_constraint_vertex_index(node, profile, names) for node in nodes]
        if any(index is None for index in resolved):
            return None
        return tuple(resolved)  # type: ignore[arg-type]

    if name == "FixedConstraint" and len(args) == 2:
        result = indices(args[:1])
    elif name == "DistanceConstraint" and len(args) == 3:
        result = indices(args[:2])
    elif (
        name in {"HorizontalConstraint", "VerticalConstraint", "CoincidentConstraint"}
        and (len(args) == 2)
        or name
        in {"EqualLengthConstraint", "ParallelEdgesConstraint", "PerpendicularEdgesConstraint"}
        and len(args) == 4
    ):
        result = indices(list(args))
    elif name in {"ParallelConstraint", "PerpendicularConstraint"} and len(args) == 2:
        # Edge-direction fallback form: each argument is `vertices[j] - vertices[i]`.
        flat: list[int] = []
        for argument in args:
            if not (isinstance(argument, ast.BinOp) and isinstance(argument.op, ast.Sub)):
                return None
            pair = indices([argument.right, argument.left])
            if pair is None:
                return None
            flat.extend(pair)
        result = tuple(flat)
    else:
        return None
    return (kind, result) if result is not None else None


def locate_constraint_statements(source: str, line: int) -> list[ConstraintStatement] | None:
    """Top-level constraint statements of the named profile at *line*, in source order.

    This ordering defines each constraint's ``index`` in the viewer payload, so
    the payload builder and the patch operations agree on which statement an
    index refers to. Both the ``profile.vertices[i]`` subscript form and the
    bare vertex-parameter-name form are recognized.

    Args:
        source: The full program text.
        line: 1-based line of the ``PolygonProfile(...)`` call.

    Returns:
        The constraint statements in source order, or None when the profile
        cannot be located or is not bound to a single name.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return None
    calls = [
        node
        for node in ast.walk(tree)
        if _is_profile_call(node) and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(calls) != 1:
        return None
    binding = next(
        (item for item in tree.body if isinstance(item, ast.Assign) and _contains(item, calls[0])),
        None,
    )
    if not (
        binding is not None
        and len(binding.targets) == 1
        and isinstance(binding.targets[0], ast.Name)
    ):
        return None
    profile = binding.targets[0].id
    names = _profile_vertex_names(calls[0], tree)
    statements: list[ConstraintStatement] = []
    for item in tree.body:
        if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)):
            continue
        parsed = _constraint_call_vertices(item.value, profile, names)
        if parsed is not None:
            statements.append(ConstraintStatement(parsed[0], parsed[1], item, item.value))
    return statements
