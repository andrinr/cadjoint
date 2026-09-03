"""Locate the calls that build features, and the plane a sketch is drawn on.

Face references make two new things addressable in the text.  A face belongs to
a *feature* — ``body = extrude(profile, depth=...)`` — and writing a reference
to it means naming the variable that feature was bound to, so
:func:`locate_feature_call` resolves a captured feature line to that binding.
Once written, the sketch's ``plane=`` argument *is* the reference, so
:func:`locate_plane_reference` reads it back: which constructor was used, whose
face it names, and the span the next edit replaces.

Both locators follow the package's rule that ambiguity is refused rather than
guessed: a feature not bound to a plain name, or a plane argument built by an
arbitrary expression, reports what it can and marks the rest unusable.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from cadjoint.viewer.source_map.nodes import (
    Span,
    _assignment_value,
    _called_name,
    _is_construction_call,
    _is_profile_call,
    _line_offsets,
    _node_span,
    parse_module,
)

# Generators that turn a sketch into a solid, in the order their faces are
# registered on the profile.
FEATURE_CALL_KINDS = frozenset({"extrude", "revolve", "loft"})

# Primitive factories; ``Solid.box(...)`` reads as the call name ``box``.
PRIMITIVE_CALL_KINDS = frozenset({"box", "sphere", "cylinder"})

# The SketchPlane constructors that derive a plane from a reference.
PLANE_CONSTRUCTORS = frozenset({"on", "offset", "tangent", "midplane"})


@dataclass(frozen=True)
class FeatureCall:
    """One feature-generating call and the variable it was bound to.

    ``variable`` is None when the call's result is not assigned to a plain
    name — ``scene = Union(extrude(p, depth=1), ...)`` — which is exactly the
    case where a face of it cannot be written back into the source.
    """

    line: int
    kind: str
    variable: str | None
    statement_line: int


def locate_feature_calls(source: str) -> list[FeatureCall] | None:
    """Every feature call in the program, in source order.

    Args:
        source: The full program text.

    Returns:
        One :class:`FeatureCall` per ``extrude``/``revolve``/``loft``/
            primitive call, or None when the source cannot be parsed.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return None

    names = FEATURE_CALL_KINDS | PRIMITIVE_CALL_KINDS
    found: list[FeatureCall] = []
    for statement in tree.body:
        bound = (
            statement.targets[0].id
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            else None
        )
        for node in ast.walk(statement):
            if _called_name(node) not in names or not _is_construction_call(node):
                continue
            # Only the call that *is* the assigned value binds the variable;
            # a nested one (inside a Union, say) has no name of its own.
            variable = bound if getattr(statement, "value", None) is node else None
            found.append(
                FeatureCall(
                    line=node.lineno,
                    kind=_called_name(node) or "",
                    variable=variable,
                    statement_line=statement.lineno,
                )
            )
    return sorted(found, key=lambda call: (call.line, call.kind))


def locate_feature_call(
    source: str, line: int, kinds: frozenset[str] | None = None
) -> FeatureCall | None:
    """The one feature call covering *line*.

    Args:
        source: The full program text.
        line: 1-based line captured when the feature was generated.
        kinds: Restrict to these call names; defaults to every feature call.

    Returns:
        The one matching feature call, or None when the line holds no such call
            or holds more than one.
    """
    calls = locate_feature_calls(source)
    if calls is None:
        return None
    allowed = kinds if kinds is not None else FEATURE_CALL_KINDS | PRIMITIVE_CALL_KINDS
    matches = [call for call in calls if call.line == line and call.kind in allowed]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class PlaneReference:
    """What a sketch's ``plane=`` argument says, read back from the source.

    ``constructor`` is the ``SketchPlane`` classmethod used (``on``,
    ``offset``, ``tangent``, ``midplane``), ``"plain"`` for a literal
    ``SketchPlane(origin=..., normal=...)``, or None when the sketch takes the
    default world plane.  ``owner`` and ``accessor`` describe the face the
    plane was derived from — ``("body", "cap", '"+"')`` for
    ``SketchPlane.on(body.cap("+"))`` — and are None for the forms that name
    no single face.
    """

    constructor: str | None
    owner: str | None
    accessor: str | None
    argument: str | None
    span: Span | None


def _plane_argument(call: ast.Call) -> ast.AST | None:
    """The ``plane`` argument of a PolygonProfile call, positional or keyword."""
    for keyword in call.keywords:
        if keyword.arg == "plane":
            return keyword.value
    return call.args[1] if len(call.args) > 1 else None


def _sketch_plane_method(node: ast.AST) -> str | None:
    """The ``SketchPlane.<method>`` a call invokes, or None."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    owner = node.func.value
    if isinstance(owner, ast.Name) and owner.id == "SketchPlane":
        return node.func.attr
    return None


def _face_accessor(node: ast.AST) -> tuple[str | None, str | None, str | None]:
    """Split ``body.cap("+")`` into ``("body", "cap", '"+"')``."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None, None, None
    owner = node.func.value
    if not isinstance(owner, ast.Name):
        return None, None, None
    argument = ast.unparse(node.args[0]) if len(node.args) == 1 else None
    return owner.id, node.func.attr, argument


def locate_plane_reference(source: str, line: int) -> PlaneReference | None:
    """Read back the plane a sketch is drawn on.

    Args:
        source: The full program text.
        line: 1-based line of the ``PolygonProfile(...)`` call.

    Returns:
        The sketch's plane reference, or None when the source cannot be parsed
            or the line does not hold exactly one profile call.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return None
    profiles = [
        node
        for node in ast.walk(tree)
        if _is_profile_call(node) and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(profiles) != 1:
        return None

    argument = _plane_argument(profiles[0])
    if argument is None:
        return PlaneReference(None, None, None, None, None)

    offsets = _line_offsets(source)
    span = _node_span(source, offsets, argument)
    node = argument
    if isinstance(node, ast.Name):
        resolved = _assignment_value(tree, node.id, node.lineno)
        if resolved is not None:
            node = resolved

    method = _sketch_plane_method(node)
    if method in PLANE_CONSTRUCTORS:
        owner, accessor, face_argument = _face_accessor(node.args[0]) if node.args else (None,) * 3
        if method == "tangent" and owner is None and node.args:
            # ``SketchPlane.tangent(body, near=[...])`` names the solid itself.
            first = node.args[0]
            owner = first.id if isinstance(first, ast.Name) else None
        return PlaneReference(method, owner, accessor, face_argument, span)
    if _called_name(node) == "SketchPlane":
        return PlaneReference("plain", None, None, None, span)
    return PlaneReference(None, None, None, None, span)
