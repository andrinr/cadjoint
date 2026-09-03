"""Locate a single construction call on a line and map it to character spans.

These are the line-addressed locators: the viewer knows which line built an
object (from :mod:`cadjoint.viewer.source_map.capture`) and asks here for the
spans of the literals it may rewrite.  Both locators refuse ambiguity — several
candidate calls covering the line, or vertices that are not literal points —
by returning None, which is how the payload marks an object non-editable.

Add a locator here when it resolves *one call at a line* to spans.  Locators
for whole classes of top-level declarations live in
:mod:`cadjoint.viewer.source_map.declarations`, and constraint statements in
:mod:`cadjoint.viewer.source_map.constraints`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from cadjoint.viewer.source_map.nodes import (
    Span,
    _called_name,
    _editable_value_node,
    _is_number,
    _is_profile_call,
    _line_offsets,
    _node_span,
    _resolved_container,
    parse_module,
)


@dataclass(frozen=True)
class ProfileCall:
    """Character spans of a ``PolygonProfile(...)`` call's vertices.

    ``element_spans`` point at the editable coordinate payloads, which may
    resolve through names into ``Vector2(value=[...])`` assignments.
    ``list_element_spans`` always point at the elements of the profile's
    vertex list itself.  Keeping both prevents structural edits from being
    inserted into a parameter constructor.
    """

    line: int
    vertices_span: Span
    element_spans: list[Span]
    list_element_spans: list[Span]


@dataclass(frozen=True)
class CallSite:
    """Character spans of a construction call's arguments."""

    line: int
    name: str
    arguments: dict[str, Span]
    """Offset just past the last argument, where a new keyword can be added."""
    arguments_end: int


def locate_call(source: str, line: int, names: set[str]) -> CallSite | None:
    """Locate a construction call at *line* and map its arguments to spans.

    Args:
        source: The full program text.
        line: 1-based line captured when the object was constructed.
        names: Acceptable function names, e.g. ``{"box", "sphere"}``.

    Returns:
        A :class:`CallSite` with one span per keyword argument, or None when the
        call is ambiguous, absent, or unparseable.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return None

    calls = [node for node in ast.walk(tree) if _called_name(node) in names]
    matches = [node for node in calls if node.lineno == line]
    if not matches:
        matches = [
            node for node in calls if node.lineno <= line <= (node.end_lineno or node.lineno)
        ]
    if len(matches) != 1:
        return None

    call = matches[0]
    offsets = _line_offsets(source)
    arguments: dict[str, Span] = {}
    end = None
    for keyword in call.keywords:
        raw_span = _node_span(source, offsets, keyword.value)
        if raw_span is None:
            continue
        end = max(end or 0, raw_span[1])
        editable = _editable_value_node(keyword.value, tree)
        span = _node_span(source, offsets, editable) if editable is not None else None
        if keyword.arg is not None and span is not None:
            arguments[keyword.arg] = span
    for argument in call.args:
        span = _node_span(source, offsets, argument)
        if span is not None:
            end = max(end or 0, span[1])
    if end is None:
        return None
    return CallSite(
        line=line,
        name=_called_name(call) or "",
        arguments=arguments,
        arguments_end=end,
    )


def _vertices_argument(call: ast.Call) -> ast.AST | None:
    """The ``vertices`` argument of a PolygonProfile call, positional or keyword."""
    for keyword in call.keywords:
        if keyword.arg == "vertices":
            return keyword.value
    return call.args[0] if call.args else None


def locate_profile_call(source: str, line: int) -> ProfileCall | None:
    """Locate the ``PolygonProfile(...)`` call created at *line*.

    Args:
        source: The full program text.
        line: 1-based line number captured when the profile was constructed.

    Returns:
        A :class:`ProfileCall` with character spans, or None when the call is
        ambiguous (several on one line), not a literal list of literal points,
        or the source cannot be parsed.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return None

    calls = [node for node in ast.walk(tree) if _is_profile_call(node)]
    exact = [node for node in calls if node.lineno == line]
    if not exact:
        # Multi-line call: the captured frame line can point inside the call.
        exact = [node for node in calls if node.lineno <= line <= (node.end_lineno or node.lineno)]
    if len(exact) != 1:
        return None

    vertices = _resolved_container(_vertices_argument(exact[0]), tree)
    if vertices is None:
        return None

    offsets = _line_offsets(source)
    vertices_span = _node_span(source, offsets, vertices)
    if vertices_span is None:
        return None

    element_spans: list[Span] = []
    list_element_spans: list[Span] = []
    for element in vertices.elts:
        list_span = _node_span(source, offsets, element)
        if list_span is None:
            return None
        list_element_spans.append(list_span)
        editable = _editable_value_node(element, tree)
        if not isinstance(editable, (ast.List, ast.Tuple)) or len(editable.elts) != 2:
            return None
        if not all(_is_number(coordinate) for coordinate in editable.elts):
            return None
        span = _node_span(source, offsets, editable)
        if span is None:
            return None
        element_spans.append(span)

    return ProfileCall(
        line=line,
        vertices_span=vertices_span,
        element_spans=element_spans,
        list_element_spans=list_element_spans,
    )
