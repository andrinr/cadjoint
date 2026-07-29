"""Rewrite sketch vertex literals in the user's Python source.

Viewer interactions are applied to the program text itself, so the editor stays
the single source of truth. Edits are pure span surgery guided by
:mod:`jaxcad.viewer._source_map`: only the characters of the targeted vertex
literal change, leaving formatting, comments, and the rest of the file byte for
byte identical.

No user code is executed here — the server can patch without spawning the
compile worker.
"""

from __future__ import annotations

import ast

from jaxcad.viewer._source_map import locate_profile_call


class PatchError(ValueError):
    """Raised when a source edit cannot be applied safely."""


def _format_coordinate(value: float) -> str:
    """Format a coordinate compactly while staying valid Python.

    Ray-plane intersections land on values like ``8.9e-16`` where the user
    clearly means zero; snapping those keeps generated code readable instead of
    littering the sketch with floating-point noise.
    """
    number = float(value)
    if abs(number) < 1e-9:
        return "0"
    return f"{number:.4g}"


def _format_vertex(xy) -> str:
    x, y = xy
    return f"[{_format_coordinate(x)}, {_format_coordinate(y)}]"


def _require_call(source: str, line: int):
    call = locate_profile_call(source, line)
    if call is None:
        raise PatchError(
            f"No editable PolygonProfile literal found at line {line}. "
            "Sketches built in a loop or from variables cannot be edited from the viewer."
        )
    return call


def _validate(source: str) -> str:
    """Guard against emitting a file that no longer parses."""
    try:
        ast.parse(source)
    except SyntaxError as error:  # pragma: no cover - defensive
        raise PatchError(f"Patched source is not valid Python: {error}") from error
    return source


def set_vertex(source: str, line: int, index: int, xy) -> str:
    """Replace the coordinates of one sketch vertex.

    Args:
        source: The program text.
        line: 1-based line of the ``PolygonProfile(...)`` call.
        index: Zero-based vertex position within the profile.
        xy: New ``(x, y)`` sketch-plane coordinates.

    Returns:
        The patched source.

    Raises:
        PatchError: If the profile or vertex cannot be located.
    """
    call = _require_call(source, line)
    if not 0 <= index < len(call.element_spans):
        raise PatchError(
            f"Vertex index {index} is out of range for the sketch at line {line} "
            f"({len(call.element_spans)} vertices)."
        )
    start, end = call.element_spans[index]
    return _validate(source[:start] + _format_vertex(xy) + source[end:])


def insert_vertex(source: str, line: int, index: int, xy) -> str:
    """Insert a new sketch vertex before position *index*.

    ``index == len(vertices)`` appends. Insertion reuses the neighbouring
    literal's span so the new entry lands inside the existing list layout.

    Args:
        source: The program text.
        line: 1-based line of the ``PolygonProfile(...)`` call.
        index: Zero-based position the new vertex should occupy.
        xy: ``(x, y)`` sketch-plane coordinates for the new vertex.

    Returns:
        The patched source.

    Raises:
        PatchError: If the profile cannot be located or the index is invalid.
    """
    call = _require_call(source, line)
    count = len(call.element_spans)
    if not 0 <= index <= count:
        raise PatchError(
            f"Insert index {index} is out of range for the sketch at line {line} "
            f"({count} vertices)."
        )

    literal = _format_vertex(xy)
    if index < count:
        start, _ = call.element_spans[index]
        return _validate(source[:start] + f"{literal}, " + source[start:])
    _, end = call.element_spans[-1]
    return _validate(source[:end] + f", {literal}" + source[end:])


def delete_vertex(source: str, line: int, index: int) -> str:
    """Remove one sketch vertex, keeping at least a triangle.

    Args:
        source: The program text.
        line: 1-based line of the ``PolygonProfile(...)`` call.
        index: Zero-based vertex position to remove.

    Returns:
        The patched source.

    Raises:
        PatchError: If the profile cannot be located, the index is invalid, or
            the profile would drop below three vertices.
    """
    call = _require_call(source, line)
    count = len(call.element_spans)
    if not 0 <= index < count:
        raise PatchError(
            f"Vertex index {index} is out of range for the sketch at line {line} "
            f"({count} vertices)."
        )
    if count <= 3:
        raise PatchError("A sketch profile needs at least 3 vertices.")

    start, end = call.element_spans[index]
    if index < count - 1:
        # Swallow the separator up to the next element.
        end = call.element_spans[index + 1][0]
    else:
        # Last element: swallow the separator after the previous one.
        start = call.element_spans[index - 1][1]
    return _validate(source[:start] + source[end:])


OPERATIONS = {
    "set_vertex": set_vertex,
    "insert_vertex": insert_vertex,
    "delete_vertex": delete_vertex,
}


def apply_operation(source: str, operation: str, **kwargs) -> str:
    """Dispatch a named patch operation.

    Args:
        source: The program text.
        operation: One of ``set_vertex``, ``insert_vertex``, ``delete_vertex``.
        **kwargs: Arguments for that operation (``line``, ``index``, ``xy``).

    Returns:
        The patched source.

    Raises:
        PatchError: On an unknown operation or a failed edit.
    """
    handler = OPERATIONS.get(operation)
    if handler is None:
        raise PatchError(f"Unknown patch operation {operation!r}.")
    try:
        return handler(source, **kwargs)
    except TypeError as error:
        raise PatchError(f"Invalid arguments for {operation!r}: {error}") from error
