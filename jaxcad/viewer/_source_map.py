"""Map construction-tree objects back to the source text that created them.

The playground edits Python source, not a scene graph, so viewer interactions
(select a sketch vertex, drag it, add one) have to be expressed as edits to the
user's program. That needs two things:

1. **Which line created this profile** — captured at construction time by
   wrapping ``PolygonProfile.__init__`` while the user program executes and
   walking the stack for the playground frame.
2. **Where each vertex literal sits in the text** — recovered afterwards by
   parsing the source and locating the ``PolygonProfile(...)`` call on that
   line, then reading the character spans of its vertex list elements.

Profiles the mapper cannot pin down unambiguously (built in a loop, vertices
passed as a variable) still render in the viewer; they are just marked
non-editable.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

PLAYGROUND_FILENAME = "<jaxcad-playground>"

Span = tuple[int, int]


@dataclass(frozen=True)
class ProfileCall:
    """Character spans of a ``PolygonProfile(...)`` call's vertex literals."""

    line: int
    vertices_span: Span
    element_spans: list[Span]


def _line_offsets(source: str) -> list[int]:
    """Absolute character offset of the start of each line (0-indexed lines)."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_span(source: str, offsets: list[int], node: ast.AST) -> Span | None:
    """Absolute character span of an AST node.

    ``col_offset`` is a UTF-8 byte offset within its line, so the prefix is
    decoded rather than sliced directly.
    """
    if getattr(node, "end_lineno", None) is None:
        return None
    lines = source.splitlines(keepends=True)

    def absolute(lineno: int, col: int) -> int:
        line = lines[lineno - 1]
        prefix = line.encode()[:col].decode()
        return offsets[lineno - 1] + len(prefix)

    return absolute(node.lineno, node.col_offset), absolute(node.end_lineno, node.end_col_offset)


def _is_profile_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "PolygonProfile"
    if isinstance(func, ast.Attribute):
        return func.attr == "PolygonProfile"
    return False


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
        tree = ast.parse(source)
    except SyntaxError:
        return None

    calls = [node for node in ast.walk(tree) if _is_profile_call(node)]
    exact = [node for node in calls if node.lineno == line]
    if not exact:
        # Multi-line call: the captured frame line can point inside the call.
        exact = [node for node in calls if node.lineno <= line <= (node.end_lineno or node.lineno)]
    if len(exact) != 1:
        return None

    vertices = _vertices_argument(exact[0])
    if not isinstance(vertices, (ast.List, ast.Tuple)):
        return None

    offsets = _line_offsets(source)
    vertices_span = _node_span(source, offsets, vertices)
    if vertices_span is None:
        return None

    element_spans: list[Span] = []
    for element in vertices.elts:
        if not isinstance(element, (ast.List, ast.Tuple)) or len(element.elts) != 2:
            return None
        if not all(_is_number(coordinate) for coordinate in element.elts):
            return None
        span = _node_span(source, offsets, element)
        if span is None:
            return None
        element_spans.append(span)

    return ProfileCall(line=line, vertices_span=vertices_span, element_spans=element_spans)


def _is_number(node: ast.AST) -> bool:
    """True for numeric literals, including negated ones like ``-1.5``."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_number(node.operand)
    return False


def _caller_line(filename: str) -> int | None:
    """Line number of the nearest frame executing *filename*."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename == filename:
            return frame.f_lineno
        frame = frame.f_back
    return None


@contextmanager
def capture_profiles(filename: str = PLAYGROUND_FILENAME):
    """Record every PolygonProfile constructed inside the block, with its line.

    Wraps ``PolygonProfile.__init__`` for the duration so profiles are captured
    wherever they are built — including ones passed straight into ``extrude()``
    that never get bound to a variable.

    Yields:
        A list of ``(profile, line)`` pairs in construction order. ``line`` is
        None when construction did not originate from *filename*.
    """
    from jaxcad.construction.sketch import PolygonProfile

    captured: list[tuple[object, int | None]] = []
    original_init = PolygonProfile.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured.append((self, _caller_line(filename)))

    PolygonProfile.__init__ = patched_init
    try:
        yield captured
    finally:
        PolygonProfile.__init__ = original_init


def build_construction_payload(
    captured: list[tuple[object, int | None]],
    source: str,
) -> list[dict]:
    """Serialize captured profiles into the viewer's construction payload.

    Args:
        captured: ``(profile, line)`` pairs from :func:`capture_profiles`.
        source: The program text the profiles were built from.

    Returns:
        One dict per profile: plane frame, vertex positions in both sketch and
        world coordinates, and the source span of each vertex literal (null when
        that vertex is not safely editable).
    """
    # One call site can build several profiles (a loop or comprehension). Their
    # literals are indistinguishable in the text, so none of them is editable.
    line_counts = Counter(line for _, line in captured if line is not None)
    shared_lines = {line for line, count in line_counts.items() if count > 1}

    payload = []
    for index, (profile, line) in enumerate(captured):
        call = (
            locate_profile_call(source, line)
            if line is not None and line not in shared_lines
            else None
        )
        spans: list[Span | None]
        if call is not None and len(call.element_spans) == len(profile.vertices):
            spans = list(call.element_spans)
        else:
            spans = [None] * len(profile.vertices)

        u, v, normal = profile.plane.frame()
        world = profile.world_vertices()
        payload.append(
            {
                "id": f"profile_{index}",
                "name": profile.name,
                "line": line,
                "editable": call is not None and spans[0] is not None,
                "plane": {
                    "origin": [float(x) for x in profile.plane.origin.xyz],
                    "u": [float(x) for x in u],
                    "v": [float(x) for x in v],
                    "normal": [float(x) for x in normal],
                },
                "vertices": [
                    {
                        "name": vertex.name,
                        "free": bool(vertex.free),
                        "uv": [float(x) for x in vertex.value],
                        "world": [float(x) for x in world[i]],
                        "span": list(spans[i]) if spans[i] is not None else None,
                    }
                    for i, vertex in enumerate(profile.vertices)
                ],
            }
        )
    return payload
