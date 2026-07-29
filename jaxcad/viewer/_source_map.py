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


def _called_name(node: ast.AST) -> str | None:
    """Name of the function a Call node invokes, bare or attribute-qualified."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_profile_call(node: ast.AST) -> bool:
    return _called_name(node) == "PolygonProfile"


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
        tree = ast.parse(source)
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
        span = _node_span(source, offsets, keyword.value)
        if span is None:
            continue
        end = max(end or 0, span[1])
        if keyword.arg is not None:
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
    """Record every construction object built inside the block, with its line.

    Wraps the construction classes' initialisers for the duration, so objects
    are captured wherever they are built — including ones passed straight into
    ``extrude()`` or ``Union()`` that never get bound to a variable.

    Yields:
        A list of ``(object, line)`` pairs in construction order. ``line`` is
        None when construction did not originate from *filename*.
    """
    from jaxcad.construction.sketch import PolygonProfile
    from jaxcad.construction.solid import ConstructionPrimitive

    captured: list[tuple[object, int | None]] = []
    originals = {cls: cls.__init__ for cls in (PolygonProfile, ConstructionPrimitive)}

    def wrap(cls, original):
        def patched_init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            captured.append((self, _caller_line(filename)))

        cls.__init__ = patched_init

    for cls, original in originals.items():
        wrap(cls, original)
    try:
        yield captured
    finally:
        for cls, original in originals.items():
            cls.__init__ = original


def build_construction_payload(
    captured: list[tuple[object, int | None]],
    source: str,
) -> list[dict]:
    """Serialize captured construction objects into the viewer's payload.

    Args:
        captured: ``(object, line)`` pairs from :func:`capture_profiles`.
        source: The program text the objects were built from.

    Returns:
        One dict per object. Every entry carries a world-space ``edges``
        wireframe so the viewer can draw any shape without knowing its topology;
        sketch profiles add their plane and per-vertex handles, primitives add
        their placement and the spans that make it editable.
    """
    from jaxcad.construction.solid import DIMENSIONS

    # One call site can build several objects (a loop or comprehension). Their
    # literals are indistinguishable in the text, so none of them is editable.
    line_counts = Counter(line for _, line in captured if line is not None)
    shared_lines = {line for line, count in line_counts.items() if count > 1}

    payload = []
    for index, (obj, line) in enumerate(captured):
        traceable = line is not None and line not in shared_lines
        if hasattr(obj, "kind") and obj.kind in DIMENSIONS:
            payload.append(_primitive_entry(obj, index, line, source, traceable))
        else:
            payload.append(_profile_entry(obj, index, line, source, traceable))
    return payload


def _plane_transform(source: str, line: int, origin) -> dict | None:
    """Locate the ``SketchPlane(...)`` that positions a profile, if it is literal.

    A sketch's placement lives on its plane rather than the profile, so moving
    the sketch means rewriting the plane's ``origin``. Planes passed in as a
    variable cannot be rewritten, and the sketch is simply not movable then.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if _called_name(node) != "SketchPlane":
            continue
        spans_line = node.lineno <= line <= (node.end_lineno or node.lineno)
        # Also accept a plane built on its own line inside the profile call.
        nested_in_profile = any(
            other.lineno <= line <= (other.end_lineno or other.lineno)
            for other in ast.walk(tree)
            if _is_profile_call(other) and _contains(other, node)
        )
        if not spans_line and not nested_in_profile:
            continue
        plane_call = locate_call(source, node.lineno, {"SketchPlane"})
        if plane_call is None or "origin" not in plane_call.arguments:
            return None
        return {
            "position": [float(x) for x in origin],
            "rotation": [0.0, 0.0, 0.0],
            "dimensions": {},
            "line": node.lineno,
            "call": "SketchPlane",
            "positionArgument": "origin",
            # Turning a sketch means reorienting its plane's normal, which is
            # not an angle triple, so rotation stays a code-only edit.
            "canRotate": False,
        }
    return None


def _contains(outer: ast.AST, inner: ast.AST) -> bool:
    return any(node is inner for node in ast.walk(outer))


def _profile_entry(profile, index: int, line: int | None, source: str, traceable: bool) -> dict:
    """Payload for a sketch profile: plane, closed edge loop, vertex handles."""
    call = locate_profile_call(source, line) if traceable else None
    spans: list[Span | None]
    if call is not None and len(call.element_spans) == len(profile.vertices):
        spans = list(call.element_spans)
    else:
        spans = [None] * len(profile.vertices)

    u, v, normal = profile.plane.frame()
    world = [[float(x) for x in point] for point in profile.world_vertices()]
    count = len(world)
    return {
        "id": f"profile_{index}",
        "kind": "profile",
        "name": profile.name,
        "line": line,
        "editable": call is not None and spans[0] is not None,
        "edges": [[world[i], world[(i + 1) % count]] for i in range(count)],
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
                "world": world[i],
                "span": list(spans[i]) if spans[i] is not None else None,
            }
            for i, vertex in enumerate(profile.vertices)
        ],
        "transform": (
            _plane_transform(source, line, profile.plane.origin.xyz) if traceable else None
        ),
        "spans": {},
    }


def _primitive_entry(primitive, index: int, line: int | None, source: str, traceable: bool) -> dict:
    """Payload for a construction primitive: outline plus editable placement."""
    from jaxcad.construction.solid import DIMENSIONS

    call = locate_call(source, line, {primitive.kind}) if traceable else None
    arguments = call.arguments if call is not None else {}
    # Placement is only editable when both literals are present to rewrite.
    editable = "position" in arguments

    dimensions = {
        key: (
            [float(x) for x in primitive.params[key].xyz]
            if key == "size"
            else float(primitive.params[key].value)
        )
        for key in DIMENSIONS[primitive.kind]
    }
    return {
        "id": f"{primitive.kind}_{index}",
        "kind": primitive.kind,
        "name": primitive.name,
        "line": line,
        "editable": editable,
        "edges": primitive.world_edges(),
        "plane": None,
        "vertices": [],
        "transform": {
            "position": [float(x) for x in primitive.position.xyz],
            "rotation": list(primitive.rotation_values()),
            "dimensions": dimensions,
            "line": line,
            "call": primitive.kind,
            "positionArgument": "position",
            "canRotate": True,
        }
        if editable
        else None,
        "spans": {name: list(span) for name, span in arguments.items()},
    }
