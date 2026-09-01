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

PLAYGROUND_FILENAME = "<cadjoint-playground>"

Span = tuple[int, int]


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


_PARAMETER_CALLS = {"Parameter", "Scalar", "Vector", "Vector2"}
_ARRAY_CALLS = {"array", "asarray"}


def _assignment_value(tree: ast.Module, name: str, before_line: int) -> ast.AST | None:
    """Value of one unambiguous module-level assignment visible at a use site.

    Following names is intentionally conservative: rebinding, tuple unpacking,
    loop targets, and function-local values are rejected rather than risking a
    source edit at the wrong definition.
    """
    values: list[ast.AST] = []
    for statement in tree.body:
        if statement.lineno >= before_line:
            continue
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ) or (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
            and statement.value is not None
        ):
            values.append(statement.value)
    return values[0] if len(values) == 1 else None


def _call_value_argument(call: ast.Call) -> ast.AST | None:
    """The numeric payload of a Parameter/array wrapper call."""
    for keyword in call.keywords:
        if keyword.arg == "value":
            return keyword.value
    return call.args[0] if call.args else None


def _editable_value_node(
    node: ast.AST,
    tree: ast.Module,
    seen: set[str] | None = None,
) -> ast.AST | None:
    """Resolve a numeric value through named parameters to its source literal."""
    seen = set() if seen is None else seen
    if _is_number(node):
        return node
    if isinstance(node, (ast.List, ast.Tuple)) and all(_is_number(item) for item in node.elts):
        return node
    if isinstance(node, ast.Name):
        if node.id in seen:
            return None
        value = _assignment_value(tree, node.id, node.lineno)
        return _editable_value_node(value, tree, seen | {node.id}) if value is not None else None
    if isinstance(node, ast.Call) and _called_name(node) in _PARAMETER_CALLS | _ARRAY_CALLS:
        value = _call_value_argument(node)
        return _editable_value_node(value, tree, seen) if value is not None else None
    return None


def _resolved_container(
    node: ast.AST,
    tree: ast.Module,
    seen: set[str] | None = None,
) -> ast.List | ast.Tuple | None:
    """Resolve a named vertex collection to its literal list or tuple."""
    seen = set() if seen is None else seen
    if isinstance(node, (ast.List, ast.Tuple)):
        return node
    if isinstance(node, ast.Name):
        if node.id in seen:
            return None
        value = _assignment_value(tree, node.id, node.lineno)
        return _resolved_container(value, tree, seen | {node.id}) if value is not None else None
    if isinstance(node, ast.Call) and _called_name(node) in _ARRAY_CALLS:
        value = _call_value_argument(node)
        return _resolved_container(value, tree, seen) if value is not None else None
    return None


def _resolved_call(
    node: ast.AST,
    tree: ast.Module,
    expected: str,
    seen: set[str] | None = None,
) -> ast.Call | None:
    """Resolve a named construction object to the call that creates it."""
    seen = set() if seen is None else seen
    if isinstance(node, ast.Call):
        return node if _called_name(node) == expected else None
    if isinstance(node, ast.Name):
        if node.id in seen:
            return None
        value = _assignment_value(tree, node.id, node.lineno)
        return (
            _resolved_call(value, tree, expected, seen | {node.id}) if value is not None else None
        )
    return None


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
    from cadjoint.construction.sketch import PolygonProfile
    from cadjoint.construction.solid import ConstructionPrimitive

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
    from cadjoint.construction.solid import DIMENSIONS

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


def build_construction_relations(
    captured: list[tuple[object, int | None]],
) -> list[dict]:
    """Serialize constraints relating whole construction-object positions."""
    from cadjoint.construction.solid import DIMENSIONS

    position_nodes: dict[int, str] = {}
    position_params: dict[int, object] = {}
    for index, (obj, _) in enumerate(captured):
        if hasattr(obj, "kind") and obj.kind in DIMENSIONS:
            position_nodes[id(obj.position)] = f"{obj.kind}_{index}"
            position_params[id(obj.position)] = obj.position

    seen: set[int] = set()
    relations: list[dict] = []
    for parameter in position_params.values():
        for constraint in parameter.get_constraints():
            if id(constraint) in seen:
                continue
            seen.add(id(constraint))
            name = constraint.__class__.__name__
            if name == "FixedConstraint":
                node_id = position_nodes.get(id(constraint.param))
                if node_id is not None:
                    relations.append(
                        {
                            "kind": "fixed",
                            "nodes": [node_id],
                            "value": [float(value) for value in constraint.target.reshape(-1)],
                        }
                    )
            elif name == "DistanceConstraint":
                node_ids = [
                    position_nodes.get(id(constraint.param1)),
                    position_nodes.get(id(constraint.param2)),
                ]
                if all(node_id is not None for node_id in node_ids):
                    relations.append(
                        {
                            "kind": "distance",
                            "nodes": node_ids,
                            "value": float(constraint.distance.value),
                        }
                    )
    return relations


def build_material_payload(namespace: dict, source: str) -> list[dict]:
    """Serialize named ``Material`` definitions for the visual material browser.

    Only direct module-level assignments such as ``brass = Material(...)`` are
    exposed. This gives every browser card a stable Python identifier that can
    be written into another construction call when the user drops the material
    onto an object.
    """
    from cadjoint.render.material import Material

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    result: list[dict] = []
    for statement in tree.body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and _called_name(statement.value) == "Material"
        ):
            continue
        variable = statement.targets[0].id
        material = namespace.get(variable)
        if not isinstance(material, Material):
            continue
        call = locate_call(source, statement.value.lineno, {"Material"})
        params = material.params
        result.append(
            {
                "id": f"material_{len(result)}",
                "name": variable,
                "line": statement.value.lineno,
                "editable": call is not None,
                "color": [float(value) for value in params["color"].value],
                "roughness": float(params["roughness"].value),
                "metallic": float(params["metallic"].value),
                "opacity": float(params["opacity"].value),
                "ior": float(params["ior"].value),
                "reflectivity": float(params["reflectivity"].value),
                "spans": (
                    {name: list(span) for name, span in call.arguments.items()}
                    if call is not None
                    else {}
                ),
            }
        )
    return result


def _material_name_from_call(call: ast.Call) -> str | None:
    """Return the named material passed to a construction call, if any."""
    value = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "material"),
        None,
    )
    return value.id if isinstance(value, ast.Name) else None


def _primitive_material(source: str, line: int, kind: str) -> str | None:
    """Named material referenced by a primitive call at *line*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called_name(node) == kind
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    return _material_name_from_call(calls[0]) if len(calls) == 1 else None


def _profile_material(source: str, line: int) -> str | None:
    """Named material on the generator consuming a profile at *line*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    profiles = [
        node
        for node in ast.walk(tree)
        if _is_profile_call(node) and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(profiles) != 1:
        return None
    statement = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.Assign) and _contains(item, profiles[0])
        ),
        None,
    )
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return None
    variable = statement.targets[0].id
    generators = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called_name(node) in {"extrude", "revolve", "loft"}
        and any(
            isinstance(argument, ast.Name) and argument.id == variable
            for argument in (node.args if _called_name(node) == "loft" else node.args[:1])
        )
    ]
    return _material_name_from_call(generators[0]) if len(generators) == 1 else None


def _plane_transform(source: str, line: int, origin) -> dict | None:
    """Locate the plane owning a profile, including named and default planes."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    profiles = [
        node
        for node in ast.walk(tree)
        if _is_profile_call(node) and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(profiles) != 1:
        return None
    profile = profiles[0]
    plane_argument = next(
        (keyword.value for keyword in profile.keywords if keyword.arg == "plane"),
        profile.args[1] if len(profile.args) > 1 else None,
    )
    plane = (
        _resolved_call(plane_argument, tree, "SketchPlane") if plane_argument is not None else None
    )

    if plane is not None:
        return {
            "position": [float(x) for x in origin],
            "rotation": [0.0, 0.0, 0.0],
            "dimensions": {},
            "line": plane.lineno,
            "call": "SketchPlane",
            "positionArgument": "origin",
            "canRotate": False,
        }

    if plane_argument is not None:
        # An arbitrary expression produced the plane and cannot be rewritten
        # without changing the program's meaning.
        return None

    # PolygonProfile() creates an identity plane by default. Moving it can be
    # represented safely by adding an explicit SketchPlane keyword.
    return {
        "position": [float(x) for x in origin],
        "rotation": [0.0, 0.0, 0.0],
        "dimensions": {},
        "line": profile.lineno,
        "call": "PolygonProfile",
        "positionArgument": "planeOrigin",
        "canRotate": False,
    }


def _contains(outer: ast.AST, inner: ast.AST) -> bool:
    return any(node is inner for node in ast.walk(outer))


CONSTRAINT_CLASS_KINDS = {
    "FixedConstraint": "fixed",
    "DistanceConstraint": "distance",
    "HorizontalConstraint": "horizontal",
    "VerticalConstraint": "vertical",
    "CoincidentConstraint": "coincident",
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
        or name in {"ParallelEdgesConstraint", "PerpendicularEdgesConstraint"}
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
        tree = ast.parse(source)
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


STUDY_CALL_KINDS = {"ThermalStudy": "thermal", "ElasticStudy": "elastic"}
"""Study constructor names mapped to their viewer payload ``kind``."""


@dataclass(frozen=True)
class StudyStatement:
    """One top-level study constructor statement, in source order.

    Statement order equals construction order for top-level declarations, so
    the position doubles as a stable index: the compile payload and the study
    patch operations agree on which statement an index refers to.  ``bcs`` is
    the resolved literal boundary-condition list (None when the argument is
    absent or not a literal container), and ``bc_spans`` are the character
    spans of its elements so single conditions can be edited or deleted.
    """

    index: int
    kind: str
    name: str | None
    variable: str | None
    statement: ast.stmt
    call: ast.Call
    call_span: Span
    bcs: ast.List | ast.Tuple | None
    bcs_span: Span | None
    bc_spans: tuple[Span, ...]


def locate_study_statements(source: str) -> list[StudyStatement] | None:
    """Top-level ``ThermalStudy``/``ElasticStudy`` constructors, in source order.

    Mirrors :func:`locate_constraint_statements`: only unambiguous top-level
    statements (an assignment or a bare expression containing exactly one
    study constructor) are located.  Studies built in loops or functions are
    not — the compile payload detects the count mismatch against the captured
    studies and marks every entry non-editable.

    Args:
        source: The full program text.

    Returns:
        The study statements in source order, or None when the source cannot
        be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    offsets = _line_offsets(source)
    statements: list[StudyStatement] = []
    for item in tree.body:
        if not isinstance(item, (ast.Assign, ast.AnnAssign, ast.Expr)):
            continue
        calls = [node for node in ast.walk(item) if _called_name(node) in STUDY_CALL_KINDS]
        if len(calls) != 1:
            continue
        call = calls[0]
        call_span = _node_span(source, offsets, call)
        if call_span is None:
            continue
        name = next(
            (
                keyword.value.value
                for keyword in call.keywords
                if keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            None,
        )
        variable = None
        if (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
        ):
            variable = item.targets[0].id
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            variable = item.target.id
        bcs_argument = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "bcs"),
            None,
        )
        bcs = _resolved_container(bcs_argument, tree) if bcs_argument is not None else None
        bcs_span = _node_span(source, offsets, bcs) if bcs is not None else None
        bc_spans: list[Span] = []
        for element in bcs.elts if bcs is not None else []:
            span = _node_span(source, offsets, element)
            if span is None:  # pragma: no cover - constructors always carry spans
                bcs = bcs_span = None
                bc_spans = []
                break
            bc_spans.append(span)
        statements.append(
            StudyStatement(
                index=len(statements),
                kind=STUDY_CALL_KINDS[_called_name(call) or ""],
                name=name,
                variable=variable,
                statement=item,
                call=call,
                call_span=call_span,
                bcs=bcs,
                bcs_span=bcs_span,
                bc_spans=tuple(bc_spans),
            )
        )
    return statements


def _runtime_constraint_entry(constraint, vertex_indices: dict[int, int]) -> dict | None:
    """Serialize one runtime constraint whose parameters are profile vertices."""
    kind = CONSTRAINT_CLASS_KINDS.get(constraint.__class__.__name__)
    if kind is None:
        return None
    if kind == "fixed":
        index = vertex_indices.get(id(constraint.param))
        if index is None:
            return None
        return {
            "kind": "fixed",
            "vertices": [index],
            "value": [float(x) for x in constraint.target.reshape(-1)],
        }
    if kind == "distance":
        indices = [
            vertex_indices.get(id(constraint.param1)),
            vertex_indices.get(id(constraint.param2)),
        ]
        if any(index is None for index in indices):
            return None
        return {
            "kind": "distance",
            "vertices": indices,
            "value": float(constraint.distance.value),
        }
    try:
        parameters = constraint.get_parameters()
    except Exception:  # pragma: no cover - defensive against foreign classes
        return None
    indices = [vertex_indices.get(id(parameter)) for parameter in parameters]
    if len(indices) not in {2, 4} or any(index is None for index in indices):
        return None
    return {"kind": kind, "vertices": indices, "value": None}


def _profile_constraints(profile, source: str, line: int | None, traceable: bool) -> list[dict]:
    """Serialize constraints attached to a profile's vertex parameters.

    Every entry carries an ``index``: its position in the serialized list,
    which for traceable profiles equals the ordinal of the matching constraint
    statement in the source. That makes the index a stable identity for the
    ``delete_constraint`` and ``set_constraint_value`` patch operations.
    Constraints visible at runtime but not matched to a statement (built in a
    loop, say) are appended after the statement-backed entries.
    """
    vertex_indices = {id(vertex): index for index, vertex in enumerate(profile.vertices)}
    seen: set[int] = set()
    entries: list[dict] = []
    for vertex in profile.vertices:
        for constraint in vertex.get_constraints():
            if id(constraint) in seen:
                continue
            seen.add(id(constraint))
            entry = _runtime_constraint_entry(constraint, vertex_indices)
            if entry is not None:
                entries.append(entry)

    statements = (
        locate_constraint_statements(source, line) if traceable and line is not None else None
    )
    if statements is None:
        for index, entry in enumerate(entries):
            entry["index"] = index
        return entries

    remaining = list(entries)
    ordered: list[dict] = []
    for position, statement in enumerate(statements):
        match = next(
            (
                entry
                for entry in remaining
                if entry["kind"] == statement.kind
                and tuple(entry["vertices"]) == statement.vertices
            ),
            None,
        )
        if match is None:
            # A statement with no runtime counterpart — e.g. an edge constraint
            # registered on derived parameters rather than the vertices — still
            # occupies its ordinal so delete/set indices stay statement-accurate.
            match = {"kind": statement.kind, "vertices": list(statement.vertices), "value": None}
        else:
            remaining.remove(match)
        match["index"] = position
        ordered.append(match)
    for offset, entry in enumerate(remaining):
        entry["index"] = len(statements) + offset
        ordered.append(entry)
    return ordered


def _profile_operators(source: str, line: int) -> list[dict]:
    """Operators in source that consume the named profile."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    calls = [
        node
        for node in ast.walk(tree)
        if _is_profile_call(node) and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(calls) != 1:
        return []
    statement = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.Assign) and any(node is calls[0] for node in ast.walk(item))
        ),
        None,
    )
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return []
    variable = statement.targets[0].id
    result = []
    for node in ast.walk(tree):
        name = _called_name(node)
        if not (isinstance(node, ast.Call) and name in {"extrude", "revolve", "loft"}):
            continue
        # A loft consumes two profiles, so its chip appears on both sketches.
        references = node.args if name == "loft" else node.args[:1]
        if not any(
            isinstance(argument, ast.Name) and argument.id == variable for argument in references
        ):
            continue
        result.append({"kind": name, "line": node.lineno})
    return result


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
        "constraints": _profile_constraints(profile, source, line, traceable),
        "operators": _profile_operators(source, line) if traceable else [],
        "material": _profile_material(source, line) if traceable else None,
    }


def _primitive_entry(primitive, index: int, line: int | None, source: str, traceable: bool) -> dict:
    """Payload for a construction primitive: outline plus editable placement."""
    from cadjoint.construction.solid import DIMENSIONS

    call = locate_call(source, line, {primitive.kind}) if traceable else None
    arguments = call.arguments if call is not None else {}
    # Missing placement keywords can be added, and parameter-backed values have
    # already been resolved to their defining literals by locate_call().
    editable = call is not None

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
        "constraints": [],
        "operators": [],
        "material": (_primitive_material(source, line, primitive.kind) if traceable else None),
    }
