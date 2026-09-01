"""Turn captured construction objects into the viewer's JSON payload.

This is the top of the source-map package: it joins the two halves — the
runtime objects captured by :mod:`cadjoint.viewer.source_map.capture` and the
character spans recovered by the locator modules — into one dict per object.
Every entry carries a world-space ``edges`` wireframe so the viewer can draw
any shape without knowing its topology; sketch profiles add their plane,
per-vertex handles, constraints and operators, primitives add their placement
and the spans that make it editable.

Add code here when it decides *what the viewer sees*.  The rule the entries
enforce is that a payload field is only marked editable when the matching
source span was located unambiguously, so a non-traceable object still renders
— it simply cannot be dragged.
"""

from __future__ import annotations

import ast
from collections import Counter

from cadjoint.viewer.source_map.calls import locate_call, locate_profile_call
from cadjoint.viewer.source_map.constraints import (
    CONSTRAINT_CLASS_KINDS,
    locate_constraint_statements,
)
from cadjoint.viewer.source_map.materials import _primitive_material, _profile_material
from cadjoint.viewer.source_map.nodes import (
    Span,
    _called_name,
    _is_profile_call,
    _resolved_call,
)


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
