"""Serialize named materials and find which object each one is assigned to.

Materials are the one construction concept the viewer browses by *variable
name*: only a direct module-level ``brass = Material(...)`` assignment is
exposed, so every browser card has a stable Python identifier that can be
written into another call when the user drops the material onto an object.

This module holds both halves of that: building the material browser payload,
and the reverse lookups that tell a primitive or profile entry which named
material it currently carries (a profile answers through the ``extrude`` /
``revolve`` / ``loft`` call that consumes it, since the material lives on the
generator, not the sketch).
"""

from __future__ import annotations

import ast

from cadjoint.viewer.source_map.calls import locate_call
from cadjoint.viewer.source_map.nodes import _called_name, _contains, _is_profile_call, parse_module


def build_material_payload(namespace: dict, source: str) -> list[dict]:
    """Serialize named ``Material`` definitions for the visual material browser.

    Only direct module-level assignments such as ``brass = Material(...)`` are
    exposed. This gives every browser card a stable Python identifier that can
    be written into another construction call when the user drops the material
    onto an object.
    """
    from cadjoint.render.material import Material

    try:
        tree = parse_module(source)
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
                # Stable across every edit that leaves the assignment alone,
                # unlike ``id``, which is a position in this payload.
                "stableId": f"assign:{variable}",
                "name": variable,
                "line": statement.value.lineno,
                "editable": call is not None,
                "color": [float(value) for value in params["color"].value],
                "roughness": float(params["roughness"].value),
                "metallic": float(params["metallic"].value),
                "opacity": float(params["opacity"].value),
                "ior": float(params["ior"].value),
                "reflectivity": float(params["reflectivity"].value),
                # Physical properties (SI) with their units and free flags,
                # for the inspector; absent values are null.
                **{
                    key: value
                    for key, value in material.describe().items()
                    if key in ("physical", "units", "free")
                },
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
        tree = parse_module(source)
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
