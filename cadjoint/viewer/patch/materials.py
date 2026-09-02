"""Patch operations for materials: defining one, and assigning it to an object.

Materials are the one construction concept addressed by *variable name* rather
than by line, so both operations here are about producing and consuming a
stable Python identifier.

Two placement rules:

- a new ``Material(...)`` definition goes **directly after the imports**, not
  above the scene assignment: a drag can assign a material to an object
  declared anywhere, and the name has to be bound before that object's
  statement runs;
- assigning to a sketch does not touch the sketch — a profile carries no
  material, so the keyword is written on the single ``extrude(...)`` that
  consumes it, and the operation refuses when there is not exactly one.

Add an operation here when it concerns material definitions or the
``material=`` keyword on a construction call.
"""

from __future__ import annotations

import ast

from cadjoint.viewer.patch.edits import (
    _ensure_import,
    _module_names,
    _set_keyword_expression,
    _validate,
)
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.patch.format import _format_keywords
from cadjoint.viewer.patch.resolvers import CONSTRUCTION_CALLS, _profile_binding
from cadjoint.viewer.patch.scene import _scene_assignment
from cadjoint.viewer.source_map.nodes import _called_name, _line_offsets


def add_material(
    source: str,
    color,
    roughness: float = 0.4,
    metallic: float = 0.0,
    opacity: float = 1.0,
    ior: float = 1.45,
    reflectivity: float = 0.0,
    name: str | None = None,
) -> str:
    """Create a named material definition above the scene assignment."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    assignment = _scene_assignment(tree)
    if assignment is None:
        raise PatchError("Add a `scene = ...` assignment before creating a material.")

    taken = _module_names(tree)
    variable = name
    if variable is None:
        index = 1
        while f"material{index}" in taken:
            index += 1
        variable = f"material{index}"
    if not variable.isidentifier() or variable in taken:
        raise PatchError(f"`{variable}` is not an available Python material name.")

    statement = (
        f"{variable} = Material(name={variable!r}, "
        + _format_keywords(
            {
                "color": color,
                "roughness": roughness,
                "metallic": metallic,
                "opacity": opacity,
                "ior": ior,
                "reflectivity": reflectivity,
            }
        )
        + ")\n"
    )
    # Materials must be defined before any earlier object can reference them
    # after a drag assignment, so place new definitions directly after imports.
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    offsets = _line_offsets(source)
    insert_line = (imports[-1].end_lineno if imports else 0) or 0
    insert = offsets[insert_line] if insert_line < len(offsets) else len(source)
    patched = source[:insert] + statement + source[insert:]
    patched = _ensure_import(patched, ast.parse(patched), "cadjoint.render", "Material")
    return _validate(patched)


def assign_material(source: str, line: int, material: str) -> str:
    """Assign a named material to a primitive or a profile's extrusion."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    definitions = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == material
        and isinstance(statement.value, ast.Call)
        and _called_name(statement.value) == "Material"
    ]
    if len(definitions) != 1:
        raise PatchError(f"`{material}` is not a named Material definition.")

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called_name(node) in CONSTRUCTION_CALLS
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(calls) != 1:
        raise PatchError(f"No single construction call found at line {line}.")
    call = calls[0]
    if _called_name(call) == "PolygonProfile":
        _, _, _, profile = _profile_binding(source, line)
        extrusions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _called_name(node) == "extrude"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == profile
        ]
        if len(extrusions) != 1:
            raise PatchError(
                f"`{profile}` needs one named extrusion before a material can be assigned."
            )
        call = extrusions[0]
    return _set_keyword_expression(source, call, "material", material)
