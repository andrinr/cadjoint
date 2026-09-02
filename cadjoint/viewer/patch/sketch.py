"""Patch operations for sketches: vertices, and the operators that consume them.

A sketch is a ``PolygonProfile`` whose vertex list is the thing the viewer
drags.  The vertex operations are the purest span surgery in the package —
they rewrite exactly the characters of one ``[x, y]`` literal — and they draw
the distinction :class:`~cadjoint.viewer.source_map.calls.ProfileCall` exists
for: coordinates are read through ``element_spans`` (which may resolve into a
``Vector2(value=[...])`` declaration), while insert and delete work on
``list_element_spans`` so a structural edit never lands inside a parameter
constructor.

The operator half — ``extrude``, ``revolve``, ``loft`` — turns a named sketch
into a solid.  All three follow the same shape: refuse when the profile
already feeds an operator, pick a free ``<profile>_body`` variable, extend the
scene first (so the AST spans stay valid), then insert the statement and its
import above the scene assignment.

:func:`set_sketch_plane` is the third kind: it re-plants an existing sketch on
a *reference* — a face of a feature the program already built, or the tangent
plane of a solid at a picked point — by rewriting the sketch's ``plane=``
argument to ``SketchPlane.on(body.cap("+"))`` and friends.  The reference is
written as source, never as coordinates, which is the whole point: the plane
then follows the parent's parameters instead of freezing a snapshot of them.

Add an operation here when it edits a sketch or turns one into geometry.
Constraint statements attached to a sketch live in
:mod:`cadjoint.viewer.patch.constraints`.
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
from cadjoint.viewer.patch.format import _format_coordinate, _format_value, _format_vertex
from cadjoint.viewer.patch.resolvers import (
    _located_feature,
    _located_sketch_call,
    _profile_binding,
    _require_call,
)
from cadjoint.viewer.patch.scene import _extend_scene_with, _scene_assignment
from cadjoint.viewer.source_map.calls import _vertices_argument
from cadjoint.viewer.source_map.nodes import _called_name, _line_offsets, _resolved_container


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
    list_spans = call.list_element_spans
    if index < count:
        start, _ = list_spans[index]
        return _validate(source[:start] + f"{literal}, " + source[start:])
    _, end = list_spans[-1]
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

    list_spans = call.list_element_spans
    start, end = list_spans[index]
    if index < count - 1:
        # Swallow the separator up to the next element.
        end = list_spans[index + 1][0]
    else:
        # Last element: swallow the separator after the previous one.
        start = list_spans[index - 1][1]
    return _validate(source[:start] + source[end:])


def add_sketch(source: str, origin, name: str | None = None) -> str:
    """Insert a standalone parameter-backed polygon sketch."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    assignment = _scene_assignment(tree)
    if assignment is None:
        raise PatchError("Add a `scene = ...` assignment before creating a sketch.")
    taken = _module_names(tree)
    variable = name
    if variable is None:
        index = 1
        while f"sketch{index}" in taken:
            index += 1
        variable = f"sketch{index}"
    statement = (
        f"{variable} = PolygonProfile("
        "[[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]], "
        f"plane=SketchPlane(origin={_format_value(origin)}), name={variable!r})\n"
    )
    offsets = _line_offsets(source)
    insert = offsets[assignment.lineno - 1]
    patched = source[:insert] + statement + source[insert:]
    patched = _ensure_import(patched, ast.parse(patched), "cadjoint.construction", "PolygonProfile")
    patched = _ensure_import(patched, ast.parse(patched), "cadjoint.construction", "SketchPlane")
    return _validate(patched)


def add_extrusion(source: str, line: int, depth: float = 0.5) -> str:
    """Extrude a named sketch and add the generated solid to ``scene``."""
    tree, _, _, profile = _profile_binding(source, line)
    already = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called_name(node) == "extrude"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == profile
    ]
    if already:
        raise PatchError(f"`{profile}` already has an extrusion.")

    taken = _module_names(tree)
    body = f"{profile}_body"
    suffix = 2
    while body in taken:
        body = f"{profile}_body{suffix}"
        suffix += 1

    patched = _extend_scene_with(source, body)
    tree = ast.parse(patched)
    assignment = _scene_assignment(tree)
    offsets = _line_offsets(patched)
    insert = offsets[assignment.lineno - 1]
    statement = f"{body} = extrude({profile}, depth={_format_coordinate(depth)})\n"
    patched = patched[:insert] + statement + patched[insert:]
    patched = _ensure_import(patched, ast.parse(patched), "cadjoint.construction", "extrude")
    return _validate(patched)


def add_revolution(source: str, line: int, offset: float = 0.0) -> str:
    """Revolve a named sketch and add the generated solid to ``scene``."""
    tree, _, _, profile = _profile_binding(source, line)
    already = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called_name(node) in {"extrude", "revolve"}
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == profile
    ]
    if already:
        raise PatchError(f"`{profile}` already has an extrusion or revolution.")

    taken = _module_names(tree)
    body = f"{profile}_body"
    suffix = 2
    while body in taken:
        body = f"{profile}_body{suffix}"
        suffix += 1

    patched = _extend_scene_with(source, body)
    tree = ast.parse(patched)
    assignment = _scene_assignment(tree)
    offsets = _line_offsets(patched)
    insert = offsets[assignment.lineno - 1]
    statement = f"{body} = revolve({profile}, offset={_format_coordinate(offset)})\n"
    patched = patched[:insert] + statement + patched[insert:]
    patched = _ensure_import(patched, ast.parse(patched), "cadjoint.construction", "revolve")
    return _validate(patched)


def add_loft(source: str, line_a: int, line_b: int, height: float = 1.0) -> str:
    """Loft between two named sketches and add the generated solid to ``scene``.

    Both lines must resolve to distinct named ``PolygonProfile`` sketches with
    equal vertex counts, and neither may already feed an operator.

    Args:
        source: The program text.
        line_a: 1-based line of the first (base) sketch's profile call.
        line_b: 1-based line of the second sketch's profile call.
        height: Total loft height along the base profile's plane normal.

    Returns:
        The patched source.

    Raises:
        PatchError: If either sketch cannot be resolved, the sketches are the
            same, the vertex counts differ, or an operator already exists.
    """
    tree, call_a, _, profile_a = _profile_binding(source, line_a)
    tree_b, call_b, _, profile_b = _profile_binding(source, line_b)
    if profile_a == profile_b:
        raise PatchError("Loft needs two different sketches.")

    for profile in (profile_a, profile_b):
        already = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _called_name(node) in {"extrude", "revolve", "loft"}
            and any(
                isinstance(argument, ast.Name) and argument.id == profile for argument in node.args
            )
        ]
        if already:
            raise PatchError(f"`{profile}` already has an operator.")

    counts: dict[str, int] = {}
    for profile, call, owner in ((profile_a, call_a, tree), (profile_b, call_b, tree_b)):
        container = _resolved_container(_vertices_argument(call), owner)
        if container is None:
            raise PatchError(f"Could not count the vertices of `{profile}`.")
        counts[profile] = len(container.elts)
    if counts[profile_a] != counts[profile_b]:
        raise PatchError(
            f"Loft needs equal vertex counts; `{profile_a}` has {counts[profile_a]} and "
            f"`{profile_b}` has {counts[profile_b]}."
        )

    taken = _module_names(tree)
    body = f"{profile_a}_body"
    suffix = 2
    while body in taken:
        body = f"{profile_a}_body{suffix}"
        suffix += 1

    patched = _extend_scene_with(source, body)
    tree = ast.parse(patched)
    assignment = _scene_assignment(tree)
    offsets = _line_offsets(patched)
    insert = offsets[assignment.lineno - 1]
    statement = f"{body} = loft({profile_a}, {profile_b}, height={_format_coordinate(height)})\n"
    patched = patched[:insert] + statement + patched[insert:]
    patched = _ensure_import(patched, ast.parse(patched), "cadjoint.construction", "loft")
    return _validate(patched)


# How each reference kind names its face, and what its argument literal is.
_FACE_ACCESSORS = {
    "cap": ("cap", lambda reference: repr(str(reference["sign"]))),
    "side": ("side", lambda reference: str(int(reference["edge"]))),
    "face": ("face", lambda reference: repr(str(reference["key"]))),
}


def set_sketch_plane(
    source: str,
    line: int,
    reference: dict,
    x_axis=None,
    flip: bool = False,
    offset: float | None = None,
) -> str:
    """Re-plant a sketch on a face, a tangent plane, or a world plane.

    The sketch's ``plane=`` argument is rewritten (or added) to the constructor
    that names the reference, so the plane stays an *expression* over the
    parent's parameters: re-dimension the parent and the sketch follows, and a
    gradient taken through the child reaches the parent.

    Args:
        source: The program text.
        line: 1-based line of the ``PolygonProfile(...)`` call to re-plant.
        reference: The plane to sit on. ``{"kind": "cap", "owner": <line>,
            "sign": "+"}`` and ``{"kind": "side", "owner": <line>, "edge": 2}``
            name a feature's analytic faces; ``{"kind": "face", "owner":
            <line>, "key": "+x"}`` names a primitive's; ``{"kind": "tangent",
            "owner": <line>, "near": [x, y, z]}`` reads the plane off the
            solid's own gradient at a picked point; ``{"kind": "world",
            "origin": [...], "normal": [...]}`` goes back to a stated plane.
        x_axis: Optional three numbers pinning the sketch's in-plane
            horizontal.
        flip: Face the plane the other way (face references only).
        offset: Optional distance to push the plane along its normal.

    Returns:
        The patched source.

    Raises:
        PatchError: If the sketch or the referenced feature cannot be located,
            the feature has no variable to name, or the feature is defined
            after the sketch that would use it.
    """
    _, call, statement = _located_sketch_call(source, line)
    kind = reference.get("kind")
    if kind == "world":
        expression = (
            f"SketchPlane(origin={_format_value(reference['origin'])}, "
            f"normal={_format_value(reference['normal'])}"
        )
        expression += f", x_axis={_format_value(x_axis)})" if x_axis is not None else ")"
    else:
        located = _located_feature(source, reference["owner"])
        if located.statement_line >= statement.lineno:
            raise PatchError(
                f"`{located.variable}` is defined at line {located.statement_line}, at or after "
                f"the sketch at line {statement.lineno}; a sketch can only sit on geometry "
                "built before it."
            )
        if kind == "tangent":
            near = _format_value(reference["near"])
            expression = f"SketchPlane.tangent({located.variable}, near={near}"
            expression += f", x_axis={_format_value(x_axis)})" if x_axis is not None else ")"
        else:
            accessor, render = _FACE_ACCESSORS[kind]
            face = f"{located.variable}.{accessor}({render(reference)})"
            expression = f"SketchPlane.on({face}"
            if x_axis is not None:
                expression += f", x_axis={_format_value(x_axis)}"
            if flip:
                expression += ", flip=True"
            expression += ")"
    if offset:
        expression = f"SketchPlane.offset({expression}, {_format_coordinate(offset)})"

    patched = _set_keyword_expression(source, call, "plane", expression)
    patched = _ensure_import(patched, ast.parse(patched), "cadjoint.construction", "SketchPlane")
    return _validate(patched)
