"""Rewrite sketch vertex literals in the user's Python source.

Viewer interactions are applied to the program text itself, so the editor stays
the single source of truth. Edits are pure span surgery guided by
:mod:`cadjoint.viewer._source_map`: only the characters of the targeted vertex
literal change, leaving formatting, comments, and the rest of the file byte for
byte identical.

No user code is executed here — the server can patch without spawning the
compile worker.
"""

from __future__ import annotations

import ast

from cadjoint.viewer._source_map import (
    Span,
    StudyStatement,
    _called_name,
    _editable_value_node,
    _line_offsets,
    _node_span,
    _resolved_container,
    _vertices_argument,
    locate_call,
    locate_constraint_statements,
    locate_profile_call,
    locate_study_statements,
)


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


def _format_value(value) -> str:
    """Format a scalar or vector literal."""
    if isinstance(value, (int, float)):
        return _format_coordinate(value)
    return "[" + ", ".join(_format_coordinate(component) for component in value) + "]"


def set_value(source: str, line: int, name: str, argument: str, value) -> str:
    """Rewrite one keyword argument of a construction call.

    Used for primitive placement — ``position``, ``rotation``, ``size``,
    ``radius`` — where the whole argument is replaced rather than one element.

    Args:
        source: The program text.
        line: 1-based line of the construction call.
        name: The called function's name, e.g. ``box``.
        argument: Keyword to rewrite.
        value: A number, or a sequence of numbers for a vector argument.

    Returns:
        The patched source.

    Raises:
        PatchError: If the call or that keyword cannot be located.
    """
    call = locate_call(source, line, {name})
    if call is None:
        raise PatchError(f"No editable {name}() call found at line {line}.")
    if name == "PolygonProfile" and argument in {"planeOrigin", "planeNormal"}:
        keyword = "origin" if argument == "planeOrigin" else "normal"
        plane = locate_call(source, line, {"SketchPlane"})
        if plane is not None:
            # The profile already carries a plane: rewrite that keyword in
            # place, or append it to the SketchPlane call when absent.
            span = plane.arguments.get(keyword)
            if span is None:
                insert = plane.arguments_end
                return _validate(
                    source[:insert] + f", {keyword}={_format_value(value)}" + source[insert:]
                )
            start, end = span
            return _validate(source[:start] + _format_value(value) + source[end:])
        insert = call.arguments_end
        patched = _validate(
            source[:insert]
            + f", plane=SketchPlane({keyword}={_format_value(value)})"
            + source[insert:]
        )
        return _ensure_import(
            patched,
            ast.parse(patched),
            "cadjoint.construction",
            "SketchPlane",
        )
    span = call.arguments.get(argument)
    if span is None:
        # The keyword is simply absent — a solid written without `rotation=`
        # should still be rotatable, so add it rather than refusing.
        insert = call.arguments_end
        return _validate(source[:insert] + f", {argument}={_format_value(value)}" + source[insert:])
    start, end = span
    return _validate(source[:start] + _format_value(value) + source[end:])


def _module_names(tree: ast.Module) -> set[str]:
    """Every name bound at module level, for choosing a fresh variable."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
    return names


def _scene_assignment(tree: ast.Module) -> ast.Assign | None:
    """The module-level ``scene = ...`` statement, if there is one."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "scene" for target in node.targets
        ):
            return node
    return None


def _ensure_import(
    source: str,
    tree: ast.Module,
    module: str,
    symbol: str,
    *,
    prefer_offset: int | None = None,
) -> str:
    """Add ``from module import symbol`` when the symbol is not already bound.

    An existing single-line ``from module import ...`` is extended in place
    rather than adding a new line: line-addressed patch operations identify
    their targets by line number, so imports must not shift the file.  When
    no such line exists and ``prefer_offset`` is given, the new import is
    inserted there (module-level imports are legal anywhere) so callers can
    keep everything above their target line untouched.
    """
    if symbol in _module_names(tree):
        return source
    offsets = _line_offsets(source)
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == module
            and node.level == 0
            and node.lineno == node.end_lineno
            and not any(alias.asname for alias in node.names)
        ):
            _, end = _node_span(source, offsets, node)
            return source[:end] + f", {symbol}" + source[end:]
    if prefer_offset is not None:
        return source[:prefer_offset] + f"from {module} import {symbol}\n" + source[prefer_offset:]
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    insert_line = (imports[-1].end_lineno if imports else 0) or 0
    offset = offsets[insert_line] if insert_line < len(offsets) else len(source)
    return source[:offset] + f"from {module} import {symbol}\n" + source[offset:]


def add_primitive(
    source: str, kind: str, position, dimensions: dict, name: str | None = None
) -> str:
    """Insert a new construction primitive and add it to the scene.

    Writes a ``Solid.<kind>(...)`` statement above the ``scene`` assignment and
    includes the new variable in the scene expression, wrapping it in a
    ``Union`` when it is not one already.

    Args:
        source: The program text.
        kind: ``box``, ``sphere``, or ``cylinder``.
        position: World position for the new solid.
        dimensions: Kind-specific arguments, e.g. ``{"radius": 0.5}``.
        name: Optional variable name; one is generated when omitted.

    Returns:
        The patched source.

    Raises:
        PatchError: If the program has no ``scene`` assignment to extend.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error

    assignment = _scene_assignment(tree)
    if assignment is None:
        raise PatchError("Add a `scene = ...` assignment before placing solids from the viewer.")

    taken = _module_names(tree)
    variable = name
    if variable is None:
        index = 1
        while f"{kind}{index}" in taken:
            index += 1
        variable = f"{kind}{index}"

    arguments = ", ".join(f"{key}={_format_value(value)}" for key, value in dimensions.items())
    statement = (
        f"{variable} = Solid.{kind}({arguments}, position={_format_value(position)}, "
        f'name="{variable}")\n'
    )

    offsets = _line_offsets(source)
    value = assignment.value
    patched = source

    # Extend the scene expression first: inserting the statement above it would
    # otherwise shift every span the AST reported.
    if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "Union":
        anchor = value.args[-1] if value.args else None
        if anchor is None:
            raise PatchError("The scene Union has no operands to extend.")
        _, end = _node_span(patched, offsets, anchor)
        patched = patched[:end] + f", {variable}" + patched[end:]
    else:
        start, end = _node_span(patched, offsets, value)
        patched = patched[:start] + f"Union({patched[start:end]}, {variable})" + patched[end:]
        patched = _ensure_import(patched, ast.parse(patched), "cadjoint.sdf.boolean", "Union")

    # Re-parse: the scene edit moved everything after it.
    tree = ast.parse(patched)
    assignment = _scene_assignment(tree)
    offsets = _line_offsets(patched)
    insert_at = offsets[assignment.lineno - 1]
    patched = patched[:insert_at] + statement + patched[insert_at:]

    patched = _ensure_import(patched, ast.parse(patched), "cadjoint.construction", "Solid")
    return _validate(patched)


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
        f"{variable} = Material(name={variable!r}, color={_format_value(color)}, "
        f"roughness={_format_coordinate(roughness)}, "
        f"metallic={_format_coordinate(metallic)}, "
        f"opacity={_format_coordinate(opacity)}, ior={_format_coordinate(ior)}, "
        f"reflectivity={_format_coordinate(reflectivity)})\n"
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


def _set_keyword_expression(source: str, call: ast.Call, keyword: str, expression: str) -> str:
    """Replace or append a call keyword with a trusted Python expression."""
    offsets = _line_offsets(source)
    existing = next((item for item in call.keywords if item.arg == keyword), None)
    if existing is not None:
        span = _node_span(source, offsets, existing.value)
        if span is None:  # pragma: no cover - defensive
            raise PatchError(f"Could not locate `{keyword}` in the source.")
        start, end = span
        return _validate(source[:start] + expression + source[end:])

    raw_spans = [
        span
        for argument in [*call.args, *(item.value for item in call.keywords)]
        if (span := _node_span(source, offsets, argument)) is not None
    ]
    if not raw_spans:
        raise PatchError("Cannot add a material to a call with no arguments.")
    insert = max(end for _, end in raw_spans)
    return _validate(source[:insert] + f", {keyword}={expression}" + source[insert:])


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


def _profile_binding(source: str, line: int):
    """Return ``(tree, call, statement, variable)`` for a named profile."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    calls = [
        node
        for node in ast.walk(tree)
        if _called_name(node) == "PolygonProfile"
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(calls) != 1:
        raise PatchError(f"No single PolygonProfile() call found at line {line}.")
    call = calls[0]
    statement = _statement_containing(tree, call)
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        raise PatchError("Name the sketch before adding constraints or operators from the viewer.")
    return tree, call, statement, statement.targets[0].id


def _after_statement(source: str, statement: ast.stmt) -> int:
    offsets = _line_offsets(source)
    return offsets[min(statement.end_lineno or statement.lineno, len(offsets) - 1)]


def _extend_scene_with(source: str, variable: str) -> str:
    """Add a named solid to the scene expression."""
    tree = ast.parse(source)
    assignment = _scene_assignment(tree)
    if assignment is None:
        raise PatchError("Add a `scene = ...` assignment before adding an extrusion.")
    offsets = _line_offsets(source)
    value = assignment.value
    if isinstance(value, ast.Call) and _called_name(value) == "Union":
        anchor = value.args[-1] if value.args else None
        if anchor is None:
            raise PatchError("The scene Union has no operands to extend.")
        _, end = _node_span(source, offsets, anchor)
        patched = source[:end] + f", {variable}" + source[end:]
    else:
        start, end = _node_span(source, offsets, value)
        patched = source[:start] + f"Union({source[start:end]}, {variable})" + source[end:]
        patched = _ensure_import(patched, ast.parse(patched), "cadjoint.sdf.boolean", "Union")
    return _validate(patched)


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


_RELATIONAL_CONSTRAINT_SYMBOLS = {
    "horizontal": "HorizontalConstraint",
    "vertical": "VerticalConstraint",
    "coincident": "CoincidentConstraint",
    "parallel": "ParallelEdgesConstraint",
    "perpendicular": "PerpendicularEdgesConstraint",
}


def add_constraint(
    source: str,
    line: int,
    kind: str,
    indices: list[int],
    value=None,
) -> str:
    """Attach a constraint statement to sketch vertices.

    ``fixed`` and ``distance`` carry a numeric ``value``; the relational kinds
    (``horizontal``, ``vertical``, ``coincident`` on two vertices, ``parallel``
    and ``perpendicular`` on two edges given as four vertices) take none.
    """
    _, _, statement, profile = _profile_binding(source, line)

    def vertex(index: int) -> str:
        return f"{profile}.vertices[{index}]"

    if kind == "fixed" and len(indices) == 1:
        constraint = f"FixedConstraint({vertex(indices[0])}, {_format_value(value)})\n"
        symbol = "FixedConstraint"
    elif kind == "distance" and len(indices) == 2:
        constraint = (
            f"DistanceConstraint({vertex(indices[0])}, {vertex(indices[1])}, "
            f"{_format_value(value)})\n"
        )
        symbol = "DistanceConstraint"
    elif kind in {"horizontal", "vertical", "coincident"} and len(indices) == 2:
        symbol = _RELATIONAL_CONSTRAINT_SYMBOLS[kind]
        constraint = f"{symbol}({vertex(indices[0])}, {vertex(indices[1])})\n"
    elif kind in {"parallel", "perpendicular"} and len(indices) == 4:
        symbol = _RELATIONAL_CONSTRAINT_SYMBOLS[kind]
        constraint = f"{symbol}({', '.join(vertex(index) for index in indices)})\n"
    else:
        raise PatchError(
            "Constraints take one vertex (`fixed`), two (`distance`, `horizontal`, "
            "`vertical`, `coincident`), or four (`parallel`, `perpendicular`)."
        )

    # Append after the profile's LAST existing constraint statement so source
    # order equals creation order — the serialized payload indices the viewer
    # chips address are statement ordinals, and prepending would reverse them.
    existing = locate_constraint_statements(source, line)
    anchor = existing[-1].statement if existing else statement
    insert = _after_statement(source, anchor)
    patched = source[:insert] + constraint + source[insert:]
    # Keep everything above the profile untouched: a first-time constraint
    # import lands next to the constraint statement, so repeated
    # line-addressed calls stay valid without recompiling in between.
    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.constraints", symbol, prefer_offset=insert
    )
    return _validate(patched)


def _located_constraint(source: str, line: int, index: int):
    """The profile's ``index``-th constraint statement, payload-order."""
    statements = locate_constraint_statements(source, line)
    if statements is None:
        raise PatchError("Name the sketch before editing constraints from the viewer.")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(statements):
        raise PatchError(
            f"Constraint index {index} is out of range; the sketch has {len(statements)}."
        )
    return statements[index]


def delete_constraint(source: str, line: int, index: int) -> str:
    """Remove one constraint statement, identified by its payload index.

    The index is the ordinal :func:`locate_constraint_statements` assigns —
    the same ordering the viewer payload carries — so a chip's ``index``
    deletes exactly the statement it displays.
    """
    located = _located_constraint(source, line, index)
    offsets = _line_offsets(source)
    start = offsets[located.statement.lineno - 1]
    end = offsets[min(located.statement.end_lineno or located.statement.lineno, len(offsets) - 1)]
    return _validate(source[:start] + source[end:])


def set_constraint_value(source: str, line: int, index: int, value) -> str:
    """Rewrite the numeric target of a fixed or distance constraint.

    Follows named-parameter indirection: when the target is a name bound to
    ``Scalar(value=...)`` (the starter program's ``wall_width`` pattern), the
    Scalar's literal is rewritten so every use of the name stays consistent.
    """
    located = _located_constraint(source, line, index)
    if located.kind == "distance":
        target = located.call.args[2] if len(located.call.args) > 2 else None
    elif located.kind == "fixed":
        target = located.call.args[1] if len(located.call.args) > 1 else None
    else:
        raise PatchError("Only `fixed` and `distance` constraints carry an editable value.")
    if target is None:
        raise PatchError("The constraint statement has no value argument to rewrite.")
    tree = ast.parse(source)
    literal = _editable_value_node(target, tree)
    if literal is None:
        raise PatchError("The constraint value is not an editable literal.")
    offsets = _line_offsets(source)
    start, end = _node_span(source, offsets, literal)

    def exact(component) -> str:
        # Typed-in values must round-trip exactly; the %.4g compaction is for
        # drag-generated coordinates, not numbers the user chose.
        return repr(float(component))

    formatted = (
        exact(value)
        if isinstance(value, (int, float))
        else "[" + ", ".join(exact(component) for component in value) + "]"
    )
    return _validate(source[:start] + formatted + source[end:])


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


def solve_sketch(
    source: str,
    line: int,
    method: str = "newton",
    iterations: int = 8,
) -> str:
    """Add or update an in-program solve step for a sketch's constraints."""
    if method not in {"newton", "adam", "sgd"}:
        raise PatchError("Solver method must be `newton`, `adam`, or `sgd`.")
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or not 1 <= iterations <= 512
    ):
        raise PatchError("Solver iterations must be an integer from 1 to 512.")

    tree, _, profile_statement, profile = _profile_binding(source, line)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _called_name(node) == "satisfy_constraints"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == profile
        ):
            patched = _set_keyword_expression(source, node, "method", repr(method))
            updated_tree = ast.parse(patched)
            updated_call = next(
                candidate
                for candidate in ast.walk(updated_tree)
                if isinstance(candidate, ast.Call)
                and _called_name(candidate) == "satisfy_constraints"
                and candidate.args
                and isinstance(candidate.args[0], ast.Name)
                and candidate.args[0].id == profile
            )
            return _set_keyword_expression(
                patched,
                updated_call,
                "steps",
                str(iterations),
            )

    statements = list(tree.body)
    index = statements.index(profile_statement)
    anchor = profile_statement
    constraint_names = {
        "FixedConstraint",
        "DistanceConstraint",
        "HorizontalConstraint",
        "VerticalConstraint",
        "CoincidentConstraint",
        "EqualLengthConstraint",
        "PointOnLineConstraint",
        "ParallelEdgesConstraint",
        "PerpendicularEdgesConstraint",
    }
    for candidate in statements[index + 1 :]:
        if (
            isinstance(candidate, ast.Expr)
            and isinstance(candidate.value, ast.Call)
            and _called_name(candidate.value) in constraint_names
        ):
            anchor = candidate
            continue
        if isinstance(candidate, (ast.Import, ast.ImportFrom)):
            # Constraint imports are placed beside their statements; they are
            # part of the block the solve step must follow.
            anchor = candidate
            continue
        break

    insert = _after_statement(source, anchor)
    patched = (
        source[:insert]
        + f"satisfy_constraints({profile}, method={method!r}, steps={iterations})\n"
        + source[insert:]
    )
    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.constraints", "satisfy_constraints"
    )
    return _validate(patched)


# ── Simulation studies ───────────────────────────────────────────────────────
# Studies are first-class code citizens (cadjoint.fem.study): the viewer edits
# them by patching their constructor source, exactly like constraints.

_STUDY_CLASSES = {"thermal": "ThermalStudy", "elastic": "ElasticStudy"}
_STUDY_DEFAULTS = {"thermal": "conductivity=1.0", "elastic": "youngs=200.0, poisson=0.3"}
_STUDY_BC_CLASSES = {
    "dirichlet": ("Dirichlet", "value"),
    "heat_flux": ("HeatFlux", "flux"),
    "fixed": ("Fixed", None),
    "traction": ("Traction", "vector"),
}
_STUDY_KIND_BC_TYPES = {
    "thermal": ("dirichlet", "heat_flux"),
    "elastic": ("fixed", "traction"),
}
_BC_CLASS_VALUE_KEYWORDS = {
    "Dirichlet": "value",
    "HeatFlux": "flux",
    "Fixed": None,
    "Traction": "vector",
}
# Constructor field order per kind, for resolving positionally written
# arguments; `name` and `bcs` are excluded from numeric-kwarg editing.
_STUDY_FIELDS = {
    "thermal": ("name", "resolution", "conductivity", "bcs", "source", "bounds", "size"),
    "elastic": ("name", "resolution", "youngs", "poisson", "bcs", "bounds", "size"),
}


def _exact_number(value) -> str:
    """Format a number so it round-trips exactly (viewer-typed values)."""
    return repr(float(value))


def _exact_value(value) -> str:
    """Format a scalar or vector with exact float round-tripping."""
    if isinstance(value, (int, float)):
        return _exact_number(value)
    return "[" + ", ".join(_exact_number(component) for component in value) + "]"


def _render_selection(payload: dict) -> str:
    """Render a normalized selection description as literal ``Nodes`` source."""
    kind = payload["kind"]
    if kind == "box":
        low, high = payload["min_corner"], payload["max_corner"]
        return f"Nodes.box({_exact_value(low)}, {_exact_value(high)})"
    if kind == "sphere":
        return (
            f"Nodes.sphere({_exact_value(payload['center'])}, {_exact_number(payload['radius'])})"
        )
    if kind == "halfspace":
        point, normal = payload["point"], payload["normal"]
        return f"Nodes.halfspace({_exact_value(point)}, {_exact_value(normal)})"
    if kind == "side":
        if payload["tol"] is None:
            return f"Nodes.side({payload['side']!r})"
        return f"Nodes.side({payload['side']!r}, tol={_exact_number(payload['tol'])})"
    if kind in {"and", "or"}:
        left, right = payload["operands"]
        operator = "&" if kind == "and" else "|"
        return f"({_render_selection(left)} {operator} {_render_selection(right)})"
    if kind == "not":
        return f"~{_render_selection(payload['operand'])}"
    raise PatchError(f"Unknown selection kind {kind!r}.")  # pragma: no cover - validated


def _selection_source(description) -> str:
    """Validate a selection description and render it as literal source.

    Round-trips through :func:`cadjoint.fem.selection.selection_from_description`
    so only selections the runtime can rebuild are ever written — predicate
    descriptions (non-serializable) are rejected here with their own message.
    """
    from cadjoint.fem.selection import selection_from_description

    if not isinstance(description, dict):
        raise PatchError("The boundary condition needs `selection` as a description object.")
    try:
        selection = selection_from_description(description)
    except (KeyError, TypeError, ValueError) as error:
        raise PatchError(f"Invalid node selection: {error}") from error
    return _render_selection(selection.describe())


def _located_study(source: str, study) -> StudyStatement:
    """Resolve a study reference — payload index or name — to its statement."""
    statements = locate_study_statements(source)
    if statements is None:
        raise PatchError("Source is not valid Python.")
    if isinstance(study, bool) or not isinstance(study, (int, str)):
        raise PatchError("A study is referenced by its name or its non-negative index.")
    if isinstance(study, int):
        if not 0 <= study < len(statements):
            raise PatchError(
                f"Study index {study} is out of range; the program declares {len(statements)}."
            )
        return statements[study]
    matches = [
        statement for statement in statements if study in (statement.name, statement.variable)
    ]
    if len(matches) != 1:
        declared = ", ".join(
            repr(statement.name or statement.variable or f"#{statement.index}")
            for statement in statements
        )
        raise PatchError(
            f"No single study named {study!r}; the program declares: {declared or 'none'}."
        )
    return matches[0]


def _reject_predicate_bc(element: ast.AST) -> None:
    """Refuse edits to a BC built on a non-serializable predicate selection."""
    if any(_called_name(node) == "predicate" for node in ast.walk(element)):
        raise PatchError(
            "This boundary condition uses a `Nodes.predicate` selection, which is not "
            "serializable; edit it directly in the code."
        )


def add_study(source: str, kind: str, name: str | None = None) -> str:
    """Declare a new simulation study at the end of the scene program.

    Appends a ``ThermalStudy``/``ElasticStudy`` constructor after the last
    existing study (or after the ``scene`` assignment when there is none) and
    imports the constructor from ``cadjoint.fem`` beside it, keeping every line
    above the insertion untouched.

    Args:
        source: The program text.
        kind: ``thermal`` or ``elastic``.
        name: Optional study display name; the generated variable name is
            used when omitted.

    Returns:
        The patched source.

    Raises:
        PatchError: On an unknown kind, a duplicate name, or a program
            without a ``scene`` assignment to anchor the study.
    """
    symbol = _STUDY_CLASSES.get(kind)
    if symbol is None:
        raise PatchError("Study `kind` must be `thermal` or `elastic`.")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    studies = locate_study_statements(source) or []
    taken_names = {study.name for study in studies if study.name is not None}
    if name is not None and name in taken_names:
        raise PatchError(f"A study named {name!r} already exists.")
    taken = _module_names(tree)
    index = 1
    while f"study{index}" in taken or f"study{index}" in taken_names:
        index += 1
    variable = f"study{index}"
    study_name = name if name is not None else variable

    if studies:
        anchor = studies[-1].statement
    else:
        anchor = _scene_assignment(tree)
        if anchor is None:
            raise PatchError(
                "Add a `scene = ...` assignment before declaring studies from the viewer."
            )
    statement = (
        f"{variable} = {symbol}(name={study_name!r}, resolution=20, "
        f"{_STUDY_DEFAULTS[kind]}, bcs=[])\n"
    )
    insert = _after_statement(source, anchor)
    patched = source[:insert] + statement + source[insert:]
    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.fem", symbol, prefer_offset=insert
    )
    return _validate(patched)


def delete_study(source: str, study) -> str:
    """Remove one study declaration, identified by payload index or name."""
    located = _located_study(source, study)
    if located.variable is not None:
        tree = ast.parse(source)
        uses = _name_references(tree, located.variable, located.statement)
        if uses:
            raise PatchError(
                f"`{located.variable}` is used elsewhere in the program, so it cannot be "
                "deleted from the viewer. Remove those uses first."
            )
    offsets = _line_offsets(source)
    start = offsets[located.statement.lineno - 1]
    end = offsets[min(located.statement.end_lineno or located.statement.lineno, len(offsets) - 1)]
    return _validate(source[:start] + source[end:])


def add_study_bc(source: str, study, bc_type: str, selection, value=None) -> str:
    """Append a boundary condition to a study's literal ``bcs`` list.

    Writes literal source such as
    ``Dirichlet(Nodes.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), value=300.0)``,
    with viewer-typed values formatted via exact float ``repr``.  The BC
    class and ``Nodes`` are imported from ``cadjoint.fem`` beside the study so
    every line above it stays untouched.

    Args:
        source: The program text.
        study: Study reference — payload index or name.
        bc_type: ``dirichlet``, ``heat_flux``, ``fixed``, or ``traction``.
        selection: Serializable node-selection description dict.
        value: Scalar for ``dirichlet``/``heat_flux``, 3-vector for
            ``traction``; ``fixed`` takes none.

    Returns:
        The patched source.

    Raises:
        PatchError: On an unknown or kind-incompatible BC type, an invalid
            selection or value, or a ``bcs`` argument that is not an
            editable literal list.
    """
    located = _located_study(source, study)
    if bc_type not in _STUDY_BC_CLASSES:
        allowed = ", ".join(_STUDY_BC_CLASSES)
        raise PatchError(f"`bc_type` must be one of: {allowed}.")
    if bc_type not in _STUDY_KIND_BC_TYPES[located.kind]:
        allowed = ", ".join(_STUDY_KIND_BC_TYPES[located.kind])
        raise PatchError(
            f"A {located.kind} study accepts {allowed} boundary conditions, not `{bc_type}`."
        )
    symbol, value_keyword = _STUDY_BC_CLASSES[bc_type]
    nodes_source = _selection_source(selection)
    if value_keyword is None:
        if value is not None:
            raise PatchError("A `fixed` boundary condition takes no value.")
        bc_source = f"{symbol}({nodes_source})"
    else:
        if bc_type == "traction":
            if not (isinstance(value, (list, tuple)) and len(value) == 3):
                raise PatchError("A `traction` boundary condition needs `value` as three numbers.")
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PatchError(f"A `{bc_type}` boundary condition needs a numeric `value`.")
        bc_source = f"{symbol}({nodes_source}, {value_keyword}={_exact_value(value)})"

    offsets = _line_offsets(source)
    anchor_line = located.statement.lineno
    if located.bcs is not None:
        anchor_line = min(anchor_line, located.bcs.lineno)
    import_offset = offsets[anchor_line - 1]

    if located.bcs is not None and located.bc_spans:
        _, end = located.bc_spans[-1]
        patched = source[:end] + f", {bc_source}" + source[end:]
    elif located.bcs is not None:
        _, end = located.bcs_span
        patched = source[: end - 1] + bc_source + source[end - 1 :]
    else:
        if any(keyword.arg == "bcs" for keyword in located.call.keywords):
            raise PatchError("The study's `bcs` argument is not an editable literal list.")
        patched = _set_keyword_expression(source, located.call, "bcs", f"[{bc_source}]")

    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.fem", symbol, prefer_offset=import_offset
    )
    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.fem", "Nodes", prefer_offset=import_offset
    )
    return _validate(patched)


def _located_study_bc(located: StudyStatement, bc) -> tuple[ast.AST, Span]:
    """The ``bc``-th element of a study's literal BC list, with its span."""
    if located.bcs is None:
        raise PatchError("The study's `bcs` argument is not an editable literal list.")
    count = len(located.bc_spans)
    if not isinstance(bc, int) or isinstance(bc, bool) or not 0 <= bc < count:
        raise PatchError(f"Boundary-condition index {bc} is out of range; the study has {count}.")
    element = located.bcs.elts[bc]
    _reject_predicate_bc(element)
    return element, located.bc_spans[bc]


def delete_study_bc(source: str, study, bc) -> str:
    """Remove one boundary condition from a study's literal ``bcs`` list."""
    located = _located_study(source, study)
    _located_study_bc(located, bc)
    spans = located.bc_spans
    start, end = spans[bc]
    if bc < len(spans) - 1:
        # Swallow the separator up to the next element.
        end = spans[bc + 1][0]
    elif bc > 0:
        # Last element: swallow the separator after the previous one.
        start = spans[bc - 1][1]
    return _validate(source[:start] + source[end:])


def _set_study_bc_value(source: str, located: StudyStatement, bc, value) -> str:
    """Rewrite the numeric payload of one boundary condition in place."""
    element, _ = _located_study_bc(located, bc)
    class_name = _called_name(element) or ""
    if class_name not in _BC_CLASS_VALUE_KEYWORDS or not isinstance(element, ast.Call):
        raise PatchError("This boundary condition is not an editable constructor call.")
    value_keyword = _BC_CLASS_VALUE_KEYWORDS[class_name]
    if value_keyword is None:
        raise PatchError("A `Fixed` boundary condition has no value to edit.")
    if class_name == "Traction":
        if not (isinstance(value, (list, tuple)) and len(value) == 3):
            raise PatchError("A `Traction` boundary condition needs `value` as three numbers.")
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PatchError(f"A `{class_name}` boundary condition needs a numeric `value`.")
    target = next(
        (keyword.value for keyword in element.keywords if keyword.arg == value_keyword),
        element.args[1] if len(element.args) > 1 else None,
    )
    if target is None:
        raise PatchError("The boundary condition has no value argument to rewrite.")
    tree = ast.parse(source)
    literal = _editable_value_node(target, tree)
    if literal is None:
        raise PatchError("The boundary-condition value is not an editable literal.")
    offsets = _line_offsets(source)
    start, end = _node_span(source, offsets, literal)
    return _validate(source[:start] + _exact_value(value) + source[end:])


def _format_study_argument(argument: str, value) -> str:
    """Format one study keyword value, keeping ``resolution`` integral."""
    if argument == "resolution":
        components = value if isinstance(value, (list, tuple)) else [value]
        if len(components) not in {1, 3}:
            raise PatchError("`resolution` must be an integer or three integers.")
        integers = []
        for component in components:
            if (
                not isinstance(component, (int, float))
                or isinstance(component, bool)
                or float(component) != int(component)
                or int(component) < 1
            ):
                raise PatchError("`resolution` must be positive whole numbers.")
            integers.append(str(int(component)))
        return integers[0] if len(integers) == 1 else "[" + ", ".join(integers) + "]"
    if argument in {"bounds", "size"}:
        if not (isinstance(value, (list, tuple)) and len(value) == 3):
            raise PatchError(f"`{argument}` must be three numbers.")
        return _exact_value(value)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PatchError(f"`{argument}` must be a number.")
    return _exact_number(value)


def _set_study_argument(source: str, located: StudyStatement, argument, value) -> str:
    """Rewrite one numeric study keyword in place (or add it when absent)."""
    fields = _STUDY_FIELDS[located.kind]
    if not isinstance(argument, str) or argument not in fields or argument in {"name", "bcs"}:
        allowed = ", ".join(field for field in fields if field not in {"name", "bcs"})
        raise PatchError(f"A {located.kind} study's editable arguments are: {allowed}.")
    expression = _format_study_argument(argument, value)
    target = next(
        (keyword.value for keyword in located.call.keywords if keyword.arg == argument),
        None,
    )
    if target is None:
        position = fields.index(argument)
        if position < len(located.call.args):
            target = located.call.args[position]
    if target is None:
        return _set_keyword_expression(source, located.call, argument, expression)
    tree = ast.parse(source)
    literal = _editable_value_node(target, tree)
    if literal is None:
        raise PatchError(f"The study's `{argument}` value is not an editable literal.")
    offsets = _line_offsets(source)
    start, end = _node_span(source, offsets, literal)
    return _validate(source[:start] + expression + source[end:])


def set_study_value(source: str, study, value, bc=None, argument=None) -> str:
    """Edit a BC's scalar/vector value or a study's numeric keyword in place.

    Args:
        source: The program text.
        study: Study reference — payload index or name.
        value: The new number(s); written with exact float ``repr`` so typed
            values round-trip (``resolution`` stays integral).
        bc: Index of the boundary condition whose value to rewrite.
        argument: Study keyword to rewrite instead (``resolution``,
            ``conductivity``, ``source``, ``youngs``, ``poisson``,
            ``bounds``, ``size``).

    Returns:
        The patched source.

    Raises:
        PatchError: When neither or both of ``bc``/``argument`` are given, or
            the target cannot be rewritten safely.
    """
    if (bc is None) == (argument is None):
        raise PatchError("set_study_value needs exactly one of `bc` or `argument`.")
    located = _located_study(source, study)
    if bc is not None:
        return _set_study_bc_value(source, located, bc, value)
    return _set_study_argument(source, located, argument, value)


CONSTRUCTION_CALLS = {"PolygonProfile", "box", "sphere", "cylinder"}


def _statement_containing(tree: ast.Module, node: ast.AST) -> ast.stmt | None:
    """The module-level statement whose subtree holds *node*."""
    for statement in tree.body:
        if any(candidate is node for candidate in ast.walk(statement)):
            return statement
    return None


def _name_references(tree: ast.Module, name: str, exclude: ast.AST) -> list[ast.Name]:
    """Every load of *name* outside the statement that defines it."""
    return [
        node
        for statement in tree.body
        if statement is not exclude
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
    ]


def delete_object(source: str, line: int) -> str:
    """Remove a construction object and its use in the scene.

    Deletes the statement that builds it and drops it from the scene
    expression. Refuses when the value is used somewhere else, since removing
    it would leave the program referring to a name that no longer exists.

    Args:
        source: The program text.
        line: 1-based line of the construction call to remove.

    Returns:
        The patched source.

    Raises:
        PatchError: If the object cannot be located or is still in use.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error

    calls = [
        node
        for node in ast.walk(tree)
        if _called_name(node) in CONSTRUCTION_CALLS
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(calls) != 1:
        raise PatchError(f"No single construction call found at line {line}.")
    call = calls[0]

    statement = _statement_containing(tree, call)
    if statement is None:
        raise PatchError("Could not find the statement that builds this object.")

    offsets = _line_offsets(source)
    scene = _scene_assignment(tree)
    edits: list[tuple[int, int]] = []

    # A solid written straight into the scene has no statement of its own to
    # remove — only its operand — so the scene assignment never takes the
    # named-variable path, which would delete the whole scene.
    if (
        statement is not scene
        and isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        variable = statement.targets[0].id
        uses = _name_references(tree, variable, statement)
        # Only a direct operand of the scene's Union can be dropped safely.
        # Anywhere else — an argument to extrude(), say — removing it would
        # silently change what the program builds.
        operands = _union_operands(scene)
        if any(not any(operand is node for operand in operands) for node in uses):
            raise PatchError(
                f"`{variable}` is used elsewhere in the program, so it cannot be deleted "
                "from the viewer. Remove those uses first."
            )
        for node in uses:
            edits.append(_argument_span(source, offsets, node))

        # If this object owns named geometric parameters, remove top-level
        # constraints that reference those now-orphaned parameters. Leaving a
        # DistanceConstraint to a deleted object's position would make the
        # next satisfy_constraints(scene) fail because that parameter is no
        # longer part of the scene tree.
        parameter_names = {
            keyword.value.id
            for keyword in call.keywords
            if keyword.arg not in {"material", "name"} and isinstance(keyword.value, ast.Name)
        }
        constraint_calls = {
            "FixedConstraint",
            "DistanceConstraint",
            "AngleConstraint",
            "ParallelConstraint",
            "PerpendicularConstraint",
            "HorizontalConstraint",
            "VerticalConstraint",
            "CoincidentConstraint",
            "EqualLengthConstraint",
            "PointOnLineConstraint",
            "ParallelEdgesConstraint",
            "PerpendicularEdgesConstraint",
        }
        for parameter in parameter_names:
            shared = any(
                other is not call
                and _called_name(other) in CONSTRUCTION_CALLS
                and any(
                    isinstance(candidate, ast.Name)
                    and isinstance(candidate.ctx, ast.Load)
                    and candidate.id == parameter
                    for candidate in ast.walk(other)
                )
                for other in ast.walk(tree)
                if isinstance(other, ast.Call)
            )
            if shared:
                continue
            for candidate in tree.body:
                if not (
                    isinstance(candidate, ast.Expr)
                    and isinstance(candidate.value, ast.Call)
                    and _called_name(candidate.value) in constraint_calls
                    and any(
                        isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id == parameter
                        for node in ast.walk(candidate.value)
                    )
                ):
                    continue
                start = offsets[candidate.lineno - 1]
                end = offsets[min(candidate.end_lineno or candidate.lineno, len(offsets) - 1)]
                edits.append((start, end))

        # Whole statement, including its line ending.
        start = offsets[statement.lineno - 1]
        end = offsets[min(statement.end_lineno or statement.lineno, len(offsets) - 1)]
        edits.append((start, end))
    else:
        # Built inline inside the scene expression: drop just that argument.
        if not any(operand is call for operand in _union_operands(scene)):
            raise PatchError(
                "This object is not a direct operand of the scene Union, so it cannot be "
                "deleted from the viewer."
            )
        edits.append(_argument_span(source, offsets, call))

    patched = source
    for start, end in sorted(edits, reverse=True):
        patched = patched[:start] + patched[end:]
    return _validate(patched)


def _contains_node(outer: ast.AST, inner: ast.AST) -> bool:
    return any(node is inner for node in ast.walk(outer))


def _union_operands(scene: ast.Assign | None) -> list[ast.AST]:
    """Positional arguments of a ``scene = Union(...)`` assignment."""
    if scene is None or not isinstance(scene.value, ast.Call):
        return []
    if getattr(scene.value.func, "id", "") != "Union":
        return []
    return list(scene.value.args)


def _argument_span(source: str, offsets, node) -> tuple[int, int]:
    """Span of one call argument, including the comma that follows it."""
    span = _node_span(source, offsets, node)
    if span is None:  # pragma: no cover - defensive
        raise PatchError("Could not locate the argument to remove.")
    start, end = span
    while end < len(source) and source[end] in ", ":
        end += 1
    return start, end


OPERATIONS = {
    "set_vertex": set_vertex,
    "insert_vertex": insert_vertex,
    "delete_vertex": delete_vertex,
    "set_value": set_value,
    "add_primitive": add_primitive,
    "add_material": add_material,
    "assign_material": assign_material,
    "add_sketch": add_sketch,
    "add_extrusion": add_extrusion,
    "add_revolution": add_revolution,
    "add_loft": add_loft,
    "add_constraint": add_constraint,
    "delete_constraint": delete_constraint,
    "set_constraint_value": set_constraint_value,
    "solve_sketch": solve_sketch,
    "delete_object": delete_object,
    "add_study": add_study,
    "delete_study": delete_study,
    "add_study_bc": add_study_bc,
    "delete_study_bc": delete_study_bc,
    "set_study_value": set_study_value,
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
