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

from jaxcad.viewer._source_map import (
    _called_name,
    _line_offsets,
    _node_span,
    locate_call,
    locate_profile_call,
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
    if name == "PolygonProfile" and argument == "planeOrigin":
        insert = call.arguments_end
        patched = _validate(
            source[:insert]
            + f", plane=SketchPlane(origin={_format_value(value)})"
            + source[insert:]
        )
        return _ensure_import(
            patched,
            ast.parse(patched),
            "jaxcad.construction",
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


def _ensure_import(source: str, tree: ast.Module, module: str, symbol: str) -> str:
    """Add ``from module import symbol`` when the symbol is not already bound."""
    if symbol in _module_names(tree):
        return source
    offsets = _line_offsets(source)
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
        patched = _ensure_import(patched, ast.parse(patched), "jaxcad.sdf.boolean", "Union")

    # Re-parse: the scene edit moved everything after it.
    tree = ast.parse(patched)
    assignment = _scene_assignment(tree)
    offsets = _line_offsets(patched)
    insert_at = offsets[assignment.lineno - 1]
    patched = patched[:insert_at] + statement + patched[insert_at:]

    patched = _ensure_import(patched, ast.parse(patched), "jaxcad.construction", "Solid")
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
    patched = _ensure_import(patched, ast.parse(patched), "jaxcad.render", "Material")
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
        patched = _ensure_import(patched, ast.parse(patched), "jaxcad.sdf.boolean", "Union")
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
    patched = _ensure_import(patched, ast.parse(patched), "jaxcad.construction", "PolygonProfile")
    patched = _ensure_import(patched, ast.parse(patched), "jaxcad.construction", "SketchPlane")
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
    patched = _ensure_import(patched, ast.parse(patched), "jaxcad.construction", "extrude")
    return _validate(patched)


def add_constraint(
    source: str,
    line: int,
    kind: str,
    indices: list[int],
    value,
) -> str:
    """Attach a fixed or distance constraint to sketch vertices."""
    _, _, statement, profile = _profile_binding(source, line)
    if kind == "fixed" and len(indices) == 1:
        constraint = f"FixedConstraint({profile}.vertices[{indices[0]}], {_format_value(value)})\n"
        symbol = "FixedConstraint"
    elif kind == "distance" and len(indices) == 2:
        constraint = (
            f"DistanceConstraint({profile}.vertices[{indices[0]}], "
            f"{profile}.vertices[{indices[1]}], {_format_value(value)})\n"
        )
        symbol = "DistanceConstraint"
    else:
        raise PatchError("Constraints must be `fixed` with one vertex or `distance` with two.")

    insert = _after_statement(source, statement)
    patched = source[:insert] + constraint + source[insert:]
    patched = _ensure_import(patched, ast.parse(patched), "jaxcad.constraints", symbol)
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
    constraint_names = {"FixedConstraint", "DistanceConstraint"}
    for candidate in statements[index + 1 :]:
        if (
            isinstance(candidate, ast.Expr)
            and isinstance(candidate.value, ast.Call)
            and _called_name(candidate.value) in constraint_names
        ):
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
        patched, ast.parse(patched), "jaxcad.constraints", "satisfy_constraints"
    )
    return _validate(patched)


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
    "add_constraint": add_constraint,
    "solve_sketch": solve_sketch,
    "delete_object": delete_object,
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
