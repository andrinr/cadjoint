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
