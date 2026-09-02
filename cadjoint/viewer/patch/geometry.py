"""Patch operations for solids: placement, creation, and deletion.

This module covers the objects that go straight into the scene —
``Solid.box``/``sphere``/``cylinder`` and, for placement purposes, the sketch
plane of a ``PolygonProfile``.

Three behaviours are worth knowing before extending it:

- :func:`set_value` **adds a keyword that is missing** rather than refusing: a
  solid written without ``rotation=`` should still be rotatable from the
  viewer.  For a profile it rewrites the plane's ``origin``/``normal``,
  synthesising a ``SketchPlane(...)`` argument when the sketch has none.
- :func:`add_primitive` **edits the scene expression before inserting the new
  statement**, because inserting first would shift every span the AST reported.
- :func:`delete_object` **refuses whenever removal would change what the
  program builds**: the name may only be used as a direct operand of the
  scene's ``Union``.  It does clean up top-level constraints left dangling on
  parameters the deleted object exclusively owned — leaving those behind would
  break the next ``satisfy_constraints(scene)``.

Sketch vertices and the operators that consume a sketch live in
:mod:`cadjoint.viewer.patch.sketch`.
"""

from __future__ import annotations

import ast

from cadjoint.viewer.patch.edits import (
    _argument_span,
    _ensure_import,
    _module_names,
    _name_references,
    _statement_containing,
    _validate,
)
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.patch.format import _format_value
from cadjoint.viewer.patch.resolvers import CONSTRUCTION_CALLS
from cadjoint.viewer.patch.scene import _all_union_operands, _scene_assignment, _union_operands
from cadjoint.viewer.source_map import locate_call
from cadjoint.viewer.source_map.nodes import _called_name, _line_offsets, _node_span


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
        # Only a direct operand of a Union can be dropped safely — the scene's
        # or a named sub-assembly's, since a scene often unions parts into a
        # body and the body into the scene. Anywhere else — an argument to
        # extrude(), say — removing it would silently change what the program
        # builds.
        operands = _all_union_operands(tree)
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
