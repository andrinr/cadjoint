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

from cadjoint.construction.solid import DIMENSIONS as _SOLID_DIMENSIONS
from cadjoint.viewer.patch.edits import (
    _ensure_import,
    _module_names,
    _name_references,
    _operand_span,
    _statement_containing,
    _validate,
)
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.patch.format import _format_keywords, _format_value
from cadjoint.viewer.patch.materials import (
    EDITABLE_PROPERTIES,
    PROPERTY_BOUNDS,
    property_range_error,
)
from cadjoint.viewer.patch.resolvers import CONSTRUCTION_CALLS
from cadjoint.viewer.patch.scene import (
    _all_union_operands,
    _scene_assignment,
    _union_assignments,
    _union_operands,
)
from cadjoint.viewer.source_map import locate_call
from cadjoint.viewer.source_map.features import FEATURE_CALL_KINDS
from cadjoint.viewer.source_map.nodes import (
    _assignment_value,
    _call_namespace,
    _called_name,
    _is_construction_call,
    _line_offsets,
    _node_span,
)

#: ``kind -> {dimension: component count}`` for every primitive the viewer
#: can place — read off the runtime's own table, so a solid the patch writes
#: is one the constructor accepts.  ``size`` is a vector; the rest scalars.
PRIMITIVE_DIMENSIONS: dict[str, dict[str, int]] = {
    kind: {key: (3 if key == "size" else 1) for key in keys}
    for kind, keys in _SOLID_DIMENSIONS.items()
}

#: ``call name -> {argument: component count}``: everything ``set_value``
#: may write, and the shape each value must have.  A keyword outside this
#: table would reach the constructor unchecked, and the program would fail
#: on its next compile instead of the request being refused now.
EDITABLE_CALLS: dict[str, dict[str, int]] = {
    **{
        kind: {"position": 3, "rotation": 3, **dimensions}
        for kind, dimensions in PRIMITIVE_DIMENSIONS.items()
    },
    "extrude": {"depth": 1},
    "revolve": {"offset": 1},
    "loft": {"height": 1},
    "PolygonProfile": {"planeOrigin": 3, "planeNormal": 3},
    # The viewport addresses a sketch's plane call directly when the sketch is
    # dragged as an object: the same two keywords, named as SketchPlane names them.
    "SketchPlane": {"origin": 3, "normal": 3},
    "Material": {"color": 3, **dict.fromkeys(EDITABLE_PROPERTIES, 1)},
}

#: Arguments that are unit intervals or brackets rather than free numbers.
_UNIT_INTERVAL_ARGUMENTS = frozenset({"color"})
_POSITIVE_ARGUMENTS = frozenset({"size", "radius", "height"})


def _checked_value(name: str, argument: str, value) -> str:
    """The literal for *value*, once it has the shape *argument* takes."""
    arguments = EDITABLE_CALLS.get(name)
    if arguments is None:
        allowed = ", ".join(sorted(EDITABLE_CALLS))
        raise PatchError(f"`set_value` edits one of these calls: {allowed}.")
    size = arguments.get(argument)
    if size is None:
        allowed = ", ".join(sorted(arguments))
        raise PatchError(f"`{name}` has no editable argument `{argument}`; expected: {allowed}.")
    if size == 1:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PatchError(f"`{argument}` needs one number.")
        components = [float(value)]
    else:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != size
            or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
            )
        ):
            raise PatchError(f"`{argument}` needs {size} numbers.")
        components = [float(item) for item in value]
    if name == "Material" and argument in PROPERTY_BOUNDS:
        low, high = PROPERTY_BOUNDS[argument]
        if not low <= components[0] <= high:
            raise PatchError(property_range_error(argument))
    if argument in _UNIT_INTERVAL_ARGUMENTS and any(not 0.0 <= item <= 1.0 for item in components):
        raise PatchError(f"`{argument}` needs {size} numbers from 0 to 1.")
    if argument in _POSITIVE_ARGUMENTS and any(item <= 0.0 for item in components):
        raise PatchError(
            f"`{argument}` needs {'a positive number' if size == 1 else f'{size} positive numbers'}."
        )
    if argument == "planeNormal" and not any(abs(item) > 1e-9 for item in components):
        raise PatchError("A sketch-plane normal must not be zero.")
    return _format_value(value)


def _call_at(tree: ast.Module, line: int, name: str) -> ast.Call | None:
    """The one call named *name* at *line*, the way :func:`locate_call` finds it."""
    calls = [node for node in ast.walk(tree) if _called_name(node) == name]
    matches = [node for node in calls if node.lineno == line]
    if not matches:
        matches = [
            node for node in calls if node.lineno <= line <= (node.end_lineno or node.lineno)
        ]
    return matches[0] if len(matches) == 1 else None


def _plane_argument(call: ast.Call) -> ast.AST | None:
    """The ``plane`` argument of a PolygonProfile call, positional or keyword."""
    for keyword in call.keywords:
        if keyword.arg == "plane":
            return keyword.value
    return call.args[1] if len(call.args) > 1 else None


def _plain_sketch_plane(node: ast.AST, tree: ast.Module) -> ast.Call | None:
    """A literal ``SketchPlane(...)`` call, reached directly or through one name.

    ``SketchPlane.on(...)`` and friends are expressions over other geometry
    and have no ``origin`` to rewrite; those answer None.
    """
    if isinstance(node, ast.Name):
        value = _assignment_value(tree, node.id, node.lineno)
        return _plain_sketch_plane(value, tree) if value is not None else None
    if (
        isinstance(node, ast.Call)
        and _called_name(node) == "SketchPlane"
        and _call_namespace(node) is None
    ):
        return node
    return None


def _set_sketch_plane_component(
    source: str, tree: ast.Module, profile: ast.Call, keyword: str, literal: str
) -> str:
    """Rewrite ``origin``/``normal`` of a sketch's literal plane, or give it one."""
    offsets = _line_offsets(source)
    argument = _plane_argument(profile)
    if argument is None:
        ends = [
            span[1]
            for item in [*profile.args, *(keyword.value for keyword in profile.keywords)]
            if (span := _node_span(source, offsets, item)) is not None
        ]
        insert = max(ends)
        patched = _validate(
            source[:insert] + f", plane=SketchPlane({keyword}={literal})" + source[insert:]
        )
        return _ensure_import(patched, ast.parse(patched), "cadjoint.construction", "SketchPlane")
    plane = _plain_sketch_plane(argument, tree)
    if plane is None:
        raise PatchError(
            "The sketch's `plane` is an expression over other geometry, not a literal "
            "`SketchPlane(...)`; re-plant it with `set_sketch_plane` or edit it in the code."
        )
    existing = next((item for item in plane.keywords if item.arg == keyword), None)
    if existing is None:
        ends = [
            span[1]
            for item in [*plane.args, *(keyword.value for keyword in plane.keywords)]
            if (span := _node_span(source, offsets, item)) is not None
        ]
        if not ends:
            _, close = _node_span(source, offsets, plane)
            return _validate(source[: close - 1] + f"{keyword}={literal}" + source[close - 1 :])
        insert = max(ends)
        return _validate(source[:insert] + f", {keyword}={literal}" + source[insert:])
    start, end = _node_span(source, offsets, existing.value)
    return _validate(source[:start] + literal + source[end:])


def set_value(source: str, line: int, name: str, argument: str, value) -> str:
    """Rewrite one keyword argument of a construction call.

    Used for primitive placement — ``position``, ``rotation``, ``size``,
    ``radius`` — where the whole argument is replaced rather than one element.
    :data:`EDITABLE_CALLS` is the contract: which calls, which keywords, and
    the shape of each value.

    Args:
        source: The program text.
        line: 1-based line of the construction call.
        name: The called function's name, e.g. ``box``.
        argument: Keyword to rewrite.
        value: A number, or a sequence of numbers for a vector argument.

    Returns:
        The patched source.

    Raises:
        PatchError: If the call cannot be located, the call does not take
            that argument, the value has the wrong shape, or the keyword is
            already present as an expression this operation will not
            overwrite.
    """
    literal = _checked_value(name, argument, value)
    call = locate_call(source, line, {name})
    if call is None:
        raise PatchError(f"No editable {name}() call found at line {line}.")
    tree = ast.parse(source)
    node = _call_at(tree, line, name)
    if node is None:  # pragma: no cover - locate_call found exactly this call
        raise PatchError(f"No editable {name}() call found at line {line}.")
    if name == "PolygonProfile":
        keyword = "origin" if argument == "planeOrigin" else "normal"
        return _set_sketch_plane_component(source, tree, node, keyword, literal)
    span = call.arguments.get(argument)
    if span is None:
        if any(keyword.arg == argument for keyword in node.keywords):
            # Present, but not a literal this layer may rewrite: appending a
            # second keyword would be a program that no longer compiles.
            raise PatchError(
                f"The {name}'s `{argument}` is not an editable literal; edit it in the code."
            )
        # The keyword is simply absent — a solid written without `rotation=`
        # should still be rotatable, so add it rather than refusing.
        insert = call.arguments_end
        return _validate(source[:insert] + f", {argument}={literal}" + source[insert:])
    start, end = span
    return _validate(source[:start] + literal + source[end:])


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
    expected = PRIMITIVE_DIMENSIONS.get(kind)
    if expected is None:
        allowed = ", ".join(sorted(PRIMITIVE_DIMENSIONS))
        raise PatchError(f"Primitive `kind` must be one of: {allowed}.")
    if not isinstance(dimensions, dict) or set(dimensions) != set(expected):
        listed_keys = ", ".join(f"`{key}`" for key in expected)
        raise PatchError(f"A `{kind}` takes exactly these dimensions: {listed_keys}.")

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

    arguments = _format_keywords({key: dimensions[key] for key in expected})
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


#: Every call ``delete_object`` removes: sketches, primitives, and the
#: features (``extrude``/``revolve``/``loft``) generated from a sketch.
DELETABLE_CALLS = frozenset(CONSTRUCTION_CALLS) | FEATURE_CALL_KINDS


def _union_operand_owner(tree: ast.Module, node: ast.AST) -> ast.Assign | None:
    """The ``name = Union(...)`` assignment *node* is a positional operand of."""
    for assignment in _union_assignments(tree):
        if any(operand is node for operand in assignment.value.args):  # type: ignore[union-attr]
            return assignment
    return None


def delete_object(source: str, line: int) -> str:
    """Remove a construction object and its use in the scene.

    Deletes the statement that builds it and drops it from the scene
    expression. Refuses when the value is used somewhere else, since removing
    it would leave the program referring to a name that no longer exists, and
    when it is the last operand of a ``Union``, since an empty union is not a
    scene the program can build.

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
        if _called_name(node) in DELETABLE_CALLS
        and _is_construction_call(node)
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
            owner = _union_operand_owner(tree, node)
            siblings = list(owner.value.args)  # type: ignore[union-attr]
            if len(siblings) == 1:
                target = owner.targets[0].id if isinstance(owner.targets[0], ast.Name) else "?"
                raise PatchError(
                    f"`{variable}` is the last operand of `{target} = Union(...)`, so deleting "
                    "it would leave an empty union. Remove that union in the code first."
                )
            edits.append(_operand_span(source, offsets, node, siblings))

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
        # A feature's positional argument is the sketch it consumes, which
        # stays; its parameters are the keywords, same as a primitive's.
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
                and _called_name(other) in DELETABLE_CALLS
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
        siblings = _union_operands(scene)
        if not any(operand is call for operand in siblings):
            raise PatchError(
                "This object is not a direct operand of the scene Union, so it cannot be "
                "deleted from the viewer."
            )
        if len(siblings) == 1:
            raise PatchError(
                "This object is the last operand of the scene Union, so deleting it would "
                "leave an empty scene. Remove the union in the code first."
            )
        edits.append(_operand_span(source, offsets, call, siblings))

    patched = source
    for start, end in sorted(edits, reverse=True):
        patched = patched[:start] + patched[end:]
    return _validate(patched)
