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
  material, so the keyword is written on the single operator (``extrude``,
  ``revolve`` or ``loft``) that consumes it, and the operation refuses when
  there is not exactly one; a feature addressed directly takes the keyword
  itself.

:func:`set_material_property` is the third operation, and the only one that
edits a material's *own* arguments.  A material carries optical properties
(always stated, because they have defaults) and physical ones in SI (stated
only when the scene cares), so the operation has to *add* a keyword as often
as it rewrites one, and ``value=None`` takes one back out.  That asymmetry —
an unstated property has no span for the inspector to drag — is the whole
reason it exists rather than reusing :func:`~.geometry.set_value`.

Add an operation here when it concerns material definitions, their property
keywords, or the ``material=`` keyword on a construction call.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass

# ``_BOUNDS`` is the one table of what a material property may hold: the
# brackets ``Material`` itself enforces when ``free=True``.  Restating them
# here would let the viewer accept a value the optimizer would then refuse,
# so the private name is imported rather than copied.
from cadjoint.render.material import (
    _BOUNDS,
    OPTICAL_PROPERTIES,
    PHYSICAL_PROPERTIES,
    UNITS,
    _display,
)
from cadjoint.viewer.patch.edits import (
    _ensure_import,
    _module_names,
    _set_keyword_expression,
    _validate,
)
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.patch.format import _exact_number, _format_keywords
from cadjoint.viewer.patch.resolvers import CONSTRUCTION_CALLS, _profile_binding
from cadjoint.viewer.patch.scene import _scene_assignment
from cadjoint.viewer.source_map.features import FEATURE_CALL_KINDS
from cadjoint.viewer.source_map.nodes import (
    _called_name,
    _editable_value_node,
    _is_construction_call,
    _line_offsets,
    _node_span,
)

#: Longest line the patched program is allowed to grow to before a keyword
#: wraps onto its own line.  Matches the repository's ruff ``line-length``.
_COLUMN_LIMIT = 100


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
        and _called_name(node) in CONSTRUCTION_CALLS | FEATURE_CALL_KINDS
        and _is_construction_call(node)
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(calls) != 1:
        raise PatchError(f"No single construction call found at line {line}.")
    call = calls[0]
    if _called_name(call) == "PolygonProfile":
        # A profile carries no material: the keyword goes on the one solid
        # generated from it, whichever operator that is.
        _, _, _, profile = _profile_binding(source, line)
        features = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _called_name(node) in FEATURE_CALL_KINDS
            and any(
                isinstance(argument, ast.Name) and argument.id == profile for argument in node.args
            )
        ]
        if len(features) != 1:
            raise PatchError(
                f"`{profile}` needs one operator (extrude, revolve or loft) before a material "
                "can be assigned."
            )
        call = features[0]
    return _set_keyword_expression(source, call, "material", material)


# ── Material properties ─────────────────────────────────────────────────────
#
# The inspector edits a material's own arguments through one operation.  Both
# families of property go through it: the scalar optical ones so that the
# inspector has a single request shape for every row it draws, and the seven
# physical ones — the point of the exercise — which are optional in
# ``Material`` and therefore usually *absent* from the call.

#: Every property ``set_material_property`` may write, in inspector order.
#: ``color`` is a vector rather than a scalar and keeps its own editor
#: (``add_material``/``set_value``), so it is deliberately not here.
EDITABLE_PROPERTIES: tuple[str, ...] = (
    *(key for key in OPTICAL_PROPERTIES if key != "color"),
    *PHYSICAL_PROPERTIES,
)

#: ``property -> (low, high)``: the same brackets ``Material`` enforces when
#: ``free=True``, so a value the viewer accepts is one the optimizer accepts.
PROPERTY_BOUNDS: dict[str, tuple[float, float]] = {
    key: (float(_BOUNDS[key][0]), float(_BOUNDS[key][1])) for key in EDITABLE_PROPERTIES
}

#: ``property -> SI unit``, named in every rejection so a refused number says
#: what it was measured in.  Optical properties are dimensionless ratios.
PROPERTY_UNITS: dict[str, str] = {key: UNITS.get(key, "-") for key in EDITABLE_PROPERTIES}

_CATALOGUE_REFUSAL = (
    "`{variable}` is built by the catalogue factory `{factory}()`, which has no property "
    "keyword to edit. Convert it to a literal `Material(...)` first — send this request "
    "again with `expand: true` to have that done for you."
)


def _magnitude(value: float) -> str:
    """A bound as a person would write it: ``1e12``, not ``1e+12``."""
    text = f"{value:g}"
    if "e" not in text:
        return text
    mantissa, _, exponent = text.partition("e")
    return f"{mantissa}e{int(exponent)}"


def property_range_error(key: str) -> str:
    """The rejection for a value outside one property's bracket.

    Args:
        key: A member of :data:`EDITABLE_PROPERTIES`.

    Returns:
        The message, naming the bracket and the unit it is measured in.
    """
    low, high = PROPERTY_BOUNDS[key]
    unit = PROPERTY_UNITS[key]
    measure = "(dimensionless)" if unit == "-" else unit
    return f"`{key}` must be a number from {_magnitude(low)} to {_magnitude(high)} {measure}."


@dataclass(frozen=True)
class _Definition:
    """One module-level material binding, literal or catalogue-built."""

    variable: str
    name: str | None
    index: int | None
    """Position among the literal definitions, i.e. the payload's index."""
    call: ast.Call
    statement: ast.Assign
    factory: str | None
    """The catalogue factory that built it, or None for ``Material(...)``."""


def _literal_name(call: ast.Call) -> str | None:
    """The string passed as ``name=`` to a call, when it is a literal."""
    return next(
        (
            keyword.value.value
            for keyword in call.keywords
            if keyword.arg == "name"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ),
        None,
    )


def _catalogue_factories() -> dict[str, str]:
    """``factory name -> catalogue key``, including the US spelling alias.

    Imported lazily: the refusal path is the only one that needs it, and the
    catalogue builds every material in it on import of its own module.
    """
    from cadjoint.materials import CATALOGUE

    return {**{key: key for key in CATALOGUE}, "aluminum_6061": "aluminium_6061"}


def _material_definitions(tree: ast.Module) -> list[_Definition]:
    """Every module-level material binding, in source order.

    Both shapes are collected: ``brass = Material(...)``, which the payload
    publishes and this operation can edit, and ``brass = copper_c11000()``,
    which it can only refuse or expand.  Collecting the second is what lets
    the refusal name the factory instead of saying the material is unknown.
    """
    factories = _catalogue_factories()
    definitions: list[_Definition] = []
    ordinal = 0
    for statement in tree.body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
        ):
            continue
        call = statement.value
        called = _called_name(call)
        if called == "Material":
            index, factory = ordinal, None
            ordinal += 1
        elif called in factories:
            index, factory = None, factories[called]
        else:
            continue
        definitions.append(
            _Definition(
                variable=statement.targets[0].id,
                name=_literal_name(call),
                index=index,
                call=call,
                statement=statement,
                factory=factory,
            )
        )
    return definitions


def _located_material(tree: ast.Module, reference, line: int | None) -> _Definition:
    """Resolve a material reference — stable id's line, payload index, or name.

    Args:
        tree: The parsed program.
        reference: The request's ``material``: a name (the Python variable or
            the literal ``name=``) or the payload's non-negative index.
        line: The line a stable ``id`` resolved to, which wins when given.

    Returns:
        The one definition the reference names.

    Raises:
        PatchError: When the reference names no material, or more than one.
    """
    definitions = _material_definitions(tree)
    if line is not None:
        matches = [
            item
            for item in definitions
            if item.call.lineno <= line <= (item.call.end_lineno or item.call.lineno)
        ]
        if len(matches) != 1:
            raise PatchError(f"No single material definition found at line {line}.")
        return matches[0]
    if isinstance(reference, bool) or not isinstance(reference, (int, str)):
        raise PatchError("A material is referenced by its name or its non-negative index.")
    if isinstance(reference, int):
        literals = [item for item in definitions if item.factory is None]
        if not 0 <= reference < len(literals):
            raise PatchError(
                f"Material index {reference} is out of range; the program declares {len(literals)}."
            )
        return literals[reference]
    matches = [item for item in definitions if reference in (item.variable, item.name)]
    if len(matches) != 1:
        declared = ", ".join(repr(item.variable) for item in definitions)
        raise PatchError(
            f"No single material named {reference!r}; the program declares: {declared or 'none'}."
        )
    return matches[0]


def _line_bounds(source: str, offset: int) -> tuple[int, int]:
    """``(start, end)`` offsets of the line *offset* falls on, newline excluded."""
    start = source.rfind("\n", 0, offset) + 1
    end = source.find("\n", offset)
    return start, len(source) if end == -1 else end


def _indentation(source: str, start: int) -> str:
    """The leading whitespace of the line beginning at *start*."""
    end = start
    while end < len(source) and source[end] in " \t":
        end += 1
    return source[start:end]


def _added_keyword(source: str, call: ast.Call, keyword: str, literal: str) -> str:
    """Append ``keyword=literal`` after a call's last argument.

    The keyword goes on the same line as the argument it follows, so no line
    number moves — unless that would push the line past the column limit, in
    which case it wraps onto its own line at the call's indentation.  A
    wrapped keyword is the one case where this operation shifts lines, and it
    shifts them by exactly one.
    """
    offsets = _line_offsets(source)
    ends = [
        span[1]
        for argument in [*call.args, *(item.value for item in call.keywords)]
        if (span := _node_span(source, offsets, argument)) is not None
    ]
    if not ends:
        raise PatchError("Cannot add a property to a material call with no arguments.")
    insert = max(ends)
    inline = f", {keyword}={literal}"
    start, end = _line_bounds(source, insert)
    if (end - start) + len(inline) <= _COLUMN_LIMIT:
        return source[:insert] + inline + source[insert:]
    indent = _indentation(source, start)
    if start == _line_bounds(source, offsets[call.lineno - 1])[0]:
        # The call opens on this line, so its arguments are one level in.
        indent += "    "
    return source[:insert] + f",\n{indent}{keyword}={literal}" + source[insert:]


def _removed_keyword(source: str, keyword: ast.keyword) -> str:
    """Delete one keyword argument, and the comma that separated it.

    A keyword that has a line to itself takes the line with it rather than
    leaving a blank one behind; anything else is cut out in place, so the
    lines around it keep their numbers.
    """
    offsets = _line_offsets(source)
    span = _node_span(source, offsets, keyword)
    if span is None:  # pragma: no cover - a keyword always has a span
        raise PatchError(f"Could not locate `{keyword.arg}` in the source.")
    start, end = span
    line_start, line_end = _line_bounds(source, start)
    tail = end
    while tail < len(source) and source[tail] in " \t":
        tail += 1
    trailing_comma = tail < len(source) and source[tail] == ","
    if trailing_comma:
        tail += 1
    if not source[line_start:start].strip() and not source[tail:line_end].strip():
        # The keyword owns its line: take the newline with it.
        return source[:line_start] + source[min(line_end + 1, len(source)) :]
    if trailing_comma:
        while tail < len(source) and source[tail] in " \t":
            tail += 1
        return source[:start] + source[tail:]
    head = start
    while head > line_start and source[head - 1] in " \t":
        head -= 1
    if head == line_start and line_start > 0:
        # The keyword opened its own line as the call's last argument (the
        # way ``_added_keyword`` wraps one): the line break and the comma
        # before it were part of the same edit, and go back out with it.
        probe = line_start - 1
        while probe > 0 and source[probe - 1] in " \t":
            probe -= 1
        if probe > 0 and source[probe - 1] == ",":
            return source[: probe - 1] + source[end:]
    if head > line_start and source[head - 1] == ",":
        head -= 1
    return source[:head] + source[end:]


def _expanded_material(source: str, located: _Definition) -> str:
    """Rewrite ``x = copper_c11000()`` as the literal ``Material(...)`` it builds.

    Only the cheap case is converted: a bare, argument-free catalogue factory,
    whose result is a plain ``Material`` this process can build and read
    without executing any of the user's own code.  The literal is written one
    keyword per line — the shape the starter program uses — so the properties
    it now states are editable rows rather than one very long line.
    """
    from cadjoint.materials import CATALOGUE

    if located.call.args or located.call.keywords:
        raise PatchError(
            f"`{located.factory}()` is called with arguments, so it cannot be expanded "
            "automatically. Write the `Material(...)` you want by hand."
        )
    material = CATALOGUE[located.factory or ""]()
    described = material.describe()
    # ``describe`` rounds the physical map but not the optical one, and a
    # float32 0.38 reads back as 0.3799999952316284 — a storage artefact, not
    # a different number, so the optical values are rounded the same way.
    arguments = [("name", json.dumps(material.name or located.variable))]
    arguments.append(
        ("color", "[" + ", ".join(_exact_number(_display(c)) for c in described["color"]) + "]")
    )
    arguments.extend(
        (key, _exact_number(_display(described[key])))
        for key in OPTICAL_PROPERTIES
        if key != "color"
    )
    arguments.extend(
        (key, _exact_number(value))
        for key, value in described["physical"].items()
        if value is not None
    )
    body = "".join(f"    {key}={text},\n" for key, text in arguments)
    statement = f"{located.variable} = Material(\n{body})\n"
    offsets = _line_offsets(source)
    start = offsets[located.statement.lineno - 1]
    end = offsets[min(located.statement.end_lineno or located.statement.lineno, len(offsets) - 1)]
    patched = source[:start] + statement + source[end:]
    return _ensure_import(patched, ast.parse(patched), "cadjoint.render", "Material")


def set_material_property(
    source: str,
    property: str,
    value: float | None,
    material=None,
    line: int | None = None,
    expand: bool = False,
) -> str:
    """Set, add, or remove one property keyword on a ``Material(...)`` call.

    Optical properties are always stated, so setting one rewrites a literal in
    place.  Physical properties usually are not, so a number the material does
    not yet carry is *added* as a new keyword — that is the operation's reason
    to exist, since an unstated property has no span for the inspector to
    drag.  ``value=None`` removes the keyword, returning the property to
    unspecified (physical) or to its default (optical).

    Args:
        source: The program text.
        property: One of :data:`EDITABLE_PROPERTIES`.
        value: The SI value to write, or None to remove the keyword.
        material: The material's name or its payload index; ignored when
            *line* is given.
        line: The line a stable ``id`` resolved to, which addresses the
            material directly.
        expand: Convert a catalogue-built material to a literal
            ``Material(...)`` first, instead of refusing the edit.

    Returns:
        The patched source.

    Raises:
        PatchError: When the material cannot be found, is built by a catalogue
            factory and *expand* is not set, or carries the property as an
            expression this operation will not overwrite.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    if property not in PROPERTY_BOUNDS:
        allowed = ", ".join(EDITABLE_PROPERTIES)
        raise PatchError(f"`{property}` is not an editable material property; expected: {allowed}.")

    located = _located_material(tree, material, line)
    if located.factory is not None:
        if not expand:
            raise PatchError(
                _CATALOGUE_REFUSAL.format(variable=located.variable, factory=located.factory)
            )
        source = _expanded_material(source, located)
        tree = ast.parse(source)
        located = _located_material(tree, located.variable, None)

    existing = next((item for item in located.call.keywords if item.arg == property), None)
    if value is None:
        # Removing what is already absent is what the caller asked for.
        return source if existing is None else _validate(_removed_keyword(source, existing))
    literal = _exact_number(value)
    if existing is None:
        return _validate(_added_keyword(source, located.call, property, literal))
    editable = _editable_value_node(existing.value, tree)
    if editable is None:
        raise PatchError(
            f"`{located.variable}`'s `{property}` is not an editable literal; "
            "edit it directly in the code."
        )
    offsets = _line_offsets(source)
    span = _node_span(source, offsets, editable)
    if span is None:  # pragma: no cover - an editable node always has a span
        raise PatchError(f"Could not locate `{property}` in the source.")
    start, end = span
    return _validate(source[:start] + literal + source[end:])
