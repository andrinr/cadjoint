"""Render Python values back into the literal source text that will replace them.

Every patch operation ends by splicing a string into the program, and this is
where that string is produced.  Two conventions live side by side and must not
be confused:

- **compact** (:func:`_format_coordinate` and friends) for values the *viewer*
  generated — drag coordinates, ray-plane intersections — where ``%.4g`` and
  snapping near-zero to ``0`` keeps generated code readable;
- **exact** (:func:`_exact_number` and friends) for values a *user typed*,
  written with ``repr(float(...))`` so the number round-trips unchanged.

Selection descriptions are rendered here too: ``_selection_source`` validates a
node-selection dict through the runtime and emits the equivalent ``Nodes.…``
expression, so only selections the runtime can rebuild are ever written.

This module knows nothing about the AST — it turns values into text.  Anything
that touches spans belongs in :mod:`cadjoint.viewer.patch.edits`.
"""

from __future__ import annotations

from cadjoint.viewer.patch.errors import PatchError


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


def _format_value(value) -> str:
    """Format a scalar or vector literal."""
    if isinstance(value, (int, float)):
        return _format_coordinate(value)
    return "[" + ", ".join(_format_coordinate(component) for component in value) + "]"


def _format_keywords(arguments: dict) -> str:
    """``key=value, key=value`` for a whole call, compactly formatted.

    Every operation that *writes a new call* — a material definition, a new
    solid — was spelling this out one keyword at a time; the argument list
    is one thing, so it is formatted in one place.
    """
    return ", ".join(f"{key}={_format_value(value)}" for key, value in arguments.items())


def _exact_number(value) -> str:
    """Format a number so it round-trips exactly (viewer-typed values)."""
    return repr(float(value))


def _exact_value(value) -> str:
    """Format a scalar or vector with exact float round-tripping."""
    if isinstance(value, (int, float)):
        return _exact_number(value)
    return "[" + ", ".join(_exact_number(component) for component in value) + "]"


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
