"""Shared helpers for the 2D sketch constraint types.

The sketch constraints (horizontal, vertical, coincident, equal-length,
point-on-line, parallel-edges, perpendicular-edges) all accept raw coordinate
pairs as well as ``Vector2`` parameters, and all read their parameters from a
name-keyed value dict. The coercion and lookup plumbing lives here so each
constraint module only spells out its residual equation.
"""

from __future__ import annotations

from jax import Array

from jaxcad.geometry.parameters import Parameter


def coerce_parameter_fields(constraint, field_names: tuple[str, ...]) -> None:
    """Replace raw values in the named dataclass fields with Parameters.

    Must run before ``Constraint.__post_init__`` registers the constraint on
    its parameters, so raw ``[x, y]`` inputs participate in constraint solving
    exactly like explicitly constructed ``Vector2`` parameters.

    Args:
        constraint: The constraint instance being initialized.
        field_names: Names of the fields to coerce, in declaration order.
    """
    from jaxcad.geometry.parameters import as_parameter

    for field_name in field_names:
        value = getattr(constraint, field_name)
        if not isinstance(value, Parameter):
            setattr(constraint, field_name, as_parameter(value))


def require_values(param_values: dict[str, Array], names: list[str]) -> list[Array]:
    """Look up each name's value, or raise naming exactly what is missing.

    Args:
        param_values: Name-keyed parameter values passed to a residual.
        names: Parameter names the residual needs, in argument order.

    Returns:
        The values in the same order as ``names``.

    Raises:
        ValueError: If any name is absent from ``param_values``.
    """
    missing = [name for name in names if name not in param_values]
    if missing:
        raise ValueError(f"Parameter values must include {missing}")
    return [param_values[name] for name in names]


def cross_2d(u: Array, w: Array) -> Array:
    """Scalar 2D cross product ``u × w`` — zero iff the vectors are parallel."""
    return u[0] * w[1] - u[1] * w[0]
