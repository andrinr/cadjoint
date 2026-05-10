"""Parameter vector utilities and residual function builder for constraint solving."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints.types.base import Constraint
from jaxcad.geometry.parameters import (
    NamedParams,
    Parameter,
    Scalar,
)


def _collect_constraints(params: NamedParams) -> list[Constraint]:
    """Discover all constraints attached to the given parameters (deduplicated)."""
    seen: set = set()
    constraints = []
    for param in params.values():
        for constraint in param.get_constraints():
            if id(constraint) not in seen:
                seen.add(id(constraint))
                constraints.append(constraint)
    return constraints


def _fixed_params(constraints: list[Constraint]) -> dict[str, Array]:
    """Collect values of non-free parameters referenced by constraints."""
    seen: set = set()
    result: dict[str, Array] = {}
    for constraint in constraints:
        for param in constraint.get_parameters():
            if param.name not in seen:
                seen.add(param.name)
                if not param.free:
                    result[param.name] = param.value
    return result


def _as_vec(v: Parameter | Array) -> Array:
    return jnp.atleast_1d(v.value if isinstance(v, Parameter) else v)


def pack_param_dict(d: dict, params: NamedParams) -> Array:
    """Pack entries of *d* into a flat vector following *params* ordering.

    Entries missing from *d* are skipped.  Values may be plain :class:`~jax.Array`
    objects or :class:`~jaxcad.geometry.parameters.Parameter` instances.
    """
    return jnp.concatenate([_as_vec(d[name]) for name in params if name in d])


def compute_param_vector(params: NamedParams) -> Array:
    """Build the initial flat parameter vector from an ordered params dict."""
    return pack_param_dict(params, params)


def unpack_param_vector(x_flat: Array, params: NamedParams) -> dict[str, Array]:
    """Inverse of compute_param_vector: split a flat array back into named values.

    Args:
        x_flat: Flat parameter vector as returned by :func:`compute_param_vector`.
        params: The same ordered NamedParams used to build ``x_flat``.

    Returns:
        Dict mapping parameter name to its value (scalars are 0-d arrays).
    """
    result: dict[str, Array] = {}
    offset = 0
    for name, p in params.items():
        size = p.value.size
        val = x_flat[offset : offset + size]
        result[name] = val[0] if isinstance(p, Scalar) else val
        offset += size
    return result


def build_residual_fn(
    constraints: list[Constraint],
    params: NamedParams,
) -> Callable[[Array], Array]:
    """Build a flat residual function over all constraints.

    Each constraint's residuals are scaled by its ``weight`` attribute before
    concatenation. Set ``constraint.weight`` at construction time to normalize
    across types or scales.
    """
    fixed_by_name = _fixed_params(constraints)
    weights: tuple[float, ...] = tuple(c.weight for c in constraints)

    def flat_fn(x_flat: Array) -> Array:
        param_values = {**fixed_by_name, **unpack_param_vector(x_flat, params)}
        return jnp.concatenate(
            [
                w * jnp.atleast_1d(c.compute_residual(param_values))
                for w, c in zip(weights, constraints)
            ]
        )

    return flat_fn
