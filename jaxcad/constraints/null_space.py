"""Null-space projection and DOF analysis for constrained parameter systems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.constraints.residual import (
    _collect_constraints,
    build_residual_fn,
    pack_param_dict,
    unpack_param_vector,
)
from jaxcad.constraints.types.base import Constraint
from jaxcad.geometry.parameters import (
    NamedParams,
    Parameter,
)


def _compute_null_space(jacobian: Array, n_params: int, tolerance: float = 1e-10) -> Array:
    """Compute the null space of the constraint Jacobian via SVD."""
    if jacobian.size == 0 or jacobian.shape[0] == 0:
        return jnp.eye(n_params)

    _, s, Vt = jnp.linalg.svd(jacobian, full_matrices=True)
    rank = jnp.sum(s > tolerance)
    V = Vt.T
    return V[:, rank:]


class NullSpaceMap:
    """Null-space projection matrix that operates directly on parameter dicts.

    Wraps the null-space matrix N (shape n_params × n_free) together with the
    parameter structure so that dict ↔ vector conversions are automatic.

    Supports:
        ``N @ reduced_vec``  →  ``dict[str, Array]``  (expand reduced → full space)
        ``full_dict @ N``    →  ``Array``              (project full → reduced space)
        ``full_vec  @ N``    →  ``Array``              (same, from a flat vector)

    The raw matrix is accessible via ``N.matrix`` if needed.
    """

    def __init__(self, matrix: Array, params: NamedParams) -> None:
        self._matrix = matrix
        self._params = params

    @property
    def matrix(self) -> Array:
        return self._matrix

    @property
    def shape(self) -> tuple[int, ...]:
        return self._matrix.shape

    def __matmul__(self, reduced: Array) -> dict[str, Array]:
        """``N @ r``: expand a reduced-space vector to a full-space dict."""
        return unpack_param_vector(self._matrix @ reduced, self._params)

    def pack(self, full_dict: dict) -> Array:
        """Pack a full-space dict into a flat vector (consistent with params ordering)."""
        return pack_param_dict(full_dict, self._params)

    def unpack(self, flat: Array) -> dict[str, Array]:
        """Unpack a flat vector into a full-space dict (inverse of pack)."""
        return unpack_param_vector(flat, self._params)

    def __rmatmul__(self, other: dict | Array) -> Array:
        """``x @ N``: project a full-space dict or flat vector to reduced coords."""
        vec = self.pack(other) if isinstance(other, dict) else other
        return vec @ self._matrix

    def __repr__(self) -> str:
        n, k = self._matrix.shape
        return f"NullSpaceMap(n_params={n}, n_free={k})"


def all_parameters(constraints: list[Constraint]) -> list[Parameter]:
    """Return a deduplicated list of all parameters referenced by constraints.

    Args:
        constraints: List of Constraint objects.

    Returns:
        Deduplicated list of Parameter objects, in order of first appearance.
    """
    seen: set = set()
    result: list[Parameter] = []
    for c in constraints:
        for p in c.get_parameters():
            if id(p) not in seen:
                seen.add(id(p))
                result.append(p)
    return result


def total_dof_reduction(constraints: list[Constraint]) -> int:
    """Return the total number of DOF removed by a list of constraints.

    Args:
        constraints: List of Constraint objects.

    Returns:
        Sum of ``dof_reduction()`` across all constraints.
    """
    return sum(c.dof_reduction() for c in constraints)


def null_space(
    free_params: dict[str, Array],
    metadata: dict[str, Parameter],
) -> NullSpaceMap:
    """Compute the null-space projection for parameters under constraints.

    Collects constraints from the metadata, linearizes the constraint manifold
    at the current free_params values, and returns a :class:`NullSpaceMap`.

    Args:
        free_params: Name-keyed dict mapping parameter names to plain Arrays
            (as returned by :func:`~jaxcad.extraction.extract_parameters`).
        metadata: Name-keyed dict mapping parameter names to Parameter objects
            (carries constraints, bounds, type info).

    Returns:
        A :class:`NullSpaceMap` ``N`` of shape ``(n_params, n_free)`` such that:
        ``N @ r`` expands reduced coords to a full-space dict, and
        ``full_dict @ N`` projects a full-space dict to reduced coords.
    """
    constraints = _collect_constraints(metadata)
    x0 = pack_param_dict(free_params, metadata)
    n_params = x0.shape[0]

    if not constraints:
        matrix = jnp.eye(n_params)
    else:
        flat_fn = build_residual_fn(constraints, metadata)
        jacobian = jax.jacobian(flat_fn)(x0)
        matrix = _compute_null_space(jacobian, n_params)

    return NullSpaceMap(matrix, metadata)
