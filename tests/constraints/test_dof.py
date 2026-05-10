"""Tests for null-space free functions and NullSpaceMap."""

import jax
import jax.numpy as jnp
import pytest

from jaxcad.constraints import (
    DistanceConstraint,
    build_residual_fn,
    null_space,
)
from jaxcad.constraints.residual import _collect_constraints, compute_param_vector
from jaxcad.geometry.parameters import Vector


def _free_and_meta(*params):
    return (
        {p.name: p.value for p in params},
        {p.name: p for p in params},
    )


# ---------------------------------------------------------------------------
# Null-space shape / DOF counting
# ---------------------------------------------------------------------------


def test_null_space_no_constraints():
    """With no constraints the null space should be the full identity."""
    p1 = Vector([0, 0, 0], free=True, name="p1_nc")
    p2 = Vector([1, 0, 0], free=True, name="p2_nc")

    N = null_space(*_free_and_meta(p1, p2))

    assert N.shape == (6, 6)


def test_null_space_with_distance_constraint():
    """One distance constraint should remove 1 DOF."""
    p1 = Vector([0, 0, 0], free=True, name="p1_dc")
    p2 = Vector([1, 0, 0], free=True, name="p2_dc")
    DistanceConstraint(p1, p2, 1.0)

    N = null_space(*_free_and_meta(p1, p2))

    assert N.shape == (6, 5)


@pytest.mark.parametrize(
    "n_points,n_constraints,expected_dof",
    [
        (2, 1, 5),
        (3, 2, 7),
        (4, 3, 9),
    ],
)
def test_dof_counting(n_points, n_constraints, expected_dof):
    points = [Vector([i, 0, 0], free=True, name=f"pdof{i}") for i in range(n_points)]
    for i in range(n_constraints):
        DistanceConstraint(points[i], points[i + 1], 1.0)

    free = {p.name: p.value for p in points}
    meta = {p.name: p for p in points}
    N = null_space(free, meta)

    assert N.shape == (n_points * 3, expected_dof)


# ---------------------------------------------------------------------------
# Null-space mathematical properties
# ---------------------------------------------------------------------------


def test_null_space_matrix_orthonormal():
    """Null space matrix should have orthonormal columns (Nᵀ N = I)."""
    p1 = Vector([0, 0, 0], free=True, name="p1_orth")
    p2 = Vector([1, 0, 0], free=True, name="p2_orth")
    DistanceConstraint(p1, p2, 1.0)

    free, meta = _free_and_meta(p1, p2)
    N = null_space(free, meta)

    assert jnp.allclose(N.matrix.T @ N.matrix, jnp.eye(N.shape[1]), atol=1e-5)


def test_null_space_is_kernel_of_jacobian():
    """J @ N ≈ 0: null space vectors should annihilate the constraint Jacobian."""
    p1 = Vector([0, 0, 0], free=True, name="p1_ker")
    p2 = Vector([1, 0, 0], free=True, name="p2_ker")
    DistanceConstraint(p1, p2, 1.0)

    free, meta = _free_and_meta(p1, p2)
    N = null_space(free, meta)

    constraints = _collect_constraints(meta)
    flat_fn = build_residual_fn(constraints, meta)
    x0 = compute_param_vector(meta)
    J = jax.jacobian(flat_fn)(x0)

    assert jnp.allclose(J @ N.matrix, jnp.zeros((J.shape[0], N.shape[1])), atol=1e-5)


def test_null_space_satisfies_constraints():
    """Small steps in the null space should approximately preserve constraints."""
    p1 = Vector([0, 0, 0], free=True, name="p1_sat")
    p2 = Vector([1, 0, 0], free=True, name="p2_sat")
    constraint = DistanceConstraint(p1, p2, 1.0)

    free, meta = _free_and_meta(p1, p2)
    N = null_space(free, meta)
    base_vec = N.pack(free)

    new_full = base_vec + N.matrix @ jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    residual = constraint.compute_residual({"p1_sat": new_full[:3], "p2_sat": new_full[3:]})

    assert jnp.abs(residual) < 0.1


# ---------------------------------------------------------------------------
# NullSpaceMap operations
# ---------------------------------------------------------------------------


def test_nullspacemap_pack_unpack_roundtrip():
    """pack and unpack should be inverses."""
    p = Vector([1, 2, 3], free=True, name="p_rt")
    free, meta = _free_and_meta(p)
    N = null_space(free, meta)  # no constraints → identity

    packed = N.pack(free)
    unpacked = N.unpack(packed)

    assert jnp.allclose(packed, jnp.array([1.0, 2.0, 3.0]), atol=1e-6)
    assert jnp.allclose(unpacked["p_rt"], free["p_rt"], atol=1e-6)


def test_nullspacemap_rmatmul_gives_reduced_coords():
    """full_dict @ N projects to reduced coordinates."""
    p1 = Vector([0, 0, 0], free=True, name="p1_rm")
    p2 = Vector([1, 0, 0], free=True, name="p2_rm")
    DistanceConstraint(p1, p2, 1.0)

    free, meta = _free_and_meta(p1, p2)
    N = null_space(free, meta)

    reduced = free @ N
    assert reduced.shape == (5,)


def test_nullspacemap_matmul_gives_full_dict():
    """N @ reduced_vec returns a full-space dict."""
    p1 = Vector([0, 0, 0], free=True, name="p1_mm")
    p2 = Vector([1, 0, 0], free=True, name="p2_mm")
    DistanceConstraint(p1, p2, 1.0)

    free, meta = _free_and_meta(p1, p2)
    N = null_space(free, meta)

    reduced = jnp.zeros(N.shape[1])
    expanded = N @ reduced

    assert set(expanded.keys()) == set(free.keys())
    assert expanded["p1_mm"].shape == (3,)


def test_nullspacemap_matmul_rmatmul_projection():
    """N @ (full @ N) equals the projection of full onto the null space."""
    p1 = Vector([0, 0, 0], free=True, name="p1_proj")
    p2 = Vector([1, 0, 0], free=True, name="p2_proj")
    DistanceConstraint(p1, p2, 1.0)

    free, meta = _free_and_meta(p1, p2)
    N = null_space(free, meta)

    # P_N = N @ N.T is the null-space projector
    P_N = N.matrix @ N.matrix.T
    projected_manual = N.unpack(P_N @ N.pack(free))
    projected_via_ops = N @ (free @ N)

    for key in free:
        assert jnp.allclose(projected_via_ops[key], projected_manual[key], atol=1e-5)


# ---------------------------------------------------------------------------
# Weight
# ---------------------------------------------------------------------------


def test_weight_default_is_one():
    p1 = Vector([0, 0, 0], free=True, name="p1_w")
    p2 = Vector([1, 0, 0], free=True, name="p2_w")
    c = DistanceConstraint(p1, p2, 1.0)
    assert c.weight == 1.0


def test_weight_on_constraint_scales_residuals():
    p1 = Vector([0, 0, 0], free=True, name="p1_ws")
    p2 = Vector([2, 0, 0], free=True, name="p2_ws")

    c_unit = DistanceConstraint(p1, p2, 1.0)
    c_half = DistanceConstraint(p1, p2, 1.0, weight=0.5)

    meta = {p1.name: p1, p2.name: p2}
    x0 = compute_param_vector(meta)

    fn_unit = build_residual_fn([c_unit], meta)
    fn_half = build_residual_fn([c_half], meta)

    assert jnp.allclose(fn_half(x0), 0.5 * fn_unit(x0), atol=1e-6)
