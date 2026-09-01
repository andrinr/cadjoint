"""Tests for constraints/solve.py — solve_constraints, project_to_manifold, constraint_residuals."""

import jax.numpy as jnp
import pytest

from cadjoint.constraints import (
    DistanceConstraint,
    constraint_residuals,
    project_to_manifold,
)
from cadjoint.constraints.solve import solve_constraints
from cadjoint.geometry.parameters import Vector
from cadjoint.sdf.primitives.sphere import Sphere
from cadjoint.sdf.transforms.affine.translate import Translate


def _free_and_meta(*params):
    return (
        {p.name: p.value for p in params},
        {p.name: p for p in params},
    )


def _sphere_constraint(distance=2.0, suffix=""):
    """Sphere: distance from origin equals `distance`."""
    anchor = Vector([0, 0, 0], free=False, name=f"anc_s{suffix}")
    p = Vector([distance, 0, 0], free=True, name=f"p_s{suffix}")
    DistanceConstraint(anchor, p, distance)
    return _free_and_meta(p)


# ---------------------------------------------------------------------------
# solve_constraints
# ---------------------------------------------------------------------------


def _trilateration_scene():
    """Trilateration: three anchors, one unknown point p."""
    anchor_a = Vector(jnp.array([0.0, 0.0, 0.0]), free=False, name="anchor_a")
    anchor_b = Vector(jnp.array([4.0, 0.0, 0.0]), free=False, name="anchor_b")
    anchor_c = Vector(jnp.array([2.0, 3.0, 0.0]), free=False, name="anchor_c")
    true_p = jnp.array([2.0, 1.0, 0.0])

    p = Vector(jnp.array([0.5, 0.5, 0.0]), free=True, name="p")
    scene = Translate(Sphere(radius=0.5), offset=p)

    DistanceConstraint(p, anchor_a, float(jnp.linalg.norm(true_p - anchor_a.value)))
    DistanceConstraint(p, anchor_b, float(jnp.linalg.norm(true_p - anchor_b.value)))
    DistanceConstraint(p, anchor_c, float(jnp.linalg.norm(true_p - anchor_c.value)))

    return scene, p, true_p


def test_solve_constraints_trilateration():
    """Solver recovers true_p from three distance constraints."""
    scene, p, true_p = _trilateration_scene()

    solved = solve_constraints(scene)

    assert jnp.allclose(solved["p"], true_p, atol=1e-5)


def test_solve_constraints_returns_free_params_dict():
    """Return value has same keys as extract_parameters free dict."""
    from cadjoint.extraction import extract_parameters

    scene, _, _ = _trilateration_scene()
    free_params, _, _ = extract_parameters(scene)
    solved = solve_constraints(scene)

    assert set(solved.keys()) == set(free_params.keys())


def test_solve_constraints_under_constrained():
    """Raises ValueError when DOF > constraints."""
    anchor = Vector(jnp.array([0.0, 0.0, 0.0]), free=False, name="anchor_uc")
    p = Vector(jnp.array([1.0, 0.0, 0.0]), free=True, name="p_uc")
    scene = Translate(Sphere(radius=0.5), offset=p)

    DistanceConstraint(p, anchor, 1.0)

    with pytest.raises(ValueError, match="Under-constrained"):
        solve_constraints(scene)


def test_solve_constraints_over_constrained():
    """Raises ValueError when constraints > DOF."""
    p = Vector(jnp.array([1.0, 0.0, 0.0]), free=True, name="p_oc")
    scene = Translate(Sphere(radius=0.5), offset=p)

    for i in range(4):
        a = Vector(jnp.array([float(i), 0.0, 0.0]), free=False, name=f"anc_oc_{i}")
        DistanceConstraint(p, a, 1.0)

    with pytest.raises(ValueError, match="Over-constrained"):
        solve_constraints(scene)


# ---------------------------------------------------------------------------
# project_to_manifold
# ---------------------------------------------------------------------------


def test_project_to_manifold_snaps_to_sphere():
    """Off-sphere point is projected onto the sphere."""
    free, meta = _sphere_constraint(distance=2.0, suffix="_snap")
    off = {"p_s_snap": jnp.array([3.0, 1.0, 0.0])}

    projected = project_to_manifold(off, meta)

    assert jnp.abs(jnp.linalg.norm(projected["p_s_snap"]) - 2.0) < 1e-5


def test_project_to_manifold_no_op_when_on_manifold():
    """Point already on the sphere should be unchanged."""
    free, meta = _sphere_constraint(distance=2.0, suffix="_noop")
    on = {"p_s_noop": jnp.array([0.0, 2.0, 0.0])}

    projected = project_to_manifold(on, meta)

    assert jnp.allclose(projected["p_s_noop"], on["p_s_noop"], atol=1e-5)


def test_project_to_manifold_no_constraints_is_identity():
    """Without any constraints, project_to_manifold is a no-op."""
    p = Vector([1.0, 2.0, 3.0], free=True, name="p_nc_proj")
    free, meta = _free_and_meta(p)

    result = project_to_manifold(free, meta)

    assert jnp.allclose(result["p_nc_proj"], free["p_nc_proj"])


def test_project_to_manifold_multiple_steps_all_satisfy():
    """All step counts should yield constraint satisfaction for the sphere."""
    free, meta = _sphere_constraint(distance=2.0, suffix="_steps")
    off = {"p_s_steps": jnp.array([4.0, 2.0, 1.0])}

    for steps in [1, 2, 5]:
        projected = project_to_manifold(off, meta, steps=steps)
        assert jnp.abs(jnp.linalg.norm(projected["p_s_steps"]) - 2.0) < 1e-5


# ---------------------------------------------------------------------------
# constraint_residuals
# ---------------------------------------------------------------------------


def test_constraint_residuals_zero_on_manifold():
    """Residuals should be zero when the constraint is exactly satisfied."""
    free, meta = _sphere_constraint(distance=2.0, suffix="_res0")
    on = {"p_s_res0": jnp.array([0.0, 0.0, 2.0])}

    r = constraint_residuals(on, meta)

    assert float(jnp.abs(r).max()) < 1e-5


def test_constraint_residuals_nonzero_off_manifold():
    """Residuals should be nonzero when the constraint is violated."""
    free, meta = _sphere_constraint(distance=2.0, suffix="_res1")
    off = {"p_s_res1": jnp.array([3.0, 0.0, 0.0])}  # |p| = 3, should be 2

    r = constraint_residuals(off, meta)

    assert float(jnp.abs(r).max()) > 0.5


def test_constraint_residuals_measures_violation_magnitude():
    """Residual magnitude should equal |p| - distance."""
    free, meta = _sphere_constraint(distance=2.0, suffix="_mag")
    off = {"p_s_mag": jnp.array([5.0, 0.0, 0.0])}  # |p| = 5, distance = 2 → residual = 3

    r = constraint_residuals(off, meta)

    assert jnp.allclose(jnp.abs(r), jnp.array([3.0]), atol=1e-5)


def test_constraint_residuals_empty_without_constraints():
    """Returns an empty array when there are no constraints."""
    p = Vector([1.0, 0.0, 0.0], free=True, name="p_empty_r")
    free, meta = _free_and_meta(p)

    r = constraint_residuals(free, meta)

    assert r.shape == (0,)
