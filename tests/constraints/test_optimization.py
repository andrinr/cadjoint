"""Tests for the optax-based constrained optimization API (make_manifold_projection)."""

import jax
import jax.numpy as jnp
import optax

from cadjoint.constraints import (
    DistanceConstraint,
    constraint_residuals,
    make_manifold_projection,
    null_space,
)
from cadjoint.geometry.parameters import Vector


def _free_and_meta(*params):
    return (
        {p.name: p.value for p in params},
        {p.name: p for p in params},
    )


def _sphere_setup(distance=2.0, suffix=""):
    """Single free point p with |p| = distance."""
    anchor = Vector([0, 0, 0], free=False, name=f"anc{suffix}")
    p = Vector([distance, 0, 0], free=True, name=f"p{suffix}")
    DistanceConstraint(anchor, p, distance)
    return _free_and_meta(p)


# ---------------------------------------------------------------------------
# make_manifold_projection unit tests
# ---------------------------------------------------------------------------


def test_make_manifold_projection_init_returns_empty_state():
    free, meta = _sphere_setup(suffix="_init")
    transform = make_manifold_projection(meta)
    assert transform.init(free) == ()


def test_make_manifold_projection_no_params_is_passthrough():
    """When params=None the transform must return updates unchanged."""
    free, meta = _sphere_setup(suffix="_nop")
    transform = make_manifold_projection(meta)
    state = transform.init(free)

    updates = {"p_nop": jnp.array([0.1, 0.2, 0.0])}
    out, _ = transform.update(updates, state, params=None)

    assert jnp.allclose(out["p_nop"], updates["p_nop"])


def test_make_manifold_projection_corrects_update_direction():
    """Effective update should land on the manifold, not just move toward it."""
    free, meta = _sphere_setup(distance=2.0, suffix="_corr")
    transform = make_manifold_projection(meta)
    state = transform.init(free)

    # A gradient step that moves off the sphere
    raw_updates = {"p_corr": jnp.array([-0.3, 0.5, 0.1])}
    corrected, _ = transform.update(raw_updates, state, params=free)

    new_params = optax.apply_updates(free, corrected)
    assert jnp.abs(jnp.linalg.norm(new_params["p_corr"]) - 2.0) < 1e-5


# ---------------------------------------------------------------------------
# Adam + make_manifold_projection
# ---------------------------------------------------------------------------


def test_manifold_projection_zero_violation_every_step():
    """Every iterate produced by adam + make_manifold_projection satisfies |p|=2."""
    free, meta = _sphere_setup(distance=2.0, suffix="_zv")
    target = jnp.array([1.0, 1.5, 0.0])

    optimizer = optax.chain(optax.adam(0.05), make_manifold_projection(meta))
    state = optimizer.init(free)
    params = free

    def objective(p):
        return jnp.sum((p["p_zv"] - target) ** 2)

    for _ in range(20):
        g = jax.grad(objective)(params)
        updates, state = optimizer.update(g, state, params)
        params = optax.apply_updates(params, updates)

        assert float(jnp.linalg.norm(constraint_residuals(params, meta))) < 1e-5


def test_manifold_projection_converges_to_constrained_optimum():
    """Adam + projection converges close to the constrained minimum."""
    free, meta = _sphere_setup(distance=2.0, suffix="_conv")
    target = jnp.array([1.0, 1.5, 0.0])
    p_star = target * (2.0 / jnp.linalg.norm(target))
    optimal_loss = float(jnp.sum((p_star - target) ** 2))

    optimizer = optax.chain(optax.adam(0.05), make_manifold_projection(meta))
    state = optimizer.init(free)
    params = free

    def objective(p):
        return jnp.sum((p["p_conv"] - target) ** 2)

    for _ in range(80):
        g = jax.grad(objective)(params)
        updates, state = optimizer.update(g, state, params)
        params = optax.apply_updates(params, updates)

    final_loss = float(jnp.sum((params["p_conv"] - target) ** 2))
    assert final_loss < optimal_loss * 1.1


# ---------------------------------------------------------------------------
# Riemannian GD: relinearized null-space SGD + make_manifold_projection
# ---------------------------------------------------------------------------


def test_riemannian_gd_zero_violation_every_step():
    """Tangent-space SGD + projection: |r(p)| = 0 at every iterate."""
    free, meta = _sphere_setup(distance=2.0, suffix="_rgd_zv")
    target = jnp.array([1.0, 1.5, 0.0])

    optimizer = optax.chain(optax.sgd(0.15), make_manifold_projection(meta))
    state = optimizer.init(free)
    params = free

    def objective(p):
        return jnp.sum((p["p_rgd_zv"] - target) ** 2)

    for _ in range(20):
        N = null_space(params, meta)
        g = jax.grad(objective)(params)
        g_tangent = N @ (g @ N)
        updates, state = optimizer.update(g_tangent, state, params)
        params = optax.apply_updates(params, updates)

        assert float(jnp.linalg.norm(constraint_residuals(params, meta))) < 1e-5


def test_riemannian_gd_converges_to_optimum():
    """Riemannian GD converges to within 2% of the constrained optimum."""
    free, meta = _sphere_setup(distance=2.0, suffix="_rgd_opt")
    target = jnp.array([1.0, 1.5, 0.0])
    p_star = target * (2.0 / jnp.linalg.norm(target))
    optimal_loss = float(jnp.sum((p_star - target) ** 2))

    optimizer = optax.chain(optax.sgd(0.15), make_manifold_projection(meta))
    state = optimizer.init(free)
    params = free

    def objective(p):
        return jnp.sum((p["p_rgd_opt"] - target) ** 2)

    for _ in range(40):
        N = null_space(params, meta)
        g = jax.grad(objective)(params)
        g_tangent = N @ (g @ N)
        updates, state = optimizer.update(g_tangent, state, params)
        params = optax.apply_updates(params, updates)

    final_loss = float(jnp.sum((params["p_rgd_opt"] - target) ** 2))
    assert final_loss < optimal_loss * 1.02
