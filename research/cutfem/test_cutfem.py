"""Coarse, fast checks of the cut-cell prototype: run with pytest from the repository root."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from research.cutfem import cutfem as C
from research.cutfem import model as M

DISC_BOX = ([-0.8, -0.8], [0.8, 0.8])


def _grad_and_fd(model, theta, s, delta=1e-5):
    obj = jax.jit(lambda t: C.objective(t, s))
    g = np.asarray(jax.grad(obj)(theta))

    def free(t):
        return C.objective(t, C.classify(model, t, s.grid, s.n_sub, s.mode, s.eps / s.h))

    return g, C.central_difference(obj, theta, delta), C.central_difference(free, theta, delta)


def test_grammar_evaluates_the_disc_exactly():
    model, theta, _ = M.disc()
    pts = np.random.default_rng(0).uniform(-1, 1, (16, 2))
    f = np.asarray(M.field(model)(theta, pts))
    assert np.allclose(f, np.linalg.norm(pts - theta[1:], axis=1) - theta[0])
    grad = np.asarray(M.spatial_gradient(model)(theta, pts))
    unit = (pts - theta[1:]) / np.linalg.norm(pts - theta[1:], axis=1)[:, None]
    assert np.allclose(grad, unit)
    at_centre = jax.grad(lambda t: M.field(model)(t, theta[None, 1:])[0])(theta)
    assert np.all(np.isfinite(at_centre))


def _monomial_integral(powers):
    """∫ over the unit simplex of Π x_i^{p_i}: Π p_i! / (Σ p_i + d)!."""
    return np.prod([math.factorial(p) for p in powers]) / math.factorial(sum(powers) + len(powers))


def test_simplex_rules_are_exact_to_their_degree():
    for dim, degree in ((2, 2), (3, 5)):
        bary, w = C._volume_rule(dim)
        assert np.all(w > 0) and abs(w.sum() - 1) < 1e-14
        x = bary[:, 1:]  # cartesian coordinates on the unit simplex
        for powers in np.ndindex(*(degree + 1,) * dim):
            if sum(powers) > degree:
                continue
            value = np.sum(w * np.prod(x ** np.array(powers), axis=1)) / math.factorial(dim)
            assert abs(value - _monomial_integral(powers)) < 1e-14, (dim, powers)


def test_quadrature_measures_the_disc():
    model, theta, _ = M.disc()
    s = C.classify(model, theta, C.grid_for(*DISC_BOX, 1 / 16), n_sub=4, mode="sharp")
    (_, w, _), (_, bw, _, normal) = C.quadrature(jnp.asarray(theta), s)
    assert np.all(np.asarray(w) >= 0)
    assert abs(float(w.sum()) - np.pi * theta[0] ** 2) < 2e-4
    assert abs(float(bw.sum()) - 2 * np.pi * theta[0]) < 2e-4
    assert np.allclose(np.linalg.norm(np.asarray(normal), axis=1), 1.0)


def test_disc_sharp_matches_analytic_and_finite_differences():
    model, theta, _ = M.disc()
    radius = theta[0]
    s = C.classify(model, theta, C.grid_for(*DISC_BOX, 1 / 16), n_sub=4, mode="sharp")
    j = float(C.objective(theta, s))
    assert abs(j - np.pi * radius**4 / 8) / (np.pi * radius**4 / 8) < 1e-2
    g, fixed, free = _grad_and_fd(model, theta, s)
    assert abs(g[0] - np.pi * radius**3 / 2) / (np.pi * radius**3 / 2) < 1e-2
    assert np.abs(g[1:]).max() < 1e-3  # translation invariance, up to discretisation
    assert np.abs(g - free).max() < 1e-6 * np.abs(g).max()
    assert np.abs(g - fixed).max() < 1e-6 * np.abs(g).max()


def test_disc_smooth_mode_is_differentiable_and_biased():
    model, theta, _ = M.disc()
    radius = theta[0]
    s = C.classify(model, theta, C.grid_for(*DISC_BOX, 1 / 16), n_sub=8, mode="smooth", alpha=0.5)
    g, fixed, free = _grad_and_fd(model, theta, s)
    assert np.abs(g - free).max() < 1e-6 * np.abs(g).max()
    assert np.abs(g - fixed).max() < 1e-6 * np.abs(g).max()
    assert abs(g[0] - np.pi * radius**3 / 2) / (np.pi * radius**3 / 2) < 3e-2


def test_blend_ad_matches_finite_differences():
    model, theta, _ = M.blended_union()
    grid = C.grid_for([-0.95, -0.55], [0.85, 0.55], 1 / 16)
    for mode, alpha, n_sub in (("sharp", 0.0, 4), ("smooth", 0.5, 8)):
        s = C.classify(model, theta, grid, n_sub=n_sub, mode=mode, alpha=alpha)
        g, fixed, free = _grad_and_fd(model, theta, s)
        # J is C¹ but not C²: a sub-corner sign flip within δ of θ costs FD O(δ).
        assert np.abs(g - free).max() < 1e-5 * np.abs(g).max()
        assert np.abs(g - fixed).max() < 1e-5 * np.abs(g).max()


def test_ghost_penalty_makes_the_system_definite():
    model, theta, _ = M.disc()
    s = C.classify(model, theta, C.grid_for(*DISC_BOX, 1 / 16), n_sub=4, mode="sharp")
    K, _ = C.assemble(theta, s, ghost=0.1)
    cond, lo, _ = C.condition_number(K)
    assert lo > 0 and cond < 1e4
    assert np.allclose(np.asarray(K), np.asarray(K).T)


def test_hadamard_formula_has_the_right_sign_and_size():
    model, theta, _ = M.disc()
    s = C.classify(model, theta, C.grid_for(*DISC_BOX, 1 / 32), n_sub=4, mode="sharp")
    had = np.asarray(C.hadamard(theta, s))
    exact = np.pi * theta[0] ** 3 / 2
    assert abs(had[0] - exact) / exact < 0.1  # first order in h, from the FE gradient on Γ


def test_sphere_smoke():
    model, theta, _ = M.sphere()
    radius = theta[0]
    s = C.classify(model, theta, C.grid_for([-0.8] * 3, [0.8] * 3, 0.2), n_sub=2, mode="sharp")
    j = float(C.objective(theta, s))
    assert abs(j - 4 * np.pi * radius**5 / 45) / (4 * np.pi * radius**5 / 45) < 0.15
    g, fixed, free = _grad_and_fd(model, theta, s)
    assert abs(g[0] - 4 * np.pi * radius**4 / 9) / (4 * np.pi * radius**4 / 9) < 0.1
    assert np.abs(g - free).max() < 1e-6 * np.abs(g).max()


def test_the_sparse_solve_agrees_with_the_dense_one_in_value_and_gradient():
    """The custom VJP through SuperLU is the dense solve's derivative."""
    model, theta, _ = M.disc()
    s = C.classify(model, theta, C.grid_for([-0.8, -0.8], [0.8, 0.8], 1 / 16), n_sub=4)
    dense = jax.jit(lambda t: C.objective(t, s, dense=True))
    sparse = jax.jit(lambda t: C.objective(t, s))
    assert abs(float(sparse(theta)) - float(dense(theta))) < 1e-12
    np.testing.assert_allclose(jax.grad(sparse)(theta), jax.grad(dense)(theta), rtol=1e-9)


def test_a_cadjoint_scene_is_a_model_too():
    """A cadjoint node table lowers into the same solver; its θ are the scene's free parameters."""
    from cadjoint.geometry.parameters import Scalar
    from cadjoint.sdf.primitives import Sphere
    from cadjoint.zeroset import lower

    table = lower(Sphere(radius=Scalar(0.6, free=True, name="r")))
    s = C.classify(table, table.theta, C.grid_for([-0.8] * 3, [0.8] * 3, 1.6 / 8), n_sub=2)
    obj = jax.jit(lambda t: C.objective(t, s))
    exact = 4 * np.pi * 0.6**5 / 45  # ∫u for −Δu = 1 in a ball: u = (R² − r²)/6
    assert abs(float(obj(table.theta)) / exact - 1) < 0.12  # −9 % at three cells per radius
    (g,) = np.asarray(jax.grad(obj)(jnp.asarray(table.theta)))
    assert abs(g / (4 * np.pi * 0.6**4 / 9) - 1) < 0.12


def test_mixed_conditions_on_an_annulus_converge_to_the_closed_form():
    """Held at 0 on the inner circle r₀, heated at q on the outer circle R.

    T = (qR/k) ln(r/r₀); the mean over the annulus and its derivative in R
    are closed forms, and the region predicates sit halfway between the
    circles so no facet is ever misfiled.  Second order in h for the mean;
    the derivative through the moving outer boundary follows.
    """
    r0, q, k = 0.25, 1.0, 2.0
    t = M.Table()
    r = t.norm([t.coord("X"), t.coord("Y"), t.lit(0.0)])
    ring = t.owning("MAX", [t.patch(t.sub(r, t.param(0))), t.patch(t.sub(t.lit(r0), r))])
    model = t.model(ring, [0.7])
    theta = jnp.asarray([0.7])

    def closed_form(R):
        return (
            (2 * q * R / k) * (R**2 / 2 * jnp.log(R / r0) - R**2 / 4 + r0**2 / 4) / (R**2 - r0**2)
        )

    mid = 0.5 * (r0 + 0.7)
    problem = C.Thermal(
        conductivity=k,
        source=0.0,
        dirichlet=((lambda p: np.linalg.norm(p, axis=1) < mid, 0.0),),
        neumann=((lambda p: np.linalg.norm(p, axis=1) > mid, q),),
    )
    errors = []
    for h in (1 / 16, 1 / 32):
        grid = C.grid_for([-0.83, -0.81], [0.83, 0.83], h)
        s = C.classify(model, theta, grid, n_sub=2, problem=problem)
        mean = jax.jit(lambda th, s=s: C.mean_temperature(th, s))
        value, slope = float(mean(theta)), float(jax.grad(mean)(theta)[0])
        exact, exact_slope = float(closed_form(0.7)), float(jax.grad(closed_form)(0.7))
        errors.append((abs(value / exact - 1), abs(slope / exact_slope - 1)))
    assert errors[0][0] < 2e-2 and errors[1][0] < 6e-3, errors
    assert errors[1][0] < 0.4 * errors[0][0], errors  # second order, roughly
    assert errors[1][1] < 3e-2, errors
