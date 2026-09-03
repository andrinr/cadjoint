"""The assembled solver: configuration, read-outs, and the physics they report."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.flow import (
    CS2,
    DEFAULT_ALPHA_MAX,
    OMEGA_CEILING,
    FlowConfig,
    SteadyOptions,
    convergence,
    heat_transfer_proxy,
    pressure,
    pressure_drop,
    recommended_alpha_max,
    solve,
    step_for,
)

SHAPE = (6, 14, 6)


def _config(**changes):
    base = {
        "shape": SHAPE,
        "inlet_speed": 0.02,
        "reynolds": 10.0,
        "characteristic_cells": 6,
        "alpha_max": DEFAULT_ALPHA_MAX,
        "steady": SteadyOptions(max_steps=6000, tol=1e-12, check_every=50),
    }
    return FlowConfig(**{**base, **changes})


class TestConfiguration:
    def test_the_reynolds_number_fixes_the_viscosity(self):
        config = _config()
        assert config.viscosity == pytest.approx(0.02 * 6 / 10.0)
        assert config.length_scale == 6

    def test_the_length_scale_defaults_to_the_duct_depth(self):
        assert _config(characteristic_cells=None).length_scale == SHAPE[2]

    def test_omega_and_viscosity_agree(self):
        from cadjoint.flow import viscosity_from_omega

        config = _config()
        assert viscosity_from_omega(config.omega) == pytest.approx(config.viscosity)

    def test_a_configuration_too_close_to_omega_two_is_refused_up_front(self):
        """A NaN an hour into a march is a much worse error message.

        Measured: 24x48x24 converges at omega 1.9440 and 20x40x20 diverges
        at 1.9531, so the ceiling sits between them.
        """
        with pytest.raises(ValueError, match="above the stable ceiling"):
            _config(reynolds=1000.0)

    def test_the_ceiling_leaves_the_working_point_alone(self):
        assert _config().omega < OMEGA_CEILING

    def test_the_recommended_drag_is_the_default(self):
        assert recommended_alpha_max(_config()) == DEFAULT_ALPHA_MAX

    def test_the_step_closure_is_cached_by_identity(self):
        """``steady_populations`` takes it as a ``nondiff_argnum``.

        JAX hashes those by identity, so a fresh closure per call would
        recompile the whole scan every time.
        """
        assert step_for(_config()) is step_for(_config())
        assert step_for(_config()) is not step_for(_config(reynolds=5.0))


class TestObjectives:
    def test_pressure_is_density_times_the_lattice_sound_speed(self):
        rho = jnp.full(SHAPE, 1.2)
        assert np.allclose(pressure(rho), 1.2 * CS2)

    def test_the_drop_is_measured_inside_the_imposed_planes(self):
        """``margin`` keeps the imposed inlet/outlet cells out of the number."""
        rho = jnp.ones(SHAPE).at[:, 1, :].set(1.1).at[:, -2, :].set(0.9)
        assert float(pressure_drop(rho)) == pytest.approx(0.2 * CS2)

    def test_the_heat_proxy_needs_both_metal_and_motion(self):
        u = jnp.ones((3, *SHAPE))
        assert float(heat_transfer_proxy(jnp.zeros(SHAPE), u)) == pytest.approx(0.0)
        assert float(heat_transfer_proxy(jnp.ones(SHAPE), jnp.zeros((3, *SHAPE)))) == pytest.approx(
            0.0, abs=1e-12
        )
        assert float(heat_transfer_proxy(jnp.ones(SHAPE), u)) > 0.0

    def test_the_proxy_scales_with_cell_volume(self):
        chi, u = jnp.ones(SHAPE), jnp.ones((3, *SHAPE))
        one = heat_transfer_proxy(chi, u, 1.0)
        assert float(heat_transfer_proxy(chi, u, 2.5)) == pytest.approx(2.5 * float(one))


class TestSolve:
    def test_a_mismatched_field_is_refused(self):
        with pytest.raises(ValueError, match="expected"):
            solve(jnp.zeros((4, 4, 4)), _config())

    def test_an_empty_duct_carries_the_flow_it_was_given(self):
        result = solve(jnp.zeros(SHAPE), _config())
        # Bulk velocity matches the inlet: mass in equals mass out.
        assert float(jnp.mean(result.velocity[1][1:-1, SHAPE[1] // 2, 1:-1])) == pytest.approx(
            0.02, rel=0.15
        )
        # The duct still resists: a positive drop, but a small one.
        assert 0.0 < float(result.pressure_drop) < 1e-2
        assert float(result.heat_transfer) == pytest.approx(0.0, abs=1e-12)

    def test_the_profile_is_the_one_a_no_slip_wall_would_produce(self):
        """Halfway bounce-back puts the wall *between* nodes, so no-slip is
        a statement about the profile, not about a node value.

        The wall face sits half a cell outside the first fluid node, so a
        parabolic profile vanishing there predicts
        ``u[1]/u_max = (0.5)(3.5)/((1.5)(2.5)) = 0.47`` on this duct.  The
        node beyond it is a ghost whose value is meaningless by
        construction -- but nothing should have leaked into it either.
        """
        profile = np.asarray(solve(jnp.zeros(SHAPE), _config()).velocity[1][:, 7, 3])
        centreline = profile.max()
        assert 0.35 < profile[1] / centreline < 0.65
        assert abs(profile[0]) < 0.02 * centreline

    def test_the_profile_is_symmetric_across_the_duct(self):
        """A duct with symmetric walls and a uniform inlet has no reason to lean."""
        profile = solve(jnp.zeros(SHAPE), _config()).velocity[1][:, SHAPE[1] // 2, 3]
        assert np.allclose(np.asarray(profile), np.asarray(profile)[::-1], atol=1e-9)

    def test_blocking_the_duct_costs_pressure(self):
        empty = solve(jnp.zeros(SHAPE), _config()).pressure_drop
        blocked = jnp.zeros(SHAPE).at[2:4, 6:9, 2:4].set(1.0)
        assert float(solve(blocked, _config()).pressure_drop) > 3.0 * float(empty)

    def test_the_flow_stays_in_the_incompressible_regime(self):
        """Above a few percent of density variation the answer is not the
        incompressible one it is being read as."""
        result = solve(jnp.zeros(SHAPE), _config())
        assert float(result.density.max() - result.density.min()) < 0.05

    def test_convergence_reports_a_falling_residual(self):
        _, history = convergence(jnp.zeros(SHAPE), _config(steady=SteadyOptions(max_steps=1000)))
        history = np.asarray(history)
        assert np.all(np.isfinite(history))
        assert history[-1] < history[0]

    def test_both_objectives_are_differentiable_in_the_design(self):
        chi = jnp.full(SHAPE, 0.0).at[2:4, 6:9, 2:4].set(0.5)
        for name in ("pressure_drop", "heat_transfer"):
            grad = jax.grad(lambda c, name=name: getattr(solve(c, _config()), name))(chi)
            assert np.all(np.isfinite(grad))
            assert float(jnp.linalg.norm(grad)) > 0.0

    def test_the_inlet_velocity_is_differentiable_too(self):
        """So an operating point can be optimized alongside the geometry."""
        grad = jax.grad(
            lambda u: solve(jnp.zeros(SHAPE), _config(), inlet_velocity=u).pressure_drop
        )(jnp.array([0.0, 0.02, 0.0]))
        assert float(grad[1]) > 0.0

    def test_solving_inside_jit_does_not_leak_the_cached_masks(self):
        """Regression: ``step_for`` is cached, so anything it captures at
        trace time escapes into later calls.  The wall masks are NumPy for
        exactly this reason."""
        config = _config(reynolds=7.0)
        compiled = jax.jit(lambda c: solve(c, config).pressure_drop)
        inside = float(compiled(jnp.zeros(SHAPE)))
        outside = float(solve(jnp.zeros(SHAPE), config).pressure_drop)
        assert inside == pytest.approx(outside, rel=1e-9)
