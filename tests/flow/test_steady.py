"""The fixed point and its adjoint: what converges, and what the gradient is.

The claim these tests exist to defend is that
:func:`~cadjoint.flow.steady_populations` returns the *same* gradient a
taped march would, without keeping the tape.  It is checked directly --
against an unrolled trajectory long enough to have converged, and against
finite differences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.flow import (
    FlowConfig,
    SteadyOptions,
    initial_populations,
    iterate_to_steady,
    residual_history,
    solid_fraction,
    steady_populations,
    step_for,
    unrolled_populations,
)

SHAPE = (6, 12, 6)
INLET = 0.02


def _options(**changes):
    base = {
        "max_steps": 8000,
        "tol": 1e-14,
        "check_every": 50,
        "adjoint_tol": 1e-12,
        "adjoint_max_steps": 2000,
        "adjoint_restart": 30,
    }
    return SteadyOptions(**{**base, **changes})


@pytest.fixture(scope="module")
def problem():
    """A small duct with a smooth centred obstacle, cheap enough to tape."""
    config = FlowConfig(
        shape=SHAPE,
        inlet_speed=INLET,
        reynolds=10.0,
        characteristic_cells=6,
        alpha_max=1.0,
        steady=_options(),
    )
    axes = [jnp.arange(n) + 0.5 for n in SHAPE]
    x, y, z = jnp.meshgrid(*axes, indexing="ij")

    def chi_of(half):
        distance = jnp.maximum(
            jnp.maximum(jnp.abs(x - 3.0) - half, jnp.abs(y - 6.0) - 2.0), jnp.abs(z - 3.0) - half
        )
        return solid_fraction(distance, 1.0)

    velocity = jnp.array([0.0, INLET, 0.0])
    return config, chi_of, velocity, initial_populations(SHAPE, velocity)


def _loss(config, chi_of, velocity, f0, half, options=None):
    chi = chi_of(half)
    theta = {"chi": chi, "inlet_velocity": velocity}
    f = steady_populations(step_for(config), options or config.steady, theta, f0)
    return jnp.sum(f * f)


class TestConvergence:
    def test_the_march_reaches_a_fixed_point(self, problem):
        config, chi_of, velocity, f0 = problem
        step_fn = step_for(config)
        theta = {"chi": chi_of(1.5), "inlet_velocity": velocity}
        f_star = iterate_to_steady(step_fn, theta, f0, config.steady)
        residual = jnp.linalg.norm(step_fn(f_star, theta) - f_star) / jnp.linalg.norm(f_star)
        assert float(residual) < 1e-11

    def test_the_residual_history_decreases(self, problem):
        config, chi_of, velocity, f0 = problem
        _, history = residual_history(
            step_for(config),
            {"chi": chi_of(1.5), "inlet_velocity": velocity},
            f0,
            _options(max_steps=2000),
        )
        history = np.asarray(history)
        assert np.all(np.isfinite(history))
        assert history[-1] < history[0] * 1e-3

    def test_the_early_exit_march_agrees_with_a_fixed_step_count(self, problem):
        """``while_loop`` and ``scan`` must land on the same fixed point."""
        config, chi_of, velocity, f0 = problem
        theta = {"chi": chi_of(1.5), "inlet_velocity": velocity}
        early = iterate_to_steady(step_for(config), theta, f0, config.steady)
        fixed = unrolled_populations(step_for(config), theta, f0, 8000)
        assert float(jnp.linalg.norm(early - fixed) / jnp.linalg.norm(fixed)) < 1e-10


class TestImplicitAdjoint:
    def test_it_matches_a_converged_unrolled_tape(self, problem):
        """The central claim, checked against the thing it replaces.

        The tape is the ground truth only once the march it tapes has
        actually converged -- so the agreement is expected to *improve* with
        trajectory length, and it does.
        """
        config, chi_of, velocity, f0 = problem
        implicit = float(jax.grad(lambda h: _loss(config, chi_of, velocity, f0, h))(1.5))

        def taped(half, steps):
            theta = {"chi": chi_of(half), "inlet_velocity": velocity}
            f = unrolled_populations(step_for(config), theta, f0, steps)
            return jnp.sum(f * f)

        errors = []
        for steps in (500, 1000, 2000):
            taped_grad = float(jax.grad(taped)(1.5, steps))
            errors.append(abs(taped_grad - implicit) / abs(implicit))
        assert errors[-1] < 1e-6, errors
        assert errors[-1] < errors[0], errors

    def test_it_matches_a_central_difference(self, problem):
        config, chi_of, velocity, f0 = problem
        analytic = float(jax.grad(lambda h: _loss(config, chi_of, velocity, f0, h))(1.5))
        h = 1e-5
        difference = float(
            (
                _loss(config, chi_of, velocity, f0, 1.5 + h)
                - _loss(config, chi_of, velocity, f0, 1.5 - h)
            )
            / (2 * h)
        )
        assert analytic == pytest.approx(difference, rel=1e-5)

    def test_the_initial_guess_receives_no_gradient(self, problem):
        """The honest answer: move the guess, the fixed point does not move.

        This is the property that distinguishes the implicit rule from the
        taped one, where ``f0`` genuinely does influence the result.
        """
        config, chi_of, velocity, f0 = problem
        theta = {"chi": chi_of(1.5), "inlet_velocity": velocity}

        def loss(start):
            f = steady_populations(step_for(config), config.steady, theta, start)
            return jnp.sum(f * f)

        assert float(jnp.abs(jax.grad(loss)(f0)).max()) == 0.0

    @pytest.mark.parametrize("solver", ["gmres", "fixed_point"])
    def test_both_adjoint_solvers_give_the_same_gradient(self, problem, solver):
        """They fail differently; on a converged forward they must agree.

        ``fixed_point`` is Richardson on the same system the forward march
        solves, so it converges exactly when the forward does; ``gmres`` is
        faster but can stall.  Disagreement means the forward had not
        converged.
        """
        config, chi_of, velocity, f0 = problem
        reference = float(
            jax.grad(
                lambda h: _loss(
                    config, chi_of, velocity, f0, h, _options(adjoint_solver="fixed_point")
                )
            )(1.5)
        )
        value = float(
            jax.grad(
                lambda h: _loss(config, chi_of, velocity, f0, h, _options(adjoint_solver=solver))
            )(1.5)
        )
        assert value == pytest.approx(reference, rel=1e-7)

    def test_an_unknown_adjoint_solver_is_refused(self, problem):
        config, chi_of, velocity, f0 = problem
        with pytest.raises(ValueError, match="adjoint_solver must be"):
            jax.grad(
                lambda h: _loss(config, chi_of, velocity, f0, h, _options(adjoint_solver="lu"))
            )(1.5)

    def test_the_gradient_survives_jit(self, problem):
        """The adjoint has to compose, not just run at the top level."""
        config, chi_of, velocity, f0 = problem
        compiled = jax.jit(jax.grad(lambda h: _loss(config, chi_of, velocity, f0, h)))
        assert float(compiled(1.5)) == pytest.approx(
            float(jax.grad(lambda h: _loss(config, chi_of, velocity, f0, h))(1.5)), rel=1e-9
        )
