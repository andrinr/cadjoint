"""The lattice Boltzmann step: its moments, its walls, and its drag."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.flow import (
    C,
    bounce_back_mask,
    duct_walls,
    equilibrium,
    initial_populations,
    macroscopic,
    step,
    stream,
)

SHAPE = (6, 10, 6)


class TestEquilibrium:
    def test_it_carries_the_density_it_was_given(self):
        rho = jnp.full(SHAPE, 1.3)
        u = jnp.zeros((3, *SHAPE))
        assert np.allclose(jnp.sum(equilibrium(rho, u), axis=0), 1.3)

    def test_it_carries_the_momentum_it_was_given(self):
        rho = jnp.ones(SHAPE)
        u = jnp.broadcast_to(jnp.array([0.01, -0.02, 0.03])[:, None, None, None], (3, *SHAPE))
        feq = equilibrium(rho, u)
        momentum = jnp.tensordot(jnp.asarray(C, dtype=feq.dtype).T, feq, axes=(1, 0))
        assert np.allclose(momentum, u, atol=1e-12)

    def test_at_rest_it_is_the_weights(self):
        from cadjoint.flow import W

        feq = equilibrium(jnp.ones((1, 1, 1)), jnp.zeros((3, 1, 1, 1)))
        assert np.allclose(feq[:, 0, 0, 0], W)


class TestMacroscopic:
    def test_without_drag_it_inverts_equilibrium(self):
        u = jnp.broadcast_to(jnp.array([0.01, 0.02, -0.01])[:, None, None, None], (3, *SHAPE))
        rho, recovered = macroscopic(equilibrium(jnp.full(SHAPE, 1.1), u), jnp.zeros(SHAPE))
        assert np.allclose(rho, 1.1)
        assert np.allclose(recovered, u, atol=1e-12)

    def test_drag_divides_the_velocity_implicitly(self):
        """``u = m / (rho + alpha/2)`` in closed form -- no explicit update.

        This is what lets ``alpha_max`` run to hundreds without the march
        going unstable.
        """
        u = jnp.broadcast_to(jnp.array([0.0, 0.05, 0.0])[:, None, None, None], (3, *SHAPE))
        f = equilibrium(jnp.ones(SHAPE), u)
        alpha = jnp.full(SHAPE, 2.0)
        _, damped = macroscopic(f, alpha)
        assert np.allclose(damped, u / (1.0 + 0.5 * 2.0), atol=1e-12)

    @pytest.mark.parametrize("alpha_max", [1e2, 1e4, 1e6])
    def test_a_huge_drag_stays_finite(self, alpha_max):
        """The implicit form cannot blow up, however hard the solid pushes."""
        f = initial_populations(SHAPE, jnp.array([0.0, 0.05, 0.0]))
        _, u = macroscopic(f, jnp.full(SHAPE, alpha_max))
        assert np.all(np.isfinite(u))
        assert float(jnp.abs(u).max()) < 0.05


class TestStreaming:
    def test_it_conserves_mass(self):
        f = initial_populations(SHAPE, jnp.array([0.0, 0.02, 0.0]))
        assert float(jnp.sum(stream(f))) == pytest.approx(float(jnp.sum(f)), rel=1e-12)

    def test_each_direction_shifts_by_its_own_velocity(self):
        f = jnp.zeros((19, *SHAPE)).at[:, 2, 3, 2].set(1.0)
        streamed = stream(f)
        for q in (0, 1, 7, 18):
            target = tuple((np.array([2, 3, 2]) + C[q]) % np.array(SHAPE))
            assert float(streamed[q][target]) == pytest.approx(1.0)


class TestWalls:
    def test_the_duct_is_open_along_y_and_closed_across_it(self):
        wall = duct_walls(SHAPE)
        assert wall[0].all() and wall[-1].all()
        assert wall[:, :, 0].all() and wall[:, :, -1].all()
        assert not wall[1:-1, :, 1:-1].any()

    def test_the_masks_are_concrete_arrays_not_traced_ones(self):
        """Regression: ``step_for`` caches these on the config.

        Built with ``jnp`` they would be tracers on a first traced call, and
        the cache would hand that tracer to every later call -- which JAX
        reports as a leaked value.  NumPy cannot leak.
        """
        wall = duct_walls(SHAPE)
        assert isinstance(wall, np.ndarray)
        assert isinstance(bounce_back_mask(wall), np.ndarray)

    def test_incoming_marks_fluid_nodes_whose_neighbour_is_a_wall(self):
        wall = duct_walls(SHAPE)
        incoming = bounce_back_mask(wall)
        assert incoming.shape == (19, *SHAPE)
        # Never set on a wall node itself: walls do not receive.
        assert not incoming[:, wall].any()
        # The rest direction can never arrive from anywhere else.
        assert not incoming[0].any()
        # A node against the x wall receives from it along +x.
        plus_x = int(np.argwhere((C == [1, 0, 0]).all(axis=1))[0, 0])
        assert incoming[plus_x, 1, 5, 3]


class TestStep:
    @staticmethod
    def _step(f, chi, u_in=0.02, alpha_max=200.0, omega=1.5):
        wall = duct_walls(SHAPE)
        return step(
            f,
            chi,
            jnp.array([0.0, u_in, 0.0]),
            omega=omega,
            alpha_max=alpha_max,
            wall=wall,
            incoming=bounce_back_mask(wall),
        )

    def test_quiescent_fluid_with_no_inflow_stays_quiescent(self):
        """A fixed point that can be checked by inspection."""
        f = initial_populations(SHAPE, jnp.array([0.0, 0.0, 0.0]))
        advanced = self._step(f, jnp.zeros(SHAPE), u_in=0.0)
        assert np.allclose(advanced, f, atol=1e-14)

    def test_halfway_bounce_back_has_a_genuine_fixed_point(self):
        """The reason the walls are halfway rather than fullway.

        Fullway bounce-back returns a population to its origin over *two*
        steps, putting an eigenvalue at -1 in the step operator: the march
        settles onto a period-2 cycle, and ``f* = T(f*)`` has no solution to
        find.  Halfway reflects within one step, so the converged state is a
        true fixed point -- which is the object the adjoint differentiates.
        """
        from cadjoint.flow import FlowConfig, SteadyOptions, iterate_to_steady, step_for

        config = FlowConfig(
            shape=SHAPE,
            inlet_speed=0.02,
            reynolds=10.0,
            characteristic_cells=6,
            alpha_max=1.0,
            steady=SteadyOptions(max_steps=4000, tol=1e-14, check_every=50),
        )
        step_fn = step_for(config)
        theta = {"chi": jnp.zeros(SHAPE), "inlet_velocity": jnp.array([0.0, 0.02, 0.0])}
        f_star = iterate_to_steady(
            step_fn, theta, initial_populations(SHAPE, theta["inlet_velocity"]), config.steady
        )
        once = jnp.linalg.norm(step_fn(f_star, theta) - f_star) / jnp.linalg.norm(f_star)
        # Fullway bounce-back leaves 4.1e-2 here, eight orders larger: the
        # bar is not tightness, it is the difference between converging and
        # cycling.
        assert float(once) < 1e-10

    def test_the_drag_stops_the_flow_where_the_solid_is(self):
        """Penalisation, end to end: fill the duct and the velocity dies."""
        f = initial_populations(SHAPE, jnp.array([0.0, 0.02, 0.0]))
        chi = jnp.ones(SHAPE)
        for _ in range(30):
            f = self._step(f, chi)
        _, u = macroscopic(f, 200.0 * chi)
        assert float(jnp.abs(u[:, 2:4, 4:6, 2:4]).max()) < 1e-3

    def test_it_is_differentiable_in_the_solid_fraction(self):
        """``chi`` is the design; a step with no gradient in it is useless."""
        f = initial_populations(SHAPE, jnp.array([0.0, 0.02, 0.0]))
        grad = jax.grad(lambda chi: jnp.sum(self._step(f, chi) ** 2))(jnp.full(SHAPE, 0.5))
        assert np.all(np.isfinite(grad))
        assert float(jnp.linalg.norm(grad)) > 0.0
