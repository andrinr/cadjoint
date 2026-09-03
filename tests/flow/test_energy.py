"""The conjugate energy equation, checked against answers it cannot produce.

Every claim :mod:`cadjoint.flow.energy` makes in prose is checked here
against something external to it: a closed-form conduction profile, a
two-layer series resistance, the analytic advection-diffusion column, and a
global energy balance.  The point of a single-domain formulation is that
temperature and heat flux are continuous at the metal-air interface *by
construction*; :class:`TestInterface` is where that stops being a claim.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.flow import EnergyConfig, conductivity, duct_walls, solve_temperature
from cadjoint.flow.energy import (
    bulk_outlet_temperature,
    energy_imbalance,
    mean_temperature,
    peak_temperature,
    thermal_resistance,
)


def _column(shape):
    """A no-flow lattice: zero velocity, no solid."""
    return jnp.zeros(shape), jnp.zeros((3, *shape))


class TestConduction:
    """With no flow the equation is Poisson's, and its answer is known."""

    @pytest.mark.parametrize(("cells", "expected"), [(8, 3.9e-3), (16, 9.8e-4), (32, 2.5e-4)])
    def test_matches_the_analytic_profile_at_second_order(self, cells, expected):
        """``-k T'' = q``, ``T(0) = 0``, ``T'(L) = 0`` on a filled duct.

        The lattice is cell-centred and the inlet Dirichlet sits half a cell
        upstream of the first centre, so the discrete answer is not the
        analytic one -- it approaches it at second order, and the parameters
        pin that rate: each refinement divides the error by four.
        """
        shape = (6, cells, 6)
        step = 1.0 / cells
        core = int((~duct_walls(shape)).sum())
        config = EnergyConfig(shape=shape, fluid_diffusivity=1.0, conductivity_ratio=1.0, tol=1e-13)
        source = jnp.full(shape, step * step)

        temperature = solve_temperature(
            jnp.ones(shape), jnp.zeros((3, *shape)), config, source=source
        )

        line = np.asarray(temperature[3, :, 3])
        y = (np.arange(cells) + 0.5) * step
        exact = y - 0.5 * y * y
        assert core > 0
        assert np.max(np.abs(line - exact)) / np.max(exact) == pytest.approx(expected, rel=0.05)

    def test_an_unsourced_duct_stays_at_the_inlet_temperature(self):
        """No source, no flow, nothing held: the answer is uniform."""
        shape = (6, 10, 6)
        config = EnergyConfig(shape=shape, fluid_diffusivity=0.1, inlet_temperature=0.0)
        chi, velocity = _column(shape)

        temperature = solve_temperature(chi, velocity, config)

        assert float(jnp.max(jnp.abs(temperature))) == pytest.approx(0.0, abs=1e-12)

    def test_the_discrete_balance_closes_cell_by_cell(self):
        """Every solved cell's fluxes sum to its source, which is what a
        finite-volume scheme is *for* -- and what a stencil sign error
        would break without moving the answer much."""
        shape = (5, 9, 5)
        config = EnergyConfig(shape=shape, fluid_diffusivity=1.0, conductivity_ratio=1.0, tol=1e-13)
        source = jnp.full(shape, 0.01)

        temperature = solve_temperature(
            jnp.zeros(shape), jnp.zeros((3, *shape)), config, source=source
        )

        laplacian = -6.0 * temperature
        for axis in range(3):
            for offset in (1, -1):
                laplacian = laplacian + jnp.roll(temperature, offset, axis=axis)
        interior = (slice(2, -2), slice(1, -1), slice(2, -2))
        assert float(jnp.max(jnp.abs(laplacian[interior] + 0.01))) < 1e-9


class TestInterface:
    """Continuity of heat flux across a metal-air face, without an interface."""

    @pytest.mark.parametrize("ratio", [10.0, 200.0, 8000.0])
    def test_two_layers_reproduce_the_exact_series_resistance(self, ratio):
        """A step in ``k`` landing on a face is resolved, not smeared.

        The harmonic face mean *is* the series resistance of the two half
        cells, so the discrete answer is the analytic one to machine
        precision at every conductivity ratio -- including aluminium in air
        at 8000, where an arithmetic mean would be wrong by a factor of
        thousands.
        """
        cells, split = 24, 12
        shape = (4, cells, 4)
        fluid = 0.05
        config = EnergyConfig(
            shape=shape, fluid_diffusivity=fluid, conductivity_ratio=ratio, tol=1e-13, restart=50
        )
        chi = np.zeros(shape)
        chi[:, split:, :] = 1.0
        held = np.zeros(shape, dtype=bool)
        held[:, -1, :] = True

        temperature = solve_temperature(
            jnp.asarray(chi), jnp.zeros((3, *shape)), config, fixed_mask=held, fixed_value=1.0
        )

        solid = fluid * ratio
        resistance = np.zeros(cells)
        resistance[0] = 0.5 / fluid
        for index in range(1, cells):
            lower = fluid if index - 1 < split else solid
            upper = fluid if index < split else solid
            resistance[index] = resistance[index - 1] + 0.5 / lower + 0.5 / upper
        exact = resistance / resistance[-1]

        assert np.max(np.abs(np.asarray(temperature[2, :, 2]) - exact)) < 1e-12

    def test_conductivity_interpolates_linearly_between_the_phases(self):
        config = EnergyConfig(shape=(2, 2, 2), fluid_diffusivity=0.01, conductivity_ratio=100.0)
        chi = jnp.array([0.0, 0.25, 1.0])

        values = conductivity(chi, config)

        assert np.allclose(np.asarray(values), [0.01, 0.2575, 1.0])


class TestAdvection:
    """The exponential blend, and the false diffusion it exists to avoid."""

    def test_the_exponential_scheme_is_nodally_exact(self):
        """One-dimensional advection-diffusion at cell Peclet 2.

        Patankar's exponential blend reproduces ``(e^{Pe s/L} - 1) /
        (e^{Pe} - 1)`` at every node exactly, at any Peclet number. This is
        the check that the convective coefficients and the half-cell inlet
        conductance are both right, because either being wrong perturbs it.
        """
        cells = 12
        shape = (4, cells, 4)
        speed, diffusivity = 0.1, 0.05
        config = EnergyConfig(
            shape=shape, fluid_diffusivity=diffusivity, conductivity_ratio=1.0, tol=1e-13
        )
        velocity = jnp.zeros((3, *shape)).at[1].set(speed)
        held = np.zeros(shape, dtype=bool)
        held[:, -1, :] = True

        temperature = solve_temperature(
            jnp.zeros(shape), velocity, config, fixed_mask=held, fixed_value=1.0
        )

        length = cells - 0.5
        peclet = speed * length / diffusivity
        station = np.arange(cells) + 0.5
        exact = np.expm1(peclet * station / length) / np.expm1(peclet)

        assert np.max(np.abs(np.asarray(temperature[2, :, 2]) - exact)) < 1e-12

    def test_upwind_carries_the_false_diffusion_the_docstring_claims(self):
        """The same column on the upwind blend is off by a fifth of its range.

        Not a defect of the implementation -- it is what ``A = 1`` means at
        cell Peclet 2, and the reason ``"exponential"`` is the default. If
        this number ever gets *small*, the exponential scheme has stopped
        being different and something is wrong with the blend.
        """
        cells = 12
        shape = (4, cells, 4)
        speed, diffusivity = 0.1, 0.05
        held = np.zeros(shape, dtype=bool)
        held[:, -1, :] = True
        velocity = jnp.zeros((3, *shape)).at[1].set(speed)
        length = cells - 0.5
        peclet = speed * length / diffusivity
        exact = np.expm1(peclet * (np.arange(cells) + 0.5) / length) / np.expm1(peclet)

        upwind = solve_temperature(
            jnp.zeros(shape),
            velocity,
            EnergyConfig(
                shape=shape,
                fluid_diffusivity=diffusivity,
                conductivity_ratio=1.0,
                scheme="upwind",
                tol=1e-13,
            ),
            fixed_mask=held,
            fixed_value=1.0,
        )

        assert np.max(np.abs(np.asarray(upwind[2, :, 2]) - exact)) == pytest.approx(0.198, abs=0.01)

    def test_the_linear_solve_survives_pure_advection(self):
        """Regression: bicgstab returned NaN here, silently, at every tolerance.

        A column with an inlet temperature at one end and a held one at the
        other is the case where the bicgstab ``rho`` recurrence collapses.
        The routine reported success and returned ``NaN``, which an
        objective would have carried into an optimizer. GMRES does not.
        """
        shape = (4, 12, 4)
        held = np.zeros(shape, dtype=bool)
        held[:, -1, :] = True
        config = EnergyConfig(shape=shape, fluid_diffusivity=0.05, conductivity_ratio=1.0)

        temperature = solve_temperature(
            jnp.zeros(shape),
            jnp.zeros((3, *shape)).at[1].set(0.1),
            config,
            fixed_mask=held,
            fixed_value=1.0,
        )

        assert bool(jnp.all(jnp.isfinite(temperature)))


class TestConservation:
    """What goes in comes out, whatever the geometry."""

    def test_the_energy_balance_closes_to_solver_tolerance(self):
        """Power in equals enthalpy out, independent of the design.

        The strongest check available without an analytic answer, because
        it holds for *any* geometry: a conservative scheme satisfies it
        whatever the solid fraction is doing.
        """
        shape = (8, 16, 8)
        config = EnergyConfig(
            shape=shape, fluid_diffusivity=0.02, conductivity_ratio=100.0, tol=1e-13
        )
        chi = np.zeros(shape)
        chi[3:5, 6:10, 3:5] = 1.0
        velocity = jnp.zeros((3, *shape)).at[1].set(0.05 * (1.0 - jnp.asarray(chi)))
        source = jnp.where(jnp.asarray(chi) > 0.5, 1.0, 0.0)
        power = float(jnp.sum(source))

        temperature = solve_temperature(jnp.asarray(chi), velocity, config, source=source)

        residual = energy_imbalance(temperature, jnp.asarray(chi), velocity, power, config)
        assert abs(float(residual)) < 1e-9

    def test_the_read_outs_agree_with_their_definitions(self):
        shape = (6, 10, 6)
        chi = jnp.zeros(shape).at[2:4, 4:6, 2:4].set(1.0)
        temperature = jnp.arange(float(np.prod(shape))).reshape(shape)
        velocity = jnp.zeros((3, *shape)).at[1].set(1.0)

        assert float(peak_temperature(temperature, chi)) == float(
            jnp.max(jnp.where(chi > 0.5, temperature, -jnp.inf))
        )
        assert float(mean_temperature(temperature, chi)) == pytest.approx(
            float(jnp.sum(chi * temperature) / jnp.sum(chi))
        )
        assert float(thermal_resistance(temperature, chi, 4.0)) == pytest.approx(
            float(peak_temperature(temperature, chi)) / 4.0
        )
        assert float(bulk_outlet_temperature(temperature, velocity)) == pytest.approx(
            float(jnp.mean(temperature[1:-1, -1, 1:-1]))
        )


class TestGradient:
    """The implicit adjoint of the linear solve, against a finite difference."""

    def test_the_temperature_gradient_matches_a_central_difference(self):
        shape = (6, 12, 6)
        config = EnergyConfig(
            shape=shape, fluid_diffusivity=0.02, conductivity_ratio=100.0, tol=1e-13
        )
        base = np.zeros(shape)
        base[2:4, 5:8, 2:4] = 1.0
        chi = jnp.asarray(base)
        velocity = jnp.zeros((3, *shape)).at[1].set(0.05)
        source = jnp.where(chi > 0.5, 1.0, 0.0)

        def objective(field):
            return mean_temperature(
                solve_temperature(field, velocity, config, source=source), field
            )

        gradient = jax.grad(objective)(chi)
        direction = jax.random.normal(jax.random.PRNGKey(0), shape)
        direction = direction / jnp.linalg.norm(direction)
        step = 1e-6
        difference = (objective(chi + step * direction) - objective(chi - step * direction)) / (
            2 * step
        )

        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(difference) == pytest.approx(float(jnp.sum(gradient * direction)), rel=1e-5)

    def test_the_gradient_flows_to_the_source_as_well(self):
        """A study optimizing where the heat goes in needs this path too."""
        shape = (5, 10, 5)
        config = EnergyConfig(shape=shape, fluid_diffusivity=0.05, tol=1e-13)
        chi = jnp.zeros(shape).at[2, 4:6, 2].set(1.0)

        def objective(power):
            source = jnp.zeros(shape).at[2, 4:6, 2].set(power)
            return mean_temperature(
                solve_temperature(chi, jnp.zeros((3, *shape)), config, source=source), chi
            )

        gradient = float(jax.grad(objective)(1.0))
        step = 1e-5
        difference = (objective(1.0 + step) - objective(1.0 - step)) / (2 * step)

        assert gradient == pytest.approx(float(difference), rel=1e-6)


class TestRefusals:
    """A configuration that cannot mean anything says so at construction."""

    def test_a_zero_diffusivity_is_refused(self):
        with pytest.raises(ValueError, match="fluid_diffusivity must be positive"):
            EnergyConfig(shape=(4, 4, 4), fluid_diffusivity=0.0)

    def test_a_negative_conductivity_ratio_is_refused(self):
        with pytest.raises(ValueError, match="conductivity_ratio must be positive"):
            EnergyConfig(shape=(4, 4, 4), fluid_diffusivity=0.1, conductivity_ratio=-1.0)

    def test_an_unknown_scheme_names_the_known_ones(self):
        with pytest.raises(ValueError, match="exponential"):
            EnergyConfig(shape=(4, 4, 4), fluid_diffusivity=0.1, scheme="quick")

    def test_a_mismatched_field_names_both_shapes(self):
        config = EnergyConfig(shape=(4, 6, 4), fluid_diffusivity=0.1)

        with pytest.raises(ValueError, match=r"expected \(4, 6, 4\)"):
            solve_temperature(jnp.zeros((4, 4, 4)), jnp.zeros((3, 4, 6, 4)), config)

    def test_a_mismatched_velocity_names_both_shapes(self):
        config = EnergyConfig(shape=(4, 6, 4), fluid_diffusivity=0.1)

        with pytest.raises(ValueError, match=r"expected \(3, 4, 6, 4\)"):
            solve_temperature(jnp.zeros((4, 6, 4)), jnp.zeros((3, 4, 4, 4)), config)
