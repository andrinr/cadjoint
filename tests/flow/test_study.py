"""``FlowStudy``: the declaration, its refusals, and the coupled gradient.

The declaration half of this file checks that a scene can say what it wants
and be told precisely when it cannot.  The solve half checks the two claims
that would be expensive to be wrong about: that a study with a still inlet
is *exactly* the pure-conduction problem (not merely close to it), and that
a derivative taken through both solvers is the derivative of the composed
answer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint import extract_parameters, functionalize
from cadjoint.fem import Nodes
from cadjoint.fem.study import capture_studies
from cadjoint.flow import (
    FLOW_STUDY_KIND,
    EnergyConfig,
    FlowStudy,
    HeatSource,
    HeldTemperature,
    Inlet,
    Outlet,
    SteadyOptions,
    Walls,
    solve_temperature,
)
from cadjoint.geometry import Vector
from cadjoint.sdf.primitives import Box


def _block(size=(0.30, 0.40, 0.47)):
    """A box in the middle of the duct, with its half-extents as the design.

    The default leaves a real passage either side of it: a block that
    nearly plugs an 8-cell duct drives the density up by 70% and stops
    being a forced-convection problem.  ``0.47`` in ``z`` rather than
    ``0.5`` keeps the surface off the cell-centre plane, which is what
    makes a central difference on that axis second-order -- see
    :class:`TestCoupledGradient`.
    """
    solid = Box(Vector(list(size), free=True, name="size"))
    free, fixed, _ = extract_parameters(solid)
    return free, fixed, functionalize(solid)


def _study(name="duct", resolution=(8, 14, 8), speed=0.02, reynolds=6.0, **kwargs):
    """A small duct study that converges in a couple of seconds."""
    options = {
        "bounds": (-1.0, -1.0, -1.0),
        "size": (2.0, 2.0, 2.0),
        "reynolds": reynolds,
        "conductivity_ratio": 50.0,
        "energy_tol": 1e-12,
        "steady": SteadyOptions(
            tol=1e-11,
            max_steps=20000,
            adjoint_solver="fixed_point",
            adjoint_tol=1e-11,
            adjoint_max_steps=6000,
        ),
        "bcs": [
            Inlet(velocity=speed),
            Outlet(),
            Walls(),
            HeatSource(Nodes.box([-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]), power=1.0),
        ],
    }
    options.update(kwargs)
    return FlowStudy(name=name, resolution=resolution, **options)


class TestDeclaration:
    """A scene declares it the way it declares a ThermalStudy."""

    def test_a_study_registers_itself_with_capture_studies(self):
        """The compile worker's registry collects it alongside mesh studies."""
        with capture_studies() as captured:
            study = _study(name="captured")

        assert captured == [study]

    def test_a_study_outside_any_context_still_constructs(self):
        """Which is what makes it usable from a plain script."""
        assert _study(name="loose").name == "loose"

    def test_every_condition_answers_whether_it_serializes(self):
        """The viewer asks the *condition*, not a selection it might not have.

        ``Inlet``, ``Outlet`` and ``Walls`` are planes of the duct rather
        than node selections; asking them for ``.nodes.serializable`` is
        what used to break the compile of any scene declaring a flow study.
        """
        study = _study(name="serializes")

        assert [bc.serializable for bc in study.bcs] == [True, True, True, True]
        assert all(hasattr(bc, "describe") for bc in study.bcs)

    def test_a_predicate_region_reports_itself_unserializable(self):
        """The one case the flag exists for still reports it."""
        source = HeatSource(Nodes.predicate(lambda points: points[:, 0] > 0), power=1.0)

        assert source.serializable is False

    def test_describe_is_json_ready_and_names_its_kind(self):
        import json

        payload = _study(name="described").describe()

        assert payload["kind"] == FLOW_STUDY_KIND
        assert payload["name"] == "described"
        assert payload["resolution"] == [8, 14, 8]
        assert [bc["type"] for bc in payload["bcs"]] == [
            "inlet",
            "outlet",
            "walls",
            "heat_source",
        ]
        assert json.loads(json.dumps(payload)) == payload

    def test_describe_reports_the_regime_it_will_solve_in(self):
        """The dimensionless numbers a reader needs are in the payload, not
        derivable only after a solve."""
        payload = _study().describe()

        assert payload["fluid"]["inlet_speed"] == pytest.approx(0.02)
        assert payload["fluid"]["viscosity"] == pytest.approx(0.02 * 8 / 6.0)
        assert payload["fluid"]["peclet_cell"] > 0.0
        assert payload["fluid"]["richardson"] >= 0.0

    def test_the_default_conditions_are_the_ones_a_duct_has(self):
        """Omitting Outlet and Walls does not change the problem solved."""
        study = FlowStudy(name="bare", resolution=(6, 10, 6), bcs=[Inlet(velocity=0.01)])

        assert isinstance(study.outlet, Outlet)
        assert study.walls.temperature is None

    def test_a_scalar_inlet_velocity_means_along_the_duct(self):
        assert Inlet(velocity=0.03).velocity == (0.0, 0.03, 0.0)
        assert Inlet(velocity=0.03).speed == pytest.approx(0.03)


class TestRefusals:
    """Each one names what to do instead."""

    def test_a_study_without_an_inlet_is_refused(self):
        with pytest.raises(ValueError, match="needs an Inlet"):
            FlowStudy(name="x", resolution=(6, 10, 6), bcs=[Outlet(), Walls()])

    def test_two_inlets_are_refused(self):
        with pytest.raises(ValueError, match="one inlet plane"):
            FlowStudy(
                name="x", resolution=(6, 10, 6), bcs=[Inlet(velocity=0.01), Inlet(velocity=0.02)]
            )

    def test_a_mesh_study_condition_is_refused_naming_its_counterpart(self):
        from cadjoint.fem import Dirichlet

        with pytest.raises(ValueError, match="HeldTemperature"):
            FlowStudy(
                name="x",
                resolution=(6, 10, 6),
                bcs=[Inlet(velocity=0.01), Dirichlet(Nodes.side("+x"), 0.0)],
            )

    def test_a_lattice_too_small_to_have_an_interior_is_refused(self):
        with pytest.raises(ValueError, match="at least 4"):
            FlowStudy(name="x", resolution=(3, 10, 6), bcs=[Inlet(velocity=0.01)])

    def test_a_still_inlet_without_a_viscosity_is_refused(self):
        """``Re = U L / nu`` fixes no viscosity when ``U`` is zero, and the
        energy solve still needs the air's diffusivity."""
        with pytest.raises(ValueError, match="State viscosity"):
            FlowStudy(name="x", resolution=(6, 10, 6), bcs=[Inlet(velocity=0.0)])

    def test_a_supersonic_inlet_is_refused_as_outside_the_model(self):
        with pytest.raises(ValueError, match="lattice sound speed"):
            Inlet(velocity=0.7)

    def test_a_region_that_falls_between_cell_centres_is_refused(self):
        """A lattice region is volumetric and coarse; silence would be worse."""
        study = FlowStudy(
            name="x",
            resolution=(6, 10, 6),
            bounds=(-1.0, -1.0, -1.0),
            size=(2.0, 2.0, 2.0),
            viscosity=0.01,
            bcs=[
                Inlet(velocity=0.0),
                HeatSource(Nodes.sphere([0.0, 0.0, 0.0], 0.001), power=1.0),
            ],
        )

        with pytest.raises(ValueError, match="covers no cell"):
            study.solve(chi=jnp.zeros((6, 10, 6)))

    def test_two_held_regions_disagreeing_on_one_cell_are_refused(self):
        study = FlowStudy(
            name="x",
            resolution=(6, 10, 6),
            bounds=(-1.0, -1.0, -1.0),
            size=(2.0, 2.0, 2.0),
            viscosity=0.01,
            bcs=[
                Inlet(velocity=0.0),
                HeldTemperature(Nodes.box([-1, -1, -1], [1, 1, 1]), 1.0),
                HeldTemperature(Nodes.box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]), 2.0),
            ],
        )

        with pytest.raises(ValueError, match="different temperature"):
            study.solve(chi=jnp.zeros((6, 10, 6)))

    def test_an_unknown_profile_names_the_known_ones(self):
        with pytest.raises(ValueError, match="smootherstep"):
            FlowStudy(name="x", resolution=(6, 10, 6), profile="step", bcs=[Inlet(velocity=0.01)])

    def test_a_still_study_refuses_to_hand_out_a_momentum_config(self):
        study = FlowStudy(
            name="x", resolution=(6, 10, 6), viscosity=0.01, bcs=[Inlet(velocity=0.0)]
        )

        with pytest.raises(ValueError, match="no momentum problem"):
            study.flow_config()

    def test_solving_without_a_scene_or_a_field_is_refused(self):
        study = FlowStudy(
            name="x", resolution=(6, 10, 6), viscosity=0.01, bcs=[Inlet(velocity=0.0)]
        )

        with pytest.raises(ValueError, match="needs a scene to sample"):
            study.solve()


class TestZeroFlow:
    """A still inlet is the pure-conduction problem exactly, not nearly."""

    @pytest.fixture
    def still(self):
        return FlowStudy(
            name="still",
            resolution=(6, 12, 6),
            bounds=(-1.0, -1.0, -1.0),
            size=(2.0, 2.0, 2.0),
            viscosity=0.71 * 0.05,
            conductivity_ratio=50.0,
            energy_tol=1e-13,
            bcs=[
                Inlet(velocity=0.0),
                Outlet(),
                Walls(),
                HeatSource(Nodes.box([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]), power=1.0),
            ],
        )

    def test_the_momentum_solve_is_skipped_and_the_fluid_is_at_rest(self, still):
        result = still.solve(chi=jnp.zeros((6, 12, 6)))

        assert float(jnp.max(jnp.abs(result.velocity))) == 0.0
        assert float(jnp.max(jnp.abs(result.density - 1.0))) == 0.0
        assert float(result.pressure_drop) == 0.0

    def test_it_equals_the_pure_conduction_solve_to_the_last_bit(self, still):
        """Not "agrees to tolerance" -- the same numbers.

        This is what makes the conjugate path checkable against
        ``ThermalStudy`` at all: any difference found downstream is a
        discretisation difference between a lattice and a mesh, never a
        difference between "with flow" and "without".
        """
        chi = jnp.zeros((6, 12, 6)).at[2:4, 5:8, 2:4].set(1.0)
        source, power, fixed, value = still._regions()

        coupled = still.solve(chi=chi).temperature
        conduction = solve_temperature(
            chi,
            jnp.zeros((3, 6, 12, 6)),
            still.energy_config(),
            source=jnp.asarray(source),
            fixed_mask=fixed,
            fixed_value=jnp.asarray(value),
        )

        assert power == pytest.approx(1.0)
        assert np.array_equal(np.asarray(coupled), np.asarray(conduction))

    def test_a_still_study_reports_no_richardson_number(self, still):
        """Buoyancy over inertia is not a diagnostic when nothing drives the
        flow -- with no forced convection to compare against, any nonzero
        answer would be meaningless rather than reassuring."""
        assert still.richardson == 0.0


class TestCoupling:
    """Flow, then temperature, and the balance between them."""

    @pytest.fixture(scope="class")
    def solved(self):
        _, fixed, evaluate = _block()
        free = extract_parameters(Box(Vector([0.45, 0.45, 0.62], free=True, name="size")))[0]
        study = _study(name="coupled")
        return study, study.solve(evaluate(free, fixed))

    def test_the_flow_converges_and_carries_the_heat_downstream(self, solved):
        _, result = solved

        assert bool(jnp.all(jnp.isfinite(result.temperature)))
        assert float(result.pressure_drop) > 0.0
        assert float(result.bulk_outlet_temperature) > 0.0
        assert float(result.peak_temperature) >= float(result.mean_temperature)

    def test_the_energy_balance_closes_against_the_injected_power(self, solved):
        """Every watt in leaves through a boundary, to round-off.

        Not "to a few percent": summing the discrete equations makes every
        interior face cancel exactly, so this holds for any geometry and
        any velocity field. Two things had to be right for it -- the
        convective flux is the mass flux ``rho u`` rather than ``u``, and
        the inlet plane is counted with its conduction as well as its
        enthalpy. Either alone leaves a percent or two on the table.
        """
        _, result = solved

        assert result.energy_imbalance is not None
        assert abs(float(result.energy_imbalance)) < 1e-9

    def test_thermal_resistance_is_the_rise_over_the_power(self, solved):
        study, result = solved

        assert float(result.thermal_resistance) == pytest.approx(
            (float(result.peak_temperature) - study.inlet.temperature) / result.power
        )

    def test_moving_air_cools_the_block(self):
        """The whole point, stated as an inequality: the same power in the
        same geometry reaches a lower peak when the air moves."""
        _, fixed, evaluate = _block()
        free = extract_parameters(Box(Vector([0.45, 0.45, 0.62], free=True, name="size")))[0]
        sdf = evaluate(free, fixed)
        moving = _study(name="moving").solve(sdf)
        still = _study(name="still-cmp", speed=0.0, viscosity=0.02 * 8 / 6.0).solve(sdf)

        assert float(moving.peak_temperature) < float(still.peak_temperature)

    def test_the_balance_is_not_reported_when_heat_has_another_way_out(self):
        """A held region or an isothermal wall is an uncounted port, and
        calling the resulting mismatch a conservation failure would be
        reporting the boundary condition instead of the solver."""
        study = _study(
            name="held",
            bcs=[
                Inlet(velocity=0.02),
                Outlet(),
                Walls(temperature=0.0),
                HeatSource(Nodes.box([-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]), power=1.0),
            ],
        )
        _, fixed, evaluate = _block()
        free = extract_parameters(Box(Vector([0.45, 0.45, 0.62], free=True, name="size")))[0]

        assert study.solve(evaluate(free, fixed)).energy_imbalance is None


class TestWarnings:
    """What the study will not let a reader assume."""

    def test_a_large_richardson_number_is_reported(self):
        """At a crawling inlet speed buoyancy stops being negligible, and the
        one-way coupling stops being exact."""
        study = _study(name="buoyant", speed=0.001, reynolds=0.4)
        result = study.solve(chi=jnp.zeros((8, 14, 8)))

        assert study.richardson > 0.1
        assert any("Richardson" in note for note in result.warnings())

    def test_a_diverged_march_is_reported_rather_than_read_as_ambient(self):
        """A lattice this coarse loses BGK stability well below
        ``OMEGA_CEILING``.  The march NaNs, and the guard in
        ``solve_temperature`` keeps that visible instead of letting GMRES
        answer a NaN operator with its zero initial guess -- which would
        read as a heat sink sitting exactly at ambient.
        """
        _, fixed, evaluate = _block()
        free = extract_parameters(Box(Vector([0.45, 0.45, 0.62], free=True, name="size")))[0]
        result = _study(name="unstable", reynolds=10.0).solve(evaluate(free, fixed))

        assert not bool(jnp.isfinite(result.peak_temperature))
        assert any("diverged" in note for note in result.warnings())

    def test_a_converged_forced_convection_study_warns_about_nothing(self):
        _, fixed, evaluate = _block()
        free = extract_parameters(Box(Vector([0.45, 0.45, 0.62], free=True, name="size")))[0]

        assert _study(name="clean").solve(evaluate(free, fixed)).warnings() == []


class TestCoupledGradient:
    """The claim most likely to be quietly wrong."""

    def test_the_derivative_through_both_solvers_matches_a_central_difference(self):
        """``d(mean temperature)/d(block size)`` runs through the momentum
        fixed point's adjoint *and* the energy solve's transposed linear
        solve.  Either one being wrong shows up here and nowhere cheaper.

        The tolerance is tight because the check is only meaningful if the
        difference itself is converging: the two step sizes are asserted to
        agree at second order, which is what a central difference does on a
        smooth objective. Without that, "matches at one ``h``" could be
        agreement with a truncation error that happens to be small.
        """
        free, fixed, evaluate = _block()
        study = _study(name="grad", resolution=(10, 18, 10), reynolds=12.0)

        def objective(parameters):
            return study.solve(evaluate(parameters, fixed)).mean_temperature

        def difference(axis, step):
            plus = dict(free, size=free["size"].at[axis].add(step))
            minus = dict(free, size=free["size"].at[axis].add(-step))
            return float((objective(plus) - objective(minus)) / (2 * step))

        gradient = np.asarray(jax.grad(objective)(free)["size"])
        assert np.all(np.isfinite(gradient))

        coarse = abs(difference(2, 1e-3) - gradient[2]) / abs(gradient[2])
        fine = abs(difference(2, 1e-4) - gradient[2]) / abs(gradient[2])

        assert fine < 5e-5
        assert coarse / fine > 30.0

    def test_the_energy_config_the_study_builds_is_the_one_it_solves(self):
        study = _study(name="cfg")
        config = study.energy_config()

        assert isinstance(config, EnergyConfig)
        assert config.fluid_diffusivity == pytest.approx(study._viscosity() / study.prandtl)
        assert config.shape == (8, 14, 8)
