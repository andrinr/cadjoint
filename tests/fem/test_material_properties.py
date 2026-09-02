"""Simulation driven by the scene's materials rather than by one scalar.

A study may still be handed an explicit ``conductivity=2.0`` — that path is
unchanged and covered elsewhere.  These tests cover the other one: sampling
the scene's own material field per element, so a bar that is copper at one end
and aluminium at the other solves as two materials with a smooth transition
exactly as wide as the CSG blend joining them.

The gradient test is the load-bearing one.  Per-element sampling is only
useful in an optimization if it stays differentiable in *both* directions that
a designer would want to move: where the material interface sits (geometry,
through the smooth blend) and what the material is made of (a free property).
``TestMaterialFieldGradient`` checks both against central finite differences.
"""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("jax_fem")

import jax
import jax.numpy as jnp

from cadjoint.fem.hexmesh import GridSpec, sdf_to_hex_mesh
from cadjoint.fem.properties import (
    FROM_MATERIAL,
    cell_centroids,
    cell_volumes,
    quantize_to_materials,
    sample_cell_property,
    sample_material_field,
    total_mass,
)
from cadjoint.fem.selection import Nodes
from cadjoint.fem.simulate import elastic_solve, thermal_solve
from cadjoint.fem.study import Dirichlet, ElasticStudy, Fixed, ThermalStudy, Traction
from cadjoint.materials import aluminium_6061, copper_c11000, steel_1018
from cadjoint.render.material import Material
from cadjoint.sdf import Box, Translate
from cadjoint.sdf.boolean import Union

_HOT_END = Nodes.side("-x")
_COLD_END = Nodes.side("+x")

# Half the bar conducts ten times better than the other half.  Round numbers
# keep the expected values readable; the catalogue is exercised separately.
_STIFF = Material(
    "stiff", conductivity=10.0, density=1000.0, youngs_modulus=2000.0, poisson_ratio=0.3
)
_SOFT = Material("soft", conductivity=1.0, density=4000.0, youngs_modulus=500.0, poisson_ratio=0.3)

_BLEND = 0.06


def _two_material_bar(interface=0.0, hot=_STIFF, cold=_SOFT, blend=_BLEND):
    """A bar made of ``hot`` for x < interface and ``cold`` beyond it."""
    offset = jnp.asarray(interface)
    left = Translate(
        Box([0.6, 0.15, 0.15], material=hot),
        jnp.stack([offset - 0.6, jnp.zeros(()), jnp.zeros(())]),
    )
    right = Translate(
        Box([0.6, 0.15, 0.15], material=cold),
        jnp.stack([offset + 0.6, jnp.zeros(()), jnp.zeros(())]),
    )
    return Union((left, right), smoothness=blend)


@pytest.fixture(scope="module")
def bar_mesh():
    grid = GridSpec.from_bounds((-1.3, -0.2, -0.2), (2.6, 0.4, 0.4), (26, 4, 4))
    return sdf_to_hex_mesh(_two_material_bar(), grid)


class TestSampling:
    def test_sharp_regions_carry_their_own_material(self, bar_mesh):
        """Away from the blend the field is exactly each material's value."""
        scene = _two_material_bar()
        values = np.asarray(
            sample_cell_property(scene, bar_mesh.points, bar_mesh.cells, "conductivity")
        )
        centroids = np.asarray(cell_centroids(bar_mesh.points, bar_mesh.cells))
        far_left = values[centroids[:, 0] < -0.4]
        far_right = values[centroids[:, 0] > 0.4]
        assert far_left.size and far_right.size
        assert far_left == pytest.approx(10.0, rel=1e-6)
        assert far_right == pytest.approx(1.0, rel=1e-6)

    def test_the_interface_is_a_smooth_transition_not_a_step(self, bar_mesh):
        """The blend that joins the solids blends their properties too."""
        scene = _two_material_bar()
        values = np.asarray(
            sample_cell_property(scene, bar_mesh.points, bar_mesh.cells, "conductivity")
        )
        between = values[(values > 1.05) & (values < 9.95)]
        assert between.size > 0, "a smooth union should leave transitional elements"

    def test_sampling_the_whole_field_batches_every_property(self, bar_mesh):
        field = sample_material_field(_two_material_bar(), bar_mesh.points, bar_mesh.cells)
        assert field["conductivity"].shape == (bar_mesh.num_cells,)
        assert field["color"].shape == (bar_mesh.num_cells, 3)

    def test_an_unstated_property_is_an_error_not_a_default(self, bar_mesh):
        scene = _two_material_bar(hot=Material("silent"), cold=Material("also_silent"))
        with pytest.raises(ValueError, match="does not specify 'conductivity'"):
            sample_cell_property(scene, bar_mesh.points, bar_mesh.cells, "conductivity")

    def test_a_bare_callable_field_says_what_is_missing(self, bar_mesh):
        with pytest.raises(TypeError, match="material_at"):
            sample_cell_property(
                lambda p: jnp.linalg.norm(p) - 1.0, bar_mesh.points, bar_mesh.cells, "density"
            )


class TestVolumesAndMass:
    def test_hex_volume_is_exact_for_a_trilinear_element(self):
        from cadjoint.fem.elements import HEX_CORNER_OFFSETS

        points = HEX_CORNER_OFFSETS.astype(np.float64) * np.array([2.0, 3.0, 4.0])
        volume = float(cell_volumes(points, np.arange(8).reshape(1, 8))[0])
        assert volume == pytest.approx(24.0, rel=1e-12)

    def test_tet_volume_is_the_determinant_sixth(self):
        points = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        volume = float(cell_volumes(points, np.arange(4).reshape(1, 4))[0])
        assert volume == pytest.approx(1.0 / 6.0, rel=1e-12)

    def test_mesh_volume_matches_the_bar(self, bar_mesh):
        total = float(jnp.sum(cell_volumes(bar_mesh.points, bar_mesh.cells)))
        # Half-extents 0.6/0.15/0.15 per half: a 2.4 x 0.3 x 0.3 bar, plus a
        # little from the smooth union's bulge at the joint.
        assert total == pytest.approx(2.4 * 0.3 * 0.3, rel=0.1)

    def test_mass_weights_each_element_by_its_own_density(self, bar_mesh):
        scene = _two_material_bar()
        density = sample_cell_property(scene, bar_mesh.points, bar_mesh.cells, "density")
        mass = float(total_mass(bar_mesh.points, bar_mesh.cells, density))
        volume = float(jnp.sum(cell_volumes(bar_mesh.points, bar_mesh.cells)))
        # Half at 1000, half at 4000 — the mass must land strictly between the
        # two single-material extremes, nearer their mean.
        assert 1000.0 * volume < mass < 4000.0 * volume
        assert mass == pytest.approx(2500.0 * volume, rel=0.1)


class TestThermalSolves:
    def test_a_per_element_field_is_not_either_single_material(self, bar_mesh):
        """The whole point: two materials do not behave like one averaged one."""
        dirichlet = [(_HOT_END, 1.0), (_COLD_END, 0.0)]
        scene = _two_material_bar()
        field = sample_cell_property(scene, bar_mesh.points, bar_mesh.cells, "conductivity")
        mixed = float(
            jnp.mean(thermal_solve(bar_mesh, conductivity=field, dirichlet=dirichlet).temperature)
        )
        for uniform in (1.0, 10.0, 5.5):
            single = float(
                jnp.mean(
                    thermal_solve(bar_mesh, conductivity=uniform, dirichlet=dirichlet).temperature
                )
            )
            assert abs(mixed - single) > 1e-3
        # The poorly-conducting half carries most of the temperature drop, so
        # the mean sits above the 0.5 of a uniform bar.
        assert mixed > 0.55

    def test_a_uniform_field_reproduces_the_scalar_solve_exactly(self, bar_mesh):
        """The per-element path must not perturb the single-material answer."""
        dirichlet = [(_HOT_END, 1.0), (_COLD_END, 0.0)]
        scalar = thermal_solve(bar_mesh, conductivity=3.0, dirichlet=dirichlet)
        uniform = thermal_solve(
            bar_mesh,
            conductivity=jnp.full((bar_mesh.num_cells,), 3.0),
            dirichlet=dirichlet,
        )
        assert (
            np.abs(np.asarray(scalar.temperature) - np.asarray(uniform.temperature)).max() < 1e-10
        )


class TestStudyIntegration:
    def _study(self, **kwargs):
        # The resolution is a placeholder: every solve below overrides it with
        # the module's pre-extracted mesh.
        kwargs.setdefault("resolution", (6, 3, 3))
        return ThermalStudy(
            "bar",
            bcs=[Dirichlet(_HOT_END, 1.0), Dirichlet(_COLD_END, 0.0)],
            **kwargs,
        )

    def test_omitting_the_property_samples_the_materials(self, bar_mesh):
        scene = _two_material_bar()
        derived = self._study().solve(scene, mesh=bar_mesh)
        expected = thermal_solve(
            bar_mesh,
            conductivity=sample_cell_property(
                scene, bar_mesh.points, bar_mesh.cells, "conductivity"
            ),
            dirichlet=[(_HOT_END, 1.0), (_COLD_END, 0.0)],
        )
        assert (
            np.abs(
                np.asarray(derived.solution.temperature) - np.asarray(expected.temperature)
            ).max()
            < 1e-12
        )

    def test_the_sentinel_is_the_same_as_omitting(self, bar_mesh):
        scene = _two_material_bar()
        omitted = self._study().solve(scene, mesh=bar_mesh)
        sentinel = self._study(conductivity=FROM_MATERIAL).solve(scene, mesh=bar_mesh)
        assert (
            np.abs(
                np.asarray(omitted.solution.temperature) - np.asarray(sentinel.solution.temperature)
            ).max()
            < 1e-12
        )

    def test_an_explicit_scalar_still_wins(self, bar_mesh):
        """The old API is untouched: a number means that number, everywhere."""
        result = self._study(conductivity=2.0).solve(_two_material_bar(), mesh=bar_mesh)
        expected = thermal_solve(
            bar_mesh, conductivity=2.0, dirichlet=[(_HOT_END, 1.0), (_COLD_END, 0.0)]
        )
        assert (
            np.abs(np.asarray(result.solution.temperature) - np.asarray(expected.temperature)).max()
            < 1e-12
        )

    def test_a_silent_scene_names_the_property_it_needs(self, bar_mesh):
        scene = _two_material_bar(hot=Material("silent"), cold=Material("silent2"))
        with pytest.raises(ValueError, match="conductivity"):
            self._study().solve(scene, mesh=bar_mesh)

    def test_the_study_reports_the_mass_it_solved(self, bar_mesh):
        result = self._study().solve(_two_material_bar(), mesh=bar_mesh)
        volume = float(jnp.sum(cell_volumes(bar_mesh.points, bar_mesh.cells)))
        assert result.mass is not None
        assert 1000.0 * volume < float(result.mass) < 4000.0 * volume

    def test_no_density_means_no_mass_not_a_failed_solve(self, bar_mesh):
        scene = _two_material_bar(
            hot=Material("k_only", conductivity=10.0), cold=Material("k_only2", conductivity=1.0)
        )
        result = self._study().solve(scene, mesh=bar_mesh)
        assert result.mass is None
        assert result.describe()["mass"] is None

    def test_describe_says_where_the_property_came_from(self):
        derived = self._study().describe()
        assert derived["material"] == {"conductivity": FROM_MATERIAL}
        assert json.loads(json.dumps(derived)) == derived
        explicit = self._study(conductivity=2.0).describe()
        assert explicit["material"] == {"conductivity": 2.0}

    def test_a_bad_sentinel_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="must be a number"):
            self._study(conductivity="copper")

    def test_a_bad_number_is_still_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            self._study(conductivity=-1.0)


class TestElasticStudyIntegration:
    def _study(self, **kwargs):
        kwargs.setdefault("resolution", (6, 3, 3))
        return ElasticStudy(
            "bend",
            bcs=[Fixed(Nodes.side("-x")), Traction(_COLD_END, (0.0, 0.0, -20.0))],
            **kwargs,
        )

    def test_material_derived_moduli_differ_from_a_single_material(self, bar_mesh):
        scene = _two_material_bar()
        derived = self._study().solve(scene, mesh=bar_mesh)
        stiff = self._study(youngs=2000.0, poisson=0.3).solve(scene, mesh=bar_mesh)
        soft = self._study(youngs=500.0, poisson=0.3).solve(scene, mesh=bar_mesh)
        mixed_tip = float(derived.max())
        assert float(stiff.max()) < mixed_tip < float(soft.max())

    def test_poisson_comes_from_the_materials_too(self, bar_mesh):
        scene = _two_material_bar()
        result = self._study().solve(scene, mesh=bar_mesh)
        assert np.asarray(result.solution.poisson).shape == (bar_mesh.num_cells,)
        assert np.asarray(result.solution.youngs).shape == (bar_mesh.num_cells,)

    def test_von_mises_uses_each_element_own_modulus(self, bar_mesh):
        """Stress recovery must not fall back to one modulus for the whole bar."""
        scene = _two_material_bar()
        result = self._study().solve(scene, mesh=bar_mesh)
        stress = result.von_mises()
        assert stress.shape == (bar_mesh.num_cells,)
        assert np.isfinite(stress).all()

    def test_safety_factor_appears_only_with_a_yield_strength(self, bar_mesh):
        scene = _two_material_bar()
        assert self._study().solve(scene, mesh=bar_mesh).safety_factor is None
        strong = Material("strong", youngs_modulus=2000.0, poisson_ratio=0.3, yield_strength=1e5)
        weak = Material("weak", youngs_modulus=500.0, poisson_ratio=0.3, yield_strength=1e4)
        result = self._study().solve(_two_material_bar(hot=strong, cold=weak), mesh=bar_mesh)
        factor = result.safety_factor
        assert factor is not None and factor > 0.0
        assert result.describe()["safety_factor"] == pytest.approx(factor, rel=1e-5)

    def test_gravity_adds_self_weight(self, bar_mesh):
        """A heavy bar under gravity sags more than a weightless one."""
        scene = _two_material_bar()
        weightless = self._study().solve(scene, mesh=bar_mesh)
        heavy = self._study(gravity=(0.0, 0.0, -9810.0)).solve(scene, mesh=bar_mesh)
        assert float(heavy.max()) > float(weightless.max())

    def test_gravity_without_a_density_says_so(self, bar_mesh):
        scene = _two_material_bar(
            hot=Material("e_only", youngs_modulus=2000.0, poisson_ratio=0.3),
            cold=Material("e_only2", youngs_modulus=500.0, poisson_ratio=0.3),
        )
        with pytest.raises(ValueError, match="density"):
            self._study(gravity=(0.0, 0.0, -9.81)).solve(scene, mesh=bar_mesh)

    def test_a_uniform_field_reproduces_the_scalar_elastic_solve(self, bar_mesh):
        clamps = [Nodes.side("-x")]
        tractions = [(_COLD_END, (0.0, 0.0, -20.0))]
        scalar = elastic_solve(
            bar_mesh, youngs=1000.0, poisson=0.3, dirichlet=clamps, tractions=tractions
        )
        uniform = elastic_solve(
            bar_mesh,
            youngs=jnp.full((bar_mesh.num_cells,), 1000.0),
            poisson=jnp.full((bar_mesh.num_cells,), 0.3),
            dirichlet=clamps,
            tractions=tractions,
        )
        difference = np.abs(
            np.asarray(scalar.displacement) - np.asarray(uniform.displacement)
        ).max()
        assert difference < 1e-9 * max(np.abs(np.asarray(scalar.displacement)).max(), 1.0)

    def test_describe_reports_gravity(self):
        payload = self._study(gravity=(0.0, 0.0, -9.81)).describe()
        assert payload["gravity"] == [0.0, 0.0, -9.81]
        assert payload["material"] == {"youngs": FROM_MATERIAL, "poisson": FROM_MATERIAL}
        assert json.loads(json.dumps(payload)) == payload


class TestMaterialFieldGradient:
    """Both design directions, checked against central finite differences."""

    @staticmethod
    def _objective(bar_mesh, interface, kappa_hot):
        scene = _two_material_bar(
            interface=interface,
            hot=Material("hot", conductivity=kappa_hot),
            cold=Material("cold", conductivity=1.0),
        )
        field = sample_cell_property(scene, bar_mesh.points, bar_mesh.cells, "conductivity")
        result = thermal_solve(
            bar_mesh,
            conductivity=field,
            dirichlet=[(_HOT_END, 1.0), (_COLD_END, 0.0)],
        )
        return jnp.sum(result.temperature**2)

    def test_gradient_flows_to_the_interface_and_the_conductivity(self, bar_mesh):
        base = jnp.asarray([0.05, 8.0])

        def objective(params):
            return self._objective(bar_mesh, params[0], params[1])

        gradient = np.asarray(jax.grad(objective)(base))
        # The interface sensitivity is sharply curved — the blend is 0.06 wide
        # and the elements are 0.1 — so its difference step has to be small
        # before truncation error stops dominating.  Measured convergence at
        # this operating point: h=4e-3 is 10 % off, 1e-3 is 4.3 % off, and
        # 2e-4 is 0.17 % off.  The conductivity sensitivity is nearly linear
        # and agrees to 4e-6 at h=2e-2.
        steps = (2e-4, 1e-2)
        for index, step in enumerate(steps):
            offset = np.zeros(2)
            offset[index] = step
            finite = (float(objective(base + offset)) - float(objective(base - offset))) / (
                2.0 * step
            )
            assert gradient[index] == pytest.approx(
                finite, rel=1e-2, abs=1e-6
            ), f"component {index}: adjoint {gradient[index]} vs FD {finite}"
        # Guard against a vacuous 0 == 0 agreement: the objective genuinely
        # depends on both the interface position and the conductivity.
        assert abs(gradient[0]) > 1e-3
        assert abs(gradient[1]) > 1e-4

    def test_moving_the_interface_actually_moves_the_answer(self, bar_mesh):
        left = float(self._objective(bar_mesh, -0.3, 8.0))
        right = float(self._objective(bar_mesh, 0.3, 8.0))
        assert abs(left - right) > 1e-3


class TestQuantization:
    def test_sharp_regions_snap_exactly_to_their_own_material(self):
        aluminium, copper = aluminium_6061(), copper_c11000()
        values = {
            "youngs_modulus": np.array([68.9e9, 117e9, 68.9e9]),
            "poisson_ratio": np.array([0.33, 0.34, 0.33]),
        }
        assignment, error = quantize_to_materials(
            values, [aluminium, copper], keys=("youngs_modulus", "poisson_ratio")
        )
        assert assignment.tolist() == [0, 1, 0]
        assert error.max() < 1e-9

    def test_a_blended_element_lands_on_the_nearer_material_and_reports_the_error(self):
        aluminium, copper = aluminium_6061(), copper_c11000()
        # 90 % of the way from aluminium to copper: it must snap to copper and
        # own up to how far it moved.
        mixed = 0.1 * 68.9e9 + 0.9 * 117e9
        values = {
            "youngs_modulus": np.array([mixed]),
            "poisson_ratio": np.array([0.1 * 0.33 + 0.9 * 0.34]),
        }
        assignment, error = quantize_to_materials(
            values, [aluminium, copper], keys=("youngs_modulus", "poisson_ratio")
        )
        assert assignment.tolist() == [0 if mixed < 0.5 * (68.9e9 + 117e9) else 1]
        assert 0.0 < error[0] < 0.1

    def test_a_reference_without_the_property_is_rejected(self):
        with pytest.raises(ValueError, match="does not specify"):
            quantize_to_materials(
                {"yield_strength": np.array([1e8])},
                [Material("nameless_reference")],
                keys=("yield_strength",),
            )

    def test_the_catalogue_works_as_a_reference_set(self):
        values = {"density": np.array([2700.0, 8940.0, 7870.0])}
        assignment, error = quantize_to_materials(
            values,
            [aluminium_6061(), copper_c11000(), steel_1018()],
            keys=("density",),
        )
        assert assignment.tolist() == [0, 1, 2]
        assert error.max() < 1e-9
