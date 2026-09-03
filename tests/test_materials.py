"""Physical material properties: the container, the catalogue, and mass.

These tests pin the contract the FEM layer relies on — that a property the
scene never stated stays *unstated* rather than defaulting to something
plausible, that stating one puts it through the same Parameter machinery as a
color (so it can be free and traced), and that the catalogue's numbers are
self-consistent engineering values rather than placeholders.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.geometry.parameters import Scalar
from cadjoint.materials import CATALOGUE, aluminium_6061, aluminum_6061, catalogue, copper_c11000
from cadjoint.render.material import (
    OPTICAL_PROPERTIES,
    PHYSICAL_PROPERTIES,
    UNITS,
    Material,
)
from cadjoint.sdf import Box, material_mass, volume


class TestPhysicalProperties:
    def test_unspecified_properties_read_as_none(self):
        """A material that says nothing about physics reports None, not a default."""
        material = Material(color=[1.0, 0.0, 0.0])
        for key in PHYSICAL_PROPERTIES:
            assert material.get(key) is None
        # …while every optical property still has its usual value.
        assert material.get("roughness") == pytest.approx(0.5)

    def test_unspecified_properties_are_nan_in_the_array_view(self):
        """The pytree keeps one static structure: unstated means NaN, not absent."""
        values = Material().as_dict()
        assert set(values) == set(OPTICAL_PROPERTIES) | set(PHYSICAL_PROPERTIES)
        for key in PHYSICAL_PROPERTIES:
            assert bool(np.isnan(np.asarray(values[key])))

    def test_specified_properties_round_trip(self):
        material = Material(
            "slug",
            density=8940.0,
            conductivity=391.0,
            specific_heat=385.0,
            youngs_modulus=117e9,
            poisson_ratio=0.34,
            thermal_expansion=17e-6,
            yield_strength=69e6,
        )
        assert material.get("density") == pytest.approx(8940.0)
        assert material.get("youngs_modulus") == pytest.approx(117e9, rel=1e-6)
        assert material.get("poisson_ratio") == pytest.approx(0.34, rel=1e-6)

    def test_properties_are_parameters(self):
        """Physical properties use the same containers as the optical ones."""
        material = Material(density=2700.0)
        assert isinstance(material.params["density"], Scalar)
        assert material.params["density"].fixed

    def test_free_marks_only_stated_properties(self):
        """A free NaN would be meaningless, so unstated properties stay fixed."""
        material = Material("mat", conductivity=167.0, free=True)
        assert material.params["conductivity"].free
        assert material.params["conductivity"].name == "mat_conductivity"
        assert material.params["conductivity"].bounds == (1e-3, 3000.0)
        assert not material.params["density"].free
        assert material.params["roughness"].free  # optical ones still go free

    def test_free_requires_a_name(self):
        with pytest.raises(ValueError, match="requires a name"):
            Material(density=2700.0, free=True)

    def test_a_property_can_carry_a_traced_value(self):
        """Free properties must survive tracing for the optimizer to tune them."""

        def total(kappa):
            return jnp.asarray(Material(conductivity=kappa).as_dict()["conductivity"]) * 3.0

        assert float(jax.grad(total)(jnp.asarray(2.0))) == pytest.approx(3.0)


class TestBlending:
    def test_physical_properties_blend_like_optical_ones(self):
        """The smooth CSG interface is a smooth *property* interface."""
        left = Material(density=1000.0, conductivity=10.0).as_dict()
        right = Material(density=3000.0, conductivity=2.0).as_dict()
        middle = Material.blend(left, right, jnp.asarray(0.25))
        assert float(middle["density"]) == pytest.approx(0.25 * 1000.0 + 0.75 * 3000.0)
        assert float(middle["conductivity"]) == pytest.approx(0.25 * 10.0 + 0.75 * 2.0)

    def test_blending_with_an_unstated_property_stays_unstated(self):
        """NaN in, NaN out — the sampling layer turns that into a clear error."""
        stated = Material(density=1000.0).as_dict()
        silent = Material().as_dict()
        blended = Material.blend(stated, silent, jnp.asarray(0.5))
        assert bool(np.isnan(np.asarray(blended["density"])))


class TestDescribePayload:
    def test_payload_is_json_ready_and_complete(self):
        material = Material("al", color=[0.8, 0.8, 0.82], density=2700.0, conductivity=167.0)
        payload = material.describe()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["color"] == pytest.approx([0.8, 0.8, 0.82], rel=1e-6)
        for key in OPTICAL_PROPERTIES:
            if key != "color":
                assert key in payload
        assert payload["physical"]["density"] == pytest.approx(2700.0)
        assert payload["physical"]["yield_strength"] is None
        assert payload["units"] == UNITS
        assert payload["free"]["density"] is False

    def test_payload_flags_free_properties(self):
        payload = Material("mat", conductivity=1.0, free=True).describe()
        assert payload["free"]["conductivity"] is True
        assert payload["free"]["density"] is False


class TestCatalogue:
    def test_every_entry_states_every_physical_property(self):
        """The catalogue exists so a scene never has to invent a number."""
        for name, factory in CATALOGUE.items():
            material = factory()
            for key in PHYSICAL_PROPERTIES:
                assert material.get(key) is not None, f"{name} is missing {key}"

    def test_values_are_physically_sane(self):
        for name, factory in CATALOGUE.items():
            material = factory()
            assert material.get("density") > 0.0, name
            assert material.get("conductivity") > 0.0, name
            assert material.get("specific_heat") > 0.0, name
            assert material.get("youngs_modulus") > 0.0, name
            assert 0.0 <= material.get("poisson_ratio") < 0.5, name
            assert material.get("thermal_expansion") >= 0.0, name
            assert material.get("yield_strength") > 0.0, name

    def test_the_documented_numbers_are_the_shipped_numbers(self):
        """Guards against a silent edit to a cited value."""
        aluminium = aluminium_6061()
        assert aluminium.get("density") == pytest.approx(2700.0)
        assert aluminium.get("conductivity") == pytest.approx(167.0)
        assert aluminium.get("youngs_modulus") == pytest.approx(68.9e9, rel=1e-6)
        assert aluminium.get("poisson_ratio") == pytest.approx(0.33, rel=1e-6)
        assert aluminium.get("yield_strength") == pytest.approx(276e6, rel=1e-6)
        copper = copper_c11000()
        assert copper.get("density") == pytest.approx(8940.0)
        assert copper.get("conductivity") == pytest.approx(391.0)
        # Copper conducts heat well over twice as well as 6061 — the whole
        # reason a design puts a copper slug into an aluminium sink.
        assert copper.get("conductivity") > 2.0 * aluminium.get("conductivity")

    def test_factories_hand_out_independent_instances(self):
        """Marking one scene's material free must not leak into the next scene."""
        first, second = aluminium_6061(), aluminium_6061()
        assert first is not second
        assert first.params["density"] is not second.params["density"]

    def test_us_spelling_is_an_alias(self):
        assert aluminum_6061 is aluminium_6061

    def test_catalogue_helper_returns_fresh_materials(self):
        materials = catalogue()
        assert len(materials) == len(CATALOGUE)
        assert {material.name for material in materials} == set(CATALOGUE)


class TestMaterialMass:
    def test_single_material_mass_is_density_times_volume(self):
        box = Box([0.5, 0.5, 0.5], material=aluminium_6061())
        measured_volume = float(volume(box, resolution=40))
        measured_mass = float(material_mass(box, resolution=40))
        assert measured_mass == pytest.approx(2700.0 * measured_volume, rel=1e-4)

    def test_mass_reports_the_material_that_is_actually_there(self):
        """Swapping aluminium for copper changes the mass, not the volume."""
        shape = [0.5, 0.5, 0.5]
        light = float(material_mass(Box(shape, material=aluminium_6061()), resolution=30))
        heavy = float(material_mass(Box(shape, material=copper_c11000()), resolution=30))
        assert heavy == pytest.approx(light * 8940.0 / 2700.0, rel=1e-6)

    def test_mass_is_differentiable_in_the_shape(self):
        """The whole point: mass can regularize a shape optimization."""

        def mass_of(half_extent):
            box = Box(jnp.stack([half_extent, half_extent, half_extent]), material=copper_c11000())
            return material_mass(box, resolution=24)

        gradient = float(jax.grad(mass_of)(jnp.asarray(0.5)))
        step = 5e-3
        finite = (float(mass_of(0.5 + step)) - float(mass_of(0.5 - step))) / (2.0 * step)
        assert gradient == pytest.approx(finite, rel=5e-2)
        assert gradient > 0.0

    def test_explicit_sample_points_match_the_lattice_form(self):
        """The form a scene's own regularizer already has: points + cell volume."""
        box = Box([0.5, 0.5, 0.5], material=copper_c11000())
        axis = jnp.linspace(-1.5, 1.5, 30)
        grid = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        cell_volume = float((3.0 / 30) ** 3)
        explicit = float(material_mass(box, grid, cell_volume, epsilon=0.02))
        lattice = float(
            material_mass(
                box, bounds=(-1.5, -1.5, -1.5), size=(3, 3, 3), resolution=30, epsilon=0.02
            )
        )
        assert explicit == pytest.approx(lattice, rel=0.05)

    def test_sample_points_without_a_cell_volume_say_so(self):
        with pytest.raises(ValueError, match="cell_volume"):
            material_mass(Box([0.5, 0.5, 0.5], material=copper_c11000()), jnp.zeros((4, 3)))

    def test_missing_density_names_the_material(self):
        with pytest.raises(ValueError, match="plain"):
            material_mass(Box([0.5, 0.5, 0.5], material=Material("plain")))
