"""The end-cap scene: that it builds what it claims, and stays differentiable.

``scenes/end_cap.py`` exists to push the modelling language harder than the
starter does — a three-deep chain of sketches on generated faces, a revolve, a
loft, two kinds of pattern, a mirror across a derived midplane, and six cuts.
The point of these tests is that all of that composes into geometry whose
*measurements* are the ones the file's comments promise, and that the two named
driving parameters still carry an exact gradient out the other end.

The scene is executed once per session: it runs a constraint solve and builds a
fair-sized SDF tree, and every test here reads the same built model.
"""

from __future__ import annotations

import importlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def cap():
    return importlib.import_module("scenes.end_cap")


class TestTheFeatureStack:
    """Each feature sits on the face of the one below it, at the stated height."""

    def test_the_flange_mounting_face_is_the_z_origin(self, cap):
        np.testing.assert_allclose(
            np.asarray(cap.flange.cap("-").origin), [0.0, 0.0, 0.0], atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(cap.flange.cap("+").origin), [0.0, 0.0, 0.18], atol=1e-6
        )

    @pytest.mark.parametrize(
        ("feature", "low", "high"),
        [
            ("flange", 0.0, 0.18),
            ("boss", 0.18, 0.60),
            ("seal_land", 0.60, 0.72),
            ("retainer_pad", 0.72, 0.78),
        ],
    )
    def test_the_chain_stacks_without_gaps(self, cap, feature, low, high):
        solid = getattr(cap, feature)
        assert float(solid.cap("-").origin[2]) == pytest.approx(low, abs=1e-5)
        assert float(solid.cap("+").origin[2]) == pytest.approx(high, abs=1e-5)

    def test_the_chain_is_three_sketch_planes_deep(self, cap):
        """boss -> seal land -> retainer pad, each on the cap below it."""
        assert cap.boss_profile.plane.reference is not None
        assert cap.land_profile.plane.reference is not None
        assert cap.pad_profile.plane.reference is not None

    def test_re_dimensioning_the_flange_lifts_the_whole_stack(self, cap):
        """The derived planes are re-derived, not stored — the CAD promise."""
        from cadjoint.construction import PolygonProfile, extrude
        from cadjoint.geometry import Scalar

        def stack_top(thickness):
            base = extrude(
                PolygonProfile.rounded_rect(
                    2.0, 2.0, 0.36, segments=4, plane=cap.flange_profile.plane, name="f"
                ),
                depth=Scalar(thickness),
            )
            boss = extrude(
                PolygonProfile.circle(
                    radius=0.62, segments=8, plane=base.cap("+").plane(offset=0.21), name="b"
                ),
                depth=Scalar(0.42),
            )
            return float(boss.cap("+").origin[2])

        assert stack_top(0.30) - stack_top(0.18) == pytest.approx(0.06, abs=1e-5)


class TestTheGeometryIsWhatItClaims:
    def _inside(self, cap, point):
        return float(cap.housing(jnp.asarray(point, dtype=jnp.float32))) < 0.0

    def test_the_bore_is_open_on_the_axis(self, cap):
        assert not self._inside(cap, [0.0, 0.0, 0.05])
        assert not self._inside(cap, [0.0, 0.0, 0.65])

    def test_the_flange_is_solid_away_from_the_bore(self, cap):
        assert self._inside(cap, [0.90, 0.0, 0.09])

    def test_the_ribs_reach_the_flange_and_the_boss(self, cap):
        """A gusset that floats above its flange is the classic silent error."""
        assert self._inside(cap, [0.80, 0.0, 0.19])  # rib just above the flange top
        assert self._inside(cap, [0.70, 0.0, 0.35])  # rib climbing toward the boss

    def test_the_ribs_are_patterned_eight_ways(self, cap):
        """Every copy must be there, and the gaps between them must be open."""
        radius, height = 0.85, 0.19

        def at(angle):
            return self._inside(cap, [radius * math.cos(angle), radius * math.sin(angle), height])

        assert all(at(2 * math.pi * i / 8) for i in range(8)), "a rib copy is missing"
        # Half-pitch between the ribs is open metal-free air -- except at
        # index 0, which is the 22.5 degrees the lubrication port occupies.
        assert not any(
            at(2 * math.pi * (i + 0.5) / 8) for i in range(1, 8)
        ), "the pattern has no gaps"
        assert at(math.radians(22.5)), "the port should fill its own gap"

    def test_the_bolt_circle_is_drilled_four_times(self, cap):
        radius = float(cap.bolt_circle.value)
        for i in range(4):
            angle = math.pi / 4 + 2 * math.pi * i / 4
            point = [radius * math.cos(angle), radius * math.sin(angle), 0.09]
            assert not self._inside(cap, point), f"bolt hole {i} is not open"

    def test_the_bolts_clear_the_rib_field(self, cap):
        """The ribs sweep out to r = 0.95; the bolts must start beyond them."""
        assert float(cap.bolt_circle.value) - 0.075 > 0.95

    def test_the_dowel_is_mirrored_under_the_flange(self, cap):
        assert self._inside(cap, [0.0, 0.80, 0.24])  # the boss above
        assert self._inside(cap, [0.0, 0.80, -0.06])  # its mirror below

    def _on_port_axis(self, cap, distance):
        """A world point `distance` out along the port's own axis."""
        origin = np.asarray(cap.port_center)
        return list(origin + distance * np.asarray(cap.port_direction))

    def test_the_lubrication_port_stands_off_the_boss(self, cap):
        # Off the port axis, which is itself bored away.
        point = np.asarray(self._on_port_axis(cap, 0.18)) + np.array([0.0, 0.0, 0.18])
        assert self._inside(cap, list(point))

    def test_the_port_is_bored_through_to_the_bearing(self, cap):
        assert not self._inside(cap, self._on_port_axis(cap, 0.18))  # on the port axis
        assert not self._inside(cap, self._on_port_axis(cap, -0.20))  # still open further in

    def test_the_port_bore_misses_the_ribs(self, cap):
        """The port sits half a rib pitch round precisely so it does not."""
        for i in range(8):
            angle = 2 * math.pi * i / 8
            point = [0.70 * math.cos(angle), 0.70 * math.sin(angle), 0.40]
            assert self._inside(cap, point), f"rib {i} was drilled away at the port height"

    def test_the_seat_counterbore_is_wider_than_the_bore(self, cap):
        """The revolved seat opens the bore out to 0.40 near the mounting face."""
        assert not self._inside(cap, [0.35, 0.0, 0.10])  # inside the seat, cut away
        assert self._inside(cap, [0.55, 0.0, 0.10])  # outside it, still metal


class TestMaterials:
    def test_the_assembly_carries_four_distinct_materials(self, cap):
        colors = {
            tuple(round(float(c), 4) for c in m.params["color"].xyz)
            for m in (cap.aluminum, cap.bronze, cap.steel, cap.nitrile)
        }
        assert len(colors) == 4

    def test_the_scene_evaluates_a_material_at_a_point(self, cap):
        material = cap.scene.material_at(jnp.array([0.90, 0.0, 0.09]))
        assert len(material["color"]) == 3


class TestConstraints:
    def test_the_rib_sketch_is_satisfied(self, cap):
        heel = np.asarray(cap.rib_heel.value)
        toe = np.asarray(cap.rib_toe.value)
        crest = np.asarray(cap.rib_crest.value)
        slope = np.asarray(cap.rib_slope.value)
        assert heel[1] == pytest.approx(toe[1], abs=1e-4)  # horizontal root
        assert np.dot(toe - heel, crest - toe) == pytest.approx(0.0, abs=1e-4)  # square corner
        assert np.linalg.norm(crest - toe) == pytest.approx(
            float(cap.rib_height.value), abs=1e-3
        )  # the driving dimension
        # rib_slope lies on the heel-crest line: the 2D cross product vanishes.
        edge, offset = crest - heel, slope - heel
        assert edge[0] * offset[1] - edge[1] * offset[0] == pytest.approx(0.0, abs=1e-4)

    def test_no_parameter_came_back_nan(self, cap):
        for name, value in cap.housing_parameters.items():
            assert np.isfinite(np.asarray(value)).all(), name


class TestDifferentiability:
    """The whole point: after all of that, the part is still traceable."""

    @pytest.mark.parametrize(("name", "step"), [("flange_thickness", 2e-3), ("bore_radius", 2e-3)])
    def test_volume_gradient_matches_finite_differences(self, cap, name, step):
        base = dict(cap.housing_parameters)

        def volume(value):
            return cap.housing_volume({**base, name: value})

        start = jnp.asarray(float(base[name]))
        analytic = float(jax.grad(volume)(start))
        finite = (float(volume(start + step)) - float(volume(start - step))) / (2 * step)
        assert analytic == pytest.approx(finite, rel=2e-3)
        assert abs(analytic) > 1e-3, "the parameter must actually move the volume"

    def test_the_two_parameters_push_the_volume_opposite_ways(self, cap):
        """A thicker flange adds metal; a wider bore removes it."""
        base = dict(cap.housing_parameters)
        gradient = jax.grad(
            lambda thickness, radius: cap.housing_volume(
                {**base, "flange_thickness": thickness, "bore_radius": radius}
            ),
            argnums=(0, 1),
        )(jnp.asarray(0.18), jnp.asarray(0.30))
        assert float(gradient[0]) > 0.0
        assert float(gradient[1]) < 0.0

    def test_both_named_parameters_reach_the_compiled_model(self, cap):
        assert "flange_thickness" in cap.housing_parameters
        assert "bore_radius" in cap.housing_parameters


class TestTheSceneIsViewerReady:
    def test_the_program_assigns_scene(self, cap):
        from cadjoint.sdf.base import SDF

        assert isinstance(cap.scene, SDF)

    def test_the_simulation_domain_is_named(self, cap):
        assert cap.housing.name == "housing"
        assert cap.cap_mesh.domain is cap.housing

    def test_the_study_points_at_the_declared_mesh(self, cap):
        assert cap.cap_study.mesh is cap.cap_mesh
        assert len(cap.cap_study.bcs) == 2

    def test_every_boundary_condition_is_serializable(self, cap):
        """A Nodes.predicate would not round-trip into the viewer."""
        assert all(bc.nodes.serializable for bc in cap.cap_study.bcs)
