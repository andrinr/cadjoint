"""Turning a face into the tool that cuts it.

A face reference is a plane plus a boundary; what a modeller actually wants to
do with one is put a hole in it. These tests pin the geometry of the tools
:class:`~cadjoint.construction.faces.Face` builds — where they start, how far
they reach, and that they arrive already oriented to the face rather than to
the world.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude
from cadjoint.geometry import Scalar

SQUARE = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]


def _plate(depth: float = 0.4):
    return extrude(PolygonProfile(SQUARE, name="plate"), depth=Scalar(depth))


class TestFaceAnchors:
    def test_center_is_the_boundary_centroid(self):
        top = _plate().cap("+")
        np.testing.assert_allclose(np.asarray(top.center()), [0.0, 0.0, 0.2], atol=1e-6)

    def test_point_maps_face_local_coordinates_to_world(self):
        top = _plate().cap("+")
        np.testing.assert_allclose(np.asarray(top.point((0.2, 0.1))), [0.2, 0.1, 0.2], atol=1e-6)

    def test_point_uses_the_faces_own_frame_not_the_world(self):
        """On a side wall, face-local x runs along the swept edge."""
        wall = _plate().side(0)
        moved = np.asarray(wall.point((0.25, 0.0))) - np.asarray(wall.origin)
        np.testing.assert_allclose(moved, 0.25 * np.asarray(wall.x_axis), atol=1e-6)

    def test_plane_is_the_face_plane(self):
        plate = _plate()
        plane = plate.cap("+").plane()
        np.testing.assert_allclose(np.asarray(plane.origin.xyz), [0.0, 0.0, 0.2], atol=1e-6)
        np.testing.assert_allclose(np.asarray(plane.normal.xyz), [0.0, 0.0, 1.0], atol=1e-6)

    def test_plane_offset_pushes_along_the_normal(self):
        plane = _plate().cap("+").plane(offset=0.15)
        np.testing.assert_allclose(np.asarray(plane.origin.xyz), [0.0, 0.0, 0.35], atol=1e-6)

    def test_plane_flip_turns_it_over(self):
        plane = _plate().cap("+").plane(flip=True)
        np.testing.assert_allclose(np.asarray(plane.normal.xyz), [0.0, 0.0, -1.0], atol=1e-6)

    def test_a_zero_offset_is_the_plain_plane(self):
        plane = _plate().cap("+").plane(offset=0.0)
        np.testing.assert_allclose(np.asarray(plane.origin.xyz), [0.0, 0.0, 0.2], atol=1e-6)


class TestHole:
    def test_the_tool_spans_from_the_face_inward(self):
        tool = _plate().cap("+").hole(0.12, depth=0.3)
        # Inside the bore, just under the face.
        assert float(tool(jnp.array([0.0, 0.0, 0.15]))) < 0.0
        # Below the tool's reach (face at 0.2, depth 0.3 -> stops at -0.1).
        assert float(tool(jnp.array([0.0, 0.0, -0.25]))) > 0.0
        # Outside it radially.
        assert float(tool(jnp.array([0.3, 0.0, 0.15]))) > 0.0

    def test_through_extends_the_tool_above_the_face(self):
        flush = _plate().cap("+").hole(0.12, depth=0.3)
        proud = _plate().cap("+").hole(0.12, depth=0.3, through=0.1)
        above = jnp.array([0.0, 0.0, 0.25])
        assert float(flush(above)) > 0.0
        assert float(proud(above)) < 0.0

    def test_at_places_the_axis_in_face_coordinates(self):
        tool = _plate().cap("+").hole(0.08, depth=0.3, at=(0.3, -0.2))
        assert float(tool(jnp.array([0.3, -0.2, 0.1]))) < 0.0
        assert float(tool(jnp.array([0.0, 0.0, 0.1]))) > 0.0

    def test_the_radius_stays_a_live_parameter(self):
        """A Face.hole shares its Scalar, which is why a bore is differentiable."""
        from cadjoint import extract_parameters, functionalize

        radius = Scalar(0.12, free=True, name="bore_radius")
        plate = _plate()
        body = plate - plate.cap("+").hole(radius, depth=0.5, through=0.05)
        free, fixed, _ = extract_parameters(body)
        assert "bore_radius" in free

        import jax

        probe = jnp.array([0.16, 0.0, 0.0])

        def field(r):
            return functionalize(body)({**free, "bore_radius": r}, fixed)(probe)

        gradient = float(jax.grad(field)(jnp.asarray(0.12)))
        step = 1e-3
        finite = (float(field(0.12 + step)) - float(field(0.12 - step))) / (2 * step)
        np.testing.assert_allclose(gradient, finite, rtol=1e-3)

    def test_a_hole_on_a_side_wall_runs_along_that_walls_normal(self):
        wall = _plate().side(0)
        tool = wall.hole(0.06, depth=0.2)
        inward = np.asarray(wall.origin) - 0.1 * np.asarray(wall.normal)
        assert float(tool(jnp.asarray(inward, dtype=jnp.float32))) < 0.0


class TestPocket:
    def test_the_pocket_takes_the_face_frame(self):
        tool = (
            _plate()
            .cap("+")
            .pocket([[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]], depth=0.15)
        )
        assert float(tool(jnp.array([0.0, 0.0, 0.15]))) < 0.0
        assert float(tool(jnp.array([0.3, 0.0, 0.15]))) > 0.0
        assert float(tool(jnp.array([0.0, 0.0, -0.1]))) > 0.0

    def test_subtracting_a_pocket_leaves_the_rest_of_the_plate(self):
        plate = _plate()
        pocketed = plate - plate.cap("+").pocket(
            [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]], depth=0.15, through=0.02
        )
        assert float(pocketed(jnp.array([0.0, 0.0, 0.15]))) > 0.0  # cut away
        assert float(pocketed(jnp.array([0.4, 0.4, 0.0]))) < 0.0  # still there


class TestFaceToolsOnADerivedStack:
    def test_a_tool_follows_a_three_deep_face_chain(self):
        base = _plate(depth=0.4)
        mid_profile = PolygonProfile.circle(
            radius=0.3, plane=base.cap("+").plane(offset=0.15), name="mid"
        )
        mid = extrude(mid_profile, depth=Scalar(0.3))
        top_profile = PolygonProfile.circle(
            radius=0.2, plane=mid.cap("+").plane(offset=0.1), name="top"
        )
        top = extrude(top_profile, depth=Scalar(0.2))
        # 0.4 plate -> cap at 0.2; mid spans [0.2, 0.5]; top spans [0.5, 0.7].
        np.testing.assert_allclose(np.asarray(top.cap("+").origin), [0.0, 0.0, 0.7], atol=1e-5)
        bore = top.cap("+").hole(0.08, depth=1.0)
        assert float(bore(jnp.array([0.0, 0.0, 0.5]))) < 0.0


class TestGeneratedOutlines:
    def test_circle_has_the_requested_segment_count(self):
        assert len(PolygonProfile.circle(radius=0.5, segments=24, name="c").vertices) == 24

    def test_circle_vertices_sit_on_the_circle(self):
        profile = PolygonProfile.circle(radius=0.5, center=(0.1, -0.2), segments=24, name="c")
        radii = np.linalg.norm(np.asarray(profile.vertex_array()) - [0.1, -0.2], axis=1)
        np.testing.assert_allclose(radii, 0.5, atol=1e-6)

    def test_generated_vertices_are_pinned_by_default(self):
        profile = PolygonProfile.circle(radius=0.5, segments=8, name="c")
        assert not any(vertex.free for vertex in profile.vertices)

    def test_generated_vertices_can_be_freed_explicitly(self):
        profile = PolygonProfile.circle(radius=0.5, segments=8, name="c", free=True)
        assert all(vertex.free for vertex in profile.vertices)

    def test_hand_written_profiles_stay_free(self):
        """The CAD default is unchanged: a typed vertex is editable."""
        assert all(vertex.free for vertex in PolygonProfile(SQUARE, name="s").vertices)

    def test_regular_start_angle_places_the_first_vertex(self):
        profile = PolygonProfile.regular(6, 1.0, start_angle=90.0, name="h")
        np.testing.assert_allclose(np.asarray(profile.vertex_array())[0], [0.0, 1.0], atol=1e-6)

    def test_regular_needs_three_sides(self):
        with pytest.raises(ValueError, match="at least 3 sides"):
            PolygonProfile.regular(2, 1.0)

    def test_rounded_rect_vertex_count(self):
        profile = PolygonProfile.rounded_rect(2.0, 1.0, 0.2, segments=4, name="r")
        assert len(profile.vertices) == 4 * 5

    def test_rounded_rect_stays_inside_its_bounding_box(self):
        vertices = np.asarray(
            PolygonProfile.rounded_rect(2.0, 1.0, 0.2, segments=6, name="r").vertex_array()
        )
        assert vertices[:, 0].min() == pytest.approx(-1.0, abs=1e-6)
        assert vertices[:, 0].max() == pytest.approx(1.0, abs=1e-6)
        assert vertices[:, 1].min() == pytest.approx(-0.5, abs=1e-6)
        assert vertices[:, 1].max() == pytest.approx(0.5, abs=1e-6)

    def test_a_zero_radius_degenerates_to_a_plain_rectangle(self):
        """Repeated arc points would be zero-length edges, i.e. a NaN distance."""
        profile = PolygonProfile.rounded_rect(2.0, 1.0, 0.0, segments=6, name="r")
        vertices = np.asarray(profile.vertex_array())
        assert len(vertices) == 4
        assert not np.isnan(np.asarray(extrude(profile, depth=0.2)(jnp.zeros(3))))

    def test_rounded_radius_is_clamped_to_the_short_side(self):
        vertices = np.asarray(
            PolygonProfile.rounded_rect(1.0, 1.0, 5.0, segments=4, name="r").vertex_array()
        )
        assert np.abs(vertices).max() == pytest.approx(0.5, abs=1e-6)

    def test_a_non_positive_size_is_refused(self):
        with pytest.raises(ValueError, match="positive size"):
            PolygonProfile.rounded_rect(0.0, 1.0, 0.1)

    def test_generated_outlines_extrude_to_solids(self):
        disc = extrude(PolygonProfile.circle(radius=0.5, segments=32, name="c"), depth=0.2)
        assert float(disc(jnp.zeros(3))) < 0.0
        assert float(disc(jnp.array([0.7, 0.0, 0.0]))) > 0.0

    def test_a_generated_outline_accepts_a_plane(self):
        plate = _plate()
        profile = PolygonProfile.circle(
            radius=0.2, plane=plate.cap("+").plane(offset=0.1), segments=16, name="c"
        )
        boss = extrude(profile, depth=0.2)
        np.testing.assert_allclose(np.asarray(boss.cap("-").origin), [0.0, 0.0, 0.2], atol=1e-6)

    def test_a_scalar_radius_is_read_but_does_not_stay_live(self):
        """Documented limitation: a generated vertex is a number, not an expression."""
        from cadjoint import extract_parameters

        radius = Scalar(0.5, free=True, name="disc_radius")
        disc = extrude(PolygonProfile.circle(radius=radius, segments=12, name="c"), depth=0.2)
        free, _, _ = extract_parameters(disc)
        assert "disc_radius" not in free


class TestMirrorAcrossAMidplane:
    def test_midplane_of_a_features_two_caps(self):
        plate = _plate(depth=0.4)
        seam = SketchPlane.midplane(plate.cap("-"), plate.cap("+"))
        np.testing.assert_allclose(np.asarray(seam.origin.xyz), [0.0, 0.0, 0.0], atol=1e-6)

    def test_a_boss_mirrors_to_the_other_side_of_the_plate(self):
        from cadjoint.sdf.operations import Mirror

        plate = _plate(depth=0.4)
        boss = extrude(
            PolygonProfile.circle(
                radius=0.1, plane=plate.cap("+").plane(offset=0.1), segments=12, name="d"
            ),
            depth=0.2,
        )
        seam = SketchPlane.midplane(plate.cap("-"), plate.cap("+"))
        under = Mirror(boss, seam)
        # The boss spans z in [0.2, 0.4]; reflected across z = 0 it spans [-0.4, -0.2].
        assert float(boss(jnp.array([0.0, 0.0, 0.3]))) < 0.0
        assert float(under(jnp.array([0.0, 0.0, -0.3]))) < 0.0
        assert float(under(jnp.array([0.0, 0.0, 0.3]))) > 0.0


def test_solid_primitives_still_carry_their_own_faces():
    """Nothing here disturbed the primitive face path."""
    block = Solid.box(size=[0.5, 0.5, 0.25], position=[0.0, 0.0, 0.0], name="block")
    np.testing.assert_allclose(np.asarray(block.face("+z").origin), [0.0, 0.0, 0.25], atol=1e-6)
