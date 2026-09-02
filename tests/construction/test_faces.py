"""Analytic faces declared by construction features.

The point of these tests is that a face is read off the *construction tree* —
the plane a feature swept, the box's own half extents — and never off the dual
contouring mesh. So they assert the exact analytic values a feature must know:
a cap at ``±depth/2``, a side wall on the plane its edge swept, a box face at
its half extent, and containment that agrees with the polygon that bounds it.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from cadjoint.construction import (
    Axis,
    Face,
    PolygonProfile,
    SketchPlane,
    Solid,
    extrude,
    loft,
    revolve,
)
from cadjoint.geometry import Scalar

SQUARE = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
SMALL = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]


def profile(vertices=SQUARE, plane=None, name="p") -> PolygonProfile:
    return PolygonProfile(vertices, plane=plane, name=name)


# ── Extrusion caps and walls ─────────────────────────────────────────────────


class TestExtrusionFaces:
    def test_declares_two_caps_and_one_wall_per_edge(self):
        solid = extrude(profile(), depth=0.8)
        assert solid.faces.keys() == ["cap+", "cap-", "side0", "side1", "side2", "side3"]
        assert solid.faces.owner_kind == "extrude"

    def test_caps_sit_at_half_the_depth_either_side_of_the_sketch_plane(self):
        solid = extrude(profile(), depth=0.8)
        assert solid.cap("+").origin == pytest.approx([0.0, 0.0, 0.4], abs=1e-6)
        assert solid.cap("-").origin == pytest.approx([0.0, 0.0, -0.4], abs=1e-6)

    def test_cap_normals_point_out_of_the_solid(self):
        solid = extrude(profile(), depth=0.8)
        assert solid.cap("+").normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
        assert solid.cap("-").normal == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)

    def test_cap_x_axis_is_the_profiles_own_u(self):
        # A +Y plane derives u = -X, and the cap must inherit that rather than
        # re-deriving a "horizontal" of its own.
        plane = SketchPlane(normal=[0.0, 1.0, 0.0])
        solid = extrude(profile(plane=plane), depth=1.0)
        u, _, _ = plane.frame()
        assert solid.cap("+").x_axis == pytest.approx([float(x) for x in u], abs=1e-6)

    def test_cap_moves_with_a_scalar_depth_parameter(self):
        depth = Scalar(0.8, free=True, name="depth")
        solid = extrude(profile(), depth=depth)
        assert float(solid.cap("+").origin[2]) == pytest.approx(0.4, abs=1e-6)
        depth.value = jnp.asarray(2.0)
        # The face is an expression, not a snapshot: rebuilt, it has moved.
        rebuilt = extrude(profile(), depth=depth)
        assert float(rebuilt.cap("+").origin[2]) == pytest.approx(1.0, abs=1e-6)

    def test_side_walls_are_normal_to_the_edge_they_swept(self):
        solid = extrude(profile(), depth=0.8)
        wall = solid.side(0)  # from (-1,-1) to (1,-1): the -Y wall
        assert wall.normal == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)
        assert wall.origin == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)

    def test_side_wall_x_axis_is_the_swept_edge_direction(self):
        solid = extrude(profile(), depth=0.8)
        assert solid.side(0).x_axis == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)

    def test_side_normals_point_outward_for_either_winding(self):
        clockwise = list(reversed(SQUARE))
        for vertices in (SQUARE, clockwise):
            solid = extrude(profile(vertices, name="w"), depth=0.5)
            for index in range(4):
                wall = solid.side(index)
                outward = wall.origin - jnp.array([0.0, 0.0, 0.0])
                assert float(jnp.dot(wall.normal, outward)) > 0.0

    def test_a_drafted_or_twisted_extrusion_declares_no_faces(self):
        # Draft tapers the walls off their swept planes and twist curves them:
        # nothing analytic would be exact, so nothing is offered.
        assert len(extrude(profile(), depth=0.5, draft=5.0).faces) == 0
        assert len(extrude(profile(), depth=0.5, twist=15.0).faces) == 0

    def test_faces_are_only_built_when_asked_for(self):
        solid = extrude(profile(), depth=0.5)
        assert solid.faces._faces is None
        solid.cap("+")
        assert solid.faces._faces is not None


# ── Face frame and extent ────────────────────────────────────────────────────


class TestFaceExtent:
    def test_contains_a_point_on_the_face(self):
        cap = extrude(profile(), depth=0.8).cap("+")
        assert bool(cap.contains(jnp.array([0.3, -0.4, 0.4])))

    def test_rejects_a_point_off_the_plane(self):
        cap = extrude(profile(), depth=0.8).cap("+")
        assert not bool(cap.contains(jnp.array([0.0, 0.0, 0.6])))

    def test_rejects_a_point_beyond_the_boundary(self):
        cap = extrude(profile(), depth=0.8).cap("+")
        assert not bool(cap.contains(jnp.array([3.0, 0.0, 0.4])))

    def test_accepts_a_point_on_the_boundary_within_tolerance(self):
        cap = extrude(profile(), depth=0.8).cap("+")
        assert bool(cap.contains(jnp.array([1.0, 0.0, 0.4]), tol=1e-4))

    def test_a_side_wall_bounds_the_sweep_in_depth(self):
        wall = extrude(profile(), depth=0.8).side(0)
        assert bool(wall.contains(jnp.array([0.2, -1.0, 0.35])))
        assert not bool(wall.contains(jnp.array([0.2, -1.0, 0.9])))

    def test_contains_is_vectorized_over_points(self):
        cap = extrude(profile(), depth=0.8).cap("+")
        hits = cap.contains(jnp.array([[0.0, 0.0, 0.4], [4.0, 0.0, 0.4]]))
        assert list(hits) == [True, False]

    def test_the_frame_is_orthonormal_and_right_handed(self):
        wall = extrude(profile(plane=SketchPlane(normal=[1.0, 2.0, 3.0])), depth=0.7).side(1)
        x, y, normal = wall.frame()
        assert float(jnp.dot(x, y)) == pytest.approx(0.0, abs=1e-6)
        assert float(jnp.dot(x, normal)) == pytest.approx(0.0, abs=1e-6)
        cross = [float(component) for component in jnp.cross(x, y)]
        assert cross == pytest.approx([float(component) for component in normal], abs=1e-6)

    def test_tolerance_scales_with_the_face(self):
        small = extrude(profile(SMALL, name="s"), depth=0.5).cap("+")
        large = extrude(profile(name="l"), depth=0.5).cap("+")
        assert small.tolerance() < large.tolerance()

    def test_describe_carries_the_accessor_that_names_the_face(self):
        payload = extrude(profile(), depth=0.8).side(2).describe()
        assert payload["reference"] == {"call": "side", "args": [2]}
        assert payload["kind"] == "side"
        assert len(payload["polygon"]) == 4
        assert payload["tolerance"] > 0.0


# ── Accessors ────────────────────────────────────────────────────────────────


class TestFaceSetAccessors:
    def test_cap_accepts_the_words_a_user_would_type(self):
        solid = extrude(profile(), depth=0.8)
        for word in ("+", "top", 1):
            assert solid.faces.cap(word).key == "cap+"
        for word in ("-", "bottom", -1):
            assert solid.faces.cap(word).key == "cap-"

    def test_an_unknown_cap_name_is_refused(self):
        with pytest.raises(ValueError, match="named '\\+' or '-'"):
            extrude(profile(), depth=0.8).cap("sideways")

    def test_an_unknown_key_names_what_is_available(self):
        with pytest.raises(KeyError, match="cap\\+"):
            extrude(profile(), depth=0.8).face("nope")

    def test_a_missing_side_is_refused(self):
        with pytest.raises(KeyError, match="side9"):
            extrude(profile(), depth=0.8).side(9)


# ── Other features ───────────────────────────────────────────────────────────


class TestLoftFaces:
    def test_declares_both_planar_ends(self):
        solid = loft(profile(name="a"), profile(SMALL, name="b"), height=2.0)
        assert solid.faces.keys() == ["cap+", "cap-"]
        assert solid.cap("+").origin == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
        assert solid.cap("-").origin == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)

    def test_each_end_carries_its_own_profile(self):
        solid = loft(profile(name="a"), profile(SMALL, name="b"), height=2.0)
        assert bool(solid.cap("-").contains(jnp.array([0.9, 0.0, -1.0])))
        # The top end is the smaller profile, so the same x is outside it.
        assert not bool(solid.cap("+").contains(jnp.array([0.9, 0.0, 1.0])))


class TestRevolveAxis:
    def test_exposes_its_axis_of_revolution(self):
        section = [[1.0, -0.2], [1.4, -0.2], [1.4, 0.2], [1.0, 0.2]]
        ring = revolve(profile(section, name="r"))
        assert isinstance(ring.axis, Axis)
        # The sketch plane's local Y is the revolution axis; on the world XY
        # plane that is +Y.
        assert ring.axis.direction == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
        assert ring.axis.point(2.0) == pytest.approx([0.0, 2.0, 0.0], abs=1e-6)

    def test_declares_no_planar_faces(self):
        section = [[1.0, -0.2], [1.4, -0.2], [1.4, 0.2], [1.0, 0.2]]
        assert len(revolve(profile(section, name="r")).faces) == 0

    def test_describe_is_serializable(self):
        section = [[1.0, -0.2], [1.4, -0.2], [1.4, 0.2], [1.0, 0.2]]
        payload = revolve(profile(section, name="r")).axis.describe()
        assert payload["kind"] == "axis"
        assert payload["origin"] == [0.0, 0.0, 0.0]


class TestPrimitiveFaces:
    def test_a_box_declares_six(self):
        block = Solid.box(size=[0.5, 0.25, 1.0], position=[1.0, 0.0, 0.0], name="block")
        assert block.faces.keys() == ["+x", "-x", "+y", "-y", "+z", "-z"]
        assert block.face("+x").origin == pytest.approx([1.5, 0.0, 0.0], abs=1e-6)
        assert block.face("-z").origin == pytest.approx([1.0, 0.0, -1.0], abs=1e-6)
        assert block.face("-z").normal == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)

    def test_a_box_face_is_bounded_by_its_half_extents(self):
        block = Solid.box(size=[0.5, 0.25, 1.0], position=[0.0, 0.0, 0.0], name="b")
        top = block.face("+z")
        assert bool(top.contains(jnp.array([0.4, 0.2, 1.0])))
        assert not bool(top.contains(jnp.array([0.4, 0.4, 1.0])))

    def test_box_faces_follow_the_primitives_rotation(self):
        turned = Solid.box(
            size=[1.0, 1.0, 1.0],
            position=[0.0, 0.0, 0.0],
            rotation=[0.0, 0.0, jnp.pi / 2],
            name="t",
        )
        assert turned.face("+x").origin == pytest.approx([0.0, 1.0, 0.0], abs=1e-5)

    def test_a_cylinder_declares_its_two_caps(self):
        pin = Solid.cylinder(radius=0.4, height=0.6, position=[0.0, 0.0, 0.0], name="pin")
        assert pin.faces.keys() == ["cap+", "cap-"]
        assert pin.cap("+").origin == pytest.approx([0.0, 0.0, 0.6], abs=1e-6)
        assert bool(pin.cap("+").contains(jnp.array([0.3, 0.0, 0.6])))
        assert not bool(pin.cap("+").contains(jnp.array([0.5, 0.0, 0.6])))

    def test_a_sphere_declares_none(self):
        ball = Solid.sphere(radius=0.5, position=[0.0, 0.0, 0.0], name="ball")
        assert len(ball.faces) == 0


# ── Registration on the sketch ───────────────────────────────────────────────


class TestFeatureRegistration:
    def test_a_generator_records_its_feature_on_the_profile(self):
        sketch = profile()
        solid = extrude(sketch, depth=0.5)
        assert [feature.kind for feature in sketch.features] == ["extrude"]
        assert sketch.features[0].faces is solid.faces

    def test_a_loft_records_itself_on_its_base_profile(self):
        base = profile(name="a")
        loft(base, profile(SMALL, name="b"), height=1.0)
        assert [feature.kind for feature in base.features] == ["loft"]

    def test_a_face_is_an_instance_of_the_public_type(self):
        assert isinstance(extrude(profile(), depth=0.5).cap("+"), Face)
