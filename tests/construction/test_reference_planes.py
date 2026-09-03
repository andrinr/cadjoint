"""Work planes derived from references, and the gradients that flow through them.

``SketchPlane.on(body.cap("+"))`` is not a coordinate — it is an expression
over the parent feature's parameters. The load-bearing test here is the last
one: the volume of a solid extruded from a derived plane, differentiated with
respect to the *parent's* depth, agreeing with finite differences. That is the
thing a B-rep modeller cannot do, because there the face is stored geometry
rather than a function of the feature that made it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude
from cadjoint.geometry import Scalar
from cadjoint.sdf import volume
from cadjoint.sdf.primitives import Sphere

SQUARE = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
SMALL = [[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]]


def axes(plane: SketchPlane) -> tuple[list[float], list[float], list[float]]:
    return tuple([float(x) for x in axis] for axis in plane.frame())


# ── The old constructor is untouched ─────────────────────────────────────────


class TestPlainConstructor:
    def test_the_world_xy_plane_is_still_the_identity_frame(self):
        u, v, n = axes(SketchPlane())
        assert u == pytest.approx([1.0, 0.0, 0.0])
        assert v == pytest.approx([0.0, 1.0, 0.0])
        assert n == pytest.approx([0.0, 0.0, 1.0])
        assert SketchPlane().is_identity() is True

    def test_a_plus_y_normal_still_derives_u_as_minus_x(self):
        u, v, _ = axes(SketchPlane(normal=[0.0, 1.0, 0.0]))
        assert u == pytest.approx([-1.0, 0.0, 0.0])
        assert v == pytest.approx([0.0, 0.0, 1.0])

    def test_a_zero_normal_is_still_refused(self):
        with pytest.raises(ValueError, match="zero-length"):
            SketchPlane(normal=[0.0, 0.0, 0.0])


class TestExplicitXAxis:
    def test_pins_the_sketchs_horizontal(self):
        plane = SketchPlane(normal=[1.0, 1.0, 0.0], x_axis=[0.0, 0.0, 1.0])
        u, v, _ = axes(plane)
        assert u == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
        assert v == pytest.approx([0.7071068, -0.7071068, 0.0], abs=1e-6)

    def test_only_the_in_plane_part_of_the_axis_is_used(self):
        # A caller passing a direction that leans out of the plane gets the
        # projection, not an error: any non-parallel direction names a
        # horizontal unambiguously.
        plane = SketchPlane(normal=[0.0, 0.0, 1.0], x_axis=[1.0, 0.0, 5.0])
        u, _, _ = axes(plane)
        assert u == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)

    def test_a_pinned_plane_is_not_the_identity_unless_it_matches(self):
        assert SketchPlane(x_axis=[0.0, 1.0, 0.0]).is_identity() is False
        assert SketchPlane(x_axis=[1.0, 0.0, 0.0]).is_identity() is True

    def test_to_world_follows_the_pinned_axis(self):
        plane = SketchPlane(origin=[0.0, 0.0, 1.0], x_axis=[0.0, 1.0, 0.0])
        world = [float(x) for x in plane.to_world(jnp.array([2.0, 0.0]))]
        assert world == pytest.approx([0.0, 2.0, 1.0], abs=1e-6)


# ── on / offset / midplane ───────────────────────────────────────────────────


class TestOn:
    def test_sits_on_the_faces_plane(self):
        body = extrude(PolygonProfile(SQUARE, name="a"), depth=0.8)
        plane = SketchPlane.on(body.cap("+"))
        assert [float(x) for x in plane.origin.xyz] == pytest.approx([0.0, 0.0, 0.4], abs=1e-6)
        assert [float(x) for x in plane.normal.xyz] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)

    def test_inherits_the_faces_x_axis(self):
        body = extrude(PolygonProfile(SQUARE, name="a"), depth=0.8)
        wall = body.side(1)  # from (1,-1) to (1,1): the +X wall, swept along +Y
        u, _, n = axes(SketchPlane.on(wall))
        assert u == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
        assert n == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)

    def test_an_override_wins_over_the_faces_axis(self):
        body = extrude(PolygonProfile(SQUARE, name="a"), depth=0.8)
        u, _, _ = axes(SketchPlane.on(body.cap("+"), x_axis=[0.0, 1.0, 0.0]))
        assert u == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)

    def test_flip_turns_the_plane_over_in_place(self):
        body = extrude(PolygonProfile(SQUARE, name="a"), depth=0.8)
        plane = SketchPlane.on(body.cap("+"), flip=True)
        assert [float(x) for x in plane.origin.xyz] == pytest.approx([0.0, 0.0, 0.4], abs=1e-6)
        assert [float(x) for x in plane.normal.xyz] == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)

    def test_records_the_face_it_came_from(self):
        body = extrude(PolygonProfile(SQUARE, name="a"), depth=0.8)
        face = body.cap("+")
        assert SketchPlane.on(face).reference is face

    def test_works_on_a_primitive_face(self):
        block = Solid.box(size=[0.5, 0.5, 0.5], position=[0.0, 0.0, 1.0], name="block")
        plane = SketchPlane.on(block.face("+z"))
        assert [float(x) for x in plane.origin.xyz] == pytest.approx([0.0, 0.0, 1.5], abs=1e-6)


class TestOffset:
    def test_pushes_a_face_along_its_normal(self):
        body = extrude(PolygonProfile(SQUARE, name="a"), depth=0.8)
        plane = SketchPlane.offset(body.cap("+"), 0.25)
        assert [float(x) for x in plane.origin.xyz] == pytest.approx([0.0, 0.0, 0.65], abs=1e-6)

    def test_accepts_a_scalar_parameter_as_the_distance(self):
        body = extrude(PolygonProfile(SQUARE, name="a"), depth=0.8)
        gap = Scalar(0.5, free=True, name="gap")
        plane = SketchPlane.offset(body.cap("+"), gap)
        assert float(plane.origin.xyz[2]) == pytest.approx(0.9, abs=1e-6)

    def test_offsets_a_plane_as_readily_as_a_face(self):
        plane = SketchPlane.offset(SketchPlane(origin=[0.0, 0.0, 1.0]), -0.5)
        assert [float(x) for x in plane.origin.xyz] == pytest.approx([0.0, 0.0, 0.5], abs=1e-6)

    def test_refuses_something_that_is_neither(self):
        with pytest.raises(TypeError, match="Face or a SketchPlane"):
            SketchPlane.offset(object(), 1.0)


class TestMidplane:
    def test_lands_halfway_between_two_opposing_faces(self):
        body = extrude(PolygonProfile(SQUARE, name="a"), depth=0.8)
        plane = SketchPlane.midplane(body.cap("+"), body.cap("-"))
        assert [float(x) for x in plane.origin.xyz] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        # The two caps point away from each other; the midplane takes the
        # first one's side rather than cancelling to nothing.
        assert [float(x) for x in plane.normal.xyz] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)

    def test_bisects_two_parallel_walls_of_different_solids(self):
        left = Solid.box(size=[0.5, 0.5, 0.5], position=[-2.0, 0.0, 0.0], name="left")
        right = Solid.box(size=[0.5, 0.5, 0.5], position=[2.0, 0.0, 0.0], name="right")
        plane = SketchPlane.midplane(left.face("+x"), right.face("-x"))
        assert [float(x) for x in plane.origin.xyz] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        assert [float(x) for x in plane.normal.xyz] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


class TestTangent:
    def test_lands_on_the_surface_with_the_fields_normal(self):
        plane = SketchPlane.tangent(Sphere(radius=1.0), near=[0.9, 0.0, 0.0])
        assert [float(x) for x in plane.origin.xyz] == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)
        assert [float(x) for x in plane.normal.xyz] == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)

    def test_projects_a_point_that_is_off_the_surface(self):
        plane = SketchPlane.tangent(Sphere(radius=1.0), near=[0.0, 0.0, 0.7])
        assert [float(x) for x in plane.origin.xyz] == pytest.approx([0.0, 0.0, 1.0], abs=1e-5)

    def test_picks_the_world_axis_most_nearly_in_plane(self):
        # Normal +X, so the sketch's horizontal must not be world X.
        u, _, _ = axes(SketchPlane.tangent(Sphere(radius=1.0), near=[0.9, 0.0, 0.0]))
        assert abs(u[0]) == pytest.approx(0.0, abs=1e-5)

    def test_an_override_wins_over_the_derived_axis(self):
        plane = SketchPlane.tangent(
            Sphere(radius=1.0), near=[0.9, 0.0, 0.0], x_axis=[0.0, 0.0, 1.0]
        )
        u, _, _ = axes(plane)
        assert u == pytest.approx([0.0, 0.0, 1.0], abs=1e-5)

    def test_the_projected_point_is_differentiable(self):
        def touch(radius):
            return SketchPlane.tangent(Sphere(radius=radius), near=[0.9, 0.05, 0.0]).origin.xyz[0]

        gradient = float(jax.grad(touch)(1.0))
        # The point rides out along the (fixed) direction of `near`, so the
        # x-coordinate grows as that direction's x component.
        direction = jnp.array([0.9, 0.05, 0.0]) / jnp.linalg.norm(jnp.array([0.9, 0.05, 0.0]))
        assert gradient == pytest.approx(float(direction[0]), abs=1e-3)

    def test_the_normal_points_out_of_a_faceted_wall(self):
        """A CSG field changes branch on its own surface; `jax.grad` there lies.

        A polygon extrusion's field is a `maximum` of a 2D polygon distance
        and a cap distance, and on the wall the first of those is exactly
        zero. Autodiff is free to return any subgradient of that, and it
        returns one pointing *into* the solid with length 0.94 — which used
        to flip the sketch frame of any pad placed on a cylinder wall.
        """
        import math

        profile = PolygonProfile.circle(radius=1.0, segments=28, name="wall")
        wall = extrude(profile, depth=0.8)
        angle = math.radians(337.5)  # between two vertices, on a facet
        near = [math.cos(angle), math.sin(angle), 0.1]
        plane = SketchPlane.tangent(wall, near=near)

        normal = jnp.asarray(plane.normal.xyz)
        assert float(jnp.linalg.norm(normal)) == pytest.approx(1.0, abs=1e-5)
        outward = jnp.asarray([math.cos(angle), math.sin(angle), 0.0])
        assert float(normal @ outward) > math.cos(math.pi / 28)

        # The projection lands on the facet, between apothem and circumradius.
        origin = jnp.asarray(plane.origin.xyz)
        assert math.cos(math.pi / 28) <= float(jnp.linalg.norm(origin[:2])) <= 1.0 + 1e-6

    def test_a_pad_on_a_faceted_wall_straddles_it(self):
        """The end the flipped normal broke: a boss half in, half out of a wall."""
        import math

        profile = PolygonProfile.circle(radius=1.0, segments=28, name="wall2")
        wall = extrude(profile, depth=0.8)
        angle = math.radians(337.5)
        plane = SketchPlane.tangent(wall, near=[math.cos(angle), math.sin(angle), 0.1])
        pad = extrude(PolygonProfile(SMALL, plane=plane, name="wallpad"), depth=0.2)
        origin = jnp.asarray(plane.origin.xyz)
        normal = jnp.asarray(plane.normal.xyz)
        assert float(pad(origin + 0.08 * normal)) < 0.0  # proud of the wall
        assert float(pad(origin - 0.08 * normal)) < 0.0  # and sunk into it
        assert float(pad(origin + 0.20 * normal)) > 0.0

    def test_a_tangent_plane_carries_a_sketch(self):
        plane = SketchPlane.tangent(Sphere(radius=1.0), near=[0.0, 0.0, 0.9])
        pad = extrude(PolygonProfile(SMALL, plane=plane, name="pad"), depth=0.2)
        # The pad straddles the sphere's north pole.
        assert float(pad(jnp.array([0.0, 0.0, 1.0]))) < 0.0
        assert float(pad(jnp.array([0.0, 0.0, 1.5]))) > 0.0


# ── Tracing and gradients ────────────────────────────────────────────────────


def stacked(depth, resolution: int = 40):
    """A boss extruded from the top cap of a base whose depth is `depth`."""
    base = extrude(PolygonProfile(SQUARE, name="base"), depth=depth)
    boss = PolygonProfile(SMALL, plane=SketchPlane.on(base.cap("+")), name="boss")
    return extrude(boss, depth=0.5)


class TestTracing:
    def test_a_derived_plane_traces_under_jit(self):
        evaluate = jax.jit(lambda depth: stacked(depth)(jnp.array([0.0, 0.0, 1.0])))
        assert float(evaluate(1.0)) == pytest.approx(0.25, abs=1e-6)

    def test_the_child_rides_on_half_the_parents_depth(self):
        # The cap sits at depth/2, so a point above the child sees its distance
        # shrink at exactly half the rate the parent grows.
        derivative = jax.grad(lambda depth: stacked(depth)(jnp.array([0.0, 0.0, 1.0])))
        assert float(derivative(1.0)) == pytest.approx(-0.5, abs=1e-6)

    def test_a_derived_plane_traces_on_a_tilted_parent(self):
        plane = SketchPlane(normal=[0.0, 1.0, 0.0])

        def evaluate(depth):
            base = extrude(PolygonProfile(SQUARE, plane=plane, name="base"), depth=depth)
            boss = PolygonProfile(SMALL, plane=SketchPlane.on(base.cap("+")), name="boss")
            return extrude(boss, depth=0.5)(jnp.array([0.0, 1.5, 0.0]))

        assert float(jax.jit(evaluate)(1.0)) == pytest.approx(0.75, abs=1e-5)
        assert float(jax.grad(evaluate)(1.0)) == pytest.approx(-0.5, abs=1e-4)

    def test_volume_gradient_through_the_parent_matches_finite_differences(self):
        def measure(depth):
            return volume(
                stacked(depth), bounds=(-2, -2, -2), size=(4, 4, 4), resolution=40, epsilon=0.02
            )

        depth = 0.8
        step = 1e-3
        analytic = float(jax.grad(measure)(depth))
        difference = (float(measure(depth + step)) - float(measure(depth - step))) / (2 * step)
        assert analytic == pytest.approx(difference, rel=2e-2)
        # And the gradient is a real signal, not a numerical zero.
        assert abs(analytic) > 1e-3
