"""Tests for construction primitives — the editable mirrors of SDF primitives."""

import math

import jax
import jax.numpy as jnp
import pytest

from jaxcad import apply_parameters, extract_parameters, functionalize
from jaxcad.construction import ConstructionPrimitive, Solid
from jaxcad.sdf.boolean import Union
from jaxcad.sdf.primitives import Box, Sphere


class TestGeneratedSolids:
    def test_box_matches_a_translated_primitive(self):
        solid = Solid.box(size=[1.0, 0.5, 0.25], position=[1.0, -0.5, 0.0], name="b")
        reference = Box(size=jnp.array([1.0, 0.5, 0.25]))
        points = jax.random.uniform(jax.random.PRNGKey(0), (300, 3), minval=-3, maxval=3)
        offset = jnp.array([1.0, -0.5, 0.0])
        error = jnp.max(jnp.abs(jax.vmap(solid)(points) - jax.vmap(reference)(points - offset)))
        assert float(error) < 1e-5

    def test_sphere_radius_and_position(self):
        solid = Solid.sphere(radius=0.5, position=[2.0, 0.0, 0.0], name="s")
        assert float(solid(jnp.array([2.0, 0.0, 0.0]))) == pytest.approx(-0.5, abs=1e-6)
        assert float(solid(jnp.array([2.5, 0.0, 0.0]))) == pytest.approx(0.0, abs=1e-6)

    def test_cylinder_axis_is_z(self):
        solid = Solid.cylinder(radius=0.5, height=1.0, position=[0.0, 0.0, 0.0], name="c")
        assert float(solid(jnp.array([0.0, 0.0, 1.0]))) == pytest.approx(0.0, abs=1e-6)
        assert float(solid(jnp.array([0.5, 0.0, 0.0]))) == pytest.approx(0.0, abs=1e-6)

    def test_rotation_is_applied(self):
        # A 90° turn about Z carries the box's long +X axis onto +Y.
        solid = Solid.box(
            size=[1.0, 0.2, 0.2], position=[0, 0, 0], rotation=[0, 0, math.pi / 2], name="r"
        )
        assert float(solid(jnp.array([0.0, 1.0, 0.0]))) == pytest.approx(0.0, abs=1e-5)
        assert float(solid(jnp.array([1.0, 0.0, 0.0]))) > 0.5

    def test_rotations_compose_in_xyz_order(self):
        primitive = ConstructionPrimitive(
            "box", size=[1.0, 0.2, 0.2], rotation=[math.pi / 2, math.pi / 2, 0.0], name="c"
        )
        # X then Y: +X → +X (unchanged by the X turn) → -Z.
        assert float(primitive.sdf()(jnp.array([0.0, 0.0, -1.0]))) == pytest.approx(0.0, abs=1e-5)

    def test_composes_with_other_sdfs(self):
        scene = Union(
            Solid.box(size=[0.5, 0.5, 0.5], position=[0, 0, 0], name="b"),
            Sphere(0.4),
            smoothness=0.0,
        )
        assert float(scene(jnp.array([0.0, 0.0, 0.0]))) < 0

    def test_rejects_unknown_kinds_and_missing_dimensions(self):
        with pytest.raises(ValueError, match="Unknown primitive kind"):
            ConstructionPrimitive("cone", radius=1.0)
        with pytest.raises(ValueError, match="needs radius"):
            ConstructionPrimitive("sphere")
        with pytest.raises(ValueError, match="three angles"):
            ConstructionPrimitive("sphere", radius=1.0, rotation=[0, 0])


class TestParameterSharing:
    def test_placement_is_extractable_and_free(self):
        solid = Solid.box(size=[0.5, 0.5, 0.5], position=[1.0, 0.0, 0.0], name="b")
        free, _, _ = extract_parameters(solid)
        assert {"b_position", "b_size"} <= set(free)

    def test_rotation_parameters_reach_the_tree_when_turned(self):
        solid = Solid.box(size=[0.5, 0.5, 0.5], position=[0, 0, 0], rotation=[0, 0.4, 0], name="b")
        free, _, _ = extract_parameters(solid)
        assert "b_ry" in free
        # Identity rotations are not emitted, so they cost nothing at trace time.
        assert "b_rx" not in free

    def test_apply_parameters_moves_the_solid(self):
        solid = Solid.box(size=[0.5, 0.5, 0.5], position=[0.0, 0.0, 0.0], name="b")
        free, fixed, _ = extract_parameters(solid)
        apply_parameters(solid, {**free, "b_position": jnp.array([2.0, 0.0, 0.0])})
        assert float(solid(jnp.array([2.0, 0.0, 0.0]))) == pytest.approx(-0.5, abs=1e-6)

    def test_gradients_flow_to_the_placement(self):
        primitive = ConstructionPrimitive("sphere", radius=0.5, position=[1.0, 0.0, 0.0], name="s")
        solid = primitive.sdf()
        free, fixed, _ = extract_parameters(solid)
        fn = functionalize(solid)

        def loss(params):
            return fn(params, fixed)(jnp.array([3.0, 0.0, 0.0]))

        grads = jax.grad(loss)(free)
        # Moving the sphere toward the probe shortens the distance.
        assert float(grads["s_position"][0]) < -0.5


class TestOutline:
    def test_box_has_twelve_edges_at_its_corners(self):
        primitive = ConstructionPrimitive("box", size=[1.0, 2.0, 3.0], name="b")
        edges = primitive.local_edges()
        assert len(edges) == 12
        corners = {point for edge in edges for point in edge}
        assert len(corners) == 8
        assert (1.0, 2.0, 3.0) in corners

    def test_sphere_and_cylinder_outlines_are_closed_rings(self):
        sphere = ConstructionPrimitive("sphere", radius=1.0, name="s")
        # Three great circles.
        assert len(sphere.local_edges()) % 3 == 0
        for start, end in sphere.local_edges():
            assert math.isclose(math.dist(start, (0, 0, 0)), 1.0, abs_tol=1e-6)
            assert math.isclose(math.dist(end, (0, 0, 0)), 1.0, abs_tol=1e-6)

        cylinder = ConstructionPrimitive("cylinder", radius=0.5, height=1.0, name="c")
        heights = {round(point[2], 6) for edge in cylinder.local_edges() for point in edge}
        assert heights == {-1.0, 1.0}

    def test_world_edges_follow_position_and_rotation(self):
        primitive = ConstructionPrimitive(
            "box",
            size=[1.0, 1.0, 1.0],
            position=[5.0, 0.0, 0.0],
            rotation=[0.0, 0.0, math.pi / 2],
            name="b",
        )
        for start, end in primitive.world_edges():
            for point in (start, end):
                # The unit cube's corners sit at distance sqrt(3) from its centre.
                assert math.isclose(math.dist(point, (5.0, 0.0, 0.0)), math.sqrt(3), abs_tol=1e-5)

    def test_outline_matches_the_solid_surface(self):
        """Every outline vertex should sit on the generated solid's surface."""
        primitive = ConstructionPrimitive(
            "box",
            size=[0.8, 0.4, 0.6],
            position=[1.0, -0.5, 0.25],
            rotation=[0.3, -0.2, 0.7],
            name="b",
        )
        solid = primitive.sdf()
        for start, _ in primitive.world_edges():
            assert abs(float(solid(jnp.array(start)))) < 1e-5
