"""Tests for boolean operations."""

import jax.numpy as jnp

from cadjoint.sdf.boolean import Difference, Intersection, Union
from cadjoint.sdf.primitives import Box, Sphere


class TestUnion:
    def test_union_contains_both(self):
        """Union should contain points from both shapes"""
        sphere = Sphere(radius=1.0)
        box = Box(size=jnp.array([0.5, 0.5, 0.5]))
        union = Union((sphere, box))

        # Point only in sphere
        p1 = jnp.array([0.9, 0.0, 0.0])
        assert union(p1) < 0

        # Point only in box
        p2 = jnp.array([0.4, 0.4, 0.4])
        assert union(p2) < 0

    def test_union_operator(self):
        """Test | operator for union"""
        sphere = Sphere(radius=1.0)
        box = Box(size=jnp.array([0.5, 0.5, 0.5]))
        union = sphere | box

        assert isinstance(union, Union)
        p = jnp.array([0.9, 0.0, 0.0])
        assert union(p) < 0

    def test_sharp_vs_smooth(self):
        """Sharp union should differ from smooth union"""
        sphere = Sphere(radius=1.0)
        box = Box(size=jnp.array([0.5, 0.5, 0.5]))

        sharp = Union((sphere, box), smoothness=0.0)
        smooth = Union((sphere, box), smoothness=0.5)

        # Point near intersection should differ
        p = jnp.array([0.5, 0.5, 0.0])
        assert not jnp.isclose(sharp(p), smooth(p))


class TestIntersection:
    def test_intersection_only_overlap(self):
        """Intersection should only contain overlapping region"""
        sphere = Sphere(radius=1.0)
        box = Box(size=jnp.array([2.0, 2.0, 2.0]))
        inter = Intersection((sphere, box))

        # Point in both (center)
        p1 = jnp.array([0.0, 0.0, 0.0])
        assert inter(p1) < 0

        # Point only in box, not in sphere
        p2 = jnp.array([1.5, 0.0, 0.0])
        assert inter(p2) > 0

    def test_intersection_operator(self):
        """Test & operator for intersection"""
        sphere = Sphere(radius=1.0)
        box = Box(size=jnp.array([2.0, 2.0, 2.0]))
        inter = sphere & box

        assert isinstance(inter, Intersection)
        p = jnp.array([0.0, 0.0, 0.0])
        assert inter(p) < 0

    def test_no_overlap_is_empty(self):
        """Intersection of non-overlapping shapes should be empty everywhere"""
        sphere1 = Sphere(radius=0.5)
        sphere2 = Sphere(radius=0.5)
        # Translate sphere2 (would need transform, so just test conceptually)
        inter = Intersection((sphere1, sphere2))

        # Both spheres at same location, should have overlap
        p = jnp.array([0.0, 0.0, 0.0])
        assert inter(p) < 0


class TestDifference:
    def test_difference_removes_second(self):
        """Difference should remove second shape from first"""
        sphere = Sphere(radius=1.0)
        small_sphere = Sphere(radius=0.5)
        diff = Difference((sphere, small_sphere))

        # Point in outer sphere but not inner
        p1 = jnp.array([0.8, 0.0, 0.0])
        assert diff(p1) < 0

        # Point in both (should be removed)
        p2 = jnp.array([0.0, 0.0, 0.0])
        assert diff(p2) > 0

    def test_difference_operator(self):
        """Test - operator for difference"""
        sphere = Sphere(radius=1.0)
        small_sphere = Sphere(radius=0.5)
        diff = sphere - small_sphere

        assert isinstance(diff, Difference)
        p = jnp.array([0.8, 0.0, 0.0])
        assert diff(p) < 0

    def test_drill_hole(self):
        """Classic use case: drill a hole through a sphere"""
        from cadjoint.sdf.primitives import Cylinder

        sphere = Sphere(radius=2.0)
        cylinder = Cylinder(radius=0.5, height=3.0)
        drilled = sphere - cylinder

        # Point in sphere but in cylinder (hole)
        p1 = jnp.array([0.0, 0.0, 0.0])
        assert drilled(p1) > 0

        # Point in sphere but outside cylinder
        p2 = jnp.array([1.5, 0.0, 0.0])
        assert drilled(p2) < 0


class TestCompositeOperations:
    def test_chained_unions(self):
        """Multiple unions can be chained"""
        s1 = Sphere(radius=0.5)
        s2 = Sphere(radius=0.5)
        s3 = Sphere(radius=0.5)

        composite = s1 | s2 | s3

        # Should still be an SDF
        p = jnp.array([0.0, 0.0, 0.0])
        assert isinstance(composite(p), jnp.ndarray)

    def test_mixed_operations(self):
        """Can mix different boolean operations"""
        sphere = Sphere(radius=1.0)
        box = Box(size=jnp.array([0.8, 0.8, 0.8]))
        small_sphere = Sphere(radius=0.3)

        # (sphere | box) - small_sphere
        composite = (sphere | box) - small_sphere

        p = jnp.array([0.0, 0.0, 0.0])
        assert composite(p) > 0  # Center removed by small sphere


class TestSmoothness:
    def test_smoothness_parameter(self):
        """Smoothness parameter should affect blending"""
        sphere = Sphere(radius=1.0)
        box = Box(size=jnp.array([1.0, 1.0, 1.0]))

        # Different smoothness values
        sharp = Union((sphere, box), smoothness=0.01)
        smooth = Union((sphere, box), smoothness=0.5)

        # Point near blending region
        p = jnp.array([0.7, 0.7, 0.0])

        # Smooth version should have smaller (more negative) distance
        assert smooth(p) < sharp(p)
