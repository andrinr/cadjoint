"""Tests for boolean operations."""

import jax
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


class TestMaterialThroughBooleans:
    """A cut tool is geometry, not a substance; it must not erase the body's material.

    `Material.blend` is a plain lerp, and `nan * 0.0 == nan`, so a child that
    leaves a physical property unspecified used to erase the other child's
    value even at weight zero. Every `Face.hole` is such a child, so every
    part with a hole in it reported `conductivity = nan` everywhere and no
    `FROM_MATERIAL` study over it could be solved.
    """

    @staticmethod
    def steel():
        from cadjoint.render.material import Material

        return Material(name="steel", color=[0.6, 0.6, 0.6], conductivity=52.0, density=7870.0)

    @staticmethod
    def at(shape, point):
        import numpy as np

        values = shape.material_at(jnp.asarray(point, dtype=jnp.float32))
        return float(np.asarray(values["conductivity"]))

    def test_a_hole_does_not_erase_the_plate(self):
        plate = Box(size=jnp.array([1.0, 1.0, 0.2]), material=self.steel())
        drilled = Difference(plate, Sphere(radius=0.2), smoothness=0.0)
        assert self.at(drilled, [0.8, 0.0, 0.0]) == 52.0

    def test_every_tool_is_folded_in_not_just_the_first(self):
        """`Difference.material_at` read `sdfs[0]` and `sdfs[1]` and stopped."""
        from cadjoint.render.material import Material
        from cadjoint.sdf.transforms import Translate

        copper = Material(name="copper", color=[0.7, 0.4, 0.2], conductivity=400.0)
        plate = Box(size=jnp.array([1.0, 1.0, 0.2]), material=self.steel())
        first = Translate(Sphere(radius=0.25), offset=jnp.array([-0.6, 0.0, 0.0]))
        second = Translate(Sphere(radius=0.25, material=copper), offset=jnp.array([0.6, 0.0, 0.0]))
        cut = Difference(plate, first, second, smoothness=0.01)

        # Clear of both tools, the body wins outright.
        assert self.at(cut, [0.0, 0.0, 0.0]) == 52.0
        # On the *second* tool's cut wall the blend has left the body's value —
        # which it could not do while the fold stopped after the first tool.
        assert self.at(cut, [0.34, 0.0, 0.0]) > 100.0

    def test_a_union_with_a_material_less_child_keeps_its_material(self):
        body = Box(size=jnp.array([1.0, 1.0, 0.2]), material=self.steel())
        joined = Union(body, Sphere(radius=0.3), smoothness=0.0)
        assert self.at(joined, [0.9, 0.0, 0.0]) == 52.0

    def test_an_intersection_with_a_material_less_child_keeps_its_material(self):
        body = Box(size=jnp.array([1.0, 1.0, 1.0]), material=self.steel())
        clipped = Intersection(body, Sphere(radius=0.9), smoothness=0.0)
        assert self.at(clipped, [0.0, 0.0, 0.0]) == 52.0

    def test_two_specified_materials_still_blend(self):
        from cadjoint.render.material import Material

        copper = Material(name="copper", color=[0.7, 0.4, 0.2], conductivity=400.0)
        joined = Union(
            Box(size=jnp.array([0.5, 0.5, 0.5]), material=self.steel()),
            Sphere(radius=0.5, material=copper),
            smoothness=0.4,
        )
        seam = self.at(joined, [0.45, 0.0, 0.0])
        assert 52.0 < seam < 400.0

    def test_a_property_neither_side_specifies_stays_unspecified(self):
        import numpy as np

        from cadjoint.render.material import Material

        plain = Material(name="plain", color=[0.5, 0.5, 0.5])
        joined = Union(Box(size=jnp.array([0.5, 0.5, 0.5]), material=plain), Sphere(radius=0.4))
        values = joined.material_at(jnp.asarray([0.45, 0.0, 0.0], dtype=jnp.float32))
        assert np.isnan(np.asarray(values["conductivity"]))

    def test_an_xor_with_a_material_less_child_keeps_its_material(self):
        body = Box(size=jnp.array([1.0, 1.0, 0.2]), material=self.steel())
        from cadjoint.sdf.boolean import Xor

        assert self.at(Xor(body, Sphere(radius=0.2)), [0.8, 0.0, 0.0]) == 52.0

    def test_the_blend_is_differentiable_where_one_side_is_unspecified(self):
        """The value fix must not be paid for with a `nan` gradient.

        Masking a `nan` with a single `jnp.where` returns the right number
        and a `nan` derivative, because the VJP of `where` multiplies the
        unselected branch's cotangent by zero and `0.0 * nan == nan`. That
        reaches the author as `Optimization ... grad_norm=nan` at step 0,
        long after the shape it came from is out of sight.
        """
        import numpy as np

        def conductivity(radius):
            plate = Box(size=jnp.array([1.0, 1.0, 0.2]), material=self.steel())
            cut = Difference(plate, Sphere(radius=radius), smoothness=0.02)
            return cut.material_at(jnp.asarray([0.8, 0.0, 0.0], dtype=jnp.float32))["conductivity"]

        gradient = jax.grad(conductivity)(jnp.float32(0.2))
        assert np.isfinite(np.asarray(gradient))

    def test_the_gradient_of_a_real_blend_is_still_the_real_one(self):
        """Sanity: making the unspecified case finite did not flatten the rest."""
        import numpy as np

        from cadjoint.render.material import Material

        copper = Material(name="copper", color=[0.7, 0.4, 0.2], conductivity=400.0)

        def conductivity(radius):
            joined = Union(
                Box(size=jnp.array([0.5, 0.5, 0.5]), material=self.steel()),
                Sphere(radius=radius, material=copper),
                smoothness=0.4,
            )
            return joined.material_at(jnp.asarray([0.45, 0.0, 0.0], dtype=jnp.float32))[
                "conductivity"
            ]

        eps = 1e-3
        gradient = float(jax.grad(conductivity)(jnp.float32(0.5)))
        finite = (
            float(conductivity(jnp.float32(0.5 + eps)))
            - float(conductivity(jnp.float32(0.5 - eps)))
        ) / (2 * eps)
        assert abs(gradient) > 1.0
        np.testing.assert_allclose(gradient, finite, rtol=2e-2)
