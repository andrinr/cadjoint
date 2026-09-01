"""Sphere primitive."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.geometry.parameters import Scalar
from cadjoint.sdf.primitives.base import Primitive


class Sphere(Primitive):
    """Sphere centered at origin with given radius.

    This is a thin wrapper that provides:
    - Fluent API: sphere.translate().rotate()
    - Operator overloading: sphere | box
    - Parameter storage for compilation

    The actual SDF computation is in the static sdf() method.

    Args:
        radius: Sphere radius (float or Scalar parameter)

    Example:
        # Direct creation
        sphere = Sphere(radius=1.0)

        # With parameter for optimization
        from cadjoint.geometry.parameters import Scalar
        radius = Scalar(value=1.0, free=True, name='radius')
        sphere = Sphere(radius=radius)

        # Fluent API
        shape = Sphere(1.0).translate([1, 0, 0]).scale(2.0)

        # Direct pure function call
        distance = Sphere.sdf(point, radius=1.0)
    """

    def __init__(self, radius: float | Scalar, material=None):
        from cadjoint.render.material import Material

        self.material = material if material is not None else Material()
        self.params = {"radius": radius}

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, radius: float) -> Array:
        """Pure SDF function for sphere.

        This is the source of truth for sphere SDF computation.
        Used directly during JAX compilation and tracing.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            radius: Sphere radius

        Returns:
            Signed distance: ||p|| - radius
        """
        return jnp.linalg.norm(p, axis=-1) - radius

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p.

        Delegates to the pure function with stored parameters.
        """
        return Sphere.sdf(p, self.params["radius"].value)

    def to_functional(self):
        """Return pure function for compilation.

        Returns:
            Pure function: Sphere.sdf(p, radius) -> sdf_value
        """
        return Sphere.sdf

    def patch_fields(self):
        """Single smooth patch: the sphere surface has no feature edges."""
        radius = self.params["radius"].value
        return [lambda p: Sphere.sdf(p, radius)]
