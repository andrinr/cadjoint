"""Cylinder primitive."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Scalar
from jaxcad.sdf.primitives.base import Primitive


class Cylinder(Primitive):
    """Cylinder along Z-axis centered at origin.

    Args:
        radius: Cylinder radius (float or Scalar parameter)
        height: Cylinder half-height (total height = 2 * height)
    """

    def __init__(self, radius: float | Scalar, height: float | Scalar, material=None):
        from jaxcad.render.material import Material

        self.material = material if material is not None else Material()
        self.params = {"radius": radius, "height": height}

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, radius: float, height: float) -> Array:
        """Pure SDF function for cylinder.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            radius: Cylinder radius
            height: Cylinder half-height

        Returns:
            Signed distance to cylinder
        """
        d_xy = jnp.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2 + 1e-20) - radius
        d_z = jnp.abs(p[..., 2]) - height
        return jnp.sqrt(
            jnp.maximum(d_xy, 0.0) ** 2 + jnp.maximum(d_z, 0.0) ** 2 + 1e-20
        ) + jnp.minimum(jnp.maximum(d_xy, d_z), 0.0)

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return Cylinder.sdf(p, self.params["radius"].value, self.params["height"].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Cylinder.sdf
