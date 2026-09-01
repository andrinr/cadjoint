"""Capsule primitive."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.geometry.parameters import Scalar
from cadjoint.sdf.primitives.base import Primitive


class Capsule(Primitive):
    """Capsule (cylinder with spherical caps) along Z-axis.

    Args:
        radius: Capsule radius (float or Scalar)
        height: Half-height of cylindrical section (float or Scalar)
    """

    def __init__(self, radius: float | Scalar, height: float | Scalar, material=None):
        from cadjoint.render.material import Material

        self.material = material if material is not None else Material()
        self.params = {"radius": radius, "height": height}

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, radius: float, height: float) -> Array:
        """Pure SDF function for capsule.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            radius: Capsule radius
            height: Half-height of cylindrical section

        Returns:
            Signed distance to capsule surface
        """
        # Clamp z coordinate to cylindrical section
        z_clamped = jnp.clip(p[..., 2], -height, height)
        # Distance to line segment along z-axis
        q = jnp.stack([p[..., 0], p[..., 1], p[..., 2] - z_clamped], axis=-1)
        return jnp.linalg.norm(q, axis=-1) - radius

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return Capsule.sdf(p, self.params["radius"].value, self.params["height"].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Capsule.sdf

    def patch_fields(self):
        """Single smooth patch: the capsule surface is C1 (no feature edges)."""
        radius = self.params["radius"].value
        height = self.params["height"].value
        return [lambda p: Capsule.sdf(p, radius, height)]
