"""Round box primitive."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.primitives.base import Primitive


class RoundBox(Primitive):
    """Box with rounded edges centered at origin.

    Args:
        size: Half-extents in each dimension (x, y, z) - Array or Vector parameter
        radius: Rounding radius (float or Scalar parameter)
    """

    def __init__(self, size: Array | Vector, radius: float | Scalar, material=None):
        from cadjoint.render.material import Material

        self.material = material if material is not None else Material()
        self.params = {"size": size, "radius": radius}

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, size: Array, radius: float) -> Array:
        """Pure SDF function for rounded box.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            size: Half-extents [x, y, z]
            radius: Rounding radius

        Returns:
            Signed distance to rounded box
        """
        q = jnp.abs(p) - size
        return (
            jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
            + jnp.minimum(jnp.max(q, axis=-1), 0.0)
            - radius
        )

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return RoundBox.sdf(p, self.params["size"].xyz, self.params["radius"].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return RoundBox.sdf

    def patch_fields(self):
        """Single smooth patch: rounding erases the box's sharp edges."""
        size = self.params["size"].xyz
        radius = self.params["radius"].value
        return [lambda p: RoundBox.sdf(p, size, radius)]
