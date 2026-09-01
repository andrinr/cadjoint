"""Intersection boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.sdf.boolean.base import BooleanOp
from cadjoint.sdf.boolean.smooth import smooth_max


class Intersection(BooleanOp):
    """Intersection of two or more SDFs (only overlapping region).

    Uses smooth maximum for differentiable blending.

    Args:
        sdfs: Tuple of SDFs to intersect
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, *sdfs, smoothness: float = 0.1):
        if len(sdfs) == 1 and isinstance(sdfs[0], (tuple, list)):
            sdfs = tuple(sdfs[0])
        self.sdfs = sdfs
        self.params = {"smoothness": smoothness}

    @staticmethod
    def sdf(child_sdfs, p: Array, smoothness: float) -> Array:
        """Pure function for intersection operation.

        Args:
            child_sdfs: Tuple of SDF functions
            p: Query point(s)
            smoothness: Blend radius

        Returns:
            Intersection SDF value
        """
        result = child_sdfs[0](p)
        for child in child_sdfs[1:]:
            d = child(p)
            result = jnp.where(
                smoothness > 0, smooth_max(result, d, smoothness), jnp.maximum(result, d)
            )
        return result

    def __call__(self, p: Array) -> Array:
        """Intersection: max over all children with smooth blending"""
        return Intersection.sdf(self.sdfs, p, self.params["smoothness"].value)

    def material_at(self, p: Array) -> dict:
        from cadjoint.render.material import Material

        k = jnp.maximum(self.params["smoothness"].value * 4.0, 1e-10)
        d1, d2 = self.sdfs[0](p), self.sdfs[1](p)
        m1, m2 = self.sdfs[0].material_at(p), self.sdfs[1].material_at(p)
        t = jnp.clip(0.5 + 0.5 * (d1 - d2) / k, 0.0, 1.0)
        return Material.blend(m1, m2, t)

    def to_functional(self):
        """Return pure function for compilation."""
        return Intersection.sdf
