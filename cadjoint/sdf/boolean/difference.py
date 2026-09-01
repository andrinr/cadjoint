"""Difference boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.sdf.boolean.base import BooleanOp
from cadjoint.sdf.boolean.smooth import smooth_max


class Difference(BooleanOp):
    """Difference of SDFs (subtract all subsequent from first).

    Uses smooth maximum for differentiable blending.

    Args:
        sdfs: Tuple of SDFs; first is the base, rest are subtracted
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, *sdfs, smoothness: float = 0.1):
        if len(sdfs) == 1 and isinstance(sdfs[0], (tuple, list)):
            sdfs = tuple(sdfs[0])
        self.sdfs = sdfs
        self.params = {"smoothness": smoothness}

    @staticmethod
    def sdf(child_sdfs, p: Array, smoothness: float) -> Array:
        """Pure function for difference operation.

        Args:
            child_sdfs: Tuple of SDF functions; first is base, rest are subtracted
            p: Query point(s)
            smoothness: Blend radius

        Returns:
            Difference SDF value
        """
        result = child_sdfs[0](p)
        for child in child_sdfs[1:]:
            d = child(p)
            result = jnp.where(
                smoothness > 0, smooth_max(result, -d, smoothness), jnp.maximum(result, -d)
            )
        return result

    def __call__(self, p: Array) -> Array:
        """Difference: subtract all subsequent SDFs from first"""
        return Difference.sdf(self.sdfs, p, self.params["smoothness"].value)

    def material_at(self, p: Array) -> dict:
        from cadjoint.render.material import Material

        k = jnp.maximum(self.params["smoothness"].value * 4.0, 1e-10)
        d1, d2 = self.sdfs[0](p), self.sdfs[1](p)
        m1, m2 = self.sdfs[0].material_at(p), self.sdfs[1].material_at(p)
        # Difference is max(d1, -d2): blend toward m2 where -d2 dominates (cut surface)
        t = jnp.clip(0.5 + 0.5 * (d1 + d2) / k, 0.0, 1.0)
        return Material.blend(m1, m2, t)

    def to_functional(self):
        """Return pure function for compilation."""
        return Difference.sdf
