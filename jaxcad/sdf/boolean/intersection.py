"""Intersection boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf.boolean.base import BooleanOp
from jaxcad.sdf.boolean.smooth import smooth_max


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
        self.sdfs = self._validate_children(sdfs, "Intersection")
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
        material_fns = tuple(sdf.material_at for sdf in self.sdfs)
        return Intersection.material(self.sdfs, material_fns, p, self.params["smoothness"].value)

    @staticmethod
    def material(child_sdfs, child_materials, p: Array, smoothness: float) -> dict:
        """Pure material selection matching the intersection distance reduction."""
        from jaxcad.render.material import Material

        k = jnp.maximum(smoothness * 4.0, 1e-10)
        result_m = child_materials[0](p)
        result_d = child_sdfs[0](p)
        for child_sdf, child_material in zip(child_sdfs[1:], child_materials[1:]):
            d = child_sdf(p)
            m = child_material(p)
            t = jnp.clip(0.5 + 0.5 * (result_d - d) / k, 0.0, 1.0)
            result_m = Material.blend(result_m, m, t)
            result_d = jnp.where(
                smoothness > 0,
                smooth_max(result_d, d, smoothness),
                jnp.maximum(result_d, d),
            )
        return result_m

    def to_functional(self):
        """Return pure function for compilation."""
        return Intersection.sdf
