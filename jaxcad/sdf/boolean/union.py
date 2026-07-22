"""Union boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf.boolean.base import BooleanOp
from jaxcad.sdf.boolean.smooth import smooth_min


class Union(BooleanOp):
    """Union of two or more SDFs (combines all shapes).

    Uses smooth minimum for differentiable blending at the intersection.

    Args:
        sdfs: Tuple of SDFs to union
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, *sdfs, smoothness: float = 0.1):
        if len(sdfs) == 1 and isinstance(sdfs[0], (tuple, list)):
            sdfs = tuple(sdfs[0])
        self.sdfs = self._validate_children(sdfs, "Union")
        self.params = {"smoothness": smoothness}

    @staticmethod
    def sdf(child_sdfs, p: Array, smoothness: float) -> Array:
        """Pure function for union operation.

        Args:
            child_sdfs: Tuple of SDF functions
            p: Query point(s)
            smoothness: Blend radius

        Returns:
            Union SDF value
        """
        result = child_sdfs[0](p)
        for child in child_sdfs[1:]:
            d = child(p)
            result = smooth_min(result, d, smoothness)
        return result

    def __call__(self, p: Array) -> Array:
        """Union: min over all children with smooth blending"""
        return Union.sdf(self.sdfs, p, self.params["smoothness"].value)

    def material_at(self, p: Array) -> dict:
        material_fns = tuple(sdf.material_at for sdf in self.sdfs)
        return Union.material(self.sdfs, material_fns, p, self.params["smoothness"].value)

    @staticmethod
    def material(child_sdfs, child_materials, p: Array, smoothness: float) -> dict:
        """Pure material selection matching the union distance reduction."""
        from jaxcad.render.material import Material

        k = jnp.maximum(smoothness * 4.0, 1e-10)
        result_m = child_materials[0](p)
        result_d = child_sdfs[0](p)
        for child_sdf, child_material in zip(child_sdfs[1:], child_materials[1:]):
            d = child_sdf(p)
            m = child_material(p)
            t = jnp.clip(0.5 + 0.5 * (d - result_d) / k, 0.0, 1.0)
            result_m = Material.blend(result_m, m, t)
            result_d = smooth_min(result_d, d, smoothness)
        return result_m

    def to_functional(self):
        """Return pure function for compilation."""
        return Union.sdf
