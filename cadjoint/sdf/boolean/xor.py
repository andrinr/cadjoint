"""XOR boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.sdf.boolean.base import BooleanOp


class Xor(BooleanOp):
    """XOR of two SDFs (symmetric difference).

    Args:
        sdfs: 2-tuple of SDFs
    """

    def __init__(self, *sdfs):
        if len(sdfs) == 1 and isinstance(sdfs[0], (tuple, list)):
            sdfs = tuple(sdfs[0])
        self.sdfs = sdfs
        self.params = {}

    @staticmethod
    def sdf(child_sdfs, p: Array) -> Array:
        """Pure function for XOR operation.

        Args:
            child_sdfs: 2-tuple of SDF functions
            p: Query point(s)

        Returns:
            XOR SDF value
        """
        d1 = child_sdfs[0](p)
        d2 = child_sdfs[1](p)
        return jnp.maximum(jnp.minimum(d1, d2), -jnp.maximum(d1, d2))

    def __call__(self, p: Array) -> Array:
        """XOR: max(min(d1, d2), -max(d1, d2))"""
        return Xor.sdf(self.sdfs, p)

    def material_at(self, p: Array) -> dict:
        from cadjoint.render.material import Material

        m1, m2 = self.sdfs[0].material_at(p), self.sdfs[1].material_at(p)
        return Material.blend(m1, m2, jnp.array(0.5))

    def to_functional(self):
        """Return pure function for compilation."""
        return Xor.sdf
