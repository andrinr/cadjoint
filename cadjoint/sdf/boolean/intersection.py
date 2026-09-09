"""Intersection boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.sdf.boolean.base import BooleanOp, _operand_patch_fields
from cadjoint.sdf.boolean.smooth import smooth_max


class Intersection(BooleanOp):
    """Intersection of two or more SDFs (only overlapping region).

    Uses smooth maximum for differentiable blending.

    Args:
        sdfs: Tuple of SDFs to intersect
        smoothness: Blend half-width in **model units**, not a fillet radius.
            0 is a sharp boolean; the default of 0.1 blends, so a plain
            ``Intersection(a, b)`` is a blended one and a caller who wants the
            sharp operation has to ask for ``smoothness=0``. The value does
            not scale with the part: 0.1 is a tenth of a unit whatever the
            unit stands for, which on a normalised engine block came to
            20 mm and removed a water jacket. See
            :mod:`cadjoint.sdf.boolean.smooth` for what the blend costs —
            it moves the field by up to ``smoothness`` per operation, and
            by the full amount wherever the operands agree.
    """

    def __init__(self, *sdfs, smoothness: float = 0.0):
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
        """Blended across every operand, not just the first two."""
        from cadjoint.sdf.boolean.base import blend_materials

        k = jnp.maximum(self.params["smoothness"].value * 4.0, 1e-10)
        result_m = self.sdfs[0].material_at(p)
        result_d = self.sdfs[0](p)
        for child in self.sdfs[1:]:
            d = child(p)
            t = jnp.clip(0.5 + 0.5 * (result_d - d) / k, 0.0, 1.0)
            result_m = blend_materials(result_m, child.material_at(p), t)
            result_d = jnp.maximum(result_d, d)
        return result_m

    def patch_fields(self):
        """Every operand's patches, operand-major, for a sharp intersection only.

        ``max`` over the operands, so the intersection's surface is made of
        pieces of theirs and the decomposition is their fields concatenated
        in operand order.  No sign change is needed: each operand already
        reads positive outside itself, and outside the intersection is
        outside at least one of them.  See
        :func:`~cadjoint.sdf.boolean.base._operand_patch_fields` for the
        order's guarantees and the ``smoothness > 0`` refusal.

        Most operands contribute patches that bound nothing here — an
        intersection keeps only the overlap, so the rest of each operand's
        surface is interior. That is the consumer's business rather than
        this method's: it drops a patch that bounds nothing with its own
        boundary probe, and a missing patch it could not recover.
        """
        return _operand_patch_fields(self)

    def to_functional(self):
        """Return pure function for compilation."""
        return Intersection.sdf
