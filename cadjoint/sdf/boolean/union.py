"""Union boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.sdf.boolean.base import BooleanOp, _operand_patch_fields
from cadjoint.sdf.boolean.smooth import smooth_min


class Union(BooleanOp):
    """Union of two or more SDFs (combines all shapes).

    Uses smooth minimum for differentiable blending at the intersection.

    Args:
        sdfs: Tuple of SDFs to union
        smoothness: Blend half-width in **model units**, not a fillet radius.
            0 is a sharp boolean; the default of 0.1 blends, so a plain
            ``Union(a, b)`` is a blended one and a caller who wants the
            sharp operation has to ask for ``smoothness=0``. The value does
            not scale with the part: 0.1 is a tenth of a unit whatever the
            unit stands for, which on a normalised engine block came to
            20 mm and removed a water jacket. See
            :mod:`cadjoint.sdf.boolean.smooth` for what the blend costs —
            it moves the field by up to ``smoothness`` per operation, and
            by the full amount wherever the operands agree.
    """

    def __init__(self, *sdfs, smoothness: float = 0.1):
        if len(sdfs) == 1 and isinstance(sdfs[0], (tuple, list)):
            sdfs = tuple(sdfs[0])
        self.sdfs = sdfs
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
        from cadjoint.sdf.boolean.base import blend_materials

        k = jnp.maximum(self.params["smoothness"].value * 4.0, 1e-10)
        result_m = self.sdfs[0].material_at(p)
        result_d = self.sdfs[0](p)
        for child in self.sdfs[1:]:
            d = child(p)
            m = child.material_at(p)
            t = jnp.clip(0.5 + 0.5 * (d - result_d) / k, 0.0, 1.0)
            result_m = blend_materials(result_m, m, t)
            result_d = smooth_min(result_d, d, self.params["smoothness"].value)
        return result_m

    def patch_fields(self):
        """Every operand's patches, operand-major, for a sharp union only.

        ``min`` over the operands means the union's surface is made of pieces
        of theirs, so the decomposition is their fields concatenated in
        operand order.  See
        :func:`~cadjoint.sdf.boolean.base._operand_patch_fields` for what the
        order guarantees and why a ``smoothness > 0`` union — whose blended
        fillet lies on no operand's zero set — declares nothing instead.

        Note that :func:`~cadjoint.meshing.patch_fields.world_frame_leaves`
        splits a scene *at* its booleans, so a top-level union never reaches
        this method; it earns its keep for a boolean nested under something
        else, such as a pattern of a two-cylinder tool.
        """
        return _operand_patch_fields(self)

    def to_functional(self):
        """Return pure function for compilation."""
        return Union.sdf
