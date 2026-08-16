"""Scale transformation for SDFs."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Vector
from jaxcad.sdf import SDF
from jaxcad.sdf.transforms.base import Transform


class Scale(Transform):
    """Scale an SDF component-wise.

    Note: Non-uniform scaling doesn't produce exact SDFs. For uniform scaling,
    we can divide the distance by the scale factor to maintain correctness.

    Args:
        sdf: The SDF to scale
        scale: Per-axis scale as Array [sx, sy, sz], Vector parameter, or float for uniform scaling
    """

    def __init__(self, sdf: SDF, scale: float | Array | Vector):
        self.sdf = sdf
        # Convert scalar to uniform 3D scale vector before auto-cast
        if isinstance(scale, (int, float)):
            scale = jnp.array([scale, scale, scale])
        self.params = {"scale": scale}

    @staticmethod
    def _transform_point(p: Array, scale: Array) -> Array:
        return p / scale

    @staticmethod
    def sdf(child_sdf, p: Array, scale: Array) -> Array:
        """Pure function for component-wise scaling.

        Args:
            child_sdf: SDF function to scale
            p: Query point(s)
            scale: Scale vector [sx, sy, sz]

        Returns:
            Scaled SDF value
        """
        # Check if uniform by comparing all components to first
        is_uniform = jnp.allclose(scale, scale[0])

        # Use jnp.where for JAX-compatible branching
        def uniform_scale():
            return child_sdf(Scale._transform_point(p, scale)) * scale[0]

        def nonuniform_scale():
            return child_sdf(Scale._transform_point(p, scale))

        return jnp.where(is_uniform, uniform_scale(), nonuniform_scale())

    def __call__(self, p: Array) -> Array:
        """Evaluate scaled SDF."""
        return Scale.sdf(self.sdf, p, self.params["scale"].xyz)

    def material_at(self, p: Array) -> dict:
        return self.sdf.material_at(Scale._transform_point(p, self.params["scale"].xyz))

    def patch_fields(self):
        """Forward the child's patch fields through a uniform scale.

        Uniform scaling maps queries into the child frame as :meth:`sdf`
        does (``p / s``) and rescales the distances by ``s``, keeping them
        exact.  Non-uniform scaling warps each patch field differently along
        each axis — no exact decomposition — so it reports ``None``.
        """
        scale = self.params["scale"].xyz
        if not bool(jnp.allclose(scale, scale[0])):
            return None
        child_fields = self.sdf.patch_fields()
        if child_fields is None:
            return None
        factor = scale[0]
        return [
            (lambda p, f=field: f(Scale._transform_point(p, scale)) * factor)
            for field in child_fields
        ]

    def to_functional(self):
        """Return pure function for compilation."""
        return Scale.sdf
