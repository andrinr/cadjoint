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
        scale_array = scale.xyz if isinstance(scale, Vector) else jnp.asarray(scale)
        if scale_array.ndim == 0:
            scale = jnp.repeat(scale_array, 3)
            scale_array = scale
        if scale_array.shape != (3,):
            raise ValueError(
                f"Scale must be a scalar or a 3D vector, got shape {scale_array.shape}."
            )
        if not bool(jnp.isfinite(scale_array).all()) or bool(jnp.any(scale_array == 0)):
            raise ValueError("Scale components must be finite and non-zero.")
        self.params = {"scale": scale}

    @property
    def is_exact(self) -> bool:
        """Whether a fixed uniform scale preserves exact distances."""
        scale = self.params["scale"]
        return (
            self.sdf.is_exact
            and not scale.free
            and bool(jnp.allclose(jnp.abs(scale.xyz), jnp.abs(scale.xyz[0])))
        )

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

        distance_scale = jnp.where(is_uniform, jnp.abs(scale[0]), 1.0)
        return child_sdf(Scale._transform_point(p, scale)) * distance_scale

    def __call__(self, p: Array) -> Array:
        """Evaluate scaled SDF."""
        return Scale.sdf(self.sdf, p, self.params["scale"].xyz)

    def material_at(self, p: Array) -> dict:
        return self.sdf.material_at(Scale._transform_point(p, self.params["scale"].xyz))

    def to_functional(self):
        """Return pure function for compilation."""
        return Scale.sdf
