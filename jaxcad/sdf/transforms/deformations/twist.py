"""Twist deformation for SDFs."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Scalar, Vector
from jaxcad.sdf.transforms.base import Transform


class Twist(Transform):
    """Twist deformation around an arbitrary axis.

    Rotates points progressively based on their projection along the axis.
    Note: This produces an approximation, not an exact SDF.

    Args:
        sdf: The SDF to twist
        strength: Twist amount (radians per unit length along axis)
        axis: Twist axis as Array [x, y, z], Vector, or string ('x', 'y', 'z'). Defaults to 'z'.
    """

    is_exact = False  # gradient magnitude is not reliable as an AO proxy

    def __init__(self, sdf, strength: float | Scalar, axis: str | Array | Vector = "z"):
        self.sdf = sdf
        if isinstance(axis, str):
            axis_map = {
                "x": jnp.array([1.0, 0.0, 0.0]),
                "y": jnp.array([0.0, 1.0, 0.0]),
                "z": jnp.array([0.0, 0.0, 1.0]),
            }
            axis = axis_map.get(axis.lower(), jnp.array([0.0, 0.0, 1.0]))
        self.params = {"strength": strength, "axis": axis}

    @staticmethod
    def _transform_point(p: Array, strength: float, axis: Array) -> Array:
        """Apply twist to a single point (1D)."""
        height = jnp.dot(p, axis)
        angle = strength * height
        c, s = jnp.cos(angle), jnp.sin(angle)
        p_along = height * axis
        p_perp = p - p_along
        cross = jnp.cross(axis, p_perp)
        return p_along + c * p_perp + s * cross

    @staticmethod
    def sdf(child_sdf, p: Array, strength: float, axis: Array) -> Array:
        """Pure function for twist deformation around an arbitrary axis.

        Args:
            child_sdf: SDF function to twist
            p: Query point(s)
            strength: Twist strength in radians per unit length along axis
            axis: Normalized twist axis [x, y, z]

        Returns:
            Twisted SDF value (approximate)
        """
        axis = axis / jnp.linalg.norm(axis)

        if p.ndim == 1:
            p_twisted = Twist._transform_point(p, strength, axis)
        else:
            height = jnp.einsum("...i,i->...", p, axis)
            angle = strength * height
            c, s = jnp.cos(angle), jnp.sin(angle)
            p_along = height[..., None] * axis
            p_perp = p - p_along
            cross = jnp.cross(axis, p_perp)
            p_twisted = p_along + c[..., None] * p_perp + s[..., None] * cross

        return child_sdf(p_twisted)

    def __call__(self, p: Array) -> Array:
        """Evaluate twisted SDF."""
        return Twist.sdf(self.sdf, p, self.params["strength"].value, self.params["axis"].xyz)

    def material_at(self, p: Array) -> dict:
        axis = self.params["axis"].xyz / jnp.linalg.norm(self.params["axis"].xyz)
        p_twisted = Twist._transform_point(p, self.params["strength"].value, axis)
        return self.sdf.material_at(p_twisted)

    def to_functional(self):
        """Return pure function for compilation."""
        return Twist.sdf
