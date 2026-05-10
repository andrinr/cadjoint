"""Infinite horizontal plane primitive."""

from __future__ import annotations

from jax import Array

from jaxcad.geometry.parameters import Scalar
from jaxcad.sdf.primitives.base import Primitive


class Plane(Primitive):
    """Infinite horizontal plane at y = height.

    SDF: p.y - height  (positive above the plane, negative below).
    Normal always points +Y.

    Args:
        height: Y-coordinate of the plane (float or Scalar parameter).
        material: Surface material.

    Example:
        ground = Plane(height=-1.0, material=Material(color=[0.4, 0.4, 0.4]))
    """

    def __init__(self, height: float | Scalar = 0.0, material=None):
        from jaxcad.render.material import Material

        self.material = material if material is not None else Material()
        self.params = {"height": height}

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, height: Array) -> Array:
        """Pure SDF for an infinite horizontal plane.

        Args:
            p: Point to evaluate, shape (3,).
            height: Y-coordinate of the plane.

        Returns:
            Signed distance (positive above the plane).
        """
        return p[..., 1] - height

    def __call__(self, p: Array) -> Array:
        return Plane.sdf(p, self.params["height"].value)

    def to_functional(self):
        return Plane.sdf
