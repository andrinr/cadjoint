"""Finite cone primitive."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Scalar
from jaxcad.sdf.primitives.base import Primitive


class Cone(Primitive):
    """Capped cone along the Z axis, centered at the origin.

    Args:
        radius: Radius of the base at ``z = -height / 2``.
        height: Total height from the base to the apex.
        material: Optional surface material.
    """

    def __init__(self, radius: float | Scalar, height: float | Scalar, material=None):
        from jaxcad.render.material import Material

        self.material = material if material is not None else Material()
        self.params = {"radius": radius, "height": height}

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, radius: float, height: float) -> Array:
        """Return the exact signed distance to a finite capped cone.

        The apex is at ``z = height / 2`` and the circular base is at
        ``z = -height / 2``. The reduction to radial/axial coordinates is the
        capped-cone distance formulation by Inigo Quilez.
        """
        half_height = height / 2.0
        radial = jnp.linalg.norm(p[..., :2], axis=-1)
        axial = p[..., 2]

        k1 = jnp.stack([jnp.zeros_like(half_height), half_height])
        k2 = jnp.stack([-radius, height])

        cap_radius = jnp.where(axial < 0.0, radius, 0.0)
        ca = jnp.stack(
            [radial - jnp.minimum(radial, cap_radius), jnp.abs(axial) - half_height],
            axis=-1,
        )

        q = jnp.stack([radial, axial], axis=-1)
        projection = jnp.sum((k1 - q) * k2, axis=-1) / jnp.sum(k2 * k2)
        projection = jnp.clip(projection, 0.0, 1.0)
        cb = q - k1 + projection[..., None] * k2

        distance_sq = jnp.minimum(jnp.sum(ca * ca, axis=-1), jnp.sum(cb * cb, axis=-1))
        inside = (cb[..., 0] < 0.0) & (ca[..., 1] < 0.0)
        return jnp.where(inside, -1.0, 1.0) * jnp.sqrt(distance_sq)

    def __call__(self, p: Array) -> Array:
        """Evaluate the cone SDF at one or more points."""
        return Cone.sdf(p, self.params["radius"].value, self.params["height"].value)

    def to_functional(self):
        """Return the pure cone distance function."""
        return Cone.sdf
