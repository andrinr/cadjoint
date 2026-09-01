"""Box primitive."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.geometry.parameters import Vector
from cadjoint.sdf.primitives.base import Primitive


class Box(Primitive):
    """Axis-aligned box centered at origin.

    Args:
        size: Half-extents in each dimension (x, y, z) - Vector parameter

    Example:
        box = Box(size=Vector([1, 2, 3], free=True, name='size'))
    """

    def __init__(self, size: Vector, material=None):
        from cadjoint.render.material import Material

        self.material = material if material is not None else Material()
        self.params = {"size": size}

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, size: Array) -> Array:
        """Pure SDF function for box.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            size: Half-extents [x, y, z]

        Returns:
            Signed distance to box

        Note:
            Branches on inside/outside instead of smoothing the outside norm
            with an epsilon: points exactly on a face then differentiate
            through the inside branch and receive a valid one-sided
            subgradient.  An epsilon-regularized ``sqrt`` has an exact
            stationary point on the face plane, which silently zeroes
            parameter gradients for on-surface samples.
        """
        q = jnp.abs(p) - size
        max_q = jnp.max(q, axis=-1)
        clamped = jnp.maximum(q, 0.0)
        squared = jnp.sum(clamped * clamped, axis=-1)
        on_or_inside = max_q <= 0.0
        safe = jnp.where(on_or_inside, 1.0, squared)
        return jnp.where(on_or_inside, max_q, jnp.sqrt(safe))

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return Box.sdf(p, self.params["size"].xyz)

    def to_functional(self):
        """Return pure function for compilation."""
        return Box.sdf

    def patch_fields(self):
        """Six face half-space distances, derived from the sdf structure.

        :meth:`sdf` is (inside) ``max_a(|p_a| - size_a)``, and each term
        splits as ``|p_a| - size_a = max(p_a - size_a, -p_a - size_a)`` — so
        the box is the max composition of six half-space distances, one per
        face.  Order: ``[+x, -x, +y, -y, +z, -z]`` (index ``2*axis + side``
        with side 0 for the positive face).
        """
        size = self.params["size"].xyz

        def face(axis: int, sign: float):
            bound = size[axis]
            return lambda p: sign * p[..., axis] - bound

        return [face(axis, sign) for axis in range(3) for sign in (1.0, -1.0)]
