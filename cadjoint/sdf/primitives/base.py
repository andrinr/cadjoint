"""Base class for primitive SDFs."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.sdf import SDF


def safe_norm(v: Array, axis: int = -1) -> Array:
    """Euclidean norm along *axis* whose gradient is finite at the zero vector.

    ``jnp.linalg.norm`` has a NaN gradient where its argument is zero, and a
    NaN gradient survives a ``max``/``where`` even when its branch is not
    selected, so one primitive evaluated at such a point poisons every
    boolean around it (issue #57).  The double ``where`` keeps the values --
    bit for bit under ``jit`` -- and makes the gradient there 0, a valid
    subgradient: the same guard ``Box.sdf`` and the polygon distance use.
    """
    squared = jnp.sum(v * v, axis=axis)
    positive = squared > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


class Primitive(SDF):
    """Base class for primitive SDFs (leaf nodes in the SDF tree).

    Primitives are the basic building blocks - spheres, boxes, cylinders, etc.
    They have no SDF children, but may have a Fluent material child so that
    ``extract_parameters`` can discover material parameters automatically.

    Subclasses must implement:
    - @staticmethod def sdf(p: Array, **params) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable
    """

    def children(self) -> list:
        mat = getattr(self, "material", None)
        return [mat] if mat is not None else []
