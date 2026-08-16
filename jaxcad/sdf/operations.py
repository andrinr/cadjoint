"""Field operations on SDFs: shell, offset, mirror, and patterns.

Each operation wraps a single child shape and evaluates it one or more times
with a modified query point and/or a modified distance value. They follow the
Transform pattern (one SDF child, static ``sdf(child_sdf, p, **params)``), so
``functionalize``/``functionalize_scene`` and the shader backends handle them
like any other unary node, and the viewer sees the wrapped shape as one node.

The child may also be a plain callable (``p -> distance``); direct evaluation
via ``__call__`` works either way, while compilation requires an SDF child.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

from jaxcad.fluent import Fluent
from jaxcad.geometry.parameters import Scalar
from jaxcad.sdf.transforms.base import Transform

_MIRROR_SIGNS = {
    "x": (-1.0, 1.0, 1.0),
    "y": (1.0, -1.0, 1.0),
    "z": (1.0, 1.0, -1.0),
}


class _Operation(Transform):
    """Shared plumbing for field operations wrapping one callable shape."""

    def children(self) -> list:
        # A plain lambda child cannot be walked as a Fluent tree node.
        return [self.sdf] if isinstance(self.sdf, Fluent) else []

    def material_at(self, p: Array) -> dict:
        if isinstance(self.sdf, Fluent):
            values = self._extract_param_values()
            return self.sdf.material_at(self.__class__._transform_point(p, **values))
        return super().material_at(p)


class Shell(_Operation):
    """Hollow shell of a shape's surface: ``|f(p)| - thickness/2``.

    The result is a wall of the given total thickness centered on the child's
    surface. Exact wherever the child field is exact.

    Args:
        sdf: Child shape (SDF node or plain callable).
        thickness: Total wall thickness.
    """

    def __init__(self, sdf, thickness: float | Scalar):
        self.sdf = sdf
        self.params = {"thickness": thickness}

    @staticmethod
    def _transform_point(p: Array, thickness: Array) -> Array:  # noqa: ARG004
        return p

    @staticmethod
    def sdf(child_sdf, p: Array, thickness: Array) -> Array:
        """Pure function for the shell operation."""
        return jnp.abs(child_sdf(p)) - thickness / 2.0

    def __call__(self, p: Array) -> Array:
        return Shell.sdf(self.sdf, p, self.params["thickness"].value)

    def to_functional(self):
        return Shell.sdf


class Offset(_Operation):
    """Offset (grow/shrink) a shape's surface: ``f(p) - distance``.

    Positive distances grow the shape outward; negative distances shrink it.

    Args:
        sdf: Child shape (SDF node or plain callable).
        distance: Offset distance.
    """

    def __init__(self, sdf, distance: float | Scalar):
        self.sdf = sdf
        self.params = {"distance": distance}

    @staticmethod
    def _transform_point(p: Array, distance: Array) -> Array:  # noqa: ARG004
        return p

    @staticmethod
    def sdf(child_sdf, p: Array, distance: Array) -> Array:
        """Pure function for the offset operation."""
        return child_sdf(p) - distance

    def __call__(self, p: Array) -> Array:
        return Offset.sdf(self.sdf, p, self.params["distance"].value)

    def to_functional(self):
        return Offset.sdf


class Mirror(_Operation):
    """Mirror a shape across a coordinate plane by reflecting the query point.

    ``Mirror(shape, 'x')`` reflects across the x=0 plane (the yz plane): the
    result is the mirror image of the child, an exact SDF.

    Args:
        sdf: Child shape (SDF node or plain callable).
        axis: Axis whose coordinate is negated: 'x', 'y', or 'z'.
    """

    def __init__(self, sdf, axis: str = "x"):
        if axis not in _MIRROR_SIGNS:
            raise ValueError(f"Mirror axis must be 'x', 'y', or 'z', got {axis!r}")
        self.sdf = sdf
        self.params = {"sign": jnp.array(_MIRROR_SIGNS[axis])}

    @staticmethod
    def _transform_point(p: Array, sign: Array) -> Array:
        return p * sign

    @staticmethod
    def sdf(child_sdf, p: Array, sign: Array) -> Array:
        """Pure function for the mirror operation."""
        return child_sdf(Mirror._transform_point(p, sign))

    def __call__(self, p: Array) -> Array:
        return Mirror.sdf(self.sdf, p, self.params["sign"].xyz)

    def to_functional(self):
        return Mirror.sdf


class LinearPattern(_Operation):
    """Union of ``count`` copies translated along a direction.

    Copy ``i`` sits at ``i * spacing`` along the normalized direction; copy 0
    is the original. Evaluated as the minimum over all translated child
    evaluations, so the count must be a static Python int.

    Args:
        sdf: Child shape (SDF node or plain callable).
        direction: Pattern direction, shape (3,) (normalized internally).
        count: Number of copies (static Python int, >= 1).
        spacing: Center-to-center spacing between neighboring copies.
    """

    def __init__(self, sdf, direction, count: int, spacing: float | Scalar):
        count = int(count)
        if count < 1:
            raise ValueError(f"LinearPattern count must be >= 1, got {count}")
        self.sdf = sdf
        self.params = {"direction": direction, "spacing": spacing, "count": count}

    @staticmethod
    def _transform_point(p: Array, direction, spacing, count) -> Array:  # noqa: ARG004
        # Material is looked up on the base copy; all copies share it anyway.
        return p

    @staticmethod
    def sdf(child_sdf, p: Array, direction: Array, spacing: Array, count: Array) -> Array:
        """Pure function for the linear pattern (count must be concrete)."""
        num = int(count)
        axis = direction / jnp.linalg.norm(direction)
        d = child_sdf(p)
        for i in range(1, num):
            d = jnp.minimum(d, child_sdf(p - axis * (spacing * i)))
        return d

    def __call__(self, p: Array) -> Array:
        return LinearPattern.sdf(
            self.sdf,
            p,
            self.params["direction"].xyz,
            self.params["spacing"].value,
            self.params["count"].value,
        )

    def to_functional(self):
        return LinearPattern.sdf


class PolarPattern(_Operation):
    """Union of ``count`` copies rotated evenly around the z axis.

    Copy ``i`` is the child rotated by ``i * 2π/count`` about the local z
    axis; evaluated as the minimum over all rotated child evaluations, so the
    count must be a static Python int.

    Args:
        sdf: Child shape (SDF node or plain callable).
        count: Number of copies (static Python int, >= 1).
        axis: Rotation axis; only 'z' is supported.
    """

    def __init__(self, sdf, count: int, axis: str = "z"):
        if axis != "z":
            raise ValueError(f"PolarPattern only supports axis 'z', got {axis!r}")
        count = int(count)
        if count < 1:
            raise ValueError(f"PolarPattern count must be >= 1, got {count}")
        self.sdf = sdf
        self.params = {"count": count}

    @staticmethod
    def _transform_point(p: Array, count: Array) -> Array:  # noqa: ARG004
        # Material is looked up on the base copy; all copies share it anyway.
        return p

    @staticmethod
    def sdf(child_sdf, p: Array, count: Array) -> Array:
        """Pure function for the polar pattern (count must be concrete)."""
        num = int(count)
        d = child_sdf(p)
        for i in range(1, num):
            theta = 2.0 * math.pi * i / num
            c, s = math.cos(theta), math.sin(theta)
            q = jnp.stack(
                [p[..., 0] * c + p[..., 1] * s, p[..., 1] * c - p[..., 0] * s, p[..., 2]],
                axis=-1,
            )
            d = jnp.minimum(d, child_sdf(q))
        return d

    def __call__(self, p: Array) -> Array:
        return PolarPattern.sdf(self.sdf, p, self.params["count"].value)

    def to_functional(self):
        return PolarPattern.sdf


def shell(shape, thickness: float | Scalar) -> Shell:
    """Hollow shell of ``shape``'s surface with the given total thickness."""
    return Shell(shape, thickness)


def offset(shape, distance: float | Scalar) -> Offset:
    """Grow (positive) or shrink (negative) ``shape`` by ``distance``."""
    return Offset(shape, distance)


def mirror(shape, axis: str = "x") -> Mirror:
    """Mirror image of ``shape`` across the plane normal to ``axis``."""
    return Mirror(shape, axis)


def linear_pattern(shape, direction, count: int, spacing: float | Scalar) -> LinearPattern:
    """``count`` copies of ``shape`` spaced along ``direction``."""
    return LinearPattern(shape, direction, count, spacing)


def polar_pattern(shape, count: int, axis: str = "z") -> PolarPattern:
    """``count`` copies of ``shape`` rotated evenly around the z axis."""
    return PolarPattern(shape, count, axis)
