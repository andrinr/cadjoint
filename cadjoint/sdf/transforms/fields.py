"""Transforms that reshape the field itself: shell, offset and mirror.

Each evaluates its child once, at a modified point or with a modified
distance, so the surface moves without the child being re-traced.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.geometry.parameters import Scalar
from cadjoint.sdf.transforms._operation import (
    _MIRROR_NORMALS,
    _child_patch_fields,
    _Operation,
    _reference_line,
)


def _mirror_plane(axis) -> tuple[Array, Array]:
    """The ``(origin, unit normal)`` of a mirror plane named by ``axis``.

    Args:
        axis: ``'x'``, ``'y'`` or ``'z'`` for the coordinate plane with that
            normal through the world origin, or any Face / SketchPlane.

    Returns:
        The plane's origin and unit normal.

    Raises:
        ValueError: If ``axis`` is neither a known axis name nor something
            carrying an ``origin`` and a ``normal``.
    """
    if isinstance(axis, str):
        if axis not in _MIRROR_NORMALS:
            raise ValueError(f"Mirror axis must be 'x', 'y', or 'z', got {axis!r}")
        return jnp.zeros(3, dtype=jnp.float32), jnp.asarray(_MIRROR_NORMALS[axis])
    plane = _reference_line(axis, "normal")
    if plane is None:
        raise ValueError(
            f"Mirror axis must be 'x', 'y', 'z', or a Face / SketchPlane, got {axis!r}"
        )
    return plane


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

    def patch_fields(self):
        """Both offsets of every child patch: outward first, then inward.

        ``sdf(p) = |f(p)| - t/2`` is the max composition
        ``max(f - t/2, -(f + t/2))``, so the wall's two surfaces sit where
        the child field reads ``+t/2`` and ``-t/2``.  Splitting that over the
        child's own patches, child patch ``f_i`` contributes exactly two
        fields, kept adjacent and always in this sense:

        - ``f_i - t/2`` — the **outward** offset, patch index ``2*i``, whose
          zero set is child patch ``i`` pushed out by half the thickness;
        - ``-(f_i + t/2)`` — the **inward** offset, patch index ``2*i + 1``,
          child patch ``i`` pulled in by half the thickness, negated so it
          too reads positive outside the wall.

        Two caveats a reader needs, neither of which the consumer minds:

        The offsets are only as exact as the child's patch field is a *true
        distance* near its own surface — ``f_i(p) = ±t/2`` has to mean "half
        a thickness off patch ``i``".  Every primitive that declares patch
        fields satisfies this (they are plane, cylinder and revolved-profile
        distances, exact along their own normal), so the offset surfaces are
        exact for them; an approximate child field would offset by the wrong
        amount.

        The *trims* between the offset patches are not the child's trims.
        Offsetting outward pushes neighbouring patches apart and the node's
        own ``|f| - t/2`` rounds the gap over, while the max composition here
        continues both patches to a sharp intersection.  The consumer
        re-derives every trim curve from surface-surface intersections rather
        than inheriting it, so the untrimmed surfaces are what matters — but
        do not read ``argmin_i |f_i|`` as exact ownership in the rounded
        sliver just outside a convex child edge.
        """
        child_fields = _child_patch_fields(self.sdf)
        if child_fields is None:
            return None
        half = self.params["thickness"].value / 2.0
        fields = []
        for field in child_fields:
            fields.append(lambda p, f=field: f(p) - half)
            fields.append(lambda p, f=field: -(f(p) + half))
        return fields

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
    """Mirror a shape across a plane by reflecting the query point.

    ``Mirror(shape, 'x')`` reflects across the x=0 plane (the yz plane): the
    result is the mirror image of the child, an exact SDF.

    The three named axes are the coordinate planes *through the world origin*,
    which is the one mirror plane a real part rarely has. Any
    :class:`~cadjoint.construction.faces.Face` or
    :class:`~cadjoint.construction.sketch.SketchPlane` is accepted instead, so
    the symmetry plane can be a face of the part or a
    :meth:`~cadjoint.construction.sketch.SketchPlane.midplane` between two of
    them — and being built from the parent's parameters, it moves when the
    part is re-dimensioned rather than stranding the mirrored copy.

    Args:
        sdf: Child shape (SDF node or plain callable).
        axis: The mirror plane. ``'x'``, ``'y'`` or ``'z'`` for the coordinate
            plane with that normal through the origin, or a ``Face`` /
            ``SketchPlane`` to mirror across it.

    Example:
        ```python
        seam = SketchPlane.midplane(housing.cap("+"), housing.cap("-"))
        both_halves = Union(lug, Mirror(lug, seam))
        ```
    """

    def __init__(self, sdf, axis: str = "x"):
        self.sdf = sdf
        origin, normal = _mirror_plane(axis)
        self.params = {"origin": origin, "normal": normal}

    @staticmethod
    def _transform_point(p: Array, origin: Array, normal: Array) -> Array:
        return p - 2.0 * jnp.sum((p - origin) * normal, axis=-1, keepdims=True) * normal

    @staticmethod
    def sdf(child_sdf, p: Array, origin: Array, normal: Array) -> Array:
        """Pure function for the mirror operation."""
        return child_sdf(Mirror._transform_point(p, origin, normal))

    def __call__(self, p: Array) -> Array:
        return Mirror.sdf(self.sdf, p, self.params["origin"].xyz, self.params["normal"].xyz)

    def patch_fields(self):
        """Forward the child's patch fields through the reflection.

        A mirror only reflects — ``sdf(p) = child(M p)`` — so its surface is
        the child's surface reflected, patch for patch: the decomposition is
        the child's fields evaluated at the reflected point, in the child's
        own order and count.  Reflection is an isometry, so distances, patch
        ownership and feature edges all survive it exactly.
        """
        child_fields = _child_patch_fields(self.sdf)
        if child_fields is None:
            return None
        origin = self.params["origin"].xyz
        normal = self.params["normal"].xyz
        return [
            (lambda p, f=field: f(Mirror._transform_point(p, origin, normal)))
            for field in child_fields
        ]

    def to_functional(self):
        return Mirror.sdf


def shell(shape, thickness: float | Scalar) -> Shell:
    """Hollow shell of ``shape``'s surface with the given total thickness."""
    return Shell(shape, thickness)


def offset(shape, distance: float | Scalar) -> Offset:
    """Grow (positive) or shrink (negative) ``shape`` by ``distance``."""
    return Offset(shape, distance)


def mirror(shape, axis: str = "x") -> Mirror:
    """Mirror image of ``shape`` across a plane.

    Args:
        shape: The shape to reflect.
        axis: ``'x'``, ``'y'`` or ``'z'`` for the coordinate plane with that
            normal through the origin, or a ``Face`` / ``SketchPlane`` to
            mirror across it.
    """
    return Mirror(shape, axis)
