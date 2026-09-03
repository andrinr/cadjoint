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

import jax
import jax.numpy as jnp
from jax import Array

from cadjoint.fluent import Fluent
from cadjoint.geometry.parameters import Scalar
from cadjoint.sdf._lowering import is_scalar_lowering
from cadjoint.sdf.transforms.base import Transform

# The world plane each named mirror axis stands for: the coordinate plane
# through the origin whose normal is that axis.
_MIRROR_NORMALS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}

# Below this squared length a direction carries no orientation; the guarded
# norm keeps value and derivative finite where a bare norm would give 0/0.
_MIN_SQUARED = 1e-12


def _unit(vector: Array) -> Array:
    """Normalize a 3-vector with a guarded norm, safe under tracing."""
    return vector / jnp.sqrt(jnp.maximum(jnp.sum(vector * vector), _MIN_SQUARED))


def _world_vector(value) -> Array:
    """A world 3-vector from a raw array or from a ``Vector`` parameter."""
    return jnp.asarray(getattr(value, "xyz", value), dtype=jnp.float32)


def _reference_line(reference, attribute: str) -> tuple[Array, Array] | None:
    """``(origin, unit direction)`` read off a geometric reference, or None.

    Duck-typed on purpose: ``cadjoint.sdf`` must not import
    ``cadjoint.construction`` — the dependency runs the other way — but the
    references a user naturally reaches for all expose the same two
    attributes. An :class:`~cadjoint.construction.faces.Axis` carries
    ``origin``/``direction``; a :class:`~cadjoint.construction.faces.Face` and
    a :class:`~cadjoint.construction.sketch.SketchPlane` carry
    ``origin``/``normal``, the plane's ``origin`` being a ``Vector``
    parameter rather than a bare array.

    Args:
        reference: The object to read.
        attribute: ``"direction"`` for a line, ``"normal"`` for a plane.

    Returns:
        The pair, or None when ``reference`` does not carry both attributes.
    """
    origin = getattr(reference, "origin", None)
    vector = getattr(reference, attribute, None)
    if origin is None or vector is None:
        return None
    return _world_vector(origin), _unit(_world_vector(vector))


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


def _pattern_axis(axis) -> tuple[Array, Array]:
    """The ``(origin, unit direction)`` of a polar pattern's axis of rotation.

    Args:
        axis: ``'z'`` for the local z axis through the origin — the historical
            default — or an :class:`~cadjoint.construction.faces.Axis`, such as
            the one a ``revolve`` declares.

    Returns:
        The axis's origin and unit direction.

    Raises:
        ValueError: If ``axis`` is another axis *name*, or is not a line.
    """
    if isinstance(axis, str):
        if axis != "z":
            raise ValueError(
                f"PolarPattern only supports axis 'z', got {axis!r}. A name cannot say "
                "WHERE the axis is, which is most of what a bolt circle means; pass an "
                "Axis instead — a revolve's solid.axis, or "
                "cadjoint.construction.Axis(origin, direction)."
            )
        return jnp.zeros(3, dtype=jnp.float32), jnp.asarray([0.0, 0.0, 1.0])
    line = _reference_line(axis, "direction")
    if line is None:
        raise ValueError(f"PolarPattern axis must be 'z' or an Axis, got {axis!r}")
    return line


def _rotate_about(p: Array, origin: Array, axis: Array, angle: float) -> Array:
    """Rotate points about an arbitrary world line, by Rodrigues' formula.

    ``angle`` is a static Python float — every pattern angle is ``i·2π/count``
    on a concrete count — so the sine and cosine fold at trace time and the
    emitted graph is the same handful of multiplies the axis-aligned form used
    to be.

    Args:
        p: Points, shape ``(..., 3)``.
        origin: A point on the axis, shape ``(3,)``.
        axis: Unit axis direction, shape ``(3,)``.
        angle: Rotation angle in radians, right-handed about ``axis``.

    Returns:
        The rotated points, shaped like ``p``.
    """
    v = p - origin
    c, s = math.cos(angle), math.sin(angle)
    parallel = axis * jnp.sum(v * axis, axis=-1, keepdims=True)
    return origin + v * c + jnp.cross(axis, v) * s + parallel * (1.0 - c)


def _rotate_about_traced(p: Array, origin: Array, axis: Array, angle: Array) -> Array:
    """Rodrigues rotation with a *traced* angle, so instances can be batched.

    :func:`_rotate_about` folds its sine and cosine at trace time, which is
    right when each instance is emitted separately.  A vectorised pattern
    instead maps one child evaluation over an array of angles, and an array
    cannot be a Python float.

    Args:
        p: Points, shape ``(..., 3)``.
        origin: A point on the axis, shape ``(3,)``.
        axis: Unit axis direction, shape ``(3,)``.
        angle: Rotation angle in radians, right-handed about ``axis``.

    Returns:
        The rotated points, shaped like ``p``.
    """
    v = p - origin
    c, s = jnp.cos(angle), jnp.sin(angle)
    parallel = axis * jnp.sum(v * axis, axis=-1, keepdims=True)
    return origin + v * c + jnp.cross(axis, v) * s + parallel * (1.0 - c)


# A pattern's suppressed instances travel as a bit mask in one scalar
# parameter: the parameter plumbing carries numbers, and a set of indices is
# not one.  Bit ``i`` set means instance ``i`` is not emitted.  Instance 0 is
# the seed the child's analytic faces are declared against, so it can never be
# suppressed, and the mask is exact in a 32-bit integer up to this many
# instances.
_MAX_SKIPPABLE = 24


def _skip_mask(skip, count: int, kind: str) -> int:
    """Pack instance indices to suppress into a bit mask.

    Args:
        skip: Iterable of instance indices in ``1 .. count - 1``.
        count: The pattern's instance count.
        kind: Class name, for the error messages.

    Returns:
        The bit mask; ``0`` when nothing is suppressed.

    Raises:
        ValueError: If an index is not an integer in ``1 .. count - 1``, or the
            count exceeds what one mask can address.
    """
    indices = sorted({int(i) for i in skip}) if skip is not None else []
    if not indices:
        return 0
    if count > _MAX_SKIPPABLE:
        raise ValueError(
            f"{kind} can suppress instances only up to count {_MAX_SKIPPABLE}, got {count}."
        )
    for index in indices:
        if index == 0:
            raise ValueError(
                f"{kind} cannot skip instance 0: it is the seed copy, and the child's "
                "face references are declared against it. Move the seed instead."
            )
        if not 1 <= index < count:
            raise ValueError(
                f"{kind} skip index {index} is outside 1 .. {count - 1} for count {count}."
            )
    return sum(1 << index for index in indices)


def _kept_instances(count: int, skip_mask) -> list[int]:
    """The instance indices a pattern actually emits, seed first."""
    mask = int(skip_mask)
    return [i for i in range(count) if not mask >> i & 1]


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
        skip: Instance indices in ``1 .. count - 1`` that are *not* emitted —
            the row of holes that a real part interrupts where something else
            has to pass. Instance 0 is the seed and cannot be skipped; the
            remaining copies keep the indices they would have had, so
            suppressing one leaves a gap rather than closing up the row.

    Example:
        ```python
        # Six holes on 0.2 centers with the third one left undrilled.
        holes = LinearPattern(hole, direction=[1, 0, 0], count=6, spacing=0.2, skip=(2,))
        ```
    """

    def __init__(self, sdf, direction, count: int, spacing: float | Scalar, *, skip=()):
        count = int(count)
        if count < 1:
            raise ValueError(f"LinearPattern count must be >= 1, got {count}")
        self.sdf = sdf
        mask = _skip_mask(skip, count, "LinearPattern")
        self.params = {
            "direction": direction,
            "spacing": spacing,
            "count": count,
            "skip_mask": mask,
        }

    # Copy 0 is the original and is not displaced, so the child's analytic
    # face references still land on this node's surface.
    inherits_faces = True

    # ``count`` decides how many instances are *emitted*, so it has to be a
    # Python int at trace time even when every other parameter is an argument.
    static_params = ("count", "skip_mask")

    @property
    def skip(self) -> tuple[int, ...]:
        """The instance indices this pattern suppresses, ascending."""
        mask = int(self.params["skip_mask"].value)
        return tuple(i for i in range(int(self.params["count"].value)) if mask >> i & 1)

    @staticmethod
    def _transform_point(p: Array, direction, spacing, count, skip_mask) -> Array:  # noqa: ARG004
        # Material is looked up on the base copy; all copies share it anyway.
        return p

    @staticmethod
    def sdf(
        child_sdf,
        p: Array,
        direction: Array,
        spacing: Array,
        count: Array,
        skip_mask: Array = 0,
    ) -> Array:
        """Pure function for the linear pattern (count must be concrete).

        The child is traced **once**, over an array of instance offsets, so the
        emitted program holds one copy of the pattern's geometry rather than
        ``count`` of them.  Offset 0 is exactly zero, so copy 0 is still the
        child evaluated at ``p`` itself.  Under
        :func:`~cadjoint.sdf._lowering.scalar_lowering` the instances are
        unrolled again, because a shader cannot carry the batch axis.
        """
        num = int(count)
        axis = direction / jnp.linalg.norm(direction)
        kept = _kept_instances(num, skip_mask)
        if len(kept) == 1:
            return child_sdf(p)
        if is_scalar_lowering():
            d = child_sdf(p)
            for i in kept[1:]:
                d = jnp.minimum(d, child_sdf(p - axis * (spacing * i)))
            return d
        offsets = axis * (spacing * jnp.asarray(kept, dtype=jnp.float32))[:, None]
        return jnp.min(jax.vmap(lambda offset: child_sdf(p - offset))(offsets), axis=0)

    def __call__(self, p: Array) -> Array:
        return LinearPattern.sdf(
            self.sdf,
            p,
            self.params["direction"].xyz,
            self.params["spacing"].value,
            self.params["count"].value,
            self.params["skip_mask"].value,
        )

    def to_functional(self):
        return LinearPattern.sdf


class PolarPattern(_Operation):
    """Union of ``count`` copies rotated evenly around an axis.

    Copy ``i`` is the child rotated by ``i * 2π/count`` about the axis;
    evaluated as the minimum over all rotated child evaluations, so the count
    must be a static Python int. Copy 0 is the original, which is why the
    child's analytic faces still land on the result.

    The default axis is the local z axis through the origin. A bolt circle
    rarely sits there, and no *letter* can be made to fit it either: a name
    says which way an axis points but not where it is, and where it is happens
    to be most of what a bolt circle means. So any other axis is given as a
    line — an :class:`~cadjoint.construction.faces.Axis`, most usefully the one
    a ``revolve`` already declares as ``solid.axis``.

    Args:
        sdf: Child shape (SDF node or plain callable).
        count: Number of copies (static Python int, >= 1).
        axis: ``'z'`` for the local z axis through the origin, or an ``Axis``
            to rotate about that line.
        skip: Instance indices in ``1 .. count - 1`` that are *not* emitted.
            A ring of ribs or bolts is a pattern right up until something else
            — a coolant gallery, a cable exit, a keyway — has to occupy one of
            its stations, and then the choice is between abandoning the pattern
            and suppressing an instance. The kept copies keep the angles they
            would have had: instance ``i`` is still at ``i * 2π/count``, so
            suppressing one leaves a gap rather than respacing the ring.
            Instance 0 is the seed the child's faces are declared against and
            cannot be suppressed — rotate the seed instead.

    Example:
        ```python
        # Six screws on the bearing housing's own axis of revolution.
        bolts = PolarPattern(screw, count=6, axis=housing.axis)

        # Eight gussets, minus the two the coolant gallery passes through.
        ribs = PolarPattern(rib, count=8, axis=housing.axis, skip=(3, 5))
        ```
    """

    def __init__(self, sdf, count: int, axis: str = "z", *, skip=()):
        origin, direction = _pattern_axis(axis)
        count = int(count)
        if count < 1:
            raise ValueError(f"PolarPattern count must be >= 1, got {count}")
        self.sdf = sdf
        mask = _skip_mask(skip, count, "PolarPattern")
        self.params = {
            "count": count,
            "origin": origin,
            "direction": direction,
            "skip_mask": mask,
        }

    # A pattern does not move its base copy, so a face reference declared on
    # the child still lands on this node's surface.
    inherits_faces = True

    # ``count`` decides how many instances are *emitted*, so it has to be a
    # Python int at trace time even when every other parameter is an argument.
    static_params = ("count", "skip_mask")

    @property
    def skip(self) -> tuple[int, ...]:
        """The instance indices this pattern suppresses, ascending."""
        mask = int(self.params["skip_mask"].value)
        return tuple(i for i in range(int(self.params["count"].value)) if mask >> i & 1)

    @staticmethod
    def _transform_point(p: Array, count, origin, direction, skip_mask) -> Array:  # noqa: ARG004
        # Material is looked up on the base copy; all copies share it anyway.
        return p

    @staticmethod
    def sdf(
        child_sdf,
        p: Array,
        count: Array,
        origin: Array,
        direction: Array,
        skip_mask: Array = 0,
    ) -> Array:
        """Pure function for the polar pattern (count must be concrete).

        Copies 1..N-1 are traced **once**, over an array of instance angles, so
        the emitted program holds two copies of the pattern's geometry (the
        unrotated original and the batched rest) rather than ``count`` of them.
        Copy 0 stays a separate, unrotated evaluation: ``origin + (p - origin)``
        is only equal to ``p`` up to rounding, and copy 0 is the one the child's
        face references are declared against.  Under
        :func:`~cadjoint.sdf._lowering.scalar_lowering` every instance is
        unrolled again, because a shader cannot carry the batch axis.
        """
        num = int(count)
        d = child_sdf(p)
        kept = _kept_instances(num, skip_mask)
        if len(kept) == 1:
            return d
        if is_scalar_lowering():
            for i in kept[1:]:
                theta = 2.0 * math.pi * i / num
                d = jnp.minimum(d, child_sdf(_rotate_about(p, origin, direction, -theta)))
            return d
        angles = -2.0 * math.pi * jnp.asarray(kept[1:], dtype=jnp.float32) / num
        rotated = jax.vmap(
            lambda angle: child_sdf(_rotate_about_traced(p, origin, direction, angle))
        )(angles)
        return jnp.minimum(d, jnp.min(rotated, axis=0))

    def __call__(self, p: Array) -> Array:
        return PolarPattern.sdf(
            self.sdf,
            p,
            self.params["count"].value,
            self.params["origin"].xyz,
            self.params["direction"].xyz,
            self.params["skip_mask"].value,
        )

    def to_functional(self):
        return PolarPattern.sdf


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


def linear_pattern(
    shape, direction, count: int, spacing: float | Scalar, *, skip=()
) -> LinearPattern:
    """``count`` copies of ``shape`` spaced along ``direction``.

    ``skip`` names instance indices in ``1 .. count - 1`` to leave out.
    """
    return LinearPattern(shape, direction, count, spacing, skip=skip)


def polar_pattern(shape, count: int, axis: str = "z", *, skip=()) -> PolarPattern:
    """``count`` copies of ``shape`` rotated evenly around an axis.

    Args:
        shape: The shape to copy.
        count: Number of copies (static Python int, >= 1).
        axis: ``'z'`` for the local z axis through the origin, or an ``Axis``
            (such as a revolve's ``solid.axis``) to rotate about that line.
        skip: Instance indices in ``1 .. count - 1`` to leave out.
    """
    return PolarPattern(shape, count, axis, skip=skip)
