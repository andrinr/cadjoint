"""How an SDF tree turns itself into JAX operations.

Two consumers trace the same tree and want opposite things from it.

*XLA* wants **structure**: an ``(N, 2)`` vertex array reduced once, a pattern
whose child is traced once under :func:`jax.vmap`, a shared subtree emitted as
one ``func.func``.  Program size then scales with the *shapes* in the tree
rather than with its unrolled leaf count, which is what makes a 168-vertex
profile cost the same to compile as a 12-vertex one.

*WGSL* wants **scalars**: the shader backend maps StableHLO tensors onto
``f32``/``vec2``–``vec4``/``mat2``–``mat4`` and nothing else, so any array with
more than four rows — every stacked vertex loop, every batched pattern
instance — is untranslatable.  Under :func:`scalar_lowering` the same tree
re-emits itself the way it always did: one op chain per vertex, one child copy
per pattern instance.

The flag is a :class:`~contextvars.ContextVar`, so it is per-thread and
per-async-task and never leaks out of the ``with`` block that set it.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass

__all__ = ["scalar_lowering", "vectorized_lowering", "is_scalar_lowering"]

_SCALAR: ContextVar[bool] = ContextVar("cadjoint_sdf_scalar_lowering", default=False)


def is_scalar_lowering() -> bool:
    """True while the tree must emit one scalar op chain per element.

    Returns:
        Whether a :func:`scalar_lowering` block is active.
    """
    return _SCALAR.get()


@contextlib.contextmanager
def scalar_lowering():
    """Emit unrolled, scalar-only operations inside this block.

    Required by the WGSL backend, whose type mapping stops at ``vec4``.

    Yields:
        None.
    """
    token = _SCALAR.set(True)
    try:
        yield
    finally:
        _SCALAR.reset(token)


@contextlib.contextmanager
def vectorized_lowering():
    """Emit array-shaped operations inside this block (the default).

    Yields:
        None.
    """
    token = _SCALAR.set(False)
    try:
        yield
    finally:
        _SCALAR.reset(token)


# ── conservative bounds ───────────────────────────────────────────────────────
#
# The second thing a consumer may want from the tree besides its operations is
# *where each node lives*.  A sphere trace evaluates the whole tree at every
# step, and a ``min`` over a far-away leaf is a ``select`` that still paid for
# the leaf.  Given a box that contains every point where a node's field is
# non-positive, a consumer can compare the distance to that box against what it
# already knows and skip the node outright.  The shader backend does exactly
# that (``cadjoint/backends/wgsl/_culling.py``); the bounds themselves are a
# property of the graph and are computed here, beside the lowering flag, so
# every backend reads one definition.
#
# The contract every rule below keeps, for a node with field ``f`` and bounds
# ``B = (center, half, scale)``: for every point ``p`` *outside* the box,
#
#     f(p)  >=  scale * dist(p, box)                                      (★)
#
# where ``dist`` is the Euclidean distance to the axis-aligned box.  ``scale``
# is 1 for everything except a non-uniform ``Scale``, which shrinks distances.
# The rules are derived in ``research/performance.md`` §13.3; each one is a
# consequence of (★) holding for the children plus a fact about the node —
# a transform is an isometry, a smooth union loses at most its blend band, a
# subtracted tool can only raise the value.  A node that cannot promise (★)
# — an infinite plane, a drafted extrusion — reports ``None`` and is never
# skipped.


@dataclass(frozen=True)
class Bounds:
    """An axis-aligned box a node's solid lies inside, in the node's own frame.

    Attributes:
        center: Box centre, shape ``(3,)``.
        half: Non-negative half extents, shape ``(3,)``.
        scale: The factor in (★): the field is at least ``scale`` times the
            distance to the box.  1 for every node but a non-uniform scale.
    """

    center: object
    half: object
    scale: object = 1.0

    @property
    def low(self):
        return self.center - self.half

    @property
    def high(self):
        return self.center + self.half


def box_distance(p, bounds: Bounds):
    """Euclidean distance from ``p`` to the box, zero inside it.

    Args:
        p: A point, shape ``(3,)``.
        bounds: The box.

    Returns:
        A scalar; the distance times ``bounds.scale``.
    """
    import jax.numpy as jnp

    q = jnp.maximum(jnp.abs(p - bounds.center) - bounds.half, 0.0)
    distance = jnp.sqrt(jnp.sum(q * q))
    if isinstance(bounds.scale, float) and bounds.scale == 1.0:
        return distance
    return bounds.scale * distance


def _hull(boxes: list) -> Bounds | None:
    """The smallest box containing every box in ``boxes`` (None if any is None)."""
    import jax.numpy as jnp

    if not boxes or any(box is None for box in boxes):
        return None
    low = boxes[0].low
    high = boxes[0].high
    scale = boxes[0].scale
    for box in boxes[1:]:
        low = jnp.minimum(low, box.low)
        high = jnp.maximum(high, box.high)
        scale = _min_scale(scale, box.scale)
    return Bounds(center=(low + high) * 0.5, half=(high - low) * 0.5, scale=scale)


def _min_scale(a, b):
    import jax.numpy as jnp

    if isinstance(a, float) and isinstance(b, float):
        return min(a, b)
    return jnp.minimum(a, b)


def _grown(bounds: Bounds | None, amount) -> Bounds | None:
    """The box grown by ``amount`` on every side (in the node's own units)."""
    if bounds is None:
        return None
    if isinstance(bounds.scale, float) and bounds.scale == 1.0:
        grow = amount
    else:
        grow = amount / bounds.scale
    return Bounds(center=bounds.center, half=bounds.half + grow, scale=bounds.scale)


def _cylinder_hull(bounds: Bounds, origin, axis) -> Bounds:
    """The box around the solid of revolution of ``bounds`` about a line.

    Everything a twist or a polar pattern does to a solid keeps each point's
    distance from the axis and its height along it, so the result lies in the
    cylinder swept by the child's bounding sphere.  The sphere is used rather
    than the eight corners because it is a handful of operations at every
    evaluation when the parameters are uniforms.
    """
    import jax.numpy as jnp

    axis = axis / jnp.sqrt(jnp.maximum(jnp.sum(axis * axis), 1e-12))
    radius_sphere = jnp.sqrt(jnp.sum(bounds.half * bounds.half))
    relative = bounds.center - origin
    height = jnp.sum(relative * axis)
    radial = relative - height * axis
    radius = jnp.sqrt(jnp.sum(radial * radial)) + radius_sphere
    low_h, high_h = height - radius_sphere, height + radius_sphere
    # Extent of {origin + h·axis + r·u : u ⊥ axis} along each world axis.
    across = radius * jnp.sqrt(jnp.maximum(1.0 - axis * axis, 0.0))
    along_low = jnp.minimum(low_h * axis, high_h * axis)
    along_high = jnp.maximum(low_h * axis, high_h * axis)
    low = origin + along_low - across
    high = origin + along_high + across
    return Bounds(center=(low + high) * 0.5, half=(high - low) * 0.5, scale=bounds.scale)


def _apply_rows(rows: list, vector):
    """``M @ vector`` where ``M`` is given as its three row vectors.

    Every consumer of these bounds has to be able to *emit* them, and the
    WGSL backend has no type wider than a ``vec4``: a ``(3, 3)`` array
    reaches its emitter as a two-dimensional slice it cannot write down.
    Keeping every linear map as three ``(3,)`` rows means the arithmetic
    here is the same arithmetic a shader can spell.
    """
    import jax.numpy as jnp

    return jnp.stack([jnp.sum(row * vector) for row in rows])


def _rotation_rows(axis, angle) -> list:
    """Rodrigues' rotation about ``axis`` by ``angle``, as three row vectors."""
    import jax.numpy as jnp

    axis = jnp.asarray(axis, jnp.float32)
    axis = axis / jnp.sqrt(jnp.maximum(jnp.sum(axis * axis), 1e-24))
    c, s = jnp.cos(angle), jnp.sin(angle)
    t = 1.0 - c
    x, y, z = axis[0], axis[1], axis[2]
    return [
        jnp.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y]),
        jnp.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x]),
        jnp.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c]),
    ]


def _reflection_rows(normal) -> list:
    """``I - 2 n nᵀ`` for a unit ``n``, as three row vectors."""
    import jax.numpy as jnp

    n = jnp.asarray(normal, jnp.float32)
    n = n / jnp.sqrt(jnp.maximum(jnp.sum(n * n), 1e-24))
    basis = [
        jnp.stack([jnp.float32(1.0), jnp.float32(0.0), jnp.float32(0.0)]),
        jnp.stack([jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0)]),
        jnp.stack([jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0)]),
    ]
    return [row - 2.0 * n[index] * n for index, row in enumerate(basis)]


def _transformed(bounds: Bounds | None, rows: list, offset) -> Bounds | None:
    """The box of ``{M a + offset : a in box}``, ``M`` given as its rows.

    A linear map sends the box to a parallelepiped; the smallest axis-aligned
    box around it has half extents ``|M| h``, because the extent along output
    axis *i* is the sum of ``|M_ij| h_j``. Exact, and conservative for the
    rotations and reflections that are the only maps reaching here.
    """
    import jax.numpy as jnp

    if bounds is None:
        return None
    return Bounds(
        center=_apply_rows(rows, bounds.center) + offset,
        half=_apply_rows([jnp.abs(row) for row in rows], bounds.half),
        scale=bounds.scale,
    )


def _profile_vertices(values: dict, prefix: str = "v") -> list:
    """The ``v0..vN`` (or ``w0..wN``) vertices of a profile primitive, in order.

    Each comes back as its own ``(2,)`` array rather than one stacked
    ``(N, 2)``: see :func:`_apply_rows` for why nothing here may be
    two-dimensional.
    """
    import jax.numpy as jnp

    names = sorted(
        (name for name in values if name.startswith(prefix) and name[len(prefix) :].isdigit()),
        key=lambda name: int(name[len(prefix) :]),
    )
    return [jnp.asarray(values[name], jnp.float32).reshape(2) for name in names]


def _fold(combine, items: list):
    """``combine`` applied left to right over a non-empty list."""
    result = items[0]
    for item in items[1:]:
        result = combine(result, item)
    return result


def _blend_band(smoothness):
    """How far a smooth minimum can fall below the plain minimum: its scaled band."""
    import jax.numpy as jnp

    return jnp.maximum(jnp.asarray(smoothness, dtype=jnp.float32) * 4.0, 1e-10)


def node_bounds(node, values: dict, children: list) -> Bounds | None:
    """The box a node's solid lies in, given its children's boxes.

    Args:
        node: The tree node.
        values: Its parameter values, as ``functionalize`` collects them —
            arrays or tracers, one per attribute in ``node.params``.
        children: The :class:`Bounds` (or ``None``) of each SDF child, in the
            frame the node hands them its query point.

    Returns:
        The node's bounds in its own frame, or ``None`` when it makes no
            promise — an unbounded field, a non-exact one, or an unknown
            node type.
    """
    import jax.numpy as jnp

    from cadjoint.sdf.boolean.difference import Difference
    from cadjoint.sdf.boolean.intersection import Intersection
    from cadjoint.sdf.boolean.union import Union
    from cadjoint.sdf.boolean.xor import Xor
    from cadjoint.sdf.operations import LinearPattern, Mirror, Offset, PolarPattern, Shell
    from cadjoint.sdf.primitives import (
        Box,
        Capsule,
        Cylinder,
        ExtrudedPolygon,
        LoftedPolygon,
        RevolvedPolygon,
        RoundBox,
        Sphere,
        Torus,
    )
    from cadjoint.sdf.transforms.affine import Rotate, Scale, Translate
    from cadjoint.sdf.transforms.deformations import Twist

    zero = jnp.zeros(3, dtype=jnp.float32)

    def vec(x, y, z):
        return jnp.stack(
            [jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32), jnp.asarray(z, jnp.float32)]
        )

    def leaf(half):
        return Bounds(center=zero, half=half)

    # ── primitives ────────────────────────────────────────────────────────
    if isinstance(node, Sphere):
        r = jnp.abs(values["radius"])
        return leaf(vec(r, r, r))
    if isinstance(node, Box):
        return leaf(jnp.abs(jnp.asarray(values["size"], jnp.float32)))
    if isinstance(node, RoundBox):
        return leaf(jnp.abs(jnp.asarray(values["size"], jnp.float32)) + jnp.abs(values["radius"]))
    if isinstance(node, Cylinder):
        r, h = jnp.abs(values["radius"]), jnp.abs(values["height"])
        return leaf(vec(r, r, h))
    if isinstance(node, Capsule):
        r, h = jnp.abs(values["radius"]), jnp.abs(values["height"])
        return leaf(vec(r, r, h + r))
    if isinstance(node, Torus):
        big, small = jnp.abs(values["major_radius"]), jnp.abs(values["minor_radius"])
        return leaf(vec(big + small, big + small, small))
    if isinstance(node, ExtrudedPolygon):
        if "draft" in values:
            # A drafted wall slides the 2D distance by a z-dependent term, so
            # the field falls below the box distance under the base cap.
            return None
        verts = _profile_vertices(values)
        half_depth = jnp.abs(values["depth"]) * 0.5
        if "twist" in values:
            # The twisted query keeps its distance from the z axis, so the
            # solid lies in the cylinder through the farthest vertex.
            radius = jnp.sqrt(_fold(jnp.maximum, [jnp.sum(v * v) for v in verts]))
            return leaf(vec(radius, radius, half_depth))
        low, high = _fold(jnp.minimum, verts), _fold(jnp.maximum, verts)
        centre = (low + high) * 0.5
        half = (high - low) * 0.5
        return Bounds(center=vec(centre[0], centre[1], 0.0), half=vec(half[0], half[1], half_depth))
    if isinstance(node, RevolvedPolygon):
        verts = _profile_vertices(values)
        radius = jnp.maximum(_fold(jnp.maximum, [v[0] for v in verts]) + values["offset"], 0.0)
        low_y = _fold(jnp.minimum, [v[1] for v in verts])
        high_y = _fold(jnp.maximum, [v[1] for v in verts])
        return Bounds(
            center=vec(0.0, (low_y + high_y) * 0.5, 0.0),
            half=vec(radius, (high_y - low_y) * 0.5, radius),
        )
    if isinstance(node, LoftedPolygon):
        verts = _profile_vertices(values, "v") + _profile_vertices(values, "w")
        low, high = _fold(jnp.minimum, verts), _fold(jnp.maximum, verts)
        centre = (low + high) * 0.5
        half = (high - low) * 0.5
        return Bounds(
            center=vec(centre[0], centre[1], 0.0),
            half=vec(half[0], half[1], jnp.abs(values["height"]) * 0.5),
        )

    # ── booleans ──────────────────────────────────────────────────────────
    if isinstance(node, Union):
        # smin never falls more than the scaled band below the plain minimum,
        # however many operands are chained (§13.3).
        return _grown(_hull(children), _blend_band(values.get("smoothness", 0.1)))
    if isinstance(node, (Difference, Intersection)):
        # A smooth maximum is never below either operand, so the base body's
        # own promise carries through every cut.
        return children[0] if children else None
    if isinstance(node, Xor):
        return _hull(children)

    # ── transforms and operations ─────────────────────────────────────────
    if not children or children[0] is None:
        return None
    child = children[0]
    if isinstance(node, Translate):
        return Bounds(center=child.center + values["offset"], half=child.half, scale=child.scale)
    if isinstance(node, Rotate):
        return _transformed(child, _rotation_rows(values["axis"], values["angle"]), zero)
    if isinstance(node, Scale):
        scale = jnp.asarray(values["scale"], jnp.float32)
        magnitude = jnp.abs(scale)
        largest = jnp.maximum(jnp.maximum(magnitude[0], magnitude[1]), magnitude[2])
        smallest = jnp.minimum(jnp.minimum(magnitude[0], magnitude[1]), magnitude[2])
        # A uniform scale multiplies the field by its factor, which exactly
        # cancels the box's growth. A non-uniform one stretches distance by
        # somewhere between the smallest and largest factor, so the promise
        # survives only at the smallest.
        return Bounds(
            center=child.center * scale,
            half=child.half * magnitude,
            scale=child.scale * (smallest / largest),
        )
    if isinstance(node, Mirror):
        origin = jnp.asarray(values["origin"], jnp.float32)
        rows = _reflection_rows(values["normal"])
        return _transformed(child, rows, origin - _apply_rows(rows, origin))
    if isinstance(node, Shell):
        return _grown(child, jnp.abs(values["thickness"]) * 0.5)
    if isinstance(node, Offset):
        return _grown(child, jnp.maximum(values["distance"], 0.0))
    if isinstance(node, Twist):
        return _cylinder_hull(child, zero, jnp.asarray(values["axis"], jnp.float32))
    if isinstance(node, LinearPattern):
        count = int(values["count"])
        if count <= 1:
            return child
        direction = jnp.asarray(values["direction"], jnp.float32)
        axis = direction / jnp.sqrt(jnp.maximum(jnp.sum(direction * direction), 1e-24))
        shift = axis * (values["spacing"] * (count - 1))
        low = jnp.minimum(child.low, child.low + shift)
        high = jnp.maximum(child.high, child.high + shift)
        return Bounds(center=(low + high) * 0.5, half=(high - low) * 0.5, scale=child.scale)
    if isinstance(node, PolarPattern):
        if int(values["count"]) <= 1:
            return child
        return _cylinder_hull(
            child,
            jnp.asarray(values["origin"], jnp.float32),
            jnp.asarray(values["direction"], jnp.float32),
        )
    return None


__all__ += ["Bounds", "box_distance", "node_bounds", "rotation_rows"]

#: Re-exported for the shader backend's polar pattern, which rotates a box
#: instance by instance and must spell the same arithmetic.
rotation_rows = _rotation_rows
