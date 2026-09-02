"""Sketch construction tree: planes and 2D profiles that generate solids.

The construction tree is a second object tree next to the SDF tree. It holds
the editable CAD scaffolding — sketch planes and parameter-bearing 2D profiles.
Generators (:func:`cadjoint.construction.extrude`,
:func:`cadjoint.construction.revolve`) turn profiles into SDF nodes that share
the *same* ``Parameter`` objects, so constraints solved on the sketch and
gradients flowing through the solid act on one set of values.

The construction tree is never rasterized by the SDF renderer; it is drawn as
a wireframe overlay on top of rendered images (see ``cadjoint.render.overlay``).

Planes from references
----------------------

``SketchPlane(origin, normal)`` states a plane in world coordinates and is the
right thing for the first sketch in a program. Every plane after that is
usually *derived* — it sits on a face of something already built — and the
alternate constructors say so directly:

- :meth:`SketchPlane.on` — the plane of a :class:`~cadjoint.construction.faces.Face`.
- :meth:`SketchPlane.offset` — a face or plane pushed along its normal, by a
  distance that may itself be a ``Scalar`` parameter.
- :meth:`SketchPlane.tangent` — the tangent plane of an SDF at a point, read
  off the field's own gradient. This is the reference for blends, fillets and
  revolved surfaces, which have no analytic face to name.
- :meth:`SketchPlane.midplane` — halfway between two faces.

A derived plane recomputes itself from the parent's parameters, so re-running
the program with a different ``depth`` moves it — which is what makes the
feature tree rebuild correctly, and is the whole point of the reference.

What it is *not* is a live edge in the compiled parameter graph. The derived
origin is evaluated when the plane is constructed and stored as an ordinary
fixed ``Parameter``, so ``extract_parameters`` on a child solid does not see
the parent's ``depth`` and ``jax.grad`` through the child w.r.t. it is exactly
zero. Re-run the program to move the child; do not expect an optimizer to
discover the link. Making it live needs derived parameters, which the leaf-
valued ``Parameter`` model does not have — see ``research/complex-scene.md``.

In-plane rotation
-----------------

``u`` and ``v`` used to be derived from the normal alone, which made a
sketch's "horizontal" an accident of the derivation on any tilted plane. A
plane now takes an explicit ``x_axis``; omitting it keeps the old derivation
exactly, so existing sketches are unchanged. The derived constructors fill it
in from the reference — the profile's ``u`` for a cap, the swept edge
direction for a side wall, the most nearly in-plane world axis for a tangent
plane.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from cadjoint.fluent import Fluent
from cadjoint.geometry.parameters import Parameter, Vector, Vector2, as_parameter

# Below this squared length a direction carries no orientation; the guarded
# norm keeps value and derivative finite where a bare norm would give 0/0.
_MIN_SQUARED = 1e-12

# The error JAX raises when a tracer is forced to a Python value (bool() or
# float()). Catching it is how this module tells "concrete scene" from
# "inside jit/grad" without type-sniffing tracer classes. Nothing else is
# caught: a genuinely bad value must still raise its own error.
_NOT_CONCRETE = jax.errors.ConcretizationTypeError


def _unit(vector: Array) -> Array:
    """Normalize a vector with a guarded norm, safe under tracing."""
    return vector / jnp.sqrt(jnp.maximum(jnp.sum(vector * vector), _MIN_SQUARED))


def _maybe_float(value) -> float | None:
    """The Python value of a scalar, or None when it is a tracer."""
    try:
        return float(value)
    except _NOT_CONCRETE:
        return None


def _scalar_value(value):
    """The current number behind a ``Scalar`` parameter, or the number itself."""
    return value.value if isinstance(value, Parameter) else value


def _plane_frame(normal: Array, x_axis: Array | None = None) -> tuple[Array, Array]:
    """Orthonormal in-plane axes (u, v) for a unit normal; right-handed u×v=n.

    With no ``x_axis`` the reference up-vector is +Y so the default +Z normal
    yields the identity frame u=(1,0,0), v=(0,1,0) — sketch coordinates on the
    world XY plane are world coordinates. The +Y fallback to +Z is written as
    a select rather than a Python branch so the derivation stays traceable
    when the normal is a parameter.

    Args:
        normal: Unit plane normal.
        x_axis: Optional in-plane "horizontal"; its component along the normal
            is removed. When None the up-vector derivation above is used.

    Returns:
        The pair ``(u, v)`` of orthonormal in-plane axes.
    """
    if x_axis is None:
        up = jnp.where(
            jnp.abs(jnp.dot(normal, jnp.array([0.0, 1.0, 0.0]))) > 0.99,
            jnp.array([0.0, 0.0, 1.0]),
            jnp.array([0.0, 1.0, 0.0]),
        )
        u = _unit(jnp.cross(up, normal))
    else:
        x_axis = jnp.asarray(x_axis)
        u = _unit(x_axis - jnp.sum(x_axis * normal) * normal)
    v = jnp.cross(normal, u)
    return u, v


def _rotation_axis(matrix: Array, angle: Array) -> Array:
    """Rotation axis of ``matrix``, as a traceable expression.

    The concrete path in :meth:`SketchPlane.axis_angle` branches on the angle;
    under a trace there is no branch to take, so both cases are evaluated and
    selected. Near ``sin(angle) == 0`` the antisymmetric part vanishes and the
    axis comes from the symmetric matrix ``(R + I) / 2`` instead — whose
    largest diagonal entry is at least 1/3 for any rotation, so the square
    root never touches zero. At ``angle == 0`` that same branch returns some
    unit axis, which is correct: the rotation is the identity either way, and
    a unit axis keeps ``Rotate`` free of a 0/0 normalization.
    """
    sine = jnp.sin(angle)
    antisymmetric = jnp.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    degenerate = jnp.abs(sine) <= 1e-6
    generic = antisymmetric / (2.0 * jnp.where(degenerate, 1.0, sine))
    symmetric = (matrix + jnp.eye(3)) / 2.0
    diagonal = jnp.diag(symmetric)
    column = jnp.argmax(diagonal)
    near_pi = symmetric[:, column] / jnp.sqrt(jnp.maximum(diagonal[column], 1e-12))
    return _unit(jnp.where(degenerate, near_pi, generic))


def _reference_frame(reference) -> tuple[Array, Array, Array]:
    """The ``(origin, normal, x_axis)`` of a Face or a SketchPlane."""
    from cadjoint.construction.faces import Face

    if isinstance(reference, Face):
        return reference.origin, reference.normal, reference.x_axis
    if isinstance(reference, SketchPlane):
        u, _, normal = reference.frame()
        return reference.origin.xyz, normal, u
    raise TypeError(f"Expected a Face or a SketchPlane, got {type(reference).__name__}.")


def _most_in_plane_axis(normal: Array) -> Array:
    """The world axis most nearly inside the plane of ``normal``.

    Picking the axis least aligned with the normal is what makes a tangent
    plane's "horizontal" predictable: on a nearly-vertical wall the sketch's
    x runs along the world axis a user would have picked anyway, and the
    choice only changes when the surface has turned far enough to make the
    old axis a worse fit than the new one.
    """
    axes = jnp.eye(3)
    return axes[jnp.argmin(jnp.abs(axes @ normal))]


class SketchPlane(Fluent):
    """A work plane the sketch entities live on.

    Defines a coordinate frame: ``origin`` plus orthonormal in-plane axes
    (u, v) derived from ``normal`` and, optionally, an explicit ``x_axis``.
    Profile coordinates (x, y) map to world space as ``origin + x·u + y·v``.

    Args:
        origin: Plane origin in world space (Vector parameter or 3D array).
        normal: Plane normal (Vector parameter or 3D array); default +Z, i.e.
            the world XY plane.
        x_axis: Optional in-plane "horizontal". Its component along the normal
            is removed, so any direction not parallel to the normal will do.
            Omit it to keep the normal-only derivation. Orientation is a
            snapshot — unlike ``origin`` it is not a tracked parameter — which
            mirrors how the generators already capture rotation at generation
            time.
    """

    def __init__(self, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), x_axis=None):
        origin = (
            as_parameter(jnp.asarray(origin, dtype=jnp.float32))
            if not isinstance(origin, Vector)
            else origin
        )
        normal = (
            as_parameter(jnp.asarray(normal, dtype=jnp.float32))
            if not isinstance(normal, Vector)
            else normal
        )
        self.params = {"origin": origin, "normal": _normalized(normal)}
        self._x_axis = None if x_axis is None else jnp.asarray(x_axis)
        self.reference = None
        """What this plane was derived from — a Face, an SDF, or None."""

    # ── derived constructors ─────────────────────────────────────────────────

    @classmethod
    def on(cls, face, x_axis=None, flip: bool = False) -> SketchPlane:
        """The plane of a face, moving with the feature's parameters.

        Args:
            face: A :class:`~cadjoint.construction.faces.Face` — ``solid.cap("+")``,
                ``solid.side(2)``, ``block.face("+x")``.
            x_axis: Override the sketch's in-plane horizontal; defaults to the
                face's own x axis (the profile's ``u`` for a cap, the swept
                edge direction for a side wall).
            flip: Face the plane the other way, keeping the same origin.

        Returns:
            A plane whose origin is an expression in the parent feature's
                parameters, so it follows the parent and carries its gradient.

        Example:
            ```python
            body = extrude(profile, depth=fin_depth)
            boss = PolygonProfile(SQUARE, plane=SketchPlane.on(body.cap("+")))
            ```
        """
        normal = -face.normal if flip else face.normal
        plane = cls(
            origin=face.origin,
            normal=normal,
            x_axis=face.x_axis if x_axis is None else x_axis,
        )
        plane.reference = face
        return plane

    @classmethod
    def offset(cls, reference, distance) -> SketchPlane:
        """A face or plane pushed along its own normal.

        Args:
            reference: A :class:`~cadjoint.construction.faces.Face` or another
                :class:`SketchPlane`.
            distance: How far to move along the normal. May be a ``Scalar``
                parameter, in which case the offset is a design variable like
                any other.

        Returns:
            The offset plane, keeping the reference's orientation.
        """
        origin, normal, x_axis = _reference_frame(reference)
        amount = distance.value if isinstance(distance, Parameter) else jnp.asarray(distance)
        plane = cls(origin=origin + amount * normal, normal=normal, x_axis=x_axis)
        plane.reference = reference
        return plane

    @classmethod
    def tangent(cls, solid, near, x_axis=None, *, max_step: float = 1.0, steps: int = 8):
        """The tangent plane of an SDF at the surface point nearest ``near``.

        This is the reference for everything with no analytic face: blends,
        fillets, the curved wall of a revolve. ``near`` is Newton-projected
        onto the zero set with
        :func:`cadjoint.fem.motion.project_points`, and the plane's normal is
        the field's gradient there — both differentiable, so the plane tracks
        the scene's parameters rather than a snapshot of them.

        Args:
            solid: Any callable scalar field on ``(3,)`` points — an SDF
                instance is one.
            near: A world point at or near the surface; a raymarch hit is the
                intended source.
            x_axis: Override the in-plane horizontal; defaults to the world
                axis most nearly inside the tangent plane.
            max_step: Cap on how far the projection may move ``near``.
            steps: Newton iterations.

        Returns:
            The tangent plane at the projected point.
        """
        from cadjoint.fem.motion import project_points

        field = lambda point: jnp.asarray(solid(point)).reshape(())  # noqa: E731
        point = project_points(field, jnp.asarray(near)[None], max_step, steps=steps)[0]
        normal = _unit(jax.grad(field)(point))
        plane = cls(
            origin=point,
            normal=normal,
            x_axis=_most_in_plane_axis(normal) if x_axis is None else x_axis,
        )
        plane.reference = solid
        return plane

    @classmethod
    def midplane(cls, face_a, face_b) -> SketchPlane:
        """The plane halfway between two faces.

        Args:
            face_a: The first face (or plane).
            face_b: The second face (or plane).

        Returns:
            A plane at the midpoint of the two origins, normal to the average
                of the two normals. A face pair usually points at each other, so
                the second normal is flipped into agreement with the first before
                averaging; that keeps the result well defined for the parallel,
                antiparallel and general cases alike.
        """
        origin_a, normal_a, x_axis = _reference_frame(face_a)
        origin_b, normal_b, _ = _reference_frame(face_b)
        aligned = jnp.where(jnp.dot(normal_a, normal_b) < 0.0, -normal_b, normal_b)
        return cls(
            origin=(origin_a + origin_b) / 2.0,
            normal=_unit(normal_a + aligned),
            x_axis=x_axis,
        )

    # ── frame ────────────────────────────────────────────────────────────────

    @property
    def origin(self) -> Vector:
        return self.params["origin"]

    @property
    def normal(self) -> Vector:
        return self.params["normal"]

    @property
    def x_axis(self) -> Array | None:
        """The explicit in-plane horizontal, or None when derived from the normal."""
        return self._x_axis

    def frame(self) -> tuple[Array, Array, Array]:
        """Return the (u, v, n) orthonormal frame as arrays."""
        n = self.normal.xyz
        u, v = _plane_frame(n, self._x_axis)
        return u, v, n

    def rotation_matrix(self) -> Array:
        """Rotation taking local (x, y, z) to world (u, v, n), columns [u v n]."""
        u, v, n = self.frame()
        return jnp.stack([u, v, n], axis=1)

    def axis_angle(self) -> tuple[Array, float | Array]:
        """Axis-angle form of :meth:`rotation_matrix` (for the Rotate transform).

        Returns:
            The pair ``(axis, angle)``. ``angle`` is a Python float whenever
                the plane's orientation is concrete — which is what lets
                :func:`cadjoint.construction.extrude._place_on_plane` drop an
                identity rotation entirely — and a traced array otherwise.
        """
        R = self.rotation_matrix()
        trace = jnp.trace(R)
        angle = jnp.arccos(jnp.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        concrete = _maybe_float(angle)
        if concrete is None:
            return _rotation_axis(R, angle), angle
        if concrete < 1e-6:
            return jnp.array([0.0, 0.0, 1.0]), 0.0
        if concrete > jnp.pi - 1e-4:
            # R is symmetric: axis from the largest diagonal of (R + I) / 2
            B = (R + jnp.eye(3)) / 2.0
            k = int(jnp.argmax(jnp.diag(B)))
            axis = B[:, k] / jnp.sqrt(B[k, k])
            return axis, float(jnp.pi)
        axis = jnp.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2.0 * jnp.sin(concrete))
        return axis, concrete

    def is_identity(self) -> bool:
        """True when the plane is the world XY plane at the origin.

        A plane whose origin or orientation is traced answers False: it cannot
        be *proved* identical to the world frame, and the placement transforms
        it then keeps are the identity anyway.
        """
        try:
            u, _, n = self.frame()
            return bool(
                jnp.allclose(n, jnp.array([0.0, 0.0, 1.0]))
                and jnp.allclose(u, jnp.array([1.0, 0.0, 0.0]))
                and jnp.allclose(self.origin.xyz, 0.0)
            )
        except _NOT_CONCRETE:
            return False

    def to_world(self, xy: Array) -> Array:
        """Map plane coordinates (..., 2) to world coordinates (..., 3)."""
        u, v, _ = self.frame()
        xy = jnp.asarray(xy)
        return self.origin.xyz + xy[..., :1] * u + xy[..., 1:2] * v

    def describe(self) -> dict:
        """Serialize the plane's frame for the viewer."""
        u, v, n = self.frame()
        return {
            "origin": [float(x) for x in self.origin.xyz],
            "u": [float(x) for x in u],
            "v": [float(x) for x in v],
            "normal": [float(x) for x in n],
        }


def _normalized(normal: Vector) -> Vector:
    """Unit-length copy of a normal parameter, keeping its name and freedom.

    ``Vector.normalize`` rejects a zero-length vector, which needs the value
    as a Python bool; under a trace there is none, so the guarded norm is used
    instead and the zero-normal check simply does not apply.
    """
    try:
        return normal.normalize()
    except _NOT_CONCRETE:
        return Vector(
            value=_unit(normal.value),
            free=normal.free,
            name=normal.name,
            bounds=normal.bounds,
        )


class PolygonProfile(Fluent):
    """Closed polygon profile in a sketch plane — a construction-tree node.

    Vertices are :class:`Vector2` parameters in plane coordinates. Raw arrays
    are wrapped as **free** parameters named ``{name}_v{i}`` — sketch vertices
    are editable until constrained, which mirrors CAD sketch semantics.
    Existing ``Vector2`` parameters are used as-is (free unnamed ones get an
    auto-generated name so extraction works).

    Args:
        vertices: Ordered vertex list — Vector2 parameters or raw (2,) arrays.
            Either winding; the polygon must be simple (non-self-intersecting).
        plane: The sketch plane; defaults to the world XY plane.
        name: Prefix for auto-generated vertex parameter names.
        free: Whether raw vertices become *free* parameters. True keeps the CAD
            sketch semantics — a vertex is editable until constrained — and is
            the default. The generated outlines (:meth:`circle`,
            :meth:`regular`, :meth:`rounded_rect`) pass False: a computed
            vertex is a consequence of the shape's dimensions, not a freedom
            of its own, and freeing dozens of them would swamp the design
            space. Existing ``Vector2`` parameters keep whatever they already
            declare.
    """

    def __init__(
        self,
        vertices,
        plane: SketchPlane | None = None,
        name: str = "profile",
        free: bool = True,
    ):
        if len(vertices) < 3:
            raise ValueError(f"PolygonProfile needs at least 3 vertices, got {len(vertices)}")
        self.plane = plane if plane is not None else SketchPlane()
        self.name = name

        wrapped: list[Vector2] = []
        for i, v in enumerate(vertices):
            if isinstance(v, Vector2):
                if v.free and v.name is None:
                    v.name = f"{name}_v{i}"
                wrapped.append(v)
            else:
                wrapped.append(
                    Vector2(
                        value=jnp.asarray(v, dtype=jnp.float32),
                        free=free,
                        name=f"{name}_v{i}",
                    )
                )
        self.vertices = wrapped
        self.params = {f"v{i}": v for i, v in enumerate(wrapped)}

    # ── generated outlines ───────────────────────────────────────────────────
    #
    # A profile is a polygon and nothing else, so every curve in this modeller
    # is a polygon fine enough to read as one. Typing a bolt-circle flange out
    # vertex by vertex is the ugly workaround these remove; they are ordinary
    # constructors that happen to compute their vertex list.
    #
    # Generated vertices are PINNED (``free=False``) unless asked otherwise.
    # Two reasons, both worth knowing before reaching for ``free=True``: an
    # individual vertex of a circle is not a design freedom — dragging one
    # makes the circle not-a-circle — and a free vertex is extracted as its own
    # optimization variable, so a 48-segment bore would quietly add 96 of them.
    # The shape's *dimensions* stay editable the ordinary way: edit the call.
    #
    # What these cannot do is keep the radius live. A generated vertex is a
    # number computed once, not an expression in ``radius``, so ``jax.grad``
    # of anything downstream w.r.t. a ``Scalar`` radius passed here is zero.
    # For a circular feature that must stay differentiable, use the exact
    # primitives — ``Solid.cylinder``, ``Face.hole``, or a ``revolve`` — whose
    # radius is a shared ``Parameter``. See research/complex-scene.md.

    @classmethod
    def circle(cls, radius, center=(0.0, 0.0), segments: int = 32, **kwargs) -> PolygonProfile:
        """A closed circular outline, approximated by ``segments`` edges.

        Args:
            radius: Circle radius, as a number (see the note above on why a
                ``Scalar`` here does not stay live).
            center: Circle centre in plane coordinates.
            segments: Number of edges; 32 reads as round at typical scales.
            **kwargs: Passed to :class:`PolygonProfile` — ``plane``, ``name``,
                and ``free``.

        Returns:
            The profile, wound counter-clockwise.
        """
        return cls.regular(segments, radius, center=center, **kwargs)

    @classmethod
    def regular(
        cls,
        sides: int,
        radius,
        center=(0.0, 0.0),
        start_angle: float = 0.0,
        **kwargs,
    ) -> PolygonProfile:
        """A regular polygon inscribed in a circle of ``radius``.

        Args:
            sides: Number of sides; at least 3.
            radius: Circumscribed radius — the distance to each *vertex*, not
                to the flats.
            center: Polygon centre in plane coordinates.
            start_angle: Angle of the first vertex, in degrees from the plane's
                x axis. Use it to put a flat, rather than a corner, where a
                wrench or a mating part needs one.
            **kwargs: Passed to :class:`PolygonProfile`.

        Returns:
            The profile, wound counter-clockwise.

        Raises:
            ValueError: If ``sides`` is less than 3.
        """
        sides = int(sides)
        if sides < 3:
            raise ValueError(f"A regular profile needs at least 3 sides, got {sides}")
        radius = float(_scalar_value(radius))
        origin = jnp.asarray(center, dtype=jnp.float32)
        offset = math.radians(float(start_angle))
        vertices = [
            [
                float(origin[0]) + radius * math.cos(offset + 2.0 * math.pi * i / sides),
                float(origin[1]) + radius * math.sin(offset + 2.0 * math.pi * i / sides),
            ]
            for i in range(sides)
        ]
        kwargs.setdefault("free", False)
        return cls(vertices, **kwargs)

    @classmethod
    def rounded_rect(
        cls,
        width,
        height,
        radius,
        center=(0.0, 0.0),
        segments: int = 6,
        **kwargs,
    ) -> PolygonProfile:
        """A rectangle with rounded corners — the flange outline of most parts.

        This is the honest way to get a rounded corner *in a profile*: the
        corner is traced as polygon vertices before the solid exists, so the
        round survives extrusion exactly and costs nothing at evaluation time.
        It is unrelated to the smooth-boolean blends used as fillets *between*
        solids, which round an intersection rather than an edge.

        Args:
            width: Full width along the plane's x axis.
            height: Full height along the plane's y axis.
            radius: Corner radius; clamped to half the shorter side.
            center: Rectangle centre in plane coordinates.
            segments: Edges per corner arc.
            **kwargs: Passed to :class:`PolygonProfile`.

        Returns:
            The profile, wound counter-clockwise.

        Raises:
            ValueError: If ``width`` or ``height`` is not positive.
        """
        width = float(_scalar_value(width))
        height = float(_scalar_value(height))
        radius = float(_scalar_value(radius))
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"rounded_rect needs a positive size, got {width} x {height}")
        radius = max(0.0, min(radius, min(width, height) / 2.0))
        segments = max(1, int(segments))
        cx, cy = (float(v) for v in jnp.asarray(center, dtype=jnp.float32))
        half_w, half_h = width / 2.0 - radius, height / 2.0 - radius
        # One arc per corner, walked counter-clockwise from the +x/+y corner.
        corners = ((half_w, half_h, 0.0), (-half_w, half_h, 90.0), (-half_w, -half_h, 180.0))
        corners += ((half_w, -half_h, 270.0),)
        # A zero radius degenerates every arc to one repeated point, and a
        # zero-length edge is a divide-by-zero in the polygon distance. Emit
        # the plain rectangle instead of a corner's worth of duplicates.
        steps = range(segments + 1) if radius > 0.0 else range(1)
        vertices: list[list[float]] = []
        for ox, oy, start in corners:
            for step in steps:
                angle = math.radians(start) + (math.pi / 2.0) * step / segments
                vertices.append(
                    [cx + ox + radius * math.cos(angle), cy + oy + radius * math.sin(angle)]
                )
        kwargs.setdefault("free", False)
        return cls(vertices, **kwargs)

    def children(self) -> list[Fluent]:
        return [self.plane]

    def vertex_array(self) -> Array:
        """Current vertex values stacked as (N, 2) in plane coordinates."""
        return jnp.stack([v.value for v in self.vertices])

    def world_vertices(self) -> Array:
        """Current vertex positions in world space, shape (N, 3)."""
        return self.plane.to_world(self.vertex_array())

    def edges_world(self) -> Array:
        """Closed edge loop in world space, shape (N+1, 3) — for overlay drawing."""
        w = self.world_vertices()
        return jnp.concatenate([w, w[:1]], axis=0)
