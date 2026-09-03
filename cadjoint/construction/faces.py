"""Analytic face references on construction features.

cadjoint is an implicit modeller: the only mesh it has is the dual-contouring
render of the surface, which is a *picture* of the geometry and must never be
allowed to define a plane.  Two things the construction tree knows instead:

1. **A feature knows its planar faces exactly.**
   :func:`~cadjoint.construction.extrude` spans ``±depth/2`` around its sketch
   plane, so its caps are ``plane.origin ± (depth/2)·plane.normal`` with the
   profile's own ``u``/``v`` as in-plane axes, and every polygon edge sweeps a
   planar side wall whose normal is ``edge_direction × plane.normal``.  A box
   knows its six faces; a revolve knows its axis.
2. **Those planes are re-derived, not stored.**  ``depth`` may be a ``Scalar``,
   so a face declared on a cap is computed from the feature's parameters every
   time the program runs: a sketch placed on it moves when the parent is
   re-dimensioned, which is what a B-rep's stored surface cannot do.  The link
   is a *rebuild* link and not a differentiable one — the derived origin is
   snapshotted into a fixed ``Parameter``, so ``jax.grad`` through a child
   solid w.r.t. the parent's ``depth`` is zero.  See
   ``research/complex-scene.md``.

A :class:`Face` carries its plane (origin, normal **and** in-plane x axis — the
sketch's "horizontal" is a choice, not an implementation accident), a
world-space boundary polygon that both bounds the face and draws its hover
highlight, and a :meth:`Face.describe` payload for the viewer.

Faces are built lazily: :class:`FaceSet` holds a builder and evaluates it the
first time a face is asked for, so every existing ``extrude()`` call keeps its
old cost.  Only exact features declare faces — a drafted or twisted extrusion
has no planar side walls, so it declares none and callers fall back to
:meth:`~cadjoint.construction.sketch.SketchPlane.tangent`, which reads the
plane straight off the SDF's gradient.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Callable

import jax.numpy as jnp
from jax import Array

# Below this squared length a direction carries no orientation; the guarded
# norm keeps both the value and its derivative finite instead of 0/0.
_MIN_SQUARED = 1e-12

# Absolute floor for a face's containment tolerance, so a tiny face still
# accepts a raymarch hit that landed a float epsilon off its plane.
_MIN_TOLERANCE = 1e-5

# Fraction of a face's own diagonal used as its default hit tolerance.
_TOLERANCE_SCALE = 1e-3


def _unit(vector) -> Array:
    """Normalize a 3-vector with a guarded norm, safe under tracing."""
    vector = jnp.asarray(vector)
    return vector / jnp.sqrt(jnp.maximum(jnp.sum(vector * vector), _MIN_SQUARED))


def _orthogonalize(x_axis, normal: Array) -> Array:
    """Component of ``x_axis`` inside the plane of ``normal``, normalized."""
    x_axis = jnp.asarray(x_axis)
    return _unit(x_axis - jnp.sum(x_axis * normal) * normal)


def _scalar(value) -> Array:
    """Current value of a number that may be a ``Scalar`` parameter."""
    from cadjoint.geometry.parameters import Parameter

    return jnp.asarray(value.value if isinstance(value, Parameter) else value)


def _statically_zero(value) -> bool:
    """True when ``value`` is a concrete number equal to zero.

    A traced value cannot be concretized and reports False, which is the
    conservative answer: the feature then declares no analytic faces.
    """
    import jax

    from cadjoint.geometry.parameters import Parameter

    if isinstance(value, Parameter):
        value = value.value
    try:
        return float(value) == 0.0
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return False


class Face:
    """One analytic planar face of a construction feature.

    A face is a *reference*, not stored geometry: its origin and boundary are
    recomputed from the feature's parameters every time the feature is built,
    so a sketch placed on it follows the parent and gradients flow through it.

    Args:
        kind: ``cap`` (an extrusion or loft end), ``side`` (a swept polygon
            edge), or ``planar`` (a primitive's flat face).
        key: Identity of the face within its owner — ``cap+``, ``side3``,
            ``+x``. Stable across rebuilds, so the viewer can address it.
        origin: A point on the face, in world space.
        normal: Outward face normal in world space; normalized on construction.
        x_axis: The face's in-plane "horizontal"; orthogonalized against the
            normal so the frame is orthonormal whatever the caller passes.
        boundary: The face's boundary loop in world space, shape ``(M, 3)``.
        owner: The feature or primitive that declared the face.
        reference: ``(method, args)`` naming the accessor that returns this
            face — ``("cap", ("+",))`` — which is what the viewer writes back
            into the user's source.
    """

    def __init__(
        self,
        kind: str,
        key: str,
        origin,
        normal,
        x_axis,
        boundary,
        *,
        owner=None,
        reference: tuple[str, tuple] | None = None,
    ):
        self.kind = kind
        self.key = key
        self.origin = jnp.asarray(origin)
        self.normal = _unit(normal)
        self.x_axis = _orthogonalize(x_axis, self.normal)
        self.boundary = jnp.asarray(boundary)
        self.owner = owner
        self.reference = reference if reference is not None else ("face", (key,))

    # ── frame ────────────────────────────────────────────────────────────────

    @property
    def y_axis(self) -> Array:
        """The in-plane axis completing a right-handed frame with the normal."""
        return jnp.cross(self.normal, self.x_axis)

    def frame(self) -> tuple[Array, Array, Array]:
        """Return the orthonormal ``(x, y, normal)`` frame of the face."""
        return self.x_axis, self.y_axis, self.normal

    def to_local(self, points) -> Array:
        """Project world points ``(..., 3)`` onto the face's ``(..., 2)`` axes."""
        delta = jnp.asarray(points) - self.origin
        return jnp.stack(
            [jnp.sum(delta * self.x_axis, axis=-1), jnp.sum(delta * self.y_axis, axis=-1)],
            axis=-1,
        )

    def polygon(self) -> Array:
        """The boundary loop in face coordinates, shape ``(M, 2)``."""
        return self.to_local(self.boundary)

    def diagonal(self) -> Array:
        """Diagonal of the face's bounding box — the scale a tolerance is read against."""
        local = self.polygon()
        extent = jnp.max(local, axis=0) - jnp.min(local, axis=0)
        return jnp.sqrt(jnp.sum(extent * extent))

    def tolerance(self) -> float:
        """Default hit tolerance for :meth:`contains`, scaled to the face."""
        return max(_MIN_TOLERANCE, _TOLERANCE_SCALE * float(self.diagonal()))

    # ── extent ───────────────────────────────────────────────────────────────

    def contains(self, point, tol: float | None = None) -> Array:
        """Whether a world point lies on this face, within ``tol``.

        Two tests, both inclusive of the tolerance: the point must sit within
        ``tol`` of the face's plane, and its in-plane projection must lie
        inside the boundary polygon (again allowing ``tol`` of slack, so a hit
        exactly on a shared edge belongs to both faces rather than neither).

        Args:
            point: World point(s), shape ``(..., 3)``.
            tol: Distance tolerance; defaults to :meth:`tolerance`.

        Returns:
            Boolean array shaped like ``point`` without its last axis.
        """
        from cadjoint.sdf.primitives.polygon import polygon_sdf_2d

        limit = self.tolerance() if tol is None else float(tol)
        point = jnp.asarray(point)
        offset = jnp.sum((point - self.origin) * self.normal, axis=-1)
        local = self.to_local(point)
        inside = polygon_sdf_2d(local, self.polygon()) <= limit
        return (jnp.abs(offset) <= limit) & inside

    # ── derived geometry ─────────────────────────────────────────────────────

    def center(self) -> Array:
        """The centroid of the face's boundary loop, in world space.

        The natural anchor for a feature placed "on the middle of this face" —
        a boss, a bore, a bolt circle's axis — and, being an average of the
        boundary, an expression in the parent's parameters like everything else
        on the face.

        Returns:
            The centroid, shape ``(3,)``.
        """
        return jnp.mean(self.boundary, axis=0)

    def point(self, at=(0.0, 0.0)) -> Array:
        """A world point at face-local coordinates ``at``, measured from the origin.

        Args:
            at: ``(x, y)`` in the face's own frame — ``x`` along
                :attr:`x_axis`, ``y`` along :attr:`y_axis`.

        Returns:
            The world point, shape ``(3,)``.
        """
        at = jnp.asarray(at)
        return self.origin + at[..., 0] * self.x_axis + at[..., 1] * self.y_axis

    def plane(self, x_axis=None, flip: bool = False, offset=0.0):
        """This face as a sketch plane, optionally pushed along its normal.

        Sugar for :meth:`~cadjoint.construction.sketch.SketchPlane.on` — and,
        with an ``offset``, for
        :meth:`~cadjoint.construction.sketch.SketchPlane.offset` — so a face
        reference reads as one phrase where it is used.

        Args:
            x_axis: Override the sketch's in-plane horizontal.
            flip: Face the plane the other way, keeping the same origin.
            offset: Distance to push along the (possibly flipped) normal. May
                be a ``Scalar`` parameter.

        Returns:
            The :class:`~cadjoint.construction.sketch.SketchPlane`.
        """
        from cadjoint.construction.sketch import SketchPlane

        plane = SketchPlane.on(self, x_axis=x_axis, flip=flip)
        if _statically_zero(offset):
            return plane
        return SketchPlane.offset(plane, offset)

    def hole(self, radius, depth, at=(0.0, 0.0), *, through: float = 0.0, material=None):
        """A cylindrical tool sunk into this face — **subtract** it to cut a hole.

        Returns the tool rather than a modified solid, because in an implicit
        modeller the cut *is* the boolean: keeping them separate is what lets
        one tool be patterned, mirrored, or subtracted from several bodies at
        once, and what keeps the hole visible in the feature tree.

        The tool is a true :class:`~cadjoint.sdf.primitives.cylinder.Cylinder`
        aligned with the face normal, not a polygonal approximation, so the
        bore stays round at every render scale and its radius stays a live
        design parameter.

        Args:
            radius: Bore radius; may be a ``Scalar`` parameter.
            depth: How far the tool reaches below the face, along ``-normal``.
                May be a ``Scalar``.
            at: Where the axis meets the face, in face-local ``(x, y)``.
            through: Extra length added *above* the face. Zero leaves the tool
                exactly flush, which is what a blind hole wants; a small
                positive value keeps a through-hole's mouth off the surface it
                is cutting, where a perfectly coincident pair of surfaces
                would otherwise leave the mesher to break the tie.
            material: Optional render material for the tool.

        Returns:
            The tool as an SDF, placed in world space.

        Example:
            ```python
            bore = flange.cap("+").hole(bore_radius, depth=0.4, through=0.02)
            housing = Difference(flange, bore)
            ```
        """
        from cadjoint.sdf.primitives import Cylinder

        length = _scalar(depth) + _scalar(through)
        tool = Cylinder(radius=radius, height=length / 2.0, material=material)
        return self._sink(tool, depth, at, through)

    def pocket(self, vertices, depth, at=(0.0, 0.0), *, through: float = 0.0, material=None):
        """A prismatic tool sunk into this face — **subtract** it to cut a pocket.

        The profile is read in the face's own frame, so a pocket is drawn in
        the same coordinates the face's other features use.

        Args:
            vertices: The pocket outline as face-local ``(x, y)`` points — a
                list, or a :class:`PolygonProfile`'s vertices.
            depth: How far the tool reaches below the face, along ``-normal``.
            at: Offset applied to the whole outline, in face-local ``(x, y)``.
            through: Extra length added above the face; see :meth:`hole`.
            material: Optional render material for the tool.

        Returns:
            The tool as an SDF, placed in world space.
        """
        from cadjoint.sdf.primitives.polygon import ExtrudedPolygon

        length = _scalar(depth) + _scalar(through)
        tool = ExtrudedPolygon(vertices, depth=length, material=material)
        return self._sink(tool, depth, at, through)

    def _sink(self, tool, depth, at, through):
        """Place a centred local-frame tool so it spans this face's cut.

        The tool arrives centred on its own origin and already the right
        length; all that is left is to sit its centre halfway between the two
        ends of the cut and turn its local z onto the face normal.
        """
        from cadjoint.construction.extrude import _place_on_plane
        from cadjoint.construction.sketch import SketchPlane

        middle = self.point(at) + (_scalar(through) - _scalar(depth)) / 2.0 * self.normal
        plane = SketchPlane(origin=middle, normal=self.normal, x_axis=self.x_axis)
        return _place_on_plane(tool, plane)

    # ── payload ──────────────────────────────────────────────────────────────

    def describe(self) -> dict:
        """Serialize the face for the viewer.

        Returns:
            A JSON-ready dict carrying the face's identity, its plane frame,
                the world-space boundary polygon the viewer highlights, the
                tolerance a hit test should use, and the accessor call that
                reproduces the face in source.
        """
        method, arguments = self.reference
        return {
            "key": self.key,
            "kind": self.kind,
            "origin": [float(x) for x in self.origin],
            "normal": [float(x) for x in self.normal],
            "xAxis": [float(x) for x in self.x_axis],
            "yAxis": [float(x) for x in self.y_axis],
            "polygon": [[float(x) for x in point] for point in self.boundary],
            "tolerance": self.tolerance(),
            "reference": {"call": method, "args": list(arguments)},
        }

    def __repr__(self) -> str:
        return f"Face({self.kind}:{self.key})"


class Axis:
    """A feature's axis of revolution, exposed as a reusable reference.

    A revolve has no planar faces to speak of — its surface is curved almost
    everywhere — but it does know the line it was swept around, which is what
    a downstream sketch, a mirror, or a circular pattern actually needs.

    Args:
        origin: A point on the axis, in world space.
        direction: Axis direction; normalized on construction.
        owner: The feature that declared the axis.
    """

    def __init__(self, origin, direction, *, owner=None):
        self.origin = jnp.asarray(origin)
        self.direction = _unit(direction)
        self.owner = owner

    def point(self, distance) -> Array:
        """A point ``distance`` along the axis from its origin."""
        return self.origin + jnp.asarray(distance) * self.direction

    def describe(self) -> dict:
        """Serialize the axis for the viewer."""
        return {
            "kind": "axis",
            "origin": [float(x) for x in self.origin],
            "direction": [float(x) for x in self.direction],
        }

    def __repr__(self) -> str:
        return f"Axis(origin={list(self.origin)}, direction={list(self.direction)})"


class FaceSet:
    """The analytic faces of one feature, built on first use.

    Args:
        owner_kind: The feature that owns the faces — ``extrude``, ``loft``,
            ``box`` and so on. Reported in the payload so the viewer can say
            what a highlighted face belongs to.
        builder: Callable returning the face list. Deferred so declaring faces
            costs nothing until something asks for one.
    """

    def __init__(self, owner_kind: str, builder: Callable[[], Sequence[Face]]):
        self.owner_kind = owner_kind
        self._builder = builder
        self._faces: list[Face] | None = None

    def all(self) -> list[Face]:
        """Every face this feature declares, in a stable order."""
        if self._faces is None:
            self._faces = list(self._builder())
        return self._faces

    def keys(self) -> list[str]:
        """The key of every declared face."""
        return [face.key for face in self.all()]

    def __iter__(self) -> Iterator[Face]:
        return iter(self.all())

    def __len__(self) -> int:
        return len(self.all())

    def face(self, key: str) -> Face:
        """The face with this key.

        Args:
            key: A key from :meth:`keys` — ``cap+``, ``side2``, ``+x``.

        Returns:
            The matching :class:`Face`.

        Raises:
            KeyError: If no face carries that key.
        """
        for face in self.all():
            if face.key == key:
                return face
        raise KeyError(f"{self.owner_kind} has no face {key!r}; declared: {self.keys()}")

    def cap(self, sign: str = "+") -> Face:
        """The cap at the ``+`` or ``-`` end of the feature's sweep.

        Args:
            sign: ``"+"``/``"top"``/``1`` for the cap at ``+depth/2``, and
                ``"-"``/``"bottom"``/``-1`` for the one at ``-depth/2``.

        Returns:
            The matching :class:`Face`.

        Raises:
            KeyError: If the feature declares no cap on that side.
            ValueError: If ``sign`` names neither end.
        """
        return self.face(f"cap{_cap_sign(sign)}")

    def side(self, index: int) -> Face:
        """The side wall swept by profile edge ``index``.

        Args:
            index: Zero-based edge index; edge ``i`` runs from profile vertex
                ``i`` to vertex ``i + 1`` (wrapping at the end).

        Returns:
            The matching :class:`Face`.

        Raises:
            KeyError: If the feature declares no such side wall.
        """
        return self.face(f"side{int(index)}")

    def describe(self) -> list[dict]:
        """Serialize every declared face for the viewer."""
        return [face.describe() for face in self.all()]

    def __repr__(self) -> str:
        return f"FaceSet({self.owner_kind}, {self.keys()})"


def _cap_sign(sign) -> str:
    """Normalize the many ways a caller names an end of a sweep to ``+``/``-``."""
    if isinstance(sign, str):
        lowered = sign.strip().lower()
        if lowered in {"+", "top", "up", "plus", "end"}:
            return "+"
        if lowered in {"-", "bottom", "down", "minus", "start"}:
            return "-"
    elif isinstance(sign, (int, float)) and not isinstance(sign, bool):
        return "+" if sign >= 0 else "-"
    raise ValueError(f"A cap is named '+' or '-', got {sign!r}.")


class Feature:
    """The record a generator leaves on the profiles it consumed.

    ``extrude(profile, depth)`` returns an SDF, and the SDF is the thing the
    renderer wants — but the *viewer* addresses geometry by the sketch that
    produced it.  Registering the feature on its profile is what lets the
    construction payload list a sketch's faces without the payload builder
    ever seeing the SDF tree.

    Args:
        kind: ``extrude``, ``revolve``, or ``loft``.
        faces: The feature's analytic faces.
        axis: The feature's axis, for revolves.
        solid: The generated SDF, for callers that want to walk back to it.
    """

    def __init__(self, kind: str, faces: FaceSet, *, axis: Axis | None = None, solid=None):
        self.kind = kind
        self.faces = faces
        self.axis = axis
        self.solid = solid

    def __repr__(self) -> str:
        return f"Feature({self.kind}, {self.faces.keys()})"


def register_feature(profile, feature: Feature) -> Feature:
    """Record a feature on the profile it was generated from.

    Args:
        profile: The :class:`~cadjoint.construction.sketch.PolygonProfile`.
        feature: The feature to record.

    Returns:
        The feature, so callers can chain.
    """
    features = getattr(profile, "features", None)
    if features is None:
        features = []
        profile.features = features
    features.append(feature)
    return feature


def attach_faces(solid, faces: FaceSet, axis: Axis | None = None):
    """Give a generated SDF its face accessors.

    The SDF tree is deliberately free of CAD semantics, so the accessors are
    bound onto the returned instance rather than added to
    :class:`~cadjoint.sdf.base.SDF`: ``extrude(...)`` still returns a plain
    SDF, and ``solid.cap("+")`` still reads like the feature API it is.

    Args:
        solid: The SDF returned by a generator.
        faces: The feature's face set.
        axis: The feature's axis, when it has one.

    Returns:
        The same SDF, with ``faces``, ``face``, ``cap``, ``side`` and — when
            given — ``axis`` bound on it.
    """
    solid.faces = faces
    solid.face = faces.face
    solid.cap = faces.cap
    solid.side = faces.side
    if axis is not None:
        solid.axis = axis
    return solid


# ── feature face builders ────────────────────────────────────────────────────


def extrusion_faces(profile, depth, *, draft=0.0, twist=0.0) -> FaceSet:
    """The analytic faces of an extrusion: two caps and one wall per edge.

    The extrusion spans ``±depth/2`` around the profile's sketch plane, so the
    caps sit at ``origin ± (depth/2)·normal`` carrying the profile's own
    ``u``/``v`` as their in-plane axes, and edge ``i`` sweeps the rectangle
    between vertices ``i`` and ``i + 1`` at both depths.

    A drafted or twisted extrusion declares **no** faces: draft tapers the
    walls off their swept planes and twist curves them outright, so nothing
    here would be exact. Use
    :meth:`~cadjoint.construction.sketch.SketchPlane.tangent` on those.

    Args:
        profile: The extruded :class:`~cadjoint.construction.sketch.PolygonProfile`.
        depth: Total extrusion depth; may be a ``Scalar`` parameter.
        draft: Draft angle in degrees, as passed to the generator.
        twist: Total twist in degrees, as passed to the generator.

    Returns:
        A lazily built :class:`FaceSet`.
    """
    exact = _statically_zero(draft) and _statically_zero(twist)

    def build() -> list[Face]:
        if not exact:
            return []
        plane = profile.plane
        u, v, normal = plane.frame()
        origin = plane.origin.xyz
        half = _scalar(depth) / 2.0
        vertices = profile.vertex_array()
        world = plane.to_world(vertices)
        faces = [
            Face(
                "cap",
                "cap+",
                origin + half * normal,
                normal,
                u,
                world + half * normal,
                reference=("cap", ("+",)),
            ),
            Face(
                "cap",
                "cap-",
                origin - half * normal,
                -normal,
                u,
                world - half * normal,
                reference=("cap", ("-",)),
            ),
        ]
        faces.extend(_swept_walls(world, normal, half, -half))
        return faces

    return FaceSet("extrude", build)


def loft_faces(profile_a, profile_b, height) -> FaceSet:
    """The two planar ends of a loft.

    The loft is placed on ``profile_a``'s plane with profile A at
    ``-height/2`` and profile B at ``+height/2``; both ends are planar, so both
    are exact references. The ruled side walls between them are planar only by
    accident and are deliberately not declared.

    Args:
        profile_a: The base profile, whose plane places the solid.
        profile_b: The top profile; only its 2D coordinates are used.
        height: Total loft height; may be a ``Scalar`` parameter.

    Returns:
        A lazily built :class:`FaceSet` with ``cap-`` (profile A) and ``cap+``
            (profile B).
    """

    def build() -> list[Face]:
        plane = profile_a.plane
        u, _, normal = plane.frame()
        origin = plane.origin.xyz
        half = _scalar(height) / 2.0
        top = plane.to_world(profile_b.vertex_array()) + half * normal
        bottom = plane.to_world(profile_a.vertex_array()) - half * normal
        return [
            Face(
                "cap",
                "cap+",
                origin + half * normal,
                normal,
                u,
                top,
                reference=("cap", ("+",)),
            ),
            Face(
                "cap",
                "cap-",
                origin - half * normal,
                -normal,
                u,
                bottom,
                reference=("cap", ("-",)),
            ),
        ]

    return FaceSet("loft", build)


def revolve_axis(profile) -> Axis:
    """The axis a profile is revolved around.

    :func:`~cadjoint.construction.revolve` sweeps the profile about the sketch
    plane's local Y axis, so the world axis is the plane's ``v`` through its
    origin.

    Args:
        profile: The revolved profile.

    Returns:
        The world-space :class:`Axis`.
    """
    _, v, _ = profile.plane.frame()
    return Axis(profile.plane.origin.xyz, v)


def _swept_walls(world: Array, normal: Array, high, low) -> list[Face]:
    """Side walls swept by each edge of a closed world-space profile loop."""
    count = int(world.shape[0])
    centroid = jnp.mean(world, axis=0)
    walls: list[Face] = []
    for index in range(count):
        start = world[index]
        end = world[(index + 1) % count]
        direction = _unit(end - start)
        raw = jnp.cross(direction, normal)
        # Orient outward whatever the profile's winding: the wall's normal
        # must point away from the profile's centroid, and a `where` keeps
        # the choice traceable when the vertices are parameters.
        middle = (start + end) / 2.0
        outward = jnp.where(jnp.sum(raw * (middle - centroid)) < 0.0, -1.0, 1.0)
        walls.append(
            Face(
                "side",
                f"side{index}",
                middle + (high + low) / 2.0 * normal,
                raw * outward,
                direction,
                jnp.stack(
                    [
                        start + low * normal,
                        end + low * normal,
                        end + high * normal,
                        start + high * normal,
                    ]
                ),
                reference=("side", (index,)),
            )
        )
    return walls


def primitive_faces(primitive) -> FaceSet:
    """The analytic faces of a construction primitive.

    A box declares its six faces keyed ``+x``/``-x``/``+y``/``-y``/``+z``/``-z``
    in its own rotated frame; a cylinder declares its two circular caps as
    ``cap+``/``cap-``; a sphere declares none — it has no planar face, and
    :meth:`~cadjoint.construction.sketch.SketchPlane.tangent` is the reference
    to use on it.

    Args:
        primitive: A :class:`~cadjoint.construction.solid.ConstructionPrimitive`.

    Returns:
        A lazily built :class:`FaceSet`.
    """

    def build() -> list[Face]:
        from cadjoint.construction.solid import _rotation_matrix

        matrix = _rotation_matrix(*primitive.rotation_values())
        center = primitive.position.xyz
        if primitive.kind == "box":
            return _box_faces(matrix, center, primitive.params["size"].xyz)
        if primitive.kind == "cylinder":
            return _cylinder_faces(
                matrix,
                center,
                primitive.params["radius"].value,
                primitive.params["height"].value,
            )
        return []

    return FaceSet(primitive.kind, build)


# Corner signs of one box face, traced as a loop in the face's own (x, y).
_QUAD = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))

# Points around a circular cap's boundary polygon — dense enough that the
# containment test and the hover highlight both read as a circle.
_CIRCLE_SEGMENTS = 48


def _box_faces(matrix: Array, center: Array, size: Array) -> list[Face]:
    """The six faces of a rotated, centred box."""
    faces = []
    for axis in range(3):
        first, second = (axis + 1) % 3, (axis + 2) % 3
        world_normal = matrix[:, axis]
        world_x = matrix[:, first]
        world_y = matrix[:, second]
        for sign in (1.0, -1.0):
            normal = world_normal * sign
            origin = center + normal * size[axis]
            # Flip the in-plane y with the face so every face's frame stays
            # right-handed against its own outward normal.
            corners = jnp.stack(
                [
                    origin + a * size[first] * world_x + sign * b * size[second] * world_y
                    for a, b in _QUAD
                ]
            )
            key = f"{'+' if sign > 0 else '-'}{'xyz'[axis]}"
            faces.append(
                Face("planar", key, origin, normal, world_x, corners, reference=("face", (key,)))
            )
    return faces


def _cylinder_faces(matrix: Array, center: Array, radius, height) -> list[Face]:
    """The two circular caps of a rotated cylinder (``height`` is the half height)."""
    world_normal = matrix[:, 2]
    world_x = matrix[:, 0]
    world_y = matrix[:, 1]
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, _CIRCLE_SEGMENTS, endpoint=False)
    faces = []
    for sign in (1.0, -1.0):
        normal = world_normal * sign
        origin = center + normal * height
        ring = (
            origin
            + radius * jnp.cos(angles)[:, None] * world_x
            + sign * radius * jnp.sin(angles)[:, None] * world_y
        )
        key = f"cap{'+' if sign > 0 else '-'}"
        faces.append(
            Face(
                "cap",
                key,
                origin,
                normal,
                world_x,
                ring,
                reference=("cap", ("+" if sign > 0 else "-",)),
            )
        )
    return faces
