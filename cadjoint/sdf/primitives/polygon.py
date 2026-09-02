"""Polygon-profile SDF primitives generated from 2D sketches.

These are the SDF-side targets of the construction layer: a 2D polygon profile
(ordered vertices in sketch-plane coordinates) turned into a solid by extrusion
or revolution. Vertex parameters are shared with the construction tree, so
constraints and gradients on sketch vertices flow through the generated solid.

The 2D polygon distance is exact for simple (non-self-intersecting) polygons of
either winding, and differentiable almost everywhere with respect to both the
query point and the vertices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from cadjoint.geometry.parameters import Scalar, Vector2
from cadjoint.sdf.primitives.base import Primitive


def _polygon_distance(p: Array, vertices: list[Array]) -> Array:
    """Polygon distance over a Python list of ``(2,)`` vertices.

    Keeping the vertices in a list rather than one ``(N, 2)`` array matters for
    shader compilation: every traced value stays a 2-vector, so the StableHLO →
    WGSL backend only ever sees ``vec2`` math instead of multi-dimensional
    slices it cannot lower.
    """
    num = len(vertices)
    d = jnp.sum((p - vertices[0]) ** 2, axis=-1)
    s = jnp.ones(p.shape[:-1])
    for i in range(num):
        j = (i + num - 1) % num
        e = vertices[j] - vertices[i]
        w = p - vertices[i]
        t = jnp.clip(jnp.sum(w * e, axis=-1) / jnp.sum(e * e), 0.0, 1.0)
        b = w - e * t[..., None]
        d = jnp.minimum(d, jnp.sum(b * b, axis=-1))
        # Even-odd crossing test; flips sign once per boundary crossing. Written
        # as pairwise equality (rather than negating each term) to stay within
        # the operations the shader backends implement.
        c1 = p[..., 1] >= vertices[i][1]
        c2 = p[..., 1] < vertices[j][1]
        c3 = e[0] * w[..., 1] > e[1] * w[..., 0]
        flip = (c1 == c2) & (c2 == c3)
        s = jnp.where(flip, -s, s)
    # Double-where sqrt guard: an epsilon inside the sqrt would zero the
    # derivative for points on (or within ~1e-9 of) the boundary — exactly
    # where surface sampling evaluates.  The guarded branch keeps the exact
    # gradient b/|b| of the active segment; truly on-boundary points fall to
    # the constant branch and are masked out downstream.
    positive = d > 1e-18
    safe = jnp.where(positive, d, 1.0)
    return s * jnp.where(positive, jnp.sqrt(safe), 0.0)


def polygon_sdf_2d(p: Array, vertices: Array) -> Array:
    """Exact signed distance from 2D point(s) to a simple polygon.

    Args:
        p: Query point(s) in profile coordinates, shape (..., 2).
        vertices: Ordered polygon vertices, shape (N, 2). Either winding.

    Returns:
        Signed distance, shape (...). Negative inside.
    """
    return _polygon_distance(p, [vertices[i] for i in range(vertices.shape[0])])


def _ordered_vertices(vertices: dict[str, Array]) -> list[Array]:
    """Order v0..v{N-1} keyword params by index, without stacking them."""
    return [vertices[name] for name in sorted(vertices, key=lambda name: int(name[1:]))]


def _profile_vertex_values(params: dict) -> list[Array]:
    """Concrete v0..v{N-1} vertex values from a primitive's params dict."""
    names = sorted(
        (name for name in params if name.startswith("v") and name[1:].isdigit()),
        key=lambda name: int(name[1:]),
    )
    return [params[name].value for name in names]


def _profile_orientation(verts) -> float | Array:
    """Winding of a closed profile: ``+1.0`` counter-clockwise, ``-1.0`` clockwise.

    Read concretely when the vertices are plain numbers, so the sign is a
    Python float fixed at construction. Under a tracer — a profile built
    inside a jitted derived plane — the sign is a traced ``where`` instead:
    piecewise constant, so it contributes no gradient, but it lets the
    construction trace rather than fail on a ``float()`` of a tracer.
    """
    points = jnp.stack([jnp.asarray(v, dtype=jnp.float32).reshape(2) for v in verts])
    rolled = jnp.roll(points, -1, axis=0)
    twice_area = jnp.sum(points[:, 0] * rolled[:, 1] - rolled[:, 0] * points[:, 1])
    try:
        return 1.0 if float(twice_area) >= 0.0 else -1.0
    except jax.errors.ConcretizationTypeError:
        return jnp.where(twice_area >= 0.0, 1.0, -1.0)


def _edge_half_plane_fields(verts: list[Array], orientation: float | None = None):
    """Outward half-plane distance fields, one per profile edge.

    Edge ``k`` runs from vertex ``k`` to vertex ``(k+1) % N``; its field is
    the signed distance to the edge's supporting line, positive on the
    polygon's outside for either input winding.  On a convex profile the 2D
    polygon distance is exactly ``max_k`` of these fields; in general each
    field still agrees with the exact distance on its own edge's patch, so
    ``argmin |f_k|`` identifies the owning edge there.

    Every step here is a JAX expression of ``verts``, so the returned fields
    stay differentiable in the profile vertices — provided ``orientation`` is
    supplied, since reading the winding off traced vertices would need a
    concrete value.

    Args:
        verts: Ordered ``(2,)`` profile vertices; may be tracers.
        orientation: The profile's winding sign from
            :func:`_profile_orientation`.  ``None`` reads it from ``verts``,
            which requires them to be concrete.

    Returns:
        List of callables mapping profile points shaped ``(..., 2)`` to
        signed distances shaped ``(...)``.
    """
    num = len(verts)
    # Positive area means counterclockwise vertices, whose outward edge
    # normal is the edge direction rotated by -90 degrees.
    orient = _profile_orientation(verts) if orientation is None else orientation

    fields = []
    for i in range(num):
        a, b = verts[i], verts[(i + 1) % num]
        direction = b - a
        length = jnp.sqrt(jnp.sum(direction**2) + 1e-20)
        normal = orient * jnp.stack([direction[1], -direction[0]]) / length

        def field(q: Array, a=a, normal=normal) -> Array:
            return jnp.sum((q - a) * normal, axis=-1)

        fields.append(field)
    return fields


def _statically_zero(value) -> bool:
    """True when ``value`` is a concrete number equal to zero.

    Used to skip draft/twist math entirely when the parameter is a known
    zero, keeping the default extrusion bit-identical to the plain form.
    Traced values (which cannot be concretized) report False so the full
    expression is emitted for them.
    """
    import jax

    try:
        return float(value) == 0.0
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return False


class ExtrudedPolygon(Primitive):
    """Solid formed by extruding a polygon profile along the local Z axis.

    The profile lies in the local XY plane; the solid spans ``±depth/2`` in Z
    (centered on the sketch plane). Use ``Translate``/``Rotate`` — or the
    construction-layer :func:`cadjoint.construction.extrude` — for placement.

    Args:
        vertices: Ordered profile vertices — Vector2 parameters (or raw (2,)
            arrays). Parameter references are preserved, so sketch constraints
            keep acting on the generated solid.
        depth: Total extrusion depth (Scalar parameter or float).
        material: Optional render material.
        draft: Draft angle in degrees. Positive values taper the walls inward
            as z increases: the profile is exact at ``z = -depth/2`` and the
            2D distance is offset by ``tan(draft)·(z + depth/2)`` above it.
            A non-zero draft makes the field a bound (gradient norm up to
            ``sec(draft)``), not an exact SDF.
        twist: Total twist in degrees over the full depth, applied by rotating
            the query's XY coordinates by an angle proportional to z (zero at
            mid-depth, ``±twist/2`` at the caps). A non-zero twist makes the
            field non-1-Lipschitz — distances can be overestimated, so
            sphere-tracing style consumers must shorten their steps.

    Both ``draft`` and ``twist`` default to 0.0 and are only added to
    ``params`` when non-zero (or given as Parameters), so plain extrusions
    keep their exact parameter layout and bit-identical field values.
    """

    def __init__(
        self,
        vertices: list[Vector2],
        depth: float | Scalar,
        material=None,
        draft: float | Scalar = 0.0,
        twist: float | Scalar = 0.0,
    ):
        from cadjoint.geometry.parameters import Parameter
        from cadjoint.render.material import Material

        if len(vertices) < 3:
            raise ValueError(f"ExtrudedPolygon needs at least 3 vertices, got {len(vertices)}")
        self.material = material if material is not None else Material()
        self.num_vertices = len(vertices)
        self.params = {f"v{i}": v for i, v in enumerate(vertices)}
        self.params["depth"] = depth
        # Read the winding once, here, from the nominal vertices: patch_fields
        # may be rebuilt with the vertices traced, and the shoelace sign is a
        # discrete reading no tracer can give.  ``params`` are still raw at
        # this point (the base class wraps them after __init__ returns).
        self._orientation = _profile_orientation([getattr(v, "value", v) for v in vertices])
        if isinstance(draft, Parameter) or not _statically_zero(draft):
            self.params["draft"] = draft
            # Drafted distances have gradient norm up to sec(draft) > 1.
            self.is_exact = False
        if isinstance(twist, Parameter) or not _statically_zero(twist):
            self.params["twist"] = twist
            # Twisted distances are not 1-Lipschitz; flag for renderers.
            self.is_exact = False

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, depth: Array, draft: Array = 0.0, twist: Array = 0.0, **vertices: Array):
        """Pure SDF: exact extrusion of the exact polygon distance.

        With the default ``draft=0``/``twist=0`` this is bit-identical to the
        plain extrusion — the extra terms are skipped for concrete zeros.

        Args:
            p: Query point(s), shape (..., 3). Profile plane is local XY.
            depth: Total extrusion depth.
            draft: Draft angle in degrees; offsets the 2D distance by
                ``tan(draft)·(z + depth/2)`` so walls taper inward with +z.
            twist: Total twist in degrees over the full depth; rotates the
                query XY by ``twist·z/depth`` (non-1-Lipschitz when non-zero).
            **vertices: v0..v{N-1} profile vertices, each shape (2,).

        Returns:
            Signed distance, shape (...).
        """
        verts = _ordered_vertices(vertices)
        xy = p[..., :2]
        z = p[..., 2]
        if not _statically_zero(twist):
            theta = jnp.deg2rad(twist) * z / depth
            c, s = jnp.cos(theta), jnp.sin(theta)
            xy = jnp.stack(
                [xy[..., 0] * c + xy[..., 1] * s, xy[..., 1] * c - xy[..., 0] * s],
                axis=-1,
            )
        d2 = _polygon_distance(xy, verts)
        if not _statically_zero(draft):
            slope = jnp.sin(jnp.deg2rad(draft)) / jnp.cos(jnp.deg2rad(draft))
            d2 = d2 + slope * (z + depth / 2.0)
        dz = jnp.abs(z) - depth / 2.0
        # Branch on inside/outside like Box.sdf so points exactly on a side
        # wall or cap receive a valid one-sided subgradient instead of the
        # exact zero an epsilon-smoothed outside norm produces there.
        max_d = jnp.maximum(d2, dz)
        squared = jnp.maximum(d2, 0.0) ** 2 + jnp.maximum(dz, 0.0) ** 2
        on_or_inside = max_d <= 0.0
        safe = jnp.where(on_or_inside, 1.0, squared)
        return jnp.where(on_or_inside, max_d, jnp.sqrt(safe))

    def __call__(self, p: Array) -> Array:
        values = {k: v.value for k, v in self.params.items() if k != "depth"}
        return ExtrudedPolygon.sdf(p, self.params["depth"].value, **values)

    def to_functional(self):
        return ExtrudedPolygon.sdf

    def patch_fields(self):
        """Per-edge wall half-planes plus the two caps.

        Order: patch ``k`` (for ``k < N``) is the wall extruded from profile
        edge ``(v_k, v_{k+1})``; patch ``N`` is the bottom cap
        (``z = -depth/2``), patch ``N+1`` the top cap.  Drafted or twisted
        extrusions have curved/tapered walls that are no longer half-planes,
        so they report ``None``.
        """
        if "draft" in self.params or "twist" in self.params:
            return None
        depth = self.params["depth"].value
        edge_fields = _edge_half_plane_fields(
            _profile_vertex_values(self.params), self._orientation
        )
        walls = [(lambda p, f=field: f(p[..., :2])) for field in edge_fields]
        caps = [
            lambda p: -p[..., 2] - depth / 2.0,
            lambda p: p[..., 2] - depth / 2.0,
        ]
        return walls + caps


class RevolvedPolygon(Primitive):
    """Solid of revolution: a polygon profile revolved around the local Y axis.

    Profile coordinates are (radial, height): the profile's X coordinate is the
    distance from the revolution axis (plus ``offset``), and its Y coordinate
    runs along the axis. The profile should stay at positive radius
    (``x + offset > 0``) for the distance to remain exact.

    Args:
        vertices: Ordered profile vertices — Vector2 parameters (or raw (2,)
            arrays), in (radial, height) coordinates.
        offset: Radial offset added before revolving (Scalar parameter or float).
        material: Optional render material.
    """

    def __init__(self, vertices: list[Vector2], offset: float | Scalar = 0.0, material=None):
        from cadjoint.render.material import Material

        if len(vertices) < 3:
            raise ValueError(f"RevolvedPolygon needs at least 3 vertices, got {len(vertices)}")
        self.material = material if material is not None else Material()
        self.num_vertices = len(vertices)
        self.params = {f"v{i}": v for i, v in enumerate(vertices)}
        self.params["offset"] = offset
        # The winding, read once from the nominal vertices (see
        # ExtrudedPolygon.__init__).
        self._orientation = _profile_orientation([getattr(v, "value", v) for v in vertices])

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, offset: Array, **vertices: Array) -> Array:
        """Pure SDF: revolve the polygon profile around the local Y axis.

        Args:
            p: Query point(s), shape (..., 3).
            offset: Radial offset of the profile.
            **vertices: v0..v{N-1} profile vertices, each shape (2,).

        Returns:
            Signed distance, shape (...).
        """
        verts = _ordered_vertices(vertices)
        radial = jnp.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2 + 1e-20) - offset
        q = jnp.stack([radial, p[..., 1]], axis=-1)
        return _polygon_distance(q, verts)

    def __call__(self, p: Array) -> Array:
        values = {k: v.value for k, v in self.params.items() if k != "offset"}
        return RevolvedPolygon.sdf(p, self.params["offset"].value, **values)

    def to_functional(self):
        return RevolvedPolygon.sdf

    def patch_fields(self):
        """One field per profile edge, in revolved (radial, height) coords.

        Patch ``k`` is the surface swept by profile edge ``(v_k, v_{k+1})``:
        the edge's outward half-plane distance evaluated at
        ``q = (sqrt(x² + z²) - offset, y)``, exactly as :meth:`sdf` maps
        queries into profile coordinates.  The profile is closed, so there
        are no separate cap fields.
        """
        offset = self.params["offset"].value
        edge_fields = _edge_half_plane_fields(
            _profile_vertex_values(self.params), self._orientation
        )

        def revolved(p: Array) -> Array:
            radial = jnp.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2 + 1e-20) - offset
            return jnp.stack([radial, p[..., 1]], axis=-1)

        return [(lambda p, f=field: f(revolved(p))) for field in edge_fields]
