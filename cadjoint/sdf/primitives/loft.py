"""Lofted polygon SDF primitive: linear interpolation between two profiles.

A loft connects two polygon profiles with the same vertex count by linearly
interpolating each vertex pair along the local Z axis. Every horizontal slice
of the solid is the exact 2D polygon distance to the interpolated profile, and
the slice distance is combined with the cap distance exactly like
:class:`~cadjoint.sdf.primitives.polygon.ExtrudedPolygon`.

The per-slice polygon distance is exact, but the assembled 3D field is a
*bound* on the true distance rather than an exact SDF: the side walls are
ruled surfaces, so the nearest surface point of a tapering wall generally lies
on a different slice than the query point. The sign is still correct wherever
each interpolated slice polygon stays simple (non-self-intersecting), which is
what meshing and rendering rely on.

"Ruled" is the general case rather than the only one: a wall whose four
corners happen to be coplanar *is* a plane, which is what every truncated
pyramid — two scaled copies of one profile — is made of.
:meth:`LoftedPolygon.patch_fields` measures that per wall and declares an
exact decomposition when every wall passes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cadjoint.geometry.parameters import Scalar, Vector2
from cadjoint.sdf._lowering import is_scalar_lowering
from cadjoint.sdf.primitives.base import Primitive
from cadjoint.sdf.primitives.polygon import _polygon_distance_stacked, _profile_orientation

# How far a wall's fourth corner may sit off the plane through the other
# three before the wall counts as genuinely ruled, as a fraction of the
# loft's own bounding diagonal.  1e-5 is chosen to straddle a wide gap: the
# vertex parameters are stored in float32, so a corner that *is* coplanar
# still reports a residual around 1e-7 of the part's size (the reference
# scenes measure 1e-9 to 1e-8), while a wall that really rules — profiles
# rotated against each other, or one edge flared and its neighbour not —
# has a residual on the order of the profile itself.  Two decades of slack
# above the noise, five below anything real.
_COPLANAR_TOLERANCE = 1e-5


def _lerped_polygon_distance(p: Array, vertices: list[Array]) -> Array:
    """Signed polygon distance where vertices may carry the query batch shape.

    Mirrors ``cadjoint.sdf.primitives.polygon._polygon_distance`` — including the
    double-where sqrt guard — but reduces with ``axis=-1`` and indexes vertex
    components with ``[..., k]`` so each vertex may be a plain ``(2,)`` array
    *or* broadcast against the query's batch shape (the lofted profile depends
    on the query's Z). For ``(2,)`` vertices the operations are identical to
    the extruded-polygon path.

    Args:
        p: Query point(s) in profile coordinates, shape (..., 2).
        vertices: Ordered polygon vertices, each shape (2,) or (..., 2).

    Returns:
        Signed distance, shape (...). Negative inside.
    """
    if not is_scalar_lowering():
        return _polygon_distance_stacked(p, jnp.stack(jnp.broadcast_arrays(*vertices)))
    num = len(vertices)
    d = jnp.sum((p - vertices[0]) ** 2, axis=-1)
    s = jnp.ones(p.shape[:-1])
    for i in range(num):
        j = (i + num - 1) % num
        e = vertices[j] - vertices[i]
        w = p - vertices[i]
        t = jnp.clip(jnp.sum(w * e, axis=-1) / jnp.sum(e * e, axis=-1), 0.0, 1.0)
        b = w - e * t[..., None]
        d = jnp.minimum(d, jnp.sum(b * b, axis=-1))
        c1 = p[..., 1] >= vertices[i][..., 1]
        c2 = p[..., 1] < vertices[j][..., 1]
        c3 = e[..., 0] * w[..., 1] > e[..., 1] * w[..., 0]
        flip = (c1 == c2) & (c2 == c3)
        s = jnp.where(flip, -s, s)
    positive = d > 1e-18
    safe = jnp.where(positive, d, 1.0)
    return s * jnp.where(positive, jnp.sqrt(safe), 0.0)


def _profile_values(params: dict, prefix: str, count: int) -> list[Array]:
    """The ``v0..`` or ``w0..`` vertex values of a loft, in index order.

    A loft carries two profiles in one params dict, so
    ``polygon._profile_vertex_values`` — which takes everything named ``v``
    followed by digits — would return only half of it and silently.

    Args:
        params: The primitive's params dict.
        prefix: ``"v"`` for profile A, ``"w"`` for profile B.
        count: Vertices per profile.

    Returns:
        The ``count`` vertex values, in index order.
    """
    return [params[f"{prefix}{i}"].value for i in range(count)]


def _planar_walls(verts_a, verts_b, height) -> bool | None:
    """Whether *every* wall of a loft is flat, read from concrete vertices.

    Wall ``k`` is the quadrilateral ``(v_k, v_{k+1}, w_{k+1}, w_k)`` spanning
    the two profile planes.  It is a plane exactly when its four corners are
    coplanar, and a general ruled surface otherwise; the test is the classic
    one, the distance of the fourth corner from the plane through the other
    three, taken relative to the loft's own bounding diagonal so it means the
    same thing on a 3 mm port and a 3 m tank.

    All-or-nothing on purpose: :meth:`LoftedPolygon.patch_fields` may only
    declare a decomposition that covers the whole surface, so one ruled wall
    disqualifies the node rather than yielding a partial cover.

    Args:
        verts_a: Profile-A vertices, each ``(2,)``.
        verts_b: Profile-B vertices, each ``(2,)``.
        height: Total loft height.

    Returns:
        True when every wall is flat, False when any is ruled or degenerate,
        and ``None`` when the vertices are traced — coplanarity is a discrete
        reading, and a tracer cannot give one.
    """
    try:
        half = float(height) / 2.0
        bottom = np.array([[float(v[0]), float(v[1]), -half] for v in verts_a])
        top = np.array([[float(v[0]), float(v[1]), half] for v in verts_b])
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return None

    corners = np.vstack([bottom, top])
    diagonal = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        return False
    # A collapsed height puts both profiles in one plane.  Every "wall" is
    # then trivially coplanar — and trivially not a wall: the normal built
    # below would come out horizontal, describing the cap rather than a side.
    if abs(2.0 * half) <= _COPLANAR_TOLERANCE * diagonal:
        return False
    # Outward orientation below is read from profile A alone, so two profiles
    # wound against each other would flip the sign of the top half's walls —
    # and such a loft self-intersects somewhere in the middle anyway.
    if float(_profile_orientation(verts_a)) != float(_profile_orientation(verts_b)):
        return False

    num = len(bottom)
    for k in range(num):
        first, second = bottom[k], bottom[(k + 1) % num]
        above = top[k]
        normal = np.cross(second - first, above - first)
        area = float(np.linalg.norm(normal))
        # A near-zero cross product means no plane through the first three
        # corners: a collapsed profile edge, or a side parallel to it.
        if area <= _COPLANAR_TOLERANCE * diagonal * diagonal:
            return False
        residual = abs(float(normal @ (top[(k + 1) % num] - first))) / area
        if residual > _COPLANAR_TOLERANCE * diagonal:
            return False
    return True


def _wall_plane_fields(verts_a, verts_b, height, orientation):
    """One outward unit-gradient plane field per loft wall.

    Wall ``k``'s plane is spanned by its bottom profile edge and the side
    joining that edge's first vertex to its partner on the far profile:

        ``n = orientation · (d_y·h, -d_x·h, d_x·e_y - d_y·e_x)``

    for bottom edge direction ``d``, side offset ``e = w_k - v_k`` and height
    ``h``.  The ``orientation`` factor is what makes ``n`` point *out* of the
    solid for either input winding: the plane contains the bottom edge, so
    ``n``'s horizontal part is ``h·(d_y, -d_x)``, which is the edge's own
    outward 2D normal up to that same winding sign.

    Every step is a JAX expression of the vertices, so the fields stay
    differentiable in both profiles — which is why ``orientation`` is passed
    in rather than read here.

    Args:
        verts_a: Profile-A vertices, each ``(2,)``; may be tracers.
        verts_b: Profile-B vertices, each ``(2,)``; may be tracers.
        height: Total loft height.
        orientation: The profiles' shared winding sign.

    Returns:
        List of callables mapping ``(..., 3)`` points to signed distances.
    """
    num = len(verts_a)
    fields = []
    for k in range(num):
        first, second = verts_a[k], verts_a[(k + 1) % num]
        direction = second - first
        side = verts_b[k] - first
        normal = orientation * jnp.stack(
            [
                direction[1] * height,
                -direction[0] * height,
                direction[0] * side[1] - direction[1] * side[0],
            ]
        )
        unit = normal / jnp.sqrt(jnp.sum(normal**2) + 1e-20)

        def field(p: Array, n=unit, a=first, h=height) -> Array:
            # A corner of the wall, on the bottom profile plane.
            corner = jnp.stack([a[0], a[1], -jnp.asarray(h) / 2.0])
            return jnp.sum((p - corner) * n, axis=-1)

        fields.append(field)
    return fields


class LoftedPolygon(Primitive):
    """Solid lofted between two polygon profiles along the local Z axis.

    Profile A lies at ``z = -height/2``, profile B at ``z = +height/2``; each
    slice in between is the per-vertex linear interpolation of the two vertex
    loops. Use ``Translate``/``Rotate`` — or the construction-layer
    :func:`cadjoint.construction.loft` — for placement.

    Vertex parameter references are preserved for both profiles, so sketch
    constraints and gradients keep acting on the generated solid.

    Args:
        vertices_a: Ordered profile-A vertices — Vector2 parameters (or raw
            (2,) arrays).
        vertices_b: Ordered profile-B vertices; must match ``vertices_a`` in
            count, with vertex ``i`` connecting to vertex ``i``.
        height: Total loft height (Scalar parameter or float).
        material: Optional render material.
    """

    def __init__(
        self,
        vertices_a: list[Vector2],
        vertices_b: list[Vector2],
        height: float | Scalar,
        material=None,
    ):
        from cadjoint.render.material import Material

        if len(vertices_a) < 3:
            raise ValueError(f"LoftedPolygon needs at least 3 vertices, got {len(vertices_a)}")
        if len(vertices_a) != len(vertices_b):
            raise ValueError(
                "LoftedPolygon profiles must have equal vertex counts, "
                f"got {len(vertices_a)} and {len(vertices_b)}"
            )
        self.material = material if material is not None else Material()
        self.num_vertices = len(vertices_a)
        self.params = {f"v{i}": v for i, v in enumerate(vertices_a)}
        self.params.update({f"w{i}": v for i, v in enumerate(vertices_b)})
        self.params["height"] = height
        # Two discrete readings taken once, here, from the nominal geometry:
        # neither the shoelace winding nor "is this wall flat" is something a
        # tracer can answer, and ``patch_fields`` may be rebuilt with the
        # profiles traced (see ExtrudedPolygon.__init__ for the same move).
        # ``params`` are still raw at this point — the base class wraps them
        # after __init__ returns.
        raw_a = [getattr(v, "value", v) for v in vertices_a]
        raw_b = [getattr(v, "value", v) for v in vertices_b]
        self._orientation = _profile_orientation(raw_a)
        self._planar_walls = _planar_walls(raw_a, raw_b, getattr(height, "value", height))

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, height: Array, **vertices: Array) -> Array:
        """Pure SDF: per-slice exact polygon distance to the lerped profile.

        Args:
            p: Query point(s), shape (..., 3). Profile plane is local XY.
            height: Total loft height; the solid spans ``±height/2`` in Z.
            **vertices: v0..v{N-1} profile-A and w0..w{N-1} profile-B
                vertices, each shape (2,).

        Returns:
            Signed distance bound, shape (...). Sign-correct while every
            interpolated slice polygon is simple.
        """
        num = len(vertices) // 2
        t = jnp.clip(p[..., 2] / height + 0.5, 0.0, 1.0)[..., None]
        verts = [
            vertices[f"v{i}"] + (vertices[f"w{i}"] - vertices[f"v{i}"]) * t for i in range(num)
        ]
        d2 = _lerped_polygon_distance(p[..., :2], verts)
        dz = jnp.abs(p[..., 2]) - height / 2.0
        # Branch on inside/outside like Box.sdf so points exactly on a side
        # wall or cap receive a valid one-sided subgradient instead of the
        # exact zero an epsilon-smoothed outside norm produces there.
        max_d = jnp.maximum(d2, dz)
        squared = jnp.maximum(d2, 0.0) ** 2 + jnp.maximum(dz, 0.0) ** 2
        on_or_inside = max_d <= 0.0
        safe = jnp.where(on_or_inside, 1.0, squared)
        return jnp.where(on_or_inside, max_d, jnp.sqrt(safe))

    def __call__(self, p: Array) -> Array:
        values = {k: v.value for k, v in self.params.items() if k != "height"}
        return LoftedPolygon.sdf(p, self.params["height"].value, **values)

    def to_functional(self):
        return LoftedPolygon.sdf

    def patch_fields(self):
        """Wall planes plus the two caps — but only when every wall is flat.

        Order matches :class:`~cadjoint.sdf.primitives.polygon.ExtrudedPolygon`:
        patch ``k`` (for ``k < N``) is the wall between profile edge
        ``(v_k, v_{k+1})`` and its partner ``(w_k, w_{k+1})``; patch ``N`` is
        the bottom cap (``z = -height/2``), patch ``N+1`` the top.

        A loft wall is in general a **ruled** surface — the straight line
        joining the two edges sweeps as it goes, and sweeps out of plane the
        moment the two edges are not parallel — and a ruled surface is none of
        the plane, cylinder, sphere or cone a consumer can certify.  It is a
        plane exactly when the wall's four corners are coplanar, which is the
        common case in practice because it covers every truncated pyramid: two
        scaled copies of one profile, which is what a flared port or a draughted
        pad is.  So :func:`_planar_walls` measures every wall, and the node
        declares all-or-nothing: one ruled wall and the whole node returns
        ``None``, because a partial decomposition would silently hide that
        wall's surface rather than fall back to one opaque patch.

        What is declared for a flat wall is the plane through its four
        corners, outward-oriented and unit-gradient.  That is the *analytic
        continuation* of what :meth:`sdf` computes there rather than a copy of
        it: the sdf measures within the horizontal slice, which is the same
        zero set scaled by ``1/|n_xy|`` — a wall tilted 60° off vertical would
        read twice the true distance and win ownership it should lose.  Two
        further differences, both outside the surface and so both harmless:
        the sdf clamps its interpolation parameter past the caps, where this
        plane simply keeps going; and the sdf is a bound rather than an exact
        distance away from the wall, while a plane's field is exact
        everywhere.

        Returns:
            ``N + 2`` fields, or ``None`` when any wall is ruled.
        """
        verts_a = _profile_values(self.params, "v", self.num_vertices)
        verts_b = _profile_values(self.params, "w", self.num_vertices)
        height = self.params["height"].value
        planar = _planar_walls(verts_a, verts_b, height)
        # Traced vertices cannot answer; fall back to the nominal reading.
        if planar is None:
            planar = self._planar_walls
        if not planar:
            return None
        walls = _wall_plane_fields(verts_a, verts_b, height, self._orientation)
        caps = [
            lambda p: -p[..., 2] - height / 2.0,
            lambda p: p[..., 2] - height / 2.0,
        ]
        return walls + caps
