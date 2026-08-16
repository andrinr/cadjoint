"""Lofted polygon SDF primitive: linear interpolation between two profiles.

A loft connects two polygon profiles with the same vertex count by linearly
interpolating each vertex pair along the local Z axis. Every horizontal slice
of the solid is the exact 2D polygon distance to the interpolated profile, and
the slice distance is combined with the cap distance exactly like
:class:`~jaxcad.sdf.primitives.polygon.ExtrudedPolygon`.

The per-slice polygon distance is exact, but the assembled 3D field is a
*bound* on the true distance rather than an exact SDF: the side walls are
ruled surfaces, so the nearest surface point of a tapering wall generally lies
on a different slice than the query point. The sign is still correct wherever
each interpolated slice polygon stays simple (non-self-intersecting), which is
what meshing and rendering rely on.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Scalar, Vector2
from jaxcad.sdf.primitives.base import Primitive


def _lerped_polygon_distance(p: Array, vertices: list[Array]) -> Array:
    """Signed polygon distance where vertices may carry the query batch shape.

    Mirrors ``jaxcad.sdf.primitives.polygon._polygon_distance`` — including the
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


class LoftedPolygon(Primitive):
    """Solid lofted between two polygon profiles along the local Z axis.

    Profile A lies at ``z = -height/2``, profile B at ``z = +height/2``; each
    slice in between is the per-vertex linear interpolation of the two vertex
    loops. Use ``Translate``/``Rotate`` — or the construction-layer
    :func:`jaxcad.construction.loft` — for placement.

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
        from jaxcad.render.material import Material

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
