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

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Scalar, Vector2
from jaxcad.sdf.primitives.base import Primitive


def polygon_sdf_2d(p: Array, vertices: Array) -> Array:
    """Exact signed distance from 2D point(s) to a simple polygon.

    Args:
        p: Query point(s) in profile coordinates, shape (..., 2).
        vertices: Ordered polygon vertices, shape (N, 2). Either winding.

    Returns:
        Signed distance, shape (...). Negative inside.
    """
    num = vertices.shape[0]
    d = jnp.sum((p - vertices[0]) ** 2, axis=-1)
    s = jnp.ones(p.shape[:-1])
    for i in range(num):
        j = (i + num - 1) % num
        e = vertices[j] - vertices[i]
        w = p - vertices[i]
        t = jnp.clip(jnp.sum(w * e, axis=-1) / jnp.sum(e * e), 0.0, 1.0)
        b = w - e * t[..., None]
        d = jnp.minimum(d, jnp.sum(b * b, axis=-1))
        # Even-odd crossing test; flips sign once per boundary crossing.
        c1 = p[..., 1] >= vertices[i, 1]
        c2 = p[..., 1] < vertices[j, 1]
        c3 = e[0] * w[..., 1] > e[1] * w[..., 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = jnp.where(flip, -s, s)
    return s * jnp.sqrt(d + 1e-20)


def _stack_vertices(vertices: dict[str, Array]) -> Array:
    """Stack v0..v{N-1} keyword params into an (N, 2) array in index order."""
    ordered = sorted(vertices, key=lambda name: int(name[1:]))
    return jnp.stack([vertices[name] for name in ordered])


class ExtrudedPolygon(Primitive):
    """Solid formed by extruding a polygon profile along the local Z axis.

    The profile lies in the local XY plane; the solid spans ``±depth/2`` in Z
    (centered on the sketch plane). Use ``Translate``/``Rotate`` — or the
    construction-layer :func:`jaxcad.construction.extrude` — for placement.

    Args:
        vertices: Ordered profile vertices — Vector2 parameters (or raw (2,)
            arrays). Parameter references are preserved, so sketch constraints
            keep acting on the generated solid.
        depth: Total extrusion depth (Scalar parameter or float).
        material: Optional render material.
    """

    def __init__(self, vertices: list[Vector2], depth: float | Scalar, material=None):
        from jaxcad.render.material import Material

        if len(vertices) < 3:
            raise ValueError(f"ExtrudedPolygon needs at least 3 vertices, got {len(vertices)}")
        self.material = material if material is not None else Material()
        self.num_vertices = len(vertices)
        self.params = {f"v{i}": v for i, v in enumerate(vertices)}
        self.params["depth"] = depth

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, depth: Array, **vertices: Array) -> Array:
        """Pure SDF: exact extrusion of the exact polygon distance.

        Args:
            p: Query point(s), shape (..., 3). Profile plane is local XY.
            depth: Total extrusion depth.
            **vertices: v0..v{N-1} profile vertices, each shape (2,).

        Returns:
            Signed distance, shape (...).
        """
        verts = _stack_vertices(vertices)
        d2 = polygon_sdf_2d(p[..., :2], verts)
        dz = jnp.abs(p[..., 2]) - depth / 2.0
        w = jnp.stack([d2, dz], axis=-1)
        outside = jnp.sqrt(jnp.sum(jnp.maximum(w, 0.0) ** 2, axis=-1) + 1e-20)
        inside = jnp.minimum(jnp.maximum(w[..., 0], w[..., 1]), 0.0)
        return inside + outside

    def __call__(self, p: Array) -> Array:
        values = {k: v.value for k, v in self.params.items() if k != "depth"}
        return ExtrudedPolygon.sdf(p, self.params["depth"].value, **values)

    def to_functional(self):
        return ExtrudedPolygon.sdf


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
        from jaxcad.render.material import Material

        if len(vertices) < 3:
            raise ValueError(f"RevolvedPolygon needs at least 3 vertices, got {len(vertices)}")
        self.material = material if material is not None else Material()
        self.num_vertices = len(vertices)
        self.params = {f"v{i}": v for i, v in enumerate(vertices)}
        self.params["offset"] = offset

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
        verts = _stack_vertices(vertices)
        radial = jnp.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2 + 1e-20) - offset
        q = jnp.stack([radial, p[..., 1]], axis=-1)
        return polygon_sdf_2d(q, verts)

    def __call__(self, p: Array) -> Array:
        values = {k: v.value for k, v in self.params.items() if k != "offset"}
        return RevolvedPolygon.sdf(p, self.params["offset"].value, **values)

    def to_functional(self):
        return RevolvedPolygon.sdf
