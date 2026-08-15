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
        verts = _ordered_vertices(vertices)
        d2 = _polygon_distance(p[..., :2], verts)
        dz = jnp.abs(p[..., 2]) - depth / 2.0
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
        verts = _ordered_vertices(vertices)
        radial = jnp.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2 + 1e-20) - offset
        q = jnp.stack([radial, p[..., 1]], axis=-1)
        return _polygon_distance(q, verts)

    def __call__(self, p: Array) -> Array:
        values = {k: v.value for k, v in self.params.items() if k != "offset"}
        return RevolvedPolygon.sdf(p, self.params["offset"].value, **values)

    def to_functional(self):
        return RevolvedPolygon.sdf
