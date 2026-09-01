"""Sketch construction tree: planes and 2D profiles that generate solids.

The construction tree is a second object tree next to the SDF tree. It holds
the editable CAD scaffolding — sketch planes and parameter-bearing 2D profiles.
Generators (:func:`cadjoint.construction.extrude`,
:func:`cadjoint.construction.revolve`) turn profiles into SDF nodes that share
the *same* ``Parameter`` objects, so constraints solved on the sketch and
gradients flowing through the solid act on one set of values.

The construction tree is never rasterized by the SDF renderer; it is drawn as
a wireframe overlay on top of rendered images (see ``cadjoint.render.overlay``).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.fluent import Fluent
from cadjoint.geometry.parameters import Vector, Vector2, as_parameter


def _plane_frame(normal: Array) -> tuple[Array, Array]:
    """Orthonormal in-plane axes (u, v) for a unit normal; right-handed u×v=n.

    The reference up-vector is +Y so the default +Z normal yields the identity
    frame u=(1,0,0), v=(0,1,0) — sketch coordinates on the world XY plane are
    world coordinates.
    """
    up = jnp.array([0.0, 1.0, 0.0])
    if abs(float(jnp.dot(normal, up))) > 0.99:
        up = jnp.array([0.0, 0.0, 1.0])
    u = jnp.cross(up, normal)
    u = u / jnp.linalg.norm(u)
    v = jnp.cross(normal, u)
    return u, v


class SketchPlane(Fluent):
    """A work plane the sketch entities live on.

    Defines a coordinate frame: ``origin`` plus orthonormal in-plane axes
    (u, v) derived from ``normal``. Profile coordinates (x, y) map to world
    space as ``origin + x·u + y·v``.

    Args:
        origin: Plane origin in world space (Vector parameter or 3D array).
        normal: Plane normal (Vector parameter or 3D array); default +Z, i.e.
            the world XY plane.
    """

    def __init__(self, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)):
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
        self.params = {"origin": origin, "normal": normal.normalize()}

    @property
    def origin(self) -> Vector:
        return self.params["origin"]

    @property
    def normal(self) -> Vector:
        return self.params["normal"]

    def frame(self) -> tuple[Array, Array, Array]:
        """Return the (u, v, n) orthonormal frame as arrays."""
        n = self.normal.xyz
        u, v = _plane_frame(n)
        return u, v, n

    def rotation_matrix(self) -> Array:
        """Rotation taking local (x, y, z) to world (u, v, n), columns [u v n]."""
        u, v, n = self.frame()
        return jnp.stack([u, v, n], axis=1)

    def axis_angle(self) -> tuple[Array, float]:
        """Axis-angle form of :meth:`rotation_matrix` (for the Rotate transform)."""
        R = self.rotation_matrix()
        trace = jnp.trace(R)
        angle = jnp.arccos(jnp.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        if float(angle) < 1e-6:
            return jnp.array([0.0, 0.0, 1.0]), 0.0
        if float(angle) > jnp.pi - 1e-4:
            # R is symmetric: axis from the largest diagonal of (R + I) / 2
            B = (R + jnp.eye(3)) / 2.0
            k = int(jnp.argmax(jnp.diag(B)))
            axis = B[:, k] / jnp.sqrt(B[k, k])
            return axis, float(jnp.pi)
        axis = jnp.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2.0 * jnp.sin(angle))
        return axis, float(angle)

    def is_identity(self) -> bool:
        """True when the plane is the world XY plane at the origin."""
        u, _, n = self.frame()
        return bool(
            jnp.allclose(n, jnp.array([0.0, 0.0, 1.0]))
            and jnp.allclose(u, jnp.array([1.0, 0.0, 0.0]))
            and jnp.allclose(self.origin.xyz, 0.0)
        )

    def to_world(self, xy: Array) -> Array:
        """Map plane coordinates (..., 2) to world coordinates (..., 3)."""
        u, v, _ = self.frame()
        xy = jnp.asarray(xy)
        return self.origin.xyz + xy[..., :1] * u + xy[..., 1:2] * v


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
    """

    def __init__(self, vertices, plane: SketchPlane | None = None, name: str = "profile"):
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
                        free=True,
                        name=f"{name}_v{i}",
                    )
                )
        self.vertices = wrapped
        self.params = {f"v{i}": v for i, v in enumerate(wrapped)}

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
