"""Construction mirrors for SDF primitives.

A sketch profile is editable in the viewer because it is a construction-layer
object whose parameters are shared with the SDF it generates. Plain primitives
had no such mirror: ``Translate(Sphere(0.5), [1, 0, 0])`` is an SDF tree with no
notion of "this is a sphere at this position", so the viewer could not draw a
handle for it or move it.

:class:`ConstructionPrimitive` supplies that mirror. It owns the position,
rotation, and dimensions as named free parameters, builds the placed SDF from
those same ``Parameter`` objects, and can report a wireframe outline the viewer
draws over the rendered solid.

Rotation is stored as three scalars rather than one vector because
:class:`~jaxcad.sdf.transforms.affine.rotate.Rotate` takes a scalar angle, and
sharing the parameter object is what keeps the mirror and the solid in step.
Angles are radians, applied X then Y then Z.
"""

from __future__ import annotations

import math
from typing import Callable

import jax.numpy as jnp
from jax import Array

from jaxcad.fluent import Fluent
from jaxcad.geometry.parameters import Scalar, Vector

# Points per full circle when tracing a curved silhouette.
_CIRCLE_SEGMENTS = 48

# Which keyword arguments each kind exposes, in the order they are written.
DIMENSIONS: dict[str, tuple[str, ...]] = {
    "box": ("size",),
    "sphere": ("radius",),
    "cylinder": ("radius", "height"),
}


def _free_vector(value, name: str) -> Vector:
    """Wrap a raw 3-vector as a named free parameter, or adopt an existing one."""
    if isinstance(value, Vector):
        if value.free and value.name is None:
            value.name = name
        return value
    return Vector(value=jnp.asarray(value, dtype=jnp.float32), free=True, name=name)


def _free_scalar(value, name: str) -> Scalar:
    """Wrap a raw number as a named free parameter, or adopt an existing one."""
    if isinstance(value, Scalar):
        if value.free and value.name is None:
            value.name = name
        return value
    return Scalar(value=jnp.asarray(value, dtype=jnp.float32), free=True, name=name)


def _rotation_matrix(rx: float, ry: float, rz: float) -> Array:
    """World rotation for intrinsic X→Y→Z angles: ``Rz @ Ry @ Rx``.

    Matches the nesting order of the generated ``Rotate`` transforms, so the
    outline lands exactly on the solid.
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    x = jnp.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    y = jnp.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    z = jnp.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return z @ y @ x


def _ring(radius: float, axis: int, offset: float = 0.0) -> list[tuple]:
    """Edges tracing a circle perpendicular to `axis` at `offset` along it."""
    points = []
    for step in range(_CIRCLE_SEGMENTS):
        angle = 2 * math.pi * step / _CIRCLE_SEGMENTS
        a, b = radius * math.cos(angle), radius * math.sin(angle)
        if axis == 0:
            points.append((offset, a, b))
        elif axis == 1:
            points.append((a, offset, b))
        else:
            points.append((a, b, offset))
    return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


class ConstructionPrimitive(Fluent):
    """A primitive with an editable placement, mirroring the SDF it generates.

    Args:
        kind: One of ``box``, ``sphere``, ``cylinder``.
        position: World position of the primitive's centre.
        rotation: Intrinsic X, Y, Z angles in radians.
        material: Optional render material for the generated SDF.
        name: Prefix for the generated parameter names.
        **dimensions: The kind's size arguments — ``size`` for a box (half
            extents), ``radius`` for a sphere, ``radius`` and ``height`` (half
            height) for a cylinder.
    """

    def __init__(
        self,
        kind: str,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        material=None,
        name: str | None = None,
        **dimensions,
    ):
        if kind not in DIMENSIONS:
            raise ValueError(
                f"Unknown primitive kind {kind!r}; expected one of {sorted(DIMENSIONS)}"
            )
        missing = set(DIMENSIONS[kind]) - set(dimensions)
        if missing:
            raise ValueError(f"{kind} needs {', '.join(sorted(missing))}")

        self.kind = kind
        self.name = name or kind
        self.material = material

        angles = jnp.asarray(rotation, dtype=jnp.float32)
        if angles.shape != (3,):
            raise ValueError(f"rotation must be three angles, got shape {angles.shape}")

        self.params = {"position": _free_vector(position, f"{self.name}_position")}
        for index, axis in enumerate("xyz"):
            self.params[f"r{axis}"] = _free_scalar(angles[index], f"{self.name}_r{axis}")
        for key in DIMENSIONS[kind]:
            value = dimensions[key]
            self.params[key] = (
                _free_vector(value, f"{self.name}_{key}")
                if key == "size"
                else _free_scalar(value, f"{self.name}_{key}")
            )

    # ── placement ────────────────────────────────────────────────────────────

    @property
    def position(self) -> Vector:
        return self.params["position"]

    @property
    def rotation(self) -> tuple[Scalar, Scalar, Scalar]:
        return (self.params["rx"], self.params["ry"], self.params["rz"])

    def rotation_values(self) -> tuple[float, float, float]:
        return tuple(float(angle.value) for angle in self.rotation)

    def sdf(self):
        """Build the placed SDF, sharing this mirror's parameter objects."""
        from jaxcad.sdf.primitives import Box, Cylinder, Sphere
        from jaxcad.sdf.transforms import Rotate, Translate

        if self.kind == "box":
            solid = Box(size=self.params["size"], material=self.material)
        elif self.kind == "sphere":
            solid = Sphere(radius=self.params["radius"], material=self.material)
        else:
            solid = Cylinder(
                radius=self.params["radius"],
                height=self.params["height"],
                material=self.material,
            )

        # Only rotate about axes that are actually turned: an identity Rotate
        # would cost a matrix multiply at every raymarch step for nothing.
        for axis, angle in zip("xyz", self.rotation):
            if float(angle.value) != 0.0:
                solid = Rotate(solid, axis=axis, angle=angle)
        return Translate(solid, offset=self.position)

    # ── outline ──────────────────────────────────────────────────────────────

    def local_edges(self) -> list[tuple]:
        """Wireframe edges in the primitive's own frame."""
        if self.kind == "box":
            sx, sy, sz = (float(v) for v in self.params["size"].xyz)
            corners = [(x * sx, y * sy, z * sz) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            # Corners differing in exactly one coordinate share an edge.
            edges = []
            for i, a in enumerate(corners):
                for b in corners[i + 1 :]:
                    if sum(1 for p, q in zip(a, b) if p != q) == 1:
                        edges.append((a, b))
            return edges

        if self.kind == "sphere":
            radius = float(self.params["radius"].value)
            return _ring(radius, 0) + _ring(radius, 1) + _ring(radius, 2)

        radius = float(self.params["radius"].value)
        height = float(self.params["height"].value)
        edges = _ring(radius, 2, height) + _ring(radius, 2, -height)
        for step in range(4):
            angle = 2 * math.pi * step / 4
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            edges.append(((x, y, -height), (x, y, height)))
        return edges

    def world_edges(self) -> list[list[list[float]]]:
        """Wireframe edges in world space, as ``[[ax, ay, az], [bx, by, bz]]``."""
        matrix = _rotation_matrix(*self.rotation_values())
        origin = self.position.xyz

        def place(point) -> list[float]:
            local = jnp.asarray(point, dtype=jnp.float32)
            return [float(v) for v in matrix @ local + origin]

        return [[place(a), place(b)] for a, b in self.local_edges()]


def _make(kind: str) -> Callable:
    """Build a ``Solid.<kind>`` factory that returns the placed SDF."""

    def factory(
        position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), material=None, name=None, **dimensions
    ):
        primitive = ConstructionPrimitive(
            kind,
            position=position,
            rotation=rotation,
            material=material,
            name=name,
            **dimensions,
        )
        return primitive.sdf()

    factory.__name__ = kind
    return factory


class Solid:
    """Factories creating a primitive and its construction mirror at once.

    Each returns the placed SDF, so solids compose directly::

        scene = Union(
            Solid.box(size=[1, 1, 0.5], position=[0, 0, 0]),
            Solid.sphere(radius=0.6, position=[1.5, 0, 0]),
        )

    The viewer picks up the mirror separately and draws its outline, which is
    what makes the result selectable and draggable.
    """

    box = staticmethod(_make("box"))
    sphere = staticmethod(_make("sphere"))
    cylinder = staticmethod(_make("cylinder"))
