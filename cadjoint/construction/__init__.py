"""Construction layer: a second tree of editable CAD scaffolding.

The construction tree holds sketch planes and parameter-bearing 2D profiles —
the objects a user edits. Generator functions turn them into SDF solids that
share the *same* Parameter objects, so sketch constraints and gradients act
directly on the generated geometry. The construction tree itself is rendered
as a wireframe overlay on top of SDF renders (``cadjoint.render.overlay``).

Construction tree:
- SketchPlane(origin, normal, x_axis) — work plane / coordinate frame
- PolygonProfile(vertices, plane) — closed 2D profile with Vector2 parameters

Face references (``faces``):
- Every generated solid carries the feature's analytic faces — ``solid.cap("+")``,
  ``solid.side(i)``, ``block.face("+z")`` — and a revolve carries ``solid.axis``.
- ``SketchPlane.on/offset/tangent/midplane`` turn those into work planes that
  move with the parent's parameters, so a gradient through a child solid
  reaches the parent's ``depth``.

Generators (construction → SDF):
- extrude(profile, depth) → ExtrudedPolygon placed on the profile's plane
- revolve(profile, offset) → RevolvedPolygon placed on the profile's plane
- loft(profile_a, profile_b, height) → LoftedPolygon placed on profile_a's plane

Geometry-entity helpers:
- from_line(line, radius) → Capsule
- from_circle(circle, height) → Cylinder
- from_point(point, radius) → Sphere
"""

from __future__ import annotations

from cadjoint.construction.extrude import extrude
from cadjoint.construction.faces import Axis, Face, FaceSet, Feature
from cadjoint.construction.from_circle import from_circle
from cadjoint.construction.from_line import from_line
from cadjoint.construction.from_point import from_point
from cadjoint.construction.loft import loft
from cadjoint.construction.revolve import revolve
from cadjoint.construction.sketch import PolygonProfile, SketchPlane
from cadjoint.construction.solid import ConstructionPrimitive, Solid

__all__ = [
    "Axis",
    "Face",
    "FaceSet",
    "Feature",
    "SketchPlane",
    "PolygonProfile",
    "ConstructionPrimitive",
    "Solid",
    "extrude",
    "revolve",
    "loft",
    "from_line",
    "from_circle",
    "from_point",
]
