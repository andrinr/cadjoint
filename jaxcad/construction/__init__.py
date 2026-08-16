"""Construction layer: a second tree of editable CAD scaffolding.

The construction tree holds sketch planes and parameter-bearing 2D profiles —
the objects a user edits. Generator functions turn them into SDF solids that
share the *same* Parameter objects, so sketch constraints and gradients act
directly on the generated geometry. The construction tree itself is rendered
as a wireframe overlay on top of SDF renders (``jaxcad.render.overlay``).

Construction tree:
- SketchPlane(origin, normal) — work plane / coordinate frame
- PolygonProfile(vertices, plane) — closed 2D profile with Vector2 parameters

Generators (construction → SDF):
- extrude(profile, depth) → ExtrudedPolygon placed on the profile's plane
- revolve(profile, offset) → RevolvedPolygon placed on the profile's plane
- loft(profile_a, profile_b, height) → LoftedPolygon placed on profile_a's plane

Legacy helpers:
- extrude(rectangle, depth) → Box
- from_line(line, radius) → Capsule
- from_circle(circle, height) → Cylinder
- from_point(point, radius) → Sphere
"""

from __future__ import annotations

from jaxcad.construction.extrude import extrude
from jaxcad.construction.from_circle import from_circle
from jaxcad.construction.from_line import from_line
from jaxcad.construction.from_point import from_point
from jaxcad.construction.loft import loft
from jaxcad.construction.revolve import revolve
from jaxcad.construction.sketch import PolygonProfile, SketchPlane
from jaxcad.construction.solid import ConstructionPrimitive, Solid

__all__ = [
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
