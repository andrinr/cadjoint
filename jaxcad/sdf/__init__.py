"""SDF (Signed Distance Function) module.

This module contains all SDF-related functionality:
- Base SDF class
- Primitives (Sphere, Box, Cylinder, etc.)
- Boolean operations (Union, Intersection, Difference)
- Transforms (Translate, Rotate, Scale, Twist)
"""

from jaxcad.functionalize import functionalize
from jaxcad.sdf import measure
from jaxcad.sdf.base import SDF
from jaxcad.sdf.boolean import (
    BooleanOp,
    Difference,
    Intersection,
    Union,
    Xor,
    difference,
    intersection,
    smooth_max,
    smooth_min,
    union,
    xor,
)
from jaxcad.sdf.measure import volume
from jaxcad.sdf.primitives import (
    Box,
    Capsule,
    Cylinder,
    Plane,
    Primitive,
    RoundBox,
    Sphere,
    Torus,
)
from jaxcad.sdf.transforms import (
    Rotate,
    Scale,
    Translate,
    Twist,
)

__all__ = [
    # Base
    "SDF",
    # Compilation
    "functionalize",
    # Measure
    "measure",
    "volume",
    # Primitives
    "Primitive",
    "Box",
    "Capsule",
    "Cylinder",
    "Plane",
    "RoundBox",
    "Sphere",
    "Torus",
    # Boolean operations
    "BooleanOp",
    "Union",
    "Intersection",
    "Difference",
    "Xor",
    "smooth_min",
    "smooth_max",
    "union",
    "intersection",
    "difference",
    "xor",
    # Transforms
    "Translate",
    "Rotate",
    "Scale",
    "Twist",
]
