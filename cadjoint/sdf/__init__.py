"""SDF (Signed Distance Function) module.

This module contains all SDF-related functionality:
- Base SDF class
- Primitives (Sphere, Box, Cylinder, etc.)
- Boolean operations (Union, Intersection, Difference)
- Transforms (Translate, Rotate, Scale, Twist)
- Field operations (Shell, Offset, Mirror, LinearPattern, PolarPattern)
"""

from cadjoint.functionalize import functionalize
from cadjoint.sdf import measure
from cadjoint.sdf.base import SDF
from cadjoint.sdf.boolean import (
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
from cadjoint.sdf.measure import material_mass, volume
from cadjoint.sdf.operations import (
    LinearPattern,
    Mirror,
    Offset,
    PolarPattern,
    Shell,
    linear_pattern,
    mirror,
    offset,
    polar_pattern,
    shell,
)
from cadjoint.sdf.primitives import (
    Box,
    Capsule,
    Cylinder,
    Plane,
    Primitive,
    RoundBox,
    Sphere,
    Torus,
)
from cadjoint.sdf.transforms import (
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
    "material_mass",
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
    # Field operations
    "Shell",
    "Offset",
    "Mirror",
    "LinearPattern",
    "PolarPattern",
    "shell",
    "offset",
    "mirror",
    "linear_pattern",
    "polar_pattern",
]
