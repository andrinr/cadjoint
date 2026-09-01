"""Geometric primitives for parametric construction.

This module contains 2D and 3D geometric entities that can be used to:
- Define parametric relationships with free/fixed parameters
- Apply geometric constraints
- Construct SDF primitives via the construction layer

Entities:
- Line: Parametric line segment in 3D
- Circle: Parametric circle in 3D
"""

from cadjoint.geometry.primitives.circle import Circle
from cadjoint.geometry.primitives.line import Line

__all__ = [
    "Line",
    "Circle",
]
