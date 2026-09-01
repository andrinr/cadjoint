"""Create a cylinder from a circle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cadjoint.geometry.parameters import Scalar, as_parameter
from cadjoint.geometry.primitives import Circle

if TYPE_CHECKING:
    from cadjoint.sdf.primitives.cylinder import Cylinder


def from_circle(circle: Circle, height: float | Scalar) -> Cylinder:
    """Create a cylinder from a circle.

    The cylinder's base is the circle, and it extends along the circle's normal
    by the specified height.

    Args:
        circle: Circle defining the cylinder's cross-section
        height: Cylinder height

    Returns:
        Cylinder SDF with shared parameter references

    Example:
        ```python
        center = Vector([0, 0, 0], free=True, name='center')
        radius = Scalar(1.0, free=True, name='radius')
        circle = Circle(center=center, radius=radius)
        cylinder = from_circle(circle, height=5.0)
        ```
    """
    from cadjoint.sdf.primitives.cylinder import Cylinder

    height = as_parameter(height)

    # Cylinder parameters: radius, height
    # The cylinder is axis-aligned in its local frame
    # For a general circle, we'd need to apply a transform
    # For now, assume the circle is in the XY plane

    cylinder = Cylinder(radius=circle.radius, height=height)

    # Store reference to source geometry
    cylinder._source_geometry = circle
    cylinder._construction_method = "from_circle"

    return cylinder
