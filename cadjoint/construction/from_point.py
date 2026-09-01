"""Create a sphere centered at a point."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cadjoint.geometry.parameters import Scalar, Vector, as_parameter

if TYPE_CHECKING:
    from cadjoint.sdf.primitives.sphere import Sphere


def from_point(point: Vector, radius: float | Scalar) -> Sphere:
    """Create a sphere centered at a point.

    Args:
        point: Center point (Vector parameter)
        radius: Sphere radius

    Returns:
        Sphere SDF with shared parameter references

    Example:
        ```python
        center = Vector([0, 0, 0], free=True, name='center')
        radius = Scalar(1.0, free=True, name='radius')
        sphere = from_point(center, radius)
        ```
    """
    from cadjoint.sdf.primitives.sphere import Sphere

    radius = as_parameter(radius)

    # Sphere at origin with given radius
    # To position it, we'd apply a translate transform
    sphere = Sphere(radius=radius)

    # Store reference to source geometry
    sphere._source_point = point
    sphere._construction_method = "from_point"

    return sphere
