"""Revolve a 2D profile around its sketch plane's Y axis into a solid SDF."""

from __future__ import annotations

from jaxcad.construction.extrude import _place_on_plane
from jaxcad.construction.sketch import PolygonProfile
from jaxcad.geometry.parameters import Scalar


def revolve(profile: PolygonProfile, offset: float | Scalar = 0.0, material=None):
    """Revolve a polygon profile around the sketch plane's local Y axis.

    Profile coordinates are (radial, height): X is the distance from the
    revolution axis (plus ``offset``), Y runs along the axis. Generates a
    :class:`~jaxcad.sdf.primitives.polygon.RevolvedPolygon` sharing the
    profile's vertex parameters, placed on the profile's sketch plane.

    Args:
        profile: PolygonProfile to revolve; should stay at positive radius.
        offset: Radial offset added before revolving.
        material: Optional render material.

    Returns:
        SDF solid sharing parameter references with the construction tree.

    Example:
        ```python
        # A washer: square cross-section revolved at radius 1.2
        profile = PolygonProfile([[1.0, -0.2], [1.4, -0.2], [1.4, 0.2], [1.0, 0.2]])
        ring = revolve(profile)
        ```
    """
    from jaxcad.sdf.primitives.polygon import RevolvedPolygon

    base = RevolvedPolygon(profile.vertices, offset=offset, material=material)
    return _place_on_plane(base, profile.plane)
