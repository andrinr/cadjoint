"""Revolve a 2D profile around its sketch plane's Y axis into a solid SDF."""

from __future__ import annotations

from cadjoint.construction.extrude import _place_on_plane
from cadjoint.construction.faces import (
    FaceSet,
    Feature,
    attach_faces,
    register_feature,
    revolve_axis,
)
from cadjoint.construction.sketch import PolygonProfile
from cadjoint.geometry.parameters import Scalar


def revolve(profile: PolygonProfile, offset: float | Scalar = 0.0, material=None):
    """Revolve a polygon profile around the sketch plane's local Y axis.

    Profile coordinates are (radial, height): X is the distance from the
    revolution axis (plus ``offset``), Y runs along the axis. Generates a
    :class:`~cadjoint.sdf.primitives.polygon.RevolvedPolygon` sharing the
    profile's vertex parameters, placed on the profile's sketch plane.

    Args:
        profile: PolygonProfile to revolve; should stay at positive radius.
        offset: Radial offset added before revolving.
        material: Optional render material.

    Returns:
        SDF solid sharing parameter references with the construction tree. A
            surface of revolution has no planar face to name, so ``solid.faces`` is
            empty and the usable reference is ``solid.axis`` — the world-space
            :class:`~cadjoint.construction.faces.Axis` it was swept around. For a
            plane *on* the curved wall, use
            :meth:`~cadjoint.construction.sketch.SketchPlane.tangent`.

    Example:
        ```python
        # A washer: square cross-section revolved at radius 1.2
        profile = PolygonProfile([[1.0, -0.2], [1.4, -0.2], [1.4, 0.2], [1.0, 0.2]])
        ring = revolve(profile)
        ```
    """
    from cadjoint.sdf.primitives.polygon import RevolvedPolygon

    base = RevolvedPolygon(profile.vertices, offset=offset, material=material)
    solid = _place_on_plane(base, profile.plane)
    faces = FaceSet("revolve", list)
    axis = revolve_axis(profile)
    attach_faces(solid, faces, axis)
    register_feature(profile, Feature("revolve", faces, axis=axis, solid=solid))
    return solid
