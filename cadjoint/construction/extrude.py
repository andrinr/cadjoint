"""Extrude a 2D profile into a solid SDF."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cadjoint.construction.faces import (
    Feature,
    attach_faces,
    extrusion_faces,
    register_feature,
)
from cadjoint.construction.sketch import PolygonProfile, SketchPlane
from cadjoint.geometry.parameters import Scalar

if TYPE_CHECKING:
    from cadjoint.sdf.base import SDF


def _place_on_plane(sdf: SDF, plane: SketchPlane) -> SDF:
    """Wrap a local-frame SDF with transforms placing it on a sketch plane.

    Rotation (orientation snapshot) then translation. The translation shares
    the plane's ``origin`` Parameter, so plane position stays differentiable
    and constraint-driven; orientation is captured at generation time.
    """
    from cadjoint.sdf.transforms import Rotate, Translate

    if plane.is_identity():
        return sdf
    axis, angle = plane.axis_angle()
    # A traced angle is not comparable, so a plane derived under jit/grad
    # always gets its Rotate — the identity shortcut is a concrete-scene
    # optimization, not part of the placement's meaning.
    identity = isinstance(angle, float) and angle == 0.0
    placed = sdf if identity else Rotate(sdf, axis=axis, angle=angle)
    return Translate(placed, offset=plane.origin)


def extrude(
    profile: PolygonProfile,
    depth: float | Scalar,
    material=None,
    draft: float | Scalar = 0.0,
    twist: float | Scalar = 0.0,
):
    """Extrude a 2D profile into a solid.

    Generates an
    :class:`~cadjoint.sdf.primitives.polygon.ExtrudedPolygon` that shares the
    profile's vertex parameters and is placed on the profile's sketch plane.
    The solid spans ``±depth/2`` around the plane.

    Args:
        profile: PolygonProfile to extrude.
        depth: Total extrusion depth.
        material: Optional render material.
        draft: Draft angle in degrees. Positive values taper the walls inward
            as local z increases; the profile is exact at the bottom cap.
        twist: Total twist in degrees over the full depth. Non-zero twist
            makes the field non-1-Lipschitz.

    Returns:
        SDF solid sharing parameter references with the construction tree, with
            its analytic faces bound on: ``solid.cap("+")`` and ``solid.cap("-")``
            for the two ends, ``solid.side(i)`` for the wall swept by profile edge
            ``i``, and ``solid.faces`` for all of them. A drafted or twisted
            extrusion declares none — see
            :func:`cadjoint.construction.faces.extrusion_faces`.

    Example:
        ```python
        profile = PolygonProfile([[0, 0], [2, 0], [2, 1], [0, 1]])
        solid = extrude(profile, depth=0.5)
        boss = PolygonProfile(SQUARE, plane=SketchPlane.on(solid.cap("+")))
        ```
    """
    from cadjoint.sdf.primitives.polygon import ExtrudedPolygon

    base = ExtrudedPolygon(
        profile.vertices, depth=depth, material=material, draft=draft, twist=twist
    )
    solid = _place_on_plane(base, profile.plane)
    faces = extrusion_faces(profile, depth, draft=draft, twist=twist)
    attach_faces(solid, faces)
    register_feature(profile, Feature("extrude", faces, solid=solid))
    return solid
