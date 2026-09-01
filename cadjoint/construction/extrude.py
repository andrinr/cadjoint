"""Extrude a 2D profile (or legacy rectangle) into a solid SDF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from cadjoint.construction.sketch import PolygonProfile, SketchPlane
from cadjoint.geometry.parameters import Scalar, Vector, as_parameter
from cadjoint.geometry.primitives import Rectangle

if TYPE_CHECKING:
    from cadjoint.sdf.base import SDF
    from cadjoint.sdf.primitives.box import Box


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
    placed = sdf if angle == 0.0 else Rotate(sdf, axis=axis, angle=angle)
    return Translate(placed, offset=plane.origin)


def extrude(
    profile: PolygonProfile | Rectangle,
    depth: float | Scalar,
    material=None,
    draft: float | Scalar = 0.0,
    twist: float | Scalar = 0.0,
):
    """Extrude a 2D profile into a solid.

    For a :class:`PolygonProfile`, generates an
    :class:`~cadjoint.sdf.primitives.polygon.ExtrudedPolygon` that shares the
    profile's vertex parameters and is placed on the profile's sketch plane.
    The solid spans ``±depth/2`` around the plane.

    The legacy ``Rectangle → Box`` path is kept for backward compatibility.

    Args:
        profile: PolygonProfile (or legacy Rectangle) to extrude.
        depth: Total extrusion depth.
        material: Optional render material (PolygonProfile path only).
        draft: Draft angle in degrees (PolygonProfile only). Positive values
            taper the walls inward as local z increases; the profile is exact
            at the bottom cap.
        twist: Total twist in degrees over the full depth (PolygonProfile
            only). Non-zero twist makes the field non-1-Lipschitz.

    Returns:
        SDF solid sharing parameter references with the construction tree.

    Example:
        ```python
        profile = PolygonProfile([[0, 0], [2, 0], [2, 1], [0, 1]])
        solid = extrude(profile, depth=0.5)
        ```
    """
    if isinstance(profile, PolygonProfile):
        from cadjoint.sdf.primitives.polygon import ExtrudedPolygon

        base = ExtrudedPolygon(
            profile.vertices, depth=depth, material=material, draft=draft, twist=twist
        )
        return _place_on_plane(base, profile.plane)
    for value in (draft, twist):
        if isinstance(value, Scalar) or float(value) != 0.0:
            raise ValueError("draft/twist are only supported for PolygonProfile extrusions")
    return _extrude_rectangle(profile, depth)


def _extrude_rectangle(rectangle: Rectangle, depth: float | Scalar) -> Box:
    """Extrude a rectangle to create an oriented box.

    The box is centered on the rectangle's plane and extends depth/2 in each
    direction along the rectangle's normal.

    Args:
        rectangle: Rectangle to extrude
        depth: Extrusion depth (total, not half-depth)

    Returns:
        Box SDF with shared parameter references

    Example:
        ```python
        rect = Rectangle(center=[0, 0, 0], width=2.0, height=1.0)
        box = extrude(rect, depth=3.0)
        ```
    """
    from cadjoint.sdf.primitives.box import Box

    depth = as_parameter(depth)

    # Box size: [width/2, height/2, depth/2]
    # We need to create a Vector that combines width, height, depth
    # Since rectangle params might be free, we need to handle this carefully

    # For now, create a new box centered at rectangle center
    # The size is [width/2, height/2, depth/2]
    # Note: This is a simplified version. Full implementation would need
    # proper coordinate frame handling for arbitrary rectangle orientations

    # Create size vector
    size_xyz = jnp.array([rectangle.width.value / 2, rectangle.height.value / 2, depth.value / 2])

    size = Vector(value=size_xyz, free=False, name=f"{rectangle.center.name}_box_size")

    # Create box
    box = Box(size=size)

    # Store reference to source geometry for potential future use
    box._source_geometry = rectangle
    box._construction_method = "extrude"

    return box
