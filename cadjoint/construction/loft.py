"""Loft between two 2D profiles into a solid SDF."""

from __future__ import annotations

from cadjoint.construction.extrude import _place_on_plane
from cadjoint.construction.faces import (
    Feature,
    attach_faces,
    loft_faces,
    register_feature,
)
from cadjoint.construction.sketch import PolygonProfile
from cadjoint.geometry.parameters import Scalar


def loft(
    profile_a: PolygonProfile,
    profile_b: PolygonProfile,
    height: float | Scalar,
    material=None,
):
    """Loft between two polygon profiles with equal vertex counts.

    Generates a :class:`~cadjoint.sdf.primitives.loft.LoftedPolygon` that shares
    *both* profiles' vertex parameters and is placed on ``profile_a``'s sketch
    plane. In the placed frame, profile A sits at ``-height/2`` and profile B
    at ``+height/2`` along the plane normal; vertex ``i`` of A connects to
    vertex ``i`` of B. ``profile_b``'s own plane is ignored — only its 2D
    vertex coordinates are used.

    Args:
        profile_a: Bottom profile; its plane places the solid.
        profile_b: Top profile; must have the same vertex count as
            ``profile_a``.
        height: Total loft height along ``profile_a``'s plane normal.
        material: Optional render material.

    Returns:
        SDF solid sharing parameter references with both profiles, with its two
            planar ends bound on as ``solid.cap("-")`` (profile A) and
            ``solid.cap("+")`` (profile B). The ruled side walls are planar only by
            accident, so they are not declared.

    Raises:
        ValueError: If the two profiles have different vertex counts.

    Example:
        ```python
        bottom = PolygonProfile([[-1, -1], [1, -1], [1, 1], [-1, 1]], name="a")
        top = PolygonProfile([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], name="b")
        solid = loft(bottom, top, height=2.0)
        ```
    """
    from cadjoint.sdf.primitives.loft import LoftedPolygon

    if len(profile_a.vertices) != len(profile_b.vertices):
        raise ValueError(
            "loft requires profiles with equal vertex counts, "
            f"got {len(profile_a.vertices)} and {len(profile_b.vertices)}"
        )
    base = LoftedPolygon(profile_a.vertices, profile_b.vertices, height=height, material=material)
    solid = _place_on_plane(base, profile_a.plane)
    faces = loft_faces(profile_a, profile_b, height)
    attach_faces(solid, faces)
    register_feature(profile_a, Feature("loft", faces, solid=solid))
    return solid
