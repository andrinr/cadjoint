"""Loft between two 2D profiles into a solid SDF."""

from __future__ import annotations

from jaxcad.construction.extrude import _place_on_plane
from jaxcad.construction.sketch import PolygonProfile
from jaxcad.geometry.parameters import Scalar


def loft(
    profile_a: PolygonProfile,
    profile_b: PolygonProfile,
    height: float | Scalar,
    material=None,
):
    """Loft between two polygon profiles with equal vertex counts.

    Generates a :class:`~jaxcad.sdf.primitives.loft.LoftedPolygon` that shares
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
        SDF solid sharing parameter references with both profiles.

    Raises:
        ValueError: If the two profiles have different vertex counts.

    Example:
        ```python
        bottom = PolygonProfile([[-1, -1], [1, -1], [1, 1], [-1, 1]], name="a")
        top = PolygonProfile([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], name="b")
        solid = loft(bottom, top, height=2.0)
        ```
    """
    from jaxcad.sdf.primitives.loft import LoftedPolygon

    if len(profile_a.vertices) != len(profile_b.vertices):
        raise ValueError(
            "loft requires profiles with equal vertex counts, "
            f"got {len(profile_a.vertices)} and {len(profile_b.vertices)}"
        )
    base = LoftedPolygon(profile_a.vertices, profile_b.vertices, height=height, material=material)
    return _place_on_plane(base, profile_a.plane)
