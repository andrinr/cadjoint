"""Tests for Cylinder SDF primitive."""

import jax.numpy as jnp
import pytest

from cadjoint.sdf.primitives import Cylinder


@pytest.mark.parametrize(
    "radius,height,point,expected_sign",
    [
        (1.0, 1.0, [0.0, 0.0, 0.0], "negative"),  # Center
        (1.0, 1.0, [1.0, 0.0, 0.0], "zero"),  # Side surface
        (1.0, 1.0, [0.0, 1.0, 0.0], "zero"),  # Side surface (y-axis)
        (1.0, 1.0, [0.0, 0.0, 1.0], "zero"),  # Top cap
        (1.0, 1.0, [0.0, 0.0, -1.0], "zero"),  # Bottom cap
        (1.0, 1.0, [2.0, 0.0, 0.0], "positive"),  # Outside
        (2.0, 3.0, [0.0, 0.0, 0.0], "negative"),  # Larger dimensions, center
    ],
)
def test_cylinder_distance(radius, height, point, expected_sign):
    """Test that cylinder SDF returns correct sign."""
    cyl = Cylinder(radius=radius, height=height)
    p = jnp.array(point)
    dist = cyl(p)

    if expected_sign == "negative":
        assert dist < 0, f"Expected negative distance at {point}, got {dist}"
    elif expected_sign == "zero":
        assert jnp.isclose(dist, 0.0, atol=1e-5), f"Expected ~0 distance at {point}, got {dist}"
    elif expected_sign == "positive":
        assert dist > 0, f"Expected positive distance at {point}, got {dist}"


def test_cylinder_radial_symmetry():
    """Test that cylinder is radially symmetric."""
    cyl = Cylinder(radius=1.0, height=2.0)

    # Points at same radius should have same distance
    points = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7071, 0.7071, 0.0],
    ]

    distances = [cyl(jnp.array(p)) for p in points]

    # All should be approximately equal (on surface)
    for i in range(len(distances) - 1):
        assert jnp.isclose(distances[i], distances[i + 1], atol=1e-3)
