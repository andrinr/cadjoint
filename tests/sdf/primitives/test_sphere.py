"""Tests for Sphere SDF primitive."""

import jax.numpy as jnp
import pytest

from cadjoint.sdf.primitives import Sphere


@pytest.mark.parametrize(
    "radius,point,expected_sign",
    [
        (1.0, [0.0, 0.0, 0.0], "negative"),  # Center
        (1.0, [1.0, 0.0, 0.0], "zero"),  # Surface (x-axis)
        (1.0, [0.0, 1.0, 0.0], "zero"),  # Surface (y-axis)
        (1.0, [0.0, 0.0, 1.0], "zero"),  # Surface (z-axis)
        (1.0, [2.0, 0.0, 0.0], "positive"),  # Outside
        (2.5, [0.0, 0.0, 0.0], "negative"),  # Larger radius, center
        (2.5, [2.5, 0.0, 0.0], "zero"),  # Larger radius, surface
    ],
)
def test_sphere_distance(radius, point, expected_sign):
    """Test that sphere SDF returns correct sign."""
    sphere = Sphere(radius=radius)
    p = jnp.array(point)
    dist = sphere(p)

    if expected_sign == "negative":
        assert dist < 0, f"Expected negative distance at {point}, got {dist}"
    elif expected_sign == "zero":
        assert jnp.isclose(dist, 0.0, atol=1e-5), f"Expected ~0 distance at {point}, got {dist}"
    elif expected_sign == "positive":
        assert dist > 0, f"Expected positive distance at {point}, got {dist}"


def test_sphere_center_exact_distance():
    """Test that distance at center equals -radius."""
    sphere = Sphere(radius=1.0)
    p = jnp.array([0.0, 0.0, 0.0])
    assert jnp.isclose(sphere(p), -1.0)


def test_sphere_surface_point_exact():
    """Test exact distance on surface."""
    sphere = Sphere(radius=3.0)
    p = jnp.array([3.0, 0.0, 0.0])
    assert jnp.isclose(sphere(p), 0.0, atol=1e-5)


def test_sphere_isotropic():
    """Test that sphere is isotropic (same distance in all directions)."""
    sphere = Sphere(radius=1.0)

    # Test points at same distance from origin
    points = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7071, 0.7071, 0.0],  # ~1/√2
    ]

    distances = [sphere(jnp.array(p)) for p in points]

    # All should be approximately zero (on surface)
    for dist in distances:
        assert jnp.isclose(dist, 0.0, atol=1e-3)
