"""Tests for Box SDF primitive."""

import jax.numpy as jnp
import pytest

from cadjoint.sdf.primitives import Box


@pytest.mark.parametrize(
    "size,point,expected_sign",
    [
        ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], "negative"),  # Center
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], "zero"),  # Corner
        ([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], "zero"),  # Face center
        ([1.0, 1.0, 1.0], [2.0, 0.0, 0.0], "positive"),  # Outside
        ([2.0, 1.0, 0.5], [0.0, 0.0, 0.0], "negative"),  # Different sizes, center
        ([2.0, 1.0, 0.5], [2.0, 1.0, 0.5], "zero"),  # Different sizes, corner
    ],
)
def test_box_distance(size, point, expected_sign):
    """Test that box SDF returns correct sign."""
    box = Box(size=jnp.array(size))
    p = jnp.array(point)
    dist = box(p)

    if expected_sign == "negative":
        assert dist < 0, f"Expected negative distance at {point}, got {dist}"
    elif expected_sign == "zero":
        assert jnp.isclose(dist, 0.0, atol=1e-5), f"Expected ~0 distance at {point}, got {dist}"
    elif expected_sign == "positive":
        assert dist > 0, f"Expected positive distance at {point}, got {dist}"


def test_box_symmetry():
    """Test that box is symmetric."""
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))

    # Points symmetric about origin should have same distance
    p1 = jnp.array([0.5, 0.0, 0.0])
    p2 = jnp.array([-0.5, 0.0, 0.0])

    assert jnp.isclose(box(p1), box(p2))


def test_box_anisotropic():
    """Test box with different dimensions."""
    box = Box(size=jnp.array([2.0, 1.0, 0.5]))

    # On x-face
    p_x = jnp.array([2.0, 0.0, 0.0])
    assert jnp.isclose(box(p_x), 0.0, atol=1e-5)

    # On y-face
    p_y = jnp.array([0.0, 1.0, 0.0])
    assert jnp.isclose(box(p_y), 0.0, atol=1e-5)

    # On z-face
    p_z = jnp.array([0.0, 0.0, 0.5])
    assert jnp.isclose(box(p_z), 0.0, atol=1e-5)
