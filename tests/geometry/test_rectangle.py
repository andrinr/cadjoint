"""Tests for Rectangle geometric entity."""

import jax.numpy as jnp

from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.geometry.primitives import Rectangle


def test_rectangle_basic():
    """Test basic rectangle creation."""
    center = Vector([0, 0, 0], name="c")
    width = Scalar(2.0, name="w")
    height = Scalar(1.0, name="h")
    normal = Vector([0, 0, 1], name="n")  # XY plane

    rect = Rectangle(center=center, width=width, height=height, normal=normal)

    assert rect.center is center
    assert rect.width is width
    assert rect.height is height


def test_rectangle_from_scalars():
    """Test rectangle creation from scalar values."""
    rect = Rectangle(center=[0, 0, 0], width=4.0, height=2.0, normal=[0, 0, 1])

    assert isinstance(rect.center, Vector)
    assert isinstance(rect.width, Scalar)
    assert isinstance(rect.height, Scalar)
    assert rect.width.value == 4.0
    assert rect.height.value == 2.0


def test_rectangle_sample_center():
    """Test sampling at rectangle center."""
    rect = Rectangle(center=[5, 5, 5], width=2.0, height=1.0, normal=[0, 0, 1])

    center_point = rect.sample(0.5, 0.5)
    assert jnp.allclose(center_point, jnp.array([5, 5, 5]))


def test_rectangle_corners():
    """Test rectangle corner calculation."""
    rect = Rectangle(center=[0, 0, 0], width=2.0, height=1.0, normal=[0, 0, 1])

    corners = rect.corners()
    assert corners.shape == (4, 3)

    # Check that corners are roughly at the right positions
    # For a 2x1 rectangle centered at origin in XY plane:
    # corners should be at (±1, ±0.5, 0)
    rect.corner(0)
    rect.corner(1)
    rect.corner(2)
    rect.corner(3)

    # Each corner should be on the rectangle's plane
    for i in range(4):
        corner = rect.corner(i)
        # Distance from center
        to_corner = corner - rect.center.xyz
        # Should be in the plane (dot with normal should be ~0)
        assert jnp.abs(jnp.dot(to_corner, rect.normal.xyz)) < 1e-6


def test_rectangle_sample_corners():
    """Test sampling at rectangle corners using parameters."""
    rect = Rectangle(center=[0, 0, 0], width=2.0, height=2.0, normal=[0, 0, 1])

    # Sample at corners using (u, v) parameters
    corner_00 = rect.sample(0.0, 0.0)  # Bottom-left
    corner_10 = rect.sample(1.0, 0.0)  # Bottom-right
    corner_11 = rect.sample(1.0, 1.0)  # Top-right
    corner_01 = rect.sample(0.0, 1.0)  # Top-left

    # All corners should be at distance sqrt(2) from center for a 2x2 square
    for corner in [corner_00, corner_10, corner_11, corner_01]:
        dist = jnp.linalg.norm(corner - rect.center.xyz)
        assert jnp.isclose(dist, jnp.sqrt(2), atol=1e-5)


def test_rectangle_local_frame():
    """Test that rectangle computes orthonormal local frame."""
    rect = Rectangle(center=[0, 0, 0], width=1.0, height=1.0, normal=[0, 0, 1])

    # U and V axes should be orthonormal
    u = rect.u_axis.xyz
    v = rect.v_axis.xyz
    n = rect.normal.xyz

    # Check normalization
    assert jnp.isclose(jnp.linalg.norm(u), 1.0)
    assert jnp.isclose(jnp.linalg.norm(v), 1.0)
    assert jnp.isclose(jnp.linalg.norm(n), 1.0)

    # Check orthogonality
    assert jnp.abs(jnp.dot(u, v)) < 1e-6
    assert jnp.abs(jnp.dot(u, n)) < 1e-6
    assert jnp.abs(jnp.dot(v, n)) < 1e-6


def test_rectangle_tilted():
    """Test rectangle with tilted normal."""
    # Rectangle in YZ plane (normal along X)
    rect = Rectangle(center=[0, 0, 0], width=2.0, height=1.0, normal=[1, 0, 0])

    # Center should still be at origin
    center = rect.sample(0.5, 0.5)
    assert jnp.allclose(center, jnp.array([0, 0, 0]), atol=1e-6)

    # Corners should have x=0 (in YZ plane)
    for i in range(4):
        corner = rect.corner(i)
        assert jnp.abs(corner[0]) < 1e-6


def test_rectangle_with_free_parameters():
    """Test rectangle with free parameters."""
    center = Vector([0, 0, 0], free=True, name="center")
    width = Scalar(3.0, free=True, name="width")
    height = Scalar(2.0, free=True, name="height")

    rect = Rectangle(center=center, width=width, height=height, normal=[0, 0, 1])

    assert rect.center.free
    assert rect.width.free
    assert rect.height.free


def test_rectangle_sample_grid():
    """Test sampling a grid of points on rectangle."""
    rect = Rectangle(center=[0, 0, 0], width=4.0, height=2.0, normal=[0, 0, 1])

    # Sample a 3x3 grid
    for u in [0.0, 0.5, 1.0]:
        for v in [0.0, 0.5, 1.0]:
            point = rect.sample(u, v)
            # Point should be in the rectangle's plane
            to_point = point - rect.center.xyz
            assert jnp.abs(jnp.dot(to_point, rect.normal.xyz)) < 1e-6
