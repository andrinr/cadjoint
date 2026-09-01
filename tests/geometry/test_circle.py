"""Tests for Circle geometric entity."""

import jax.numpy as jnp
import pytest

from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.geometry.primitives import Circle


def test_circle_basic():
    """Test basic circle creation."""
    center = Vector([0, 0, 0], name="c")
    radius = Scalar(1.0, name="r")
    normal = Vector([0, 0, 1], name="n")

    circle = Circle(center=center, radius=radius, normal=normal)

    assert circle.center is center
    assert circle.radius is radius


def test_circle_from_scalars():
    """Test circle creation from scalar values."""
    circle = Circle(center=[0, 0, 0], radius=2.5, normal=[0, 0, 1])

    assert isinstance(circle.center, Vector)
    assert isinstance(circle.radius, Scalar)
    assert circle.radius.value == 2.5


def test_circle_sample():
    """Test sampling points on circle."""
    circle = Circle(center=[0, 0, 0], radius=1.0, normal=[0, 0, 1])

    # Sample at angle 0
    p0 = circle.sample(0.0)
    assert jnp.linalg.norm(p0 - circle.center.xyz) == pytest.approx(1.0)

    # Sample at π/2
    p1 = circle.sample(jnp.pi / 2)
    assert jnp.linalg.norm(p1 - circle.center.xyz) == pytest.approx(1.0)

    # Sample at π
    p2 = circle.sample(jnp.pi)
    assert jnp.linalg.norm(p2 - circle.center.xyz) == pytest.approx(1.0)


def test_circle_sample_uniform():
    """Test uniform sampling around circle."""
    circle = Circle(center=[0, 0, 0], radius=2.0, normal=[0, 0, 1])

    points = circle.sample_uniform(8)
    assert points.shape == (8, 3)

    # All points should be at radius distance from center
    for point in points:
        dist = jnp.linalg.norm(point - circle.center.xyz)
        assert jnp.isclose(dist, 2.0)


def test_circle_tangent():
    """Test tangent vector calculation."""
    circle = Circle(center=[0, 0, 0], radius=1.0, normal=[0, 0, 1])

    # Tangent at angle 0
    t0 = circle.tangent(0.0)
    # Should be perpendicular to radius
    p0 = circle.sample(0.0)
    radius_vec = p0 - circle.center.xyz
    assert jnp.abs(jnp.dot(t0, radius_vec)) < 1e-6

    # Tangent should be unit length
    assert jnp.isclose(jnp.linalg.norm(t0), 1.0)


def test_circle_area():
    """Test circle area calculation."""
    circle = Circle(center=[0, 0, 0], radius=2.0, normal=[0, 0, 1])

    area = circle.area()
    expected_area = jnp.pi * 4.0  # π * r²
    assert jnp.isclose(area, expected_area)


def test_circle_circumference():
    """Test circle circumference calculation."""
    circle = Circle(center=[0, 0, 0], radius=3.0, normal=[0, 0, 1])

    circumference = circle.circumference()
    expected = 2 * jnp.pi * 3.0
    assert jnp.isclose(circumference, expected)


def test_circle_closest_point():
    """Test finding closest point on circle."""
    circle = Circle(center=[0, 0, 0], radius=1.0, normal=[0, 0, 1])

    # Point outside circle in the plane
    closest = circle.closest_point(jnp.array([3, 0, 0]))
    assert jnp.allclose(closest, jnp.array([1, 0, 0]), atol=1e-6)

    # Point inside circle in the plane
    closest = circle.closest_point(jnp.array([0.5, 0, 0]))
    assert jnp.linalg.norm(closest - circle.center.xyz) == pytest.approx(1.0)


def test_circle_distance_to_point():
    """Test distance from point to circle."""
    circle = Circle(center=[0, 0, 0], radius=1.0, normal=[0, 0, 1])

    # Point on the circle
    dist = circle.distance_to_point(jnp.array([1, 0, 0]))
    assert jnp.isclose(dist, 0.0, atol=1e-6)

    # Point outside the circle
    dist = circle.distance_to_point(jnp.array([3, 0, 0]))
    assert jnp.isclose(dist, 2.0)

    # Point at center
    dist = circle.distance_to_point(jnp.array([0, 0, 0]))
    assert jnp.isclose(dist, 1.0)


def test_circle_local_frame():
    """Test that circle computes orthonormal local frame."""
    circle = Circle(center=[0, 0, 0], radius=1.0, normal=[0, 0, 1])

    u = circle.u_axis.xyz
    v = circle.v_axis.xyz
    n = circle.normal.xyz

    # Check normalization
    assert jnp.isclose(jnp.linalg.norm(u), 1.0)
    assert jnp.isclose(jnp.linalg.norm(v), 1.0)
    assert jnp.isclose(jnp.linalg.norm(n), 1.0)

    # Check orthogonality
    assert jnp.abs(jnp.dot(u, v)) < 1e-6
    assert jnp.abs(jnp.dot(u, n)) < 1e-6
    assert jnp.abs(jnp.dot(v, n)) < 1e-6


def test_circle_tilted():
    """Test circle with tilted normal."""
    # Circle in YZ plane (normal along X)
    circle = Circle(center=[0, 0, 0], radius=2.0, normal=[1, 0, 0])

    # Sample points should have x ≈ 0 (in YZ plane)
    for theta in jnp.linspace(0, 2 * jnp.pi, 8):
        point = circle.sample(theta)
        assert jnp.abs(point[0]) < 1e-6


def test_circle_with_free_parameters():
    """Test circle with free parameters."""
    center = Vector([0, 0, 0], free=True, name="center")
    radius = Scalar(1.5, free=True, name="radius")

    circle = Circle(center=center, radius=radius, normal=[0, 0, 1])

    assert circle.center.free
    assert circle.radius.free


def test_circle_offset_center():
    """Test circle with offset center."""
    circle = Circle(center=[10, 20, 30], radius=5.0, normal=[0, 0, 1])

    # Sample point
    point = circle.sample(0.0)

    # Distance from center should be radius
    dist = jnp.linalg.norm(point - circle.center.xyz)
    assert jnp.isclose(dist, 5.0)


def test_circle_large_radius():
    """Test circle with large radius."""
    circle = Circle(center=[0, 0, 0], radius=100.0, normal=[0, 0, 1])

    area = circle.area()
    expected_area = jnp.pi * 10000.0
    assert jnp.isclose(area, expected_area)


def test_circle_small_radius():
    """Test circle with small radius."""
    circle = Circle(center=[0, 0, 0], radius=0.01, normal=[0, 0, 1])

    circumference = circle.circumference()
    expected = 2 * jnp.pi * 0.01
    assert jnp.isclose(circumference, expected)
