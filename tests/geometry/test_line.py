"""Tests for Line geometric entity."""

import jax.numpy as jnp

from cadjoint.geometry.parameters import Vector
from cadjoint.geometry.primitives import Line


def test_line_basic():
    """Test basic line creation."""
    p1 = Vector([0, 0, 0], name="p1")
    p2 = Vector([1, 0, 0], name="p2")
    line = Line(start=p1, end=p2)

    assert line.start is p1
    assert line.end is p2


def test_line_from_arrays():
    """Test line creation from raw arrays."""
    line = Line(start=[0, 0, 0], end=[5, 0, 0])

    assert isinstance(line.start, Vector)
    assert isinstance(line.end, Vector)
    assert jnp.allclose(line.start.xyz, jnp.array([0, 0, 0]))
    assert jnp.allclose(line.end.xyz, jnp.array([5, 0, 0]))


def test_line_length():
    """Test line length calculation."""
    line = Line(start=[0, 0, 0], end=[3, 4, 0])
    assert jnp.isclose(line.length(), 5.0)

    line2 = Line(start=[1, 2, 3], end=[4, 6, 3])
    assert jnp.isclose(line2.length(), 5.0)  # 3-4-5 triangle in XY


def test_line_direction():
    """Test line direction vector."""
    line = Line(start=[0, 0, 0], end=[10, 0, 0])

    # Normalized direction
    direction = line.direction(normalized=True)
    assert jnp.allclose(direction, jnp.array([1, 0, 0]))
    assert jnp.isclose(jnp.linalg.norm(direction), 1.0)

    # Unnormalized direction
    direction_raw = line.direction(normalized=False)
    assert jnp.allclose(direction_raw, jnp.array([10, 0, 0]))


def test_line_midpoint():
    """Test line midpoint."""
    line = Line(start=[0, 0, 0], end=[4, 6, 8])
    midpoint = line.midpoint()
    assert jnp.allclose(midpoint, jnp.array([2, 3, 4]))


def test_line_sample():
    """Test sampling points along line."""
    line = Line(start=[0, 0, 0], end=[10, 0, 0])

    # Start point
    assert jnp.allclose(line.sample(0.0), jnp.array([0, 0, 0]))

    # End point
    assert jnp.allclose(line.sample(1.0), jnp.array([10, 0, 0]))

    # Middle point
    assert jnp.allclose(line.sample(0.5), jnp.array([5, 0, 0]))

    # Quarter point
    assert jnp.allclose(line.sample(0.25), jnp.array([2.5, 0, 0]))


def test_line_tangent():
    """Test line tangent vector."""
    line = Line(start=[0, 0, 0], end=[5, 0, 0])

    # Tangent is constant for a line
    t1 = line.tangent(0.0)
    t2 = line.tangent(0.5)
    t3 = line.tangent(1.0)

    assert jnp.allclose(t1, jnp.array([1, 0, 0]))
    assert jnp.allclose(t2, jnp.array([1, 0, 0]))
    assert jnp.allclose(t3, jnp.array([1, 0, 0]))


def test_line_closest_point():
    """Test finding closest point on line."""
    line = Line(start=[0, 0, 0], end=[10, 0, 0])

    # Point above line
    closest = line.closest_point(jnp.array([5, 5, 0]))
    assert jnp.allclose(closest, jnp.array([5, 0, 0]))

    # Point before start
    closest = line.closest_point(jnp.array([-5, 0, 0]))
    assert jnp.allclose(closest, jnp.array([0, 0, 0]))

    # Point after end
    closest = line.closest_point(jnp.array([15, 0, 0]))
    assert jnp.allclose(closest, jnp.array([10, 0, 0]))


def test_line_distance_to_point():
    """Test distance from point to line."""
    line = Line(start=[0, 0, 0], end=[10, 0, 0])

    # Point on the line
    dist = line.distance_to_point(jnp.array([5, 0, 0]))
    assert jnp.isclose(dist, 0.0)

    # Point above the line
    dist = line.distance_to_point(jnp.array([5, 3, 0]))
    assert jnp.isclose(dist, 3.0)

    # Point in 3D space
    dist = line.distance_to_point(jnp.array([5, 3, 4]))
    assert jnp.isclose(dist, 5.0)  # 3-4-5 triangle


def test_line_with_free_parameters():
    """Test line with free parameters."""
    p1 = Vector([0, 0, 0], free=True, name="start")
    p2 = Vector([1, 1, 1], free=True, name="end")

    line = Line(start=p1, end=p2)

    assert line.start.free
    assert line.end.free
    assert line.start.name == "start"
    assert line.end.name == "end"


def test_line_degenerate():
    """Test degenerate line (start == end)."""
    line = Line(start=[5, 5, 5], end=[5, 5, 5])

    assert jnp.isclose(line.length(), 0.0)

    # Closest point should be the start/end point
    closest = line.closest_point(jnp.array([10, 10, 10]))
    assert jnp.allclose(closest, jnp.array([5, 5, 5]))


def test_line_diagonal():
    """Test diagonal line in 3D."""
    line = Line(start=[0, 0, 0], end=[1, 1, 1])

    length = line.length()
    assert jnp.isclose(length, jnp.sqrt(3))

    direction = line.direction(normalized=True)
    expected = jnp.array([1, 1, 1]) / jnp.sqrt(3)
    assert jnp.allclose(direction, expected)
