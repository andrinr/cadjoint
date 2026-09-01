"""Tests for from_line construction function."""

import jax.numpy as jnp

from cadjoint.construction import from_line
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.geometry.primitives import Line
from cadjoint.sdf.primitives.capsule import Capsule


def test_from_line_basic():
    """Test basic line to capsule conversion."""
    p1 = Vector([0, 0, -2.5], name="p1")
    p2 = Vector([0, 0, 2.5], name="p2")
    line = Line(start=p1, end=p2)

    capsule = from_line(line, radius=0.5)

    assert isinstance(capsule, Capsule)
    assert capsule._construction_method == "from_line"
    assert capsule._source_geometry is line
    # Line length is 5, so half-height should be 2.5
    assert jnp.isclose(capsule.params["height"].value, 2.5)


def test_from_line_with_scalar_radius():
    """Test capsule creation with Scalar radius parameter."""
    p1 = Vector([0, 0, -3], name="start")
    p2 = Vector([0, 0, 3], name="end")
    line = Line(start=p1, end=p2)

    radius = Scalar(1.5, name="radius")
    capsule = from_line(line, radius=radius)

    assert isinstance(capsule, Capsule)
    assert capsule.params["radius"] is radius
    assert capsule.params["radius"].value == 1.5
    # Line length is 6, half-height is 3
    assert jnp.isclose(capsule.params["height"].value, 3.0)


def test_from_line_with_free_points():
    """Test that capsule is created from line with free points."""
    p1 = Vector([0, 0, -1], free=True, name="p1")
    p2 = Vector([0, 0, 1], free=True, name="p2")
    line = Line(start=p1, end=p2)

    capsule = from_line(line, radius=0.3)

    # Capsule stores reference to source geometry
    assert capsule._source_geometry.start is p1
    assert capsule._source_geometry.end is p2
    assert p1.free
    assert p2.free


def test_from_line_vertical():
    """Test capsule from vertical line."""
    p1 = Vector([0, 0, 0], name="bottom")
    p2 = Vector([0, 0, 10], name="top")
    line = Line(start=p1, end=p2)

    capsule = from_line(line, radius=2.0)

    # Line length is 10, so half-height is 5
    assert jnp.isclose(capsule.params["height"].value, 5.0)
    assert capsule.params["radius"].value == 2.0


def test_from_line_short():
    """Test capsule from short line."""
    p1 = Vector([0, 0, -0.5], name="p1")
    p2 = Vector([0, 0, 0.5], name="p2")
    line = Line(start=p1, end=p2)

    radius = Scalar(0.75, name="r")
    capsule = from_line(line, radius=radius)

    # Line length is 1, half-height is 0.5
    assert jnp.isclose(capsule.params["height"].value, 0.5)
    assert capsule.params["radius"].value == 0.75


def test_from_line_zero_length():
    """Test capsule from zero-length line (becomes sphere)."""
    p1 = Vector([0, 0, 5], name="p1")
    p2 = Vector([0, 0, 5], name="p2")
    line = Line(start=p1, end=p2)

    capsule = from_line(line, radius=1.0)

    # Line length is 0, half-height is 0
    assert jnp.isclose(capsule.params["height"].value, 0.0)
    # This becomes a sphere with the given radius
    assert capsule.params["radius"].value == 1.0
