"""Tests for from_circle construction function."""

from cadjoint.construction import from_circle
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.geometry.primitives import Circle
from cadjoint.sdf.primitives.cylinder import Cylinder


def test_from_circle_basic():
    """Test basic circle to cylinder conversion."""
    center = Vector([0, 0, 0], name="center")
    radius = Scalar(1.0, name="radius")
    circle = Circle(center=center, radius=radius, normal=Vector([0, 0, 1], name="n"))

    cylinder = from_circle(circle, height=5.0)

    assert isinstance(cylinder, Cylinder)
    assert cylinder._construction_method == "from_circle"
    assert cylinder._source_geometry is circle
    assert cylinder.params["radius"] is radius


def test_from_circle_with_scalar_height():
    """Test cylinder creation with Scalar height parameter."""
    center = Vector([0, 0, 0], name="c")
    radius = Scalar(2.5, name="r")
    circle = Circle(center=center, radius=radius, normal=Vector([0, 0, 1], name="n"))

    height = Scalar(10.0, name="height")
    cylinder = from_circle(circle, height=height)

    assert isinstance(cylinder, Cylinder)
    assert cylinder.params["height"] is height
    assert cylinder.params["height"].value == 10.0


def test_from_circle_with_free_parameters():
    """Test that free parameters are preserved."""
    center = Vector([0, 0, 0], free=True, name="center")
    radius = Scalar(3.0, free=True, name="radius")
    circle = Circle(center=center, radius=radius, normal=Vector([0, 0, 1], name="n"))

    cylinder = from_circle(circle, height=7.0)

    # Cylinder should reference the same radius
    assert cylinder.params["radius"] is radius
    assert radius.free
    assert cylinder._source_geometry.center is center


def test_from_circle_small():
    """Test cylinder from small circle."""
    center = Vector([0, 0, 0], name="c")
    radius = Scalar(0.1, name="r")
    circle = Circle(center=center, radius=radius, normal=Vector([0, 0, 1], name="n"))

    cylinder = from_circle(circle, height=0.5)

    assert cylinder.params["radius"].value == 0.1
    assert cylinder.params["height"].value == 0.5


def test_from_circle_large():
    """Test cylinder from large circle."""
    center = Vector([5, 5, 5], name="c")
    radius = Scalar(50.0, name="r")
    circle = Circle(center=center, radius=radius, normal=Vector([0, 0, 1], name="n"))

    height = Scalar(100.0, name="h")
    cylinder = from_circle(circle, height=height)

    assert cylinder.params["radius"].value == 50.0
    assert cylinder.params["height"].value == 100.0
