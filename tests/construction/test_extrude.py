"""Tests for extrude construction function."""

import jax.numpy as jnp

from cadjoint.construction import extrude
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.geometry.primitives import Rectangle
from cadjoint.sdf.primitives.box import Box


def test_extrude_basic():
    """Test basic rectangle extrusion."""
    rect = Rectangle(
        center=Vector([0, 0, 0], name="center"),
        width=Scalar(2.0, name="width"),
        height=Scalar(1.0, name="height"),
        normal=Vector([0, 0, 1], name="normal"),
    )

    box = extrude(rect, depth=3.0)

    assert isinstance(box, Box)
    assert box._construction_method == "extrude"
    assert box._source_geometry is rect


def test_extrude_with_scalar_depth():
    """Test extrusion with Scalar depth parameter."""
    rect = Rectangle(
        center=Vector([0, 0, 0], name="center"),
        width=Scalar(4.0, name="width"),
        height=Scalar(2.0, name="height"),
        normal=Vector([0, 0, 1], name="normal"),
    )

    depth = Scalar(5.0, name="depth")
    box = extrude(rect, depth=depth)

    assert isinstance(box, Box)
    # Size should be [width/2, height/2, depth/2] = [2, 1, 2.5]
    expected_size = jnp.array([2.0, 1.0, 2.5])
    assert jnp.allclose(box.params["size"].xyz, expected_size)


def test_extrude_with_free_parameters():
    """Test extrusion preserves free parameter references."""
    center = Vector([0, 0, 0], free=True, name="center")
    width = Scalar(2.0, free=True, name="width")
    height = Scalar(1.0, free=True, name="height")

    rect = Rectangle(
        center=center, width=width, height=height, normal=Vector([0, 0, 1], name="normal")
    )
    box = extrude(rect, depth=3.0)

    # The box should reference the original rectangle
    assert box._source_geometry.center is center
    assert box._source_geometry.width is width
    assert box._source_geometry.height is height


def test_extrude_dimensions():
    """Test that extruded box has correct dimensions."""
    rect = Rectangle(
        center=Vector([0, 0, 0], name="c"),
        width=Scalar(6.0, name="w"),
        height=Scalar(4.0, name="h"),
        normal=Vector([0, 0, 1], name="n"),
    )

    box = extrude(rect, depth=2.0)

    # Box size should be [3, 2, 1] (half-sizes)
    expected_size = jnp.array([3.0, 2.0, 1.0])
    assert jnp.allclose(box.params["size"].xyz, expected_size)
