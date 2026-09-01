"""Tests for from_point construction function."""

import jax.numpy as jnp

from cadjoint.construction import from_point
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.primitives.sphere import Sphere


def test_from_point_basic():
    """Test basic point to sphere conversion."""
    center = Vector([0, 0, 0], name="center")
    radius = Scalar(1.0, name="radius")

    sphere = from_point(center, radius)

    assert isinstance(sphere, Sphere)
    assert sphere._construction_method == "from_point"
    assert sphere._source_point is center


def test_from_point_with_float_radius():
    """Test sphere creation with float radius."""
    center = Vector([1, 2, 3], name="c")
    sphere = from_point(center, radius=2.5)

    assert isinstance(sphere, Sphere)
    assert sphere.params["radius"].value == 2.5
    assert sphere._source_point is center


def test_from_point_with_scalar_radius():
    """Test sphere creation with Scalar radius parameter."""
    center = Vector([0, 0, 0], name="c")
    radius = Scalar(3.5, name="r")

    sphere = from_point(center, radius=radius)

    assert isinstance(sphere, Sphere)
    assert sphere.params["radius"] is radius
    assert sphere.params["radius"].value == 3.5


def test_from_point_with_free_parameters():
    """Test that free parameters are preserved."""
    center = Vector([1, 1, 1], free=True, name="center")
    radius = Scalar(2.0, free=True, name="radius")

    sphere = from_point(center, radius)

    # Sphere should reference the same parameters
    assert sphere.params["radius"] is radius
    assert sphere._source_point is center
    assert center.free
    assert radius.free


def test_from_point_at_origin():
    """Test sphere at origin."""
    center = Vector([0, 0, 0], name="origin")
    sphere = from_point(center, radius=1.0)

    assert jnp.allclose(sphere._source_point.value, jnp.array([0, 0, 0]))
    assert sphere.params["radius"].value == 1.0


def test_from_point_offset():
    """Test sphere at non-origin position."""
    center = Vector([10, -5, 3], name="offset")
    radius = Scalar(7.5, name="r")

    sphere = from_point(center, radius=radius)

    assert jnp.allclose(sphere._source_point.xyz, jnp.array([10, -5, 3]))
    assert sphere.params["radius"].value == 7.5


def test_from_point_small_sphere():
    """Test very small sphere."""
    center = Vector([0, 0, 0], name="c")
    sphere = from_point(center, radius=0.01)

    assert sphere.params["radius"].value == 0.01


def test_from_point_large_sphere():
    """Test large sphere."""
    center = Vector([0, 0, 0], name="c")
    radius = Scalar(1000.0, name="big_r")
    sphere = from_point(center, radius=radius)

    assert sphere.params["radius"].value == 1000.0
