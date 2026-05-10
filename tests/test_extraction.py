"""Tests for parameter extraction from SDF trees."""

import jax.numpy as jnp

from jaxcad import extract_parameters
from jaxcad.geometry.parameters import Scalar, Vector
from jaxcad.sdf.primitives.box import Box
from jaxcad.sdf.primitives.sphere import Sphere


def test_extract_free_parameter():
    """Test extraction of a single free parameter."""
    radius = Scalar(1.0, free=True, name="radius")
    sphere = Sphere(radius=radius)

    free_params, fixed_params, metadata = extract_parameters(sphere)

    assert len(free_params) == 1
    assert "radius" in free_params
    assert jnp.array_equal(free_params["radius"], radius.value)
    assert metadata["radius"] is radius


def test_extract_fixed_parameter():
    """Test extraction of a fixed parameter."""
    radius = Scalar(2.0, free=False, name="radius")
    sphere = Sphere(radius=radius)

    free_params, fixed_params, metadata = extract_parameters(sphere)

    assert len(free_params) == 0
    assert "sphere_0.radius" in fixed_params


def test_extract_mixed_parameters():
    """Test extraction of mixed free and fixed parameters."""
    size = Vector([1, 2, 3], free=True, name="size")
    box = Box(size=size)

    free_params, fixed_params, metadata = extract_parameters(box)

    assert "size" in free_params
    assert jnp.array_equal(free_params["size"], size.value)
    assert metadata["size"] is size


def test_extract_multiple_primitives():
    """Test parameter extraction maintains unique node IDs."""
    r1 = Scalar(1.0, free=True, name="r1")
    r2 = Scalar(2.0, free=True, name="r2")

    sphere1 = Sphere(radius=r1)
    sphere2 = Sphere(radius=r2)

    # Extract from each separately
    free1, _, _ = extract_parameters(sphere1)
    free2, _, _ = extract_parameters(sphere2)

    assert "r1" in free1
    assert "r2" in free2


def test_extract_no_free_parameters():
    """Test extraction when no parameters are free."""
    radius = Scalar(1.0, free=False, name="radius")
    sphere = Sphere(radius=radius)

    free_params, fixed_params, metadata = extract_parameters(sphere)

    assert len(free_params) == 0
    assert "sphere_0.radius" in fixed_params
