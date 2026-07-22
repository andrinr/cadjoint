"""Tests for SDF compilation to pure JAX functions."""

import jax.numpy as jnp

from jaxcad import extract_parameters, functionalize, functionalize_scene
from jaxcad.geometry.parameters import Scalar, Vector
from jaxcad.render import Material
from jaxcad.sdf.primitives.box import Box
from jaxcad.sdf.primitives.sphere import Sphere


def test_compile_sphere_basic():
    """Test compiling a sphere to a function."""
    radius = Scalar(1.0, free=True, name="radius")
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)

    # Query at origin (inside sphere)
    point = jnp.array([0.0, 0.0, 0.0])
    free_vals = {"radius": 1.0}
    fixed_vals = {}

    distance = sdf_fn(free_vals, fixed_vals)(point)

    # At origin, distance should be -radius
    assert jnp.isclose(distance, -1.0)


def test_compile_sphere_outside():
    """Test compiled sphere SDF for point outside."""
    radius = Scalar(1.0, free=False, name="radius")
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)

    # Query at distance 2 from origin
    point = jnp.array([2.0, 0.0, 0.0])
    free_vals = {}
    fixed_vals = {"sphere_0.radius": 1.0}

    distance = sdf_fn(free_vals, fixed_vals)(point)

    # Distance should be 2 - 1 = 1
    assert jnp.isclose(distance, 1.0)


def test_compile_with_parameter_variation():
    """Test that compiled function responds to parameter changes."""
    radius = Scalar(1.0, free=True, name="radius")
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)
    point = jnp.array([2.0, 0.0, 0.0])

    # Test with radius = 1.0
    dist1 = sdf_fn({"radius": 1.0}, {})(point)
    assert jnp.isclose(dist1, 1.0)

    # Test with radius = 1.5
    dist2 = sdf_fn({"radius": 1.5}, {})(point)
    assert jnp.isclose(dist2, 0.5)

    # Test with radius = 2.0
    dist3 = sdf_fn({"radius": 2.0}, {})(point)
    assert jnp.isclose(dist3, 0.0)


def test_compile_box():
    """Test compiling a box to a function."""
    size = Vector([1.0, 1.0, 1.0], free=True, name="size")
    box = Box(size=size)

    sdf_fn = functionalize(box)

    # Query at origin (inside box)
    point = jnp.array([0.0, 0.0, 0.0])
    free_vals = {"size": jnp.array([1.0, 1.0, 1.0])}
    fixed_vals = {}

    distance = sdf_fn(free_vals, fixed_vals)(point)

    # Should be inside the box
    assert distance < 0


def test_compile_consistency():
    """Test that compiled function gives same results as direct evaluation."""
    radius = Scalar(1.5, free=False, name="radius")
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)

    # Test multiple points
    test_points = [
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([2.0, 0.0, 0.0]),
        jnp.array([1.0, 1.0, 0.0]),
    ]

    for point in test_points:
        # Direct evaluation using __call__
        direct_dist = sphere(point)

        # Compiled evaluation
        compiled_dist = sdf_fn({}, {"sphere_0.radius": 1.5})(point)

        # Should be very close
        assert jnp.isclose(direct_dist, compiled_dist, atol=1e-6)


def test_functionalized_material_preserves_forward_render_properties():
    material = Material(reflectivity=0.7, roughness=0.2, metallic=0.9)
    sphere = Sphere(radius=1.0, material=material)
    free, fixed, _ = extract_parameters(sphere)
    _, material_fn = functionalize_scene(sphere)(free, fixed)

    compiled = material_fn(jnp.array([1.0, 0.0, 0.0]))
    assert set(compiled) == {
        "color",
        "roughness",
        "metallic",
        "opacity",
        "ior",
        "reflectivity",
    }
    assert jnp.isclose(compiled["reflectivity"], 0.7)
