"""Tests for SDF compilation to pure JAX functions."""

import jax.numpy as jnp
import pytest

from jaxcad import functionalize, functionalize_scene
from jaxcad.extraction import extract_parameters
from jaxcad.geometry.parameters import Scalar, Vector
from jaxcad.render.material import Material
from jaxcad.sdf.boolean import Difference, Intersection, Union, Xor
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


def test_functionalized_material_preserves_every_property_and_free_values():
    reflectivity = Scalar(0.8, free=True, name="reflectivity", bounds=(0.0, 1.0))
    sphere = Sphere(1.0, material=Material(reflectivity=reflectivity))
    free, fixed, _ = extract_parameters(sphere)
    _, material_fn = functionalize_scene(sphere)(free, fixed)

    material = material_fn(jnp.zeros(3))

    assert set(material) == set(sphere.material_at(jnp.zeros(3)))
    assert jnp.isclose(material["reflectivity"], 0.8)

    _, updated_material_fn = functionalize_scene(sphere)({**free, "reflectivity": 0.2}, fixed)
    assert jnp.isclose(updated_material_fn(jnp.zeros(3))["reflectivity"], 0.2)


@pytest.mark.parametrize(
    ("operation", "child_count"),
    [(Union, 3), (Intersection, 3), (Difference, 3), (Xor, 2)],
)
def test_functionalized_boolean_material_matches_direct_evaluation(operation, child_count):
    children = (
        Sphere(1.0, Material(color=[1.0, 0.0, 0.0])).translate([-0.5, 0.0, 0.0]),
        Sphere(0.9, Material(color=[0.0, 1.0, 0.0])).translate([0.5, 0.0, 0.0]),
        Sphere(0.7, Material(color=[0.0, 0.0, 1.0])).translate([0.0, 0.5, 0.0]),
    )[:child_count]
    shape = operation(children)
    free, fixed, _ = extract_parameters(shape)
    _, material_fn = functionalize_scene(shape)(free, fixed)
    point = jnp.array([0.2, 0.1, 0.0])

    direct = shape.material_at(point)
    compiled = material_fn(point)

    assert set(compiled) == set(direct)
    for key in direct:
        assert jnp.allclose(compiled[key], direct[key])
