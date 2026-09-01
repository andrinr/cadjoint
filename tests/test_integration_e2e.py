"""End-to-end integration tests for the complete cadjoint system.

These tests verify that all layers work together:
1. Geometry layer (Line, Circle)
2. Constraints layer (DistanceConstraint, AngleConstraint, etc.)
3. Construction layer (from_line, from_circle, from_point)
4. Compiler layer (extract_parameters, compile_functionalize)
"""

import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import DistanceConstraint, all_parameters, total_dof_reduction
from cadjoint.construction import from_circle, from_line, from_point
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.geometry.primitives import Circle, Line
from cadjoint.sdf.primitives import Sphere


def test_e2e_constrained_geometry_to_sdf():
    """Test creating constrained geometry and converting to SDF."""
    # Layer 1: Create geometry with free parameters
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([0, 0, 5], free=True, name="p2")

    # Layer 2: Add distance constraint
    constraints = [DistanceConstraint(p1, p2, distance=5.0)]

    # Verify constraint DOF reduction
    params = all_parameters(constraints)
    total_dof = sum(3 if isinstance(p, Vector) else 1 for p in params)
    assert total_dof == 6  # 2 points * 3 DOF
    assert total_dof_reduction(constraints) == 1  # Distance constraint removes 1 DOF

    # Layer 3: Convert to SDF via construction
    line = Line(start=p1, end=p2)
    capsule = from_line(line, radius=0.5)

    # Verify construction preserves references
    assert capsule._source_geometry is line
    assert capsule._construction_method == "from_line"


def test_e2e_parametric_sphere_optimization():
    """Test full pipeline: geometry → SDF → compilation → gradient."""
    # Layer 1: Parametric geometry
    center = Vector([0, 0, 0], free=False, name="center")
    radius = Scalar(1.0, free=True, name="radius")

    # Layer 3: Construct SDF
    sphere = from_point(center, radius)

    # Layer 4: Extract and compile
    free_params, fixed_params, _ = extract_parameters(sphere)

    # Should have one free parameter (radius)
    assert len(free_params) == 1
    assert "radius" in free_params

    # Compile to pure function
    sdf_fn = functionalize(sphere)

    # Test gradient computation
    def loss_fn(r):
        params = {"radius": r}
        point = jnp.array([2.0, 0.0, 0.0])
        return sdf_fn(params, {})(point) ** 2

    # Compute gradient
    grad = jax.grad(loss_fn)(1.0)

    # Gradient should be non-zero
    assert not jnp.isclose(grad, 0.0)


def test_e2e_constrained_two_spheres():
    """Test two spheres with distance constraint between centers."""
    # Layer 1: Two points with constraints
    center1 = Vector([0, 0, 0], free=True, name="c1")
    center2 = Vector([3, 0, 0], free=True, name="c2")

    # Layer 2: Distance constraint
    constraints = [DistanceConstraint(center1, center2, distance=3.0)]

    # Total DOF: 6 (2 points * 3), DOF reduction: 1
    params = all_parameters(constraints)
    total_dof = sum(3 if isinstance(p, Vector) else 1 for p in params)
    assert total_dof == 6
    assert total_dof_reduction(constraints) == 1

    # Layer 3: Construct SDFs
    radius1 = Scalar(1.0, free=False, name="r1")
    radius2 = Scalar(0.5, free=False, name="r2")

    sphere1 = from_point(center1, radius1)
    sphere2 = from_point(center2, radius2)

    # Verify construction
    assert sphere1._source_point is center1
    assert sphere2._source_point is center2


def test_e2e_circle_to_cylinder_with_constraints():
    """Test circle with radius constraint, converted to cylinder."""
    # Layer 1: Parametric circle
    center = Vector([0, 0, 0], free=True, name="center")
    radius = Scalar(2.0, free=True, name="radius")
    normal = Vector([0, 0, 1], free=False, name="normal")

    circle = Circle(center=center, radius=radius, normal=normal)

    # Verify circle properties
    area = circle.area()
    assert jnp.isclose(area, jnp.pi * 4.0)

    # Layer 3: Convert to cylinder
    height = Scalar(5.0, free=True, name="height")
    cylinder = from_circle(circle, height=height)

    # Verify construction preserves parameters
    assert cylinder.params["radius"] is radius
    assert cylinder.params["height"] is height

    # Layer 4: Compile
    sdf_fn = functionalize(cylinder)

    # Test evaluation
    point = jnp.array([0.0, 0.0, 0.0])
    dist = sdf_fn({"radius": 2.0, "height": 5.0}, {})(point)

    # At origin, should be inside cylinder
    assert dist < 0


def test_e2e_line_properties_and_capsule():
    """Test line geometry properties and capsule construction."""
    # Layer 1: Parametric line
    p1 = Vector([0, 0, -2], free=True, name="p1")
    p2 = Vector([0, 0, 2], free=True, name="p2")

    line = Line(start=p1, end=p2)

    # Test geometry properties
    assert jnp.isclose(line.length(), 4.0)
    midpoint = line.midpoint()
    assert jnp.allclose(midpoint, jnp.array([0, 0, 0]))

    # Layer 3: Convert to capsule
    radius = Scalar(0.5, free=True, name="radius")
    capsule = from_line(line, radius=radius)

    # Verify capsule properties
    assert capsule.params["radius"] is radius
    assert jnp.isclose(capsule.params["height"].value, 2.0)  # Half of line length

    # Layer 4: Extract parameters
    free_params, fixed_params, _ = extract_parameters(capsule)

    # Should have one free parameter (radius)
    assert "radius" in free_params


def test_e2e_multi_constraint_system():
    """Test complex system with multiple constraints."""
    # Layer 1: Three points forming a triangle
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([3, 0, 0], free=True, name="p2")
    p3 = Vector([0, 4, 0], free=True, name="p3")

    # Layer 2: Multiple constraints
    constraints = [
        DistanceConstraint(p1, p2, distance=3.0),
        DistanceConstraint(p1, p3, distance=4.0),
        DistanceConstraint(p2, p3, distance=5.0),
    ]

    # Total DOF: 9 (3 points * 3)
    # Constraints: 3 distance constraints = 3 DOF removed
    params = all_parameters(constraints)
    total_dof = sum(3 if isinstance(p, Vector) else 1 for p in params)
    assert total_dof == 9
    assert total_dof_reduction(constraints) == 3

    # Verify constraint count
    assert len(constraints) == 3


def test_e2e_gradient_based_optimization_setup():
    """Test that the system is ready for gradient-based optimization."""
    # Create a simple parametric sphere
    radius = Scalar(1.0, free=True, name="radius")
    sphere = Sphere(radius=radius)

    # Extract parameters
    free_params, fixed_params, _ = extract_parameters(sphere)

    # Compile to function
    sdf_fn = functionalize(sphere)

    # Define loss function (distance to target point)
    def loss(r):
        params_dict = {"radius": r}
        target_point = jnp.array([2.0, 0.0, 0.0])
        distance = sdf_fn(params_dict, {})(target_point)
        return distance**2

    # Compute gradient using JAX
    grad_fn = jax.grad(loss)
    gradient = grad_fn(1.0)

    # Gradient should exist and be finite
    assert jnp.isfinite(gradient)

    # Can also compute hessian
    hess_fn = jax.hessian(loss)
    hessian = hess_fn(1.0)
    assert jnp.isfinite(hessian)


def test_e2e_parameter_reference_sharing():
    """Test that parameter references are properly shared across layers."""
    # Layer 1: Create shared parameters
    center = Vector([0, 0, 0], free=True, name="center")
    radius = Scalar(2.0, free=True, name="radius")

    # Layer 3: Multiple constructions share the same parameters
    sphere1 = from_point(center, radius)

    # Verify parameter identity (same object, not copy)
    assert sphere1.params["radius"] is radius
    assert sphere1._source_point is center

    # If we modify the parameter value, it should affect the sphere
    original_value = radius.value
    radius.value = jnp.array(3.0)

    # The sphere's radius should reference the updated value
    assert sphere1.params["radius"].value == 3.0

    # Restore
    radius.value = original_value
