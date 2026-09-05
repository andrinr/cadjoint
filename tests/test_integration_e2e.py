"""End-to-end integration tests across the layers.

These tests verify that the layers work together:
1. Parameters (Vector, Scalar)
2. Constraints (DistanceConstraint, DOF accounting)
3. SDF primitives sharing those parameters
4. Compilation (extract_parameters, functionalize) and gradients
"""

import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import DistanceConstraint, all_parameters, total_dof_reduction
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.primitives import Sphere


def test_e2e_parametric_sphere_optimization():
    """Test full pipeline: parameters → SDF → compilation → gradient."""
    radius = Scalar(1.0, free=True, name="radius")
    sphere = Sphere(radius=radius)

    free_params, fixed_params, _ = extract_parameters(sphere)
    assert len(free_params) == 1
    assert "radius" in free_params

    sdf_fn = functionalize(sphere)

    def loss_fn(r):
        point = jnp.array([2.0, 0.0, 0.0])
        return sdf_fn({"radius": r}, {})(point) ** 2

    grad = jax.grad(loss_fn)(1.0)
    assert not jnp.isclose(grad, 0.0)


def test_e2e_constrained_two_points():
    """Two points with a distance constraint between them."""
    center1 = Vector([0, 0, 0], free=True, name="c1")
    center2 = Vector([3, 0, 0], free=True, name="c2")
    constraints = [DistanceConstraint(center1, center2, distance=3.0)]

    params = all_parameters(constraints)
    total_dof = sum(3 if isinstance(p, Vector) else 1 for p in params)
    assert total_dof == 6
    assert total_dof_reduction(constraints) == 1


def test_e2e_multi_constraint_system():
    """Three points forming a triangle held by three distances."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([3, 0, 0], free=True, name="p2")
    p3 = Vector([0, 4, 0], free=True, name="p3")
    constraints = [
        DistanceConstraint(p1, p2, distance=3.0),
        DistanceConstraint(p1, p3, distance=4.0),
        DistanceConstraint(p2, p3, distance=5.0),
    ]

    params = all_parameters(constraints)
    total_dof = sum(3 if isinstance(p, Vector) else 1 for p in params)
    assert total_dof == 9
    assert total_dof_reduction(constraints) == 3


def test_e2e_gradient_based_optimization_setup():
    """The compiled SDF is differentiable to second order."""
    radius = Scalar(1.0, free=True, name="radius")
    sphere = Sphere(radius=radius)
    sdf_fn = functionalize(sphere)

    def loss(r):
        target_point = jnp.array([2.0, 0.0, 0.0])
        return sdf_fn({"radius": r}, {})(target_point) ** 2

    assert jnp.isfinite(jax.grad(loss)(1.0))
    assert jnp.isfinite(jax.hessian(loss)(1.0))


def test_e2e_parameter_reference_sharing():
    """A primitive holds the parameter object itself, not a copy."""
    radius = Scalar(2.0, free=True, name="radius")
    sphere = Sphere(radius=radius)
    assert sphere.params["radius"] is radius

    original_value = radius.value
    radius.value = jnp.array(3.0)
    assert sphere.params["radius"].value == 3.0
    radius.value = original_value
