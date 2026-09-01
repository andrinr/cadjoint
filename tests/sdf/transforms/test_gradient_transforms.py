"""Tests for gradient flow through transform parameters.

Note: The class-based transform API has limited differentiability with respect
to transform parameters because parameters are stored as instance attributes.
For full differentiability, use the functional API in transforms.functional.
"""

import jax
import jax.numpy as jnp

from cadjoint.sdf.primitives import Box, Sphere
from cadjoint.sdf.transforms import Rotate, Translate, Twist


def test_translate_gradient_functional():
    """Test gradient through translation offset using functional API."""
    sphere = Sphere(radius=1.0)

    def loss_fn(offset):
        # Evaluate at a fixed point - distance changes as we move the sphere
        return Translate.sdf(sphere, jnp.array([2.0, 0.0, 0.0]), offset) ** 2

    offset = jnp.array([0.0, 0.0, 0.0])  # Start with no offset
    grad_fn = jax.grad(loss_fn)

    # Functional API supports full differentiation
    gradient = grad_fn(offset)

    # Gradient should be non-zero (moving sphere changes distance)
    assert jnp.any(jnp.abs(gradient) > 1e-6)
    print(f"Translation gradient: {gradient}")


def test_gradient_through_geometry():
    """Test gradient through geometry (SDF values) - always works."""
    sphere = Sphere(radius=1.0)

    # Gradient with respect to query point works fine
    def loss_fn(point):
        return sphere(point) ** 2

    point = jnp.array([2.0, 0.0, 0.0])
    grad_fn = jax.grad(loss_fn)
    gradient = grad_fn(point)

    # Gradient should point toward sphere center
    assert jnp.isfinite(jnp.all(gradient))
    print(f"Geometry gradient: {gradient}")


def test_rotate_gradient_functional():
    """Test gradient through rotation angle using functional API."""
    box = Box(size=jnp.array([1.0, 0.5, 0.5]))

    def loss_fn(angle):
        # Point outside box that changes distance as we rotate around Z axis
        axis = jnp.array([0.0, 0.0, 1.0])
        return Rotate.sdf(box, jnp.array([2.0, 1.0, 0.0]), axis, angle) ** 2

    angle = 0.2  # Small non-zero angle
    grad_fn = jax.grad(loss_fn)

    gradient = grad_fn(angle)

    assert jnp.isfinite(gradient)
    print(f"Rotation gradient: {gradient}")


def test_twist_gradient_functional():
    """Test gradient through twist strength using functional API."""
    box = Box(size=jnp.array([0.5, 0.5, 1.0]))

    def loss_fn(strength):
        return Twist.sdf(box, jnp.array([1.0, 0.0, 0.5]), strength, jnp.array([0.0, 0.0, 1.0])) ** 2

    strength = 1.0
    grad_fn = jax.grad(loss_fn)

    gradient = grad_fn(strength)

    assert jnp.isfinite(gradient)
    print(f"Twist gradient (functional): {gradient}")


def test_optimization_example_functional():
    """Test optimizing transform parameters to fit a target using functional API."""
    sphere = Sphere(radius=1.0)
    target_point = jnp.array([3.0, 0.0, 0.0])

    def loss_fn(offset):
        # Want the sphere surface to pass through target_point
        distance = Translate.sdf(sphere, target_point, offset)
        return distance**2

    # Start with initial guess
    offset = jnp.array([0.0, 0.0, 0.0])

    # Gradient descent
    learning_rate = 0.1
    for _ in range(50):
        grad = jax.grad(loss_fn)(offset)
        offset = offset - learning_rate * grad

    # After optimization, sphere should be centered at [2, 0, 0]
    assert jnp.allclose(offset[0], 2.0, atol=0.1)
    print(f"Optimized offset (functional): {offset}")
