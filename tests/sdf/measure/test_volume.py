"""Tests for differentiable volume estimation."""

import jax
import jax.numpy as jnp

from cadjoint.sdf.measure import volume
from cadjoint.sdf.primitives import Box, Sphere

BOUNDS = (-2, -2, -2)
SIZE = (4, 4, 4)
RESOLUTION = 80


def test_sphere_volume_accuracy():
    """Volume of unit sphere should be close to (4/3)π ≈ 4.189."""
    v = volume(Sphere(radius=1.0), bounds=BOUNDS, size=SIZE, resolution=RESOLUTION)
    assert jnp.isclose(v, 4 / 3 * jnp.pi, atol=0.2)


def test_box_volume_accuracy():
    """Volume of 1×1×1 box centred at origin should be close to 1.0.

    Box takes half-extents, so (0.5, 0.5, 0.5) → 1×1×1 cube, volume = 1.
    """
    v = volume(
        Box(size=jnp.array([0.5, 0.5, 0.5])), bounds=BOUNDS, size=SIZE, resolution=RESOLUTION
    )
    assert jnp.isclose(v, 1.0, atol=0.05)


def test_larger_sphere_volume():
    """Volume scales as r³: radius-2 sphere ≈ 8× unit sphere."""
    v1 = volume(Sphere(radius=1.0), bounds=(-3, -3, -3), size=(6, 6, 6), resolution=RESOLUTION)
    v2 = volume(Sphere(radius=2.0), bounds=(-3, -3, -3), size=(6, 6, 6), resolution=RESOLUTION)
    assert jnp.isclose(v2 / v1, 8.0, atol=0.2)


def test_volume_differentiable_wrt_radius():
    """jax.grad should work through volume and give dV/dr ≈ 4πr²."""

    def vol_fn(r):
        return volume(Sphere(radius=r), bounds=BOUNDS, size=SIZE, resolution=RESOLUTION)

    grad = jax.grad(vol_fn)(1.0)
    assert jnp.isclose(grad, 4 * jnp.pi, atol=0.5)


def test_volume_positive():
    """Volume estimate must always be positive."""
    v = volume(Sphere(radius=1.0), bounds=BOUNDS, size=SIZE, resolution=50)
    assert v > 0


def test_volume_increases_with_radius():
    """Larger radius → larger volume."""
    v1 = volume(Sphere(radius=0.5), bounds=BOUNDS, size=SIZE, resolution=50)
    v2 = volume(Sphere(radius=1.5), bounds=BOUNDS, size=SIZE, resolution=50)
    assert v2 > v1


def test_epsilon_convergence():
    """Smaller epsilon should give a more accurate volume estimate."""
    sphere = Sphere(radius=1.0)
    analytical = 4 / 3 * jnp.pi
    err_large = abs(
        volume(sphere, bounds=BOUNDS, size=SIZE, resolution=50, epsilon=0.5) - analytical
    )
    err_small = abs(
        volume(sphere, bounds=BOUNDS, size=SIZE, resolution=50, epsilon=0.01) - analytical
    )
    assert err_small < err_large
