"""Camera model and ray generation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _normalize(v: Array) -> Array:
    n = jnp.sqrt(jnp.sum(v**2) + 1e-12)
    return v / n


def _camera_rays(
    camera_pos: Array,
    look_at: Array,
    resolution: tuple[int, int],
    fov: float,
) -> Array:
    """Build a flat array of unit ray directions for a perspective camera.

    Constructs an orthonormal camera frame (right, down, forward) aligned to
    world-up = [0, 1, 0].  Image rows run top-to-bottom in screen space.

    Args:
        camera_pos: World-space camera position, shape (3,).
        look_at: World-space point the camera looks toward, shape (3,).
        resolution: (height, width) in pixels.
        fov: Half-width field-of-view in world units at unit depth.

    Returns:
        Unit ray directions, shape (H*W, 3).
    """
    h, w = resolution
    forward = _normalize(look_at - camera_pos)
    world_up = jnp.array([0.0, 1.0, 0.0])
    right = _normalize(jnp.cross(forward, world_up))
    down = _normalize(jnp.cross(right, forward))

    fx = fov
    fy = fx / w * h
    # linspace works with abstract fov inside jax.jit (mgrid requires concrete bounds)
    xs = jnp.linspace(-fx, fx, w)
    ys = jnp.linspace(fy, -fy, h)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    y = yy.reshape(-1)
    x = xx.reshape(-1)
    dirs = jnp.stack([x, y, jnp.ones_like(x)], axis=1)
    R = jnp.stack([right, down, forward])
    dirs = dirs @ R
    norms = jnp.sqrt(jnp.sum(dirs**2, axis=1, keepdims=True) + 1e-12)
    return dirs / norms
