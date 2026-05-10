"""Environment-map sampling for background and reflections."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _sample_env_map(env_map: Array, direction: Array) -> Array:
    """Sample an equirectangular HDR environment map for a ray direction.

    Args:
        env_map: Float32 image of shape ``(H, W, 3)`` in equirectangular
            (latitude-longitude) projection with values in ``[0, ∞)``.  Any
            standard HDR image (e.g. loaded via ``imageio.imread``) works here;
            pass it through ``jnp.asarray`` first.
        direction: Unit ray direction, shape ``(3,)``.

    Returns:
        RGB sample from the map, shape ``(3,)``.
    """
    # Spherical → UV
    theta = jnp.arctan2(direction[0], direction[2])  # longitude [-π, π]
    phi = jnp.arcsin(jnp.clip(direction[1], -1.0, 1.0))  # latitude  [-π/2, π/2]

    u = (theta / (2.0 * jnp.pi) + 0.5) % 1.0  # [0, 1]
    v = 0.5 - phi / jnp.pi  # [0, 1]  (top = sky)

    H, W = env_map.shape[0], env_map.shape[1]
    ix = jnp.clip(jnp.floor(u * W).astype(jnp.int32), 0, W - 1)
    iy = jnp.clip(jnp.floor(v * H).astype(jnp.int32), 0, H - 1)
    return env_map[iy, ix]


def make_gradient_sky(
    sky_color: Array | list = (0.20, 0.48, 0.90),
    horizon_color: Array | list = (0.85, 0.72, 0.52),
    ground_color: Array | list = (0.18, 0.16, 0.14),
    resolution: tuple[int, int] = (256, 512),
) -> Array:
    """Create a simple procedural gradient sky as a (H, W, 3) env-map array.

    The latitude axis is split into ground (y < 0) and sky (y ≥ 0), with a
    smooth cosine blend in between.  Suitable as a drop-in replacement for a
    real HDR file.

    Args:
        sky_color: RGB color at the zenith (top of the sky).
        horizon_color: RGB color at the horizon.
        ground_color: RGB color below the horizon.
        resolution: (H, W) of the generated map.

    Returns:
        Float32 JAX array of shape ``(H, W, 3)`` with values in ``[0, 1]``.
    """
    H, W = resolution
    sky_color = jnp.asarray(sky_color, dtype=jnp.float32)
    horizon_color = jnp.asarray(horizon_color, dtype=jnp.float32)
    ground_color = jnp.asarray(ground_color, dtype=jnp.float32)

    # v ∈ [0, 1]: 0 = top (zenith), 0.5 = horizon, 1 = bottom (nadir)
    v = (jnp.arange(H, dtype=jnp.float32) + 0.5) / H  # (H,)

    # Sky half: v ∈ [0, 0.5] → t ∈ [0, 1] (zenith → horizon)
    t_sky = jnp.clip(v * 2.0, 0.0, 1.0)
    t_ground = jnp.clip((v - 0.5) * 2.0, 0.0, 1.0)

    sky_blend = sky_color * (1.0 - t_sky[:, None]) + horizon_color * t_sky[:, None]
    ground_blend = horizon_color * (1.0 - t_ground[:, None]) + ground_color * t_ground[:, None]

    # Combine: above horizon = sky_blend, below = ground_blend
    row_color = jnp.where(v[:, None] < 0.5, sky_blend, ground_blend)  # (H, 3)

    # Tile across width (gradient is latitude-only; uniform in longitude)
    return jnp.broadcast_to(row_color[:, None, :], (H, W, 3))
