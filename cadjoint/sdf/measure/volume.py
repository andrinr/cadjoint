"""Differentiable volume estimation for SDFs."""

import jax
import jax.numpy as jnp
from jax import Array

from cadjoint.sdf.base import SDF


def volume(
    sdf: SDF,
    bounds: tuple[float, float, float] = (-3, -3, -3),
    size: tuple[float, float, float] = (6, 6, 6),
    resolution: int = 50,
    epsilon: float = 0.01,
) -> Array:
    """Estimate the volume of an SDF as a differentiable JAX scalar.

    Uses a smooth sigmoid indicator σ(−d/ε) as a soft approximation of the
    interior indicator (1 where d < 0, 0 where d > 0).  The result is
    differentiable with respect to any free parameters of the SDF.

    Args:
        sdf:        The SDF to measure.
        bounds:     Lower corner (x, y, z) of the sampling box.
        size:       Extent (dx, dy, dz) of the sampling box.
        resolution: Number of samples per axis (resolution³ total).
        epsilon:    Smoothing width.  Smaller → sharper / more accurate;
                    larger → smoother gradients.

    Returns:
        Differentiable scalar volume estimate.
    """
    x = jnp.linspace(bounds[0], bounds[0] + size[0], resolution)
    y = jnp.linspace(bounds[1], bounds[1] + size[1], resolution)
    z = jnp.linspace(bounds[2], bounds[2] + size[2], resolution)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    distances = jax.vmap(sdf)(points)
    indicators = jax.nn.sigmoid(-distances / epsilon)

    voxel_vol = (size[0] / resolution) * (size[1] / resolution) * (size[2] / resolution)
    return jnp.sum(indicators) * voxel_vol
