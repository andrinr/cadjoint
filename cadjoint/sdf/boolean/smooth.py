"""Smooth blending functions for boolean operations."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def smooth_min(a: Array, b: Array, k: float = 0.1) -> Array:
    """Smooth minimum function for blending SDFs (Inigo Quilez's version).

    Args:
        a: First SDF value(s)
        b: Second SDF value(s)
        k: Smoothness parameter (larger = smoother blend)

    Returns:
        Smoothly blended minimum value
    """
    k_scaled = jnp.maximum(k * 4.0, 1e-10)
    h = jnp.maximum(k_scaled - jnp.abs(a - b), 0.0)
    return jnp.minimum(a, b) - h * h * 0.25 / k_scaled


def smooth_max(a: Array, b: Array, k: float = 0.1) -> Array:
    """Smooth maximum function for blending SDFs.

    Args:
        a: First SDF value(s)
        b: Second SDF value(s)
        k: Smoothness parameter (larger = smoother blend)

    Returns:
        Smoothly blended maximum value
    """
    return -smooth_min(-a, -b, k)
