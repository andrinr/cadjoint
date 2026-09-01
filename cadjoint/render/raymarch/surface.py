"""Surface-position helpers shared by secondary ray paths."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.render.raymarch._constants import _SECONDARY_RAY_OFFSET


def _offset_surface(position: Array, normal: Array, hit_epsilon: float) -> Array:
    """Move a ray origin beyond the configured surface hit band."""
    distance = jnp.maximum(_SECONDARY_RAY_OFFSET, 4.0 * hit_epsilon)
    return position + distance * normal
