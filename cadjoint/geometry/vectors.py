"""Vector arithmetic every layer needs, written once.

`cadjoint.geometry` is what the SDF, construction and constraint layers all
already depend on, which makes it the one place a shared helper can live
without inventing an import edge.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["MIN_SQUARED", "unit"]

#: Floor on a squared norm before it is square-rooted.
#:
#: Not a tolerance on the answer but a guard on its *derivative*: the
#: gradient of ``sqrt`` is unbounded at zero, so normalising a vector that
#: can vanish under tracing produces a NaN that propagates through the whole
#: reverse pass. Clamping the radicand keeps the value right everywhere it
#: matters and the gradient finite where it does not.
MIN_SQUARED = 1e-12


def unit(vector: Array) -> Array:
    """Normalize a vector with a guarded norm, safe under tracing.

    Args:
        vector: Any array-like; converted, so a plain list works.

    Returns:
        The vector scaled to unit length, or a vector of magnitude
            ``|v| / sqrt(MIN_SQUARED)`` when it is shorter than the guard —
            which for a genuinely zero vector is zero.
    """
    vector = jnp.asarray(vector)
    return vector / jnp.sqrt(jnp.maximum(jnp.sum(vector * vector), MIN_SQUARED))
