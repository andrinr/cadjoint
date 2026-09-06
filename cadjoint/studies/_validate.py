"""Argument validation shared by every declared study.

One home for the checks a study's constructor repeats: the three-number
vectors that name a region, a point, an axis or a lattice origin.  These
were four byte-identical copies before this module existed — one each in
the selection language, the simulation mesh, the FEM study layer and the
flow study layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["require_triplet"]


def require_triplet(value: Any, label: str) -> tuple[float, float, float]:
    """Three finite floats, or a ``ValueError`` naming the argument.

    Args:
        value: Anything array-like of shape ``(3,)``.
        label: The argument's name, used in the error message.

    Returns:
        The value as a plain tuple of three floats.

    Raises:
        ValueError: If the value is not three finite numbers.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain three finite numbers, got {value!r}.")
    return (float(array[0]), float(array[1]), float(array[2]))
