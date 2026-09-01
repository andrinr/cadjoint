"""Flat-key arithmetic for integer lattice coordinates.

Several host-side stages need set- or dictionary-style lookups over 3-D
lattice coordinates — cells, lattice vertices, padded neighborhoods.
Encoding each coordinate triple as one row-major integer key turns those
lookups into sorted-array operations; these helpers keep the stride
arithmetic in a single place so every stage encodes identically.
"""

from __future__ import annotations

import numpy as np


def _strides(shape: tuple[int, int, int]) -> np.ndarray:
    """Row-major strides for flat keys over a lattice of ``shape``."""
    return np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)


def _flatten(coordinates: np.ndarray, strides: np.ndarray) -> np.ndarray:
    """Encode integer coordinate triples as scalar row-major keys."""
    return coordinates.astype(np.int64) @ strides


def _unflatten(keys: np.ndarray, strides: np.ndarray) -> np.ndarray:
    """Decode scalar keys back into coordinate triples, shaped ``(count, 3)``."""
    return np.stack(
        [keys // strides[0], (keys % strides[0]) // strides[1], keys % strides[1]],
        axis=1,
    )
