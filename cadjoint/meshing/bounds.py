"""Fit an extraction lattice to a part, at whatever scale it was modelled.

:data:`~cadjoint.meshing.DEFAULT_BOUNDS` is a ±3 box: right for a scene
modelled in the playground's units, and wrong for a 100 mm part modelled in
millimetres, which it clips to a 6 mm chunk without a word — the surface
crossing its boundary is a warning the export suppresses, and a box wholly
inside the part has no surface to cross at all (issue #49).
:func:`fit_bounds` finds the part instead:

- **Growing scales.**  Cubes of half-width 1, 4, 16 … 4096 are scanned at
  :data:`SCAN_CELLS` cells per axis, and the smallest cube whose counted
  samples stay off its boundary holds the part.
- **The near-surface rule.**  A sample counts when the field is below half
  a cell diagonal.  The field never overestimates the distance to the
  surface, so every surface point has a counted sample within that half
  diagonal: a plate thinner than the scan spacing is found even when no
  sample lands inside it.
- **One refinement.**  The cube found can be four times the part with a
  pad of a sixteenth of the cube; the same rule on :data:`SCAN_CELLS`
  cells across the coarse box bounds the part to a few percent of its size.

:func:`boundary_inside` is the clipping check that does not depend on the
surface crossing the boundary: it evaluates the field on the lattice's six
faces and reports whether any of it lies inside the part.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cadjoint.meshing.edge_detection import GridSpec

__all__ = ["SCAN_CELLS", "SCAN_HALF_WIDTHS", "boundary_inside", "fit_bounds"]

#: Half-widths of the cubes :func:`fit_bounds` scans, smallest first.
SCAN_HALF_WIDTHS = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0)
#: Cells per axis of each scan.
SCAN_CELLS = 48
#: Padding per side, in scan cells, around the counted samples.
_PAD_CELLS = 1.5


def _near(field: Callable, points: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Samples inside the part or within half a cell diagonal of its surface."""
    values = np.asarray(field(points.astype(np.float32)), dtype=np.float64).reshape(-1)
    return values < 0.5 * float(np.linalg.norm(spacing))


def _lattice(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    axes = [np.linspace(a, b, SCAN_CELLS + 1) for a, b in zip(low, high)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def fit_bounds(
    sdf: Callable, padding: float = 0.0
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(origin, size)`` of a box around the part, found at its own scale.

    Args:
        sdf: The field, mapping ``(n, 3)`` points to ``(n,)`` values.
        padding: Extra margin per side, in model units, on top of the one
            and a half fine-scan cells every fit carries.

    Returns:
        The lower corner and the extent, ready for
        :meth:`~cadjoint.meshing.GridSpec.from_bounds`.

    Raises:
        ValueError: When no cube up to half-width ``SCAN_HALF_WIDTHS[-1]``
            holds a closed part: the object is empty, or unbounded (an
            infinite plane, a missing intersection) and needs explicit
            bounds.
    """
    import jax

    # Every scan is the same shape, so the field compiles once.
    field = jax.jit(sdf)
    unit = _lattice(-np.ones(3), np.ones(3))
    for half in SCAN_HALF_WIDTHS:
        spacing = np.full(3, 2.0 * half / SCAN_CELLS)
        points = unit * half
        near = _near(field, points, spacing)
        if not near.any():
            continue
        low, high = points[near].min(axis=0), points[near].max(axis=0)
        edge = half - 0.5 * spacing[0]
        if (low <= -edge).any() or (high >= edge).any():
            continue  # the part reaches the scan cube's boundary: look wider
        pad = _PAD_CELLS * spacing
        low, high = _refine(field, low - pad, high + pad)
        low, high = low - padding, high + padding
        return (
            tuple(float(v) for v in low),  # type: ignore[return-value]
            tuple(float(v) for v in high - low),
        )
    raise ValueError(
        "No closed part found inside a cube of half-width "
        f"{SCAN_HALF_WIDTHS[-1]:g}: the object is empty or unbounded (an infinite "
        "plane, a missing intersection). Give it explicit bounds."
    )


def _refine(field: Callable, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tighten a coarse box with one scan of the same size inside it."""
    points = _lattice(low, high)
    spacing = (high - low) / SCAN_CELLS
    near = _near(field, points, spacing)
    if not near.any():
        return low, high
    pad = _PAD_CELLS * spacing
    return points[near].min(axis=0) - pad, points[near].max(axis=0) + pad


def boundary_inside(sdf: Callable, grid: GridSpec) -> bool:
    """Whether any lattice vertex on the grid's six faces lies inside the part.

    True means a mesh extracted on *grid* is cut open there — or, if the
    whole grid is inside, empty.  Unlike the extractor's boundary warning
    this does not need the surface to cross the boundary.
    """
    axes = [
        origin + spacing * np.arange(count + 1)
        for origin, spacing, count in zip(grid.origin, grid.spacing, grid.cells)
    ]
    for fixed in range(3):
        free = [axes[i] for i in range(3) if i != fixed]
        a, b = np.meshgrid(*free, indexing="ij")
        for end in (axes[fixed][0], axes[fixed][-1]):
            columns = [a.ravel(), b.ravel()]
            columns.insert(fixed, np.full(a.size, end))
            face = np.stack(columns, axis=-1).astype(np.float32)
            if (np.asarray(sdf(face)) < 0.0).any():
                return True
    return False
