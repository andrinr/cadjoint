"""Sharp-feature detection from Hermite edge data.

Second half of the edge-detection stage: given the crossing edges and Hermite
samples produced by :mod:`jaxcad.meshing.edge_detection`, decide which grid
cells contain smooth surface, a crease, or a corner.

Two complementary detectors are provided:

- :func:`classify_feature_cells` is geometric and works on any black-box
  field.  It measures the angular spread of the Hermite normals incident to a
  cell through the singular values of their mean-centered covariance.  Both
  spread directions near zero mean locally planar surface, one significant
  direction means the normals fan across a crease, two mean they span a
  corner.
- :func:`detect_branch_changes` is structural and exact for hard CSG.
  ``min``/``max`` operations select an active child field at every point, and
  a sharp crease lies exactly where the active branch switches on the
  surface.  Marking each Hermite sample with its active branch and looking
  for cells with mixed branches finds those seams without thresholds.

Cell incidence is discrete and computed host-side; the singular-value
measures are ordinary JAX values and differentiate with respect to design
parameters.  Exactly-degenerate spread directions (singular value zero, the
generic case for flat geometry) carry a zero subgradient instead of the
``inf``/NaN a raw ``sqrt`` would produce.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxcad.meshing.edge_detection import CrossingEdges, GridSpec, HermiteData

FACE = 0
CREASE = 1
CORNER = 2

# Lattice offsets from a cell to the start vertices of its twelve edges,
# grouped by edge axis.  Cell (i, j, k) owns x-edges at (i, j + dj, k + dk)
# and so on; equivalently, an axis-``a`` edge touches the up-to-four cells
# found by subtracting these offsets on the two non-``a`` axes.
_CELL_EDGE_OFFSETS = (
    ((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)),
    ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)),
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
)


class CellIncidence(NamedTuple):
    """Mapping from active cells to the crossing edges on their boundary.

    Attributes:
        cells: Lattice index of each active cell, shaped ``(cell_count, 3)``.
        edge_ids: Indices into the crossing-edge arrays, shaped
            ``(cell_count, 12)`` and padded with ``-1``.
        counts: Number of valid entries per row of ``edge_ids``.
    """

    cells: np.ndarray
    edge_ids: np.ndarray
    counts: np.ndarray

    @property
    def count(self) -> int:
        return int(self.cells.shape[0])


class FeatureCells(NamedTuple):
    """Per-cell feature classification.

    Attributes:
        singular_values: Descending singular values of each cell's
            mean-centered, count-normalized unit normals, shaped
            ``(cell_count, 3)``.  JAX values.
        crease_measure: Leading singular value per cell; for two normal
            clusters at angle ``2α`` around their mean this equals
            ``sin(α)``, so it grows monotonically with crease sharpness.
            JAX values.
        corner_measure: Second singular value per cell; near zero for faces
            and creases, significant when the normals span a corner.  JAX
            values.
        classes: Concrete labels per cell: :data:`FACE`, :data:`CREASE`, or
            :data:`CORNER`.
    """

    singular_values: Array
    crease_measure: Array
    corner_measure: Array
    classes: np.ndarray


def cell_edge_incidence(edges: CrossingEdges, grid: GridSpec) -> CellIncidence:
    """Group crossing edges by the grid cells they border.

    Every axis-aligned edge touches up to four cells; cells outside the grid
    are dropped.  The result covers exactly the active cells: cells with at
    least one crossing edge on their boundary.
    """
    if edges.count == 0:
        empty = np.empty((0, 3), dtype=np.int32)
        return CellIncidence(
            cells=empty,
            edge_ids=np.empty((0, 12), dtype=np.int32),
            counts=np.empty((0,), dtype=np.int32),
        )

    candidate_cells = []
    candidate_edges = []
    for axis in range(3):
        selected = np.flatnonzero(edges.axis == axis)
        if selected.size == 0:
            continue
        offsets = np.asarray(_CELL_EDGE_OFFSETS[axis], dtype=np.int32)
        cells = edges.index[selected][:, None, :] - offsets[None, :, :]
        candidate_cells.append(cells.reshape((-1, 3)))
        candidate_edges.append(np.repeat(selected.astype(np.int32), offsets.shape[0]))

    cells = np.concatenate(candidate_cells)
    edge_ids = np.concatenate(candidate_edges)
    in_bounds = np.all((cells >= 0) & (cells < np.asarray(grid.cells, dtype=np.int32)), axis=1)
    cells = cells[in_bounds]
    edge_ids = edge_ids[in_bounds]

    strides = np.asarray(
        [grid.cells[1] * grid.cells[2], grid.cells[2], 1],
        dtype=np.int64,
    )
    flat = cells.astype(np.int64) @ strides
    unique_flat, inverse = np.unique(flat, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    counts = np.bincount(sorted_inverse, minlength=unique_flat.shape[0])
    row_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    positions = np.arange(sorted_inverse.shape[0]) - row_starts[sorted_inverse]

    table = np.full((unique_flat.shape[0], 12), -1, dtype=np.int32)
    table[sorted_inverse, positions] = edge_ids[order]
    unique_cells = np.stack(
        [
            unique_flat // strides[0],
            (unique_flat % strides[0]) // strides[1],
            unique_flat % strides[1],
        ],
        axis=1,
    ).astype(np.int32)
    return CellIncidence(cells=unique_cells, edge_ids=table, counts=counts.astype(np.int32))


def classify_feature_cells(
    hermite: HermiteData,
    incidence: CellIncidence,
    *,
    crease_threshold: float = 0.25,
    corner_threshold: float = 0.25,
) -> FeatureCells:
    """Classify active cells as face, crease, or corner from normal spread.

    For each cell, the incident Hermite unit normals are centered about their
    mean and the singular values of the count-normalized spread measure how
    many independent directions they fan across.  The leading singular value
    is compared against ``crease_threshold`` and the second against
    ``corner_threshold``.  For two normal clusters at angle ``2α`` around
    their mean the leading singular value equals ``sin(α)``, so it is
    monotonic in crease sharpness all the way to opposed normals, and the
    default of ``0.25`` fires at a normal fan of roughly 29 degrees while a
    smooth surface at cell size ``h`` and curvature radius ``r`` stays near
    ``h / (2 r)``.

    The singular values and measures are differentiable JAX values; exactly
    degenerate directions (flat geometry) get a zero subgradient.  The class
    labels are concrete and frozen per extraction, like the edge set.
    """
    if not 0 < crease_threshold < 1:
        raise ValueError("crease_threshold must be between 0 and 1.")
    if not 0 < corner_threshold < 1:
        raise ValueError("corner_threshold must be between 0 and 1.")

    normals = hermite.unit_normals()
    edge_ids = jnp.asarray(np.maximum(incidence.edge_ids, 0))
    mask = jnp.asarray((incidence.edge_ids >= 0).astype(np.float32))
    gathered = normals[edge_ids]
    counts = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    means = jnp.sum(gathered * mask[..., None], axis=1) / counts[:, None]
    centered = (gathered - means[:, None, :]) * mask[..., None]
    covariance = jnp.einsum("cei,cej->cij", centered, centered) / counts[:, None, None]
    eigenvalues = jnp.linalg.eigvalsh(covariance)[..., ::-1]
    # sqrt has infinite slope at zero, and flat cells produce exactly-zero
    # eigenvalues; the double-where keeps the degenerate branch constant so
    # it cannot NaN the reverse-mode gradients of every other cell.
    positive = eigenvalues > 1e-12
    safe = jnp.where(positive, eigenvalues, 1.0)
    singular_values = jnp.where(positive, jnp.sqrt(safe), 0.0)

    crease_measure = singular_values[..., 0]
    corner_measure = singular_values[..., 1]

    concrete_crease = np.asarray(jax.lax.stop_gradient(crease_measure))
    concrete_corner = np.asarray(jax.lax.stop_gradient(corner_measure))
    classes = np.where(
        concrete_corner > corner_threshold,
        CORNER,
        np.where(concrete_crease > crease_threshold, CREASE, FACE),
    ).astype(np.int8)
    return FeatureCells(
        singular_values=singular_values,
        crease_measure=crease_measure,
        corner_measure=corner_measure,
        classes=classes,
    )


def active_branches(
    child_fields: Sequence[Callable[[Array], Array]],
    points: Array,
    *,
    mode: str = "min",
) -> np.ndarray:
    """Report which child field a hard CSG ``min``/``max`` selects per point.

    Args:
        child_fields: The operand fields of a union (``mode="min"``) or
            intersection (``mode="max"``).
        points: Query positions shaped ``(point_count, 3)``.
        mode: ``"min"`` or ``"max"``.

    Returns:
        Concrete child indices shaped ``(point_count,)``.
    """
    if len(child_fields) < 2:
        raise ValueError("active_branches needs at least two child fields.")
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'.")
    stacked = jnp.stack([jax.vmap(field)(points) for field in child_fields])
    winner = jnp.argmin(stacked, axis=0) if mode == "min" else jnp.argmax(stacked, axis=0)
    return np.asarray(jax.lax.stop_gradient(winner), dtype=np.int32)


def detect_branch_changes(
    branches: np.ndarray,
    incidence: CellIncidence,
) -> np.ndarray:
    """Mark cells whose incident Hermite samples disagree on the active branch.

    A hard CSG crease lies exactly where the surface switches from one
    ``min``/``max`` operand to another, so a cell whose surface samples carry
    different active branches straddles a seam.  This is exact for hard
    Booleans and needs no angular threshold; combine it with
    :func:`classify_feature_cells` for black-box or smoothly blended fields.

    Args:
        branches: Active branch per Hermite sample, from
            :func:`active_branches`, shaped ``(edge_count,)``.
        incidence: Cell-to-edge mapping for the same edge set.

    Returns:
        Boolean mask shaped ``(cell_count,)``; ``True`` where a seam crosses
        the cell.
    """
    branch_array = np.asarray(branches)
    if branch_array.ndim != 1:
        raise ValueError(f"branches must be one-dimensional; received shape {branch_array.shape}.")
    valid = incidence.edge_ids >= 0
    gathered = branch_array[np.maximum(incidence.edge_ids, 0)]
    first = gathered[np.arange(incidence.count), 0]
    return np.any(valid & (gathered != first[:, None]), axis=1)
