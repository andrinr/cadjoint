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

from jaxcad.meshing._lattice import _flatten, _strides, _unflatten
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

# The eight corners of a cell in bit order: corner id ``4 dx + 2 dy + dz``.
_CELL_CORNER_OFFSETS = np.array(
    [[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
    dtype=np.int64,
)

# The twelve cell edges as corner-id pairs: corners differing in one bit.
_CORNER_EDGE_PAIRS = tuple(
    (corner, corner | bit) for corner in range(8) for bit in (4, 2, 1) if not corner & bit
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


def _gather_incident(incidence: CellIncidence, normals):
    """Gather per-edge unit normals into each cell's padded slot table.

    Returns ``(gathered, mask)``: the normals in every cell's twelve edge
    slots, and a mask keeping only real slots (the table pads with ``-1``)
    that hold a non-degenerate normal.  A degenerate sample (zero unit
    normal from a dead field subgradient) must not drag cell statistics
    toward zero — that inflates the centered spread and misclassifies
    smooth cells as creases.

    Only operations shared by JAX and NumPy arrays are used, so JAX inputs
    stay traceable and NumPy inputs stay concrete.
    """
    gathered = normals[np.maximum(incidence.edge_ids, 0)]
    mask = ((gathered**2).sum(axis=-1) > 0.25) & (incidence.edge_ids >= 0)
    return gathered, mask


def _masked_mean(gathered, mask):
    """Mean over each cell's masked slots; empty cells divide by one.

    Returns ``(mean, counts)``; the clamped counts are reused by callers
    as normalizers.  Same JAX/NumPy duality as :func:`_gather_incident`.
    """
    counts = mask.sum(axis=1).clip(1)
    return (gathered * mask[..., None]).sum(axis=1) / counts[:, None], counts


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

    strides = _strides(grid.cells)
    flat = _flatten(cells, strides)
    unique_flat, inverse = np.unique(flat, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    counts = np.bincount(sorted_inverse, minlength=unique_flat.shape[0])
    row_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    positions = np.arange(sorted_inverse.shape[0]) - row_starts[sorted_inverse]

    table = np.full((unique_flat.shape[0], 12), -1, dtype=np.int32)
    table[sorted_inverse, positions] = edge_ids[order]
    unique_cells = _unflatten(unique_flat, strides).astype(np.int32)
    return CellIncidence(cells=unique_cells, edge_ids=table, counts=counts.astype(np.int32))


def manifold_cell_incidence(
    edges: CrossingEdges,
    grid: GridSpec,
    inside: np.ndarray,
) -> CellIncidence:
    """Group crossing edges by cell *and* by inside-corner component.

    Uniform dual contouring places one vertex per active cell, which cannot
    represent two surface sheets crossing the same cell: their quads share
    the vertex and mesh edges end up bordered by four triangles.  Following
    Manifold Dual Contouring, this variant splits every active cell into the
    connected components of its *inside* corners under the cell's edge
    adjacency (corners connected along the twelve cell edges) and emits one
    incidence row per component.  Each crossing edge joins the component of
    its inside endpoint, so a cell crossed by two sheets contributes two
    rows — two QEF vertices — and the quads of the two sheets stay disjoint.
    Cells crossed by a single sheet produce exactly the rows (and slot
    order) of :func:`cell_edge_incidence`.

    Args:
        edges: Crossing edges from
            :func:`jaxcad.meshing.edge_detection.find_crossing_edges`.
        grid: The sampling grid the edges were detected on.
        inside: Boolean lattice shaped like :attr:`GridSpec.lattice_shape`;
            ``True`` where ``value - level < 0``, matching the
            ``start_inside`` convention of the edge set.  Only the corners
            of active cells are consulted.

    Returns:
        Incidence rows keyed by (cell, component); ``cells`` rows may repeat
        the same lattice cell.
    """
    inside_lattice = np.asarray(inside, dtype=bool)
    if inside_lattice.shape != grid.lattice_shape:
        raise ValueError(
            f"inside must be shaped {grid.lattice_shape}; received {inside_lattice.shape}."
        )
    if edges.count == 0:
        return CellIncidence(
            cells=np.empty((0, 3), dtype=np.int32),
            edge_ids=np.empty((0, 12), dtype=np.int32),
            counts=np.empty((0,), dtype=np.int32),
        )

    # Candidate (cell, edge, inside-corner) triples, in the same order as
    # cell_edge_incidence so single-component cells reproduce its rows.
    axis_units = np.eye(3, dtype=np.int32)
    candidate_cells = []
    candidate_edges = []
    candidate_corners = []
    for axis in range(3):
        selected = np.flatnonzero(edges.axis == axis)
        if selected.size == 0:
            continue
        offsets = np.asarray(_CELL_EDGE_OFFSETS[axis], dtype=np.int32)
        starts = edges.index[selected].astype(np.int32)
        cells = starts[:, None, :] - offsets[None, :, :]
        inward = np.where(edges.start_inside[selected, None], 0, axis_units[axis][None, :])
        corner = starts[:, None, :] + inward[:, None, :] - cells  # components in {0, 1}
        candidate_cells.append(cells.reshape((-1, 3)))
        candidate_edges.append(np.repeat(selected.astype(np.int32), offsets.shape[0]))
        candidate_corners.append(
            (corner[..., 0] * 4 + corner[..., 1] * 2 + corner[..., 2]).reshape((-1,))
        )

    cells = np.concatenate(candidate_cells)
    edge_ids = np.concatenate(candidate_edges)
    corners = np.concatenate(candidate_corners)
    in_bounds = np.all((cells >= 0) & (cells < np.asarray(grid.cells, dtype=np.int32)), axis=1)
    cells = cells[in_bounds]
    edge_ids = edge_ids[in_bounds]
    corners = corners[in_bounds]

    strides = _strides(grid.cells)
    flat = _flatten(cells, strides)
    unique_flat, inverse = np.unique(flat, return_inverse=True)
    unique_cells = _unflatten(unique_flat, strides)

    # Connected components of each active cell's inside corners: label every
    # inside corner with its own id and propagate the minimum across the
    # twelve corner-pair edges (both endpoints inside) to a fixed point.
    corner_lattice = unique_cells[:, None, :] + _CELL_CORNER_OFFSETS[None, :, :]
    corner_inside = inside_lattice[
        corner_lattice[..., 0], corner_lattice[..., 1], corner_lattice[..., 2]
    ]
    labels = np.where(corner_inside, np.arange(8, dtype=np.int8)[None, :], np.int8(8))
    while True:
        previous = labels.copy()
        for a, b in _CORNER_EDGE_PAIRS:
            both = corner_inside[:, a] & corner_inside[:, b]
            merged = np.minimum(labels[:, a], labels[:, b])
            labels[:, a] = np.where(both, merged, labels[:, a])
            labels[:, b] = np.where(both, merged, labels[:, b])
        if np.array_equal(previous, labels):
            break

    components = labels[inverse, corners]
    if np.any(components == 8):
        raise ValueError(
            "inside is inconsistent with the edge set: a crossing edge's inside "
            "endpoint is not marked inside."
        )

    # Group candidates by (cell, component); rows sort by cell then component.
    group_keys = inverse.astype(np.int64) * 8 + components
    unique_groups, group_inverse = np.unique(group_keys, return_inverse=True)
    order = np.argsort(group_inverse, kind="stable")
    sorted_inverse = group_inverse[order]
    counts = np.bincount(sorted_inverse, minlength=unique_groups.shape[0])
    row_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    positions = np.arange(sorted_inverse.shape[0]) - row_starts[sorted_inverse]

    table = np.full((unique_groups.shape[0], 12), -1, dtype=np.int32)
    table[sorted_inverse, positions] = edge_ids[order]
    row_cells = unique_cells[unique_groups // 8].astype(np.int32)
    return CellIncidence(cells=row_cells, edge_ids=table, counts=counts.astype(np.int32))


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

    gathered, mask = _gather_incident(incidence, hermite.unit_normals())
    means, counts = _masked_mean(gathered, mask)
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


# Half of the 26-neighborhood: one representative per unordered offset pair.
_HALF_NEIGHBORHOOD = np.array(
    [
        offset
        for offset in ((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))
        if offset > (0, 0, 0)
    ],
    dtype=np.int64,
)


def feature_cell_links(
    feature_mask: np.ndarray,
    incidence: CellIncidence,
    grid: GridSpec,
    *,
    junction_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Link feature cells that are lattice neighbors (diagonals included).

    A feature curve — a crease, a corner fan, a CSG seam — crosses cells
    diagonally as often as it crosses them face-to-face, so consecutive
    feature cells along the curve are generally *not* connected by a mesh
    edge; drawing the sharp mesh edges instead produces a staircase around
    the true curve.  Because feature-aware vertex placement puts every
    feature cell's vertex exactly on the curve, connecting neighboring
    feature vertices directly yields chords of the true curve.

    Args:
        feature_mask: Boolean mask over incidence rows selecting feature
            cells (creases, corners, seams).
        incidence: Cell-to-edge mapping whose rows the mask indexes.
        grid: The sampling grid (for lattice strides).
        junction_mask: Optional boolean mask over incidence rows marking
            junction cells (corners).  A link between two cells that share a
            junction neighbor is a shortcut across the junction — the true
            feature path runs through it — and is dropped.

    Returns:
        Unordered incidence-row pairs shaped ``(link_count, 2)``; each pair
        of cells is within Chebyshev distance 1 of the other.
    """
    mask = np.asarray(feature_mask, dtype=bool)
    if mask.shape != (incidence.count,):
        raise ValueError(
            f"feature_mask must have shape ({incidence.count},); received {mask.shape}."
        )
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    cells = incidence.cells[rows].astype(np.int64)
    padded = (grid.cells[0] + 2, grid.cells[1] + 2, grid.cells[2] + 2)
    strides = _strides(padded)
    keys = _flatten(cells + 1, strides)  # shift by one so negative neighbors stay unique
    order = np.argsort(keys)
    sorted_keys = keys[order]
    links = []
    for offset in _HALF_NEIGHBORHOOD:
        neighbor_keys = _flatten(cells + 1 + offset, strides)
        found = np.searchsorted(sorted_keys, neighbor_keys)
        found = np.clip(found, 0, sorted_keys.size - 1)
        hit = sorted_keys[found] == neighbor_keys
        links.append(np.stack([rows[hit], rows[order[found[hit]]]], axis=1))
    if not links:
        return np.empty((0, 2), dtype=np.int32)
    pairs = np.concatenate(links).astype(np.int32).reshape((-1, 2))
    if junction_mask is None or pairs.shape[0] == 0:
        return pairs

    junctions = np.asarray(junction_mask, dtype=bool)
    adjacency: dict[int, set[int]] = {}
    for a, b in pairs:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    keep = np.ones(pairs.shape[0], dtype=bool)
    for row, (a, b) in enumerate(pairs):
        if junctions[a] or junctions[b]:
            continue
        common = adjacency[int(a)] & adjacency[int(b)]
        if any(junctions[cell] for cell in common):
            keep[row] = False
    return pairs[keep]


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
    first = gathered[:, 0]
    return np.any(valid & (gathered != first[:, None]), axis=1)
