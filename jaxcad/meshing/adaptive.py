"""Adaptive grid search: octree pruning for surface-cell discovery.

Dense detection evaluates the field at every lattice vertex, although almost
all cells are far from the surface.  This module descends an octree over the
cell lattice instead, discarding whole blocks the field provably cannot
cross: for a block with center ``c`` and half-diagonal ``ρ``, any zero of an
``L``-Lipschitz field inside the block requires ``|f(c) - level| ≤ ρ·L``.
Work becomes proportional to the surface area instead of the volume.

The bound is safe only when ``L`` really bounds the field's gradient.  Exact
SDF primitives and rigid transforms are 1-Lipschitz, and ``min``/``max`` CSG
preserves that; scaling and smooth blends need a larger bound.  When in
doubt, pass a generous ``lipschitz`` — pruning weakens gracefully, missing
cells does not.

The octree adapts the *search*, not the mesh: surviving leaves are ordinary
unit cells of the uniform grid, so the returned edge set — and therefore the
extracted mesh — is identical to dense detection at the same resolution, and
every downstream stage works unchanged.  Multi-size cells in the mesh itself
(coarse leaves in flat regions) additionally need 2:1 balancing and
transition stitching; that is the later, Manifold-Dual-Contouring stage.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxcad.meshing._lattice import _flatten, _strides, _unflatten
from jaxcad.meshing.edge_detection import CrossingEdges, GridSpec
from jaxcad.meshing.features import _CELL_EDGE_OFFSETS

_CORNER_OFFSETS = np.array(
    [[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
    dtype=np.int64,
)
# Absolute slack on the pruning threshold absorbs float32 evaluation noise
# so a block whose true center value sits exactly on the bound is never
# wrongly discarded.
_PRUNE_SLACK = 1e-5
# Every batch is padded to this exact size so XLA compiles one program for
# the whole descent; per-level shapes would otherwise recompile at every
# octree level and dominate the wall clock.
_CHUNK = 8192


def _chunked_evaluator(
    sdf: Callable[[Array], Array],
) -> Callable[[np.ndarray], np.ndarray]:
    batched = jax.jit(jax.vmap(sdf))

    def evaluate(points: np.ndarray) -> np.ndarray:
        count = points.shape[0]
        values = np.empty((count,), dtype=np.float64)
        for start in range(0, count, _CHUNK):
            chunk = points[start : start + _CHUNK]
            if chunk.shape[0] < _CHUNK:
                padded = np.repeat(chunk[:1], _CHUNK, axis=0)
                padded[: chunk.shape[0]] = chunk
                chunk_values = batched(jnp.asarray(padded, dtype=jnp.float32))
                values[start:] = np.asarray(chunk_values, dtype=np.float64)[: chunk.shape[0]]
            else:
                values[start : start + _CHUNK] = np.asarray(
                    batched(jnp.asarray(chunk, dtype=jnp.float32)), dtype=np.float64
                )
        return values

    return evaluate


def surface_cells(
    sdf: Callable[[Array], Array],
    grid: GridSpec,
    *,
    level: float = 0.0,
    lipschitz: float = 1.0,
    stats: dict | None = None,
) -> np.ndarray:
    """Find every unit cell that may contain the level set, by octree descent.

    Returns lattice cell indices shaped ``(cell_count, 3)``, a conservative
    superset of the cells the surface actually crosses.
    """
    return _surface_cells(
        _chunked_evaluator(sdf), grid, level=level, lipschitz=lipschitz, stats=stats
    )


def _surface_cells(
    evaluator: Callable[[np.ndarray], np.ndarray],
    grid: GridSpec,
    *,
    level: float,
    lipschitz: float,
    stats: dict | None,
) -> np.ndarray:
    if not lipschitz > 0:
        raise ValueError("lipschitz must be positive.")
    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    cells = np.asarray(grid.cells, dtype=np.int64)

    evaluations = 0
    block = 1 << int(np.ceil(np.log2(cells.max())))
    blocks = np.zeros((1, 3), dtype=np.int64)
    while True:
        lo = blocks * block
        hi = np.minimum(lo + block, cells)
        centers = origin + (lo + hi) * 0.5 * spacing
        half_diagonal = np.linalg.norm((hi - lo) * 0.5 * spacing, axis=1)
        values = evaluator(centers)
        evaluations += centers.shape[0]
        keep = np.abs(values - level) <= half_diagonal * lipschitz + _PRUNE_SLACK
        blocks = blocks[keep]
        if block == 1 or blocks.shape[0] == 0:
            break
        block //= 2
        children = (blocks[:, None, :] * 2 + _CORNER_OFFSETS[None, :, :]).reshape((-1, 3))
        in_bounds = np.all(children * block < cells, axis=1)
        blocks = children[in_bounds]

    if stats is not None:
        stats["evaluations"] = evaluations
    return blocks.astype(np.int32)


def sparse_crossing_edges(
    sdf: Callable[[Array], Array],
    grid: GridSpec,
    *,
    level: float = 0.0,
    lipschitz: float = 1.0,
    stats: dict | None = None,
    return_inside: bool = False,
) -> CrossingEdges | tuple[CrossingEdges, np.ndarray]:
    """Detect crossing edges without sampling the full lattice.

    Octree pruning finds the candidate cells, then only their corner
    vertices are evaluated.  The result is identical to
    :func:`jaxcad.meshing.edge_detection.find_crossing_edges` over a dense
    sample — same edges, same order, same frozen orientations — whenever
    ``lipschitz`` genuinely bounds the field.

    Args:
        sdf: JAX-traceable scalar field ``(point[3]) -> scalar``.
        grid: Sampling grid.
        level: Isovalue to extract.
        lipschitz: Gradient bound the pruning may assume; see the module
            docstring.
        stats: Optional dict that receives ``evaluations`` (total field
            evaluations, pruning plus corners) and ``candidate_cells``.
        return_inside: Also return a boolean inside lattice shaped like
            :attr:`GridSpec.lattice_shape`, filled from the corner values
            already evaluated here (``value - level < 0``; unevaluated
            vertices read ``False``).  Every active cell is a pruning
            candidate, so all corners consulted by
            :func:`jaxcad.meshing.features.manifold_cell_incidence` carry
            evaluated values, matching a dense sample exactly.

    Returns:
        The crossing edge set, or ``(edges, inside)`` when
        ``return_inside`` is true.
    """
    # One evaluator for pruning and corners: tracing/compiling the field is
    # the dominant cost for real scenes, and must be paid exactly once.
    evaluator = _chunked_evaluator(sdf)
    candidates = _surface_cells(evaluator, grid, level=level, lipschitz=lipschitz, stats=stats)
    if stats is not None:
        stats["candidate_cells"] = int(candidates.shape[0])
    if candidates.shape[0] == 0:
        empty_index = np.empty((0, 3), dtype=np.int32)
        empty = CrossingEdges(
            axis=np.empty((0,), dtype=np.int8),
            index=empty_index,
            start_inside=np.empty((0,), dtype=bool),
        )
        if return_inside:
            return empty, np.zeros(grid.lattice_shape, dtype=bool)
        return empty

    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    strides = _strides(grid.lattice_shape)

    corners = (candidates[:, None, :].astype(np.int64) + _CORNER_OFFSETS[None, :, :]).reshape(
        (-1, 3)
    )
    corner_keys = np.unique(_flatten(corners, strides))
    corner_index = _unflatten(corner_keys, strides)
    # The same float64 expression as lattice_points(), so evaluation happens
    # at bit-identical coordinates to a dense sample.
    positions = origin + corner_index.astype(np.float64) * spacing
    corner_values = evaluator(positions)
    if stats is not None:
        stats["evaluations"] += positions.shape[0]
    if not np.isfinite(corner_values).all():
        raise ValueError("The implicit field returned a non-finite value.")

    def value_at(keys: np.ndarray) -> np.ndarray:
        return corner_values[np.searchsorted(corner_keys, keys)]

    axis_labels = []
    axis_indices = []
    axis_orientation = []
    for axis in range(3):
        offsets = np.asarray(_CELL_EDGE_OFFSETS[axis], dtype=np.int64)
        starts = (candidates[:, None, :].astype(np.int64) + offsets[None, :, :]).reshape((-1, 3))
        start_keys = np.unique(_flatten(starts, strides))
        step = strides[axis]
        start_values = value_at(start_keys)
        end_values = value_at(start_keys + step)
        start_inside = (start_values - level) < 0
        end_inside = (end_values - level) < 0
        crossing = start_inside != end_inside
        keys = start_keys[crossing]
        index = _unflatten(keys, strides)
        axis_labels.append(np.full((keys.shape[0],), axis, dtype=np.int8))
        axis_indices.append(index.astype(np.int32))
        axis_orientation.append(start_inside[crossing])

    edges = CrossingEdges(
        axis=np.concatenate(axis_labels),
        index=np.concatenate(axis_indices).reshape((-1, 3)),
        start_inside=np.concatenate(axis_orientation),
    )
    if return_inside:
        inside = np.zeros(grid.lattice_shape, dtype=bool)
        inside.reshape((-1,))[corner_keys] = (corner_values - level) < 0
        return edges, inside
    return edges
