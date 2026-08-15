"""Mesh generation from Hermite edge data: differentiable dual contouring.

Third stage of the meshing pipeline.  One vertex is placed in every active
cell by minimizing a regularized quadratic error against the cell's Hermite
tangent planes, and cells are connected by a quad around every crossing edge.

The split established by the earlier stages carries through unchanged:

- **Continuous and differentiable**: vertex positions solve
  ``(Σ nᵢnᵢᵀ + λI)(v - m) = Σ nᵢ nᵢ·(pᵢ - m)`` around the cell's Hermite
  mass point ``m`` — plain batched linear algebra on stage-1 outputs, so
  vertices inherit exact parameter derivatives through the moving Hermite
  inputs.  The Tikhonov term makes the system uniformly well posed: planar
  cells stay near their mass point instead of sliding along the face, and no
  SVD/eigendecomposition (whose gradients blow up at the repeated singular
  values planar cells always have) appears in the gradient path.
- **Discrete and frozen**: which cells exist, how they connect, and the
  triangulation of each quad.  Winding is deterministic: the frozen
  ``start_inside`` orientation of each crossing edge decides the face
  direction, with no dependence on computed normals.

An optimization loop re-extracts per step and differentiates through
:func:`qef_vertices` with frozen edges/incidence::

    edges = find_crossing_edges(sample_grid(compiled(free, fixed), grid))
    incidence = cell_edge_incidence(edges, grid)

    def loss(candidate_free):
        hermite = edge_hermite_data(compiled(candidate_free, fixed), grid, edges)
        vertices, _ = qef_vertices(hermite, incidence, grid)
        return mesh_loss(vertices, faces)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxcad.meshing.edge_detection import (
    CrossingEdges,
    GridSpec,
    HermiteData,
    edge_hermite_data,
    find_crossing_edges,
    sample_grid,
)
from jaxcad.meshing.features import CellIncidence, cell_edge_incidence


class Mesh(NamedTuple):
    """Dual-contour surface mesh.

    Attributes:
        vertices: One vertex per active cell, shaped ``(cell_count, 3)``.
            JAX values; differentiable with respect to design parameters.
        faces: Triangle indices into ``vertices``, shaped ``(tri_count, 3)``,
            counterclockwise seen from outside.  Concrete and frozen.
        quads: Source quads before triangulation, shaped ``(quad_count, 4)``.
        normals: Per-vertex normals (masked mean of incident Hermite unit
            normals), shaped ``(cell_count, 3)``.  JAX values.
        cells: Lattice cell index of each vertex row, shaped
            ``(cell_count, 3)``.  Concrete.
    """

    vertices: Array
    faces: np.ndarray
    quads: np.ndarray
    normals: Array
    cells: np.ndarray


def qef_vertices(
    hermite: HermiteData,
    incidence: CellIncidence,
    grid: GridSpec,
    *,
    regularization: float = 1e-3,
) -> tuple[Array, Array]:
    """Place one vertex per active cell from its Hermite tangent planes.

    Minimizes ``Σ (nᵢ · (v - pᵢ))²`` plus a Tikhonov pull toward the cell's
    Hermite mass point, scaled so ``regularization`` is relative to the
    normal count.  Planar cells (rank-1 systems) land on the face near the
    mass point, crease cells slide onto the crease line, and corner cells
    reproduce the corner; the bias a corner inherits from the
    regularization is of order ``regularization × cell size``.

    Fully JAX-traceable: call under ``jax.grad``/``jax.jit`` with a
    ``hermite`` built from traced parameters to differentiate vertex
    positions.  Vertices are clamped to their cell as a final safety net;
    the clamp binds only on degenerate cells.

    Args:
        hermite: Stage-1 Hermite data for the frozen edge set.
        incidence: Cell-to-edge mapping for the same edge set.
        grid: The sampling grid (cell bounds for clamping).
        regularization: Relative Tikhonov weight; must be positive.

    Returns:
        ``(vertices, normals)``, both shaped ``(cell_count, 3)``.  Normals
        are the safely averaged incident unit normals; a cell whose normals
        cancel (opposed thin-sheet subgradients) falls back to its first
        sample's normal so downstream orientation and shading stay finite.
    """
    if not regularization > 0:
        raise ValueError("regularization must be positive.")

    ids = jnp.asarray(np.maximum(incidence.edge_ids, 0))
    mask = jnp.asarray((incidence.edge_ids >= 0).astype(np.float32))
    normals = hermite.unit_normals()[ids] * mask[..., None]
    points = hermite.points[ids]
    counts = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    mass_point = jnp.sum(points * mask[..., None], axis=1) / counts[:, None]

    system = jnp.einsum("cei,cej->cij", normals, normals)
    plane_offsets = jnp.einsum("cei,cei->ce", normals, points - mass_point[:, None, :])
    rhs = jnp.einsum("cei,ce->ci", normals, plane_offsets)
    damping = regularization * counts
    regularized = system + damping[:, None, None] * jnp.eye(3, dtype=system.dtype)
    vertices = mass_point + jnp.linalg.solve(regularized, rhs[..., None])[..., 0]

    cell_min = jnp.asarray(
        np.asarray(grid.origin, dtype=np.float64)
        + incidence.cells * np.asarray(grid.spacing, dtype=np.float64)
    )
    vertices = jnp.clip(vertices, cell_min, cell_min + jnp.asarray(grid.spacing))

    mean_normal = jnp.sum(normals, axis=1) / counts[:, None]
    squared = jnp.sum(mean_normal**2, axis=-1, keepdims=True)
    valid = squared > 1e-12
    safe = jnp.where(valid, squared, 1.0)
    averaged = jnp.where(valid, mean_normal / jnp.sqrt(safe), normals[:, 0, :])
    return vertices, averaged


def sharp_qef_vertices(
    hermite: HermiteData,
    incidence: CellIncidence,
    grid: GridSpec,
    *,
    rcond: float = 5e-2,
) -> np.ndarray:
    """Place vertices with a rank-revealing QEF that lands exactly on features.

    The truncated pseudo-inverse keeps only the constraint directions the
    cell's Hermite planes actually determine: planar cells project the mass
    point onto their face, crease cells land on the crease line, and corner
    cells reproduce the corner — with none of the mass-point bias the
    Tikhonov solve trades for smooth gradients.  On a uniform grid that
    bias differs per cell (it depends on how the grid slices the surface),
    which is exactly the wiggle visible along feature curves.

    Concrete forward placement only: singular-value truncation has no
    usable derivative, so optimization losses should keep differentiating
    :func:`qef_vertices`.  The two solutions differ by the regularization
    bias, of order ``regularization × cell size``.
    """
    if not 0 < rcond < 1:
        raise ValueError("rcond must be between 0 and 1.")

    edge_ids = np.maximum(incidence.edge_ids, 0)
    mask = (incidence.edge_ids >= 0).astype(np.float64)
    normals = np.asarray(hermite.unit_normals(), dtype=np.float64)[edge_ids] * mask[..., None]
    points = np.asarray(hermite.points, dtype=np.float64)[edge_ids]
    counts = np.maximum(mask.sum(axis=1), 1.0)
    mass_point = (points * mask[..., None]).sum(axis=1) / counts[:, None]

    offsets = np.einsum("cei,cei->ce", normals, points - mass_point[:, None, :])
    u, singular, vt = np.linalg.svd(normals, full_matrices=False)
    leading = np.maximum(singular[:, :1], 1e-30)
    inverse = np.where(singular > rcond * leading, 1.0 / np.maximum(singular, 1e-30), 0.0)
    solution = np.einsum(
        "cji,cj->ci",
        vt,
        inverse * np.einsum("cej,ce->cj", u, offsets),
    )
    vertices = mass_point + solution

    cell_min = np.asarray(grid.origin, dtype=np.float64) + incidence.cells * np.asarray(
        grid.spacing, dtype=np.float64
    )
    return np.clip(vertices, cell_min, cell_min + np.asarray(grid.spacing, dtype=np.float64))


# Neighbor cells around an axis-``a`` edge, as (du, dv) lattice offsets to
# subtract on the two other axes ``u = (a+1) % 3`` and ``v = (a+2) % 3``.
# The order walks counterclockwise in the (u, v) plane seen from +a, so a
# surface whose outward normal points along +a gets front-facing quads.
_QUAD_NEIGHBOR_OFFSETS = ((1, 1), (0, 1), (0, 0), (1, 0))


def dual_faces(
    edges: CrossingEdges,
    incidence: CellIncidence,
    grid: GridSpec,
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Connect cell vertices with one oriented quad per crossing edge.

    Orientation is decided purely by the frozen ``start_inside`` flag: the
    outward normal of the surface crossing an edge points from its inside
    endpoint to its outside endpoint.  Quads are triangulated along their
    shorter diagonal.  Edges on the grid boundary lack one of their four
    cells and are skipped.

    Args:
        edges: Crossing edges from :func:`find_crossing_edges`.
        incidence: Cell-to-edge mapping for the same edge set.
        grid: The sampling grid.
        vertices: Concrete vertex positions ``(cell_count, 3)`` used only to
            pick the shorter quad diagonal (a frozen, discrete choice).

    Returns:
        ``(quads, triangles, skipped_boundary_edges)``.
    """
    vertex_rows = np.full(grid.cells, -1, dtype=np.int32)
    vertex_rows[tuple(incidence.cells.T)] = np.arange(incidence.count, dtype=np.int32)
    positions = np.asarray(vertices, dtype=np.float64)

    quads: list[tuple[int, int, int, int]] = []
    skipped_boundary = 0
    axis_array = np.asarray(edges.axis)
    index_array = np.asarray(edges.index)
    inside_array = np.asarray(edges.start_inside)
    for edge_id in range(edges.count):
        axis = int(axis_array[edge_id])
        u = (axis + 1) % 3
        v = (axis + 2) % 3
        index = index_array[edge_id]
        if not (0 < index[u] < grid.cells[u] and 0 < index[v] < grid.cells[v]):
            skipped_boundary += 1
            continue
        rows = []
        for du, dv in _QUAD_NEIGHBOR_OFFSETS:
            cell = index.copy()
            cell[u] -= du
            cell[v] -= dv
            rows.append(int(vertex_rows[tuple(cell)]))
        if min(rows) < 0:
            # Cannot happen for a well-formed incidence (each neighbor owns
            # this crossing edge), but never emit an invalid quad.
            continue
        if not inside_array[edge_id]:
            rows.reverse()
        quads.append(tuple(rows))

    quad_array = np.asarray(quads, dtype=np.int32).reshape((-1, 4))
    triangles: list[tuple[int, int, int]] = []
    for a, b, c, d in quad_array:
        diagonal_ac = np.sum((positions[a] - positions[c]) ** 2)
        diagonal_bd = np.sum((positions[b] - positions[d]) ** 2)
        if diagonal_ac <= diagonal_bd:
            triangles.extend(((a, b, c), (a, c, d)))
        else:
            triangles.extend(((a, b, d), (b, c, d)))
    triangle_array = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    return quad_array, triangle_array, skipped_boundary


def extract_mesh(
    sdf: Callable[[Array], Array],
    grid: GridSpec,
    *,
    level: float = 0.0,
    bisection_iterations: int = 16,
    newton_steps: int = 1,
    regularization: float = 1e-3,
    sharp: bool = True,
    lipschitz: float | None = None,
) -> Mesh:
    """Extract a feature-preserving dual-contour mesh from an implicit field.

    Convenience composition of the pipeline stages with concrete inputs:
    sample, detect crossing edges, refine Hermite data, place QEF vertices,
    and build oriented dual faces.  With ``sharp=True`` (the default) the
    forward vertices come from :func:`sharp_qef_vertices`, which lands
    exactly on creases and corners; ``sharp=False`` keeps the Tikhonov
    placement everywhere.  Passing ``lipschitz`` switches detection to
    octree pruning (:mod:`jaxcad.meshing.adaptive`): identical output,
    surface-proportional cost, safe only when the value genuinely bounds
    the field's gradient.  For parameter gradients, freeze the detected
    edges/incidence and differentiate through :func:`edge_hermite_data` +
    :func:`qef_vertices` as shown in the module docstring.

    Warns when the surface crosses the grid boundary; the returned mesh is
    open there.
    """
    if lipschitz is None:
        edges = find_crossing_edges(sample_grid(sdf, grid), level=level)
    else:
        from jaxcad.meshing.adaptive import sparse_crossing_edges

        edges = sparse_crossing_edges(sdf, grid, level=level, lipschitz=lipschitz)
    incidence = cell_edge_incidence(edges, grid)
    if edges.count == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return Mesh(
            vertices=jnp.asarray(empty),
            faces=np.empty((0, 3), dtype=np.int32),
            quads=np.empty((0, 4), dtype=np.int32),
            normals=jnp.asarray(empty.copy()),
            cells=np.empty((0, 3), dtype=np.int32),
        )
    hermite = edge_hermite_data(
        sdf,
        grid,
        edges,
        level=level,
        bisection_iterations=bisection_iterations,
        newton_steps=newton_steps,
    )
    vertices, normals = qef_vertices(hermite, incidence, grid, regularization=regularization)
    if sharp:
        vertices = jnp.asarray(sharp_qef_vertices(hermite, incidence, grid), dtype=vertices.dtype)
    quads, faces, skipped_boundary = dual_faces(edges, incidence, grid, np.asarray(vertices))
    if skipped_boundary:
        warnings.warn(
            f"The isosurface crosses the extraction boundary on {skipped_boundary} "
            "grid edges; the returned mesh is open.",
            stacklevel=2,
        )
    return Mesh(
        vertices=vertices,
        faces=faces,
        quads=quads,
        normals=normals,
        cells=incidence.cells.copy(),
    )
