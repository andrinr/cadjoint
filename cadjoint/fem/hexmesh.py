"""Volumetric HEX8 meshing of signed distance fields.

What belongs here: turning an SDF into a :class:`HexMesh` and nothing more.
Voxelization on a regular lattice (cells whose center is inside are kept),
the boundary-vertex snapping that projects vertices onto the zero set along
the field gradient with an inversion guard, and the grouping of the outer
quads by dominant gradient axis.  The result uses VTK/meshio
``hexahedron`` corner ordering, which is also what jax-fem's ``HEX8``
element consumes.

What does *not* belong here, and where it lives instead:

- element topology tables — :mod:`cadjoint.fem.elements`
- quality metrics (``scaled_jacobians``, ``aspect_ratios``,
  ``corner_tet_volumes``) — :mod:`cadjoint.fem.quality`
- boundary faces and their selection (:class:`~cadjoint.fem.boundary.FaceGroup`,
  ``select_faces``, ``faces_from_nodes``) — :mod:`cadjoint.fem.boundary`
- differentiable node motion (``project_points``, ``recompute_points``) —
  :mod:`cadjoint.fem.motion`
- solving — :mod:`cadjoint.fem.jaxfem` / :mod:`cadjoint.fem.backends`

Those four modules are shared with :mod:`cadjoint.fem.tetmesh`, so the two
element families are layered identically.  The names below are re-exported
for callers that have always imported them from here.

The projection runs through JAX, so with frozen connectivity the node
positions can be recomputed differentiably for a traced SDF via
:func:`~cadjoint.fem.motion.recompute_points` — the hook used by the
end-to-end design-parameter -> mesh -> FEM gradient path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from cadjoint.fem.boundary import (
    FaceGroup,
    _boundary_face_rows,
    _face_geometry,
    faces_from_nodes,
    select_faces,
)
from cadjoint.fem.elements import HEX_CORNER_OFFSETS
from cadjoint.fem.motion import project_points, recompute_points
from cadjoint.fem.quality import aspect_ratios, corner_tet_volumes, scaled_jacobians
from cadjoint.meshing import GridSpec

__all__ = [
    "FaceGroup",
    "GridSpec",
    "HexMesh",
    "aspect_ratios",
    "corner_tet_volumes",
    "faces_from_nodes",
    "project_points",
    "recompute_points",
    "scaled_jacobians",
    "sdf_to_hex_mesh",
    "select_faces",
]


@dataclass(frozen=True)
class HexMesh:
    """A HEX8 volume mesh extracted from an SDF.

    Attributes:
        points: Vertex positions after snapping, shaped ``(N, 3)`` float64.
        cells: HEX8 connectivity in VTK/meshio corner order, ``(C, 8)`` int32.
        boundary_faces: Outer quads (faces used by exactly one cell) grouped
            by the dominant SDF-gradient axis at their centers, keyed
            ``"+x"``/``"-x"``/... — so e.g. all faces of a box lying on its
            +x side land in one group.
        base_points: Unsnapped lattice positions of the same vertices,
            ``(N, 3)``.  Together with ``snap_mask`` this freezes the
            topology so :func:`~cadjoint.fem.motion.recompute_points` can
            rebuild ``points`` differentiably for a traced SDF.
        snap_mask: Boolean ``(N,)`` mask of vertices that were projected
            onto the zero set (after the inversion guard).
        max_step: Projection displacement clamp used for snapping (half the
            cell diagonal).
        grid: The sampling grid the mesh was extracted from.
    """

    points: np.ndarray
    cells: np.ndarray
    boundary_faces: dict[str, FaceGroup]
    base_points: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    snap_mask: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    max_step: float = 0.0
    grid: GridSpec | None = None

    @property
    def num_points(self) -> int:
        """Number of vertices."""
        return int(self.points.shape[0])

    @property
    def num_cells(self) -> int:
        """Number of hexahedra."""
        return int(self.cells.shape[0])

    def all_boundary_faces(self) -> FaceGroup:
        """All boundary quads concatenated across gradient-axis groups."""
        groups = list(self.boundary_faces.values())
        return FaceGroup(
            nodes=np.concatenate([g.nodes for g in groups], axis=0),
            centers=np.concatenate([g.centers for g in groups], axis=0),
            normals=np.concatenate([g.normals for g in groups], axis=0),
        )


def _evaluate_sdf(sdf: Callable[[Any], Any], points: np.ndarray) -> np.ndarray:
    """Evaluate ``sdf`` on ``(M, 3)`` points, returning a float64 array."""
    import jax.numpy as jnp

    return np.asarray(sdf(jnp.asarray(points)), dtype=np.float64).reshape(-1)


def _group_boundary_faces(
    sdf: Callable[[Any], Any], points: np.ndarray, faces: np.ndarray
) -> dict[str, FaceGroup]:
    """Group boundary quads by the dominant SDF-gradient axis at their centers."""
    import jax
    import jax.numpy as jnp

    centers, normals = _face_geometry(points, faces)
    gradient = np.asarray(
        jax.vmap(jax.grad(lambda p: jnp.asarray(sdf(p)).reshape(())))(jnp.asarray(centers))
    )
    axis = np.argmax(np.abs(gradient), axis=-1)
    positive = np.take_along_axis(gradient, axis[:, None], axis=-1)[:, 0] >= 0.0
    groups: dict[str, FaceGroup] = {}
    for axis_index, axis_name in enumerate("xyz"):
        for sign_positive, sign in ((True, "+"), (False, "-")):
            mask = (axis == axis_index) & (positive == sign_positive)
            if np.any(mask):
                groups[f"{sign}{axis_name}"] = FaceGroup(
                    nodes=faces[mask], centers=centers[mask], normals=normals[mask]
                )
    return groups


def _snap_boundary_vertices(
    sdf: Callable[[Any], Any],
    points: np.ndarray,
    cells: np.ndarray,
    boundary_faces: np.ndarray,
    max_step: float,
    snap_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project eligible boundary vertices onto the zero set, guarding inversions.

    Candidates are the vertices of boundary faces lying within ``snap_range``
    of the zero set.  All candidates are projected at once (displacement
    clamped to ``max_step``); then any vertex whose move would drive an
    incident hex's corner-tet volume non-positive is reverted, iterating
    until every hex is valid.

    Returns:
        ``(new_points, snap_mask)``.
    """
    candidate = np.zeros(points.shape[0], dtype=bool)
    candidate[np.unique(boundary_faces)] = True
    values = _evaluate_sdf(sdf, points[candidate])
    keep = np.abs(values) <= snap_range
    indices = np.flatnonzero(candidate)[keep]
    if indices.size == 0:
        return points.copy(), np.zeros(points.shape[0], dtype=bool)

    projected = np.asarray(project_points(sdf, points[indices], max_step), dtype=np.float64)

    snap_mask = np.zeros(points.shape[0], dtype=bool)
    snap_mask[indices] = True
    proposed = points.copy()
    proposed[indices] = projected
    projected_by_index = dict(zip(indices.tolist(), projected))

    # Volumes below this threshold count as (nearly) inverted.  Relative to
    # the unsnapped cell volume so the guard is scale-independent.
    cell_volume = float(np.prod([max(step, 1e-30) for step in _cell_spacing(points, cells)]))
    threshold = 1e-9 * cell_volume

    for _ in range(16):
        volumes = corner_tet_volumes(proposed, cells)
        bad_cells = np.flatnonzero(np.any(volumes <= threshold, axis=1))
        if bad_cells.size == 0:
            break
        bad_vertices = np.unique(cells[bad_cells])
        revert = bad_vertices[snap_mask[bad_vertices]]
        if revert.size == 0:
            break
        snap_mask[revert] = False
        proposed[revert] = points[revert]
    final = points.copy()
    moved = np.flatnonzero(snap_mask)
    final[moved] = np.array([projected_by_index[int(i)] for i in moved])
    return final, snap_mask


def _cell_spacing(points: np.ndarray, cells: np.ndarray) -> tuple[float, float, float]:
    """Edge lengths of the first hex along its three lattice axes."""
    first = points[cells[0]]
    return (
        float(np.linalg.norm(first[1] - first[0])),
        float(np.linalg.norm(first[3] - first[0])),
        float(np.linalg.norm(first[4] - first[0])),
    )


def sdf_to_hex_mesh(sdf: Callable[[Any], Any], grid: GridSpec, *, snap: bool = True) -> HexMesh:
    """Extract a HEX8 volume mesh from an SDF on a regular grid.

    Cells whose center lies inside (``sdf < 0``) are kept, sharing vertices
    through the lattice.  With ``snap=True`` boundary vertices within one
    cell diagonal of the zero set are Newton-projected onto the surface
    along the field gradient, with total displacement clamped to half the
    cell diagonal; a vertex is left unsnapped if moving it would invert any
    incident hex (corner-tet volume check).

    Args:
        sdf: Signed distance field callable on ``(..., 3)`` points.
        grid: Sampling lattice.
        snap: Project boundary vertices onto the zero set.

    Returns:
        The extracted :class:`HexMesh`.

    Raises:
        ValueError: If no cell center lies inside the field.
    """
    nx, ny, nz = grid.cells
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    origin = np.asarray(grid.origin, dtype=np.float64)

    index = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    centers = origin + (index + 0.5) * spacing
    inside = _evaluate_sdf(sdf, centers) < 0.0
    if not np.any(inside):
        raise ValueError("No cell center lies inside the SDF; enlarge the grid or resolution.")
    kept = index[inside]  # (C, 3) lower lattice corners

    corner_index = kept[:, None, :] + HEX_CORNER_OFFSETS[None, :, :]  # (C, 8, 3)
    flat = (corner_index[..., 0] * (ny + 1) + corner_index[..., 1]) * (nz + 1) + corner_index[
        ..., 2
    ]
    used, cells = np.unique(flat, return_inverse=True)
    cells = cells.reshape(flat.shape).astype(np.int32)

    used_index = np.stack(
        (used // ((ny + 1) * (nz + 1)), (used // (nz + 1)) % (ny + 1), used % (nz + 1)), axis=-1
    )
    base_points = origin + used_index.astype(np.float64) * spacing

    boundary = _boundary_face_rows(cells)
    max_step = 0.5 * float(np.linalg.norm(spacing))
    if snap:
        points, snap_mask = _snap_boundary_vertices(
            sdf, base_points, cells, boundary, max_step, snap_range=float(np.linalg.norm(spacing))
        )
    else:
        points = base_points.copy()
        snap_mask = np.zeros(points.shape[0], dtype=bool)

    return HexMesh(
        points=points,
        cells=cells,
        boundary_faces=_group_boundary_faces(sdf, points, boundary),
        base_points=base_points,
        snap_mask=snap_mask,
        max_step=max_step,
        grid=grid,
    )
