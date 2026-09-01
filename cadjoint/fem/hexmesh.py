"""Volumetric HEX8 meshing of signed distance fields.

Voxelizes an SDF on a regular lattice (cells whose center is inside are
kept), then Newton-projects boundary vertices onto the zero set along the
field gradient so element faces track the surface.  The result is a
:class:`HexMesh` whose cells use VTK/meshio ``hexahedron`` corner ordering,
which is also what jax-fem's ``HEX8`` element consumes.

The projection runs through JAX, so with frozen connectivity the node
positions can be recomputed differentiably for a traced SDF via
:func:`recompute_points` — the hook used by the end-to-end
design-parameter -> mesh -> FEM gradient path.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

import numpy as np

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

# VTK/meshio hexahedron corner ordering for the cell with lower lattice
# corner (i, j, k): index offsets (di, dj, dk) per corner.
_HEX_CORNER_OFFSETS = np.array(
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ],
    dtype=np.int64,
)

# The six quad faces of a VTK hex, each listed with outward orientation.
_HEX_FACES = np.array(
    [
        (0, 4, 7, 3),  # -x
        (1, 2, 6, 5),  # +x
        (0, 1, 5, 4),  # -y
        (3, 7, 6, 2),  # +y
        (0, 3, 2, 1),  # -z
        (4, 5, 6, 7),  # +z
    ],
    dtype=np.int64,
)

# Corner tetrahedra (a, b, c, d): det[p_b - p_a, p_c - p_a, p_d - p_a] is
# positive at every corner of the reference cube, so a positive determinant
# at all eight corners certifies a non-inverted hex.
_CORNER_TETS = np.array(
    [
        (0, 1, 3, 4),
        (1, 2, 0, 5),
        (2, 3, 1, 6),
        (3, 0, 2, 7),
        (4, 7, 5, 0),
        (5, 4, 6, 1),
        (6, 5, 7, 2),
        (7, 6, 4, 3),
    ],
    dtype=np.int64,
)


class FaceGroup(NamedTuple):
    """A batch of boundary quads tagged for boundary-condition selection.

    Attributes:
        nodes: Vertex indices per face, shaped ``(M, 4)``, outward oriented.
        centers: Face centroids, shaped ``(M, 3)``.
        normals: Unit outward geometric normals, shaped ``(M, 3)``.
    """

    nodes: np.ndarray
    centers: np.ndarray
    normals: np.ndarray


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
            topology so :func:`recompute_points` can rebuild ``points``
            differentiably for a traced SDF.
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


def project_points(sdf: Callable[[Any], Any], points: Any, max_step: float, *, steps: int = 8):
    """Newton-project points onto the SDF zero set along the field gradient.

    Mirrors the seam projection used by the viewer's feature extraction but
    for a single field: each iteration moves ``x`` by
    ``-f(x) grad f / |grad f|^2``, and the total displacement from the
    starting point is clamped to ``max_step`` so callers can bound how far
    vertices wander (keeping interior elements well-shaped).

    Pure JAX, fixed iteration count — safe to trace and differentiate with
    respect to SDF parameters.

    Args:
        sdf: Scalar field callable on ``(3,)`` points (batched via vmap).
        points: Starting positions, shaped ``(M, 3)``.
        max_step: Maximum total displacement per point.
        steps: Newton iterations.

    Returns:
        Projected positions as a JAX array, shaped ``(M, 3)``.
    """
    import jax
    import jax.numpy as jnp

    value_and_grad = jax.vmap(jax.value_and_grad(lambda p: jnp.asarray(sdf(p)).reshape(())))
    start = jnp.asarray(points)
    x = start
    for _ in range(steps):
        value, gradient = value_and_grad(x)
        squared = jnp.sum(gradient * gradient, axis=-1, keepdims=True)
        step = value[:, None] * gradient / jnp.maximum(squared, 1e-12)
        x = x - step
        displacement = x - start
        # Guarded norm: at zero displacement (vertex already on the surface)
        # a bare norm's gradient is 0/0, and the NaN would leak through
        # minimum() into otherwise-clean cotangents.
        squared_displacement = jnp.sum(displacement * displacement, axis=-1, keepdims=True)
        length = jnp.sqrt(jnp.maximum(squared_displacement, 1e-24))
        scale = jnp.minimum(1.0, max_step / length)
        x = start + displacement * scale
    return x


def corner_tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Signed corner-tetrahedron volumes (determinants) of each hex.

    Args:
        points: Vertex positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)``.

    Returns:
        Determinants shaped ``(C, 8)``; all positive for well-oriented,
        non-inverted hexes.
    """
    corners = points[cells]  # (C, 8, 3)
    tets = corners[:, _CORNER_TETS]  # (C, 8, 4, 3)
    edges = tets[:, :, 1:, :] - tets[:, :, :1, :]  # (C, 8, 3, 3)
    return np.linalg.det(edges)


# The twelve edges of a VTK hex (bottom ring, top ring, verticals).
_HEX_EDGES = np.array(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ],
    dtype=np.int64,
)


def scaled_jacobians(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Per-element scaled Jacobian quality metric of each hex.

    At each of the eight corners the Jacobian determinant (the corner-tet
    volume of :func:`corner_tet_volumes`) is normalized by the product of
    the lengths of the three edges meeting at that corner; the element value
    is the minimum over its corners.  A perfect cube scores ``1.0``; values
    approach ``0`` as corners flatten and turn negative on inversion.

    Args:
        points: Vertex positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)``.

    Returns:
        Scaled Jacobian per cell, shaped ``(C,)``, in ``[-1, 1]``.
    """
    corners = points[cells]  # (C, 8, 3)
    tets = corners[:, _CORNER_TETS]  # (C, 8, 4, 3)
    edges = tets[:, :, 1:, :] - tets[:, :, :1, :]  # (C, 8, 3, 3)
    determinants = np.linalg.det(edges)  # (C, 8)
    lengths = np.linalg.norm(edges, axis=-1)  # (C, 8, 3)
    denominator = np.prod(lengths, axis=-1)  # (C, 8)
    return np.min(determinants / np.maximum(denominator, 1e-30), axis=1)


def aspect_ratios(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Per-element edge aspect ratio of each hex.

    The ratio of the longest to the shortest of the twelve element edges.
    A perfect cube scores ``1.0``; larger is worse.

    Args:
        points: Vertex positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)``.

    Returns:
        Aspect ratio per cell, shaped ``(C,)``, at least ``1.0``.
    """
    corners = points[cells]  # (C, 8, 3)
    vectors = corners[:, _HEX_EDGES[:, 1]] - corners[:, _HEX_EDGES[:, 0]]  # (C, 12, 3)
    lengths = np.linalg.norm(vectors, axis=-1)  # (C, 12)
    return lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-30)


def _boundary_face_rows(cells: np.ndarray) -> np.ndarray:
    """Outward-oriented quads used by exactly one cell, shaped ``(M, 4)``."""
    faces = cells[:, _HEX_FACES].reshape(-1, 4)  # (C*6, 4), oriented
    keys = np.sort(faces, axis=1)
    _, first_index, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return faces[first_index[counts == 1]]


def _face_geometry(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroids and unit normals of quads (area-weighted diagonal cross)."""
    quad = points[faces]  # (M, 4, 3)
    centers = quad.mean(axis=1)
    normals = 0.5 * np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    return centers, normals / np.maximum(lengths, 1e-30)


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


def select_faces(mesh: HexMesh, predicate: Callable[..., Any]) -> FaceGroup:
    """Select boundary faces whose center (and normal) satisfy a predicate.

    Args:
        mesh: The hex mesh.
        predicate: Called per face as ``predicate(center)`` or
            ``predicate(center, normal)`` (arity is inspected); truthy return
            keeps the face.

    Returns:
        A :class:`FaceGroup` of the matching faces.
    """
    faces = mesh.all_boundary_faces()
    try:
        arity = len(inspect.signature(predicate).parameters)
    except (TypeError, ValueError):
        arity = 1
    if arity >= 2:
        mask = np.array(
            [bool(predicate(c, n)) for c, n in zip(faces.centers, faces.normals)], dtype=bool
        )
    else:
        mask = np.array([bool(predicate(c)) for c in faces.centers], dtype=bool)
    return FaceGroup(
        nodes=faces.nodes[mask], centers=faces.centers[mask], normals=faces.normals[mask]
    )


def faces_from_nodes(mesh: HexMesh, nodes: Any) -> FaceGroup:
    """Boundary faces spanned by a node set (all four corners selected).

    The bridge from node selections to area-integrated boundary conditions
    (tractions, heat fluxes): a boundary quad carries the load exactly when
    every one of its corner nodes belongs to ``nodes``.

    Args:
        mesh: The hex mesh.
        nodes: Node indices (any integer array-like).

    Returns:
        A :class:`FaceGroup` of the spanned faces (possibly empty).
    """
    indices = np.asarray(nodes).reshape(-1)
    faces = mesh.all_boundary_faces()
    mask = np.isin(faces.nodes, indices).all(axis=1)
    return FaceGroup(
        nodes=faces.nodes[mask], centers=faces.centers[mask], normals=faces.normals[mask]
    )


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

    corner_index = kept[:, None, :] + _HEX_CORNER_OFFSETS[None, :, :]  # (C, 8, 3)
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


def recompute_points(sdf: Callable[[Any], Any], mesh: HexMesh):
    """Recompute mesh vertex positions differentiably with frozen topology.

    Interior lattice vertices stay at their frozen positions; the vertices
    recorded in ``mesh.snap_mask`` are re-projected onto the (possibly
    traced) SDF's zero set with the same clamp used at extraction.  Because
    connectivity, the snapped-vertex set, and the base lattice are all
    frozen, the output is a differentiable function of the SDF's parameters
    — the mesh half of the design -> mesh -> FEM gradient chain.

    Args:
        sdf: Signed distance field, possibly closing over traced parameters.
        mesh: Mesh extracted at the nominal design by :func:`sdf_to_hex_mesh`.

    Returns:
        Vertex positions as a JAX array shaped like ``mesh.points``.
    """
    import jax.numpy as jnp

    base = jnp.asarray(mesh.base_points)
    indices = np.flatnonzero(mesh.snap_mask)
    if indices.size == 0:
        return base
    projected = project_points(sdf, base[indices], mesh.max_step)
    return base.at[indices].set(projected)
