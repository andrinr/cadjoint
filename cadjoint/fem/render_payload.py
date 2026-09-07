"""Render-ready surface payloads for FEM results in the browser viewer.

The playground's simulate mode does not ship the volume mesh to the
browser: it draws the *boundary* of the hex mesh as an indexed triangle
list with one scalar per vertex (temperature, von Mises stress mapped to
nodes, ...).  This module turns a :class:`~cadjoint.fem.hexmesh.HexMesh`
plus a nodal scalar field into that payload, including the face-group
catalog the boundary-condition UI is built from and per-group triangle
ranges so hovering a group can tint exactly its faces.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadjoint.fem.discretization import Surface

__all__ = ["boundary_render_payload", "cell_to_node_scalar", "face_group_catalog"]


def _quad_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area of each quad via the half cross product of its diagonals."""
    quad = points[faces]
    normals = 0.5 * np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
    return np.linalg.norm(normals, axis=-1)


def face_group_catalog(mesh: Any) -> list[dict[str, Any]]:
    """:func:`surface_catalog` of the mesh's own surface."""
    return surface_catalog(mesh.surface())


def cell_to_node_scalar(mesh: Any, cell_values: np.ndarray) -> np.ndarray:
    """Average a per-cell scalar onto the mesh nodes.

    Each node receives the mean of the values of its incident elements —
    the usual nodal projection for cell-centered quantities such as the
    von Mises stress evaluated at element centers.  Works for any fixed
    nodes-per-cell connectivity (HEX8, TET4, TET10 midside nodes
    included), so results stay meshing-method-agnostic.

    Args:
        mesh: The volume mesh (hex or tet).
        cell_values: Scalar per cell, shaped ``(C,)``.

    Returns:
        Scalar per node, shaped ``(N,)`` float64.
    """
    values = np.asarray(cell_values, dtype=np.float64).reshape(-1)
    if values.shape[0] != mesh.num_cells:
        raise ValueError(f"Expected one value per cell ({mesh.num_cells}), got {values.shape[0]}.")
    cells = np.asarray(mesh.cells)
    sums = np.zeros(mesh.num_points, dtype=np.float64)
    counts = np.zeros(mesh.num_points, dtype=np.float64)
    flat = cells.reshape(-1)
    np.add.at(sums, flat, np.repeat(values, cells.shape[1]))
    np.add.at(counts, flat, 1.0)
    return sums / np.maximum(counts, 1.0)


def surface_render_payload(surface: Surface, node_scalar: np.ndarray) -> dict[str, Any]:
    """The viewer's boundary-surface payload for any discretization.

    Faces are emitted group by group (in the surface's group order); quads
    are split into two triangles each, preserving outward orientation, and
    triangles pass through.  Only the vertices the faces use are shipped;
    indices refer to the compacted vertex list.

    Args:
        surface: The discretization's :meth:`~cadjoint.fem.discretization.Discretization.surface`.
        node_scalar: One scalar per point of the surface, shaped ``(N,)``.

    Returns:
        ``{"positions", "scalars", "indices", "groups", "range", "vertex_count"}``
        where ``positions`` is a flat xyz list, ``scalars`` has one float per
        compacted vertex, ``indices`` is a flat triangle list, and each group
        entry carries ``{"id", "axis", "side", "center", "area", "faces",
        "start", "count"}`` — its range in ``indices`` (in index units) and
        the catalog fields of :func:`surface_catalog`.
    """
    scalar = np.asarray(node_scalar, dtype=np.float64).reshape(-1)
    points = np.asarray(surface.points)
    if scalar.shape[0] != points.shape[0]:
        raise ValueError(
            f"Expected one scalar per node ({points.shape[0]}), got {scalar.shape[0]}."
        )
    catalog = {entry["id"]: entry for entry in surface_catalog(surface)}
    all_faces = np.concatenate([np.asarray(faces) for _, faces in surface.groups], axis=0)
    used, remapped = np.unique(all_faces.reshape(-1), return_inverse=True)
    compact = remapped.reshape(all_faces.shape).astype(np.int64)
    triangles = _triangulate(compact)
    positions = points[used]
    scalars = scalar[used]
    finite = scalars[np.isfinite(scalars)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    groups: list[dict[str, Any]] = []
    offset = 0
    for group_id, faces in surface.groups:
        count = int(np.asarray(faces).shape[0]) * (6 if np.asarray(faces).shape[1] == 4 else 3)
        groups.append({**catalog[group_id], "start": offset, "count": count})
        offset += count
    return {
        "positions": [round(float(value), 5) for value in positions.reshape(-1)],
        "scalars": [round(float(value), 6) for value in scalars],
        "indices": [int(value) for value in triangles.reshape(-1)],
        "groups": groups,
        "range": [round(low, 6), round(high, 6)],
        "vertex_count": int(used.shape[0]),
    }


def _triangulate(faces: np.ndarray) -> np.ndarray:
    """Outward quads ``(a, b, c, d)`` as ``(a, b, c) + (a, c, d)``; triangles as they are."""
    if faces.shape[1] == 4:
        return np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=1).reshape(-1, 3)
    return faces


def surface_catalog(surface: Surface) -> list[dict[str, Any]]:
    """One entry per surface group: id, axis and side (for a hex mesh's gradient-axis
    groups such as ``"+x"``; None otherwise), the area-weighted center, the
    area and the face count."""
    points = np.asarray(surface.points)
    catalog: list[dict[str, Any]] = []
    for group_id, faces in surface.groups:
        faces = np.asarray(faces)
        corners = points[faces]
        if faces.shape[1] == 4:
            areas = _quad_areas(points, faces)
        else:
            areas = 0.5 * np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=-1
            )
        total = float(areas.sum())
        weights = areas / max(total, 1e-30)
        center = (corners.mean(axis=1) * weights[:, None]).sum(axis=0)
        axis_group = len(group_id) == 2 and group_id[0] in "+-" and group_id[1] in "xyz"
        catalog.append(
            {
                "id": group_id,
                "axis": group_id[1] if axis_group else None,
                "side": group_id[0] if axis_group else None,
                "center": [round(float(value), 5) for value in center],
                "area": round(total, 6),
                "faces": int(faces.shape[0]),
            }
        )
    return catalog


def boundary_render_payload(mesh: Any, node_scalar: np.ndarray) -> dict[str, Any]:
    """:func:`surface_render_payload` of the mesh's own surface."""
    return surface_render_payload(mesh.surface(), node_scalar)
