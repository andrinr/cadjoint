"""Render-ready surface payloads for FEM results in the browser viewer.

The playground's simulate mode does not ship the volume mesh to the
browser: it draws the *boundary* of the hex mesh as an indexed triangle
list with one scalar per vertex (temperature, von Mises stress mapped to
nodes, ...).  This module turns a :class:`~jaxcad.fem.hexmesh.HexMesh`
plus a nodal scalar field into that payload, including the face-group
catalog the boundary-condition UI is built from and per-group triangle
ranges so hovering a group can tint exactly its faces.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jaxcad.fem.hexmesh import HexMesh

__all__ = ["boundary_render_payload", "cell_to_node_scalar", "face_group_catalog"]


def _quad_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area of each quad via the half cross product of its diagonals."""
    quad = points[faces]
    normals = 0.5 * np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
    return np.linalg.norm(normals, axis=-1)


def face_group_catalog(mesh: HexMesh) -> list[dict[str, Any]]:
    """Describe each boundary face group for the boundary-condition UI.

    Args:
        mesh: Hex mesh whose ``boundary_faces`` were tagged by the dominant
            SDF-gradient axis at extraction time.

    Returns:
        One entry per group, sorted by id, shaped
        ``{"id", "axis", "side", "center", "area", "faces"}`` — the id is
        the gradient-axis key (``"+x"``, ``"-z"``, ...), the center is the
        area-weighted mean of the group's face centroids.
    """
    catalog: list[dict[str, Any]] = []
    for group_id in sorted(mesh.boundary_faces):
        group = mesh.boundary_faces[group_id]
        areas = _quad_areas(mesh.points, group.nodes)
        total = float(areas.sum())
        weights = areas / max(total, 1e-30)
        center = (group.centers * weights[:, None]).sum(axis=0)
        catalog.append(
            {
                "id": group_id,
                "axis": group_id[1],
                "side": group_id[0],
                "center": [round(float(value), 5) for value in center],
                "area": round(total, 6),
                "faces": int(group.nodes.shape[0]),
            }
        )
    return catalog


def cell_to_node_scalar(mesh: HexMesh, cell_values: np.ndarray) -> np.ndarray:
    """Average a per-cell scalar onto the mesh nodes.

    Each node receives the mean of the values of its incident hexes — the
    usual nodal projection for cell-centered quantities such as the
    von Mises stress evaluated at element centers.

    Args:
        mesh: The hex mesh.
        cell_values: Scalar per cell, shaped ``(C,)``.

    Returns:
        Scalar per node, shaped ``(N,)`` float64.
    """
    values = np.asarray(cell_values, dtype=np.float64).reshape(-1)
    if values.shape[0] != mesh.num_cells:
        raise ValueError(f"Expected one value per cell ({mesh.num_cells}), got {values.shape[0]}.")
    sums = np.zeros(mesh.num_points, dtype=np.float64)
    counts = np.zeros(mesh.num_points, dtype=np.float64)
    flat = mesh.cells.reshape(-1)
    np.add.at(sums, flat, np.repeat(values, 8))
    np.add.at(counts, flat, 1.0)
    return sums / np.maximum(counts, 1.0)


def boundary_render_payload(mesh: HexMesh, node_scalar: np.ndarray) -> dict[str, Any]:
    """Build the indexed-triangle surface payload the viewer renders.

    Boundary quads are emitted group by group (sorted by group id) and
    split into two triangles each, preserving outward orientation.  Only
    the vertices used by boundary faces are shipped; indices refer to the
    compacted vertex list.

    Args:
        mesh: The hex mesh.
        node_scalar: One scalar per mesh node, shaped ``(N,)``.

    Returns:
        ``{"positions", "scalars", "indices", "groups", "range", "vertex_count"}``
        where ``positions`` is a flat xyz list, ``scalars`` has one float per
        compacted vertex, ``indices`` is a flat triangle list, and each group
        entry carries ``{"id", "start", "count"}`` — its range in ``indices``
        (in index units) — merged with the catalog fields of
        :func:`face_group_catalog`.
    """
    scalar = np.asarray(node_scalar, dtype=np.float64).reshape(-1)
    if scalar.shape[0] != mesh.num_points:
        raise ValueError(
            f"Expected one scalar per node ({mesh.num_points}), got {scalar.shape[0]}."
        )

    catalog = {entry["id"]: entry for entry in face_group_catalog(mesh)}
    quads_per_group: list[tuple[str, np.ndarray]] = [
        (group_id, mesh.boundary_faces[group_id].nodes) for group_id in sorted(mesh.boundary_faces)
    ]
    all_quads = np.concatenate([quads for _, quads in quads_per_group], axis=0)

    used, remapped = np.unique(all_quads.reshape(-1), return_inverse=True)
    quads = remapped.reshape(-1, 4).astype(np.int64)
    # Split each outward-oriented quad (a, b, c, d) into (a, b, c) + (a, c, d).
    triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=1)

    positions = mesh.points[used]
    scalars = scalar[used]
    finite = scalars[np.isfinite(scalars)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0

    groups: list[dict[str, Any]] = []
    offset = 0
    for group_id, group_quads in quads_per_group:
        count = int(group_quads.shape[0]) * 6
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
