"""Boundary faces of a volume mesh: extraction, geometry and selection.

What belongs here: everything that answers *which* faces or nodes a
boundary condition acts on, for either element family — the
:class:`FaceGroup` payload both mesh types return, the "faces used by
exactly one cell" extraction (quads on hexes, triangles on tets), face
centroids and outward normals, the predicate- and node-set-driven
selections that bridge :mod:`cadjoint.fem.selection` to area-integrated
conditions, and the TET10 midside completion a quadratic patch needs.

What does *not* belong here: mesh construction (:mod:`cadjoint.fem.hexmesh`
/ :mod:`cadjoint.fem.tetmesh`), quality metrics
(:mod:`cadjoint.fem.quality`), or anything that solves.  Functions take
meshes structurally — a hex or tet mesh is anything exposing ``points`` /
``cells`` / ``all_boundary_faces()`` (and, for the TET10 helpers,
``edge_parents`` / ``num_corner_points``) — so this module sits *below* the
mesh modules and they import from it, not the other way around.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np

from cadjoint.fem.elements import HEX_FACES, TET_FACES

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from cadjoint.fem.hexmesh import HexMesh
    from cadjoint.fem.tetmesh import TetMesh

__all__ = [
    "FaceGroup",
    "faces_from_nodes",
    "select_faces",
    "tet10_complete_nodes",
    "tet10_face_midsides",
    "tet_boundary_faces",
    "tet_faces_from_nodes",
]


class FaceGroup(NamedTuple):
    """A batch of boundary faces tagged for boundary-condition selection.

    Attributes:
        nodes: Vertex indices per face, outward oriented — shaped
            ``(M, 4)`` for hex quads, ``(M, 3)`` for tet triangles.
        centers: Face centroids, shaped ``(M, 3)``.
        normals: Unit outward geometric normals, shaped ``(M, 3)``.
    """

    nodes: np.ndarray
    centers: np.ndarray
    normals: np.ndarray


def _boundary_face_rows(cells: np.ndarray) -> np.ndarray:
    """Outward-oriented hex quads used by exactly one cell, shaped ``(M, 4)``."""
    faces = cells[:, HEX_FACES].reshape(-1, 4)  # (C*6, 4), oriented
    keys = np.sort(faces, axis=1)
    _, first_index, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return faces[first_index[counts == 1]]


def tet_boundary_faces(cells: np.ndarray) -> np.ndarray:
    """Outward-oriented boundary corner triangles (faces used by exactly one tet).

    Accepts TET4 or TET10 connectivity (only the four corner columns are
    read), like every tet metric.
    """
    faces = np.asarray(cells, dtype=np.int64)[:, :4][:, TET_FACES].reshape(-1, 3)
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


def _tri_geometry(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroids and unit normals of triangles (area-weighted edge cross)."""
    tris = points[faces]  # (M, 3, 3)
    centers = tris.mean(axis=1)
    normals = 0.5 * np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    return centers, normals / np.maximum(lengths, 1e-30)


def select_faces(mesh: HexMesh | TetMesh, predicate: Callable[..., Any]) -> FaceGroup:
    """Select boundary faces whose center (and normal) satisfy a predicate.

    Works on either element family: the mesh only has to expose
    ``all_boundary_faces()``.

    Args:
        mesh: The volume mesh.
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
    """Boundary quads spanned by a node set (all four corners selected).

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


def tet_faces_from_nodes(mesh: TetMesh, nodes: Any) -> np.ndarray:
    """Boundary triangles all three of whose corners belong to ``nodes``.

    The tet analog of :func:`faces_from_nodes`: the bridge from node
    selections to area-integrated boundary conditions.  Returns the raw
    ``(M, 3)`` corner triples (not a :class:`FaceGroup`) because the tet
    solver path targets faces by connectivity, not by geometry.
    """
    indices = np.asarray(nodes).reshape(-1)
    mask = np.isin(mesh.boundary_tris, indices).all(axis=1)
    return mesh.boundary_tris[mask]


def tet10_complete_nodes(mesh: TetMesh, nodes: Any) -> np.ndarray:
    """Extend a corner node set with the midsides of fully contained edges.

    Node selections resolve to corner boundary vertices; a TET10 boundary
    condition must also cover the midside nodes of the selected patch
    (clamps pin the whole quadratic face, and jax-fem's membership face
    selection requires *all six* face nodes in the set).  A midside node
    joins the set exactly when both its corner parents are selected.  On a
    TET4 mesh this is the identity (modulo uniqueness/int32).

    Args:
        mesh: The tet mesh.
        nodes: Corner node indices (any integer array-like).

    Returns:
        Sorted unique int32 node indices including qualifying midsides.
    """
    indices = np.unique(np.asarray(nodes).reshape(-1)).astype(np.int64)
    if mesh.edge_parents is None:
        return indices.astype(np.int32)
    both = np.isin(mesh.edge_parents, indices).all(axis=1)
    midsides = mesh.num_corner_points + np.flatnonzero(both)
    return np.concatenate([indices, midsides]).astype(np.int32)


def tet10_face_midsides(mesh: TetMesh, faces: np.ndarray) -> np.ndarray:
    """Midside node indices of each corner triangle's three edges.

    ``faces`` are corner triangles (e.g. rows of ``boundary_tris``); the
    result row ``k`` holds the midside nodes of edges ``(f0, f1)``,
    ``(f1, f2)``, ``(f2, f0)`` of face ``k`` — the TRI6 completion used
    for quadratic surface integrals
    (:func:`~cadjoint.fem.postprocess.load_work_tri6` takes the
    concatenation ``[faces, midsides]``).

    Args:
        mesh: A TET10 mesh (``edge_parents`` set).
        faces: Corner triangles, ``(M, 3)``.

    Returns:
        Midside node indices, ``(M, 3)`` int64.

    Raises:
        ValueError: On a TET4 mesh, or if a face edge is not a mesh edge.
    """
    if mesh.edge_parents is None:
        raise ValueError("mesh has no midside nodes; promote it with tet10_mesh first.")
    parents = np.asarray(mesh.edge_parents, dtype=np.int64)
    corner_count = mesh.num_corner_points
    keys = parents[:, 0] * corner_count + parents[:, 1]  # sorted (lexicographic rows)
    tri = np.asarray(faces, dtype=np.int64)[:, :3]
    edges = np.sort(np.stack([tri[:, (0, 1)], tri[:, (1, 2)], tri[:, (2, 0)]], axis=1), axis=2)
    wanted = edges[..., 0] * corner_count + edges[..., 1]  # (M, 3)
    position = np.searchsorted(keys, wanted)
    valid = (position < keys.size) & (keys[np.minimum(position, keys.size - 1)] == wanted)
    if not valid.all():
        raise ValueError("faces contain an edge that is not an edge of the mesh.")
    return corner_count + position
