"""Element size and shape metrics, uniform across element families.

What belongs here: any per-element scalar computed from ``(points, cells)``
alone — signed volumes, inversion determinants, and the shape metrics the
viewer's mesh cards and :class:`~cadjoint.fem.simmesh.SimMesh` report.  The
hex family (``corner_tet_volumes``/``scaled_jacobians``/``aspect_ratios``)
and the tet family (``tet_volumes``/``tet_radius_ratios``/
``tet_aspect_ratios``) live side by side so the two paths report quality
through one module instead of each mesh module carrying its own metrics.

What does *not* belong here: mesh construction, boundary extraction
(:mod:`cadjoint.fem.boundary`), or anything that evaluates an SDF.  Every
function is pure NumPy, accepts TET4 or TET10 connectivity (only the four
corner columns are read on tets), and never mutates its inputs.
"""

from __future__ import annotations

import numpy as np

from cadjoint.fem.elements import HEX_CORNER_TETS, HEX_EDGES, TET10_EDGES, TET_FACES

__all__ = [
    "aspect_ratios",
    "corner_tet_volumes",
    "scaled_jacobians",
    "tet_aspect_ratios",
    "tet_radius_ratios",
    "tet_volumes",
]


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
    tets = corners[:, HEX_CORNER_TETS]  # (C, 8, 4, 3)
    edges = tets[:, :, 1:, :] - tets[:, :, :1, :]  # (C, 8, 3, 3)
    return np.linalg.det(edges)


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
    tets = corners[:, HEX_CORNER_TETS]  # (C, 8, 4, 3)
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
    vectors = corners[:, HEX_EDGES[:, 1]] - corners[:, HEX_EDGES[:, 0]]  # (C, 12, 3)
    lengths = np.linalg.norm(vectors, axis=-1)  # (C, 12)
    return lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-30)


def tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Signed tetrahedron volumes, ``(T,)``; positive for correct orientation."""
    corners = np.asarray(points)[np.asarray(cells)[:, :4]]
    return np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0


def tet_radius_ratios(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Normalized radius ratio ``3 r_in / r_circ`` per tet, in ``(0, 1]``.

    The regular tetrahedron scores 1; slivers approach 0.  Uses the exact
    inradius (``3V / A_total``) and circumradius (Cayley–Menger-free
    formula via opposite-edge products).
    """
    corners = np.asarray(points, dtype=np.float64)[np.asarray(cells)[:, :4]]  # (T, 4, 3)
    volume = np.abs(tet_volumes(points, cells))
    faces = corners[:, TET_FACES]  # (T, 4, 3, 3)
    face_areas = 0.5 * np.linalg.norm(
        np.cross(faces[..., 1, :] - faces[..., 0, :], faces[..., 2, :] - faces[..., 0, :]),
        axis=-1,
    )  # (T, 4)
    inradius = 3.0 * volume / np.maximum(face_areas.sum(axis=1), 1e-30)
    # Circumradius: R = sqrt((a q_a)^2 ... ) / (24 V) with products of
    # opposite edge lengths a = |v1-v0||v3-v2| etc.
    e = [corners[:, j] - corners[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (2, 3), (1, 3), (1, 2))]
    a = np.linalg.norm(e[0], axis=1) * np.linalg.norm(e[3], axis=1)
    b = np.linalg.norm(e[1], axis=1) * np.linalg.norm(e[4], axis=1)
    c = np.linalg.norm(e[2], axis=1) * np.linalg.norm(e[5], axis=1)
    p = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
    circumradius = np.sqrt(np.maximum(p, 0.0)) / np.maximum(24.0 * volume, 1e-30)
    return 3.0 * inradius / np.maximum(circumradius, 1e-30)


def tet_aspect_ratios(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Longest-to-shortest edge ratio per tet, at least 1."""
    corners = np.asarray(points, dtype=np.float64)[np.asarray(cells)[:, :4]]
    edges = corners[:, TET10_EDGES[:, 1]] - corners[:, TET10_EDGES[:, 0]]
    lengths = np.linalg.norm(edges, axis=-1)
    return lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-30)
