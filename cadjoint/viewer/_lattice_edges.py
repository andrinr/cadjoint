"""The overlay's public fallback: both layers read off the lattice.

What :mod:`cadjoint.viewer._edge_overlay` draws when nothing fills the
``feature_edges`` plugin kind.  The extraction is the viewer's own — the
raw sharp QEF on the overlay grid, with no repair — and the sharp layer is
:func:`cadjoint.meshing.lattice_feature_edges` on it, which says what this
tier is and is not.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _lattice_layers(scene: Any, grid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The overlay's two layers read off the lattice.

    The wire layer is the dual-contour mesh's native quad edges
    (triangulation diagonals carry no surface information), drawn on the
    same seam-projected vertices as the sharp layer, so the two agree.

    Args:
        scene: Root SDF node.
        grid: The :class:`~cadjoint.meshing.GridSpec` to extract on.

    Returns:
        ``(vertices, quad_edges, sharp_chords)`` — the re-solved dual
            vertices, the quad edge pairs indexing them, and the sharp
            chords as ``(n, 2, 3)`` point pairs.
    """
    import jax.numpy as jnp

    from cadjoint.meshing import (
        dual_faces,
        edge_hermite_data,
        find_crossing_edges,
        manifold_cell_incidence,
        sample_grid,
        sharp_qef_vertices,
    )
    from cadjoint.meshing.feature_edges import lattice_feature_edges

    sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
    values = sample_grid(sdf, grid)
    edges = find_crossing_edges(values)
    if edges.count == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, np.empty((0, 2), dtype=np.int64), np.empty((0, 2, 3), dtype=np.float64)
    # Manifold incidence: one row per inside-corner component, so cells
    # crossed by two surface sheets keep the sheets on separate vertices.
    incidence = manifold_cell_incidence(edges, grid, values < 0.0)
    hermite = edge_hermite_data(sdf, grid, edges)
    vertices = sharp_qef_vertices(hermite, incidence, grid)
    quads, _triangles, _skipped = dual_faces(edges, incidence, grid, vertices)
    quad_edges = np.concatenate(
        [quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]
    )
    vertices, links = lattice_feature_edges(scene, grid, hermite, incidence, vertices, quads)
    return vertices, quad_edges, vertices[links]
