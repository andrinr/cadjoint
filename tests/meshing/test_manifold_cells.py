"""Manifold dual contouring: multi-vertex cells for multi-sheet geometry.

Two disjoint thin slabs (z half-extent 0.03, centers 0.1 apart in z) whose
surfaces cross the same 0.125-thick cell layer: each slab straddles one of
the two adjacent lattice planes at z = -0.05 and z = 0.075, and the slabs
end in the same cell column in x (walls at x = 0.03 and x = -0.03, both
inside the column [-0.05, 0.075]).  In that column's middle-layer cells the
inside corners split into two *diagonal* groups — box_lo's at the bottom
west, box_hi's at the top east — that are not connected along any cell
edge, so :func:`jaxcad.meshing.features.manifold_cell_incidence` emits two
rows (two QEF vertices) per cell.  Uniform dual contouring's single shared
vertex fused the two slabs there into edges bordered by four triangles;
with the split the extraction yields two separate watertight pancakes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxcad.meshing import (
    GridSpec,
    extract_mesh,
    find_crossing_edges,
    manifold_cell_incidence,
    sample_grid,
)
from jaxcad.sdf.primitives import Box
from tests.meshing.test_scenes import euler_characteristic, undirected_edge_counts

SLAB_HALF = jnp.array([0.215, 0.4, 0.03])
LO_CENTER = jnp.array([-0.185, 0.0, -0.05])  # x in [-0.4, 0.03], z in [-0.08, -0.02]
HI_CENTER = jnp.array([0.185, 0.0, 0.05])  # x in [-0.03, 0.4], z in [0.02, 0.08]


def two_slabs(p):
    return jnp.minimum(Box.sdf(p - LO_CENTER, SLAB_HALF), Box.sdf(p - HI_CENTER, SLAB_HALF))


# 0.125-thick cells with lattice planes at z = -0.05 (inside the lower slab)
# and z = 0.075 (inside the upper slab); the x column [-0.05, 0.075] contains
# both slab end walls.
GRID = GridSpec.from_bounds((-0.55, -0.55, -0.3), (1.125, 1.125, 0.625), (9, 9, 5))


def face_components(faces: np.ndarray) -> int:
    """Number of connected components under face adjacency (shared edges)."""
    parent = np.arange(faces.shape[0])

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    by_edge: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(faces):
        for corner in range(3):
            key = tuple(sorted((int(face[corner]), int(face[(corner + 1) % 3]))))
            if key in by_edge:
                parent[find(by_edge[key])] = find(face_index)
            else:
                by_edge[key] = face_index
    return len({find(index) for index in range(faces.shape[0])})


def test_shared_cells_get_one_row_per_inside_component():
    values = sample_grid(two_slabs, GRID)
    edges = find_crossing_edges(values)
    incidence = manifold_cell_incidence(edges, GRID, values < 0.0)
    cells, counts = np.unique(incidence.cells, axis=0, return_counts=True)
    doubled = cells[counts == 2]
    # The stagger column (x cell 4) in the middle z layer (z cell 2), for
    # every y cell the slabs span: two diagonal inside-corner components.
    assert doubled.shape[0] > 0
    assert set(map(tuple, doubled)) == {(4, y, 2) for y in range(1, 8)}
    assert counts.max() == 2


def test_two_slabs_extract_as_two_watertight_components():
    mesh = extract_mesh(two_slabs, GRID)
    counts = undirected_edge_counts(mesh.faces)
    np.testing.assert_array_equal(np.unique(counts), [2])
    assert face_components(mesh.faces) == 2
    # Two genus-0 closed pancakes: Euler characteristic 2 + 2.
    assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == 4


def test_two_slab_extraction_is_deterministic():
    first = extract_mesh(two_slabs, GRID)
    second = extract_mesh(two_slabs, GRID)
    np.testing.assert_array_equal(first.faces, second.faces)
    np.testing.assert_array_equal(first.quads, second.quads)
    np.testing.assert_array_equal(first.cells, second.cells)
    np.testing.assert_array_equal(np.asarray(first.vertices), np.asarray(second.vertices))
