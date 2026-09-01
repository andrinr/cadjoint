"""Reference-element topology tables shared by every FEM module.

What belongs here: the *constant* combinatorics of an element type — the
corner ordering, which local vertices form each face, each edge, each
corner tetrahedron, and the reference-coordinate corner signs.  Nothing
here touches a mesh, an SDF or a solver; the arrays are plain integer (or
sign) tables in VTK/meshio ordering, which is also what jax-fem's
``HEX8``/``TET4``/``TET10`` elements consume.

What does *not* belong here: anything that reads vertex positions.  Metrics
computed from the tables live in :mod:`cadjoint.fem.quality`, boundary
extraction in :mod:`cadjoint.fem.boundary`, and the shape functions the
tables feed in :mod:`cadjoint.fem.postprocess` / the solver layer.

Having one home for these tables is what lets the hex and tet paths share
their quality, motion and boundary code instead of each carrying a private
copy of the same combinatorics.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HEX_CORNER_OFFSETS",
    "HEX_CORNER_SIGNS",
    "HEX_CORNER_TETS",
    "HEX_EDGES",
    "HEX_FACES",
    "TET10_EDGES",
    "TET_FACES",
]

# VTK/meshio hexahedron corner ordering for the cell with lower lattice
# corner (i, j, k): index offsets (di, dj, dk) per corner.
HEX_CORNER_OFFSETS = np.array(
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
HEX_FACES = np.array(
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
HEX_CORNER_TETS = np.array(
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

# The twelve edges of a VTK hex (bottom ring, top ring, verticals).
HEX_EDGES = np.array(
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

# Trilinear HEX8 corner signs in reference coordinates [-1, 1]^3 (VTK
# order): dN_i/dxi at the element center is HEX_CORNER_SIGNS[i] / 8.
HEX_CORNER_SIGNS = np.array(
    [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ],
    dtype=np.float64,
)

# The four triangular faces of a positive-volume tet (v0, v1, v2, v3),
# each listed with outward orientation (face i is opposite vertex i).
TET_FACES = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)], dtype=np.int64)

# The six edges of a tet in meshio ``tetra10`` midside order: nodes 4..9
# are the midpoints of edges (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
TET10_EDGES = np.array([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)], dtype=np.int64)
