"""Which world-frame leaves belong to the design, for the feature-edge overlay.

Shared by the graph path (:mod:`cadjoint.viewer._edge_overlay`) and the
lattice fallback (:mod:`cadjoint.viewer._lattice_edges`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadjoint.meshing.patch_fields import world_frame_leaves

#: The overlay's name for the leaf split it shares with the mesher and the
#: graph: the design-subtree rule below is defined over exactly these
#: leaves, and the derived B-rep's decomposition carries the same list in
#: the same order.
_world_frame_leaves = world_frame_leaves


def _carries_construction(node: Any) -> bool:
    """Whether *node*'s subtree carries a construction-layer face set."""
    from cadjoint.construction.faces import FaceSet

    if isinstance(getattr(node, "faces", None), FaceSet):
        return True
    children = getattr(node, "children", None)
    return callable(children) and any(_carries_construction(child) for child in children())


def _design_leaves(leaves: list[Any]) -> np.ndarray | None:
    """Which world-frame leaves the construction tree owns, or ``None``.

    **The design-subtree rule.**  Sharp feature edges are the design's own
    creases and CSG seams.  Context primitives dropped in around it — a
    board, a die, screw heads, decoupling caps — are scenery: the user is
    not authoring their edges, and drawing feature curves on them clutters
    the overlay with geometry the physics never sees.

    No new flag is needed, because the scene already draws the line.  Every
    construction generator attaches a :class:`FaceSet` to the SDF it returns
    — ``extrude``, ``revolve`` and ``loft`` through ``attach_faces``, and the
    ``Solid.*`` primitive mirrors through the same call — so a leaf belongs
    to the design exactly when its subtree carries one.  A leaf assembled by
    hand out of raw SDF primitives carries nothing, and is scenery.

    Returns ``None`` when *no* leaf carries a mirror: a scene written
    directly as an SDF expression is all design, and restricting it to
    nothing would delete its sharp layer entirely.

    **Exactly one thing is restricted**, and it is the sharp layer's: which
    of the graph's edges are drawn (the ``feature_edges`` component lifts
    the rule from leaves to patches, and an edge is drawn when *both* its
    patches pass — a curve half on scenery is not a crease the user is authoring).
    Everything else stays whole-scene.  In particular the graph is derived
    over every leaf, so the wire layer keeps its full mesh wireframe with
    every vertex re-solved, and a vertex on scenery is classified as scenery
    rather than handed to whichever design operand happens to be nearest.

    The net effect: scenery keeps its full mesh wireframe, solved exactly as
    the design's is, and grows no sharp edges.
    """
    marked = np.array([_carries_construction(leaf) for leaf in leaves], dtype=bool)
    return np.flatnonzero(marked) if marked.any() else None
