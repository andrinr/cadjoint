"""Feature-edge overlay for the viewer's mesh view.

Everything behind ``mode="mesh"``: dual-contour the scene on the viewer's
own grid and split the result into the two overlay layers the frontend
draws — the mesh's native quad edges ("wire") and the chords of its true
feature curves ("sharp").

**Where the sharp layer comes from depends on the tier.**  A feature curve
is where two patch zero sets meet, and the derived B-rep finds exactly
that: one dual-contouring pass discovers which patch owns which region of
the surface, the boundary chain between two regions is an edge, and every
point on it is placed by the projection kernel rather than by the lattice.
That is the private tier's, reached here through the ``feature_edges``
plugin kind (:class:`cadjoint.plugins.contracts.FeatureEdges`), and it is
what :func:`_graph_layers` asks for.

Without it — public cadjoint alone — the sharp layer is read off the
lattice instead (:mod:`cadjoint.viewer._lattice_edges`): feature cells are
classified from normal spreads and exact ``min``/``max`` CSG seams, the
seam vertices are Newton-projected onto the common zero set of the
operands meeting there, and the surviving cell links are the chords.  It
is the same layer the overlay shipped before the graph existed, and it is
honest about what it is: the staircase, the near-parallel rails and the
orphan ticks are fought with tangent, chain-degree and debris tests rather
than solved.  A rim is a polygon of cell links, not a circle.
:mod:`cadjoint.tier` is where the difference is reported.

The wire layer is the dual-contour quad edges either way, drawn on the
re-solved vertex positions of whichever pass produced them so the two
layers agree everywhere.

Only the extraction lives here.  The worker mode that calls it (executing
the user's program first) stays in :mod:`cadjoint.viewer.worker.main`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadjoint.enums import PluginKind
from cadjoint.meshing import DEFAULT_BOUNDS, DEFAULT_SIZE
from cadjoint.meshing.patch_fields import world_frame_leaves
from cadjoint.viewer._lattice_edges import _lattice_layers

#: The overlay's name for the leaf split it shares with the mesher and the
#: graph: the design-subtree rule below is defined over exactly these
#: leaves, and the derived B-rep's decomposition carries the same list in
#: the same order.
_world_frame_leaves = world_frame_leaves


# Mesh-edge view settings.  The grid matches the raymarcher's view volume.
# Detection stays dense rather than Lipschitz-pruned: user-written fields
# can exceed any assumed gradient bound, and a hole in the viewer is worse
# than the ~100 ms this costs.
_MESH_EDGE_BOUNDS = DEFAULT_BOUNDS
_MESH_EDGE_SIZE = DEFAULT_SIZE
_MESH_EDGE_RESOLUTION = 64

# Newton iterations per projection asked of the ``feature_edges`` component.
# Its own default is eight, which converges from a cold seed anywhere in its
# cell; every seed the overlay produces is a mesh-edge midpoint or a chord of
# an already-solved polyline, so it starts within a fraction of a cell and
# four iterations reach the same float32 floor for half the cost.
_PROJECTION_STEPS = 4


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
    of the graph's edges are drawn (:func:`_design_patches` lifts the rule
    from leaves to patches, and an edge is drawn when *both* its patches
    pass — a curve half on scenery is not a crease the user is authoring).
    Everything else stays whole-scene.  In particular the graph is derived
    over every leaf, so the wire layer keeps its full mesh wireframe with
    every vertex re-solved, and a vertex on scenery is classified as scenery
    rather than handed to whichever design operand happens to be nearest.

    The net effect: scenery keeps its full mesh wireframe, solved exactly as
    the design's is, and grows no sharp edges.
    """
    marked = np.array([_carries_construction(leaf) for leaf in leaves], dtype=bool)
    return np.flatnonzero(marked) if marked.any() else None


def _overlay_grid() -> Any:
    """The viewer's own extraction grid — the raymarcher's view volume."""
    from cadjoint.meshing import GridSpec

    return GridSpec.from_bounds(_MESH_EDGE_BOUNDS, _MESH_EDGE_SIZE, _MESH_EDGE_RESOLUTION)


def _graph_layers(
    component: Any, scene: Any, grid: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two layers from the ``feature_edges`` kind — the private tier's.

    One extraction serves both layers: the component runs its own
    dual-contouring pass and hands back the mesh it built the graph over in
    ``stats["mesh"]``, so the wire layer draws that pass's re-solved
    vertices rather than paying for a second sweep.  A component that
    reports no mesh is not an error — the caller runs the public pass for
    the wire layer instead.

    Args:
        component: The object filling the ``feature_edges`` kind.
        scene: Root SDF node.
        grid: The overlay's grid.

    Returns:
        ``(vertices, quad_edges, sharp_chords)`` — see
            :func:`_lattice_layers`.
    """
    import warnings

    with warnings.catch_warnings():
        # The overlay's grid is a fixed view volume the scene is *expected*
        # to leave — a ground plane spans it edge to edge — so the mesher's
        # open-boundary warning would fire on every compile and say nothing
        # a viewer user can act on.  Clipping to the view is the point.
        warnings.filterwarnings("ignore", message="The isosurface crosses the extraction boundary")
        edges = component.feature_edges(
            scene,
            grid,
            design_leaves=_design_leaves(_world_frame_leaves(scene)),
            steps=_PROJECTION_STEPS,
        )
    mesh = edges.stats.get("mesh")
    if mesh is None:
        vertices, quad_edges, _sharp = _lattice_layers(scene, grid)
    else:
        points, quads = mesh
        vertices = np.asarray(points, dtype=np.float64)
        quads = np.asarray(quads, dtype=np.int64)
        quad_edges = np.concatenate(
            [quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]
        )
    return vertices, quad_edges, edges.chords()


def _mesh_edge_layers(scene: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Extract the overlay's two layers, from the graph where it is installed.

    Asks for the ``feature_edges`` kind and falls back to
    :func:`_lattice_layers` when nothing fills it, so the overlay is always
    drawn.

    Returns:
        ``(vertices, wire_edges, sharp, source)`` — the re-solved dual
            vertices, the unique quad edges indexing them, the sharp chords
            as ``(m, 2, 3)``, and which layer produced them (``"graph"`` or
            ``"lattice"``).
    """
    from cadjoint import tier

    grid = _overlay_grid()
    component = tier.component(PluginKind.FEATURE_EDGES.value)
    source = "lattice" if component is None else "graph"
    if component is None:
        vertices, quad_edges, sharp = _lattice_layers(scene, grid)
    else:
        vertices, quad_edges, sharp = _graph_layers(component, scene, grid)
    wire_edges = np.unique(np.sort(quad_edges, axis=1), axis=0)
    return vertices, wire_edges, np.asarray(sharp, dtype=np.float64).reshape(-1, 2, 3), source


def _segments(pairs: np.ndarray) -> list[list[list[float]]]:
    return [[[round(float(value), 3) for value in point] for point in pair] for pair in pairs]


def _mesh_edge_result(scene: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The overlay payload, and the seed a server refreshes it from.

    ``payload["edges"]`` says which layer produced the sharp chords.  The
    seed (:func:`cadjoint.viewer._overlay_refresh.refresh_seed`) is the same
    points with the scene's node table, for re-solving them at another
    design without extracting again; it is optional on top of optional, and
    its own failure leaves the payload intact.

    Optional viewer data: any failure prints a note (captured into the
    compile output) and returns ``None`` rather than failing the compile.
    """
    try:
        vertices, wire_edges, sharp, source = _mesh_edge_layers(scene)
    except Exception as error:  # noqa: BLE001 - viewer extra must never break compiles
        print(f"mesh edge view unavailable: {error}")
        return None, None
    payload = {
        "wire": _segments(vertices[wire_edges]),
        "sharp": _segments(sharp),
        "resolution": _MESH_EDGE_RESOLUTION,
        "edges": source,
    }
    try:
        from cadjoint.viewer._overlay_refresh import refresh_seed

        seed = refresh_seed(scene, vertices, wire_edges, sharp, _MESH_EDGE_RESOLUTION, source)
    except Exception as error:  # noqa: BLE001 - a refresh is a faster path, never the only one
        print(f"mesh edge refresh unavailable: {error}")
        seed = None
    return payload, seed


def _mesh_edge_payload(scene: Any) -> dict[str, Any] | None:
    """The overlay payload alone — see :func:`_mesh_edge_result`."""
    return _mesh_edge_result(scene)[0]
