"""Differentiable meshing pipeline for jaxcad implicit fields.

Built bottom-up in tested stages:

1. :mod:`jaxcad.meshing.edge_detection` — grid-edge crossing detection and
   differentiable Hermite data (roots + gradients per crossing edge).
2. :mod:`jaxcad.meshing.features` — sharp-feature classification (face,
   crease, corner cells) from Hermite data, plus exact CSG seam detection
   from ``min``/``max`` branch changes.
3. :mod:`jaxcad.meshing.dual_contouring` — mesh generation: differentiable
   QEF vertex placement over the Hermite data and oriented dual
   connectivity, with exact sharp-feature placement for the forward mesh.
4. :mod:`jaxcad.meshing.adaptive` — octree-pruned surface-cell discovery:
   identical output to dense detection at surface-proportional cost.
5. :mod:`jaxcad.meshing.export` — planar-patch merging into sparse polygon
   meshes and OBJ/STL/STEP serialization.

Remaining stages add mesh diagnostics and multi-size octree leaves.
Discrete choices (edge sets, incidence, connectivity, class labels) are
frozen per extraction; all continuous quantities carry exact JAX
derivatives with respect to design parameters.
"""

from jaxcad.meshing.adaptive import sparse_crossing_edges, surface_cells
from jaxcad.meshing.dual_contouring import (
    Mesh,
    dual_faces,
    extract_mesh,
    qef_vertices,
    sharp_qef_vertices,
)
from jaxcad.meshing.edge_detection import (
    CrossingEdges,
    GridSpec,
    HermiteData,
    detect_edges,
    edge_hermite_data,
    find_crossing_edges,
    sample_grid,
)
from jaxcad.meshing.export import merge_planar_faces, save_obj, save_step, save_stl
from jaxcad.meshing.features import (
    CORNER,
    CREASE,
    FACE,
    CellIncidence,
    FeatureCells,
    active_branches,
    cell_edge_incidence,
    classify_feature_cells,
    detect_branch_changes,
    feature_cell_links,
)

__all__ = [
    "Mesh",
    "qef_vertices",
    "sharp_qef_vertices",
    "dual_faces",
    "extract_mesh",
    "surface_cells",
    "sparse_crossing_edges",
    "merge_planar_faces",
    "save_obj",
    "save_stl",
    "save_step",
    "GridSpec",
    "CrossingEdges",
    "HermiteData",
    "sample_grid",
    "find_crossing_edges",
    "edge_hermite_data",
    "detect_edges",
    "CellIncidence",
    "FeatureCells",
    "FACE",
    "CREASE",
    "CORNER",
    "cell_edge_incidence",
    "classify_feature_cells",
    "active_branches",
    "detect_branch_changes",
    "feature_cell_links",
]
