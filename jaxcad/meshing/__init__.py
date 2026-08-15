"""Differentiable meshing pipeline for jaxcad implicit fields.

Built bottom-up in tested stages:

1. :mod:`jaxcad.meshing.edge_detection` — grid-edge crossing detection and
   differentiable Hermite data (roots + gradients per crossing edge).
2. :mod:`jaxcad.meshing.features` — sharp-feature classification (face,
   crease, corner cells) from Hermite data, plus exact CSG seam detection
   from ``min``/``max`` branch changes.
3. :mod:`jaxcad.meshing.dual_contouring` — mesh generation: differentiable
   QEF vertex placement over the Hermite data and oriented dual
   connectivity.

Later stages add quality diagnostics and adaptivity.  Discrete choices
(edge sets, incidence, connectivity, class labels) are frozen per
extraction; all continuous quantities carry exact JAX derivatives with
respect to design parameters.
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
