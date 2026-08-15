"""Differentiable meshing pipeline for jaxcad implicit fields.

Built bottom-up in tested stages:

1. :mod:`jaxcad.meshing.edge_detection` — grid-edge crossing detection and
   differentiable Hermite data (roots + gradients per crossing edge).
2. :mod:`jaxcad.meshing.features` — sharp-feature classification (face,
   crease, corner cells) from Hermite data, plus exact CSG seam detection
   from ``min``/``max`` branch changes.

Later stages add mesh generation from the Hermite data, quality diagnostics,
and adaptivity.  Discrete choices (edge sets, incidence, class labels) are
frozen per extraction; all continuous quantities carry exact JAX derivatives
with respect to design parameters.
"""

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
)

__all__ = [
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
]
