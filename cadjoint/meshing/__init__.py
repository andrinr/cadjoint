"""Differentiable meshing pipeline for cadjoint implicit fields.

Built bottom-up in tested stages:

1. :mod:`cadjoint.meshing.edge_detection` — grid-edge crossing detection and
   differentiable Hermite data (roots + gradients per crossing edge).
2. :mod:`cadjoint.meshing.features` — sharp-feature classification (face,
   crease, corner cells) from Hermite data, plus exact CSG seam detection
   from ``min``/``max`` branch changes.
3. :mod:`cadjoint.meshing.dual_contouring` — mesh generation: differentiable
   QEF vertex placement over the Hermite data and oriented dual
   connectivity, with exact sharp-feature placement for the forward mesh.
4. :mod:`cadjoint.meshing.adaptive` — octree-pruned surface-cell discovery:
   identical output to dense detection at surface-proportional cost.
5. :mod:`cadjoint.meshing.export` — planar-patch merging into sparse polygon
   meshes and OBJ/STL/STEP serialization.
6. :mod:`cadjoint.meshing.diagnostics` — post-extraction mesh checks: sampled
   surface deviation and self-intersection tests, triangle-quality
   statistics, watertightness, and Euler characteristic.
7. :mod:`cadjoint.meshing.patch_fields` — per-primitive smooth patch fields
   and ``(leaf, patch)`` surface signatures: exact feature edges as the
   loci where ``argmin |field|`` ownership switches.
8. :mod:`cadjoint.meshing.simplify` — post-extraction, topology-safe mesh
   simplification: error-bounded, feature-preserving half-edge collapses
   that merge coplanar and gently curved regions (multi-size leaves after
   the fact).

Remaining stages add multi-size octree leaves inside the extractor.
Discrete choices (edge sets, incidence, connectivity, class labels) are
frozen per extraction; all continuous quantities carry exact JAX
derivatives with respect to design parameters.
"""

from cadjoint.meshing.adaptive import sparse_crossing_edges, surface_cells
from cadjoint.meshing.diagnostics import (
    mesh_report,
    self_intersections,
    surface_deviation,
    triangle_quality,
)
from cadjoint.meshing.dual_contouring import (
    Mesh,
    dual_faces,
    extract_mesh,
    qef_vertices,
    sharp_qef_vertices,
)
from cadjoint.meshing.edge_detection import (
    CrossingEdges,
    GridSpec,
    HermiteData,
    detect_edges,
    edge_hermite_data,
    find_crossing_edges,
    sample_grid,
)
from cadjoint.meshing.export import merge_planar_faces, save_obj, save_step, save_stl
from cadjoint.meshing.features import (
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
    manifold_cell_incidence,
)
from cadjoint.meshing.patch_fields import (
    ScenePatchFields,
    exact_feature_mask,
    patch_signatures,
    scene_patch_fields,
    signature_function,
    world_frame_leaves,
)
from cadjoint.meshing.simplify import simplify_mesh

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
    "surface_deviation",
    "self_intersections",
    "triangle_quality",
    "mesh_report",
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
    "manifold_cell_incidence",
    "classify_feature_cells",
    "active_branches",
    "detect_branch_changes",
    "feature_cell_links",
    "ScenePatchFields",
    "world_frame_leaves",
    "scene_patch_fields",
    "signature_function",
    "patch_signatures",
    "exact_feature_mask",
    "simplify_mesh",
]
