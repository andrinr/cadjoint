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
9. :mod:`cadjoint.meshing.feature_edges` — the part's creases and CSG
   seams read off the lattice, as links between one extraction's vertices.
10. :mod:`cadjoint.meshing.surface` — the front door: :func:`surface_grid`
    (by cell count or cell size, fitted or over given bounds),
    :class:`SurfaceOptions` and :func:`extract_surface`, which returns a
    :class:`Surface`: the welded mesh, crease-split shading normals, feature
    edges, and whether the lattice clipped the part.
11. :mod:`cadjoint.meshing.walls` — :func:`wall_thickness`, the thinnest
    wall measured on the field, in lattice cells.

``docs/integrating.qmd`` is the guide for a program that meshes cadjoint
parts: the functions above that make up the stable surface, and what each
promises.

Remaining stages add multi-size octree leaves inside the extractor.
Discrete choices (edge sets, incidence, connectivity, class labels) are
frozen per extraction; all continuous quantities carry exact JAX
derivatives with respect to design parameters.
"""

from cadjoint.meshing.adaptive import sparse_crossing_edges, surface_cells
from cadjoint.meshing.bounds import boundary_inside, fit_bounds
from cadjoint.meshing.diagnostics import (
    MeshTopology,
    mesh_report,
    mesh_topology,
    self_intersections,
    surface_deviation,
    triangle_quality,
)
from cadjoint.meshing.dual_contouring import (
    Extraction,
    Mesh,
    dual_faces,
    extract_mesh,
    extraction,
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
from cadjoint.meshing.export import (
    merge_planar_faces,
    remove_slivers,
    save_obj,
    save_step,
    save_stl,
)
from cadjoint.meshing.feature_edges import design_leaves, lattice_feature_edges
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
    stacked_values,
    world_frame_leaves,
)
from cadjoint.meshing.simplify import simplify_mesh
from cadjoint.meshing.surface import (
    CREASE_ANGLE,
    RenderMesh,
    Surface,
    SurfaceOptions,
    extract_surface,
    surface_grid,
)
from cadjoint.meshing.walls import MIN_WALL_CELLS, WALL_CELLS, WallThickness, wall_thickness

#: The default sampling volume — the viewer's own view volume — for meshes
#: and studies that declare no box of their own, and the overlay's grid.
DEFAULT_BOUNDS = (-3.0, -3.0, -3.0)
DEFAULT_SIZE = (6.0, 6.0, 6.0)

__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_SIZE",
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
    "remove_slivers",
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
    "stacked_values",
    "signature_function",
    "patch_signatures",
    "exact_feature_mask",
    "simplify_mesh",
    "SurfaceOptions",
    "extract_surface",
    "surface_grid",
    "Surface",
    "RenderMesh",
    "CREASE_ANGLE",
    "Extraction",
    "extraction",
    "fit_bounds",
    "boundary_inside",
    "MeshTopology",
    "mesh_topology",
    "lattice_feature_edges",
    "design_leaves",
    "WallThickness",
    "wall_thickness",
    "MIN_WALL_CELLS",
    "WALL_CELLS",
]
