"""Finite element analysis on SDF geometry.

The package is layered so the two element families (HEX8 and TET4/TET10)
travel the same path, and so no mesh module depends on a solver:

1. :mod:`~cadjoint.fem.elements` — reference-element topology tables.
2. :mod:`~cadjoint.fem.quality`, :mod:`~cadjoint.fem.boundary`,
   :mod:`~cadjoint.fem.motion` — the concerns both families share: element
   metrics, boundary faces and their selection, and differentiable node
   motion under frozen topology.
3. :mod:`~cadjoint.fem.hexmesh`, :mod:`~cadjoint.fem.tetmesh` — mesh
   construction only.
4. :mod:`~cadjoint.fem.backends` (solver ABI + registry),
   :mod:`~cadjoint.fem.jaxfem` (every direct jax-fem solve, both
   families), :mod:`~cadjoint.fem.calculix`, and
   :mod:`~cadjoint.fem.postprocess` (quantities derived from a solution).
5. :mod:`~cadjoint.fem.simulate`, :mod:`~cadjoint.fem.study` — patch
   resolution and the declarative entry points.

Layers 1-3 have no solver dependency.  The solver layer requires the
``fem`` extra (jax-fem); the tesseract interop layer requires the
``tesseract`` extra — both are imported lazily so this package imports
cleanly without them.
"""

from __future__ import annotations

from cadjoint.fem.boundary import (
    FaceGroup,
    faces_from_nodes,
    select_faces,
    tet_faces_from_nodes,
)
from cadjoint.fem.hexmesh import GridSpec, HexMesh, sdf_to_hex_mesh
from cadjoint.fem.motion import project_points, recompute_points, recompute_tet_points
from cadjoint.fem.quality import (
    aspect_ratios,
    corner_tet_volumes,
    scaled_jacobians,
    tet_aspect_ratios,
    tet_radius_ratios,
)
from cadjoint.fem.selection import Nodes, NodeSelection, selection_from_description
from cadjoint.fem.simmesh import SimMesh, capture_sim_meshes
from cadjoint.fem.tetmesh import TetMesh, sdf_to_tet_mesh

__all__ = [
    "FaceGroup",
    "GridSpec",
    "HexMesh",
    "NodeSelection",
    "Nodes",
    "SimMesh",
    "TetMesh",
    "aspect_ratios",
    "capture_sim_meshes",
    "corner_tet_volumes",
    "faces_from_nodes",
    "project_points",
    "recompute_points",
    "recompute_tet_points",
    "scaled_jacobians",
    "sdf_to_hex_mesh",
    "sdf_to_tet_mesh",
    "select_faces",
    "selection_from_description",
    "tet_aspect_ratios",
    "tet_faces_from_nodes",
    "tet_radius_ratios",
]

# Declarative study layer (code-first simulation, capture registry for the
# compile worker).  Appended additively; no solver import happens here.
from cadjoint.fem.result import SimulationResult  # noqa: E402
from cadjoint.fem.study import (  # noqa: E402
    Dirichlet,
    ElasticStudy,
    Fixed,
    HeatFlux,
    ThermalStudy,
    Traction,
    capture_studies,
)

__all__ += [
    "Dirichlet",
    "ElasticStudy",
    "Fixed",
    "HeatFlux",
    "SimulationResult",
    "ThermalStudy",
    "Traction",
    "capture_studies",
]
