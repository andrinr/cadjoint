"""Finite element analysis on SDF geometry.

The mesher (:mod:`cadjoint.fem.hexmesh`) has no solver dependency.  The
solver layer (:mod:`cadjoint.fem.simulate`) requires the ``fem`` extra
(jax-fem); the tesseract interop layer requires the ``tesseract`` extra —
both are imported lazily so this package imports cleanly without them.
"""

from __future__ import annotations

from cadjoint.fem.hexmesh import (
    FaceGroup,
    GridSpec,
    HexMesh,
    aspect_ratios,
    corner_tet_volumes,
    faces_from_nodes,
    project_points,
    recompute_points,
    scaled_jacobians,
    sdf_to_hex_mesh,
    select_faces,
)
from cadjoint.fem.selection import Nodes, NodeSelection, selection_from_description
from cadjoint.fem.simmesh import SimMesh, capture_sim_meshes

__all__ = [
    "FaceGroup",
    "GridSpec",
    "HexMesh",
    "NodeSelection",
    "Nodes",
    "SimMesh",
    "aspect_ratios",
    "capture_sim_meshes",
    "corner_tet_volumes",
    "faces_from_nodes",
    "project_points",
    "recompute_points",
    "scaled_jacobians",
    "sdf_to_hex_mesh",
    "select_faces",
    "selection_from_description",
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
