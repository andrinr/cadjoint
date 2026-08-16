"""Finite element analysis on SDF geometry.

The mesher (:mod:`jaxcad.fem.hexmesh`) has no solver dependency.  The
solver layer (:mod:`jaxcad.fem.simulate`) requires the ``fem`` extra
(jax-fem); the tesseract interop layer requires the ``tesseract`` extra —
both are imported lazily so this package imports cleanly without them.
"""

from __future__ import annotations

from jaxcad.fem.hexmesh import (
    FaceGroup,
    GridSpec,
    HexMesh,
    corner_tet_volumes,
    project_points,
    recompute_points,
    sdf_to_hex_mesh,
    select_faces,
)

__all__ = [
    "FaceGroup",
    "GridSpec",
    "HexMesh",
    "corner_tet_volumes",
    "project_points",
    "recompute_points",
    "sdf_to_hex_mesh",
    "select_faces",
]
