"""Derived boundary representation: faces, edges and vertices from patch ownership.

cadjoint models implicitly, so it has no stored B-rep — and does not need
one.  Every hard primitive is a ``min``/``max`` over smooth patch fields
with exact surface ownership, which makes the boundary representation
*derivable*: a face is the surviving part of one patch's zero set, an edge
where two meet, a vertex where three do.  Each element is analytic in the
design parameters, so the whole graph differentiates by the implicit
function theorem while its topology stays discrete and frozen per
extraction.

The three products this serves share almost all of their code:

- **Clean extraction and export** — :mod:`cadjoint.brep.graph` finds the
  graph, :mod:`cadjoint.brep.step` writes analytic STEP surfaces from it and
  OBJ/STL from the same tessellation.
- **A draggable B-rep** — :mod:`cadjoint.brep.drag` moves a vertex or edge
  handle by solving for the design parameters that put it there, with the
  sketch constraints held.
- **A differentiable simulation mesh** — :mod:`cadjoint.brep.plc` turns the
  graph into a piecewise-linear complex for TetGen and tags every node with
  its owning face / edge / vertex, so node motion under a parameter change
  is the same projection.

What they share is :mod:`cadjoint.brep.project`: one Newton kernel onto the
intersection of one, two or three zero sets, with an implicit-function
adjoint.  It is the only place positions are computed.
"""

from cadjoint.brep.drag import DragResult, drag_handle, handle_position
from cadjoint.brep.graph import (
    ANALYTIC_KINDS,
    AnalyticSurface,
    BRep,
    BRepEdge,
    BRepFace,
    BRepVertex,
    Patch,
    extract_brep,
)
from cadjoint.brep.plc import PLC, brep_plc, plc_quality, plc_tet_mesh, recompute_plc_points
from cadjoint.brep.project import (
    field_residuals,
    project,
    project_fields,
    stacked_fields,
    transversal,
)
from cadjoint.brep.step import (
    brep_loops,
    brep_triangles,
    brep_volume,
    save_brep_obj,
    save_brep_step,
    save_brep_stl,
    simplify_loop,
)

__all__ = [
    "ANALYTIC_KINDS",
    "AnalyticSurface",
    "BRep",
    "BRepEdge",
    "BRepFace",
    "BRepVertex",
    "DragResult",
    "PLC",
    "Patch",
    "brep_loops",
    "brep_plc",
    "brep_triangles",
    "brep_volume",
    "drag_handle",
    "extract_brep",
    "field_residuals",
    "handle_position",
    "plc_quality",
    "plc_tet_mesh",
    "recompute_plc_points",
    "project",
    "project_fields",
    "save_brep_obj",
    "save_brep_step",
    "save_brep_stl",
    "simplify_loop",
    "stacked_fields",
    "transversal",
]
