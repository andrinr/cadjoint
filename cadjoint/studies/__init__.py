"""What a declared study needs, whatever physics solves it.

`cadjoint.fem` meshes a part and solves conduction or elasticity on it;
`cadjoint.flow` fills a lattice and solves Brinkman-penalised flow on it.
Neither is built on the other, but a *declared study* needs the same three
things either way, and they live here so that a flow study does not have to
import a finite-element package to get them:

- :func:`~cadjoint.studies.capture.capture_studies` and
  :func:`~cadjoint.studies.capture.register_study` — the registry a study
  lands in when a scene program constructs it.
- :class:`~cadjoint.studies.selection.Nodes` and
  :class:`~cadjoint.studies.selection.NodeSelection` — the region language
  boundary conditions are written in, with two evaluations: volumetric over
  any points (:meth:`~cadjoint.studies.selection.NodeSelection.contains`,
  what a lattice asks) and restricted to a mesh's boundary nodes
  (:meth:`~cadjoint.studies.selection.NodeSelection.resolve`, what a solve
  on a meshed part asks).
- :func:`~cadjoint.studies._validate.require_triplet` — the three-finite-
  numbers check every constructor here repeats.

What does *not* live here is the physics: the boundary-condition
vocabularies (``Dirichlet``/``Traction`` against ``Inlet``/``Outlet``), the
study classes, the meshes and the solvers all stay in their own packages.
"""

from __future__ import annotations

from cadjoint.studies._validate import require_triplet
from cadjoint.studies.capture import capture_studies, declared_studies, register_study
from cadjoint.studies.selection import (
    BoundaryMesh,
    Nodes,
    NodeSelection,
    boundary_node_mask,
    selection_from_description,
)

__all__ = [
    "BoundaryMesh",
    "NodeSelection",
    "Nodes",
    "boundary_node_mask",
    "capture_studies",
    "declared_studies",
    "register_study",
    "require_triplet",
    "selection_from_description",
]
