"""What a discretization of a scene is, to everything that is not one.

A study solves on one, the optimiser moves one, the viewer draws one and a
selection resolves on one — and none of them should know whether it is a
hex lattice, a tet fill or a cut-cell structure.  This module is the
contract those callers write against; :class:`~cadjoint.fem.hexmesh.HexMesh`,
:class:`~cadjoint.fem.tetmesh.TetMesh` and :class:`~cadjoint.fem.cutfem.CutMesh`
implement it, each carrying its own patch resolution, solver dispatch,
frozen-topology motion, surface and element quality.

The point of the seam is where the branching used to be.  Adding cut cells
touched six modules because each asked ``isinstance(mesh, TetMesh)`` for
its own reason; now a family answers those questions itself, once.

What a caller says to a discretization is a *problem* — conditions as
selections, materials as numbers or per-element fields — and gets back the
family's result object (``temperature`` or ``displacement`` on its nodes,
possibly traced).  What a caller moves it with is a *placement*: whatever
:meth:`Discretization.moved` returned for a traced design field, handed
back to :meth:`Discretization.thermal`/``elastic`` — node positions for a
mesh, the field itself for cut cells, which have no node map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from cadjoint.fem.boundary import FaceGroup

__all__ = ["Discretization", "ElasticProblem", "Surface", "ThermalProblem"]


@dataclass(frozen=True)
class Surface:
    """The boundary a discretization shows and selects on.

    Attributes:
        points: The positions the faces index, ``(N, 3)`` — the mesh's own
            nodes for a hex or tet mesh, the extracted surface vertices for
            cut cells.
        groups: ``(id, faces)`` pairs, faces ``(F, 3)`` triangles or
            ``(F, 4)`` outward quads.  A hex mesh groups its faces by the
            dominant field-gradient axis (``"+x"``, ``"-z"``, …); a
            triangulated surface is one group, ``"surface"``.
    """

    points: np.ndarray
    groups: tuple[tuple[str, np.ndarray], ...]

    @property
    def faces(self) -> np.ndarray:
        """Every face, group by group in id order."""
        return np.concatenate([faces for _, faces in self.groups], axis=0)


@dataclass(frozen=True)
class ThermalProblem:
    """Steady heat conduction: ``−∇·(k∇T) = q`` with conditions by selection.

    ``conductivity`` is a number, a per-element ``(C,)`` field, or a traced
    scalar; ``dirichlet`` pairs a selection with a prescribed temperature
    (possibly traced), ``neumann`` a selection with an inflow per area.
    """

    conductivity: Any
    source: float = 0.0
    dirichlet: tuple[tuple[Any, Any], ...] = ()
    neumann: tuple[tuple[Any, float], ...] = ()


@dataclass(frozen=True)
class ElasticProblem:
    """Small-strain linear elasticity: clamped selections, tractions by selection."""

    youngs: Any
    poisson: Any
    fixed: tuple[Any, ...] = ()
    tractions: tuple[tuple[Any, Any], ...] = ()
    body_force: Any = None


@runtime_checkable
class Discretization(Protocol):
    """A discretized scene: what studies, the optimiser, the viewer and selections need.

    Structural, like :class:`~cadjoint.studies.selection.BoundaryMesh` which
    it extends: a family implements these on its own class.
    """

    points: Any
    cells: Any  #: element connectivity, or None where there are no elements

    @property
    def num_points(self) -> int: ...

    @property
    def num_cells(self) -> int: ...

    @property
    def family(self) -> str:
        """``"hex"``, ``"tet"`` or ``"cut"`` — for reports, never for branching."""
        ...

    def all_boundary_faces(self) -> FaceGroup: ...

    def surface(self) -> Surface: ...

    def quality(self) -> dict[str, np.ndarray]:
        """Per-element metrics, ``{}`` where there are no elements to grade."""
        ...

    def node_patch(self, selection: Any) -> np.ndarray:
        """Node indices a node-valued condition applies to."""
        ...

    def face_patch(self, selection: Any) -> tuple[np.ndarray, Any]:
        """``(nodes, faces)`` an area-integrated condition applies to; raises if it spans no face."""
        ...

    def unresolvable(self, bcs: list) -> str | None:
        """The first condition that resolves to nothing here, described; None if all resolve."""
        ...

    def moved(self, field: Any, *, smooth_passes: int = 0, design: Any = None) -> Any:
        """The placement of this topology under ``field`` (possibly traced): see the module note."""
        ...

    def thermal(
        self, problem: ThermalProblem, *, placement: Any = None, backend: Any = None
    ) -> Any: ...

    def elastic(
        self, problem: ElasticProblem, *, placement: Any = None, backend: Any = None
    ) -> Any: ...

    def traction_work(self, positions: Any, displacement: Any, selection: Any, vector: Any) -> Any:
        """``∫ t · u`` over the faces a traction selection spans, at ``positions``."""
        ...


# ── what every discretization needs to satisfy the contract above ─────────
#
# Shared implementation, not protocol: each family answers `node_patch`,
# `face_patch` and `unresolvable` its own way, but the argument check, the
# prescribed-value rule and the "does this condition still resolve" sweep
# are the same three answers for all of them.  They lived in `hexmesh` and
# were imported from there by `tetmesh`, which is the wrong direction —
# nothing here is about hexahedra.


def _require_selection(patch: Any) -> None:
    from cadjoint.studies import NodeSelection

    if not isinstance(patch, NodeSelection):
        raise TypeError(
            f"Boundary patches are Nodes selections, got {patch!r}. Build one via "
            "Nodes.box/sphere/halfspace/cylinder/side/predicate."
        )


def _scalar_or_traced(value: Any) -> Any:
    """A prescribed value: a plain number as a float, anything traced untouched."""
    return float(value) if isinstance(value, (int, float)) else value


def _unresolvable_on_mesh(mesh: Any, bcs: list) -> str | None:
    """The first condition that finds no nodes, or spans no face, on a volume mesh.

    Selections are anchored in space, so a re-meshed design can move a loaded
    surface out of its selection; node-valued conditions need nodes, the
    area-integrated ones (``HeatFlux``, ``Traction``) a complete boundary face.
    """
    from cadjoint.fem.study import HeatFlux, Traction

    for bc in bcs:
        label = f"boundary condition {type(bc).__name__} {bc.nodes.describe()}"
        try:
            bc.nodes.resolve(mesh)
        except ValueError:
            return f"{label} matched no surface nodes"
        if isinstance(bc, (HeatFlux, Traction)):
            try:
                mesh.face_patch(bc.nodes)
            except ValueError:
                return f"{label} spans no complete boundary face"
    return None
