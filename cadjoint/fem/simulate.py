"""Thermal and structural simulation on SDF-extracted hex and tet meshes.

What belongs here: the imperative entry points and the *patch resolution*
between them and the solver ABI — turning a user's selection into the
node index sets and exact face lists a backend consumes, and wrapping the
returned field in a result object.  Public entry points
:func:`thermal_solve` and :func:`elastic_solve` resolve boundary patches
(:class:`~cadjoint.studies.selection.NodeSelection` values) against the mesh,
hand array-level BCs to a pluggable solver backend
(:mod:`cadjoint.fem.backends`; direct in-process jax-fem by default), and
return small result objects with VTK export for ParaView.

What does *not* belong here: the finite-element formulations
(:mod:`cadjoint.fem.jaxfem`), the boundary-face rules the patch resolution
calls into (:mod:`cadjoint.fem.boundary`), or the derived quantities the
result objects expose (:mod:`cadjoint.fem.postprocess`).

Patch semantics: a ``NodeSelection`` used for a node-valued condition
(prescribed temperature, clamp) applies to its selected node set directly;
used for an area-integrated condition (traction, heat flux) it spans the
boundary faces all of whose corners are selected
(:func:`~cadjoint.fem.boundary.faces_from_nodes` on hex meshes,
:func:`~cadjoint.fem.boundary.tet_faces_from_nodes` on tet meshes).

Both entry points accept a :class:`~cadjoint.fem.tetmesh.TetMesh`
(``SimMesh(method="tet4"/"tet10")``) with identical semantics: selections
resolve to corner boundary nodes, TET10 boundary conditions are completed
with the patch's midside nodes, and area-integrated conditions target the
exact boundary triangles the selection spans (jax-fem's node-membership
face rule over-selects on tets otherwise).  Tet meshes always solve on the
direct jax-fem path; passing another ``backend`` raises.

Requires the ``fem`` extra (jax-fem) for the default backend; add the
``tesseract`` extra for ``backend="tesseract"``.

For end-to-end design gradients pass ``points=recompute_points(sdf, mesh)``
— a traced array with frozen connectivity — and differentiate the returned
fields with ``jax.grad``; gradients flow through the solver's adjoint into
the projection and on to the SDF parameters.  Solves run concretely (PETSc
assembly), so do not place them under ``jax.jit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cadjoint.fem.backends import SolverBackend
from cadjoint.fem.discretization import ElasticProblem, ThermalProblem
from cadjoint.fem.postprocess import hex_von_mises, tet_von_mises
from cadjoint.studies import NodeSelection

__all__ = ["ElasticResult", "ThermalResult", "elastic_solve", "thermal_solve"]

#: A boundary patch: a node selection.
Patch = NodeSelection

#: A solvable volume mesh (HEX8, or TET4/TET10 via the tet path).
SolveMesh = Any  # any :class:`~cadjoint.fem.discretization.Discretization`


def _require_selection(patch: Any) -> None:
    """Reject anything that is not a :class:`NodeSelection`, naming the fix."""
    if not isinstance(patch, NodeSelection):
        raise TypeError(
            f"Boundary patches are Nodes selections, got {patch!r}. Build one via "
            "Nodes.box/sphere/halfspace/cylinder/side/predicate."
        )


#: meshio cell type per connectivity width (HEX8 / TET4 / TET10).
_MESHIO_CELL_TYPES = {8: "hexahedron", 4: "tetra", 10: "tetra10"}


def _export_vtk(path: str, mesh: SolveMesh, point_data: dict, cell_data: dict) -> None:
    """Write the mesh and fields as VTK via meshio (guarded)."""
    try:
        import meshio
    except ImportError as error:
        raise ImportError(
            "VTK export requires meshio (installed with the 'fem' extra: pip install cadjoint[fem])."
        ) from error
    width = int(np.asarray(mesh.cells).shape[1])
    meshio.Mesh(
        points=np.asarray(mesh.points, dtype=np.float64),
        cells=[(_MESHIO_CELL_TYPES[width], np.asarray(mesh.cells, dtype=np.int64))],
        point_data={k: np.asarray(v) for k, v in point_data.items()},
        cell_data={k: [np.asarray(v)] for k, v in cell_data.items()},
    ).write(path)


def _property_value(value: Any) -> Any:
    """Normalize a material property argument for the solver ABI.

    A plain Python number becomes a ``float`` (the historical coercion, which
    also rejects nonsense early); anything array-like — a per-element ``(C,)``
    field sampled from the scene's materials, or a traced scalar — passes
    through untouched so the backend can broadcast and differentiate it.
    """
    if isinstance(value, bool):
        raise TypeError("Material properties must be numeric, got a bool.")
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _require_direct_backend(backend: Any, what: str) -> None:
    """Tet meshes solve in-process via jax-fem only (no backend registry)."""
    if backend is None:
        return
    name = backend if isinstance(backend, str) else getattr(backend, "name", None)
    if name != "jaxfem":
        raise ValueError(
            f"{what} on tet meshes run on the direct jax-fem path only; "
            f"got backend {name!r}. Use a hex SimMesh for other backends."
        )


@dataclass(frozen=True)
class ThermalResult:
    """Steady-state thermal solution.

    Attributes:
        temperature: Per-node temperature, shaped ``(N,)``.
        mesh: The mesh that was solved on.
    """

    temperature: Any
    mesh: SolveMesh

    def vtk_export(self, path: str) -> None:
        """Write mesh + temperature as a VTK file for ParaView."""
        _export_vtk(path, self.mesh, {"temperature": self.temperature}, {})


@dataclass(frozen=True)
class ElasticResult:
    """Small-strain linear elastic solution.

    Attributes:
        displacement: Per-node displacement, shaped ``(N, 3)``.
        mesh: The mesh that was solved on.
        youngs: Young's modulus used — a scalar, or a per-element ``(C,)``
            array when the study derived it from the scene's materials.
        poisson: Poisson ratio used, scalar or per element like ``youngs``.
    """

    displacement: Any
    mesh: SolveMesh
    youngs: float
    poisson: float

    def von_mises(self) -> np.ndarray:
        """Per-cell von Mises stress evaluated at each element center.

        On hex meshes the displacement gradient is taken from the
        trilinear (HEX8) basis at the element center
        (:func:`~cadjoint.fem.postprocess.hex_von_mises`); on tet meshes
        from the TET4/TET10 basis at the centroid
        (:func:`~cadjoint.fem.postprocess.tet_von_mises`).

        Returns:
            Von Mises stress per cell, shaped ``(C,)``.
        """
        points = np.asarray(self.mesh.points, dtype=np.float64)
        displacement = np.asarray(self.displacement, dtype=np.float64)
        cells = np.asarray(self.mesh.cells)
        recover = hex_von_mises if cells.shape[1] == 8 else tet_von_mises
        return recover(points, cells, displacement, youngs=self.youngs, poisson=self.poisson)

    def vtk_export(self, path: str) -> None:
        """Write mesh + displacement + von Mises stress as VTK for ParaView."""
        _export_vtk(
            path,
            self.mesh,
            {"displacement": self.displacement},
            {"von_mises": self.von_mises()},
        )


def thermal_solve(
    mesh: SolveMesh,
    *,
    conductivity: Any,
    dirichlet: list[tuple[Patch, float]],
    neumann: list[tuple[Patch, float]] | None = None,
    source: float = 0.0,
    backend: str | SolverBackend | None = None,
    points: Any = None,
) -> ThermalResult:
    """Solve steady-state heat conduction ``-div(k grad T) = q`` on the mesh.

    The discretization resolves the patches and dispatches its own solver
    (:mod:`cadjoint.fem.discretization`): hex meshes through the backend
    registry, tet meshes on the direct jax-fem path, cut cells on their own.

    Args:
        mesh: Any discretization.
        conductivity: Thermal conductivity ``k`` — a number, a per-element
            ``(C,)`` field, or a traced scalar.
        dirichlet: ``(patch, value)`` prescribed temperatures; a value may
            be traced.
        neumann: ``(patch, flux)`` inflows per area on the faces a patch spans.
        source: Volumetric heat source ``q``.
        backend: Backend name (``"jaxfem"`` default, ``"tesseract"``) or a
            :class:`~cadjoint.fem.backends.SolverBackend` instance; hex only.
        points: The placement — what the mesh's ``moved`` returned for a
            traced design — for differentiable frozen-topology solves.
            Patches always resolve on the nominal mesh.

    Returns:
        The family's thermal result; ``temperature`` is a JAX array with an
        adjoint VJP w.r.t. the placement, ``conductivity``, and ``source``.
    """
    for patch, _ in [*dirichlet, *(neumann or [])]:
        _require_selection(patch)
    problem = ThermalProblem(
        conductivity=conductivity,
        source=float(source),
        dirichlet=tuple((patch, value) for patch, value in dirichlet),
        neumann=tuple((patch, float(value)) for patch, value in (neumann or [])),
    )
    return mesh.thermal(problem, placement=points, backend=backend)


def elastic_solve(
    mesh: SolveMesh,
    *,
    youngs: Any,
    poisson: Any,
    dirichlet: list[Patch],
    tractions: list[tuple[Patch, Any]],
    backend: str | SolverBackend | None = None,
    points: Any = None,
    body_force: Any = None,
) -> ElasticResult:
    """Solve small-strain linear elasticity on the mesh.

    Args:
        mesh: Any discretization.
        youngs: Young's modulus — a number, a per-element field, or traced.
        poisson: Poisson's ratio, likewise.
        dirichlet: Patches clamped in all three components.
        tractions: ``(patch, vector)`` tractions on the faces a patch spans.
        backend: Backend name or instance; hex only.
        points: The placement (see :func:`thermal_solve`).
        body_force: Optional per-element ``(C, 3)`` body force.

    Returns:
        The family's elastic result; ``displacement`` is a JAX array with
        an adjoint VJP w.r.t. the placement.
    """
    for patch in [*dirichlet, *(patch for patch, _ in tractions)]:
        _require_selection(patch)
    problem = ElasticProblem(
        youngs=youngs,
        poisson=poisson,
        fixed=tuple(dirichlet),
        tractions=tuple((patch, vector) for patch, vector in tractions),
        body_force=body_force,
    )
    return mesh.elastic(problem, placement=points, backend=backend)
