"""Thermal and structural simulation on SDF-extracted hex and tet meshes.

What belongs here: the imperative entry points and the *patch resolution*
between them and the solver ABI — turning a user's selection or predicate
into the node index sets and exact face lists a backend consumes, and
wrapping the returned field in a result object.  Public entry points
:func:`thermal_solve` and :func:`elastic_solve` resolve boundary patches —
:class:`~cadjoint.fem.selection.NodeSelection` values or legacy face
predicates — against the mesh, hand array-level BCs to a pluggable solver
backend (:mod:`cadjoint.fem.backends`; direct in-process jax-fem by
default), and return small result objects with VTK export for ParaView.

What does *not* belong here: the finite-element formulations
(:mod:`cadjoint.fem.jaxfem`), the boundary-face rules the patch resolution
calls into (:mod:`cadjoint.fem.boundary`), or the derived quantities the
result objects expose (:mod:`cadjoint.fem.postprocess`).

Patch semantics: a ``NodeSelection`` used for a node-valued condition
(prescribed temperature, clamp) applies to its selected node set directly;
used for an area-integrated condition (traction, heat flux) it spans the
boundary faces all of whose corners are selected
(:func:`~cadjoint.fem.boundary.faces_from_nodes` on hex meshes,
:func:`~cadjoint.fem.boundary.tet_faces_from_nodes` on tet meshes).  A
callable patch is the legacy face-predicate form resolved via
:func:`~cadjoint.fem.select_faces`.

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
from typing import Any, Callable

import numpy as np

from cadjoint.fem.backends import ElasticBCs, SolverBackend, ThermalBCs, get_backend
from cadjoint.fem.boundary import (
    faces_from_nodes,
    select_faces,
    tet10_complete_nodes,
    tet10_face_midsides,
    tet_faces_from_nodes,
)
from cadjoint.fem.hexmesh import HexMesh
from cadjoint.fem.jaxfem import tet_elastic_solve, tet_thermal_solve
from cadjoint.fem.postprocess import hex_von_mises, tet_von_mises
from cadjoint.fem.selection import NodeSelection
from cadjoint.fem.tetmesh import TetMesh

__all__ = ["ElasticResult", "ThermalResult", "elastic_solve", "thermal_solve"]

Predicate = Callable[..., Any]
#: A boundary patch: a node selection or a legacy face predicate.
Patch = NodeSelection | Predicate

#: A solvable volume mesh (HEX8, or TET4/TET10 via the tet path).
SolveMesh = HexMesh | TetMesh


def _patch_nodes(mesh: HexMesh, predicate: Predicate) -> np.ndarray:
    """Unique vertex indices of the boundary faces matching ``predicate``."""
    group = select_faces(mesh, predicate)
    if group.nodes.size == 0:
        raise ValueError("Boundary-condition predicate selected no boundary faces.")
    return np.unique(group.nodes).astype(np.int32)


def _node_patch(mesh: HexMesh, patch: Patch) -> np.ndarray:
    """Node indices for a node-valued condition (Dirichlet / clamp).

    A :class:`NodeSelection` applies to its selected node set directly; a
    legacy predicate resolves to the unique nodes of its matching faces.
    """
    if isinstance(patch, NodeSelection):
        return patch.resolve(mesh)
    return _patch_nodes(mesh, patch)


def _face_patch(mesh: HexMesh, patch: Patch) -> np.ndarray:
    """Node indices spanning an area-integrated condition (traction / flux).

    A :class:`NodeSelection` spans the boundary faces whose four corners
    are all selected; the returned set is the union of those corners so a
    backend applies the load to exactly the spanned faces.
    """
    if isinstance(patch, NodeSelection):
        group = faces_from_nodes(mesh, patch.resolve(mesh))
        if group.nodes.size == 0:
            raise ValueError(
                f"Selection {patch.describe()} spans no complete boundary face; "
                "area-integrated conditions need all four corners of at least one "
                "boundary quad selected."
            )
        return np.unique(group.nodes).astype(np.int32)
    return _patch_nodes(mesh, patch)


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


def _tet_node_patch(mesh: TetMesh, patch: Patch) -> np.ndarray:
    """Node indices for a node-valued condition on a tet mesh.

    Selections (and legacy predicates) resolve to corner boundary nodes;
    on TET10 the set is completed with the midside nodes both of whose
    corner parents are selected, so the whole quadratic patch is pinned.
    """
    if isinstance(patch, NodeSelection):
        indices = patch.resolve(mesh)
    else:
        group = select_faces(mesh, patch)
        if group.nodes.size == 0:
            raise ValueError("Boundary-condition predicate selected no boundary faces.")
        indices = np.unique(group.nodes)
    return tet10_complete_nodes(mesh, indices)


def _tet_face_patch(mesh: TetMesh, patch: Patch) -> tuple[np.ndarray, np.ndarray]:
    """Node set and exact boundary triangles of an area-integrated condition.

    Returns:
        ``(nodes, faces)`` — the spanning node set (corners plus, on
            TET10, the faces' midside nodes: jax-fem selects a face for a
            surface map only when *all* its nodes are in the set) and the
            ``(M, 3)`` corner triangles used for exact face targeting.
    """
    if isinstance(patch, NodeSelection):
        faces = tet_faces_from_nodes(mesh, patch.resolve(mesh))
        if faces.shape[0] == 0:
            raise ValueError(
                f"Selection {patch.describe()} spans no complete boundary face; "
                "area-integrated conditions need every corner of at least one "
                "boundary face selected."
            )
    else:
        group = select_faces(mesh, patch)
        faces = group.nodes
        if faces.shape[0] == 0:
            raise ValueError("Boundary-condition predicate selected no boundary faces.")
    nodes = np.unique(faces)
    if mesh.edge_parents is not None:
        nodes = np.concatenate([nodes, np.unique(tet10_face_midsides(mesh, faces))])
    return nodes.astype(np.int32), np.asarray(faces)


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

    Args:
        mesh: Hex mesh from :func:`cadjoint.fem.sdf_to_hex_mesh`, or a
            :class:`~cadjoint.fem.tetmesh.TetMesh` (solved with TET4/TET10
            elements on the direct jax-fem path; same BC semantics, with
            TET10 midside completion and exact flux-face targeting).
        conductivity: Thermal conductivity ``k`` — a scalar for a
            single-material domain, or a per-element ``(C,)`` array sampled
            from the scene's material field
            (:func:`cadjoint.fem.properties.sample_cell_property`), which the
            direct backend carries as a jax-fem internal variable.
        dirichlet: ``(patch, temperature)`` pairs; each patch is a
            :class:`~cadjoint.fem.Nodes` selection (applied to its node set
            directly) or a legacy face predicate for
            :func:`cadjoint.fem.select_faces`.  With the default direct
            backend a temperature may be a traced JAX scalar: the solve is
            then differentiable w.r.t. the prescribed value (lifted
            formulation).
        neumann: ``(patch, flux)`` pairs prescribing a heat inflow per area
            (positive heats the body) on the boundary faces spanned by the
            patch.  Direct backend only for now.
        source: Volumetric heat source ``q``.
        backend: Backend name (``"jaxfem"`` default, ``"tesseract"``) or a
            :class:`~cadjoint.fem.backends.SolverBackend` instance.
        points: Optional traced override of ``mesh.points`` (same shape) for
            differentiable frozen-topology solves; BC patches are always
            resolved on the nominal ``mesh.points``.

    Returns:
        A :class:`ThermalResult`; ``temperature`` is a JAX array with an
        adjoint VJP w.r.t. ``points``, ``conductivity``, and ``source``.
    """
    if isinstance(mesh, TetMesh):
        _require_direct_backend(backend, "Thermal solves")
        flux_patches = [_tet_face_patch(mesh, patch) for patch, _ in (neumann or [])]
        tet_bcs = ThermalBCs(
            dirichlet_nodes=[_tet_node_patch(mesh, patch) for patch, _ in dirichlet],
            dirichlet_values=[
                float(value) if isinstance(value, (int, float)) else value for _, value in dirichlet
            ],
            flux_nodes=[nodes for nodes, _ in flux_patches],
            flux_values=[float(value) for _, value in (neumann or [])],
        )
        temperature = tet_thermal_solve(
            mesh.points if points is None else points,
            mesh.cells,
            tet_bcs,
            conductivity=_property_value(conductivity),
            source=float(source),
            ele_type=mesh.ele_type,
            base_points=mesh.points,
            flux_faces=[faces for _, faces in flux_patches] if flux_patches else None,
        )
        return ThermalResult(temperature=temperature, mesh=mesh)
    bcs = ThermalBCs(
        dirichlet_nodes=[_node_patch(mesh, patch) for patch, _ in dirichlet],
        dirichlet_values=[
            float(value) if isinstance(value, (int, float)) else value for _, value in dirichlet
        ],
        flux_nodes=[_face_patch(mesh, patch) for patch, _ in (neumann or [])],
        flux_values=[float(value) for _, value in (neumann or [])],
    )
    solver = get_backend(backend)
    solve_points = mesh.points if points is None else points
    temperature = solver.thermal(
        solve_points,
        mesh.cells,
        bcs,
        conductivity=_property_value(conductivity),
        source=float(source),
        base_points=mesh.points,
    )
    return ThermalResult(temperature=temperature, mesh=mesh)


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
        mesh: Hex mesh from :func:`cadjoint.fem.sdf_to_hex_mesh`, or a
            :class:`~cadjoint.fem.tetmesh.TetMesh` (solved with TET4/TET10
            elements on the direct jax-fem path; same BC semantics, with
            TET10 midside completion and exact traction-face targeting).
        youngs: Young's modulus — a scalar, or a per-element ``(C,)``
            array sampled from the scene's material field.
        poisson: Poisson ratio, scalar or per element like ``youngs``.
        dirichlet: Patches picking fully-clamped node sets (all displacement
            components fixed to zero) — :class:`~cadjoint.fem.Nodes`
            selections applied directly, or legacy face predicates.
        tractions: ``(patch, vector)`` pairs applying a constant traction
            (force per area) on the boundary faces spanned by the patch.
        backend: Backend name or instance (see :func:`thermal_solve`).
        points: Optional traced override of ``mesh.points`` (same shape) for
            differentiable frozen-topology solves.
        body_force: Optional body force density in N/m^3, ``(3,)`` or
            ``(C, 3)`` — ``density * gravity`` for self-weight.  Direct
            backend only.

    Returns:
        An :class:`ElasticResult`; ``displacement`` is a JAX array with an
        adjoint VJP w.r.t. ``points``.
    """
    if isinstance(mesh, TetMesh):
        _require_direct_backend(backend, "Elastic solves")
        traction_patches = [_tet_face_patch(mesh, patch) for patch, _ in tractions]
        tet_bcs = ElasticBCs(
            fixed_nodes=[_tet_node_patch(mesh, patch) for patch in dirichlet],
            traction_nodes=[nodes for nodes, _ in traction_patches],
            traction_vectors=[np.asarray(vector, dtype=np.float64) for _, vector in tractions],
        )
        displacement = tet_elastic_solve(
            mesh.points if points is None else points,
            mesh.cells,
            tet_bcs,
            youngs=_property_value(youngs),
            poisson=_property_value(poisson),
            ele_type=mesh.ele_type,
            base_points=mesh.points,
            traction_faces=[faces for _, faces in traction_patches] if traction_patches else None,
            body_force=body_force,
        )
        return ElasticResult(
            displacement=displacement,
            mesh=mesh,
            youngs=_property_value(youngs),
            poisson=_property_value(poisson),
        )
    bcs = ElasticBCs(
        fixed_nodes=[_node_patch(mesh, patch) for patch in dirichlet],
        traction_nodes=[_face_patch(mesh, patch) for patch, _ in tractions],
        traction_vectors=[np.asarray(vector, dtype=np.float64) for _, vector in tractions],
    )
    solver = get_backend(backend)
    solve_points = mesh.points if points is None else points
    elastic_kwargs: dict[str, Any] = {}
    if body_force is not None:
        elastic_kwargs["body_force"] = body_force
    displacement = solver.elastic(
        solve_points,
        mesh.cells,
        bcs,
        youngs=_property_value(youngs),
        poisson=_property_value(poisson),
        base_points=mesh.points,
        **elastic_kwargs,
    )
    return ElasticResult(
        displacement=displacement,
        mesh=mesh,
        youngs=_property_value(youngs),
        poisson=_property_value(poisson),
    )
