"""Thermal and structural simulation on SDF-extracted hex meshes.

Public entry points :func:`thermal_solve` and :func:`elastic_solve` resolve
boundary patches — :class:`~cadjoint.fem.selection.NodeSelection` values or
legacy face predicates — against the mesh, hand array-level BCs to a
pluggable solver backend (:mod:`cadjoint.fem.backends`; direct in-process
jax-fem by default), and return small result objects with VTK export for
ParaView.

Patch semantics: a ``NodeSelection`` used for a node-valued condition
(prescribed temperature, clamp) applies to its selected node set directly;
used for an area-integrated condition (traction, heat flux) it spans the
boundary faces all four of whose corners are selected
(:func:`~cadjoint.fem.hexmesh.faces_from_nodes`).  A callable patch is the
legacy face-predicate form resolved via :func:`~cadjoint.fem.select_faces`.

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
from cadjoint.fem.hexmesh import HexMesh, faces_from_nodes, select_faces
from cadjoint.fem.selection import NodeSelection

__all__ = ["ElasticResult", "ThermalResult", "elastic_solve", "thermal_solve"]

Predicate = Callable[..., Any]
#: A boundary patch: a node selection or a legacy face predicate.
Patch = NodeSelection | Predicate

# Trilinear corner signs of the VTK hex in reference coordinates [-1, 1]^3;
# dN_i/dxi at the element center is _CORNER_SIGNS[i] / 8.
_CORNER_SIGNS = np.array(
    [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ],
    dtype=np.float64,
)


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


def _export_vtk(path: str, mesh: HexMesh, point_data: dict, cell_data: dict) -> None:
    """Write the mesh and fields as VTK via meshio (guarded)."""
    try:
        import meshio
    except ImportError as error:
        raise ImportError(
            "VTK export requires meshio (installed with the 'fem' extra: pip install cadjoint[fem])."
        ) from error
    meshio.Mesh(
        points=np.asarray(mesh.points, dtype=np.float64),
        cells=[("hexahedron", np.asarray(mesh.cells, dtype=np.int64))],
        point_data={k: np.asarray(v) for k, v in point_data.items()},
        cell_data={k: [np.asarray(v)] for k, v in cell_data.items()},
    ).write(path)


@dataclass(frozen=True)
class ThermalResult:
    """Steady-state thermal solution.

    Attributes:
        temperature: Per-node temperature, shaped ``(N,)``.
        mesh: The mesh that was solved on.
    """

    temperature: Any
    mesh: HexMesh

    def vtk_export(self, path: str) -> None:
        """Write mesh + temperature as a VTK file for ParaView."""
        _export_vtk(path, self.mesh, {"temperature": self.temperature}, {})


@dataclass(frozen=True)
class ElasticResult:
    """Small-strain linear elastic solution.

    Attributes:
        displacement: Per-node displacement, shaped ``(N, 3)``.
        mesh: The mesh that was solved on.
        youngs: Young's modulus used.
        poisson: Poisson ratio used.
    """

    displacement: Any
    mesh: HexMesh
    youngs: float
    poisson: float

    def von_mises(self) -> np.ndarray:
        """Per-cell von Mises stress evaluated at each hex center.

        The displacement gradient is taken from the trilinear (HEX8) basis
        at the element center.

        Returns:
            Von Mises stress per cell, shaped ``(C,)``.
        """
        points = np.asarray(self.mesh.points, dtype=np.float64)
        displacement = np.asarray(self.displacement, dtype=np.float64)
        cells = np.asarray(self.mesh.cells)
        grad_ref = _CORNER_SIGNS / 8.0  # (8, 3): dN/dxi at the center
        corner_positions = points[cells]  # (C, 8, 3)
        corner_disp = displacement[cells]  # (C, 8, 3)
        jacobian = np.einsum("cia,ib->cab", corner_positions, grad_ref)  # dx/dxi
        grad_phys = np.einsum("ib,cab->cia", grad_ref, np.linalg.inv(jacobian).transpose(0, 2, 1))
        u_grad = np.einsum("cia,cib->cab", corner_disp, grad_phys)  # du_a/dx_b
        strain = 0.5 * (u_grad + u_grad.transpose(0, 2, 1))
        lame_lambda = self.youngs * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson))
        lame_mu = self.youngs / (2 * (1 + self.poisson))
        trace = np.trace(strain, axis1=1, axis2=2)
        stress = lame_lambda * trace[:, None, None] * np.eye(3) + 2.0 * lame_mu * strain
        deviator = stress - np.trace(stress, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
        return np.sqrt(1.5 * np.einsum("cab,cab->c", deviator, deviator))

    def vtk_export(self, path: str) -> None:
        """Write mesh + displacement + von Mises stress as VTK for ParaView."""
        _export_vtk(
            path,
            self.mesh,
            {"displacement": self.displacement},
            {"von_mises": self.von_mises()},
        )


def thermal_solve(
    mesh: HexMesh,
    *,
    conductivity: float,
    dirichlet: list[tuple[Patch, float]],
    neumann: list[tuple[Patch, float]] | None = None,
    source: float = 0.0,
    backend: str | SolverBackend | None = None,
    points: Any = None,
) -> ThermalResult:
    """Solve steady-state heat conduction ``-div(k grad T) = q`` on the mesh.

    Args:
        mesh: Hex mesh from :func:`cadjoint.fem.sdf_to_hex_mesh`.
        conductivity: Thermal conductivity ``k`` (constant).
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
        conductivity=float(conductivity),
        source=float(source),
        base_points=mesh.points,
    )
    return ThermalResult(temperature=temperature, mesh=mesh)


def elastic_solve(
    mesh: HexMesh,
    *,
    youngs: float,
    poisson: float,
    dirichlet: list[Patch],
    tractions: list[tuple[Patch, Any]],
    backend: str | SolverBackend | None = None,
    points: Any = None,
) -> ElasticResult:
    """Solve small-strain linear elasticity on the mesh.

    Args:
        mesh: Hex mesh from :func:`cadjoint.fem.sdf_to_hex_mesh`.
        youngs: Young's modulus.
        poisson: Poisson ratio.
        dirichlet: Patches picking fully-clamped node sets (all displacement
            components fixed to zero) — :class:`~cadjoint.fem.Nodes`
            selections applied directly, or legacy face predicates.
        tractions: ``(patch, vector)`` pairs applying a constant traction
            (force per area) on the boundary faces spanned by the patch.
        backend: Backend name or instance (see :func:`thermal_solve`).
        points: Optional traced override of ``mesh.points`` (same shape) for
            differentiable frozen-topology solves.

    Returns:
        An :class:`ElasticResult`; ``displacement`` is a JAX array with an
        adjoint VJP w.r.t. ``points``.
    """
    bcs = ElasticBCs(
        fixed_nodes=[_node_patch(mesh, patch) for patch in dirichlet],
        traction_nodes=[_face_patch(mesh, patch) for patch, _ in tractions],
        traction_vectors=[np.asarray(vector, dtype=np.float64) for _, vector in tractions],
    )
    solver = get_backend(backend)
    solve_points = mesh.points if points is None else points
    displacement = solver.elastic(
        solve_points,
        mesh.cells,
        bcs,
        youngs=float(youngs),
        poisson=float(poisson),
        base_points=mesh.points,
    )
    return ElasticResult(
        displacement=displacement, mesh=mesh, youngs=float(youngs), poisson=float(poisson)
    )
