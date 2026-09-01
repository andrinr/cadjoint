"""Direct in-process jax-fem solving for every element family.

What belongs here: the actual finite-element problems — the weak forms,
the Dirichlet lift, the traction/flux surface maps, the linear solvers and
jax-fem's adjoint wiring — for HEX8 (:class:`JaxFemBackend`) and for
TET4/TET10 (:func:`tet_thermal_solve`, :func:`tet_elastic_solve`).  Both
families sit in this one module so the thermal and elastic formulations
stay demonstrably the same across element types; the tet entry points are
plain functions rather than a second backend class because tet meshes
deliberately bypass the backend registry (they always solve on the direct
jax-fem path, see :func:`cadjoint.fem.simulate._require_direct_backend`).

What does *not* belong here: the array-level ABI, the backend protocol and
the registry (:mod:`cadjoint.fem.backends`), patch resolution
(:mod:`cadjoint.fem.simulate`), meshing (:mod:`cadjoint.fem.hexmesh` /
:mod:`cadjoint.fem.tetmesh`) or stress recovery
(:mod:`cadjoint.fem.postprocess`).  Nothing here reads a mesh object:
solves take ``(points, cells, bcs)`` arrays, so this module imports no mesh
module and the mesh layer stays free of solver dependencies.

Differentiability: ``points`` may be traced; returned fields carry an
adjoint VJP through jax-fem's ``ad_wrapper``.  The thermal solves are
additionally differentiable w.r.t. the prescribed Dirichlet values via the
lifted formulation.  Forward solves assemble through PETSc, so no call here
may sit under ``jax.jit``.

Precision: every entry point runs inside :func:`~cadjoint.fem.backends._x64_scope`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadjoint.fem.backends import (
    ElasticBCs,
    ThermalBCs,
    _membership_location,
    _require_jax_fem,
    _x64_scope,
)

__all__ = [
    "JaxFemBackend",
    "tet_elastic_solve",
    "tet_thermal_solve",
]


class JaxFemBackend:
    """Direct in-process jax-fem backend for HEX8 meshes (the default).

    Gradients flow through jax-fem's adjoint (``ad_wrapper``): the forward
    solve runs concretely (PETSc-assembled Newton), the VJP solves the
    adjoint system and back-propagates through the residual — including
    through the nodal coordinates via
    ``Problem.initialize_geometric_quantities``.
    """

    name = "jaxfem"

    def thermal(self, points, cells, bcs, *, conductivity, source, base_points=None):
        """See :meth:`~cadjoint.fem.backends.SolverBackend.thermal`.

        Dirichlet values are differentiable: jax-fem bakes prescribed values
        into the DOF elimination at problem construction (outside the
        adjoint's parameter path), so the solve is lifted — ``T = u0 + g``
        with ``g`` the nodal field interpolating the prescribed boundary
        values and ``u0`` solved under *homogeneous* Dirichlet conditions
        with the extra flux ``k grad(g)`` in the weak form.  ``g`` enters
        through ``set_params`` (as an internal variable via its quad-point
        gradient), so ``d(objective)/d(dirichlet value)`` flows through the
        adjoint.

        Heat-flux (Neumann) patches enter the weak form as surface
        integrals ``-integral(v * q)`` over the faces spanned by each
        ``flux_nodes`` set, so ``k grad(T) . n = q`` on the patch (``q``
        positive heats the body).  The lift composes unchanged: the surface
        term does not involve ``g``.
        """
        _require_jax_fem()
        with _x64_scope():
            import jax.numpy as jnp
            from jax_fem.generate_mesh import Mesh
            from jax_fem.problem import Problem
            from jax_fem.solver import ad_wrapper

            flux_values = [float(value) for value in bcs.flux_values]

            class _Thermal(Problem):
                def get_tensor_map(self):
                    def tensor_map(u_grad, kappa, _source, lift_grad):
                        # u_grad: (vec=1, dim); lift_grad: (dim,) broadcasts in.
                        return kappa * (u_grad + lift_grad)

                    return tensor_map

                def get_mass_map(self):
                    def mass_map(u, _x, _kappa, source_value, _lift_grad):
                        # Weak form: residual += integral(v * mass_map); the
                        # source q enters as -q so that -div(k grad T) = q.
                        return -source_value * jnp.ones_like(u)

                    return mass_map

                def get_surface_maps(self):
                    # Weak form: residual += integral(v * surface_map); a
                    # prescribed inflow q enters as -q so k grad(T).n = q.
                    return [
                        (lambda u, _x, value=value: -value * jnp.ones_like(u))
                        for value in flux_values
                    ]

                def set_params(self, params):
                    params_points, kappa, source_value, lift_nodal = params
                    self.initialize_geometric_quantities([params_points])
                    fe = self.fes[0]
                    shape = (fe.num_cells, fe.num_quads)
                    # grad(g) at the quad points from the (possibly traced)
                    # nodal lift: (C, 8) x (C, Q, 8, dim) -> (C, Q, dim).
                    lift_grad = jnp.einsum("cn,cqnd->cqd", lift_nodal[fe.cells], fe.shape_grads)
                    self.internal_vars = [
                        kappa * jnp.ones(shape),
                        source_value * jnp.ones(shape),
                        lift_grad,
                    ]

            if base_points is None:
                base_points = points
            base_points = np.asarray(base_points, dtype=np.float64)
            mesh = Mesh(base_points, np.asarray(cells), ele_type="HEX8")
            dirichlet = [
                [_membership_location(nodes) for nodes in bcs.dirichlet_nodes],
                [0] * len(bcs.dirichlet_nodes),
                [(lambda _point: 0.0) for _ in bcs.dirichlet_nodes],
            ]
            problem = _Thermal(
                mesh=mesh,
                vec=1,
                dim=3,
                ele_type="HEX8",
                dirichlet_bc_info=dirichlet,
                location_fns=[_membership_location(nodes) for nodes in bcs.flux_nodes],
            )
            forward = ad_wrapper(problem)

            lift = jnp.zeros(base_points.shape[0], dtype=jnp.float64)
            for nodes, value in zip(bcs.dirichlet_nodes, bcs.dirichlet_values):
                lift = lift.at[jnp.asarray(np.asarray(nodes, dtype=np.int32))].set(value)
            solution = forward((jnp.asarray(points), conductivity, source, lift))
            return solution[0][:, 0] + lift

    def elastic(self, points, cells, bcs, *, youngs, poisson, base_points=None):
        """See :meth:`~cadjoint.fem.backends.SolverBackend.elastic`."""
        _require_jax_fem()
        with _x64_scope():
            import jax.numpy as jnp
            from jax_fem.generate_mesh import Mesh
            from jax_fem.problem import Problem
            from jax_fem.solver import ad_wrapper

            lame_lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
            lame_mu = youngs / (2.0 * (1.0 + poisson))
            tractions = [np.asarray(vector, dtype=np.float64) for vector in bcs.traction_vectors]

            class _Elastic(Problem):
                def get_tensor_map(self):
                    def stress(u_grad):
                        strain = 0.5 * (u_grad + u_grad.T)
                        return lame_lambda * jnp.trace(strain) * jnp.eye(3) + 2.0 * lame_mu * strain

                    return stress

                def get_surface_maps(self):
                    # Weak form: residual += integral(v * surface_map); a
                    # traction t enters as -t so that sigma.n = t on the patch.
                    return [
                        (lambda _u, _x, vector=vector: -jnp.asarray(vector)) for vector in tractions
                    ]

                def set_params(self, params):
                    self.initialize_geometric_quantities([params])

            if base_points is None:
                base_points = points
            mesh = Mesh(
                np.asarray(base_points, dtype=np.float64), np.asarray(cells), ele_type="HEX8"
            )
            fixed_locations = [_membership_location(nodes) for nodes in bcs.fixed_nodes]
            dirichlet = [
                [location for location in fixed_locations for _ in range(3)],
                [component for _ in fixed_locations for component in range(3)],
                [(lambda _point: 0.0) for _ in fixed_locations for _ in range(3)],
            ]
            problem = _Elastic(
                mesh=mesh,
                vec=3,
                dim=3,
                ele_type="HEX8",
                dirichlet_bc_info=dirichlet,
                location_fns=[_membership_location(nodes) for nodes in bcs.traction_nodes],
            )
            forward = ad_wrapper(problem)
            return forward(jnp.asarray(points))[0]


def _tet_direct_linear_solver(A: Any, b: Any, _x0: Any, _options: dict) -> Any:
    """Layered robust direct solve for sliver-tet stiffness systems.

    Preserving the DC surface verbatim leaves sliver tets whose
    conditioning defeats every single off-the-shelf solver somewhere
    (all observed, see research/tet-vs-hex.md): jax-fem's default
    BiCGStab diverges outright; PETSc LU hits (near-)zero pivots on some
    elastic sliver meshes even with a nonzero factor shift; SuperLU
    survives those but its COLAMD ordering blew up (hours of fill) on one
    thermal mesh.  So: try PETSc LU with a nonzero pivot shift first
    (fast, fill-safe nested-dissection ordering), verify the residual,
    and fall back to SuperLU orderings when the factorization was bad.

    Signature per jax-fem's ``custom_solver`` contract: ``(A, b, x0,
    linear_options) -> x`` with ``A`` a PETSc AIJ matrix.
    """
    import scipy.sparse
    import scipy.sparse.linalg
    from petsc4py import PETSc

    rhs = np.asarray(b, dtype=np.float64)
    scale = max(float(np.linalg.norm(rhs)), 1e-30)

    def residual(x: np.ndarray) -> float:
        y = PETSc.Vec().createSeq(len(rhs))
        vec = PETSc.Vec().createSeq(len(rhs))
        vec.setValues(range(len(rhs)), x)
        A.mult(vec, y)
        return float(np.linalg.norm(y.getArray() - rhs)) / scale

    petsc_rhs = PETSc.Vec().createSeq(len(rhs))
    petsc_rhs.setValues(range(len(rhs)), rhs)
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.pc.setType("lu")
    ksp.pc.setFactorShift(PETSc.Mat.FactorShiftType.NONZERO, 1e-12)
    solution = PETSc.Vec().createSeq(len(rhs))
    ksp.solve(petsc_rhs, solution)
    x = np.array(solution.getArray())
    if np.isfinite(x).all() and residual(x) < 1e-8:
        return x

    indptr, indices, data = A.getValuesCSR()
    matrix = scipy.sparse.csr_matrix((data, indices, indptr))
    best = x
    best_residual = residual(x) if np.isfinite(x).all() else np.inf
    for ordering in ("COLAMD", "MMD_AT_PLUS_A"):
        candidate = scipy.sparse.linalg.spsolve(matrix, rhs, permc_spec=ordering)
        if not np.isfinite(candidate).all():
            continue
        candidate_residual = residual(candidate)
        if candidate_residual < best_residual:
            best, best_residual = candidate, candidate_residual
        if candidate_residual < 1e-8:
            return candidate
    if best_residual < 1e-4:
        return best
    raise RuntimeError(
        f"Direct linear solve failed on the tet system (best relative residual "
        f"{best_residual:.2e}); the mesh likely contains degenerate sliver tets — "
        "re-extract at a different resolution."
    )


# Both the forward Newton steps and the adjoint run through the layered
# direct solver above.
_TET_SOLVER_OPTIONS = {"custom_solver": _tet_direct_linear_solver}


def _rows_in(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Boolean mask of which ``rows`` (2-D int64) appear as rows of ``table``."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    table = np.ascontiguousarray(table, dtype=np.int64)
    void = np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    return np.isin(rows.view(void).reshape(-1), table.view(void).reshape(-1))


def _restrict_surface_faces(problem: Any, surface_faces: list[np.ndarray]) -> None:
    """Prune jax-fem's face selection to exactly the given boundary triangles.

    jax-fem selects a (cell, local face) pair for a surface map whenever
    *all* the face's nodes satisfy the location function.  With node-set
    membership locations on a tet mesh this over-selects: an interior
    face whose three corners all happen to lie on the loaded surface
    patch is selected once per adjacent cell, double-loading a face that
    is not even on the boundary (observed on the bracket web at fine
    resolutions).  This helper prunes each patch's selection (traction or
    heat-flux alike) to the faces whose corner triple matches the
    requested boundary triangles, and rebuilds the dependent structures
    (``cells_list_face_list`` and the face blocks of the assembly
    sparsity pattern ``I``/``J``) so value and index arrays stay aligned.
    Surface quadrature data is recomputed from the pruned selection by
    ``set_params`` before every solve.
    """
    finite_element = problem.fes[0]
    face_inds = np.asarray(finite_element.face_inds)
    # Local corner slots per face: for TET4 all three face nodes are
    # corners; for TET10 the corners are the local indices below 4.
    corner_slots = np.stack([np.sort(local[local < 4])[:3] for local in face_inds])
    cells0 = np.asarray(finite_element.cells)

    def flat_dof_ids(cells_arrays: list[np.ndarray]) -> np.ndarray:
        parts = []
        for i, cells_arr in enumerate(cells_arrays):
            vec = problem.fes[i].vec
            ids = (
                vec * np.asarray(cells_arr)[:, :, None]
                + np.arange(vec)[None, None, :]
                + problem.offset[i]
            )
            parts.append(ids.reshape(len(cells_arr), -1))
        return np.concatenate(parts, axis=1)

    inds = flat_dof_ids(problem.cells_list)
    pattern_i = np.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
    pattern_j = np.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)
    new_cells_face_list = []
    for patch, target in enumerate(surface_faces):
        binds = np.asarray(problem.boundary_inds_list[patch])
        slots = corner_slots[binds[:, 1]]
        corner_ids = np.take_along_axis(cells0[binds[:, 0]], slots, axis=1)
        keys = np.sort(corner_ids, axis=1)
        target_keys = np.sort(np.asarray(target, dtype=np.int64)[:, :3], axis=1)
        mask = _rows_in(keys, target_keys)
        if int(mask.sum()) != target_keys.shape[0]:
            raise ValueError(
                f"Surface patch {patch}: matched {int(mask.sum())} of "
                f"{target_keys.shape[0]} requested boundary faces; the patch node "
                "set must contain every corner of every requested face."
            )
        pruned = binds[mask]
        problem.boundary_inds_list[patch] = pruned
        cells_face = [np.asarray(c)[pruned[:, 0]] for c in problem.cells_list]
        new_cells_face_list.append(cells_face)
        inds_face = flat_dof_ids(cells_face)
        pattern_i = np.hstack(
            [pattern_i, np.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)]
        )
        pattern_j = np.hstack(
            [pattern_j, np.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)]
        )
    problem.cells_list_face_list = new_cells_face_list
    problem.I = pattern_i
    problem.J = pattern_j


def tet_elastic_solve(
    points: Any,
    cells: np.ndarray,
    bcs: ElasticBCs,
    *,
    youngs: float,
    poisson: float,
    ele_type: str = "TET4",
    base_points: np.ndarray | None = None,
    traction_faces: list[np.ndarray] | None = None,
) -> Any:
    """Small-strain linear elasticity on a tet mesh via jax-fem.

    Mirrors :meth:`JaxFemBackend.elastic` with the element type opened up:
    ``"TET4"`` or ``"TET10"`` (both confirmed in jax-fem's element tables;
    connectivity must be meshio ``tetra`` / ``tetra10`` order, as produced
    by TetGen resp. :func:`~cadjoint.fem.tetmesh.tet10_from_tet4`).
    ``points`` may be traced; the displacement participates in the
    surrounding autodiff graph via jax-fem's adjoint.

    Args:
        points: Node positions, ``(N, 3)`` (traced allowed).
        cells: Connectivity, ``(T, 4)`` or ``(T, 10)``.
        bcs: Array-level boundary conditions (the backend ABI).  For
            ``TET10``, node sets must include midside nodes (a face
            carries a traction when *all* its nodes are in the set).
        youngs: Young's modulus.
        poisson: Poisson ratio.
        ele_type: ``"TET4"`` or ``"TET10"``.
        base_points: Concrete positions for problem construction when
            ``points`` is traced (defaults to ``points``).
        traction_faces: Optional exact face targeting: one ``(M, >=3)``
            array of *corner* node triples per traction patch (boundary
            triangles).  When given, jax-fem's node-membership face
            selection is pruned to exactly these faces — closing the
            interior-face double-count hole of pure node membership (see
            :func:`_restrict_surface_faces`).  Every corner must also be
            in the corresponding ``bcs.traction_nodes`` set.

    Returns:
        Per-node displacement, ``(N, 3)`` JAX array.
    """
    if ele_type not in ("TET4", "TET10"):
        raise ValueError(f"ele_type must be 'TET4' or 'TET10', got {ele_type!r}.")
    _require_jax_fem()
    with _x64_scope():
        import jax.numpy as jnp
        from jax_fem.generate_mesh import Mesh
        from jax_fem.problem import Problem
        from jax_fem.solver import ad_wrapper

        lame_lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        lame_mu = youngs / (2.0 * (1.0 + poisson))
        tractions = [np.asarray(vector, dtype=np.float64) for vector in bcs.traction_vectors]

        class _Elastic(Problem):
            def get_tensor_map(self):
                def stress(u_grad):
                    strain = 0.5 * (u_grad + u_grad.T)
                    return lame_lambda * jnp.trace(strain) * jnp.eye(3) + 2.0 * lame_mu * strain

                return stress

            def get_surface_maps(self):
                return [
                    (lambda _u, _x, vector=vector: -jnp.asarray(vector)) for vector in tractions
                ]

            def set_params(self, params):
                self.initialize_geometric_quantities([params])

        if base_points is None:
            base_points = points
        mesh = Mesh(np.asarray(base_points, dtype=np.float64), np.asarray(cells), ele_type=ele_type)
        fixed_locations = [_membership_location(nodes) for nodes in bcs.fixed_nodes]
        dirichlet = [
            [location for location in fixed_locations for _ in range(3)],
            [component for _ in fixed_locations for component in range(3)],
            [(lambda _point: 0.0) for _ in fixed_locations for _ in range(3)],
        ]
        problem = _Elastic(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet,
            location_fns=[_membership_location(nodes) for nodes in bcs.traction_nodes],
        )
        if traction_faces is not None:
            if len(traction_faces) != len(bcs.traction_nodes):
                raise ValueError(
                    "traction_faces must provide one face array per traction patch "
                    f"({len(traction_faces)} given for {len(bcs.traction_nodes)} patches)."
                )
            _restrict_surface_faces(problem, traction_faces)
        forward = ad_wrapper(
            problem,
            solver_options=dict(_TET_SOLVER_OPTIONS),
            adjoint_solver_options=dict(_TET_SOLVER_OPTIONS),
        )
        return forward(jnp.asarray(points))[0]


def tet_thermal_solve(
    points: Any,
    cells: np.ndarray,
    bcs: ThermalBCs,
    *,
    conductivity: float,
    source: float = 0.0,
    ele_type: str = "TET4",
    base_points: np.ndarray | None = None,
    flux_faces: list[np.ndarray] | None = None,
) -> Any:
    """Steady-state heat conduction on a tet mesh via jax-fem.

    Mirrors :meth:`JaxFemBackend.thermal` — the same lifted Dirichlet
    formulation (prescribed values enter through a nodal lift field ``g``,
    so the solve is differentiable w.r.t. traced Dirichlet values) and the
    same Neumann surface integral for heat-flux patches — with the element
    type opened up to ``"TET4"``/``"TET10"``.  ``points`` may be traced;
    the temperature participates in the surrounding autodiff graph via
    jax-fem's adjoint.

    Args:
        points: Node positions, ``(N, 3)`` (traced allowed).
        cells: Connectivity, ``(T, 4)`` or ``(T, 10)``.
        bcs: Array-level thermal boundary conditions (the backend ABI).
            For ``TET10``, node sets must include midside nodes
            (:func:`~cadjoint.fem.boundary.tet10_complete_nodes`).
        conductivity: Thermal conductivity ``k`` (may be traced).
        source: Volumetric heat source ``q`` (may be traced).
        ele_type: ``"TET4"`` or ``"TET10"``.
        base_points: Concrete positions for problem construction when
            ``points`` is traced (defaults to ``points``).
        flux_faces: Optional exact face targeting: one ``(M, >=3)`` array
            of *corner* node triples per heat-flux patch, pruning jax-fem's
            node-membership face selection to exactly these boundary
            triangles (see :func:`_restrict_surface_faces`).

    Returns:
        Per-node temperature, ``(N,)`` JAX array.
    """
    if ele_type not in ("TET4", "TET10"):
        raise ValueError(f"ele_type must be 'TET4' or 'TET10', got {ele_type!r}.")
    _require_jax_fem()
    with _x64_scope():
        import jax.numpy as jnp
        from jax_fem.generate_mesh import Mesh
        from jax_fem.problem import Problem
        from jax_fem.solver import ad_wrapper

        flux_values = [float(value) for value in bcs.flux_values]

        class _Thermal(Problem):
            def get_tensor_map(self):
                def tensor_map(u_grad, kappa, _source, lift_grad):
                    # u_grad: (vec=1, dim); lift_grad: (dim,) broadcasts in.
                    return kappa * (u_grad + lift_grad)

                return tensor_map

            def get_mass_map(self):
                def mass_map(u, _x, _kappa, source_value, _lift_grad):
                    # Weak form: residual += integral(v * mass_map); the
                    # source q enters as -q so that -div(k grad T) = q.
                    return -source_value * jnp.ones_like(u)

                return mass_map

            def get_surface_maps(self):
                # Weak form: residual += integral(v * surface_map); a
                # prescribed inflow q enters as -q so k grad(T).n = q.
                return [
                    (lambda u, _x, value=value: -value * jnp.ones_like(u)) for value in flux_values
                ]

            def set_params(self, params):
                params_points, kappa, source_value, lift_nodal = params
                self.initialize_geometric_quantities([params_points])
                fe = self.fes[0]
                shape = (fe.num_cells, fe.num_quads)
                # grad(g) at the quad points from the (possibly traced)
                # nodal lift: (C, n) x (C, Q, n, dim) -> (C, Q, dim).
                lift_grad = jnp.einsum("cn,cqnd->cqd", lift_nodal[fe.cells], fe.shape_grads)
                self.internal_vars = [
                    kappa * jnp.ones(shape),
                    source_value * jnp.ones(shape),
                    lift_grad,
                ]

        if base_points is None:
            base_points = points
        base_points = np.asarray(base_points, dtype=np.float64)
        mesh = Mesh(base_points, np.asarray(cells), ele_type=ele_type)
        dirichlet = [
            [_membership_location(nodes) for nodes in bcs.dirichlet_nodes],
            [0] * len(bcs.dirichlet_nodes),
            [(lambda _point: 0.0) for _ in bcs.dirichlet_nodes],
        ]
        problem = _Thermal(
            mesh=mesh,
            vec=1,
            dim=3,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet,
            location_fns=[_membership_location(nodes) for nodes in bcs.flux_nodes],
        )
        if flux_faces is not None:
            if len(flux_faces) != len(bcs.flux_nodes):
                raise ValueError(
                    "flux_faces must provide one face array per flux patch "
                    f"({len(flux_faces)} given for {len(bcs.flux_nodes)} patches)."
                )
            _restrict_surface_faces(problem, flux_faces)
        forward = ad_wrapper(
            problem,
            solver_options=dict(_TET_SOLVER_OPTIONS),
            adjoint_solver_options=dict(_TET_SOLVER_OPTIONS),
        )

        lift = jnp.zeros(base_points.shape[0], dtype=jnp.float64)
        for nodes, value in zip(bcs.dirichlet_nodes, bcs.dirichlet_values):
            lift = lift.at[jnp.asarray(np.asarray(nodes, dtype=np.int32))].set(value)
        solution = forward((jnp.asarray(points), conductivity, source, lift))
        return solution[0][:, 0] + lift
