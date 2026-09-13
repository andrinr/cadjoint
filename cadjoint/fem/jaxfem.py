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
from cadjoint.fem.rungs import PaddedTets, pad_cell_field, pad_indices, pad_tets, rung

__all__ = [
    "JaxFemBackend",
    "tet_elastic_solve",
    "tet_thermal_solve",
]


def _cellwise(value: Any, shape: tuple[int, int]) -> Any:
    """Broadcast a scalar or per-cell ``(C,)`` value to ``(num_cells, num_quads)``.

    The one place a heterogeneous material property becomes a jax-fem
    ``internal_vars`` entry.  A scalar takes the identical code path it always
    did (``value * ones(shape)``), so single-material solves are unchanged;
    a per-cell array is broadcast along the quadrature axis, making the
    property piecewise constant per element.
    """
    import jax.numpy as jnp

    array = jnp.asarray(value)
    if array.ndim == 0:
        return array * jnp.ones(shape)
    if array.shape != (shape[0],):
        raise ValueError(
            f"Per-element property must be scalar or shaped ({shape[0]},); got {array.shape}."
        )
    return jnp.broadcast_to(array[:, None], shape)


def _cellwise_vector(value: Any, shape: tuple[int, int]) -> Any:
    """Broadcast a ``(3,)`` or ``(C, 3)`` vector to ``(num_cells, num_quads, 3)``."""
    import jax.numpy as jnp

    array = jnp.asarray(value)
    if array.shape == (3,):
        return jnp.broadcast_to(array, (*shape, 3))
    if array.shape != (shape[0], 3):
        raise ValueError(f"Body force must be shaped (3,) or ({shape[0]}, 3); got {array.shape}.")
    return jnp.broadcast_to(array[:, None, :], (*shape, 3))


def _lame(youngs: Any, poisson: Any) -> tuple[Any, Any]:
    """Lame constants ``(lambda, mu)`` from ``(E, nu)``; elementwise on arrays."""
    import jax.numpy as jnp

    youngs = jnp.asarray(youngs)
    poisson = jnp.asarray(poisson)
    lame_lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    lame_mu = youngs / (2.0 * (1.0 + poisson))
    return lame_lambda, lame_mu


def _elastic_problem(
    tractions: list[np.ndarray],
    *,
    youngs: Any,
    poisson: Any,
    body_force: Any = None,
) -> tuple[Any, Any]:
    """Build the elasticity ``Problem`` subclass and its ``ad_wrapper`` inputs.

    Two shapes of problem come out of here, and which one you get is decided
    purely by the arguments:

    * **Homogeneous** — scalar ``youngs``/``poisson``, no body force.  The Lame
      constants are baked into the tensor map's closure and the only parameter
      is ``points``, exactly as before per-element properties existed.  This is
      the path every single-material solve takes, byte for byte.
    * **Heterogeneous** — per-element ``(C,)`` moduli and/or a body force.  The
      constants move into ``internal_vars`` (so they are also differentiable
      through jax-fem's adjoint), and a body force ``b`` adds the mass-map term
      ``residual += integral(v * (-b))``, i.e. ``div(sigma) + b = 0``.

    Args:
        tractions: One constant traction vector per surface patch.
        youngs: Young's modulus — scalar, or per element ``(C,)``.
        poisson: Poisson ratio — scalar, or per element ``(C,)``.
        body_force: Optional body force density in N/m^3, ``(3,)`` or ``(C, 3)``
            (self-weight is ``density * gravity``).

    Returns:
        ``(problem_class, make_params)`` where ``make_params(points)`` builds
        the argument ``ad_wrapper``'s forward is called with.
    """
    import jax.numpy as jnp
    from jax_fem.problem import Problem

    lame_lambda, lame_mu = _lame(youngs, poisson)
    has_force = body_force is not None
    heterogeneous = has_force or lame_lambda.ndim > 0 or lame_mu.ndim > 0

    def surface_maps(_self):
        # Weak form: residual += integral(v * surface_map); a traction t
        # enters as -t so that sigma.n = t on the patch.
        return [(lambda _u, _x, vector=vector: -jnp.asarray(vector)) for vector in tractions]

    def hooke(u_grad, lmbda, shear):
        strain = 0.5 * (u_grad + u_grad.T)
        return lmbda * jnp.trace(strain) * jnp.eye(3) + 2.0 * shear * strain

    if not heterogeneous:

        class _Elastic(Problem):
            def get_tensor_map(self):
                def stress(u_grad):
                    return hooke(u_grad, lame_lambda, lame_mu)

                return stress

            get_surface_maps = surface_maps

            def set_params(self, params):
                self.initialize_geometric_quantities([params])

        return _Elastic, (lambda points: jnp.asarray(points))

    class _Elastic(Problem):
        def get_tensor_map(self):
            if has_force:

                def stress(u_grad, lmbda, shear, _force):
                    return hooke(u_grad, lmbda, shear)
            else:

                def stress(u_grad, lmbda, shear):
                    return hooke(u_grad, lmbda, shear)

            return stress

        get_surface_maps = surface_maps

        def set_params(self, params):
            if has_force:
                params_points, lmbda, shear, force = params
            else:
                params_points, lmbda, shear = params
            self.initialize_geometric_quantities([params_points])
            fe = self.fes[0]
            shape = (fe.num_cells, fe.num_quads)
            self.internal_vars = [_cellwise(lmbda, shape), _cellwise(shear, shape)]
            if has_force:
                self.internal_vars.append(_cellwise_vector(force, shape))

    if has_force:

        def get_mass_map(_self):
            # Weak form: residual += integral(v * mass_map); a body force b
            # enters as -b so that div(sigma) + b = 0.
            def mass_map(_u, _x, _lmbda, _shear, force):
                return -force

            return mass_map

        _Elastic.get_mass_map = get_mass_map

    if has_force:
        force_array = jnp.asarray(body_force, dtype=jnp.float64)

        def make_params(points):
            return (jnp.asarray(points), lame_lambda, lame_mu, force_array)
    else:

        def make_params(points):
            return (jnp.asarray(points), lame_lambda, lame_mu)

    return _Elastic, make_params


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
                        _cellwise(kappa, shape),
                        _cellwise(source_value, shape),
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

    def elastic(self, points, cells, bcs, *, youngs, poisson, base_points=None, body_force=None):
        """See :meth:`~cadjoint.fem.backends.SolverBackend.elastic`.

        ``youngs`` and ``poisson`` may each be a scalar (the homogeneous path,
        unchanged) or a per-element ``(C,)`` array sampled from the scene's
        material field; ``body_force`` optionally adds a body force density in
        N/m^3 (``density * gravity`` for self-weight), ``(3,)`` or ``(C, 3)``.
        In the heterogeneous case the moduli travel through ``internal_vars``,
        so the solve is differentiable w.r.t. them as well as w.r.t. ``points``
        (see :func:`_elastic_problem`).
        """
        _require_jax_fem()
        with _x64_scope():
            import jax.numpy as jnp
            from jax_fem.generate_mesh import Mesh
            from jax_fem.solver import ad_wrapper

            tractions = [np.asarray(vector, dtype=np.float64) for vector in bcs.traction_vectors]
            problem_class, make_params = _elastic_problem(
                tractions, youngs=youngs, poisson=poisson, body_force=body_force
            )

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
            problem = problem_class(
                mesh=mesh,
                vec=3,
                dim=3,
                ele_type="HEX8",
                dirichlet_bc_info=dirichlet,
                location_fns=[_membership_location(nodes) for nodes in bcs.traction_nodes],
            )
            forward = ad_wrapper(problem)
            return forward(make_params(jnp.asarray(points)))[0]


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


def _rebuild_face_blocks(problem: Any) -> None:
    """Realign a problem's face structures with its current face selection.

    ``boundary_inds_list`` is the (cell, local face) selection each surface
    map integrates over, and two things are derived from it at construction:
    ``cells_list_face_list``, the connectivity the face residual scatters
    through, and the face blocks of the assembly sparsity pattern
    ``I``/``J``.  Anything that rewrites the selection — the pruning of
    :func:`_prune_surface_faces`, the padding of :func:`_pad_surface_faces` —
    has to rebuild both, or the value and index arrays of the tangent stop
    lining up.  Surface quadrature data needs no help here: ``set_params``
    recomputes it from the selection before every solve.
    """

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
    faces = []
    for binds in problem.boundary_inds_list:
        cells_face = [np.asarray(c)[np.asarray(binds)[:, 0]] for c in problem.cells_list]
        faces.append(cells_face)
        inds_face = flat_dof_ids(cells_face)
        pattern_i = np.hstack(
            [pattern_i, np.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)]
        )
        pattern_j = np.hstack(
            [pattern_j, np.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)]
        )
    problem.cells_list_face_list = faces
    problem.I = pattern_i
    problem.J = pattern_j


def _prune_surface_faces(problem: Any, surface_faces: list[np.ndarray]) -> None:
    """Prune jax-fem's face selection to exactly the given boundary triangles.

    jax-fem selects a (cell, local face) pair for a surface map whenever
    *all* the face's nodes satisfy the location function.  With node-set
    membership locations on a tet mesh this over-selects: an interior
    face whose three corners all happen to lie on the loaded surface
    patch is selected once per adjacent cell, double-loading a face that
    is not even on the boundary (observed on the bracket web at fine
    resolutions).  This helper prunes each patch's selection (traction or
    heat-flux alike) to the faces whose corner triple matches the
    requested boundary triangles; the caller rebuilds the dependent
    structures with :func:`_rebuild_face_blocks`.
    """
    finite_element = problem.fes[0]
    face_inds = np.asarray(finite_element.face_inds)
    # Local corner slots per face: for TET4 all three face nodes are
    # corners; for TET10 the corners are the local indices below 4.
    corner_slots = np.stack([np.sort(local[local < 4])[:3] for local in face_inds])
    cells0 = np.asarray(finite_element.cells)
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
        problem.boundary_inds_list[patch] = binds[mask]


def _pad_surface_faces(problem: Any, ghost_cell: int) -> None:
    """Grow every face selection to a rung with faces of the ghost body.

    The selections are the last mesh-dependent shapes in the solve: one
    surface kernel is compiled per patch per face count, and the count of
    boundary triangles a patch carries moves with every design edit even
    when the node and cell counts have been pinned.  A padded row names
    local face 0 of a padded cell, whose nodes are ghost nodes, so its
    contribution is scattered into rows the Dirichlet elimination replaces
    and reaches no real degree of freedom.  Empty selections are left
    alone: an empty patch is already a stable shape, and padding it would
    turn "this patch carries nothing" into "this patch carries something",
    which is a different problem even when the something is inert.
    """
    for patch, binds in enumerate(problem.boundary_inds_list):
        selection = np.asarray(binds)
        target = rung(int(selection.shape[0]))
        if target <= selection.shape[0]:
            continue
        filler = np.tile(
            np.asarray([[ghost_cell, 0]], dtype=selection.dtype), (target - len(selection), 1)
        )
        problem.boundary_inds_list[patch] = np.concatenate([selection, filler])


def _pad_dirichlet_selection(problem: Any, ghost_node: int) -> None:
    """Grow every Dirichlet selection to a rung with a ghost degree of freedom.

    ``node_inds_list`` is what the row elimination indexes with, and it is
    the *selected* count, not the node set the caller passed: jax-fem
    evaluates the location function over every node and keeps the distinct
    matches, so repeating an index in the caller's set does not reach here.
    The count moves with the design — a clamped face gains a node when the
    surface does — and each patch is a shape in ``apply_bc_vec``,
    ``assign_bc`` and ``copy_bc`` alike.

    Padding with a *ghost* node is what makes this exact rather than nearly
    so.  ``apply_bc_vec`` reads ``res.at[nodes, vecs].set(sol[...])``
    followed by ``.at[nodes, vecs].add(-values)``, and the second one
    accumulates: a repeated *real* index would subtract its prescribed
    value twice and move the boundary.  A ghost index cannot, whatever any
    patch prescribes — the ghost body shares no degree of freedom with the
    mesh and is cut off the answer — so the padded rows are inert by
    construction and not by arithmetic.  A patch that selected nothing is
    left alone; it is already a stable shape.

    The concatenation runs on the host, and it stays there.  Building the
    padded arrays in ``jax.numpy`` would put the *unpadded* count back into
    the shapes — a dozen one-op programs keyed on exactly the count being
    padded away, which is the trap ``research/performance.md`` §17.7
    describes from the other side.  Handing the finished NumPy arrays back
    to the device with ``jnp.asarray`` was tried and measured *worse*
    (six more eager programs per solve, §18.5): jax-fem is happy to index
    with a host array, and the transfers are not free.
    """
    finite_element = problem.fes[0]
    for patch, nodes in enumerate(finite_element.node_inds_list):
        indices = np.asarray(nodes)
        count = int(indices.shape[0])
        target = rung(count)
        if target <= count:
            continue
        fill = target - count
        component = np.asarray(finite_element.vec_inds_list[patch])
        values = np.asarray(finite_element.vals_list[patch])
        finite_element.node_inds_list[patch] = np.concatenate(
            [indices, np.full(fill, ghost_node, dtype=indices.dtype)]
        )
        finite_element.vec_inds_list[patch] = np.concatenate(
            [component, np.full(fill, component[0], dtype=component.dtype)]
        )
        finite_element.vals_list[patch] = np.concatenate(
            [values, np.zeros(fill, dtype=values.dtype)]
        )


def _pinning_sets(patches: list[np.ndarray], ghosts: np.ndarray) -> list[np.ndarray]:
    """The Dirichlet node sets with the ghost body pinned, each on a rung.

    The ghosts go into the first patch because they have to go somewhere —
    an unpinned ghost node owns an empty row of the tangent — and the first
    patch is the one that exists whenever padding is on at all.  Every set
    is then grown to a rung, which stabilises the shape of the membership
    test each location function closes over.  Repetition is the filler and
    it changes nothing: the location function asks whether a node is in the
    set, and jax-fem keeps the distinct matches.
    """
    filler = int(ghosts[-1])
    return [
        pad_indices(np.concatenate([np.asarray(nodes, dtype=np.int32), ghosts]), filler)
        if patch == 0
        else pad_indices(nodes)
        for patch, nodes in enumerate(patches)
    ]


def _padded_thermal_bcs(bcs: ThermalBCs, ghosts: np.ndarray) -> ThermalBCs:
    """``bcs`` with the ghost body pinned and every node set on a rung."""
    return ThermalBCs(
        dirichlet_nodes=_pinning_sets(bcs.dirichlet_nodes, ghosts),
        dirichlet_values=list(bcs.dirichlet_values),
        flux_nodes=[pad_indices(nodes) for nodes in bcs.flux_nodes],
        flux_values=list(bcs.flux_values),
    )


def _padded_elastic_bcs(bcs: ElasticBCs, ghosts: np.ndarray) -> ElasticBCs:
    """``bcs`` with the ghost body clamped and every node set on a rung."""
    return ElasticBCs(
        fixed_nodes=_pinning_sets(bcs.fixed_nodes, ghosts),
        traction_nodes=[pad_indices(nodes) for nodes in bcs.traction_nodes],
        traction_vectors=list(bcs.traction_vectors),
    )


def _pad_problem(
    points: Any, base_points: Any, cells: np.ndarray, pinned: list[np.ndarray]
) -> PaddedTets | None:
    """The mesh grown to the ladder, or ``None`` when it must be solved as given.

    Padding needs somewhere to pin the ghost body, so a problem with no
    Dirichlet patch at all is solved on the raw mesh: an unpinned ghost
    block is a singular tangent, which is a wrong answer where an
    unstable shape is only a slow one.
    """
    if not any(len(np.asarray(nodes)) for nodes in pinned):
        return None
    return pad_tets(points, base_points, cells)


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
    body_force: Any = None,
) -> Any:
    """Small-strain linear elasticity on a tet mesh via jax-fem.

    Mirrors :meth:`JaxFemBackend.elastic` with the element type opened up:
    ``"TET4"`` or ``"TET10"`` (both confirmed in jax-fem's element tables;
    connectivity must be meshio ``tetra`` / ``tetra10`` order, as produced
    by TetGen resp. :func:`~cadjoint.fem.tetmesh.tet10_from_tet4`).
    ``points`` may be traced; the displacement participates in the
    surrounding autodiff graph via jax-fem's adjoint.

    The mesh is solved on a *rung* — grown to a fixed size with a pinned,
    disconnected ghost body, so a run of similar designs shares one
    compiled assembly instead of compiling a fresh one per design
    (:mod:`cadjoint.fem.rungs`, ``research/performance.md`` §18).  The
    returned displacement is cut back to the caller's ``N`` rows and
    differs from the unpadded solve only by the rounding of a larger
    factorization; ``CADJOINT_FEM_RUNGS=off`` solves the raw mesh.

    Args:
        points: Node positions, ``(N, 3)`` (traced allowed).
        cells: Connectivity, ``(T, 4)`` or ``(T, 10)``.
        bcs: Array-level boundary conditions (the backend ABI).  For
            ``TET10``, node sets must include midside nodes (a face
            carries a traction when *all* its nodes are in the set).
            At least one ``fixed_nodes`` patch must be non-empty for the
            mesh to be padded; without one the ghost body cannot be
            pinned and the raw mesh is solved instead.
        youngs: Young's modulus — scalar, or per element ``(T,)``.
        poisson: Poisson ratio — scalar, or per element ``(T,)``.
        ele_type: ``"TET4"`` or ``"TET10"``.
        base_points: Concrete positions for problem construction when
            ``points`` is traced (defaults to ``points``).
        traction_faces: Optional exact face targeting: one ``(M, >=3)``
            array of *corner* node triples per traction patch (boundary
            triangles).  When given, jax-fem's node-membership face
            selection is pruned to exactly these faces — closing the
            interior-face double-count hole of pure node membership (see
            :func:`_prune_surface_faces`).  Every corner must also be
            in the corresponding ``bcs.traction_nodes`` set.
        body_force: Optional body force density in N/m^3, ``(3,)`` or
            ``(T, 3)`` — ``density * gravity`` for self-weight.

    Returns:
        Per-node displacement, ``(N, 3)`` JAX array.
    """
    if ele_type not in ("TET4", "TET10"):
        raise ValueError(f"ele_type must be 'TET4' or 'TET10', got {ele_type!r}.")
    _require_jax_fem()
    with _x64_scope():
        import jax.numpy as jnp
        from jax_fem.generate_mesh import Mesh
        from jax_fem.solver import ad_wrapper

        if base_points is None:
            base_points = points
        base_points = np.asarray(base_points, dtype=np.float64)
        cells = np.asarray(cells)
        if traction_faces is not None and len(traction_faces) != len(bcs.traction_nodes):
            raise ValueError(
                "traction_faces must provide one face array per traction patch "
                f"({len(traction_faces)} given for {len(bcs.traction_nodes)} patches)."
            )
        # Padding comes before the problem class is built: the class bakes
        # the Lame constants into its tensor map and closes over the body
        # force, so a field padded afterwards would never reach the solve.
        padding = _pad_problem(points, base_points, cells, bcs.fixed_nodes)
        if padding is not None:
            points, base_points, cells = padding.points, padding.base_points, padding.cells
            bcs = _padded_elastic_bcs(bcs, padding.ghost_nodes)
            youngs = pad_cell_field(youngs, padding, 1)
            poisson = pad_cell_field(poisson, padding, 1)
            body_force = pad_cell_field(body_force, padding, 2)
        tractions = [np.asarray(vector, dtype=np.float64) for vector in bcs.traction_vectors]
        problem_class, make_params = _elastic_problem(
            tractions, youngs=youngs, poisson=poisson, body_force=body_force
        )
        mesh = Mesh(base_points, cells, ele_type=ele_type)
        fixed_locations = [_membership_location(nodes) for nodes in bcs.fixed_nodes]
        dirichlet = [
            [location for location in fixed_locations for _ in range(3)],
            [component for _ in fixed_locations for component in range(3)],
            [(lambda _point: 0.0) for _ in fixed_locations for _ in range(3)],
        ]
        problem = problem_class(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet,
            location_fns=[_membership_location(nodes) for nodes in bcs.traction_nodes],
        )
        if traction_faces is not None:
            _prune_surface_faces(problem, traction_faces)
        if padding is not None:
            _pad_surface_faces(problem, padding.ghost_cell)
            _pad_dirichlet_selection(problem, int(padding.ghost_nodes[0]))
        if traction_faces is not None or padding is not None:
            _rebuild_face_blocks(problem)
        forward = ad_wrapper(
            problem,
            solver_options=dict(_TET_SOLVER_OPTIONS),
            adjoint_solver_options=dict(_TET_SOLVER_OPTIONS),
        )
        displacement = forward(make_params(jnp.asarray(points)))[0]
        return displacement if padding is None else displacement[: padding.node_count]


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

    The mesh is solved on a *rung* — grown to a fixed size with a pinned,
    disconnected ghost body, so a run of similar designs shares one
    compiled assembly instead of compiling a fresh one per design
    (:mod:`cadjoint.fem.rungs`, ``research/performance.md`` §18).  The
    returned temperature is cut back to the caller's ``N`` rows and differs
    from the unpadded solve only by the rounding of a larger factorization;
    ``CADJOINT_FEM_RUNGS=off`` solves the raw mesh.

    Args:
        points: Node positions, ``(N, 3)`` (traced allowed).
        cells: Connectivity, ``(T, 4)`` or ``(T, 10)``.
        bcs: Array-level thermal boundary conditions (the backend ABI).
            For ``TET10``, node sets must include midside nodes
            (:func:`~cadjoint.fem.boundary.tet10_complete_nodes`).
            At least one ``dirichlet_nodes`` patch must be non-empty for
            the mesh to be padded; without one the ghost body cannot be
            pinned and the raw mesh is solved instead.
        conductivity: Thermal conductivity ``k`` (may be traced).
        source: Volumetric heat source ``q`` (may be traced).
        ele_type: ``"TET4"`` or ``"TET10"``.
        base_points: Concrete positions for problem construction when
            ``points`` is traced (defaults to ``points``).
        flux_faces: Optional exact face targeting: one ``(M, >=3)`` array
            of *corner* node triples per heat-flux patch, pruning jax-fem's
            node-membership face selection to exactly these boundary
            triangles (see :func:`_prune_surface_faces`).

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
                    _cellwise(kappa, shape),
                    _cellwise(source_value, shape),
                    lift_grad,
                ]

        if base_points is None:
            base_points = points
        base_points = np.asarray(base_points, dtype=np.float64)
        cells = np.asarray(cells)
        if flux_faces is not None and len(flux_faces) != len(bcs.flux_nodes):
            raise ValueError(
                "flux_faces must provide one face array per flux patch "
                f"({len(flux_faces)} given for {len(bcs.flux_nodes)} patches)."
            )
        padding = _pad_problem(points, base_points, cells, bcs.dirichlet_nodes)
        if padding is not None:
            points, base_points, cells = padding.points, padding.base_points, padding.cells
            bcs = _padded_thermal_bcs(bcs, padding.ghost_nodes)
            conductivity = pad_cell_field(conductivity, padding, 1)
            source = pad_cell_field(source, padding, 1)
        mesh = Mesh(base_points, cells, ele_type=ele_type)
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
            _prune_surface_faces(problem, flux_faces)
        if padding is not None:
            _pad_surface_faces(problem, padding.ghost_cell)
            _pad_dirichlet_selection(problem, int(padding.ghost_nodes[0]))
        if flux_faces is not None or padding is not None:
            _rebuild_face_blocks(problem)
        forward = ad_wrapper(
            problem,
            solver_options=dict(_TET_SOLVER_OPTIONS),
            adjoint_solver_options=dict(_TET_SOLVER_OPTIONS),
        )

        # Built from the padded node sets, so its scatter has a rung's shape
        # like everything else.  The ghost body lands in patch 0 and takes
        # that patch's temperature; being one constant over a disconnected,
        # pinned block, it has no gradient inside a ghost element and no
        # degree of freedom outside one.
        lift = jnp.zeros(base_points.shape[0], dtype=jnp.float64)
        for nodes, value in zip(bcs.dirichlet_nodes, bcs.dirichlet_values, strict=True):
            lift = lift.at[jnp.asarray(np.asarray(nodes, dtype=np.int32))].set(value)
        solution = forward((jnp.asarray(points), conductivity, source, lift))
        temperature = solution[0][:, 0] + lift
        return temperature if padding is None else temperature[: padding.node_count]
