"""Pluggable FEM solver backends.

The solver layer is split so jax-fem is one backend rather than a
hard-wired dependency:

- **ABI**: boundary conditions and mesh data cross the backend boundary as
  plain arrays (:class:`ThermalBCs` / :class:`ElasticBCs` hold node index
  sets and values, resolved from user predicates by
  :mod:`cadjoint.fem.simulate`).  Anything expressible as arrays can also
  cross a Tesseract schema, so third-party — even non-JAX — solvers can
  plug in by shipping a tesseract with ``apply`` + ``vector_jacobian_product``
  endpoints.
- **Default**: :class:`JaxFemBackend` runs jax-fem in-process (native JAX
  composition, no serialization boundary) — the performance baseline.
- **Interop**: :class:`TesseractBackend` routes through
  ``tesseract_jax.apply_tesseract`` and the packaged reference tesseract in
  :mod:`cadjoint.fem.tesseracts`, proving the plugin contract.

Differentiability contract: ``thermal``/``elastic`` accept possibly-traced
``points`` and return JAX arrays whose VJP w.r.t. ``points`` (and scalar
material parameters where noted) is defined — via jax-fem's adjoint
(``ad_wrapper``) for the direct backend, via the tesseract's
``vector_jacobian_product`` endpoint for tesseract backends.  The direct
thermal solve is additionally differentiable w.r.t. the prescribed
Dirichlet *values* (lifted formulation; see
:meth:`JaxFemBackend.thermal`).  Forward solves are not jax-traceable
end-to-end (PETSc assembly), so none of these calls may sit under
``jax.jit``.

Precision contract: each backend call enables jax's x64 mode only for its
own duration (see :func:`_x64_scope`) so forward solves never leak float64
into the rest of the process.  Callers that differentiate *through* a solve
run the backward pass after that scope has exited and must enable x64
process-wide themselves (``jax.config.update("jax_enable_x64", True)``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

_FEM_EXTRA_MESSAGE = (
    "jax-fem is not installed. Install the 'fem' extra: pip install cadjoint[fem] "
    "(plus its runtime dependencies: fenics-basix, meshio, gmsh, petsc4py, pyfiglet, scipy)."
)
_TESSERACT_EXTRA_MESSAGE = (
    "tesseract-core / tesseract-jax are not installed. "
    "Install the 'tesseract' extra: pip install cadjoint[tesseract]."
)


@dataclass(frozen=True)
class ThermalBCs:
    """Array-level thermal boundary conditions (the backend ABI).

    Attributes:
        dirichlet_nodes: One int array of node indices per prescribed patch.
        dirichlet_values: Prescribed temperature per patch — plain floats or
            (for the direct backend) traced JAX scalars, in which case the
            solve is differentiable w.r.t. the prescribed values.
        flux_nodes: One int array of the vertex indices spanning each
            heat-flux (Neumann) patch; a boundary face carries the flux when
            all of its vertices belong to the set.
        flux_values: Prescribed heat inflow per area for each flux patch
            (positive heats the body).
    """

    dirichlet_nodes: list[np.ndarray] = field(default_factory=list)
    dirichlet_values: list[Any] = field(default_factory=list)
    flux_nodes: list[np.ndarray] = field(default_factory=list)
    flux_values: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ElasticBCs:
    """Array-level elastic boundary conditions (the backend ABI).

    Attributes:
        fixed_nodes: One int array of node indices per fully-clamped patch.
        traction_nodes: One int array of the vertex indices spanning each
            traction patch; a boundary face carries the traction when all
            of its vertices belong to the set.
        traction_vectors: Traction vector (force per area) per patch.
    """

    fixed_nodes: list[np.ndarray] = field(default_factory=list)
    traction_nodes: list[np.ndarray] = field(default_factory=list)
    traction_vectors: list[np.ndarray] = field(default_factory=list)


@runtime_checkable
class SolverBackend(Protocol):
    """Protocol every FEM backend implements.

    ``points`` may be a traced JAX array; implementations must return JAX
    arrays participating in the surrounding autodiff graph (adjoint VJP at
    minimum).  ``cells`` and BC index arrays are static.
    """

    name: str

    def thermal(
        self,
        points: Any,
        cells: np.ndarray,
        bcs: ThermalBCs,
        *,
        conductivity: float,
        source: float,
        base_points: np.ndarray | None = None,
    ) -> Any:
        """Solve -div(k grad T) = q; returns per-node temperature ``(N,)``.

        ``base_points`` is a concrete snapshot of the node positions used
        for problem construction (BC bookkeeping) when ``points`` is a
        traced array; it defaults to ``points`` itself.
        """
        ...

    def elastic(
        self,
        points: Any,
        cells: np.ndarray,
        bcs: ElasticBCs,
        *,
        youngs: float,
        poisson: float,
        base_points: np.ndarray | None = None,
    ) -> Any:
        """Solve small-strain linear elasticity; returns displacement ``(N, 3)``."""
        ...


def _require_jax_fem():
    """Import jax-fem or raise naming the extra (no global config changes)."""
    try:
        import jax_fem  # noqa: F401
    except ImportError as error:
        raise ImportError(_FEM_EXTRA_MESSAGE) from error
    import logging

    from jax_fem import logger

    logger.setLevel(logging.WARNING)


@contextmanager
def _x64_scope() -> Iterator[None]:
    """Enable jax's x64 mode for the duration, restoring the caller's setting.

    jax-fem requires float64, but flipping ``jax_enable_x64`` permanently
    would leak into every float32 computation that runs later in the same
    process (importing ``jax_fem.solver`` alone flips it at module scope —
    this scope also undoes that).  Restoring after the forward solve is safe
    for forward-only callers such as the viewer's preview path; callers who
    differentiate *through* a solve run the backward pass after this scope
    has exited and must therefore enable x64 process-wide themselves (the
    FEM test suite and the optimization example both do).
    """
    import jax

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _membership_location(node_set: np.ndarray) -> Callable[..., Any]:
    """A 2-arg jax-fem location function selecting nodes by index."""
    import jax.numpy as jnp

    indices = jnp.asarray(np.asarray(node_set, dtype=np.int32))

    def location(point, index):  # noqa: ARG001 - jax-fem calls with (point, index)
        return jnp.isin(index, indices)

    return location


class JaxFemBackend:
    """Direct in-process jax-fem backend (default; performance baseline).

    Gradients flow through jax-fem's adjoint (``ad_wrapper``): the forward
    solve runs concretely (PETSc-assembled Newton), the VJP solves the
    adjoint system and back-propagates through the residual — including
    through the nodal coordinates via
    ``Problem.initialize_geometric_quantities``.
    """

    name = "jaxfem"

    def thermal(self, points, cells, bcs, *, conductivity, source, base_points=None):
        """See :meth:`SolverBackend.thermal`.

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
        """See :meth:`SolverBackend.elastic`."""
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


class TesseractBackend:
    """Backend routing through Tesseracts (interop ABI reference).

    Executes the packaged jax-fem thermal and elastic tesseracts locally
    (no Docker) via ``Tesseract.from_tesseract_api`` and composes them into
    JAX autodiff with ``tesseract_jax.apply_tesseract``, which dispatches to
    the tesseract's ``vector_jacobian_product`` endpoint under ``jax.grad``.
    Third-party solvers plug in the same way: point ``api_path`` /
    ``elastic_api_path`` at their ``tesseract_api.py`` (or adapt this class
    to a served/containerized tesseract via ``Tesseract.from_image``).
    """

    name = "tesseract"

    def __init__(
        self,
        api_path: str | Path | None = None,
        *,
        elastic_api_path: str | Path | None = None,
    ):
        try:
            from tesseract_core import Tesseract  # noqa: F401
        except ImportError as error:
            raise ImportError(_TESSERACT_EXTRA_MESSAGE) from error
        _require_jax_fem()  # the packaged reference tesseracts run jax-fem in-process
        tesseracts_dir = Path(__file__).parent / "tesseracts"
        if api_path is None:
            api_path = tesseracts_dir / "thermal_jaxfem" / "tesseract_api.py"
        if elastic_api_path is None:
            elastic_api_path = tesseracts_dir / "elastic_jaxfem" / "tesseract_api.py"
        self._api_paths = {"thermal": Path(api_path), "elastic": Path(elastic_api_path)}
        self._tesseracts: dict[str, Any] = {}

    def _tesseract_for(self, kind: str):
        """Load the tesseract for ``kind`` lazily (kept warm per instance)."""
        if kind not in self._tesseracts:
            from tesseract_core import Tesseract

            self._tesseracts[kind] = Tesseract.from_tesseract_api(str(self._api_paths[kind]))
        return self._tesseracts[kind]

    def thermal(self, points, cells, bcs, *, conductivity, source, base_points=None):
        """See :meth:`SolverBackend.thermal`.

        ``base_points`` is unused: the tesseract runtime hands its endpoints
        concrete arrays, so construction-time concreteness is guaranteed.
        """
        del base_points
        import jax.numpy as jnp
        from tesseract_jax import apply_tesseract

        if bcs.dirichlet_nodes:
            nodes = np.concatenate([np.asarray(n, dtype=np.int32) for n in bcs.dirichlet_nodes])
            values = np.concatenate(
                [
                    np.full(len(n), value, dtype=np.float64)
                    for n, value in zip(bcs.dirichlet_nodes, bcs.dirichlet_values)
                ]
            )
        else:
            nodes = np.zeros(0, dtype=np.int32)
            values = np.zeros(0, dtype=np.float64)
        if bcs.flux_nodes:
            flux_nodes = np.concatenate([np.asarray(n, dtype=np.int32) for n in bcs.flux_nodes])
            flux_values = np.asarray(bcs.flux_values, dtype=np.float64)
        else:
            flux_nodes = np.zeros(0, dtype=np.int32)
            flux_values = np.zeros(0, dtype=np.float64)
        flux_offsets = np.concatenate(
            [[0], np.cumsum([len(n) for n in bcs.flux_nodes], dtype=np.int64)]
        ).astype(np.int32)
        with _x64_scope():
            outputs = apply_tesseract(
                self._tesseract_for("thermal"),
                {
                    "points": jnp.asarray(points, dtype=jnp.float64),
                    "cells": np.asarray(cells, dtype=np.int32),
                    "dirichlet_nodes": nodes,
                    "dirichlet_values": jnp.asarray(values),
                    "flux_nodes": flux_nodes,
                    "flux_offsets": flux_offsets,
                    "flux_values": flux_values,
                    # Exact-face targeting is a tet feature; hex meshes use
                    # pure node membership (empty offsets = disabled).
                    "flux_faces": np.zeros((0, 3), dtype=np.int32),
                    "flux_face_offsets": np.zeros(0, dtype=np.int32),
                    "conductivity": jnp.asarray(conductivity, dtype=jnp.float64),
                    "source": jnp.asarray(source, dtype=jnp.float64),
                },
            )
            return outputs["temperature"]

    def elastic(self, points, cells, bcs, *, youngs, poisson, base_points=None):
        """See :meth:`SolverBackend.elastic`.

        ``base_points`` is unused: the tesseract runtime hands its endpoints
        concrete arrays, so construction-time concreteness is guaranteed.
        Clamped patches cross the boundary as one union node set (all
        components pinned to zero, so patch identity is irrelevant);
        traction patches keep their identity via prefix offsets.
        """
        del base_points
        import jax.numpy as jnp
        from tesseract_jax import apply_tesseract

        if bcs.fixed_nodes:
            fixed = np.unique(
                np.concatenate([np.asarray(n, dtype=np.int32) for n in bcs.fixed_nodes])
            ).astype(np.int32)
        else:
            fixed = np.zeros(0, dtype=np.int32)
        if bcs.traction_nodes:
            traction_nodes = np.concatenate(
                [np.asarray(n, dtype=np.int32) for n in bcs.traction_nodes]
            )
            traction_vectors = np.asarray(bcs.traction_vectors, dtype=np.float64)
        else:
            traction_nodes = np.zeros(0, dtype=np.int32)
            traction_vectors = np.zeros((0, 3), dtype=np.float64)
        offsets = np.concatenate(
            [[0], np.cumsum([len(n) for n in bcs.traction_nodes], dtype=np.int64)]
        ).astype(np.int32)
        with _x64_scope():
            outputs = apply_tesseract(
                self._tesseract_for("elastic"),
                {
                    "points": jnp.asarray(points, dtype=jnp.float64),
                    "cells": np.asarray(cells, dtype=np.int32),
                    "fixed_nodes": fixed,
                    "traction_nodes": traction_nodes,
                    "traction_offsets": offsets,
                    "traction_vectors": traction_vectors,
                    # Exact-face targeting is a tet feature; hex meshes use
                    # pure node membership (empty offsets = disabled).
                    "traction_faces": np.zeros((0, 3), dtype=np.int32),
                    "traction_face_offsets": np.zeros(0, dtype=np.int32),
                    "youngs": np.asarray(youngs, dtype=np.float64),
                    "poisson": np.asarray(poisson, dtype=np.float64),
                },
            )
            return outputs["displacement"]


def _calculix_backend() -> SolverBackend:
    """Factory for the CalculiX backend (lazy import keeps ccx optional)."""
    from cadjoint.fem.calculix import CalculixBackend

    return CalculixBackend()


_REGISTRY: dict[str, Callable[[], SolverBackend]] = {
    "jaxfem": JaxFemBackend,
    "tesseract": TesseractBackend,
    "calculix": _calculix_backend,
}


def register_backend(name: str, factory: Callable[[], SolverBackend]) -> None:
    """Register a solver backend factory under ``name``."""
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    """Names of registered backends (not necessarily importable)."""
    return sorted(_REGISTRY)


def get_backend(backend: str | SolverBackend | None = None) -> SolverBackend:
    """Resolve a backend name (or pass an instance through).

    ``None`` selects the default direct jax-fem backend.

    Raises:
        ImportError: If the backend's dependencies are missing (message
            names the extra to install).
        KeyError: If the name is not registered.
    """
    if backend is None:
        backend = "jaxfem"
    if isinstance(backend, str):
        try:
            factory = _REGISTRY[backend]
        except KeyError:
            raise KeyError(
                f"Unknown FEM backend {backend!r}; registered: {available_backends()}"
            ) from None
        return factory()
    return backend
