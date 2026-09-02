"""The FEM solver ABI, the backend registry, and the Tesseract interop path.

What belongs here: the contract every solver plugs into, and nothing that
solves.  Concretely — the array-level boundary-condition payloads
(:class:`ThermalBCs` / :class:`ElasticBCs`), the :class:`SolverBackend`
protocol, the name -> factory registry (:func:`get_backend`,
:func:`register_backend`), the cross-process interop backend
(:class:`TesseractBackend`), and the small shared utilities every solver
needs (:func:`_x64_scope`, :func:`_require_jax_fem`,
:func:`_membership_location`).

What does *not* belong here: finite-element formulations.  Every direct
jax-fem solve — HEX8 *and* TET4/TET10 — lives in :mod:`cadjoint.fem.jaxfem`;
CalculiX lives in :mod:`cadjoint.fem.calculix`.  The registry reaches them
through lazy factories, so this module imports no solver implementation and
stays importable without jax-fem, tesseract-core or ccx.

The split exists so jax-fem is one backend rather than a hard-wired
dependency:

- **ABI**: boundary conditions and mesh data cross the backend boundary as
  plain arrays (:class:`ThermalBCs` / :class:`ElasticBCs` hold node index
  sets and values, resolved from user predicates by
  :mod:`cadjoint.fem.simulate`).  Anything expressible as arrays can also
  cross a Tesseract schema, so third-party — even non-JAX — solvers can
  plug in by shipping a tesseract with ``apply`` + ``vector_jacobian_product``
  endpoints.
- **Default**: :class:`~cadjoint.fem.jaxfem.JaxFemBackend` runs jax-fem
  in-process (native JAX composition, no serialization boundary) — the
  performance baseline.
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
:meth:`~cadjoint.fem.jaxfem.JaxFemBackend.thermal`).  Forward solves are not
jax-traceable end-to-end (PETSc assembly), so none of these calls may sit
under ``jax.jit``.

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
        conductivity: Any,
        source: float,
        base_points: np.ndarray | None = None,
    ) -> Any:
        """Solve -div(k grad T) = q; returns per-node temperature ``(N,)``.

        ``conductivity`` is either a scalar (one material for the whole
        domain) or a per-element ``(C,)`` array sampled from the scene's
        material field (:mod:`cadjoint.fem.properties`); a backend that
        cannot represent a heterogeneous field says so rather than
        silently averaging.

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
        youngs: Any,
        poisson: Any,
        base_points: np.ndarray | None = None,
        body_force: Any = None,
    ) -> Any:
        """Solve small-strain linear elasticity; returns displacement ``(N, 3)``.

        ``youngs`` and ``poisson`` are scalars or per-element ``(C,)`` arrays
        (see :meth:`thermal`).  ``body_force`` optionally prescribes a body
        force density in N/m^3 — ``density * gravity`` for self-weight —
        shaped ``(3,)`` or ``(C, 3)``; ``None`` means no body force.
        """
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


def _as_cell_array(value: Any, num_cells: int) -> np.ndarray:
    """A per-element property as a ``(C,)`` array, or an empty one if scalar.

    The wire form used by the packaged tesseracts: an empty array means "the
    scalar input applies to the whole domain", which is what every existing
    caller sends, so the scalar path stays the schema default.

    A traced value comes back *as a JAX array*, never concretized — the
    per-element properties are differentiable inputs of the tesseracts, so
    concretizing here would sever the gradient before the call is even made
    (and raise a tracer-conversion error under ``jax.grad``).  Only the
    array's shape is inspected, which is static.
    """
    import jax.numpy as jnp

    array = jnp.asarray(value)
    if array.ndim == 0:
        return jnp.zeros(0, dtype=jnp.float64)
    if array.shape != (num_cells,):
        raise ValueError(
            f"Per-element property must be scalar or shaped ({num_cells},); got {array.shape}."
        )
    return array.astype(jnp.float64)


def _membership_location(node_set: np.ndarray) -> Callable[..., Any]:
    """A 2-arg jax-fem location function selecting nodes by index."""
    import jax.numpy as jnp

    indices = jnp.asarray(np.asarray(node_set, dtype=np.int32))

    def location(point, index):  # noqa: ARG001 - jax-fem calls with (point, index)
        return jnp.isin(index, indices)

    return location


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

        A per-element ``conductivity`` crosses the boundary as the schema's
        optional ``cell_conductivity`` array; a scalar leaves that array empty
        so the wire payload is exactly what it always was.
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
        cell_conductivity = _as_cell_array(conductivity, int(np.asarray(cells).shape[0]))
        scalar_conductivity = 0.0 if cell_conductivity.size else conductivity
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
                    "conductivity": jnp.asarray(scalar_conductivity, dtype=jnp.float64),
                    "cell_conductivity": jnp.asarray(cell_conductivity, dtype=jnp.float64),
                    "source": jnp.asarray(source, dtype=jnp.float64),
                },
            )
            return outputs["temperature"]

    def elastic(self, points, cells, bcs, *, youngs, poisson, base_points=None, body_force=None):
        """See :meth:`SolverBackend.elastic`.

        ``base_points`` is unused: the tesseract runtime hands its endpoints
        concrete arrays, so construction-time concreteness is guaranteed.
        Clamped patches cross the boundary as one union node set (all
        components pinned to zero, so patch identity is irrelevant);
        traction patches keep their identity via prefix offsets.

        Per-element moduli and a body force cross as the schema's optional
        ``cell_youngs`` / ``cell_poisson`` / ``body_force`` arrays, all empty
        for the scalar single-material path.
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
        num_cells = int(np.asarray(cells).shape[0])
        cell_youngs = _as_cell_array(youngs, num_cells)
        cell_poisson = _as_cell_array(poisson, num_cells)
        scalar_youngs = 0.0 if cell_youngs.size else youngs
        scalar_poisson = 0.0 if cell_poisson.size else poisson
        if body_force is None:
            force = jnp.zeros((0, 3), dtype=jnp.float64)
        else:
            force = jnp.broadcast_to(jnp.asarray(body_force, dtype=jnp.float64), (num_cells, 3))
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
                    "youngs": np.asarray(scalar_youngs, dtype=np.float64),
                    "poisson": np.asarray(scalar_poisson, dtype=np.float64),
                    "cell_youngs": jnp.asarray(cell_youngs, dtype=jnp.float64),
                    "cell_poisson": jnp.asarray(cell_poisson, dtype=jnp.float64),
                    "body_force": force,
                },
            )
            return outputs["displacement"]


def _jaxfem_backend() -> SolverBackend:
    """Factory for the direct jax-fem backend (lazy import keeps the layers apart).

    The implementation lives in :mod:`cadjoint.fem.jaxfem`, which imports
    this module for the ABI; resolving it lazily keeps the dependency
    one-directional.
    """
    from cadjoint.fem.jaxfem import JaxFemBackend

    return JaxFemBackend()


def _calculix_backend() -> SolverBackend:
    """Factory for the CalculiX backend (lazy import keeps ccx optional)."""
    from cadjoint.fem.calculix import CalculixBackend

    return CalculixBackend()


_REGISTRY: dict[str, Callable[[], SolverBackend]] = {
    "jaxfem": _jaxfem_backend,
    "tesseract": TesseractBackend,
    "calculix": _calculix_backend,
}


def __getattr__(name: str) -> Any:
    """Resolve :class:`~cadjoint.fem.jaxfem.JaxFemBackend` for legacy importers.

    ``JaxFemBackend`` was defined here before the solver formulations moved
    to :mod:`cadjoint.fem.jaxfem`; the packaged tesseracts and downstream
    code still import it from this module.  Serving it through PEP 562
    keeps ``from cadjoint.fem.backends import JaxFemBackend`` working
    without a module-level import back into the solver layer (which would
    make the two modules circular).
    """
    if name == "JaxFemBackend":
        from cadjoint.fem.jaxfem import JaxFemBackend

        return JaxFemBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
