"""The Tesseract chains behind ``gradient_path="tesseract"``/``"tesseract-dc"``.

Two frozen chains live here, differing only in **where the black box is
cut**:

* :func:`freeze_study_chain` (``gradient_path="tesseract"``) wraps the
  whole meshing pipeline in the ``mesher`` tesseract and carries
  ``d(points)/d(field samples)`` with its surface-interpolation VJP.
* :func:`freeze_study_chain_dc` (``gradient_path="tesseract-dc"``) keeps
  dual contouring in JAX — frozen crossing edges riding the **true SDF**
  under clamped Newton projection, QEF through a differentiable linear
  solve, then the same Newton projection ``tetmesh.sdf_to_tet_mesh``
  applies before TetGen — and wraps only TetGen, in the ``tetfill``
  tesseract, whose VJP is an exact pass-through on the vertices ``-Y``
  preserves.  Its frozen interior then follows the boundary through the
  direct path's Laplacian relaxation (:func:`_interior_relaxation`).  This
  is the narrow cut the user asked for: only the tet meshing is a black
  box, the rest is natively differentiable.

:func:`freeze_study_chain` freezes a study's mesh topology through the
packaged mesher tesseract (dual-contour surface + TetGen, or voxelize+snap
for hex) at the current design, resolves the study's boundary conditions on
that frozen mesh, and returns a :class:`FrozenChain` whose
:meth:`~FrozenChain.metric_value` is one traced function

    lattice samples -> mesher tesseract -> solver tesseract -> metric

differentiable end to end: the mesher's surface-interpolation VJP carries
``d(points)/d(samples)`` and the solver tesseract's adjoint carries
``d(field)/d(points)``.  This is the alternative gradient path the
``cadjoint.optimize`` seam selects with ``gradient_path="tesseract"`` — the
DIRECT path (Newton re-projection onto the true SDF) remains the default;
the trade is measured in ``research/tet-vs-hex.md``.

The chain differs from the direct path in one honest way: the mesher
tesseract operates on the *trilinear interpolant* of the lattice samples,
so its frozen mesh is not bit-identical to ``SimMesh.build``'s (which
meshes the true SDF), and its gradient carries only the normal component
of boundary-vertex motion (mesher gauge; see the mesher tesseract's
docstring).  The solver stage is exact: identical solves and adjoints to
the in-process backends (parity 1e-9, ``tests/fem/test_tesseract_tet.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

_TESSERACT_EXTRA_MESSAGE = (
    "tesseract-core / tesseract-jax are not installed. "
    "Install the 'tesseract' extra: pip install cadjoint[tesseract]."
)

#: Mesher tesseract ``element`` codes per SimMesh method.
_ELEMENT_CODES = {"tet4": 0, "hex": 1, "tet10": 2}

_TESSERACTS_DIR = Path(__file__).parent
_LOADED: dict[str, Any] = {}


def _tesseract(name: str):
    """Load a packaged tesseract once per process (kept warm)."""
    if name not in _LOADED:
        try:
            from tesseract_core import Tesseract
        except ImportError as error:
            raise ImportError(_TESSERACT_EXTRA_MESSAGE) from error
        _LOADED[name] = Tesseract.from_tesseract_api(
            str(_TESSERACTS_DIR / name / "tesseract_api.py")
        )
    return _LOADED[name]


@dataclass(frozen=True)
class FrozenChain:
    """A frozen-topology two-tesseract chain for one study.

    Attributes:
        mesh: The frozen mesh (a real ``HexMesh``/``TetMesh``, so node
            selections and ``recompute_*`` fallbacks work on it).
        lattice: The sampling lattice points, ``(L, 3)`` — evaluate the
            (traced) design field here and hand the samples to
            :meth:`metric_value`.
        study: The study whose BCs are frozen into the solver inputs.
    """

    mesh: Any
    lattice: np.ndarray
    study: Any
    _kind: str
    _solve: Callable[[Any], tuple[Any, Any]]

    def metric_value(self, samples: Any, metric: str) -> Any:
        """The study metric as a traced JAX scalar of the lattice samples.

        ``mean``/``max`` mirror ``SimulationResult.mean()/.max()``
        (temperature, resp. guarded displacement magnitude); ``compliance``
        reuses the optimizer's traction-work helper on the traced points.
        """
        import jax.numpy as jnp

        points, field = self._solve(jnp.asarray(samples))
        return _metric_scalar(self.study, self.mesh, self._kind, metric, points, field)


def _metric_scalar(study: Any, mesh: Any, kind: str, metric: str, points: Any, field: Any) -> Any:
    """A study metric from a solved field on (possibly traced) ``points``.

    ``mean``/``max`` mirror ``SimulationResult.mean()/.max()`` (temperature,
    resp. guarded displacement magnitude); ``compliance`` reuses the
    optimizer's traction-work helper.  Shared by both frozen chains.
    """
    import jax.numpy as jnp

    if kind == "thermal":
        scalar = field
    else:
        if metric == "compliance":
            from cadjoint.optimize import _compliance

            return _compliance(study, SimpleNamespace(displacement=field), mesh, points)
        scalar = jnp.sqrt(jnp.sum(field**2, axis=-1) + 1e-30)
    return jnp.mean(scalar) if metric == "mean" else jnp.max(scalar)


def _offsets(sets: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum([len(s) for s in sets], dtype=np.int64)]).astype(np.int32)


def _concat(sets: list[np.ndarray], width: int | None = None) -> np.ndarray:
    if not sets:
        if width is None:
            return np.zeros(0, dtype=np.int32)
        return np.zeros((0, width), dtype=np.int32)
    return np.concatenate([np.asarray(s, dtype=np.int32) for s in sets]).astype(np.int32)


def _discover_mesh(sim_mesh: Any, samples: np.ndarray, grid: Any) -> tuple[Any, dict, int]:
    """Run the mesher tesseract concretely and wrap its output as a mesh.

    Mirrors ``SimMesh.build``'s sharp -> Tikhonov fallback for the tet
    methods.  Returns ``(mesh, templates, element)`` where ``templates``
    are the frozen-topology inputs for the traced mesher call (element and
    sharp flags included).

    Raises:
        RuntimeError: When the mesher rejects the design (the optimizer's
            refreeze fallback catches exactly this).
    """
    from cadjoint.fem.boundary import (
        FaceGroup,
        _boundary_face_rows,
        _face_geometry,
        tet_boundary_faces,
    )
    from cadjoint.fem.hexmesh import HexMesh
    from cadjoint.fem.tetmesh import TetMesh, tet10_mesh

    mesher = _tesseract("mesher")
    method = sim_mesh.method
    element = _ELEMENT_CODES[method]
    # Discovery always runs the corner-level mesher (TET10 promotes the
    # TET4 mesh locally — deterministic match to the element-2 apply).
    discovery_element = 1 if method == "hex" else 0

    def static(sharp: int) -> dict:
        return {
            "origin": np.asarray(grid.origin),
            "spacing": np.asarray(grid.spacing),
            "element": np.int32(discovery_element),
            "sharp": np.int32(sharp),
            "min_ratio": np.float64(1.5),
            "min_dihedral": np.float64(10.0),
        }

    found = error = None
    sharp_used = 1
    for sharp_used in (1, 0) if method != "hex" else (0,):
        try:
            found = mesher.apply(
                dict(
                    field_values=samples,
                    point_ids=np.zeros(0, np.int32),
                    cell_template=np.zeros((0, 8 if method == "hex" else 4), np.int32),
                    num_surface=np.int32(0),
                    **static(sharp_used),
                )
            )
            break
        except Exception as exc:  # the tesseract wraps the TetGen RuntimeError
            error = exc
    if found is None:
        raise RuntimeError(f"mesher tesseract rejected the design: {error}")

    points = np.asarray(found["points"])
    cells = np.asarray(found["cells"]).astype(np.int32)
    mask = np.asarray(found["surface_mask"]).astype(bool)
    max_step = 0.5 * float(np.linalg.norm(grid.spacing))
    if method == "hex":
        boundary = _boundary_face_rows(cells)
        centers, normals = _face_geometry(points, boundary)
        mesh: Any = HexMesh(
            points=points,
            cells=cells,
            boundary_faces={"all": FaceGroup(boundary, centers, normals)},
            base_points=points,
            snap_mask=mask,
            max_step=max_step,
            grid=grid,
        )
    else:
        mesh = TetMesh(
            points=points,
            cells=cells,
            num_surface=int(mask.sum()),
            boundary_tris=tet_boundary_faces(cells),
            base_points=points,
            max_step=max_step,
            grid=grid,
        )
        if method == "tet10":
            mesh = tet10_mesh(mesh)
    nodes_per_cell = int(np.asarray(mesh.cells).shape[1])
    templates = {
        "point_ids": np.arange(mesh.num_points, dtype=np.int32),
        "cell_template": np.zeros((mesh.num_cells, nodes_per_cell), np.int32),
        "num_surface": np.int32(mesh.num_surface if isinstance(mesh, TetMesh) else int(mask.sum())),
        "origin": np.asarray(grid.origin),
        "spacing": np.asarray(grid.spacing),
        "element": np.int32(element),
        "sharp": np.int32(sharp_used),
        "min_ratio": np.float64(1.5),
        "min_dihedral": np.float64(10.0),
    }
    return mesh, templates, element


def _node_patch(mesh: Any, selection: Any) -> np.ndarray:
    """Node-valued patch (Dirichlet / clamp) on the frozen mesh."""
    from cadjoint.fem.boundary import tet10_complete_nodes
    from cadjoint.fem.tetmesh import TetMesh

    indices = selection.resolve(mesh)
    if isinstance(mesh, TetMesh):
        return tet10_complete_nodes(mesh, indices)
    return np.asarray(indices, dtype=np.int32)


def _face_patch(mesh: Any, selection: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Area-integrated patch: spanning node set + exact tet faces (or None)."""
    from cadjoint.fem.boundary import faces_from_nodes, tet10_face_midsides, tet_faces_from_nodes
    from cadjoint.fem.tetmesh import TetMesh

    indices = selection.resolve(mesh)
    if isinstance(mesh, TetMesh):
        faces = tet_faces_from_nodes(mesh, indices)
        nodes = np.unique(faces)
        if mesh.edge_parents is not None:
            nodes = np.concatenate([nodes, np.unique(tet10_face_midsides(mesh, faces))])
        return nodes.astype(np.int32), faces.astype(np.int32)
    return np.unique(faces_from_nodes(mesh, indices).nodes).astype(np.int32), None


def _scalar_property(study: Any, name: str) -> float:
    """A study's material property as a scalar, or a clear refusal.

    The frozen chain is handed a *functionalized* design field, not the scene
    object, so it has no ``material_at`` to sample and cannot serve a study
    that derives its properties from the scene's materials.  The solver
    tesseracts themselves accept a per-element array (``cell_conductivity`` /
    ``cell_youngs`` / ``cell_poisson``); it is only this chain's inputs that
    cannot be built without the material field.

    Args:
        study: The study being frozen.
        name: The property attribute, e.g. ``"conductivity"``.

    Returns:
        The property as a float.

    Raises:
        ValueError: If the study derives the property from its materials.
    """
    value = getattr(study, name)
    if isinstance(value, str):
        raise ValueError(
            f"Study {study.name!r} derives {name!r} from the scene's materials, which the "
            "frozen tesseract chain cannot sample: it is given a functionalized design "
            f"field with no material_at. Set an explicit {name} on the study for the "
            'chain paths, or run the study with gradient_path="direct".'
        )
    return float(value)


def _solver_stage(study: Any, mesh: Any) -> tuple[str, str, str, dict]:
    """Resolve the study's BCs on ``mesh`` into solver-tesseract inputs.

    Shared by both frozen chains (the solver stage is identical; only the
    meshing stage in front of it differs).

    Args:
        study: A ``ThermalStudy`` or ``ElasticStudy``.
        mesh: The frozen mesh its selections resolve on.

    Returns:
        ``(kind, solver_name, output_field, inputs)`` — everything but the
        (traced) ``points`` entry of the solver tesseract's payload.
    """
    from cadjoint.fem.study import Dirichlet, Fixed, HeatFlux, ThermalStudy, Traction

    if isinstance(study, ThermalStudy):
        holds = [
            (_node_patch(mesh, bc.nodes), float(bc.value))
            for bc in study.bcs
            if isinstance(bc, Dirichlet)
        ]
        fluxes = [
            (_face_patch(mesh, bc.nodes), float(bc.flux))
            for bc in study.bcs
            if isinstance(bc, HeatFlux)
        ]
        flux_nodes = [nodes for (nodes, _faces), _ in fluxes]
        flux_faces = [faces for (_nodes, faces), _ in fluxes if faces is not None]
        inputs = {
            "cells": np.asarray(mesh.cells, dtype=np.int32),
            "dirichlet_nodes": _concat([nodes for nodes, _ in holds]),
            "dirichlet_values": np.concatenate(
                [np.full(len(nodes), value, dtype=np.float64) for nodes, value in holds]
            )
            if holds
            else np.zeros(0, dtype=np.float64),
            "flux_nodes": _concat(flux_nodes),
            "flux_offsets": _offsets(flux_nodes),
            "flux_values": np.asarray([value for _, value in fluxes], dtype=np.float64),
            "flux_faces": _concat(flux_faces, width=3),
            "flux_face_offsets": _offsets(flux_faces) if flux_faces else np.zeros(0, np.int32),
            "conductivity": np.float64(_scalar_property(study, "conductivity")),
            "source": np.float64(study.source),
            # Per-element properties do not move with the mesh, so they are a
            # pure pass-through here: an empty array selects the scalar above.
            # Sent explicitly rather than left to the schema's default so the
            # payload names every input the solver tesseract declares — the
            # chain's frozen inputs should not depend on defaulting.
            "cell_conductivity": np.zeros(0, dtype=np.float64),
        }
        return "thermal", "thermal_jaxfem", "temperature", inputs

    clamps = [_node_patch(mesh, bc.nodes) for bc in study.bcs if isinstance(bc, Fixed)]
    tractions = [
        (_face_patch(mesh, bc.nodes), np.asarray(bc.vector, dtype=np.float64))
        for bc in study.bcs
        if isinstance(bc, Traction)
    ]
    traction_nodes = [nodes for (nodes, _faces), _ in tractions]
    traction_faces = [faces for (_nodes, faces), _ in tractions if faces is not None]
    inputs = {
        "cells": np.asarray(mesh.cells, dtype=np.int32),
        "fixed_nodes": np.unique(_concat(clamps)).astype(np.int32),
        "traction_nodes": _concat(traction_nodes),
        "traction_offsets": _offsets(traction_nodes),
        "traction_vectors": np.asarray([vector for _, vector in tractions]).reshape(-1, 3),
        "traction_faces": _concat(traction_faces, width=3),
        "traction_face_offsets": _offsets(traction_faces)
        if traction_faces
        else np.zeros(0, np.int32),
        "youngs": np.float64(_scalar_property(study, "youngs")),
        "poisson": np.float64(_scalar_property(study, "poisson")),
        # Pass-through, for the reason given in the thermal branch above.
        "cell_youngs": np.zeros(0, dtype=np.float64),
        "cell_poisson": np.zeros(0, dtype=np.float64),
        "body_force": np.zeros((0, 3), dtype=np.float64),
    }
    return "elastic", "elastic_jaxfem", "displacement", inputs


def freeze_study_chain(study: Any, sim_mesh: Any, field: Callable[[Any], Any]) -> FrozenChain:
    """Freeze the two-tesseract chain for ``study`` at the current design.

    Args:
        study: A ``ThermalStudy`` or ``ElasticStudy`` (BCs are resolved on
            the frozen mesh, values baked into the solver inputs).
        sim_mesh: The study's ``SimMesh`` — its method picks the mesher
            mode (hex / tet4 / tet10), its grid the sampling lattice.
        field: The design field at the current (concrete) parameters; a
            callable on ``(..., 3)`` points.

    Returns:
        The :class:`FrozenChain`.

    Raises:
        RuntimeError: When the mesher tesseract rejects the design.
        ImportError: Without the ``tesseract`` extra.
    """
    import jax.numpy as jnp

    try:
        from tesseract_jax import apply_tesseract
    except ImportError as error:
        raise ImportError(_TESSERACT_EXTRA_MESSAGE) from error

    grid = sim_mesh.grid(field)
    lattice = np.asarray(grid.lattice_points())
    samples0 = np.asarray(field(jnp.asarray(lattice)), dtype=np.float64)
    mesh, templates, _element = _discover_mesh(sim_mesh, samples0, grid)
    from cadjoint.optimize import _unresolvable_bc

    problem = _unresolvable_bc(study, mesh)
    if problem is not None:
        raise RuntimeError(f"{problem} on the mesher tesseract's frozen mesh")
    mesher = _tesseract("mesher")
    kind, solver_name, output, inputs = _solver_stage(study, mesh)
    solver = _tesseract(solver_name)

    def solve(samples):
        meshed = apply_tesseract(mesher, dict(field_values=samples, **templates))
        solved = apply_tesseract(solver, dict(points=meshed["points"], **inputs))
        return meshed["points"], solved[output]

    return FrozenChain(mesh=mesh, lattice=lattice, study=study, _kind=kind, _solve=solve)


# ── the narrow cut: JAX dual contouring + the tetfill tesseract ──────────────

#: Newton projection steps applied to the DC vertices before TetGen (the
#: same count ``sdf_to_tet_mesh`` / the mesher tesseract use).
_PROJECTION_STEPS = 12
#: TetGen quality bounds, matching ``sdf_to_tet_mesh``'s defaults.
_MIN_RATIO = 1.5
_MIN_DIHEDRAL = 10.0
#: Jacobi-Laplacian sweeps that let the frozen interior (Steiner) nodes
#: follow the boundary, applied in JAX to the tetfill tesseract's returned
#: nodes -- the same count and the same operator
#: ``tetmesh.recompute_tet_points`` uses on the direct path.
_INTERIOR_PASSES = 2


@dataclass(frozen=True)
class FrozenDCChain:
    """A frozen-topology chain whose only black box is TetGen.

    The gradient path is ``design field -> JAX dual contouring (frozen
    crossing anchors projected onto the true SDF, differentiable QEF) ->
    Newton projection -> tetfill tesseract (pass-through VJP) -> interior
    relaxation -> solver tesseract (adjoint) -> metric``.  Unlike
    :class:`FrozenChain` the
    entry point is the *field callable*, not lattice samples: dual
    contouring evaluates the true SDF at its own refined points, which is
    precisely the fidelity the whole-pipeline mesher gives up.

    Attributes:
        mesh: The frozen ``TetMesh`` (BC selections and ``recompute_*``
            fallbacks work on it, as with :class:`FrozenChain`).
        study: The study whose BCs are frozen into the solver inputs.
        surface_faces: The frozen DC triangulation, ``(F, 3)``.
        surface_points: The DC surface vertices at the freeze design,
            ``(V, 3)`` — the leading ``V`` rows of ``mesh.points``.
    """

    mesh: Any
    study: Any
    surface_faces: np.ndarray
    surface_points: np.ndarray
    _kind: str
    _extract: Callable[[Any], Any]
    _solve: Callable[[Any], tuple[Any, Any]]

    def dc_surface(self, field: Callable[[Any], Any]) -> Any:
        """The DC surface vertices at ``field``, as a traced ``(V, 3)`` array."""
        return self._extract(field)

    def metric_value(self, field: Callable[[Any], Any], metric: str) -> Any:
        """The study metric as a traced JAX scalar of the design field.

        Args:
            field: The (possibly traced) SDF callable at the current
                design — a callable on ``(3,)`` points.
            metric: ``"mean"``, ``"max"``, or ``"compliance"``.
        """
        points, solved = self._solve(self._extract(field))
        return _metric_scalar(self.study, self.mesh, self._kind, metric, points, solved)


def _freeze_dc_surface(
    field: Callable[[Any], Any], grid: Any
) -> tuple[np.ndarray, np.ndarray, Callable[[Any], Any]]:
    """Freeze the DC topology and return the differentiable vertex map.

    Runs the :mod:`cadjoint.meshing` pipeline once concretely to fix the
    discrete choices (crossing edge set, manifold cell incidence, quad
    triangulation, and the crossing points on the edges) and returns a
    closure that re-derives the vertex positions from any — traced — SDF
    over that frozen topology: the frozen crossings ride the traced zero
    set by clamped Newton projection, :func:`~cadjoint.meshing.qef_vertices`
    fits one vertex per active cell to the planes they span (a
    differentiable linear solve), and the same clamped Newton projection
    ``sdf_to_tet_mesh`` applies lands it on the surface.

    **Why the crossings are anchored rather than re-solved.**  The obvious
    map re-runs :func:`~cadjoint.meshing.edge_hermite_data` on the frozen
    edge set every traced call.  That is wrong, and it is what used to break
    the starter heat sink at step 5 of its own optimization: the root search
    is bracketed by the *frozen sign pattern at the lattice vertices*, which
    is discrete topology and expires.  Measured on ``scenes/starter.py``,
    **168 of 858 brackets (20%) are already invalid one optimizer step
    later** — the fin faces sit within 0.005 of a lattice plane, so a whole
    plane of vertices changes sign at once.  On an expired edge bisection
    collapses toward an endpoint, so the cell is fitted to a sample that is
    not on the surface, ``qef_vertices``' cell clamp then pins neighbouring
    cells' vertices onto their shared face, and the boundary tets built on
    them go to zero volume (204 degenerate tets at step 2, ``min |vol|``
    1.6e-21) until the FEM system is unsolvable.  Projecting the frozen
    crossings instead needs no bracket at all, is defined for every design,
    and keeps every sample a genuine surface point with a genuine normal —
    so the QEF's plane fit still recovers **tangential** vertex motion, the
    property the bar exhibit in ``research/tet-vs-hex.md`` shows plain
    per-vertex re-projection throws away.

    Sharp-feature placement (:func:`~cadjoint.meshing.sharp_qef_vertices`)
    is deliberately *not* used: its singular-value truncation has no usable
    derivative, and mixing it into the freeze would break the fixed point
    (the frozen mesh's boundary would not equal the traced surface at the
    nominal design).  The Tikhonov QEF differs from it by the
    regularization bias, of order ``regularization x cell size``.

    Returns:
        ``(faces, vertices, extract)`` — the frozen triangulation, the
        concrete vertices at this design, and the traced vertex map.

    Raises:
        RuntimeError: When the field does not cross the grid, or the
            surface is open at the grid boundary (TetGen needs it closed).
    """
    import jax
    import jax.numpy as jnp

    from cadjoint.fem.motion import project_points
    from cadjoint.meshing import (
        dual_faces,
        edge_hermite_data,
        find_crossing_edges,
        manifold_cell_incidence,
        qef_vertices,
        sample_grid,
    )
    from cadjoint.meshing.edge_detection import HermiteData

    values = sample_grid(field, grid)
    edges = find_crossing_edges(values)
    if edges.count == 0:
        raise RuntimeError("the design field does not cross the meshing grid; nothing to mesh")
    incidence = manifold_cell_incidence(edges, grid, values < 0)
    max_step = 0.5 * float(np.linalg.norm(grid.spacing))

    # The frozen crossings: located once, by the bracketed root search, on
    # the concrete design.  From here on they are anchors that ride the
    # traced surface, never re-solved against the expiring bracket.
    frozen = edge_hermite_data(field, grid, edges)
    anchors = jnp.asarray(np.asarray(frozen.points, dtype=np.float64))
    frozen_t = jax.lax.stop_gradient(frozen.t)
    # Edge direction times a fraction of an edge, toward the inside
    # endpoint: the offset edge_hermite_data itself uses to re-evaluate a
    # dead subgradient.  Landing bit-exactly on a wall whose SDF has no
    # gradient there is common, not exotic (epsilon-smoothed norms, polygon
    # boundaries), and without this the QEF loses those planes outright.
    directions = np.eye(3, dtype=np.float64)[np.asarray(edges.axis, dtype=np.int32)] * np.asarray(
        grid.spacing, dtype=np.float64
    )
    inward = jnp.asarray(np.where(edges.start_inside, -1e-3, 1e-3)[:, None] * directions)

    def extract(sdf: Callable[[Any], Any]) -> Any:
        value_and_grad = jax.vmap(jax.value_and_grad(lambda p: jnp.asarray(sdf(p)).reshape(())))
        samples = project_points(sdf, anchors, max_step, steps=_PROJECTION_STEPS)
        residuals, gradients = value_and_grad(samples)
        fallback = jax.vmap(jax.grad(lambda p: jnp.asarray(sdf(p)).reshape(())))(
            jax.lax.stop_gradient(samples) + inward
        )
        degenerate = jnp.sum(jax.lax.stop_gradient(gradients) ** 2, axis=-1) < 1e-12
        gradients = jnp.where(degenerate[:, None], fallback, gradients)
        # ``t`` keeps the frozen crossing parameter for provenance — a
        # projected sample no longer lives on its edge — and nothing
        # qef_vertices reads depends on it.
        hermite = HermiteData(points=samples, gradients=gradients, t=frozen_t, values=residuals)
        vertices, _normals = qef_vertices(hermite, incidence, grid)
        return project_points(sdf, vertices, max_step, steps=_PROJECTION_STEPS)

    vertices = np.asarray(extract(field), dtype=np.float64)
    _quads, faces, skipped = dual_faces(edges, incidence, grid, vertices)
    if skipped:
        raise RuntimeError(
            f"the isosurface crosses the extraction boundary on {skipped} grid edges; "
            "the DC surface is open there and TetGen needs a watertight one — enlarge "
            "the meshing box"
        )
    return faces.astype(np.int32), vertices, extract


def _interior_relaxation(mesh: Any) -> Callable[[Any], Any]:
    """The JAX map that lets a frozen fill's interior follow its boundary.

    The tetfill tesseract holds the interior (Steiner) nodes at their frozen
    positions, so a boundary that has marched away from them leaves the
    straddling tets progressively worse shaped.  This composes
    :func:`~cadjoint.fem.motion.smooth_interior_delta` — a fixed number of
    Jacobi–Laplacian sweeps over the frozen connectivity, boundary pinned —
    onto the tesseract's returned ``nodes``, and re-derives TET10 midsides
    as midpoints of the relaxed corners.

    Deliberately applied **here** rather than inside the tesseract.  The
    tesseract's forward stays a pure gather and its VJP stays the exact
    transpose of one (``tests/fem/test_tetfill.py``, agreement ~1e-16), with
    no smoothing operator to transpose by hand; JAX carries the relaxation's
    own derivative.  It also tightens the chain rather than loosening it:
    the relaxed nodes depend on ``nodes`` only through the preserved
    boundary block, so the cotangent that reaches the tesseract is supported
    exactly where its pass-through is exact, and the solver's interior-node
    sensitivity is *transported to the boundary* instead of being dropped.

    Args:
        mesh: The frozen ``TetMesh`` (TET4 or TET10).

    Returns:
        ``nodes -> nodes``, the identity when there is nothing to relax.
    """
    import jax.numpy as jnp

    from cadjoint.fem.motion import smooth_interior_delta

    corner_count = mesh.num_corner_points
    count = mesh.num_surface
    if _INTERIOR_PASSES <= 0 or corner_count <= count:
        return lambda nodes: nodes
    base = jnp.asarray(np.asarray(mesh.points[:corner_count], dtype=np.float64))
    parents = None if mesh.edge_parents is None else jnp.asarray(mesh.edge_parents)

    def relax(nodes: Any) -> Any:
        corners = base + smooth_interior_delta(mesh, nodes[:count] - base[:count], _INTERIOR_PASSES)
        if parents is None:
            return corners
        return jnp.concatenate([corners, corners[parents].mean(axis=1)], axis=0)

    return relax


def freeze_study_chain_dc(
    study: Any,
    sim_mesh: Any,
    field: Callable[[Any], Any],
    *,
    freeze_interior: bool = True,
) -> FrozenDCChain:
    """Freeze the DC-surface + tetfill + solver chain for ``study``.

    Args:
        study: A ``ThermalStudy`` or ``ElasticStudy`` (BCs are resolved on
            the frozen mesh, values baked into the solver inputs).
        sim_mesh: The study's ``SimMesh``; must declare a tet method
            (``"tet4"``/``"tet10"``) — this chain fills a DC surface, so
            there is no hex mode.
        field: The design field at the current (concrete) parameters; a
            callable on ``(..., 3)`` points.
        freeze_interior: Hold the interior (Steiner) nodes across the
            traced calls, so the tesseract re-evaluates the frozen fill
            instead of re-running TetGen (the default, and the same
            contract ``recompute_tet_points`` honours on the direct path).
            TetGen's Steiner insertion is not continuous in the surface —
            measured: a 1e-4 design step already changes the Steiner count
            on the box bar — so ``False`` re-runs the black box per step
            and is only usable where the topology happens to survive.
            Frozen mode additionally relaxes the held interior toward the
            moving boundary (:func:`_interior_relaxation`), so the two
            modes' derivatives differ there: TetGen-in-the-loop mode drops
            the solver's interior cotangents, frozen mode transports them
            onto the boundary through the relaxation's transpose.

    Returns:
        The :class:`FrozenDCChain`.

    Raises:
        RuntimeError: When the DC surface is unusable or TetGen rejects it
            (the optimizer's refreeze fallback catches exactly this).
        ValueError: On a hex ``SimMesh``.
        ImportError: Without the ``tesseract`` extra.
    """
    try:
        from tesseract_jax import apply_tesseract
    except ImportError as error:
        raise ImportError(_TESSERACT_EXTRA_MESSAGE) from error

    from cadjoint.fem.boundary import tet_boundary_faces
    from cadjoint.fem.tetmesh import TetMesh, tet10_mesh

    method = sim_mesh.method
    if method not in ("tet4", "tet10"):
        raise ValueError(
            f"freeze_study_chain_dc needs a tet SimMesh (method='tet4'/'tet10'), got {method!r}; "
            "the DC chain fills a dual-contour surface, which has no hex counterpart."
        )
    grid = sim_mesh.grid(field)
    faces, vertices, extract = _freeze_dc_surface(field, grid)

    tetfill = _tesseract("tetfill")
    static = {
        "triangles": faces,
        "min_ratio": np.float64(_MIN_RATIO),
        "min_dihedral": np.float64(_MIN_DIHEDRAL),
    }
    try:
        # Discovery always runs the corner-level fill; TET10 promotion is
        # deterministic and reproduced locally so the frozen mesh matches
        # the element-2 apply node for node.
        found = tetfill.apply(
            dict(
                points=vertices,
                element=np.int32(0),
                interior_points=np.zeros((0, 3), np.float64),
                node_ids=np.zeros(0, np.int32),
                cell_template=np.zeros((0, 4), np.int32),
                **static,
            )
        )
    except Exception as error:  # the tesseract wraps TetGen's RuntimeError
        raise RuntimeError(f"tetfill tesseract rejected the DC surface: {error}") from error

    points = np.asarray(found["nodes"], dtype=np.float64)
    cells = np.asarray(found["cells"]).astype(np.int32)
    mesh: Any = TetMesh(
        points=points,
        cells=cells,
        num_surface=int(vertices.shape[0]),
        boundary_tris=tet_boundary_faces(cells),
        base_points=points,
        max_step=0.5 * float(np.linalg.norm(grid.spacing)),
        grid=grid,
    )
    if method == "tet10":
        mesh = tet10_mesh(mesh)

    from cadjoint.optimize import _unresolvable_bc

    problem = _unresolvable_bc(study, mesh)
    if problem is not None:
        raise RuntimeError(f"{problem} on the DC chain's frozen mesh")

    cells_out = np.asarray(mesh.cells, dtype=np.int32)
    templates = {
        "element": np.int32(_ELEMENT_CODES[method]),
        "node_ids": np.arange(mesh.num_points, dtype=np.int32),
        # Frozen-fill mode passes the real connectivity (and the held
        # interior); TetGen-in-the-loop mode passes shapes only.
        "cell_template": cells_out if freeze_interior else np.zeros(cells_out.shape, np.int32),
        "interior_points": points[int(vertices.shape[0]) :]
        if freeze_interior
        else np.zeros((0, 3), np.float64),
        **static,
    }
    kind, solver_name, output, inputs = _solver_stage(study, mesh)
    solver = _tesseract(solver_name)
    relax = _interior_relaxation(mesh) if freeze_interior else None

    def solve(surface_points):
        filled = apply_tesseract(tetfill, dict(points=surface_points, **templates))
        nodes = filled["nodes"] if relax is None else relax(filled["nodes"])
        solved = apply_tesseract(solver, dict(points=nodes, **inputs))
        return nodes, solved[output]

    return FrozenDCChain(
        mesh=mesh,
        study=study,
        surface_faces=faces,
        surface_points=vertices,
        _kind=kind,
        _extract=extract,
        _solve=solve,
    )
