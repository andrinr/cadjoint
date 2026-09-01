"""The mesher + solver two-Tesseract chain behind ``gradient_path="tesseract"``.

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
        if self._kind == "thermal":
            scalar = field
        else:
            if metric == "compliance":
                from cadjoint.optimize import _compliance

                result = SimpleNamespace(displacement=field)
                return _compliance(self.study, result, self.mesh, points)
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
    from cadjoint.fem.hexmesh import (
        FaceGroup,
        HexMesh,
        _boundary_face_rows,
        _face_geometry,
    )
    from cadjoint.fem.tetmesh import TetMesh, tet10_mesh, tet_boundary_faces

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
    from cadjoint.fem.tetmesh import TetMesh, tet10_complete_nodes

    indices = selection.resolve(mesh)
    if isinstance(mesh, TetMesh):
        return tet10_complete_nodes(mesh, indices)
    return np.asarray(indices, dtype=np.int32)


def _face_patch(mesh: Any, selection: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Area-integrated patch: spanning node set + exact tet faces (or None)."""
    from cadjoint.fem.hexmesh import faces_from_nodes
    from cadjoint.fem.tetmesh import TetMesh, tet10_face_midsides, tet_faces_from_nodes

    indices = selection.resolve(mesh)
    if isinstance(mesh, TetMesh):
        faces = tet_faces_from_nodes(mesh, indices)
        nodes = np.unique(faces)
        if mesh.edge_parents is not None:
            nodes = np.concatenate([nodes, np.unique(tet10_face_midsides(mesh, faces))])
        return nodes.astype(np.int32), faces.astype(np.int32)
    return np.unique(faces_from_nodes(mesh, indices).nodes).astype(np.int32), None


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

    from cadjoint.fem.study import Dirichlet, Fixed, HeatFlux, ThermalStudy, Traction

    grid = sim_mesh.grid(field)
    lattice = np.asarray(grid.lattice_points())
    samples0 = np.asarray(field(jnp.asarray(lattice)), dtype=np.float64)
    mesh, templates, _element = _discover_mesh(sim_mesh, samples0, grid)
    from cadjoint.optimize import _unresolvable_bc

    problem = _unresolvable_bc(study, mesh)
    if problem is not None:
        raise RuntimeError(f"{problem} on the mesher tesseract's frozen mesh")
    mesher = _tesseract("mesher")

    if isinstance(study, ThermalStudy):
        kind, solver_name, output = "thermal", "thermal_jaxfem", "temperature"
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
            "conductivity": np.float64(study.conductivity),
            "source": np.float64(study.source),
        }
    else:
        kind, solver_name, output = "elastic", "elastic_jaxfem", "displacement"
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
            "youngs": np.float64(study.youngs),
            "poisson": np.float64(study.poisson),
        }
    solver = _tesseract(solver_name)

    def solve(samples):
        meshed = apply_tesseract(mesher, dict(field_values=samples, **templates))
        solved = apply_tesseract(solver, dict(points=meshed["points"], **inputs))
        return meshed["points"], solved[output]

    return FrozenChain(mesh=mesh, lattice=lattice, study=study, _kind=kind, _solve=solve)
