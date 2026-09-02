"""DC surface -> tetrahedral volume mesh (TET4 / TET10).

What belongs here: building a :class:`TetMesh` and nothing more.  The
differentiable dual-contour surface from
:func:`cadjoint.meshing.extract_mesh` is handed to TetGen as a piecewise
linear complex with boundary splitting disabled (``-Y``), so every DC
surface vertex survives *verbatim* as the first ``num_surface`` nodes of
the volume mesh and only interior Steiner vertices are added.  That makes
the frozen-topology doctrine carry over unchanged:

- connectivity, the surface/interior split, and BC node sets are frozen
  per extraction;
- the boundary vertices are exactly the DC vertices, which are already
  differentiable w.r.t. design parameters (the meshing pipeline's
  contract) — :func:`~cadjoint.fem.motion.recompute_tet_points` re-derives
  them per candidate design by Newton re-projection onto the traced SDF
  (the same projection :func:`~cadjoint.fem.motion.recompute_points` uses
  on hexes);
- interior Steiner vertices stay frozen: their discrete sensitivity is a
  mesh-motion term that vanishes in the continuous limit (Hadamard's
  shape-derivative structure — only normal boundary motion changes the
  shape), and is measured to be orders below the boundary sensitivity in
  ``research/tet-vs-hex.md``.  An optional Laplacian pass
  (:func:`~cadjoint.fem.motion.smooth_interior_delta`) propagates boundary
  motion inward differentiably for larger design steps.

Also here: the straight-sided TET4 -> TET10 promotion
(:func:`tet10_from_tet4`, :func:`tet10_mesh`), whose midside nodes are
edge midpoints of the — possibly traced — corner positions, so gradients
flow through them while faces stay straight-sided.

What does *not* belong here, and where it lives instead:

- element topology tables — :mod:`cadjoint.fem.elements`
- quality metrics (``tet_volumes``, ``tet_radius_ratios``,
  ``tet_aspect_ratios``) — :mod:`cadjoint.fem.quality`
- boundary triangles, node-set selection and TET10 midside completion —
  :mod:`cadjoint.fem.boundary`
- differentiable node motion — :mod:`cadjoint.fem.motion`
- the TET4/TET10 solves (``tet_thermal_solve``, ``tet_elastic_solve``) —
  :mod:`cadjoint.fem.jaxfem`, alongside the HEX8 ones
- stress recovery and load work — :mod:`cadjoint.fem.postprocess`

Those modules are shared with :mod:`cadjoint.fem.hexmesh`, so the two
element families are layered identically.  The mesh-layer names are
re-exported below; the solver- and postprocessing-layer names resolve
through :func:`__getattr__` so this module keeps no import edge *up* into
the solver layer (the ``tetfill`` / ``mesher`` tesseract images depend on
it without shipping jax-fem).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from cadjoint.fem.boundary import (
    FaceGroup,
    _tri_geometry,
    tet10_complete_nodes,
    tet10_face_midsides,
    tet_boundary_faces,
    tet_faces_from_nodes,
)
from cadjoint.fem.elements import TET10_EDGES
from cadjoint.fem.motion import project_points, recompute_tet_points, smooth_interior_delta
from cadjoint.fem.quality import tet_aspect_ratios, tet_radius_ratios, tet_volumes
from cadjoint.meshing import GridSpec, extract_mesh
from cadjoint.meshing.diagnostics import self_intersections

__all__ = [
    "TetMesh",
    "recompute_tet_points",
    "refine_resolution",
    "sdf_to_tet_mesh",
    "smooth_interior_delta",
    "surface_to_tet_mesh",
    "tet10_complete_nodes",
    "tet10_face_midsides",
    "tet10_from_tet4",
    "tet10_mesh",
    "tet_aspect_ratios",
    "tet_boundary_faces",
    "tet_faces_from_nodes",
    "tet_radius_ratios",
    "tet_volumes",
]

#: Deprecated alias of :data:`cadjoint.fem.elements.TET10_EDGES`, kept
#: because callers have long reached for it through this module.
_TET10_EDGES = TET10_EDGES

#: Names that moved to the solver / postprocessing layer, resolved lazily
#: by :func:`__getattr__` so importing a mesh does not import a solver.
_MOVED = {
    "tet_elastic_solve": "cadjoint.fem.jaxfem",
    "tet_thermal_solve": "cadjoint.fem.jaxfem",
    "load_work_quads": "cadjoint.fem.postprocess",
    "load_work_tri6": "cadjoint.fem.postprocess",
    "load_work_tris": "cadjoint.fem.postprocess",
    "tet_von_mises": "cadjoint.fem.postprocess",
}

_TETGEN_MESSAGE = (
    "tetgen is not installed (PyPI wheels exist for macOS arm64 / Python 3.14): pip install tetgen"
)


def __getattr__(name: str) -> Any:
    """Resolve names that moved out of this module (see :data:`_MOVED`).

    The TET4/TET10 solves and the derived-quantity helpers used to live
    here; they now sit in the solver and postprocessing layers alongside
    their hex counterparts.  Serving them through PEP 562 keeps every
    ``from cadjoint.fem.tetmesh import ...`` call site working while
    leaving this module's own imports pointing strictly *downwards*.
    """
    module = _MOVED.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True)
class TetMesh:
    """A TET4/TET10 volume mesh whose boundary vertices are DC surface vertices.

    Duck-compatible with :class:`~cadjoint.fem.selection.NodeSelection`
    resolution (``num_points`` / ``points`` / ``all_boundary_faces`` /
    ``grid``), so ``Nodes`` selections resolve on tet meshes unchanged
    (selections resolve to the *corner* boundary nodes; TET10 midside
    completion happens at BC assembly via
    :func:`~cadjoint.fem.boundary.tet10_complete_nodes`).

    Attributes:
        points: Vertex positions, ``(N, 3)`` float64.  The first
            ``num_surface`` rows are the DC surface vertices verbatim,
            followed by interior Steiner vertices; a TET10 mesh appends
            the shared midside nodes after all corner vertices.
        cells: Connectivity (meshio ``tetra``/``tetra10`` order, positive
            volumes), ``(T, 4)`` or ``(T, 10)`` int32.
        num_surface: Number of leading DC surface (corner) vertices.
        boundary_tris: Outward-oriented boundary corner triangles (faces
            used by exactly one tet), ``(M, 3)`` int64.
        base_points: Frozen nominal positions, ``(N, 3)`` — the anchor for
            :func:`~cadjoint.fem.motion.recompute_tet_points` (for TET10,
            midside rows are the midpoints of the corner base positions).
        max_step: Newton re-projection displacement clamp.
        grid: The DC sampling grid the surface came from (``None`` when
            built from a raw surface).
        edge_parents: ``None`` for TET4.  For TET10 the ``(E, 2)`` corner
            index pairs whose midpoints the appended midside nodes are
            (row ``k`` describes node ``num_corner_points + k``; rows are
            sorted pairs in lexicographic order).
        refinement: What the automatic refinement ladder of
            :func:`sdf_to_tet_mesh` had to do to produce this mesh, or
            None when the mesh did not come through it (a raw surface, or
            a chain/tesseract fill).  See
            :func:`sdf_to_tet_mesh` for the record's shape.
    """

    points: np.ndarray
    cells: np.ndarray
    num_surface: int
    boundary_tris: np.ndarray
    base_points: np.ndarray
    max_step: float
    grid: GridSpec | None = None
    edge_parents: np.ndarray | None = None
    refinement: dict[str, Any] | None = None

    @property
    def num_points(self) -> int:
        """Number of vertices (including TET10 midside nodes)."""
        return int(self.points.shape[0])

    @property
    def num_cells(self) -> int:
        """Number of tetrahedra."""
        return int(self.cells.shape[0])

    @property
    def order(self) -> int:
        """Element order: 1 for TET4, 2 for TET10."""
        return 1 if self.edge_parents is None else 2

    @property
    def ele_type(self) -> str:
        """The jax-fem element type string (``"TET4"`` / ``"TET10"``)."""
        return "TET4" if self.edge_parents is None else "TET10"

    @property
    def num_corner_points(self) -> int:
        """Number of corner vertices (excludes TET10 midside nodes)."""
        if self.edge_parents is None:
            return self.num_points
        return self.num_points - int(self.edge_parents.shape[0])

    def all_boundary_faces(self) -> FaceGroup:
        """Boundary triangles as a :class:`FaceGroup` (nodes shaped ``(M, 3)``)."""
        centers, normals = _tri_geometry(self.points, self.boundary_tris)
        return FaceGroup(nodes=self.boundary_tris, centers=centers, normals=normals)


def surface_to_tet_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    base_vertices: np.ndarray | None = None,
    grid: GridSpec | None = None,
    min_ratio: float = 1.5,
    min_dihedral: float = 10.0,
) -> TetMesh:
    """Tet-mesh the inside of a watertight triangulated surface with TetGen.

    Runs TetGen in PLC mode with boundary splitting disabled (``-Y``,
    ``nobisect``) plus radius-edge and dihedral quality bounds for the
    interior, so the input surface triangulation is preserved exactly: the
    first ``len(vertices)`` output nodes are the input vertices verbatim
    and all added Steiner vertices are strictly interior.

    Args:
        vertices: Surface vertex positions, ``(V, 3)``.
        faces: Watertight triangle connectivity, ``(F, 3)``.
        base_vertices: Frozen Newton starting positions for the surface
            vertices (defaults to ``vertices``).  :func:`sdf_to_tet_mesh`
            passes the raw DC vertices here while meshing their projected
            positions, so :func:`~cadjoint.fem.motion.recompute_tet_points`
            reproduces the meshed geometry exactly at the nominal design.
        grid: Optional DC sampling grid, recorded on the mesh and used to
            derive the re-projection clamp (half the cell diagonal, like
            the hex mesher).  Without it the clamp falls back to half the
            median surface edge length.
        min_ratio: TetGen radius-edge quality bound (``-q``); smaller is
            stricter.
        min_dihedral: TetGen minimum dihedral angle bound in degrees.

    Returns:
        The :class:`TetMesh`.

    Raises:
        ImportError: If tetgen is not installed.
        RuntimeError: If TetGen rejects the surface (e.g. the DC surface
            self-intersects at coarse resolutions — re-extract at higher
            resolution or with ``sharp=False``).
    """
    try:
        import tetgen
    except ImportError as error:
        raise ImportError(_TETGEN_MESSAGE) from error

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    generator = tetgen.TetGen(vertices, faces)
    try:
        nodes, cells, *_ = generator.tetrahedralize(
            plc=True,
            nobisect=True,
            quality=True,
            minratio=float(min_ratio),
            mindihedral=float(min_dihedral),
            quiet=True,
        )
    except RuntimeError as error:
        raise RuntimeError(f"{_TETGEN_PREFIX}{error}. {_SINGLE_GRID_ADVICE}") from error
    nodes = np.asarray(nodes, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int32)
    count = vertices.shape[0]
    if nodes.shape[0] < count or not np.allclose(nodes[:count], vertices, atol=1e-12):
        raise RuntimeError(
            "TetGen did not preserve the input surface vertices as its leading nodes; "
            "the frozen-topology contract needs nobisect (-Y) preservation."
        )
    volumes = tet_volumes(nodes, cells)
    if np.any(volumes <= 0.0):
        raise RuntimeError(f"TetGen produced {int((volumes <= 0).sum())} non-positive tets.")
    # Positive but numerically zero volumes (observed down to 1e-20 on
    # borderline DC surfaces) make the stiffness singular for every direct
    # solver; reject them here so callers get the designed remesh-at-a-
    # different-resolution error instead of a garbage solve.
    degenerate_threshold = 1e-9 * float(np.median(volumes))
    if float(volumes.min()) < degenerate_threshold:
        raise RuntimeError(
            f"TetGen produced numerically degenerate tets (min volume {volumes.min():.2e} "
            f"vs median {np.median(volumes):.2e}); the surface is borderline at this "
            "resolution — re-extract on a different grid or with sharp=False."
        )
    if grid is not None:
        max_step = 0.5 * float(np.linalg.norm(grid.spacing))
    else:
        edges = nodes[faces[:, 1]] - nodes[faces[:, 0]]
        max_step = 0.5 * float(np.median(np.linalg.norm(edges, axis=1)))
    base_points = nodes.copy()
    if base_vertices is not None:
        base_vertices = np.asarray(base_vertices, dtype=np.float64)
        if base_vertices.shape != vertices.shape:
            raise ValueError("base_vertices must match vertices in shape.")
        base_points[:count] = base_vertices
    return TetMesh(
        points=nodes,
        cells=cells,
        num_surface=count,
        boundary_tris=tet_boundary_faces(cells),
        base_points=base_points,
        max_step=max_step,
        grid=grid,
    )


#: Prefix every TetGen failure carries, whether it came from one grid or
#: from the whole refinement ladder.  Callers match on it, so both paths
#: keep it and neither repeats it.
_TETGEN_PREFIX = "TetGen rejected the surface: "

#: What :func:`surface_to_tet_mesh` advises when it has only the one grid
#: it was handed.  :func:`sdf_to_tet_mesh` has already tried that advice by
#: the time it fails, so it strips this sentence and gives its own.
_SINGLE_GRID_ADVICE = (
    "DC surfaces can self-intersect at coarse resolutions; re-extract on a "
    "finer grid or with sharp=False."
)

#: Grid scale factors the refinement ladder in :func:`sdf_to_tet_mesh`
#: walks after the declared resolution, in order.  1.5x is the smallest
#: step that reliably moves a feature sitting on ~1.3 cells past two
#: cells, and 2.25x (its square) is the second rung.
_REFINEMENT_FACTORS = (1.5, 2.25)

#: Non-adjacent triangle pairs the pre-TetGen diagnostic samples, as a
#: multiple of the surface's triangle count and clamped to this ceiling.
#: The check is a *sampled* one (see
#: :func:`cadjoint.meshing.diagnostics.self_intersections`): a hit proves
#: the surface is folded, a miss proves nothing.
#:
#: It does not buy speed.  Measured on the end cap's housing (2026-09-02),
#: it costs 0.12-0.16 s per rung at 145k-200k pairs -- about 1% of the
#: ~15 s the rung's extraction and projection cost, but *more* than the
#: 0.06-0.16 s TetGen itself takes to reject a folded surface.  What it
#: buys is the diagnosis: the rung is recorded as ``"self-intersecting"``
#: with a fold count instead of as an opaque TetGen string, and the same
#: number is what tells a caller that refining is the wrong answer.
_DIAGNOSTIC_PAIRS_PER_TRIANGLE = 32
_DIAGNOSTIC_PAIRS_CEILING = 200_000


def refine_resolution(resolution: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Scale a per-axis cell count by ``factor``, rounding each axis up.

    Args:
        resolution: Cells per axis.
        factor: Scale factor; values below 1 shrink the grid.

    Returns:
        The scaled cell counts, each at least 1.
    """
    return tuple(max(1, int(math.ceil(count * factor))) for count in resolution)  # type: ignore[return-value]


def _rescaled_grid(grid: GridSpec, cells: tuple[int, int, int]) -> GridSpec:
    """The same sampled box as ``grid``, re-diced into ``cells`` cells per axis."""
    extent = [span * count for span, count in zip(grid.spacing, grid.cells)]
    spacing = tuple(float(span / count) for span, count in zip(extent, cells))
    return GridSpec(origin=grid.origin, spacing=spacing, cells=tuple(int(c) for c in cells))


def _bare_reason(message: str) -> str:
    """A TetGen failure stripped of the wrapper the ladder re-adds itself.

    Args:
        message: The message :func:`surface_to_tet_mesh` raised.

    Returns:
        Just what TetGen objected to, without the shared prefix, without
        the single-grid advice the ladder supersedes, and without the
        trailing period the ladder's own sentence supplies.
    """
    reason = message.removeprefix(_TETGEN_PREFIX).strip()
    return reason.removesuffix(_SINGLE_GRID_ADVICE).strip().rstrip(".")


def _count_self_intersections(vertices: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    """Sampled self-intersection count of a surface, sized to the surface.

    Wraps :func:`cadjoint.meshing.diagnostics.self_intersections` with a
    pair budget proportional to the triangle count (see
    :data:`_DIAGNOSTIC_PAIRS_PER_TRIANGLE`).

    Args:
        vertices: Surface vertex positions ``(V, 3)``.
        faces: Triangle connectivity ``(F, 3)``.

    Returns:
        ``{"count": int, "tested": int}`` — intersecting pairs found, and
        non-adjacent pairs actually tested.
    """
    budget = min(
        _DIAGNOSTIC_PAIRS_CEILING,
        max(4096, _DIAGNOSTIC_PAIRS_PER_TRIANGLE * int(faces.shape[0])),
    )
    # ``self_intersections`` only reads ``.vertices`` / ``.faces``, so the
    # surface under test need not be a dual_contouring ``Mesh`` (here it
    # is the *projected* surface, which is what TetGen actually sees).
    report = self_intersections(SimpleNamespace(vertices=vertices, faces=faces), pairs=budget)
    return {"count": int(report["count"]), "tested": int(report["tested"])}


def sdf_to_tet_mesh(
    sdf: Callable[[Any], Any],
    grid: GridSpec,
    *,
    sharp: bool = True,
    min_ratio: float = 1.5,
    min_dihedral: float = 10.0,
    max_refinements: int = 2,
) -> TetMesh:
    """Extract the DC surface of ``sdf`` on ``grid`` and tet-mesh its inside.

    The DC vertices are Newton-projected onto the zero set *before*
    tetrahedralization (QEF vertices sit near but not exactly on the
    surface), so the meshed boundary is the projected surface and
    :func:`~cadjoint.fem.motion.recompute_tet_points` — which runs the
    identical projection from the frozen raw DC positions — reproduces
    ``mesh.points`` exactly at the nominal design.  Without this,
    re-projection at solve time would move the boundary of an
    already-meshed volume and collapse sliver tets.

    **Automatic refinement.**  Tets need a finer grid than hexes on thin
    features: the hex mesher only has to decide in/out per cell, while
    TetGen needs the DC surface to be a valid PLC, and a wall thinner than
    about two cells makes dual contouring fold that surface over itself.
    So this function walks a ladder of grids over the same box — the
    declared resolution, then ``x1.5`` and ``x2.25`` (rounded up per axis,
    see :data:`_REFINEMENT_FACTORS`) — and at each rung tries exact
    sharp-feature placement first and the more robust Tikhonov placement
    second, exactly as a caller would by hand.  Before each TetGen call
    the projected surface goes through the sampled self-intersection
    diagnostic, so a rung it catches is recorded as folded, with a count,
    rather than as an opaque TetGen string (it is a sampled check, and
    cheap, but it is not a speed-up -- see
    :data:`_DIAGNOSTIC_PAIRS_PER_TRIANGLE`).  The first rung that TetGen
    accepts wins.

    Args:
        sdf: Signed distance field callable on ``(..., 3)`` points.
        grid: DC sampling lattice (must fully contain the surface); its
            box is held fixed while the ladder re-dices it.
        sharp: Try exact sharp-feature vertex placement at each rung
            (``False`` uses only the Tikhonov QEF placement, which is more
            robust against self-intersections at coarse grids).
        min_ratio: TetGen radius-edge quality bound.
        min_dihedral: TetGen minimum dihedral angle bound in degrees.
        max_refinements: How many refinement rungs to try after the
            declared resolution (0 restores the pre-refinement behaviour
            of failing at the declared grid).

    Returns:
        The :class:`TetMesh`; its first ``num_surface`` points are the
        projected DC surface vertices, its ``base_points`` hold the raw
        DC positions the projection restarts from, its ``grid`` is the
        rung that succeeded, and its
        :attr:`~TetMesh.refinement` records the ladder::

            {"declared": (26, 26, 13),      # the resolution asked for
             "used": (39, 39, 20),          # the resolution that worked
             "factor": 1.5,                 # scale of the winning rung
             "refined": True,               # used != declared
             "attempts": [                  # every rung/placement tried
                 {"resolution": (26, 26, 13), "factor": 1.0, "sharp": True,
                  "self_intersections": 3, "pairs_tested": 146432,
                  "outcome": "self-intersecting"},
                 ...
                 {"resolution": (39, 39, 20), "factor": 1.5, "sharp": True,
                  "self_intersections": 0, "pairs_tested": 331264,
                  "outcome": "meshed"}]}

        ``outcome`` is ``"meshed"``, ``"self-intersecting"`` (the
        diagnostic fired, TetGen was not run) or ``"rejected"`` (TetGen
        ran and refused; the attempt also carries ``"error"``).

    Raises:
        RuntimeError: If no rung of the ladder produces a mesh.  The
            message names the declared and the finest attempted
            resolution and the thinnest-feature heuristic.
    """
    declared = tuple(int(count) for count in grid.cells)
    factors = (1.0, *_REFINEMENT_FACTORS[: max(0, int(max_refinements))])
    placements = (True, False) if sharp else (False,)
    attempts: list[dict[str, Any]] = []
    last_error = ""
    for factor in factors:
        cells = declared if factor == 1.0 else refine_resolution(declared, factor)
        level = grid if cells == tuple(grid.cells) else _rescaled_grid(grid, cells)
        for placement in placements:
            surface = extract_mesh(sdf, level, sharp=placement)
            raw = np.asarray(surface.vertices, dtype=np.float64)
            faces = np.asarray(surface.faces, dtype=np.int64).reshape((-1, 3))
            max_step = 0.5 * float(np.linalg.norm(level.spacing))
            projected = np.asarray(project_points(sdf, raw, max_step), dtype=np.float64)
            folds = _count_self_intersections(projected, faces)
            attempt: dict[str, Any] = {
                "resolution": cells,
                "factor": float(factor),
                "sharp": bool(placement),
                "self_intersections": folds["count"],
                "pairs_tested": folds["tested"],
            }
            attempts.append(attempt)
            if folds["count"] > 0:
                attempt["outcome"] = "self-intersecting"
                last_error = (
                    f"the surface self-intersects ({folds['count']} folded triangle "
                    f"pairs found in {folds['tested']} sampled)"
                )
                continue
            try:
                mesh = surface_to_tet_mesh(
                    projected,
                    faces,
                    base_vertices=raw,
                    grid=level,
                    min_ratio=min_ratio,
                    min_dihedral=min_dihedral,
                )
            except RuntimeError as error:
                attempt["outcome"] = "rejected"
                attempt["error"] = str(error)
                last_error = _bare_reason(str(error))
                continue
            attempt["outcome"] = "meshed"
            return replace(
                mesh,
                refinement={
                    "declared": declared,
                    "used": cells,
                    "factor": float(factor),
                    "refined": cells != declared,
                    "attempts": attempts,
                },
            )
    finest = attempts[-1]["resolution"] if attempts else declared
    raise RuntimeError(
        f"{_TETGEN_PREFIX}{last_error}. The surface stays self-intersecting "
        f"up to {finest} (declared {declared}, refined "
        f"{' and '.join(f'x{factor:g}' for factor in factors[1:]) or 'not at all'}); the "
        "part likely has features thinner than two cells at the declared resolution — "
        "raise the declared resolution or use method='hex'."
    )


def tet10_from_tet4(
    points: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Promote a TET4 mesh to straight-sided TET10 (meshio ``tetra10`` order).

    Midside nodes are shared through unique undirected edges and appended
    after the corner nodes, so corner indices are unchanged.

    Args:
        points: TET4 vertex positions, ``(N, 3)``.
        cells: TET4 connectivity, ``(T, 4)``.

    Returns:
        ``(points10, cells10, edge_parents)`` where ``points10`` is
        ``(N + E, 3)``, ``cells10`` is ``(T, 10)`` int32, and
        ``edge_parents`` is ``(E, 2)`` — the corner indices whose midpoint
        each appended node is (the hook for differentiable recomputation:
        ``points10 = concat(corners, corners[parents].mean(1))``).
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    cell_edges = np.sort(cells[:, TET10_EDGES], axis=2).reshape(-1, 2)  # (T*6, 2)
    unique_edges, inverse = np.unique(cell_edges, axis=0, return_inverse=True)
    midside = unique_edges.shape[0]
    midpoints = points[unique_edges].mean(axis=1)
    cells10 = np.concatenate([cells, points.shape[0] + inverse.reshape(-1, 6)], axis=1).astype(
        np.int32
    )
    points10 = np.concatenate([points, midpoints], axis=0)
    assert points10.shape[0] == points.shape[0] + midside
    return points10, cells10, unique_edges


def tet10_mesh(mesh: TetMesh) -> TetMesh:
    """Promote a TET4 :class:`TetMesh` to a straight-sided TET10 one.

    Shared midside nodes are appended after all corner vertices (via
    :func:`tet10_from_tet4`); the surface/interior split, the boundary
    corner triangles, and the frozen-topology contract carry over —
    :func:`~cadjoint.fem.motion.recompute_tet_points` re-projects the
    corner surface vertices and rebuilds midsides as traced corner
    midpoints, reproducing ``points`` exactly at the nominal design.

    Args:
        mesh: A TET4 mesh from :func:`sdf_to_tet_mesh`.

    Returns:
        The TET10 :class:`TetMesh` (``edge_parents`` set).

    Raises:
        ValueError: If ``mesh`` is already quadratic.
    """
    if mesh.edge_parents is not None:
        raise ValueError("mesh is already a TET10 mesh.")
    points10, cells10, parents = tet10_from_tet4(mesh.points, mesh.cells)
    base10 = np.concatenate([mesh.base_points, mesh.base_points[parents].mean(axis=1)], axis=0)
    return TetMesh(
        points=points10,
        cells=cells10,
        num_surface=mesh.num_surface,
        boundary_tris=mesh.boundary_tris,
        base_points=base10,
        max_step=mesh.max_step,
        grid=mesh.grid,
        edge_parents=parents,
        refinement=mesh.refinement,
    )
