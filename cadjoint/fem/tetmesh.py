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

from dataclasses import dataclass
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

__all__ = [
    "TetMesh",
    "recompute_tet_points",
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
    """

    points: np.ndarray
    cells: np.ndarray
    num_surface: int
    boundary_tris: np.ndarray
    base_points: np.ndarray
    max_step: float
    grid: GridSpec | None = None
    edge_parents: np.ndarray | None = None

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
        raise RuntimeError(
            f"TetGen rejected the surface: {error}. DC surfaces can self-intersect at "
            "coarse resolutions; re-extract on a finer grid or with sharp=False."
        ) from error
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


def sdf_to_tet_mesh(
    sdf: Callable[[Any], Any],
    grid: GridSpec,
    *,
    sharp: bool = True,
    min_ratio: float = 1.5,
    min_dihedral: float = 10.0,
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

    Args:
        sdf: Signed distance field callable on ``(..., 3)`` points.
        grid: DC sampling lattice (must fully contain the surface).
        sharp: Use exact sharp-feature vertex placement for the surface
            (``False`` falls back to Tikhonov QEF placement, which is
            more robust against self-intersections at coarse grids).
        min_ratio: TetGen radius-edge quality bound.
        min_dihedral: TetGen minimum dihedral angle bound in degrees.

    Returns:
        The :class:`TetMesh`; its first ``num_surface`` points are the
        projected DC surface vertices, its ``base_points`` hold the raw
        DC positions the projection restarts from.
    """
    surface = extract_mesh(sdf, grid, sharp=sharp)
    raw = np.asarray(surface.vertices, dtype=np.float64)
    max_step = 0.5 * float(np.linalg.norm(grid.spacing))
    projected = np.asarray(project_points(sdf, raw, max_step), dtype=np.float64)
    return surface_to_tet_mesh(
        projected,
        np.asarray(surface.faces),
        base_vertices=raw,
        grid=grid,
        min_ratio=min_ratio,
        min_dihedral=min_dihedral,
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
    )
