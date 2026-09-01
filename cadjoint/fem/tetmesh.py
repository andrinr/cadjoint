"""Prototype: DC surface -> tetrahedral volume mesh -> jax-fem TET4/TET10.

Research prototype for the "dual contouring, then volume mesh" route (see
``research/tet-vs-hex.md``): the differentiable dual-contour surface from
:func:`cadjoint.meshing.extract_mesh` is handed to TetGen as a piecewise
linear complex with boundary splitting disabled (``-Y``), so every DC
surface vertex survives *verbatim* as the first ``num_surface`` nodes of
the volume mesh and only interior Steiner vertices are added.  That makes
the frozen-topology doctrine carry over unchanged:

- connectivity, the surface/interior split, and BC node sets are frozen
  per extraction;
- the boundary vertices are exactly the DC vertices, which are already
  differentiable w.r.t. design parameters (the meshing pipeline's
  contract) — :func:`recompute_tet_points` re-derives them per candidate
  design by Newton re-projection onto the traced SDF (mirroring
  :func:`cadjoint.fem.hexmesh.recompute_points`);
- interior Steiner vertices stay frozen: their discrete sensitivity is a
  mesh-motion term that vanishes in the continuous limit (Hadamard's
  shape-derivative structure — only normal boundary motion changes the
  shape), and is measured to be orders below the boundary sensitivity in
  the research note.  An optional Laplacian pass propagates boundary
  motion inward differentiably for larger design steps.

The elastic solve mirrors :meth:`cadjoint.fem.backends.JaxFemBackend.
elastic` with the element type opened up to jax-fem's ``TET4``/``TET10``
(quadratic tets via :func:`tet10_from_tet4`; midside nodes are edge
midpoints of the — possibly traced — corner positions, so gradients flow
through them, but faces stay straight-sided).  This module deliberately
does not touch the production backend registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cadjoint.fem.backends import ElasticBCs, _membership_location, _require_jax_fem, _x64_scope
from cadjoint.fem.hexmesh import FaceGroup, project_points
from cadjoint.meshing import GridSpec, extract_mesh

__all__ = [
    "TetMesh",
    "load_work_quads",
    "load_work_tri6",
    "load_work_tris",
    "recompute_tet_points",
    "sdf_to_tet_mesh",
    "surface_to_tet_mesh",
    "tet10_from_tet4",
    "tet_aspect_ratios",
    "tet_boundary_faces",
    "tet_elastic_solve",
    "tet_faces_from_nodes",
    "tet_radius_ratios",
    "tet_volumes",
]

_TETGEN_MESSAGE = (
    "tetgen is not installed (PyPI wheels exist for macOS arm64 / Python 3.14): pip install tetgen"
)

# The four triangular faces of a positive-volume tet (v0, v1, v2, v3),
# each listed with outward orientation (face i is opposite vertex i).
_TET_FACES = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)], dtype=np.int64)

# The six edges of a tet in meshio ``tetra10`` midside order: nodes 4..9
# are the midpoints of edges (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
_TET10_EDGES = np.array([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)], dtype=np.int64)


@dataclass(frozen=True)
class TetMesh:
    """A TET4 volume mesh whose boundary vertices are DC surface vertices.

    Duck-compatible with :class:`~cadjoint.fem.selection.NodeSelection`
    resolution (``num_points`` / ``points`` / ``all_boundary_faces`` /
    ``grid``), so ``Nodes`` selections resolve on tet meshes unchanged.

    Attributes:
        points: Vertex positions, ``(N, 3)`` float64.  The first
            ``num_surface`` rows are the DC surface vertices verbatim;
            the rest are interior Steiner vertices.
        cells: TET4 connectivity (meshio ``tetra`` order, positive
            volumes), ``(T, 4)`` int32.
        num_surface: Number of leading DC surface vertices.
        boundary_tris: Outward-oriented boundary triangles (faces used by
            exactly one tet), ``(M, 3)`` int64.
        base_points: Frozen nominal positions, ``(N, 3)`` — the anchor for
            :func:`recompute_tet_points`.
        max_step: Newton re-projection displacement clamp.
        grid: The DC sampling grid the surface came from (``None`` when
            built from a raw surface).
    """

    points: np.ndarray
    cells: np.ndarray
    num_surface: int
    boundary_tris: np.ndarray
    base_points: np.ndarray
    max_step: float
    grid: GridSpec | None = None

    @property
    def num_points(self) -> int:
        """Number of vertices."""
        return int(self.points.shape[0])

    @property
    def num_cells(self) -> int:
        """Number of tetrahedra."""
        return int(self.cells.shape[0])

    def all_boundary_faces(self) -> FaceGroup:
        """Boundary triangles as a :class:`FaceGroup` (nodes shaped ``(M, 3)``)."""
        tris = self.points[self.boundary_tris]  # (M, 3, 3)
        centers = tris.mean(axis=1)
        normals = 0.5 * np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        return FaceGroup(
            nodes=self.boundary_tris,
            centers=centers,
            normals=normals / np.maximum(lengths, 1e-30),
        )


def tet_boundary_faces(cells: np.ndarray) -> np.ndarray:
    """Outward-oriented boundary triangles (faces used by exactly one tet)."""
    faces = np.asarray(cells, dtype=np.int64)[:, _TET_FACES].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    _, first_index, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return faces[first_index[counts == 1]]


def tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Signed tetrahedron volumes, ``(T,)``; positive for correct orientation."""
    corners = np.asarray(points)[np.asarray(cells)]
    return np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0


def tet_radius_ratios(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Normalized radius ratio ``3 r_in / r_circ`` per tet, in ``(0, 1]``.

    The regular tetrahedron scores 1; slivers approach 0.  Uses the exact
    inradius (``3V / A_total``) and circumradius (Cayley–Menger-free
    formula via opposite-edge products).
    """
    corners = np.asarray(points, dtype=np.float64)[np.asarray(cells)]  # (T, 4, 3)
    volume = np.abs(tet_volumes(points, cells))
    faces = corners[:, _TET_FACES]  # (T, 4, 3, 3)
    face_areas = 0.5 * np.linalg.norm(
        np.cross(faces[..., 1, :] - faces[..., 0, :], faces[..., 2, :] - faces[..., 0, :]),
        axis=-1,
    )  # (T, 4)
    inradius = 3.0 * volume / np.maximum(face_areas.sum(axis=1), 1e-30)
    # Circumradius: R = sqrt((a q_a)^2 ... ) / (24 V) with products of
    # opposite edge lengths a = |v1-v0||v3-v2| etc.
    e = [corners[:, j] - corners[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (2, 3), (1, 3), (1, 2))]
    a = np.linalg.norm(e[0], axis=1) * np.linalg.norm(e[3], axis=1)
    b = np.linalg.norm(e[1], axis=1) * np.linalg.norm(e[4], axis=1)
    c = np.linalg.norm(e[2], axis=1) * np.linalg.norm(e[5], axis=1)
    p = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
    circumradius = np.sqrt(np.maximum(p, 0.0)) / np.maximum(24.0 * volume, 1e-30)
    return 3.0 * inradius / np.maximum(circumradius, 1e-30)


def tet_aspect_ratios(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Longest-to-shortest edge ratio per tet, at least 1."""
    corners = np.asarray(points, dtype=np.float64)[np.asarray(cells)]
    edges = corners[:, _TET10_EDGES[:, 1]] - corners[:, _TET10_EDGES[:, 0]]
    lengths = np.linalg.norm(edges, axis=-1)
    return lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-30)


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
            positions, so :func:`recompute_tet_points` reproduces the
            meshed geometry exactly at the nominal design.
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
    :func:`recompute_tet_points` — which runs the identical projection
    from the frozen raw DC positions — reproduces ``mesh.points`` exactly
    at the nominal design.  Without this, re-projection at solve time
    would move the boundary of an already-meshed volume and collapse
    sliver tets.

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


def tet_faces_from_nodes(mesh: TetMesh, nodes: Any) -> np.ndarray:
    """Boundary triangles all three of whose corners belong to ``nodes``.

    The tet analog of :func:`cadjoint.fem.hexmesh.faces_from_nodes`: the
    bridge from node selections to area-integrated boundary conditions.
    """
    indices = np.asarray(nodes).reshape(-1)
    mask = np.isin(mesh.boundary_tris, indices).all(axis=1)
    return mesh.boundary_tris[mask]


def _neighbor_lists(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected vertex adjacency of the tet mesh, as flat pairs.

    Returns:
        ``(sources, targets, degrees)`` — for every directed adjacency
        pair ``sources[k] -> targets[k]``, plus per-node degree.
    """
    edges = np.asarray(mesh.cells, dtype=np.int64)[:, _TET10_EDGES].reshape(-1, 2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    sources = np.concatenate([edges[:, 0], edges[:, 1]])
    targets = np.concatenate([edges[:, 1], edges[:, 0]])
    degrees = np.bincount(targets, minlength=mesh.num_points)
    return sources, targets, degrees


def recompute_tet_points(
    sdf: Callable[[Any], Any], mesh: TetMesh, *, smooth_passes: int = 0
) -> Any:
    """Recompute tet-mesh vertex positions differentiably with frozen topology.

    The leading ``num_surface`` boundary vertices are Newton re-projected
    from their frozen nominal positions onto the (possibly traced) SDF's
    zero set — the same clamped projection the hex mesher uses, so the
    output is a differentiable function of the SDF's parameters.  Interior
    Steiner vertices stay frozen by default; ``smooth_passes > 0`` runs
    that many differentiable Jacobi–Laplacian passes propagating the
    boundary *displacement* inward (boundary values pinned), which keeps
    interior elements better shaped under larger design motions.

    Args:
        sdf: Signed distance field, possibly closing over traced parameters.
        mesh: Mesh extracted at the nominal design.
        smooth_passes: Number of interior displacement-smoothing passes.

    Returns:
        Vertex positions as a JAX array shaped like ``mesh.points``.
    """
    import jax.numpy as jnp

    base = jnp.asarray(mesh.base_points)
    count = mesh.num_surface
    projected = project_points(sdf, base[:count], mesh.max_step)
    if smooth_passes <= 0:
        return jnp.concatenate([projected, base[count:]], axis=0)
    boundary_delta = projected - base[:count]
    sources, targets, degrees = _neighbor_lists(mesh)
    weights = 1.0 / jnp.maximum(jnp.asarray(degrees, dtype=base.dtype), 1.0)
    delta = jnp.zeros_like(base).at[:count].set(boundary_delta)
    for _ in range(smooth_passes):
        averaged = jnp.zeros_like(base).at[targets].add(delta[sources]) * weights[:, None]
        delta = averaged.at[:count].set(boundary_delta)
    return base + delta


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
    cell_edges = np.sort(cells[:, _TET10_EDGES], axis=2).reshape(-1, 2)  # (T*6, 2)
    unique_edges, inverse = np.unique(cell_edges, axis=0, return_inverse=True)
    midside = unique_edges.shape[0]
    midpoints = points[unique_edges].mean(axis=1)
    cells10 = np.concatenate([cells, points.shape[0] + inverse.reshape(-1, 6)], axis=1).astype(
        np.int32
    )
    points10 = np.concatenate([points, midpoints], axis=0)
    assert points10.shape[0] == points.shape[0] + midside
    return points10, cells10, unique_edges


def _rows_in(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Boolean mask of which ``rows`` (2-D int64) appear as rows of ``table``."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    table = np.ascontiguousarray(table, dtype=np.int64)
    void = np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    return np.isin(rows.view(void).reshape(-1), table.view(void).reshape(-1))


def _restrict_traction_faces(problem: Any, traction_faces: list[np.ndarray]) -> None:
    """Prune jax-fem's face selection to exactly the given boundary triangles.

    jax-fem selects a (cell, local face) pair for a surface map whenever
    *all* the face's nodes satisfy the location function.  With node-set
    membership locations on a tet mesh this over-selects: an interior
    face whose three corners all happen to lie on the loaded surface
    patch is selected once per adjacent cell, double-loading a face that
    is not even on the boundary (observed on the bracket web at fine
    resolutions).  This helper prunes each patch's selection to the faces
    whose corner triple matches the requested boundary triangles, and
    rebuilds the dependent structures (``cells_list_face_list`` and the
    face blocks of the assembly sparsity pattern ``I``/``J``) so value
    and index arrays stay aligned.  Surface quadrature data is recomputed
    from the pruned selection by ``set_params`` before every solve.
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
    for patch, target in enumerate(traction_faces):
        binds = np.asarray(problem.boundary_inds_list[patch])
        slots = corner_slots[binds[:, 1]]
        corner_ids = np.take_along_axis(cells0[binds[:, 0]], slots, axis=1)
        keys = np.sort(corner_ids, axis=1)
        target_keys = np.sort(np.asarray(target, dtype=np.int64)[:, :3], axis=1)
        mask = _rows_in(keys, target_keys)
        if int(mask.sum()) != target_keys.shape[0]:
            raise ValueError(
                f"Traction patch {patch}: matched {int(mask.sum())} of "
                f"{target_keys.shape[0]} requested boundary faces; the traction node "
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

    Mirrors :meth:`cadjoint.fem.backends.JaxFemBackend.elastic` with the
    element type opened up: ``"TET4"`` or ``"TET10"`` (both confirmed in
    jax-fem's element tables; connectivity must be meshio ``tetra`` /
    ``tetra10`` order, as produced by TetGen resp.
    :func:`tet10_from_tet4`).  ``points`` may be traced; the displacement
    participates in the surrounding autodiff graph via jax-fem's adjoint.

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
            :func:`_restrict_traction_faces`).  Every corner must also be
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
            _restrict_traction_faces(problem, traction_faces)
        forward = ad_wrapper(problem)
        return forward(jnp.asarray(points))[0]


def load_work_tris(points: Any, displacement: Any, faces: np.ndarray, traction: Any) -> Any:
    """Work of a constant traction over linear (TRI3) boundary faces.

    ``W = sum_f area_f * t . mean(u at corners)`` — exact for linear
    interpolation, differentiable in both ``points`` and ``displacement``.
    Equal to the classical compliance ``f . u`` (twice the strain energy)
    when the faces carry the only load and supports are homogeneous.
    """
    import jax.numpy as jnp

    points = jnp.asarray(points)
    displacement = jnp.asarray(displacement)
    tri = points[np.asarray(faces)]  # (M, 3, 3)
    areas = 0.5 * jnp.linalg.norm(jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    mean_u = displacement[np.asarray(faces)].mean(axis=1)  # (M, 3)
    return jnp.sum(areas * (mean_u @ jnp.asarray(traction)))


def load_work_tri6(points: Any, displacement: Any, faces6: np.ndarray, traction: Any) -> Any:
    """Work of a constant traction over straight-sided quadratic (TRI6) faces.

    On a straight triangle the exact integral of a quadratic field uses
    the midside-only rule: ``integral(u) = area * mean(u at midsides)``.
    ``faces6`` is ``(M, 6)`` — three corners then the midsides opposite
    them (any consistent order; only corner/midside split matters).
    """
    import jax.numpy as jnp

    points = jnp.asarray(points)
    displacement = jnp.asarray(displacement)
    faces6 = np.asarray(faces6)
    tri = points[faces6[:, :3]]
    areas = 0.5 * jnp.linalg.norm(jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    mean_mid = displacement[faces6[:, 3:]].mean(axis=1)
    return jnp.sum(areas * (mean_mid @ jnp.asarray(traction)))


def load_work_quads(points: Any, displacement: Any, faces: np.ndarray, traction: Any) -> Any:
    """Work of a constant traction over bilinear (QUAD4) boundary faces.

    2x2 Gauss on the isoparametric bilinear map — matching jax-fem's own
    surface integration on HEX8 faces; differentiable in both arguments.
    """
    import jax.numpy as jnp

    points = jnp.asarray(points)
    displacement = jnp.asarray(displacement)
    corners = points[np.asarray(faces)]  # (M, 4, 3)
    corner_u = displacement[np.asarray(faces)]  # (M, 4, 3)
    g = 1.0 / np.sqrt(3.0)
    total = jnp.zeros(())
    for xi, eta in ((-g, -g), (g, -g), (g, g), (-g, g)):
        shape = 0.25 * jnp.asarray(
            [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]
        )
        d_xi = 0.25 * jnp.asarray([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        d_eta = 0.25 * jnp.asarray([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        tangent_u = jnp.einsum("i,mid->md", d_xi, corners)
        tangent_v = jnp.einsum("i,mid->md", d_eta, corners)
        jacobian = jnp.linalg.norm(jnp.cross(tangent_u, tangent_v), axis=-1)  # (M,)
        u_gauss = jnp.einsum("i,mid->md", shape, corner_u)  # (M, 3)
        total = total + jnp.sum(jacobian * (u_gauss @ jnp.asarray(traction)))
    return total
