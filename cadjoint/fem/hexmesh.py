"""Volumetric HEX8 meshing of signed distance fields.

What belongs here: turning an SDF into a :class:`HexMesh` and nothing more.
Voxelization on a regular lattice (cells whose center is inside are kept),
the boundary-vertex snapping that projects vertices onto the zero set along
the field gradient with an inversion guard, and the grouping of the outer
quads by dominant gradient axis.  The result uses VTK/meshio
``hexahedron`` corner ordering, which is also what jax-fem's ``HEX8``
element consumes.

What does *not* belong here, and where it lives instead:

- element topology tables — :mod:`cadjoint.fem.elements`
- quality metrics (``scaled_jacobians``, ``aspect_ratios``,
  ``corner_tet_volumes``) — :mod:`cadjoint.fem.quality`
- boundary faces and their selection (:class:`~cadjoint.fem.boundary.FaceGroup`,
  ``select_faces``, ``faces_from_nodes``) — :mod:`cadjoint.fem.boundary`
- differentiable node motion (``project_points``, ``recompute_points``) —
  :mod:`cadjoint.fem.motion`
- solving — :mod:`cadjoint.fem.jaxfem` / :mod:`cadjoint.fem.backends`

Those four modules are shared with :mod:`cadjoint.fem.tetmesh`, so the two
element families are layered identically.  The names below are re-exported
for callers that have always imported them from here.

The projection runs through JAX, so with frozen connectivity the node
positions can be recomputed differentiably for a traced SDF via
:func:`~cadjoint.fem.motion.recompute_points` — the hook used by the
end-to-end design-parameter -> mesh -> FEM gradient path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from cadjoint.fem.boundary import (
    FaceGroup,
    _boundary_face_rows,
    _face_geometry,
    faces_from_nodes,
    select_faces,
)
from cadjoint.fem.discretization import (
    Surface,
    _require_selection,
    _scalar_or_traced,
    _unresolvable_on_mesh,
)
from cadjoint.fem.elements import HEX_CORNER_OFFSETS
from cadjoint.fem.motion import project_points, recompute_points
from cadjoint.fem.quality import aspect_ratios, corner_tet_volumes, scaled_jacobians
from cadjoint.meshing import GridSpec

__all__ = [
    "FaceGroup",
    "GridSpec",
    "HexMesh",
    "aspect_ratios",
    "corner_tet_volumes",
    "faces_from_nodes",
    "project_points",
    "recompute_points",
    "scaled_jacobians",
    "sdf_to_hex_mesh",
    "select_faces",
]


@dataclass(frozen=True)
class HexMesh:
    """A HEX8 volume mesh extracted from an SDF.

    Attributes:
        points: Vertex positions after snapping, shaped ``(N, 3)`` float64.
        cells: HEX8 connectivity in VTK/meshio corner order, ``(C, 8)`` int32.
        boundary_faces: Outer quads (faces used by exactly one cell) grouped
            by the dominant SDF-gradient axis at their centers, keyed
            ``"+x"``/``"-x"``/... — so e.g. all faces of a box lying on its
            +x side land in one group.
        base_points: Unsnapped lattice positions of the same vertices,
            ``(N, 3)``.  Together with ``snap_mask`` this freezes the
            topology so :func:`~cadjoint.fem.motion.recompute_points` can
            rebuild ``points`` differentiably for a traced SDF.
        snap_mask: Boolean ``(N,)`` mask of vertices that were projected
            onto the zero set (after the inversion guard).
        max_step: Projection displacement clamp used for snapping (half the
            cell diagonal).
        grid: The sampling grid the mesh was extracted from.
    """

    points: np.ndarray
    cells: np.ndarray
    boundary_faces: dict[str, FaceGroup]
    base_points: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    snap_mask: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    max_step: float = 0.0
    grid: GridSpec | None = None
    #: The scene's node table and, per snapped vertex, the census surfaces it
    #: lies on — set by :func:`with_table` when the scene is known, so
    #: :meth:`moved` can hold a crease vertex on both faces that meet there.
    table: Any = field(repr=False, default=None)
    incidence: Any = field(repr=False, default=None)

    @property
    def num_points(self) -> int:
        """Number of vertices."""
        return int(self.points.shape[0])

    @property
    def num_cells(self) -> int:
        """Number of hexahedra."""
        return int(self.cells.shape[0])

    def all_boundary_faces(self) -> FaceGroup:
        """All boundary quads concatenated across gradient-axis groups."""
        groups = list(self.boundary_faces.values())
        return FaceGroup(
            nodes=np.concatenate([g.nodes for g in groups], axis=0),
            centers=np.concatenate([g.centers for g in groups], axis=0),
            normals=np.concatenate([g.normals for g in groups], axis=0),
        )

    # ── the discretization protocol (cadjoint.fem.discretization) ─────────

    @property
    def family(self) -> str:
        return "hex"

    def surface(self) -> Surface:
        return Surface(
            points=self.points,
            groups=tuple(
                (group_id, self.boundary_faces[group_id].nodes)
                for group_id in sorted(self.boundary_faces)
            ),
        )

    def quality(self) -> dict[str, np.ndarray]:
        from cadjoint.fem.quality import aspect_ratios, scaled_jacobians

        return {
            "scaled_jacobian": scaled_jacobians(self.points, self.cells),
            "aspect_ratio": aspect_ratios(self.points, self.cells),
        }

    def node_patch(self, selection: Any) -> np.ndarray:
        _require_selection(selection)
        return selection.resolve(self)

    def face_patch(self, selection: Any) -> tuple[np.ndarray, None]:
        """The union of the corners of the boundary quads the selection spans, and no faces.

        A backend applies an area-integrated condition to exactly the faces
        all of whose corners are in the set.
        """
        _require_selection(selection)
        group = faces_from_nodes(self, selection.resolve(self))
        if group.nodes.size == 0:
            raise ValueError(
                f"Selection {selection.describe()} spans no complete boundary face; "
                "area-integrated conditions need all four corners of at least one "
                "boundary quad selected."
            )
        return np.unique(group.nodes).astype(np.int32), None

    def unresolvable(self, bcs: list) -> str | None:
        return _unresolvable_on_mesh(self, bcs)

    def moved(self, field: Any, *, smooth_passes: int = 0, design: Any = None) -> Any:  # noqa: ARG002 - the protocol's signature
        """Node positions under the design: snapped vertices re-projected, the lattice frozen.

        With a table and the design's parameters, each snapped vertex is
        solved onto the census surfaces it was classified on — a crease
        vertex onto both faces at once — which is the derivative a single
        field's Newton gets wrong there.  Without, the single field's.
        """
        from cadjoint.fem.motion import recompute_points

        if self.table is None or design is None:
            return recompute_points(field, self)
        import jax.numpy as jnp

        from cadjoint.zeroset.project import project_table, theta_array

        indices = np.flatnonzero(self.snap_mask)
        base = jnp.asarray(self.points)
        theta = theta_array(self.table, design[1])
        projected = project_table(
            self.table, theta, base[indices], self.incidence, max_step=self.max_step
        )
        return base.at[indices].set(projected)

    def thermal(self, problem: Any, *, placement: Any = None, backend: Any = None) -> Any:
        """Steady conduction through the backend registry (``jaxfem`` by default)."""
        from cadjoint.fem.backends import ThermalBCs, get_backend
        from cadjoint.fem.simulate import ThermalResult, _property_value

        bcs = ThermalBCs(
            dirichlet_nodes=[self.node_patch(patch) for patch, _ in problem.dirichlet],
            dirichlet_values=[_scalar_or_traced(value) for _, value in problem.dirichlet],
            flux_nodes=[self.face_patch(patch)[0] for patch, _ in problem.neumann],
            flux_values=[float(value) for _, value in problem.neumann],
        )
        temperature = get_backend(backend).thermal(
            self.points if placement is None else placement,
            self.cells,
            bcs,
            conductivity=_property_value(problem.conductivity),
            source=float(problem.source),
            base_points=self.points,
        )
        return ThermalResult(temperature=temperature, mesh=self)

    def elastic(self, problem: Any, *, placement: Any = None, backend: Any = None) -> Any:
        """Linear elasticity through the backend registry (``jaxfem`` by default)."""
        from cadjoint.fem.backends import ElasticBCs, get_backend
        from cadjoint.fem.simulate import ElasticResult, _property_value

        bcs = ElasticBCs(
            fixed_nodes=[self.node_patch(patch) for patch in problem.fixed],
            traction_nodes=[self.face_patch(patch)[0] for patch, _ in problem.tractions],
            traction_vectors=[np.asarray(v, dtype=np.float64) for _, v in problem.tractions],
        )
        extra: dict[str, Any] = {}
        if problem.body_force is not None:
            extra["body_force"] = problem.body_force
        displacement = get_backend(backend).elastic(
            self.points if placement is None else placement,
            self.cells,
            bcs,
            youngs=_property_value(problem.youngs),
            poisson=_property_value(problem.poisson),
            base_points=self.points,
            **extra,
        )
        return ElasticResult(
            displacement=displacement,
            mesh=self,
            youngs=_property_value(problem.youngs),
            poisson=_property_value(problem.poisson),
        )

    def traction_work(self, positions: Any, displacement: Any, selection: Any, vector: Any) -> Any:
        from cadjoint.fem.postprocess import load_work_quads

        quads = faces_from_nodes(self, selection.resolve(self)).nodes
        return load_work_quads(positions, displacement, quads, vector)


def with_table(mesh: Any, scene: Any, *, tolerance: float | None = None) -> Any:
    """The mesh with the scene's node table and its surface vertices classified onto it.

    Lowers ``scene`` (a construction object; a bare callable has no table
    and the mesh is returned as is), names the census surfaces each
    surface vertex lies on, and places the vertices with the incidence-aware
    kernel at the nominal design so that :meth:`moved` at that design is a
    no-op.  A vertex on no surface within ``tolerance`` (a twentieth of the
    finest spacing by default) keeps its single-field placement.
    """
    if not hasattr(scene, "params") and not hasattr(scene, "children"):
        return mesh
    if not hasattr(mesh, "base_points"):
        # Cut cells have no node map to correct: their placement *is* the
        # field, and `moved` hands the field straight back.
        return mesh
    import dataclasses

    import jax.numpy as jnp

    from cadjoint.zeroset import lower
    from cadjoint.zeroset.project import classify, project_table

    try:
        table = lower(scene)
    except Exception:  # noqa: BLE001 - a scene the table cannot express keeps the single field
        return mesh
    if isinstance(mesh, HexMesh):
        indices = np.flatnonzero(mesh.snap_mask)
    else:
        indices = np.arange(mesh.num_surface)
    if indices.size == 0:
        return dataclasses.replace(mesh, table=table, incidence=[])
    spacing = min(mesh.grid.spacing) if mesh.grid is not None else mesh.max_step
    tolerance = 0.05 * float(spacing) if tolerance is None else tolerance
    theta = jnp.asarray(table.theta)
    surface = np.asarray(mesh.points)[indices]
    incidence = classify(table, theta, surface, tolerance=tolerance)
    # Converged, not merely stepped: `moved` re-solves from these points at
    # every design, so a placement that is not a fixed point of its own
    # projection would move the mesh under a design that did not change.
    placed = surface
    for _ in range(4):
        stepped = np.asarray(project_table(table, theta, placed, incidence, max_step=mesh.max_step))
        if np.abs(stepped - placed).max() < 1e-12:
            placed = stepped
            break
        placed = stepped
    points, incidence = _guard_inversions(mesh, indices, placed, incidence)
    return dataclasses.replace(mesh, points=points, table=table, incidence=incidence)


def _guard_inversions(
    mesh: Any, indices: np.ndarray, placed: np.ndarray, incidence: list[list[int]]
) -> tuple[np.ndarray, list[list[int]]]:
    """Take the crease placement only where it costs no element any quality.

    Landing a vertex exactly on the edge two faces make can pull it further
    than the mesher's own snap did — the mesher guards its snap against
    inversion (:func:`_snap_boundary_vertices`) and this guards against
    *degradation*, which is stricter and is the property a mesh is judged
    on: an element whose metric would drop has the vertices that moved in
    it put back.  A reverted vertex keeps its position and the single
    surface nearest it, so it moves exactly as it did before.

    **How many vertices this reverts is a property of the tet fill, not of
    the classifier — and it is worth reading before you believe a crease
    count.**  Because the test is "no element gets worse *at all*", a mesh
    with slivers in it trips the guard everywhere.  Measured on
    ``scenes/starter.py`` (``research/performance.md`` §16.8): in float32
    :func:`~cadjoint.zeroset.project.classify` finds 133 crease vertices and
    this reverts 41 of them, leaving 92; in float64 it finds 124 — the same
    answer, so detection is stable — and this reverts **118**, leaving 6.
    The difference is not precision reaching the classifier.  Enabling x64
    moves the dual-contour crossings, TetGen fills the surface differently,
    and the resulting mesh's worst raw element is 4.6x worse (minimum radius
    ratio 0.134 against 0.029); on that mesh the placement would degrade
    1327 of 3104 elements instead of 222 of 2962, so the guard takes almost
    all of it back.  A high revert rate is therefore a sliver alarm about
    the fill upstream, and a crease count taken *after* this function is
    measuring the mesher's luck as much as the geometry.
    """
    from cadjoint.fem.quality import scaled_jacobians, tet_radius_ratios

    points = np.array(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells)
    hexes = cells.shape[1] == 8
    corners = cells if hexes else cells[:, :4]

    def metric(x: np.ndarray) -> np.ndarray:
        return scaled_jacobians(x, cells) if hexes else tet_radius_ratios(x, corners)

    baseline = metric(points)
    proposed = points.copy()
    proposed[indices] = placed
    moved = np.zeros(points.shape[0], dtype=bool)
    moved[indices] = True
    for _ in range(16):
        worse = np.flatnonzero(metric(proposed) < baseline - 1e-9)
        if worse.size == 0:
            break
        revert = np.unique(corners[worse])
        revert = revert[moved[revert]]
        if revert.size == 0:
            break
        moved[revert] = False
        proposed[revert] = points[revert]
    reverted = set(np.flatnonzero(~moved).tolist())
    incidence = [
        (row[:1] if int(index) in reverted else row) for index, row in zip(indices, incidence)
    ]
    if getattr(mesh, "edge_parents", None) is not None:
        parents = proposed[: mesh.num_corner_points]
        proposed[mesh.num_corner_points :] = parents[mesh.edge_parents].mean(axis=1)
    return proposed, incidence


def _evaluate_sdf(sdf: Callable[[Any], Any], points: np.ndarray) -> np.ndarray:
    """Evaluate ``sdf`` on ``(M, 3)`` points, returning a float64 array."""
    import jax.numpy as jnp

    return np.asarray(sdf(jnp.asarray(points)), dtype=np.float64).reshape(-1)


def _group_boundary_faces(
    sdf: Callable[[Any], Any], points: np.ndarray, faces: np.ndarray
) -> dict[str, FaceGroup]:
    """Group boundary quads by the dominant SDF-gradient axis at their centers."""
    import jax
    import jax.numpy as jnp

    centers, normals = _face_geometry(points, faces)
    gradient = np.asarray(
        jax.vmap(jax.grad(lambda p: jnp.asarray(sdf(p)).reshape(())))(jnp.asarray(centers))
    )
    axis = np.argmax(np.abs(gradient), axis=-1)
    positive = np.take_along_axis(gradient, axis[:, None], axis=-1)[:, 0] >= 0.0
    groups: dict[str, FaceGroup] = {}
    for axis_index, axis_name in enumerate("xyz"):
        for sign_positive, sign in ((True, "+"), (False, "-")):
            mask = (axis == axis_index) & (positive == sign_positive)
            if np.any(mask):
                groups[f"{sign}{axis_name}"] = FaceGroup(
                    nodes=faces[mask], centers=centers[mask], normals=normals[mask]
                )
    return groups


def _snap_boundary_vertices(
    sdf: Callable[[Any], Any],
    points: np.ndarray,
    cells: np.ndarray,
    boundary_faces: np.ndarray,
    max_step: float,
    snap_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project eligible boundary vertices onto the zero set, guarding inversions.

    Candidates are the vertices of boundary faces lying within ``snap_range``
    of the zero set.  All candidates are projected at once (displacement
    clamped to ``max_step``); then any vertex whose move would drive an
    incident hex's corner-tet volume non-positive is reverted, iterating
    until every hex is valid.

    Returns:
        ``(new_points, snap_mask)``.
    """
    candidate = np.zeros(points.shape[0], dtype=bool)
    candidate[np.unique(boundary_faces)] = True
    values = _evaluate_sdf(sdf, points[candidate])
    keep = np.abs(values) <= snap_range
    indices = np.flatnonzero(candidate)[keep]
    if indices.size == 0:
        return points.copy(), np.zeros(points.shape[0], dtype=bool)

    projected = np.asarray(project_points(sdf, points[indices], max_step), dtype=np.float64)

    snap_mask = np.zeros(points.shape[0], dtype=bool)
    snap_mask[indices] = True
    proposed = points.copy()
    proposed[indices] = projected
    projected_by_index = dict(zip(indices.tolist(), projected))

    # Volumes below this threshold count as (nearly) inverted.  Relative to
    # the unsnapped cell volume so the guard is scale-independent.
    cell_volume = float(np.prod([max(step, 1e-30) for step in _cell_spacing(points, cells)]))
    threshold = 1e-9 * cell_volume

    for _ in range(16):
        volumes = corner_tet_volumes(proposed, cells)
        bad_cells = np.flatnonzero(np.any(volumes <= threshold, axis=1))
        if bad_cells.size == 0:
            break
        bad_vertices = np.unique(cells[bad_cells])
        revert = bad_vertices[snap_mask[bad_vertices]]
        if revert.size == 0:
            break
        snap_mask[revert] = False
        proposed[revert] = points[revert]
    final = points.copy()
    moved = np.flatnonzero(snap_mask)
    final[moved] = np.array([projected_by_index[int(i)] for i in moved])
    return final, snap_mask


def _cell_spacing(points: np.ndarray, cells: np.ndarray) -> tuple[float, float, float]:
    """Edge lengths of the first hex along its three lattice axes."""
    first = points[cells[0]]
    return (
        float(np.linalg.norm(first[1] - first[0])),
        float(np.linalg.norm(first[3] - first[0])),
        float(np.linalg.norm(first[4] - first[0])),
    )


def sdf_to_hex_mesh(sdf: Callable[[Any], Any], grid: GridSpec, *, snap: bool = True) -> HexMesh:
    """Extract a HEX8 volume mesh from an SDF on a regular grid.

    Cells whose center lies inside (``sdf < 0``) are kept, sharing vertices
    through the lattice.  With ``snap=True`` boundary vertices within one
    cell diagonal of the zero set are Newton-projected onto the surface
    along the field gradient, with total displacement clamped to half the
    cell diagonal; a vertex is left unsnapped if moving it would invert any
    incident hex (corner-tet volume check).

    Args:
        sdf: Signed distance field callable on ``(..., 3)`` points.
        grid: Sampling lattice.
        snap: Project boundary vertices onto the zero set.

    Returns:
        The extracted :class:`HexMesh`.

    Raises:
        ValueError: If no cell center lies inside the field.
    """
    nx, ny, nz = grid.cells
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    origin = np.asarray(grid.origin, dtype=np.float64)

    index = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    centers = origin + (index + 0.5) * spacing
    inside = _evaluate_sdf(sdf, centers) < 0.0
    if not np.any(inside):
        raise ValueError("No cell center lies inside the SDF; enlarge the grid or resolution.")
    kept = index[inside]  # (C, 3) lower lattice corners

    corner_index = kept[:, None, :] + HEX_CORNER_OFFSETS[None, :, :]  # (C, 8, 3)
    flat = (corner_index[..., 0] * (ny + 1) + corner_index[..., 1]) * (nz + 1) + corner_index[
        ..., 2
    ]
    used, cells = np.unique(flat, return_inverse=True)
    cells = cells.reshape(flat.shape).astype(np.int32)

    used_index = np.stack(
        (used // ((ny + 1) * (nz + 1)), (used // (nz + 1)) % (ny + 1), used % (nz + 1)), axis=-1
    )
    base_points = origin + used_index.astype(np.float64) * spacing

    boundary = _boundary_face_rows(cells)
    max_step = 0.5 * float(np.linalg.norm(spacing))
    if snap:
        points, snap_mask = _snap_boundary_vertices(
            sdf, base_points, cells, boundary, max_step, snap_range=float(np.linalg.norm(spacing))
        )
    else:
        points = base_points.copy()
        snap_mask = np.zeros(points.shape[0], dtype=bool)

    return HexMesh(
        points=points,
        cells=cells,
        boundary_faces=_group_boundary_faces(sdf, points, boundary),
        base_points=base_points,
        snap_mask=snap_mask,
        max_step=max_step,
        grid=grid,
    )
