"""The ownership graph: a B-rep derived from patch ownership, not stored.

Every hard primitive in cadjoint is a ``min``/``max`` over smooth **patch
fields** with exact surface ownership
(:mod:`cadjoint.meshing.patch_fields`).  That makes a boundary
representation *derivable* rather than authored:

- a **face** is the part of one patch field's zero set that survived the
  booleans,
- an **edge** is where two patch zero sets meet,
- a **vertex** is where three meet.

None of those is stored geometry.  A vertex is the solution of
``f_a = f_b = f_c = 0``, an edge point the solution of ``f_a = f_b = 0``
nearest its seed, a face point the solution of ``f_a = 0`` — all three the
same :func:`cadjoint.brep.project.project` call at a different arity, and
all three differentiable in the design parameters by the implicit-function
theorem.  What is stored is *which* patches meet where, and that is
discrete: frozen per extraction exactly like the crossing-edge set and the
cell incidence one stage below.

**Dual contouring is the discovery tool, not the geometry.**  The extractor
finds the graph — which patches own which region of the surface, which
regions are adjacent, in what cyclic order — and hands over seed points.
Every seed is then re-solved by the kernel, so the answer is independent of
where the lattice happened to cut.  :attr:`BRep.points` is the whole mesh
re-solved that way: each vertex carries the patch set it belongs to
(:attr:`BRep.owner_patches`) and is placed by a 1-, 2- or 3-field
projection accordingly.

**Blends are the one genuinely smooth surface type.**  A smooth union
(``smoothness > 0``) creates surface that lies on *no* patch's zero set.
The test is exact and needs no threshold on angle: project a candidate
surface point onto the scene's own zero set, then ask its owning patch
field for its value there.  Zero means the patch owns the point and the face
is analytic; anything of order the blend radius means the point is on a
fillet, and that face has no analytic surface and must be tessellated.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cadjoint.brep.project import batched_residuals, project_batched, project_fields
from cadjoint.meshing.dual_contouring import Mesh, extract_mesh
from cadjoint.meshing.edge_detection import GridSpec
from cadjoint.meshing.patch_fields import ScenePatchFields, scene_patch_fields

__all__ = [
    "AnalyticSurface",
    "BRep",
    "BRepEdge",
    "BRepFace",
    "BRepVertex",
    "Patch",
    "extract_brep",
]

#: Face kinds that carry an analytic surface a STEP writer can emit exactly.
ANALYTIC_KINDS = ("plane", "cylinder", "sphere", "cone")

# A quad (a, b, c, d) contributes these consecutive index pairs as edges.
_QUAD_EDGE_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))


class Patch(NamedTuple):
    """One smooth patch field of one world-frame leaf.

    Attributes:
        index: Global patch index (the graph's own numbering).
        leaf: Index of the world-frame leaf in
            :attr:`~cadjoint.meshing.patch_fields.ScenePatchFields.leaves`.
        patch: Index of the field within that leaf's decomposition.
        kind: Structural surface type of the patch, read off the primitive
            it came from — one of :data:`ANALYTIC_KINDS`, ``"revolution"``
            for a swept profile edge with no closed form here, or
            ``"opaque"`` when the leaf declares no patch decomposition.
        field: The world-frame callable ``f(p)`` whose zero set the patch is.
    """

    index: int
    leaf: int
    patch: int
    kind: str
    field: Callable[[Array], Array]


@dataclass(frozen=True)
class AnalyticSurface:
    """A fitted analytic surface for one face, with its measured residual.

    The *type* is structural — it comes from the primitive the patch belongs
    to, never from guessing at a point cloud — and only the placement is
    fitted, from the face's own re-projected sample points and the patch
    field's gradients there.  ``residual`` is the largest distance from a
    sample point to the fitted surface, so a caller can refuse to emit an
    analytic STEP surface it cannot certify.

    Attributes:
        kind: ``plane``, ``cylinder``, ``sphere``, ``cone``, or
            ``freeform`` when no closed form was fitted.
        origin: A point of the surface — on the plane, on the axis, the
            centre of the sphere, the apex of the cone.
        axis: Plane normal, or cylinder/cone axis; zeros for a sphere.
        radius: Cylinder or sphere radius; zero otherwise.
        half_angle: Cone half angle in radians; zero otherwise.
        residual: Largest sample distance from the fitted surface.
        sense: ``+1`` when the surface's own outward direction (the plane
            normal, the radial direction away from a cylinder or sphere
            centre) is the *solid's* outward normal, ``-1`` when the face is
            a bore and the solid lies outside the fitted surface.  Measured
            against the scene's own gradient, not the patch field's: a
            subtracted cylinder's field grows away from its axis whichever
            side the material is on.
    """

    kind: str
    origin: np.ndarray
    axis: np.ndarray
    radius: float
    half_angle: float
    residual: float
    sense: float = 1.0


@dataclass(frozen=True)
class BRepFace:
    """A face: the surviving part of one patch field's zero set.

    Attributes:
        index: Face index within the :class:`BRep`.
        patch: Global patch index, or ``-1`` for a blend face (which lies on
            no patch's zero set).
        leaf: World-frame leaf the face belongs to.
        kind: :attr:`Patch.kind` for an analytic face, ``"blend"`` for a
            smooth-union fillet.
        analytic: Whether :attr:`surface` is a certified closed form.
        surface: The fitted :class:`AnalyticSurface`.
        quads: Indices into ``mesh.quads`` of the region the face covers.
        loops: Boundary loops as ordered mesh-vertex indices, outer loop
            first; empty when the region's boundary is not a disjoint union
            of simple loops (reported in :attr:`BRep.stats`).
        area: Summed quad area, as a size measure for reports.
    """

    index: int
    patch: int
    leaf: int
    kind: str
    analytic: bool
    surface: AnalyticSurface
    quads: np.ndarray
    loops: list[list[int]]
    area: float


@dataclass(frozen=True)
class BRepEdge:
    """An edge: where two patch zero sets meet.

    Attributes:
        index: Edge index within the :class:`BRep`.
        faces: The two face indices the edge separates.
        patches: The two global patch indices, or ``-1`` where the
            neighbouring face is a blend.
        polyline: Seed points re-solved by the two-field projection,
            ordered along the curve, shaped ``(k, 3)``.
        vertices: Indices of the :class:`BRepVertex` endpoints, ``-1`` where
            the chain does not end at a triple point.
        closed: Whether the chain closes on itself (a circle, e.g. a
            cylinder's rim).
        analytic: Whether both neighbours are analytic, so the polyline
            really is on an exact intersection curve.
        residual: Largest ``|f|`` over both fields along the polyline.
    """

    index: int
    faces: tuple[int, int]
    patches: tuple[int, int]
    polyline: np.ndarray
    vertices: tuple[int, int]
    closed: bool
    analytic: bool
    residual: float


@dataclass(frozen=True)
class BRepVertex:
    """A vertex: where three patch zero sets meet.

    Attributes:
        index: Vertex index within the :class:`BRep`.
        mesh_vertex: The dual-contour vertex that seeded it.
        faces: Every face incident to it (three for a clean corner; more
            marks an ambiguity, reported in :attr:`BRep.stats`).
        patches: The three global patch indices actually solved for.
        point: The solved position, shaped ``(3,)``.
        analytic: Whether three analytic patches were available.
        residual: Largest ``|f|`` over the three fields at :attr:`point`.
    """

    index: int
    mesh_vertex: int
    faces: tuple[int, ...]
    patches: tuple[int, ...]
    point: np.ndarray
    analytic: bool
    residual: float


@dataclass(frozen=True)
class BRep:
    """A derived boundary representation over a frozen ownership graph.

    Attributes:
        faces: Every face, analytic and blend alike.
        edges: Every face-pair boundary chain.
        vertices: Every triple point.
        patches: The scene's global patch table.
        mesh: The dual-contour mesh the graph was discovered on.
        grid: The sampling grid.
        points: Every mesh vertex re-solved by the projection kernel at its
            own arity, shaped ``(n, 3)``.
        owner_patches: Per mesh vertex, the global patch indices it was
            solved against, shaped ``(n, 3)`` and ``-1``-padded.
        owner_arity: Per mesh vertex, how many fields that was (0 for a
            vertex whose owners include a blend, which is left where dual
            contouring put it).
        quad_face: Face index of every quad, shaped ``(q,)``.
        decomposition: The scene's :class:`ScenePatchFields`.
        stats: Counts and ambiguity reports — see :meth:`report`.
    """

    faces: list[BRepFace]
    edges: list[BRepEdge]
    vertices: list[BRepVertex]
    patches: list[Patch]
    mesh: Mesh
    grid: GridSpec
    points: np.ndarray
    owner_patches: np.ndarray
    owner_arity: np.ndarray
    quad_face: np.ndarray
    decomposition: ScenePatchFields
    stats: dict = field(default_factory=dict)

    def analytic_faces(self) -> list[BRepFace]:
        """Faces whose surface is a certified closed form."""
        return [face for face in self.faces if face.analytic]

    def blend_faces(self) -> list[BRepFace]:
        """Faces that lie on no patch zero set — smooth-union fillets."""
        return [face for face in self.faces if face.kind == "blend"]

    def report(self) -> dict:
        """A flat, printable summary: counts per kind plus the ambiguities."""
        kinds: dict[str, int] = {}
        for face in self.faces:
            kinds[face.kind] = kinds.get(face.kind, 0) + 1
        return {
            "faces": len(self.faces),
            "edges": len(self.edges),
            "vertices": len(self.vertices),
            "patches": len(self.patches),
            "face_kinds": kinds,
            "analytic_faces": len(self.analytic_faces()),
            "blend_faces": len(self.blend_faces()),
            **self.stats,
        }


# ── patch table ──────────────────────────────────────────────────────────────


def _unwrap(node: Any) -> Any:
    """Descend a leaf's transform chain to the primitive underneath."""
    from cadjoint.sdf.primitives.base import Primitive
    from cadjoint.sdf.transforms.base import Transform

    seen = 0
    while isinstance(node, Transform) and seen < 32:
        child = getattr(node, "sdf", None)
        if child is None:
            return node
        node = child
        seen += 1
    return node if isinstance(node, Primitive) else node


def _revolved_edge_kind(primitive: Any, index: int) -> str:
    """Surface type swept by profile edge ``index`` of a revolved polygon.

    A profile edge at constant radius sweeps a cylinder, one at constant
    height sweeps an annulus in a plane, and anything else sweeps a cone.
    The profile is closed, so edge ``k`` runs from vertex ``k`` to ``k+1``.
    """
    count = int(primitive.num_vertices)
    values = [np.asarray(primitive.params[f"v{i}"].value, dtype=np.float64) for i in range(count)]
    start = values[index]
    end = values[(index + 1) % count]
    delta = end - start
    scale = max(float(np.abs(values).max()), 1e-9)
    if abs(float(delta[0])) <= 1e-9 * scale:
        return "cylinder"
    if abs(float(delta[1])) <= 1e-9 * scale:
        return "plane"
    return "cone"


def _patch_kind(leaf: Any, index: int, exact: bool) -> str:
    """Structural surface type of patch ``index`` of a world-frame leaf.

    Read off the primitive's own patch decomposition — the same ordering the
    primitive's :meth:`~cadjoint.sdf.base.SDF.patch_fields` documents — so
    the type is known rather than inferred from geometry.  Transforms in
    between are isometries or uniform scales and do not change the type.
    """
    if not exact:
        return "opaque"
    from cadjoint.sdf.primitives.box import Box
    from cadjoint.sdf.primitives.cylinder import Cylinder
    from cadjoint.sdf.primitives.plane import Plane
    from cadjoint.sdf.primitives.polygon import ExtrudedPolygon, RevolvedPolygon
    from cadjoint.sdf.primitives.sphere import Sphere

    primitive = _unwrap(leaf)
    if isinstance(primitive, (Box, Plane)):
        return "plane"
    if isinstance(primitive, Sphere):
        return "sphere"
    if isinstance(primitive, Cylinder):
        return "cylinder" if index == 0 else "plane"
    if isinstance(primitive, ExtrudedPolygon):
        return "plane"
    if isinstance(primitive, RevolvedPolygon):
        return _revolved_edge_kind(primitive, index)
    return "freeform"


def _patch_table(decomposition: ScenePatchFields) -> tuple[list[Patch], np.ndarray]:
    """Flatten the per-leaf decomposition into one global patch table.

    Returns:
        ``(patches, offsets)`` — the table and, per leaf, the global index
            its patch 0 sits at, so ``(leaf, patch)`` maps to
            ``offsets[leaf] + patch``.
    """
    patches: list[Patch] = []
    offsets = np.zeros(len(decomposition.leaves) + 1, dtype=np.int64)
    for leaf_id, (leaf, fields, exact) in enumerate(
        zip(decomposition.leaves, decomposition.fields, decomposition.exact)
    ):
        offsets[leaf_id] = len(patches)
        for patch_id, patch_field in enumerate(fields):
            patches.append(
                Patch(
                    index=len(patches),
                    leaf=leaf_id,
                    patch=patch_id,
                    kind=_patch_kind(leaf, patch_id, exact),
                    field=patch_field,
                )
            )
    offsets[-1] = len(patches)
    return patches, offsets


# ── ownership ────────────────────────────────────────────────────────────────


def _own_patch(
    decomposition: ScenePatchFields, offsets: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Owning global patch index and ``|f|`` there, for surface points.

    Ownership is the two-stage ``argmin`` of
    :func:`cadjoint.meshing.patch_fields.signature_function`: the nearest
    leaf first, then the nearest patch *within* that leaf.  Going straight to
    a global ``argmin`` would let another solid's unbounded half-space field
    (a cap plane extends forever) steal a point that is nowhere near it.
    """
    probes = jnp.asarray(np.asarray(points, dtype=np.float64).reshape(-1, 3), dtype=jnp.float32)
    leaf_values = np.stack(
        [
            np.abs(np.asarray(jax.vmap(leaf)(probes), dtype=np.float64))
            for leaf in decomposition.leaves
        ]
    )
    leaf_ids = np.argmin(leaf_values, axis=0)
    per_leaf = [
        np.stack([np.abs(np.asarray(jax.vmap(f)(probes), dtype=np.float64)) for f in fields])
        for fields in decomposition.fields
    ]
    owner = np.zeros(probes.shape[0], dtype=np.int64)
    magnitude = np.zeros(probes.shape[0], dtype=np.float64)
    for leaf_id, values in enumerate(per_leaf):
        rows = np.flatnonzero(leaf_ids == leaf_id)
        if rows.size == 0:
            continue
        local = np.argmin(values[:, rows], axis=0)
        owner[rows] = offsets[leaf_id] + local
        magnitude[rows] = values[local, rows]
    return owner, magnitude


def _components(adjacency: list[set[int]], key: np.ndarray) -> np.ndarray:
    """Connected components of an adjacency list, split by ``key``."""
    labels = np.full(len(adjacency), -1, dtype=np.int64)
    next_label = 0
    for seed in range(len(adjacency)):
        if labels[seed] >= 0:
            continue
        labels[seed] = next_label
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if labels[neighbor] < 0 and key[neighbor] == key[current]:
                    labels[neighbor] = next_label
                    queue.append(neighbor)
        next_label += 1
    return labels


def _quad_adjacency(quads: np.ndarray) -> tuple[list[set[int]], dict[tuple[int, int], list[int]]]:
    """Quad neighbours across shared mesh edges, and the edge-to-quad map."""
    users: dict[tuple[int, int], list[int]] = {}
    for quad_id in range(quads.shape[0]):
        row = quads[quad_id]
        for i, j in _QUAD_EDGE_PAIRS:
            a, b = int(row[i]), int(row[j])
            key = (a, b) if a < b else (b, a)
            users.setdefault(key, []).append(quad_id)
    neighbors: list[set[int]] = [set() for _ in range(quads.shape[0])]
    for owners in users.values():
        for a in owners:
            for b in owners:
                if a != b:
                    neighbors[a].add(b)
    return neighbors, users


def _region_loops(quads: np.ndarray, region: np.ndarray) -> list[list[int]]:
    """Ordered boundary loops of a quad region, or ``[]`` when non-simple.

    Boundary edges are the ones used by exactly one quad of the region;
    their directions come from the quads' consistent winding and chain into
    closed loops.  Unlike the single-loop merge in
    :mod:`cadjoint.meshing.export` this keeps every loop, because a real
    face may well have holes (a plate with a bore).
    """
    usage: dict[tuple[int, int], int] = {}
    directed: dict[tuple[int, int], tuple[int, int]] = {}
    for quad_id in region:
        row = quads[quad_id]
        for i, j in _QUAD_EDGE_PAIRS:
            a, b = int(row[i]), int(row[j])
            key = (a, b) if a < b else (b, a)
            usage[key] = usage.get(key, 0) + 1
            directed[key] = (a, b)
    successor: dict[int, int] = {}
    for key, count in usage.items():
        if count != 1:
            continue
        a, b = directed[key]
        if a in successor:
            return []
        successor[a] = b
    loops: list[list[int]] = []
    remaining = set(successor)
    while remaining:
        start = min(remaining)
        loop = [start]
        remaining.discard(start)
        current = successor[start]
        while current != start:
            if current not in remaining:
                return []
            loop.append(current)
            remaining.discard(current)
            current = successor[current]
        if len(loop) >= 3:
            loops.append(loop)
    loops.sort(key=len, reverse=True)
    return loops


# ── analytic surface fitting ─────────────────────────────────────────────────


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(lengths > 0, lengths, 1.0)


def _patch_normals(patch_field: Callable[[Array], Array], points: np.ndarray) -> np.ndarray:
    """Outward unit normals of a patch at surface points.

    A patch field is negative inside its primitive, so its gradient points
    out of the solid — the same orientation convention the exporters use.
    """
    probes = jnp.asarray(points, dtype=jnp.float32)
    gradients = jax.vmap(jax.grad(lambda p: jnp.asarray(patch_field(p)).reshape(())))(probes)
    return _unit_rows(np.asarray(gradients, dtype=np.float64))


def _fit_plane(points: np.ndarray, normals: np.ndarray) -> AnalyticSurface:
    normal = _unit_rows(normals.mean(axis=0)[None, :])[0]
    origin = points.mean(axis=0)
    residual = float(np.abs((points - origin) @ normal).max()) if points.size else 0.0
    return AnalyticSurface("plane", origin, normal, 0.0, 0.0, residual, 1.0)


def _fit_sphere(points: np.ndarray, normals: np.ndarray) -> AnalyticSurface:
    # |x|^2 - 2 c.x + |c|^2 = r^2  is linear in (c, |c|^2 - r^2).
    design = np.concatenate([2.0 * points, np.ones((points.shape[0], 1))], axis=1)
    target = np.sum(points * points, axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center = solution[:3]
    radius = float(np.sqrt(max(solution[3] + float(center @ center), 0.0)))
    residual = float(np.abs(np.linalg.norm(points - center, axis=1) - radius).max())
    sense = 1.0 if float(np.mean(np.sum(normals * (points - center), axis=1))) >= 0 else -1.0  # noqa: E501
    return AnalyticSurface("sphere", center, np.zeros(3), radius, 0.0, residual, sense)


def _axis_from_normals(normals: np.ndarray) -> np.ndarray:
    """Cylinder axis: the direction every surface normal is orthogonal to."""
    _u, _s, vt = np.linalg.svd(normals - 0.0, full_matrices=True)
    return vt[-1]


def _fit_cylinder(
    points: np.ndarray, normals: np.ndarray, solid_normals: np.ndarray
) -> AnalyticSurface:
    axis = _axis_from_normals(normals)
    # Circle fit in the plane orthogonal to the axis.
    basis = _orthonormal_basis(axis)
    planar = points @ basis.T
    design = np.concatenate([2.0 * planar, np.ones((planar.shape[0], 1))], axis=1)
    target = np.sum(planar * planar, axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center2 = solution[:2]
    radius = float(np.sqrt(max(solution[2] + float(center2 @ center2), 0.0)))
    origin = basis.T @ center2
    residual = float(np.abs(np.linalg.norm(planar - center2, axis=1) - radius).max())
    radial = points - origin - ((points - origin) @ axis)[:, None] * axis
    sense = 1.0 if float(np.mean(np.sum(solid_normals * radial, axis=1))) >= 0 else -1.0
    return AnalyticSurface("cylinder", origin, axis, radius, 0.0, residual, sense)


def _orthonormal_basis(axis: np.ndarray) -> np.ndarray:
    """Two unit vectors spanning the plane orthogonal to ``axis``, as rows."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return np.stack([first, second])


def _fit_cone(
    points: np.ndarray, normals: np.ndarray, solid_normals: np.ndarray
) -> AnalyticSurface:
    # Every tangent plane of a cone contains the apex: n.(x - q) = 0.
    apex, *_ = np.linalg.lstsq(normals, np.sum(normals * points, axis=1), rcond=None)
    rays = points - apex
    lengths = np.linalg.norm(rays, axis=1, keepdims=True)
    unit_rays = rays / np.where(lengths > 0, lengths, 1.0)
    axis = _unit_rows(unit_rays.mean(axis=0)[None, :])[0]
    cosines = unit_rays @ axis
    half_angle = float(np.arccos(np.clip(float(np.mean(cosines)), -1.0, 1.0)))
    # Distance to the cone surface, measured along the surface normal.
    radial = np.linalg.norm(rays - (rays @ axis)[:, None] * axis, axis=1)
    axial = rays @ axis
    residual = float(np.abs(radial * np.cos(half_angle) - axial * np.sin(half_angle)).max())
    away = rays - axial[:, None] * axis
    sense = 1.0 if float(np.mean(np.sum(solid_normals * away, axis=1))) >= 0 else -1.0
    return AnalyticSurface("cone", apex, axis, 0.0, half_angle, residual, sense)


def _fit_surface(
    kind: str,
    points: np.ndarray,
    normals: np.ndarray,
    tolerance: float,
    solid_normals: np.ndarray | None = None,
) -> AnalyticSurface:
    """Fit the structurally known surface type, or fall back to freeform."""
    empty = AnalyticSurface(
        "freeform",
        points.mean(axis=0) if points.size else np.zeros(3),
        np.zeros(3),
        0.0,
        0.0,
        float("inf"),
        1.0,
    )
    if points.shape[0] < 3:
        return empty
    outward = normals if solid_normals is None else solid_normals
    try:
        if kind == "plane":
            fitted = _fit_plane(points, normals)
        elif kind == "sphere":
            fitted = _fit_sphere(points, outward)
        elif kind == "cylinder":
            fitted = _fit_cylinder(points, normals, outward)
        elif kind == "cone":
            fitted = _fit_cone(points, normals, outward)
        else:
            return empty
    except np.linalg.LinAlgError:
        return empty
    if not np.isfinite(fitted.residual) or fitted.residual > tolerance:
        return AnalyticSurface(
            "freeform",
            fitted.origin,
            fitted.axis,
            fitted.radius,
            fitted.half_angle,
            fitted.residual,
            fitted.sense,
        )
    return fitted


# ── extraction ───────────────────────────────────────────────────────────────


def _quad_centroids(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    return vertices[quads].mean(axis=1)


def _quad_areas(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    points = vertices[quads]
    rolled = np.roll(points, -1, axis=1)
    return np.linalg.norm(0.5 * np.sum(np.cross(points, rolled), axis=1), axis=1)


def _project_scene(scene: Any, points: np.ndarray, max_step: float, steps: int) -> np.ndarray:
    """One-field projection of points onto the *scene's* zero set."""
    return project_fields([scene], points, max_step=max_step, steps=steps)


def _solve_owned_points(
    patches: list[Patch],
    owner_sets: list[tuple[int, ...]],
    seeds: np.ndarray,
    max_step: float,
    steps: int,
) -> np.ndarray:
    """Re-solve seeds grouped by their owning patch set, one call per arity.

    This is where the graph stops trusting dual contouring: every point is
    replaced by the solution of its own 1-, 2- or 3-field system.  Points
    whose owner set is empty (a blend neighbourhood, where no patch owns the
    surface) keep their seed.  Grouping is by *arity* rather than by patch
    set — :func:`~cadjoint.brep.project.project_batched` lets every point
    gather its own fields, so three programs cover the whole mesh instead of
    one per distinct subset.
    """
    solved = np.asarray(seeds, dtype=np.float64).copy()
    field_table = [patch.field for patch in patches]
    by_arity: dict[int, list[int]] = {}
    for row, owners in enumerate(owner_sets):
        if owners and len(owners) <= 3:
            by_arity.setdefault(len(owners), []).append(row)
    for _arity, rows in sorted(by_arity.items()):
        index = np.asarray(rows, dtype=np.int64)
        members = np.asarray([owner_sets[row] for row in rows], dtype=np.int32)
        solved[index] = project_batched(
            field_table, members, solved[index], max_step=max_step, steps=steps
        )
    return solved


def extract_brep(
    scene: Any,
    grid: GridSpec,
    *,
    mesh: Mesh | None = None,
    blend_tolerance: float | None = None,
    fit_tolerance: float | None = None,
    max_step: float | None = None,
    steps: int = 8,
    fit_surfaces: bool = True,
) -> BRep:
    """Derive a B-rep ownership graph from a scene and a sampling grid.

    Args:
        scene: Root SDF node.
        grid: The sampling grid dual contouring discovers the graph on.
        mesh: A dual-contour mesh already extracted on ``grid``; extracted
            here when omitted.
        blend_tolerance: ``|f_patch|`` above which a surface point is
            declared to lie on a blend rather than on its patch.  Defaults
            to ``1e-3`` times the grid diagonal, which is far below any
            usable blend radius and far above projection noise.
        fit_tolerance: Largest sample deviation an analytic surface fit may
            have and still be certified.  Defaults to ``1e-2`` times the
            cell diagonal.
        max_step: Displacement clamp for every projection; defaults to half
            the cell diagonal, the clamp the tet mesher uses.
        steps: Newton iterations per projection.  Eight converges from a
            cold seed anywhere in its cell; a caller whose seeds are already
            within a fraction of a cell (the viewer overlay, which seeds
            from mesh-edge midpoints) can halve this and halve the cost,
            because the cost of an eager JAX program is per call and the
            loop is unrolled.
        fit_surfaces: Whether to fit and certify each analytic face's closed
            form.  A caller that only wants the graph's topology and its
            edge curves — the viewer overlay does — can turn this off and
            skip one projection program plus a gradient program per face;
            every face then carries a ``freeform`` surface and
            :attr:`BRepFace.analytic` is ``False`` throughout, while
            :attr:`BRepFace.kind` and every edge stay exactly as they were.

    Returns:
        The :class:`BRep`.

    Raises:
        ValueError: If the extracted mesh has no quads.
    """
    decomposition = scene_patch_fields(scene)
    patches, offsets = _patch_table(decomposition)
    if mesh is None:
        mesh = extract_mesh(scene, grid)
    quads = np.asarray(mesh.quads, dtype=np.int64)
    if quads.shape[0] == 0:
        raise ValueError("The extraction produced no quads; nothing to derive a B-rep from.")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    spacing = np.asarray(grid.spacing, dtype=np.float64)
    cell_diagonal = float(np.linalg.norm(spacing))
    if max_step is None:
        max_step = 0.5 * cell_diagonal
    if blend_tolerance is None:
        blend_tolerance = 1e-3 * float(np.linalg.norm(spacing * np.asarray(grid.cells)))
    if fit_tolerance is None:
        fit_tolerance = 1e-2 * cell_diagonal

    # 1. Ownership of every quad, decided on the scene's own zero set so the
    #    blend test compares like with like.
    centroids = _project_scene(scene, _quad_centroids(vertices, quads), max_step, steps)
    owner, magnitude = _own_patch(decomposition, offsets, centroids)
    blend = magnitude > blend_tolerance
    # A blend quad keeps its leaf (fillets belong to a solid) but loses its
    # patch, so blend bands split per leaf instead of fusing into one face.
    leaf_of_patch = np.asarray([patch.leaf for patch in patches], dtype=np.int64)
    region_key = np.where(blend, -1 - leaf_of_patch[owner], owner)

    # 2. Faces are connected components of same-key quads.
    neighbors, edge_users = _quad_adjacency(quads)
    quad_face = _components(neighbors, region_key)
    face_count = int(quad_face.max()) + 1
    areas = _quad_areas(vertices, quads)

    # 2b. Every analytic face's fit samples in ONE projection program.  Per
    #     face would be hundreds of JAX calls, and the cost of a call in
    #     eager mode is independent of how many points it moves.
    field_table = [patch.field for patch in patches]
    regions = [np.flatnonzero(quad_face == face_id) for face_id in range(face_count)]
    sample_blocks: list[np.ndarray] = []
    sample_members: list[np.ndarray] = []
    sample_spans: dict[int, tuple[int, int]] = {}
    cursor = 0
    for face_id, region in enumerate(regions):
        key = int(region_key[region[0]])
        if key < 0 or not fit_surfaces:
            continue
        # Corners as well as centroids: a one-quad sliver has a single
        # centroid, and a plane cannot be fitted through one point.
        block = np.concatenate([centroids[region], vertices[np.unique(quads[region])]])
        sample_blocks.append(block)
        sample_members.append(np.full((block.shape[0], 1), key, dtype=np.int32))
        sample_spans[face_id] = (cursor, cursor + block.shape[0])
        cursor += block.shape[0]
    if sample_blocks:
        fitted_samples = project_batched(
            field_table,
            np.concatenate(sample_members),
            np.concatenate(sample_blocks),
            max_step=max_step,
            steps=steps,
        )
        # The solid's own outward normal at every sample, in one program:
        # it decides each fitted surface's ``sense`` (a subtracted cylinder's
        # patch field grows away from its axis whichever side the material
        # is on, so the patch gradient cannot answer that).
        solid_normals = _patch_normals(scene, fitted_samples)
    else:
        fitted_samples = np.zeros((0, 3))
        solid_normals = np.zeros((0, 3))

    faces: list[BRepFace] = []
    non_simple = 0
    for face_id, region in enumerate(regions):
        key = int(region_key[region[0]])
        is_blend = key < 0
        patch_index = -1 if is_blend else int(owner[region[0]])
        leaf_id = -1 - key if is_blend else patches[patch_index].leaf
        kind = "blend" if is_blend else patches[patch_index].kind
        if is_blend or not fit_surfaces:
            surface = AnalyticSurface(
                "freeform",
                centroids[region].mean(axis=0),
                np.zeros(3),
                0.0,
                0.0,
                float("inf"),
                1.0,
            )
        else:
            low, high = sample_spans[face_id]
            samples = fitted_samples[low:high]
            normals = _patch_normals(patches[patch_index].field, samples)
            surface = _fit_surface(kind, samples, normals, fit_tolerance, solid_normals[low:high])
        loops = _region_loops(quads, region)
        if not loops:
            non_simple += 1
        faces.append(
            BRepFace(
                index=face_id,
                patch=patch_index,
                leaf=int(leaf_id),
                kind=kind,
                analytic=surface.kind in ANALYTIC_KINDS,
                surface=surface,
                quads=region,
                loops=loops,
                area=float(areas[region].sum()),
            )
        )

    # 3. Per-vertex ownership: the distinct faces a mesh vertex touches
    #    decide the arity of its projection.
    vertex_faces: list[set[int]] = [set() for _ in range(vertices.shape[0])]
    for quad_id in range(quads.shape[0]):
        for index in quads[quad_id]:
            vertex_faces[int(index)].add(int(quad_face[quad_id]))

    owner_sets: list[tuple[int, ...]] = []
    owner_patches = np.full((vertices.shape[0], 3), -1, dtype=np.int64)
    owner_arity = np.zeros(vertices.shape[0], dtype=np.int64)
    for index, incident in enumerate(vertex_faces):
        patch_ids = sorted({faces[face_id].patch for face_id in incident})
        if not incident or -1 in patch_ids or not 1 <= len(patch_ids) <= 3:
            owner_sets.append(())
            continue
        owner_sets.append(tuple(patch_ids))
        owner_patches[index, : len(patch_ids)] = patch_ids
        owner_arity[index] = len(patch_ids)
    points = _solve_owned_points(patches, owner_sets, vertices, max_step, steps)

    # 4. Edges: the boundary chains between two faces.
    edges, edge_of_pair, edge_ends = _build_edges(
        faces, patches, edge_users, quad_face, points, max_step, steps
    )

    # 5. Vertices: mesh vertices whose incident faces number three or more.
    vertices_out, ambiguous = _build_vertices(faces, vertex_faces, patches, points)
    edges = _link_edge_vertices(edges, edge_ends, vertices_out)

    stats = {
        "quads": int(quads.shape[0]),
        "mesh_vertices": int(vertices.shape[0]),
        "blend_quads": int(blend.sum()),
        "non_simple_faces": non_simple,
        "ambiguous_vertices": ambiguous,
        "edge_pairs": len(edge_of_pair),
        "tangent_or_blend_edges": sum(1 for edge in edges if not edge.analytic),
        "freeform_faces": sum(1 for face in faces if face.surface.kind == "freeform"),
    }
    return BRep(
        faces=faces,
        edges=edges,
        vertices=vertices_out,
        patches=patches,
        mesh=mesh,
        grid=grid,
        points=points,
        owner_patches=owner_patches,
        owner_arity=owner_arity,
        quad_face=quad_face,
        decomposition=decomposition,
        stats=stats,
    )


def _chain_segments(segments: list[tuple[int, int]]) -> list[list[int]]:
    """Order undirected segments into maximal vertex chains.

    A face pair usually meets along one curve, but a bore through a plate
    gives two, and a chain that closes on itself (a cylinder rim) has no
    endpoints — all three come out of the same walk.
    """
    adjacency: dict[int, list[int]] = {}
    for a, b in segments:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    unused = {(min(a, b), max(a, b)) for a, b in segments}
    chains: list[list[int]] = []
    endpoints = [node for node, links in adjacency.items() if len(links) != 2]
    starts = sorted(endpoints) + sorted(adjacency)
    for start in starts:
        for neighbor in adjacency.get(start, []):
            key = (min(start, neighbor), max(start, neighbor))
            if key not in unused:
                continue
            unused.discard(key)
            chain = [start, neighbor]
            current, previous = neighbor, start
            while True:
                step = None
                for candidate in adjacency[current]:
                    edge = (min(current, candidate), max(current, candidate))
                    if candidate != previous and edge in unused:
                        step = candidate
                        break
                if step is None:
                    break
                unused.discard((min(current, step), max(current, step)))
                chain.append(step)
                previous, current = current, step
                if step == chain[0]:
                    break
            chains.append(chain)
    return chains


def _build_edges(
    faces: list[BRepFace],
    patches: list[Patch],
    edge_users: dict[tuple[int, int], list[int]],
    quad_face: np.ndarray,
    points: np.ndarray,
    max_step: float,
    steps: int,
) -> tuple[list[BRepEdge], dict[tuple[int, int], list[int]], np.ndarray]:
    """Chain the mesh edges that separate two faces into B-rep edges.

    Every analytic chain's seeds are projected in one batched call, for the
    reason :func:`_solve_owned_points` gives: a body has a hundred-odd edges
    and a hundred-odd JAX programs is the whole cost.

    Returns:
        ``(edges, edge_of_pair, ends)`` — the edges, the edge indices per
            face pair, and per edge the two mesh vertices its chain ended
            on (``-1`` for a closed chain), which
            :func:`_link_edge_vertices` turns into :class:`BRepVertex`
            endpoints once the vertices exist.
    """
    pair_segments: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (a, b), owners in edge_users.items():
        incident = {int(quad_face[owner]) for owner in owners}
        if len(incident) != 2:
            continue
        left, right = sorted(incident)
        pair_segments.setdefault((left, right), []).append((a, b))

    raw: list[tuple[tuple[int, int], tuple[int, int], np.ndarray, bool, bool, tuple[int, int]]] = []
    for (left, right), segments in sorted(pair_segments.items()):
        patch_a, patch_b = faces[left].patch, faces[right].patch
        analytic = patch_a >= 0 and patch_b >= 0
        for chain in _chain_segments(segments):
            closed = len(chain) > 2 and chain[0] == chain[-1]
            nodes = chain[:-1] if closed else chain
            if closed:
                seeds = 0.5 * (points[nodes] + points[np.roll(nodes, -1)])
            else:
                seeds = 0.5 * (points[nodes[:-1]] + points[nodes[1:]])
            ends = (-1, -1) if closed else (int(chain[0]), int(chain[-1]))
            raw.append(((left, right), (patch_a, patch_b), seeds, closed, analytic, ends))

    field_table = [patch.field for patch in patches]
    blocks = [entry[2] for entry in raw if entry[4] and entry[2].shape[0]]
    members = [
        np.tile(np.asarray(entry[1], dtype=np.int32), (entry[2].shape[0], 1))
        for entry in raw
        if entry[4] and entry[2].shape[0]
    ]
    if blocks:
        stacked = np.concatenate(blocks)
        stacked_members = np.concatenate(members)
        solved = project_batched(
            field_table, stacked_members, stacked, max_step=max_step, steps=steps
        )
        residual_all = batched_residuals(field_table, stacked_members, solved)
    else:
        solved = np.zeros((0, 3))
        residual_all = np.zeros(0)

    edges: list[BRepEdge] = []
    cursor = 0
    for face_pair, patch_pair, seeds, closed, analytic, _ends in raw:
        if analytic and seeds.shape[0]:
            span = slice(cursor, cursor + seeds.shape[0])
            polyline = solved[span]
            residual = float(residual_all[span].max())
            cursor += seeds.shape[0]
        else:
            polyline = np.asarray(seeds, dtype=np.float64)
            residual = float("inf")
        edges.append(
            BRepEdge(
                index=len(edges),
                faces=face_pair,
                patches=patch_pair,
                polyline=polyline,
                vertices=(-1, -1),
                closed=closed,
                analytic=analytic,
                residual=residual,
            )
        )
    edge_of_pair: dict[tuple[int, int], list[int]] = {}
    for edge in edges:
        edge_of_pair.setdefault(edge.faces, []).append(edge.index)
    ends = np.asarray([entry[5] for entry in raw], dtype=np.int64).reshape(len(raw), 2)
    return edges, edge_of_pair, ends


def _link_edge_vertices(
    edges: list[BRepEdge], ends: np.ndarray, vertices: list[BRepVertex]
) -> list[BRepEdge]:
    """Name each open edge's triple-point endpoints, once the vertices exist.

    An edge chain ends on a mesh vertex; that mesh vertex becomes a
    :class:`BRepVertex` exactly when three or more faces meet there.  The
    two passes run in that order, so the endpoints are attached here rather
    than left at the ``(-1, -1)`` :func:`_build_edges` emits.

    The link is what lets a consumer close the graph at its corners: an
    edge's polyline is seeded from mesh-edge *midpoints*, so it stops half a
    cell short of the corner, and only the vertex says where the corner
    actually is.

    Args:
        edges: Edges as :func:`_build_edges` returned them.
        ends: Per edge, the two mesh vertices its chain ended on, ``-1``
            for a closed chain.
        vertices: The solved triple points.

    Returns:
        The edges with :attr:`BRepEdge.vertices` filled in.
    """
    vertex_of_mesh = {vertex.mesh_vertex: vertex.index for vertex in vertices}
    linked: list[BRepEdge] = []
    for edge, (start, stop) in zip(edges, ends):
        linked.append(
            replace(
                edge,
                vertices=(
                    vertex_of_mesh.get(int(start), -1),
                    vertex_of_mesh.get(int(stop), -1),
                ),
            )
        )
    return linked


def _build_vertices(
    faces: list[BRepFace],
    vertex_faces: list[set[int]],
    patches: list[Patch],
    points: np.ndarray,
) -> tuple[list[BRepVertex], int]:
    """Solve every mesh vertex that three or more faces meet at.

    A clean corner has exactly three incident faces and three analytic
    patches.  Four faces meeting at a point, or a blend among them, is a
    genuine ambiguity of the *scene* rather than of this code — it is
    counted and reported, not silently resolved.
    """
    candidates = [index for index, incident in enumerate(vertex_faces) if len(incident) >= 3]
    field_table = [patch.field for patch in patches]
    rows: list[int] = []
    members: list[tuple[int, ...]] = []
    prepared: list[tuple[int, tuple[int, ...], tuple[int, ...], bool]] = []
    ambiguous = 0
    for index in candidates:
        incident = tuple(sorted(vertex_faces[index]))
        patch_ids = sorted({faces[face_id].patch for face_id in incident})
        if len(incident) > 3 or -1 in patch_ids:
            ambiguous += 1
        usable = tuple(pid for pid in patch_ids if pid >= 0)[:3]
        analytic = len(usable) == 3 and len(incident) == 3
        prepared.append((index, incident, usable, analytic))
        if analytic:
            rows.append(index)
            members.append(usable)
    if rows:
        residual_all = batched_residuals(
            field_table, np.asarray(members, dtype=np.int32), points[np.asarray(rows)]
        )
    else:
        residual_all = np.zeros(0)

    result: list[BRepVertex] = []
    cursor = 0
    for index, incident, usable, analytic in prepared:
        if analytic:
            residual = float(residual_all[cursor])
            cursor += 1
        else:
            residual = float("inf")
        result.append(
            BRepVertex(
                index=len(result),
                mesh_vertex=index,
                faces=incident,
                patches=usable,
                point=points[index].copy(),
                analytic=analytic,
                residual=residual,
            )
        )
    return result, ambiguous
