"""Meshing from the graph: a piecewise-linear complex with owned nodes.

The simulation mesh today is built from the dual-contour quad soup: TetGen
gets one triangle pair per grid cell that the surface crosses, preserves them
exactly (``-Y``), and fills the inside.  The surface triangulation is then
*lattice-driven* — its edge lengths, its aspect ratios and its node count come
from where the grid cut the model, not from the model.

The graph offers a different input.  A planar face is a polygon; it does not
need one triangle per cell to be represented exactly, it needs its own
boundary.  So the PLC handed to TetGen is:

- every analytic **planar** face re-triangulated from its simplified boundary
  loop — for the starter's fin comb, twenty-four polygons instead of six
  thousand quads,
- every **curved or blend** face kept as its dual-contour triangles, because
  a blend has no closed form and its tessellation *is* its definition,
- with the vertices shared by a coarsened and an uncoarsened face pinned, so
  the complex stays conforming across that boundary.

Every node keeps its owner: the patch set it was solved against
(:attr:`~cadjoint.brep.graph.BRep.owner_patches`).  That is what makes the
mesh differentiable in the same way the graph is — a node on a face moves by
a one-field projection, a node on an edge by two, a corner by three, all
through :func:`cadjoint.brep.project.project_batched`, and the Steiner nodes
TetGen adds follow through the existing Laplacian
(:func:`cadjoint.fem.motion.smooth_interior_delta`).

This module is a measured spike, not the replacement: it builds the complex,
meshes it, and reports quality against the current path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cadjoint.brep.graph import BRep
from cadjoint.brep.project import project_batched
from cadjoint.brep.step import brep_loops, brep_triangles

__all__ = ["PLC", "brep_plc", "plc_quality", "plc_tet_mesh", "recompute_plc_points"]


@dataclass(frozen=True)
class PLC:
    """A piecewise-linear complex derived from the ownership graph.

    Attributes:
        points: Vertex positions ``(n, 3)``, a compaction of
            :attr:`BRep.points`.
        triangles: Watertight connectivity ``(t, 3)``, outward wound.
        owner_patches: Per point, the global patch indices it was solved
            against, ``(n, 3)`` and ``-1``-padded.
        owner_arity: Per point, how many of those there are (0 where the
            point sits in a blend neighbourhood and keeps its dual-contour
            position).
        face_of_triangle: Owning B-rep face of each triangle.
        coarsened: Face indices that were re-triangulated from their loops.
        stats: Counts, including the triangle reduction against the
            dual-contour tessellation.
    """

    points: np.ndarray
    triangles: np.ndarray
    owner_patches: np.ndarray
    owner_arity: np.ndarray
    face_of_triangle: np.ndarray
    coarsened: list[int]
    stats: dict


def _plane_basis(normal: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, helper)
    first /= np.linalg.norm(first)
    return np.stack([first, np.cross(normal, first)])


def _ear_clip(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple counterclockwise 2D polygon by ear clipping.

    Args:
        polygon: Vertices ``(m, 2)``, counterclockwise, without repetition.

    Returns:
        Triangles as index triples into ``polygon``; empty when the polygon
            is degenerate or not simple enough to clip.
    """
    count = polygon.shape[0]
    if count < 3:
        return []
    remaining = list(range(count))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3 and guard < 4 * count:
        guard += 1
        clipped = False
        for position in range(len(remaining)):
            previous = remaining[position - 1]
            current = remaining[position]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = polygon[previous], polygon[current], polygon[following]
            cross = float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if cross <= 0.0:
                continue  # Reflex or collinear: not an ear.
            others = [index for index in remaining if index not in (previous, current, following)]
            if others and _any_inside(polygon[others], a, b, c):
                continue
            triangles.append((previous, current, following))
            remaining.pop(position)
            clipped = True
            break
        if not clipped:
            return []
    if len(remaining) == 3:
        triangles.append(tuple(remaining))  # type: ignore[arg-type]
    return triangles


def _any_inside(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Whether any of ``points`` lies strictly inside triangle ``abc``."""
    v0, v1, v2 = b - a, c - a, points - a
    denominator = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denominator) < 1e-30:
        return False
    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / denominator
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / denominator
    return bool(np.any((u > 1e-12) & (v > 1e-12) & (u + v < 1.0 - 1e-12)))


def brep_plc(brep: BRep, *, coarsen: bool = True, tolerance: float | None = None) -> PLC:
    """Build a piecewise-linear complex from the ownership graph.

    Args:
        brep: The extracted graph.
        coarsen: Re-triangulate single-loop planar faces from their
            simplified boundary loops.  ``False`` keeps every dual-contour
            triangle, which isolates the effect of the re-projection alone.
        tolerance: Loop-simplification tolerance; see
            :func:`~cadjoint.brep.step.brep_loops`.

    Returns:
        The :class:`PLC`.
    """
    triangles, face_ids = brep_triangles(brep)
    coarsenable: list[int] = []
    blocked = 0
    if coarsen:
        candidate = {
            face.index
            for face in brep.faces
            if face.analytic and face.kind == "plane" and len(face.loops) == 1
        }
        # All-or-nothing across a shared edge.  Coarsening one side of a
        # boundary and not the other leaves the kept side splitting a
        # segment the coarsened side draws whole — a T-junction, and TetGen
        # sees a crack.  Pinning the shared vertices instead only moves the
        # problem into the triangulation, where a facet with collinear
        # boundary vertices has no ear to clip.  So a face coarsens only
        # when every face it borders does too, and the count of the ones
        # this rules out is reported rather than papered over.
        neighbours: dict[int, set[int]] = {face.index: set() for face in brep.faces}
        for edge in brep.edges:
            left, right = edge.faces
            neighbours[left].add(right)
            neighbours[right].add(left)
        changed = True
        while changed:
            changed = False
            for index in sorted(candidate):
                if not neighbours[index].issubset(candidate):
                    candidate.discard(index)
                    blocked += 1
                    changed = True
        coarsenable = sorted(candidate)
        loops = brep_loops(brep, tolerance=tolerance)
    else:
        loops = [[] for _ in brep.faces]

    output: list[np.ndarray] = []
    owners: list[int] = []
    coarsened: list[int] = []
    for face_index in coarsenable:
        loop = loops[face_index][0]
        if len(loop) < 3:
            continue
        normal = np.asarray(brep.faces[face_index].surface.axis, dtype=np.float64)
        basis = _plane_basis(normal)
        planar = brep.points[loop] @ basis.T
        clipped = _ear_clip(planar)
        if not clipped:
            continue  # Non-simple after simplification: keep the DC triangles.
        indices = np.asarray(loop, dtype=np.int64)
        output.append(indices[np.asarray(clipped, dtype=np.int64)])
        owners.extend([face_index] * len(clipped))
        coarsened.append(face_index)
    replaced = np.isin(face_ids, coarsened)
    output.append(triangles[~replaced].astype(np.int64))
    owners.extend(int(face_id) for face_id in face_ids[~replaced])

    connectivity = np.concatenate(output).astype(np.int64)
    used = np.unique(connectivity)
    remap = np.full(brep.points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return PLC(
        points=brep.points[used].copy(),
        triangles=remap[connectivity].astype(np.int32),
        owner_patches=brep.owner_patches[used].copy(),
        owner_arity=brep.owner_arity[used].copy(),
        face_of_triangle=np.asarray(owners, dtype=np.int64),
        coarsened=coarsened,
        stats={
            "dc_triangles": int(triangles.shape[0]),
            "plc_triangles": int(connectivity.shape[0]),
            "dc_points": int(brep.points.shape[0]),
            "plc_points": int(used.size),
            "coarsened_faces": len(coarsened),
            "coarsenable_faces": len(coarsenable),
            "coarsen_blocked_faces": blocked,
        },
    )


def plc_tet_mesh(brep: BRep, plc: PLC, *, min_ratio: float = 1.5, min_dihedral: float = 10.0):
    """Tet-mesh a PLC through the existing TetGen path.

    Args:
        brep: The graph the PLC came from (its grid sets the motion clamp).
        plc: The complex to fill.
        min_ratio: TetGen radius-edge bound.
        min_dihedral: TetGen minimum dihedral angle in degrees.

    Returns:
        A :class:`~cadjoint.fem.tetmesh.TetMesh` whose leading
            ``num_surface`` nodes are the PLC's, in order.

    Raises:
        ImportError: If tetgen is not installed.
        RuntimeError: If TetGen rejects the complex.
    """
    from cadjoint.fem.tetmesh import surface_to_tet_mesh

    return surface_to_tet_mesh(
        plc.points,
        plc.triangles,
        grid=brep.grid,
        min_ratio=min_ratio,
        min_dihedral=min_dihedral,
    )


def recompute_plc_points(
    brep: BRep, plc: PLC, mesh: Any, *, smooth_passes: int = 0, max_step: float | None = None
) -> np.ndarray:
    """Move a PLC-based tet mesh under the graph's own projections.

    Every boundary node is re-solved against *its own* owner patches — one
    field for a face node, two for an edge node, three for a corner — instead
    of the single scene projection
    :func:`cadjoint.fem.motion.recompute_tet_points` uses.  A corner
    therefore stays a corner exactly, rather than sliding along whichever
    branch the scene SDF happens to select there.

    Args:
        brep: The graph (its patch table supplies the fields).
        plc: The complex whose points are the mesh's leading nodes.
        mesh: The :class:`~cadjoint.fem.tetmesh.TetMesh` built from it.
        smooth_passes: Laplacian passes carrying boundary motion inward.
        max_step: Projection clamp; defaults to the mesh's own.

    Returns:
        Corner-node positions, shaped like ``mesh.points[:num_corner_points]``.
    """
    import jax.numpy as jnp

    from cadjoint.fem.motion import smooth_interior_delta

    if max_step is None:
        max_step = mesh.max_step
    surface = np.asarray(mesh.base_points[: mesh.num_surface], dtype=np.float64)
    field_table = [patch.field for patch in brep.patches]
    solved = surface.copy()
    by_arity: dict[int, list[int]] = {}
    for row in range(mesh.num_surface):
        arity = int(plc.owner_arity[row])
        if arity:
            by_arity.setdefault(arity, []).append(row)
    for arity, rows in sorted(by_arity.items()):
        index = np.asarray(rows, dtype=np.int64)
        members = plc.owner_patches[index, :arity].astype(np.int32)
        solved[index] = project_batched(field_table, members, solved[index], max_step=max_step)
    base = np.asarray(mesh.base_points[: mesh.num_corner_points], dtype=np.float64)
    if smooth_passes <= 0:
        return np.concatenate([solved, base[mesh.num_surface :]], axis=0)
    delta = smooth_interior_delta(
        mesh, jnp.asarray(solved - base[: mesh.num_surface]), smooth_passes
    )
    return base + np.asarray(delta, dtype=np.float64)


def plc_quality(points: np.ndarray, cells: np.ndarray) -> dict:
    """Summary tet-quality statistics, using the FEM suite's own metrics.

    Args:
        points: Node positions ``(n, 3)``.
        cells: Tet connectivity ``(t, 4)`` or ``(t, 10)``.

    Returns:
        Tet and node counts, the radius-ratio floor / first percentile /
            mean (1 is regular, 0 is a sliver), the aspect-ratio ceiling and
            mean, the smallest tet volume, and the total volume.
    """
    from cadjoint.fem.quality import tet_aspect_ratios, tet_radius_ratios, tet_volumes

    radius = tet_radius_ratios(points, cells)
    aspect = tet_aspect_ratios(points, cells)
    volumes = tet_volumes(points, cells)
    return {
        "tets": int(np.asarray(cells).shape[0]),
        "nodes": int(np.asarray(points).shape[0]),
        "radius_ratio_min": float(radius.min()),
        "radius_ratio_p01": float(np.percentile(radius, 1.0)),
        "radius_ratio_mean": float(radius.mean()),
        "aspect_max": float(aspect.max()),
        "aspect_mean": float(aspect.mean()),
        "volume_min": float(volumes.min()),
        "volume": float(volumes.sum()),
    }
