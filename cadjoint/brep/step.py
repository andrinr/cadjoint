"""Export from the ownership graph: analytic STEP, and OBJ/STL beside it.

:mod:`cadjoint.meshing.export` writes a *mesh*: it recovers flat regions by
growing coplanar quads under an angle threshold and emits one ``PLANE`` per
region, with everything curved left as triangles.  That is the best a
consumer of a quad soup can do.

The graph knows more.  A face already *is* one patch's zero set, so its
surface type is known rather than recovered, its boundary loops come from
ownership rather than from a 1° threshold, and its loop vertices have been
re-solved onto the exact intersection curves they lie on.  Three consequences
show up in the file:

- **Loops collapse.**  A box face's boundary is 64 dual-contour vertices, but
  they lie on four straight lines to within float noise, so the exported
  ``FACE_OUTER_BOUND`` has four vertices.  The collapse is measured
  (:func:`simplify_loop` keeps a deviation bound), not assumed.
- **Holes survive.**  A plate with a bore is one face with two loops —
  ``FACE_OUTER_BOUND`` plus ``FACE_BOUND`` — where the merge-based writer has
  to give up and fall back to triangles.
- **Cylinders stay cylinders.**  A full cylindrical band bounded by two rim
  circles is emitted as ``CYLINDRICAL_SURFACE`` with ``CIRCLE`` edges, with
  no faceting at all.

Everything the graph cannot certify — blend faces, and any curved face whose
trimming loops are not circles — falls back to the faceted representation, so
a file is always complete.  Which faces took which path is returned, so a
caller can see exactly how much of the model is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cadjoint.brep.graph import BRep, BRepFace
from cadjoint.meshing.export import (
    _STEP_BOILERPLATE,
    _STEP_HEADER,
    _polygon_normals,
    _step_point,
    _step_real,
    _unit,
)

__all__ = [
    "brep_loops",
    "brep_triangles",
    "brep_volume",
    "save_brep_obj",
    "save_brep_step",
    "save_brep_stl",
    "simplify_loop",
]


def simplify_loop(
    points: np.ndarray,
    loop: list[int],
    tolerance: float,
    protected: set[int] | None = None,
) -> list[int]:
    """Drop loop vertices that lie on the straight run between their neighbours.

    A dual-contour boundary walks one vertex per cell along a straight edge;
    once those vertices have been re-solved onto the exact edge curve they
    are collinear, and keeping them says nothing.  A vertex is removed when
    its distance to the segment joining the surviving neighbours stays under
    ``tolerance``, which bounds how far the simplified loop can move.

    ``protected`` is what keeps a shell sewable.  A boundary is shared with
    the face on the other side; simplifying it on one side only leaves two
    faces claiming different curves between the same two points, and no
    kernel will sew that.  So a vertex any *faceted* face still uses stays.

    Args:
        points: Vertex positions ``(n, 3)``.
        loop: Ordered vertex indices.
        tolerance: Maximum deviation permitted per removed vertex.
        protected: Vertex indices that must be kept.

    Returns:
        The kept indices, in order; the input is returned unchanged when
            fewer than four vertices would survive.
    """
    if len(loop) < 4:
        return list(loop)
    keep_set = protected or set()
    positions = points[np.asarray(loop, dtype=np.int64)]
    previous = np.roll(positions, 1, axis=0)
    following = np.roll(positions, -1, axis=0)
    span = following - previous
    lengths = np.linalg.norm(span, axis=1)
    cross = np.linalg.norm(np.cross(span, positions - previous), axis=1)
    offset = np.where(lengths > 0, cross / np.where(lengths > 0, lengths, 1.0), lengths)
    # Judged against each vertex's ORIGINAL neighbours, never against the
    # survivors of an earlier removal.  The order-dependent greedy version
    # keeps different subsets of one straight run depending on where the
    # walk started, and the two faces sharing that run then disagree about
    # the curve between the same two points — eight free edges in a shell of
    # nineteen hundred, measured, and no solid.
    kept = [
        index
        for position, index in enumerate(loop)
        if offset[position] > tolerance or index in keep_set
    ]
    return kept if len(kept) >= 3 else list(loop)


def _weld_aligned(
    points: np.ndarray, loops: list[list[int]], tolerance: float = 1e-7
) -> list[list[int]]:
    """:func:`~cadjoint.meshing.export._weld_degenerate_edges`, index-aligned.

    The meshing version drops loops that welding dissolves, which loses the
    correspondence between a loop and the face it belongs to.  Here every
    input loop keeps its slot, empty when it dissolved — one union-find over
    all of them, so the analytic faces and the facets agree on which
    vertices coincide.
    """
    parent: dict[int, int] = {}

    def find(index: int) -> int:
        root = index
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(index, index) != index:
            parent[index], index = root, parent[index]
        return root

    for loop in loops:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if a != b and float(np.linalg.norm(points[b] - points[a])) <= tolerance:
                left, right = find(a), find(b)
                if left != right:
                    parent[max(left, right)] = min(left, right)
    result: list[list[int]] = []
    for loop in loops:
        mapped = [find(index) for index in loop]
        collapsed = [v for i, v in enumerate(mapped) if v != mapped[i - 1]]
        result.append(collapsed if len(collapsed) >= 3 else [])
    return result


def brep_loops(
    brep: BRep, *, tolerance: float | None = None, protected: set[int] | None = None
) -> list[list[list[int]]]:
    """Per face, its boundary loops simplified onto the exact curves.

    Args:
        brep: The extracted graph.
        tolerance: Deviation bound for :func:`simplify_loop`; defaults to
            ``1e-6`` times the grid diagonal, far below any modelling
            feature and far above the projection's own residual.
        protected: Vertex indices no loop may drop — see
            :func:`simplify_loop`.

    Returns:
        One list of index loops per face, indexing :attr:`BRep.points`.
    """
    if tolerance is None:
        extent = np.asarray(brep.grid.spacing) * np.asarray(brep.grid.cells)
        tolerance = 1e-6 * float(np.linalg.norm(extent))
    return [
        [simplify_loop(brep.points, loop, tolerance, protected) for loop in face.loops]
        for face in brep.faces
    ]


def brep_triangles(brep: BRep) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate the whole graph over the re-solved points.

    Each quad is split along its shorter diagonal on the *projected*
    positions, and triangles that projection collapsed to zero area (two
    corners of a quad landing on the same edge point) are dropped.

    Args:
        brep: The extracted graph.

    Returns:
        ``(triangles, face_ids)`` — connectivity ``(t, 3)`` into
            :attr:`BRep.points`, and the owning face of each triangle.
    """
    points = brep.points
    quads = np.asarray(brep.mesh.quads, dtype=np.int64)
    a, b, c, d = quads.T
    diagonal_ac = np.sum((points[a] - points[c]) ** 2, axis=-1)
    diagonal_bd = np.sum((points[b] - points[d]) ** 2, axis=-1)
    shorter = (diagonal_ac <= diagonal_bd)[:, None]
    first = np.where(shorter, np.stack([a, b, c], axis=1), np.stack([a, b, d], axis=1))
    second = np.where(shorter, np.stack([a, c, d], axis=1), np.stack([b, c, d], axis=1))
    triangles = np.stack([first, second], axis=1).reshape((-1, 3))
    face_ids = np.repeat(brep.quad_face, 2)
    edges_ab = points[triangles[:, 1]] - points[triangles[:, 0]]
    edges_ac = points[triangles[:, 2]] - points[triangles[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(edges_ab, edges_ac), axis=1)
    keep = areas > 0.0
    return triangles[keep].astype(np.int32), face_ids[keep]


def brep_volume(brep: BRep) -> float:
    """Enclosed volume of the graph's tessellation, by the divergence theorem.

    Args:
        brep: The extracted graph.

    Returns:
        Signed volume; positive for outward-wound triangles.
    """
    triangles, _ = brep_triangles(brep)
    corners = brep.points[triangles]
    return float(
        np.sum(np.einsum("ti,ti->t", corners[:, 0], np.cross(corners[:, 1], corners[:, 2]))) / 6.0
    )


def save_brep_obj(brep: BRep, path: str | Path) -> None:
    """Write the graph as OBJ: n-gons for simple analytic faces, triangles else.

    A face with a single boundary loop becomes one polygon; a face with holes
    (which OBJ cannot express) falls back to its triangles, as does every
    blend face.

    Args:
        brep: The extracted graph.
        path: Output file path.
    """
    loops = brep_loops(brep)
    triangles, face_ids = brep_triangles(brep)
    lines = ["# cadjoint b-rep export"]
    lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in brep.points)
    polygonal = set()
    for face in brep.faces:
        if face.analytic and face.kind == "plane" and len(loops[face.index]) == 1:
            polygonal.add(face.index)
            lines.append("f " + " ".join(str(index + 1) for index in loops[face.index][0]))
    for triangle, face_id in zip(triangles, face_ids):
        if int(face_id) in polygonal:
            continue
        lines.append(f"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def save_brep_stl(brep: BRep, path: str | Path, *, binary: bool = True) -> None:
    """Write the graph's tessellation as STL.

    Args:
        brep: The extracted graph.
        path: Output file path.
        binary: Write the binary format (default) or ASCII.
    """
    from cadjoint.meshing.dual_contouring import Mesh
    from cadjoint.meshing.export import save_stl

    triangles, _ = brep_triangles(brep)
    save_stl(
        Mesh(
            vertices=brep.points,
            faces=triangles,
            quads=np.asarray(brep.mesh.quads, dtype=np.int32),
            normals=brep.mesh.normals,
            cells=brep.mesh.cells,
        ),
        path,
        binary=binary,
    )


# ── STEP ─────────────────────────────────────────────────────────────────────


def _loop_circle(
    points: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fit a full circle to a closed loop, or ``None``.

    Both a cylinder's rim and the bore in the plate it passes through are
    the *same* circle, and STEP wants them to be the same ``EDGE_CURVE``:
    two faces sharing one edge is what makes a shell sew into a solid.  So
    circles are recognised from the loop alone — its own plane, its own
    centre — and cached by geometry in :class:`_StepWriter`, which is what
    lets the plate's hole and the bore wall meet on one entity.

    Returns:
        ``(centre, axis, radius)`` with ``axis`` following the loop's
            winding, or ``None`` when the loop is not a full circle.
    """
    if points.shape[0] < 5:
        return None
    area_vector = 0.5 * np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
    length = float(np.linalg.norm(area_vector))
    if length <= 0.0:
        return None
    axis = area_vector / length
    centre_guess = points.mean(axis=0)
    offsets = points - centre_guess
    if float(np.abs(offsets @ axis).max()) > tolerance:
        return None
    basis = np.stack([_orthogonal(axis), np.cross(axis, _orthogonal(axis))])
    planar = (points - centre_guess) @ basis.T
    design = np.concatenate([2.0 * planar, np.ones((planar.shape[0], 1))], axis=1)
    target = np.sum(planar * planar, axis=1)
    try:
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    centre2 = solution[:2]
    radius = float(np.sqrt(max(solution[2] + float(centre2 @ centre2), 0.0)))
    if radius <= 0.0:
        return None
    distances = np.linalg.norm(planar - centre2, axis=1)
    if float(np.abs(distances - radius).max()) > max(tolerance, 1e-9 * radius):
        return None
    # A full turn, not an arc: the swept angle about the axis must close.
    angles = np.arctan2(planar[:, 1] - centre2[1], planar[:, 0] - centre2[0])
    steps = np.diff(np.concatenate([angles, angles[:1]]))
    steps = (steps + np.pi) % (2.0 * np.pi) - np.pi
    if abs(abs(float(steps.sum())) - 2.0 * np.pi) > 1e-6:
        return None
    return centre_guess + basis.T @ centre2, axis, radius


def _canonical(axis: np.ndarray) -> tuple[np.ndarray, float]:
    """A direction and the sign that maps ``axis`` onto it, for cache keys."""
    dominant = int(np.argmax(np.abs(axis)))
    sign = 1.0 if axis[dominant] >= 0 else -1.0
    return axis * sign, sign


def _orthogonal(axis: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(axis, helper))


class _StepWriter:
    """Accumulates STEP entities with shared points, vertices and curves."""

    def __init__(
        self, points: np.ndarray, tolerance: float, protected: set[int] | None = None
    ) -> None:
        self.points = points
        self.tolerance = tolerance
        self.protected = protected or set()
        self.entities: list[str] = list(_STEP_BOILERPLATE)
        self._cartesian: dict[int, int] = {}
        self._vertex: dict[int, int] = {}
        self._line_edge: dict[tuple[int, int], int] = {}
        self._circle_edge: dict[tuple, int] = {}
        self.counts: dict[str, int] = {}

    def add(self, body: str) -> int:
        self.entities.append(body)
        name = body.split("(", 1)[0].strip("(")
        self.counts[name] = self.counts.get(name, 0) + 1
        return len(self.entities)

    def cartesian(self, index: int) -> int:
        if index not in self._cartesian:
            self._cartesian[index] = self.add(
                f"CARTESIAN_POINT('',{_step_point(self.points[index])})"
            )
        return self._cartesian[index]

    def free_point(self, position: np.ndarray) -> int:
        return self.add(f"CARTESIAN_POINT('',{_step_point(position)})")

    def vertex(self, index: int) -> int:
        if index not in self._vertex:
            self._vertex[index] = self.add(f"VERTEX_POINT('',#{self.cartesian(index)})")
        return self._vertex[index]

    def direction(self, vector: np.ndarray) -> int:
        return self.add(f"DIRECTION('',{_step_point(_unit(np.asarray(vector, dtype=np.float64)))})")

    def placement(self, origin_id: int, axis: np.ndarray, reference: np.ndarray) -> int:
        axis_id = self.direction(axis)
        reference_id = self.direction(reference)
        return self.add(f"AXIS2_PLACEMENT_3D('',#{origin_id},#{axis_id},#{reference_id})")

    def line_edge(self, a: int, b: int) -> tuple[int, bool]:
        """Shared ``EDGE_CURVE`` for the segment ``a-b``; also its direction."""
        key = (a, b) if a < b else (b, a)
        forward = key == (a, b)
        if key not in self._line_edge:
            low, high = key
            delta = self.points[high] - self.points[low]
            direction_id = self.direction(delta if np.any(delta) else np.array([1.0, 0.0, 0.0]))
            vector_id = self.add(
                f"VECTOR('',#{direction_id},{_step_real(float(np.linalg.norm(delta)))})"
            )
            line_id = self.add(f"LINE('',#{self.cartesian(low)},#{vector_id})")
            self._line_edge[key] = self.add(
                f"EDGE_CURVE('',#{self.vertex(low)},#{self.vertex(high)},#{line_id},.T.)"
            )
        return self._line_edge[key], forward

    def polyline_loop(self, loop: list[int]) -> int:
        oriented = []
        for a, b in zip(loop, loop[1:] + loop[:1]):
            edge_id, forward = self.line_edge(a, b)
            oriented.append(
                self.add(f"ORIENTED_EDGE('',*,*,#{edge_id},{'.T.' if forward else '.F.'})")
            )
        return self.add("EDGE_LOOP('',(" + ",".join(f"#{i}" for i in oriented) + "))")

    def circle_loop(self, centre: np.ndarray, axis: np.ndarray, radius: float) -> int:
        """``EDGE_LOOP`` of one closed ``CIRCLE`` edge, shared between faces.

        Two faces meeting on a circular edge must reference *one*
        ``EDGE_CURVE``, or OCCT sews nothing and the shell falls apart.  The
        cache key quantizes the circle's geometry at the file's own
        tolerance, and the loop's own winding decides only the orientation
        flag.
        """
        canonical_axis, _sign = _canonical(axis)
        quantum = max(self.tolerance, 1e-12)
        key = (
            tuple(np.round(centre / quantum).astype(np.int64)),
            tuple(np.round(canonical_axis / 1e-9).astype(np.int64)),
            int(round(radius / quantum)),
        )
        if key not in self._circle_edge:
            reference = _orthogonal(canonical_axis)
            centre_id = self.free_point(centre)
            placement_id = self.placement(centre_id, canonical_axis, reference)
            circle_id = self.add(f"CIRCLE('',#{placement_id},{_step_real(float(radius))})")
            seam_id = self.free_point(centre + radius * reference)
            vertex_id = self.add(f"VERTEX_POINT('',#{seam_id})")
            self._circle_edge[key] = self.add(
                f"EDGE_CURVE('',#{vertex_id},#{vertex_id},#{circle_id},.T.)"
            )
        forward = float(axis @ canonical_axis) >= 0
        oriented_id = self.add(
            f"ORIENTED_EDGE('',*,*,#{self._circle_edge[key]},{'.T.' if forward else '.F.'})"
        )
        return self.add(f"EDGE_LOOP('',(#{oriented_id}))")

    def bound(self, loop: list[int], outer: bool) -> int | None:
        """A face bound for one loop: a shared circle when it is one, else lines."""
        if len(loop) < 3:
            return None
        # A circle is only usable when the face on the other side of the
        # loop will draw the same circle.  A loop touching a faceted face
        # (every such vertex is ``protected``) must stay a polyline, or the
        # two faces claim different curves and the shell will not sew.
        circle = (
            None
            if self.protected.intersection(loop)
            else _loop_circle(self.points[loop], self.tolerance)
        )
        loop_id = self.circle_loop(*circle) if circle is not None else self.polyline_loop(loop)
        keyword = "FACE_OUTER_BOUND" if outer else "FACE_BOUND"
        return self.add(f"{keyword}('',#{loop_id},.T.)")


def _write_plane_face(
    writer: _StepWriter, face: BRepFace, face_loops: list[list[int]]
) -> int | None:
    """One ``ADVANCED_FACE`` on a ``PLANE``, outer loop first, holes after."""
    normal = np.asarray(face.surface.axis, dtype=np.float64)
    bounds = [
        bound
        for bound in (writer.bound(loop, position == 0) for position, loop in enumerate(face_loops))
        if bound is not None
    ]
    if not bounds:
        return None
    origin_id = writer.cartesian(face_loops[0][0])
    span = writer.points[face_loops[0][1]] - writer.points[face_loops[0][0]]
    reference = _unit(span - float(span @ normal) * normal)
    if not np.any(reference):
        reference = _orthogonal(normal)
    placement_id = writer.placement(origin_id, normal, reference)
    plane_id = writer.add(f"PLANE('',#{placement_id})")
    return writer.add(f"ADVANCED_FACE('',({','.join(f'#{i}' for i in bounds)}),#{plane_id},.T.)")


def _write_cylinder_face(
    writer: _StepWriter, face: BRepFace, face_loops: list[list[int]]
) -> int | None:
    """One ``ADVANCED_FACE`` on a ``CYLINDRICAL_SURFACE`` bounded by rim circles."""
    surface = face.surface
    axis = np.asarray(surface.axis, dtype=np.float64)
    circles = [_loop_circle(writer.points[loop], writer.tolerance) for loop in face_loops]
    if any(circle is None for circle in circles):
        return None
    bounds = [writer.bound(loop, position == 0) for position, loop in enumerate(face_loops)]
    if any(bound is None for bound in bounds):
        return None
    centre = circles[0][0]
    reference = _orthogonal(axis)
    origin_id = writer.free_point(centre)
    placement_id = writer.placement(origin_id, axis, reference)
    cylinder_id = writer.add(
        f"CYLINDRICAL_SURFACE('',#{placement_id},{_step_real(float(surface.radius))})"
    )
    # ``.T.`` regardless of :attr:`AnalyticSurface.sense`: the wire order
    # already tells the kernel which side the material is on, and OCCT
    # round-trips the plate's bore to its analytic volume this way.  ``sense``
    # is carried for callers who want to know, not to steer the flag.
    return writer.add(f"ADVANCED_FACE('',({','.join(f'#{i}' for i in bounds)}),#{cylinder_id},.T.)")


def _analytic_plan(
    brep: BRep,
    loops: list[list[list[int]]],
    tolerance: float,
    protected: set[int] | None = None,
) -> dict[int, str]:
    """Decide how each face is written: ``plane``, ``cylinder`` or ``facet``."""
    keep = protected or set()
    plan: dict[int, str] = {}
    for face in brep.faces:
        face_loops = loops[face.index]
        if not face.analytic or not face_loops:
            plan[face.index] = "facet"
        elif face.kind == "plane":
            plan[face.index] = "plane"
        elif (
            face.kind == "cylinder"
            and len(face_loops) == 2
            and not any(keep.intersection(loop) for loop in face_loops)
            and all(_loop_circle(brep.points[loop], tolerance) is not None for loop in face_loops)
        ):
            plan[face.index] = "cylinder"
        else:
            plan[face.index] = "facet"
    return plan


def save_brep_step(
    brep: BRep,
    path: str | Path,
    *,
    analytic: bool = True,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Write the graph as STEP AP214, analytic where the graph can certify it.

    Planar faces become a ``PLANE`` trimmed by their own boundary loops
    (holes included); a full cylindrical band bounded by two rim circles
    becomes a ``CYLINDRICAL_SURFACE`` with ``CIRCLE`` edges; everything else
    — blend faces above all — is faceted into planar triangles from the same
    tessellation :func:`brep_triangles` produces, so the shell always closes.

    Args:
        brep: The extracted graph.
        path: Output file path.
        analytic: Emit analytic surfaces; ``False`` facets everything, which
            is the comparison baseline against
            :func:`cadjoint.meshing.export.save_step`.
        tolerance: Loop-simplification and circle-detection tolerance;
            defaults to ``1e-6`` times the grid diagonal.

    Returns:
        A report: ``faces`` per strategy, ``entities`` written, and the
            per-keyword entity counts under ``keywords``.

    Raises:
        ValueError: If the graph has no writable face.
    """
    extent = np.asarray(brep.grid.spacing) * np.asarray(brep.grid.cells)
    if tolerance is None:
        tolerance = 1e-6 * float(np.linalg.norm(extent))

    loops = brep_loops(brep, tolerance=tolerance)
    plan = (
        _analytic_plan(brep, loops, tolerance)
        if analytic
        else {face.index: "facet" for face in brep.faces}
    )
    # Second pass: a vertex a faceted face still uses cannot be simplified
    # away on the analytic side, or the two faces disagree about the curve
    # between the same two points and the shell will not sew.
    quads = np.asarray(brep.mesh.quads, dtype=np.int64)
    faceted_quads = np.asarray(
        [plan[int(face_id)] == "facet" for face_id in brep.quad_face], dtype=bool
    )
    protected = set(np.unique(quads[faceted_quads]).tolist()) if faceted_quads.any() else set()
    if protected:
        loops = brep_loops(brep, tolerance=tolerance, protected=protected)
        plan = _analytic_plan(brep, loops, tolerance, protected) if analytic else plan

    triangles, triangle_faces = brep_triangles(brep)
    faceted = {face_id for face_id, strategy in plan.items() if strategy == "facet"}
    facet_loops = [
        [int(a), int(b), int(c)]
        for (a, b, c), face_id in zip(triangles, triangle_faces)
        if int(face_id) in faceted
    ]
    analytic_loops = {
        face_id: [list(loop) for loop in loops[face_id] if len(loop) >= 3]
        for face_id, strategy in plan.items()
        if strategy in ("plane", "cylinder")
    }

    # Weld across every loop at once so the polyline faces and the facets
    # agree on which vertices are the same vertex, keeping the loop order so
    # each welded loop stays matched to the face it came from.
    flat: list[list[int]] = list(facet_loops)
    keys: list[tuple[str, int]] = [("facet", -1) for _ in facet_loops]
    for face_id, face_loops in analytic_loops.items():
        for loop in face_loops:
            flat.append(list(loop))
            keys.append(("analytic", face_id))
    if not flat:
        raise ValueError("The B-rep has no writable face.")
    welded = _weld_aligned(brep.points, flat)
    rebuilt_facets: list[list[int]] = []
    rebuilt_analytic: dict[int, list[list[int]]] = {face_id: [] for face_id in analytic_loops}
    for (kind, face_id), loop in zip(keys, welded):
        if len(loop) < 3:
            continue
        if kind == "facet":
            rebuilt_facets.append(loop)
        else:
            rebuilt_analytic[face_id].append(loop)

    writer = _StepWriter(brep.points, tolerance, protected)
    face_ids: list[int] = []
    written = {"plane": 0, "cylinder": 0, "facet": 0, "dropped": 0}
    for face in brep.faces:
        strategy = plan[face.index]
        if strategy == "facet":
            continue
        face_loops = rebuilt_analytic.get(face.index) or []
        if not face_loops:
            written["dropped"] += 1
            continue
        if strategy == "plane":
            entity = _write_plane_face(writer, face, face_loops)
        else:
            entity = _write_cylinder_face(writer, face, face_loops)
        if entity is None:
            written["dropped"] += 1
            continue
        face_ids.append(entity)
        written[strategy] += 1
    for loop in rebuilt_facets:
        loop_id = writer.polyline_loop(loop)
        bound_id = writer.add(f"FACE_OUTER_BOUND('',#{loop_id},.T.)")
        normal = _unit(_polygon_normals(brep.points, np.asarray([loop], dtype=np.int64))[0])
        first_edge = _unit(brep.points[loop[1]] - brep.points[loop[0]])
        reference = _unit(first_edge - float(first_edge @ normal) * normal)
        if not np.any(reference):
            reference = _orthogonal(normal)
        placement_id = writer.placement(writer.cartesian(loop[0]), normal, reference)
        plane_id = writer.add(f"PLANE('',#{placement_id})")
        face_ids.append(writer.add(f"ADVANCED_FACE('',(#{bound_id}),#{plane_id},.T.)"))
        written["facet"] += 1

    if not face_ids:
        raise ValueError("The B-rep produced no STEP face.")
    shell_id = writer.add("CLOSED_SHELL('',(" + ",".join(f"#{i}" for i in face_ids) + "))")
    brep_id = writer.add(f"MANIFOLD_SOLID_BREP('cadjoint_brep',#{shell_id})")
    representation_id = writer.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{brep_id}),#13)")
    writer.add(f"SHAPE_DEFINITION_REPRESENTATION(#8,#{representation_id})")

    data = "\n".join(f"#{i + 1} = {body};" for i, body in enumerate(writer.entities))
    Path(path).write_text(_STEP_HEADER.format(data=data), encoding="ascii")
    return {
        "faces": written,
        "step_faces": len(face_ids),
        "entities": len(writer.entities),
        "keywords": dict(sorted(writer.counts.items())),
        "volume": brep_volume(brep),
    }
