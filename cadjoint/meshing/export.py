"""Mesh export: planar-face merging and OBJ / STL / STEP writers.

Dual-contour meshes tessellate flat regions with many coplanar quads.  For
export this module first rebuilds a *sparse polygon mesh*: quads are merged
into large planar n-gons wherever the surface is genuinely flat (a box
collapses to its 6 faces) while curved regions keep their triangles.  The
writers then serialize that mixed polygon/triangle representation:

- :func:`save_obj` — Wavefront OBJ with native n-gon faces.
- :func:`save_stl` — binary (default) or ASCII STL triangles.
- :func:`save_step` — minimal STEP AP214 boundary representation.

Everything here is concrete host-side numpy + stdlib; export happens after
extraction, outside any differentiation path.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cadjoint.meshing.dual_contouring import Mesh

# A quad (a, b, c, d) contributes directed boundary-candidate edges in
# winding order; consecutive pairs below index into the quad row.
_QUAD_EDGE_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))


def _polygon_normals(vertices: np.ndarray, loops: np.ndarray) -> np.ndarray:
    """Area-weighted (Newell) normals for index loops, shaped like ``loops[:, 0]``.

    Args:
        vertices: Vertex positions ``(n, 3)``.
        loops: Integer index loops ``(k, m)``.

    Returns:
        Unnormalized area vectors ``(k, 3)``; direction follows the winding.
    """
    points = vertices[loops]
    rolled = np.roll(points, -1, axis=1)
    return 0.5 * np.sum(np.cross(points, rolled), axis=1)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros_like(vector)


def _quad_triangle_rows(mesh: Mesh, quad_ids: list[int]) -> np.ndarray:
    """Triangles covering the given quads, taken from the mesh triangulation.

    ``dual_faces`` emits triangles ``2i`` and ``2i + 1`` for quad ``i``; when
    that pairing does not hold (foreign meshes) the quads are split along
    their first diagonal instead.
    """
    faces = np.asarray(mesh.faces, dtype=np.int32)
    quads = np.asarray(mesh.quads, dtype=np.int32)
    if faces.shape[0] == 2 * quads.shape[0]:
        rows = np.asarray(
            [row for quad_id in quad_ids for row in (2 * quad_id, 2 * quad_id + 1)],
            dtype=np.int64,
        )
        return faces[rows].reshape((-1, 3)).copy()
    split = []
    for quad_id in quad_ids:
        a, b, c, d = (int(v) for v in quads[quad_id])
        split.extend(((a, b, c), (a, c, d)))
    return np.asarray(split, dtype=np.int32).reshape((-1, 3))


def _boundary_loops(quads: np.ndarray, region: list[int]) -> list[list[int]] | None:
    """Ordered boundary loops of a quad region, or ``None`` if non-simple.

    Boundary edges are the undirected edges used by exactly one quad of the
    region; their directions (inherited from the quads' consistent winding)
    are chained into closed loops.  Returns ``None`` when the boundary is
    not a disjoint union of simple loops.
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
    boundary_count = 0
    for key, count in usage.items():
        if count != 1:
            continue
        a, b = directed[key]
        if a in successor:
            return None  # A vertex with two outgoing boundary edges: non-simple.
        successor[a] = b
        boundary_count += 1
    if boundary_count < 3:
        return None
    loops: list[list[int]] = []
    remaining = set(successor)
    while remaining:
        start = min(remaining)
        loop = [start]
        remaining.discard(start)
        current = successor[start]
        while current != start:
            if current not in remaining:
                return None  # Open chain: the boundary never closes.
            loop.append(current)
            remaining.discard(current)
            current = successor[current]
        if len(loop) < 3:
            return None
        loops.append(loop)
    return loops


def merge_planar_faces(
    mesh: Mesh,
    *,
    angle_degrees: float = 1.0,
    plane_tolerance: float | None = None,
) -> tuple[list[list[int]], np.ndarray]:
    """Merge coplanar mesh quads into large polygon faces.

    Regions are grown across shared quad edges while each candidate quad's
    normal stays within ``angle_degrees`` of the region seed's normal, then
    accepted only if every region vertex lies within a small tolerance of the
    region's mean plane.  An accepted region with exactly one boundary loop
    becomes a single n-gon wound counterclockwise around its outward normal;
    regions with holes (multiple boundary loops), non-simple boundaries,
    failed planarity, or a single quad (merging gains nothing, which keeps
    curved meshes fully triangulated) fall back to their triangles.

    Args:
        mesh: A dual-contour :class:`~cadjoint.meshing.dual_contouring.Mesh`.
        angle_degrees: Maximum angle between a quad normal and its region
            seed's normal during region growth.
        plane_tolerance: Maximum vertex distance from a region's mean plane;
            defaults to ``1e-5`` times the mesh bounding-box diagonal.

    Returns:
        ``(polygons, triangles)``: ordered vertex-index loops (one simple
        polygon each, indices into ``mesh.vertices``) and the leftover
        triangles ``(t, 3)`` covering every quad not absorbed by a polygon.
    """
    quads = np.asarray(mesh.quads, dtype=np.int32)
    quad_count = quads.shape[0]
    if quad_count == 0:
        return [], np.asarray(mesh.faces, dtype=np.int32).reshape((-1, 3)).copy()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if plane_tolerance is None:
        spans = vertices.max(axis=0) - vertices.min(axis=0)
        plane_tolerance = max(1e-5 * float(np.linalg.norm(spans)), 1e-9)

    normals = _polygon_normals(vertices, quads)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    unit_normals = np.where(valid[:, None], normals / np.where(valid, lengths, 1.0)[:, None], 0.0)

    edge_users: dict[tuple[int, int], list[int]] = {}
    for quad_id in range(quad_count):
        row = quads[quad_id]
        for i, j in _QUAD_EDGE_PAIRS:
            a, b = int(row[i]), int(row[j])
            key = (a, b) if a < b else (b, a)
            edge_users.setdefault(key, []).append(quad_id)
    neighbors: list[set[int]] = [set() for _ in range(quad_count)]
    for users in edge_users.values():
        for a in users:
            for b in users:
                if a != b:
                    neighbors[a].add(b)

    cos_threshold = np.cos(np.radians(angle_degrees))
    visited = np.zeros(quad_count, dtype=bool)
    polygons: list[list[int]] = []
    leftover_ids: list[int] = []
    for seed in range(quad_count):
        if visited[seed]:
            continue
        visited[seed] = True
        if not valid[seed]:
            leftover_ids.append(seed)
            continue
        seed_normal = unit_normals[seed]
        region = [seed]
        queue = deque([seed])
        while queue:
            quad_id = queue.popleft()
            for neighbor in sorted(neighbors[quad_id]):
                if visited[neighbor] or not valid[neighbor]:
                    continue
                if float(unit_normals[neighbor] @ seed_normal) < cos_threshold:
                    continue
                visited[neighbor] = True
                region.append(neighbor)
                queue.append(neighbor)
        if len(region) < 2:
            leftover_ids.extend(region)
            continue
        region_normal = _unit(np.sum(normals[region], axis=0))
        region_vertices = np.unique(quads[region])
        offsets = (vertices[region_vertices] - vertices[region_vertices].mean(axis=0)) @ (
            region_normal
        )
        if float(np.abs(offsets).max()) > plane_tolerance:
            leftover_ids.extend(region)
            continue
        loops = _boundary_loops(quads, region)
        if loops is None or len(loops) != 1:
            leftover_ids.extend(region)  # Holes or a non-simple boundary.
            continue
        loop = loops[0]
        loop_normal = _polygon_normals(vertices, np.asarray([loop], dtype=np.int64))[0]
        if float(loop_normal @ region_normal) < 0:
            loop.reverse()
        polygons.append([int(v) for v in loop])
    return polygons, _quad_triangle_rows(mesh, sorted(leftover_ids))


def save_obj(mesh: Mesh, path: str | Path, *, merge_planar: bool = True) -> None:
    """Write a Wavefront OBJ file with merged planar n-gon faces.

    Args:
        mesh: A dual-contour :class:`~cadjoint.meshing.dual_contouring.Mesh`.
        path: Output file path.
        merge_planar: Merge coplanar quads into n-gons via
            :func:`merge_planar_faces`; when ``False`` every triangle of
            ``mesh.faces`` is written unmerged.

    All mesh vertices are written (1-based OBJ indexing keeps the original
    row order), so faces of either kind index the same vertex list.  The
    output is deterministic for a given mesh.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if merge_planar:
        polygons, triangles = merge_planar_faces(mesh)
    else:
        polygons = []
        triangles = np.asarray(mesh.faces, dtype=np.int32).reshape((-1, 3))
    lines = ["# cadjoint mesh export"]
    lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices)
    for loop in polygons:
        lines.append("f " + " ".join(str(index + 1) for index in loop))
    for a, b, c in triangles:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _triangle_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    edges_ab = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edges_ac = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    normals = np.cross(edges_ab, edges_ac)
    lengths = np.linalg.norm(normals, axis=1)
    safe = np.where(lengths > 0, lengths, 1.0)
    return np.where(lengths[:, None] > 0, normals / safe[:, None], 0.0)


def save_stl(mesh: Mesh, path: str | Path, *, binary: bool = True) -> None:
    """Write the mesh triangles as an STL file.

    Args:
        mesh: A dual-contour :class:`~cadjoint.meshing.dual_contouring.Mesh`;
            every triangle of ``mesh.faces`` is written.
        path: Output file path.
        binary: Write the 80-byte-header binary format (default); ``False``
            writes the ASCII ``solid`` format instead.

    Triangle normals are computed from the counterclockwise winding, so they
    point outward for a well-oriented mesh.  The output is deterministic.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 3))
    normals = _triangle_normals(vertices, triangles)
    corners = vertices[triangles]  # (t, 3, 3)
    if binary:
        records = np.zeros(
            triangles.shape[0],
            dtype=[("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attribute", "<u2")],
        )
        records["normal"] = normals.astype(np.float32)
        records["corners"] = corners.astype(np.float32)
        header = b"cadjoint binary STL export".ljust(80, b"\0")
        with open(path, "wb") as stream:
            stream.write(header)
            stream.write(np.uint32(triangles.shape[0]).tobytes())
            stream.write(records.tobytes())
        return
    lines = ["solid cadjoint"]
    for normal, triangle in zip(normals, corners):
        lines.append(f"  facet normal {normal[0]:.8e} {normal[1]:.8e} {normal[2]:.8e}")
        lines.append("    outer loop")
        lines.extend(f"      vertex {x:.8e} {y:.8e} {z:.8e}" for x, y, z in triangle)
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid cadjoint")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _step_real(value: float) -> str:
    """Format a float as a STEP real (always contains a decimal point)."""
    text = f"{value:.10G}"
    if "E" in text:
        mantissa, exponent = text.split("E")
        if "." not in mantissa:
            mantissa += "."
        return f"{mantissa}E{exponent}"
    if "." not in text:
        text += "."
    return text


def _step_point(point: np.ndarray) -> str:
    return "(" + ",".join(_step_real(float(value)) for value in point) + ")"


# Distance below which two points are the same point per the exported file's
# own UNCERTAINTY_MEASURE_WITH_UNIT declaration (entity #12 below).  Edges
# shorter than this are degenerate: OCCT cannot build a LINE from a
# zero-magnitude VECTOR and drops the whole containing wire, splitting the
# shell.  ``save_step`` welds such edge endpoints together instead.
_STEP_DISTANCE_ACCURACY = 1e-7

_STEP_HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('cadjoint faceted mesh export'),'2;1');
FILE_NAME('mesh','',('cadjoint'),(''),'cadjoint','cadjoint','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 1 1 1 1 }}'));
ENDSEC;
DATA;
{data}
ENDSEC;
END-ISO-10303-21;
"""

_STEP_BOILERPLATE = (
    "APPLICATION_CONTEXT('automotive design')",
    "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2010,#1)",
    "PRODUCT_CONTEXT('',#1,'mechanical')",
    "PRODUCT('cadjoint_mesh','cadjoint_mesh','',(#3))",
    "PRODUCT_DEFINITION_FORMATION('','',#4)",
    "PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design')",
    "PRODUCT_DEFINITION('design','',#5,#6)",
    "PRODUCT_DEFINITION_SHAPE('','',#7)",
    "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))",
    "(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))",
    "(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())",
    "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0E-7),#9,'distance_accuracy_value','')",
    "(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#12))"
    "GLOBAL_UNIT_ASSIGNED_CONTEXT((#9,#10,#11))REPRESENTATION_CONTEXT('',''))",
)


def _weld_degenerate_edges(
    vertices: np.ndarray,
    loops: list[list[int]],
    tolerance: float = _STEP_DISTANCE_ACCURACY,
) -> list[list[int]]:
    """Collapse loop edges shorter than ``tolerance`` onto one endpoint.

    Sharp dual-contour placement can land two adjacent cells' vertices on the
    same feature-curve point, giving distinct indices with coincident
    coordinates.  The endpoints of every such degenerate edge are merged
    (union-find, lowest index wins), consecutive duplicates are removed from
    each loop, and loops left with fewer than three vertices are dropped.

    Args:
        vertices: Vertex positions ``(n, 3)``.
        loops: Ordered vertex-index loops (polygons and triangles alike).
        tolerance: Maximum length of an edge to weld away.

    Returns:
        Fresh loop lists with welded indices; the input loops are untouched.
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
            if a != b and float(np.linalg.norm(vertices[b] - vertices[a])) <= tolerance:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    if not parent:
        return [list(loop) for loop in loops]

    welded: list[list[int]] = []
    for loop in loops:
        mapped = [find(index) for index in loop]
        collapsed = [v for i, v in enumerate(mapped) if v != mapped[i - 1]]
        if len(collapsed) >= 3:
            welded.append(collapsed)
    return welded


def save_step(mesh: Mesh, path: str | Path) -> None:
    """Write a minimal faceted STEP AP214 boundary representation.

    Builds one planar ``ADVANCED_FACE`` per merged polygon (via
    :func:`merge_planar_faces`) and per leftover triangle, over shared
    ``VERTEX_POINT`` and ``EDGE_CURVE``/``LINE`` topology, gathered into a
    ``CLOSED_SHELL`` inside a ``MANIFOLD_SOLID_BREP`` with the minimal AP214
    product and unit boilerplate.  Every referenced entity id is defined and
    the output is deterministic for a given mesh.

    Degenerate loop edges (endpoints closer than the file's declared
    distance accuracy, which sharp feature placement can produce) are welded
    away first — OCCT rejects zero-length ``LINE`` geometry and would split
    the shell.  Output is validated against the OCCT kernel in
    ``tests/meshing/test_step_kernel.py``: files read back as a single valid
    closed ``MANIFOLD_SOLID_BREP`` with matching face counts and volume.
    Units are metres (one mesh unit = 1 m); readers defaulting to
    millimetres scale coordinates by 1000 on import.

    Args:
        mesh: A dual-contour :class:`~cadjoint.meshing.dual_contouring.Mesh`;
            must contain at least one face.
        path: Output file path.

    Raises:
        ValueError: If the mesh has no faces.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    polygons, triangles = merge_planar_faces(mesh)
    loops = [list(polygon) for polygon in polygons]
    loops.extend([int(a), int(b), int(c)] for a, b, c in triangles)
    if not loops:
        raise ValueError("Cannot export an empty mesh to STEP.")
    loops = _weld_degenerate_edges(vertices, loops)
    if not loops:
        raise ValueError("Every face of the mesh is degenerate at STEP accuracy.")

    entities: list[str] = list(_STEP_BOILERPLATE)

    def add(body: str) -> int:
        entities.append(body)
        return len(entities)

    used = sorted({index for loop in loops for index in loop})
    point_ids: dict[int, int] = {}
    vertex_ids: dict[int, int] = {}
    for index in used:
        point_ids[index] = add(f"CARTESIAN_POINT('',{_step_point(vertices[index])})")
        vertex_ids[index] = add(f"VERTEX_POINT('',#{point_ids[index]})")

    edge_keys = sorted(
        {(a, b) if a < b else (b, a) for loop in loops for a, b in zip(loop, loop[1:] + loop[:1])}
    )
    edge_curve_ids: dict[tuple[int, int], int] = {}
    for a, b in edge_keys:
        direction = _unit(vertices[b] - vertices[a])
        if not np.any(direction):
            direction = np.array([1.0, 0.0, 0.0])
        direction_id = add(f"DIRECTION('',{_step_point(direction)})")
        length = _step_real(float(np.linalg.norm(vertices[b] - vertices[a])))
        vector_id = add(f"VECTOR('',#{direction_id},{length})")
        line_id = add(f"LINE('',#{point_ids[a]},#{vector_id})")
        edge_curve_ids[(a, b)] = add(
            f"EDGE_CURVE('',#{vertex_ids[a]},#{vertex_ids[b]},#{line_id},.T.)"
        )

    face_ids: list[int] = []
    for loop in loops:
        oriented_ids = []
        for a, b in zip(loop, loop[1:] + loop[:1]):
            key = (a, b) if a < b else (b, a)
            flag = ".T." if (a, b) == key else ".F."
            oriented_ids.append(add(f"ORIENTED_EDGE('',*,*,#{edge_curve_ids[key]},{flag})"))
        loop_id = add("EDGE_LOOP('',(" + ",".join(f"#{i}" for i in oriented_ids) + "))")
        bound_id = add(f"FACE_OUTER_BOUND('',#{loop_id},.T.)")
        normal = _unit(_polygon_normals(vertices, np.asarray([loop], dtype=np.int64))[0])
        first_edge = _unit(vertices[loop[1]] - vertices[loop[0]])
        reference = _unit(first_edge - float(first_edge @ normal) * normal)
        if not np.any(reference):
            reference = np.array([1.0, 0.0, 0.0])
        normal_id = add(f"DIRECTION('',{_step_point(normal)})")
        reference_id = add(f"DIRECTION('',{_step_point(reference)})")
        placement_id = add(
            f"AXIS2_PLACEMENT_3D('',#{point_ids[loop[0]]},#{normal_id},#{reference_id})"
        )
        plane_id = add(f"PLANE('',#{placement_id})")
        face_ids.append(add(f"ADVANCED_FACE('',(#{bound_id}),#{plane_id},.T.)"))

    shell_id = add("CLOSED_SHELL('',(" + ",".join(f"#{i}" for i in face_ids) + "))")
    brep_id = add(f"MANIFOLD_SOLID_BREP('cadjoint_mesh',#{shell_id})")
    representation_id = add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{brep_id}),#13)")
    add(f"SHAPE_DEFINITION_REPRESENTATION(#8,#{representation_id})")

    data = "\n".join(f"#{i + 1} = {body};" for i, body in enumerate(entities))
    Path(path).write_text(_STEP_HEADER.format(data=data), encoding="ascii")
