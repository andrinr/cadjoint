"""The axiomatic battery: simple solids with known B-reps, measured, not assumed.

One list of named cases, each an SDF built from cadjoint's own primitives
and booleans together with the answer a textbook B-rep would give — face,
edge and vertex counts, the Euler characteristic, and wherever it has a
closed form, every edge curve and every corner.  ``tests/brep/test_axioms.py``
pins each case; ``python -m tests.brep.axioms`` renders them into
``research/brep-axioms/``.  Both go through :func:`measure`, so the number a
test asserts is the number the gallery draws.

Nothing here edits ``cadjoint``: the battery observes ``extract_brep`` as it
is.  Where a helper would belong in the library (the analytic-curve distance
and coverage, the Euler count over a graph with holes and closed edges, the
normal-crossing test along an edge) it lives here and says so.

Euler characteristic.  ``V - E + F = 2`` only when every face is a disk and
every edge an open arc.  The graph reports closed edges (a rim circle with
no vertex on it) and faces with holes (a cap around a bore), so the count is
taken over open cells with compactly supported Euler characteristics:
``chi = sum_faces (2 - loops) - open_edges + vertices``.  A disk contributes
1, an annulus 0, a whole sphere 2, a closed edge 0; a plate with a bore has
genus 1 and chi 0.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.meshing.edge_detection import GridSpec

__all__ = [
    "CASES",
    "Case",
    "Curve",
    "Measurement",
    "case",
    "grid_for",
    "measure",
    "segment",
    "circle",
]

#: Lattice offsets, in cells, the "hard" cases are re-extracted at.
OFFSETS = (0.0, 0.37, 0.71)
#: Cells per axis the tests run at; the gallery adds 64.
TEST_RESOLUTION = 32
#: Margin around the shape's bounding box, in the shape's own units.
MARGIN = 0.33
#: The viewer overlay's residual gate: an edge whose two-field residual is
#: above this many cells is refused (``_edge_overlay``).
REFUSE_CELLS = 0.1
#: Coverage reach: a true-curve sample is covered when an extracted edge
#: passes within this many cells of it.
COVER_CELLS = 0.5


# ── analytic curves ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Curve:
    """An expected edge curve, sampled densely for distance and coverage.

    Attributes:
        name: Label for reports and legends.
        kind: ``segment``, ``circle``, ``ellipse`` or ``param``.
        closed: Whether the curve closes on itself.
        sampler: ``n -> (n, 3)`` points along the curve.
        tag: ``analytic`` for an edge the graph should find exactly,
            ``crease`` for a hard crease a smooth union replaces by a blend
            (expected only when the classifier rounds it back to an edge),
            ``tangent`` for a seam where the two surfaces share a normal.
    """

    name: str
    kind: str
    closed: bool
    sampler: Callable[[int], np.ndarray]
    tag: str = "analytic"

    def samples(self, count: int = 512) -> np.ndarray:
        return np.asarray(self.sampler(count), dtype=np.float64)

    def length(self) -> float:
        pts = self.samples(2048)
        if self.closed:
            pts = np.concatenate([pts, pts[:1]])
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def segment(a, b, name: str = "segment", tag: str = "analytic") -> Curve:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return Curve(
        name, "segment", False, lambda n: a[None] + np.linspace(0, 1, n)[:, None] * (b - a), tag
    )


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def circle(center, axis, radius: float, name: str = "circle", tag: str = "analytic") -> Curve:
    c = np.asarray(center, dtype=np.float64)
    u, v = _basis(np.asarray(axis, dtype=np.float64))

    def sample(n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)[:, None]
        return c[None] + radius * (np.cos(t) * u[None] + np.sin(t) * v[None])

    return Curve(name, "circle", True, sample, tag)


def ellipse(center, u, v, name: str = "ellipse", tag: str = "analytic") -> Curve:
    """Closed curve ``c + u cos t + v sin t`` for semi-axis vectors ``u, v``."""
    c = np.asarray(center, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    def sample(n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)[:, None]
        return c[None] + np.cos(t) * u[None] + np.sin(t) * v[None]

    return Curve(name, "ellipse", True, sample, tag)


def param(fn, closed: bool, name: str = "curve", tag: str = "analytic") -> Curve:
    """Curve from ``fn(t)`` with ``t`` in ``[0, 2pi)`` (closed) or ``[0, 1]``."""

    def sample(n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False) if closed else np.linspace(0, 1, n)
        return np.stack([np.asarray(fn(float(s)), dtype=np.float64) for s in t])

    return Curve(name, "param", closed, sample, tag)


# ── axis-aligned box helpers ─────────────────────────────────────────────────


def _box_bounds(half, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(half, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return center - half, center + half


def _box_edges(lo, hi) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """The twelve edges as ``(axis, start, end)``."""
    edges = []
    for axis in range(3):
        i, j = [k for k in range(3) if k != axis]
        for si in (lo[i], hi[i]):
            for sj in (lo[j], hi[j]):
                start = np.zeros(3)
                end = np.zeros(3)
                start[i] = end[i] = si
                start[j] = end[j] = sj
                start[axis], end[axis] = lo[axis], hi[axis]
                edges.append((axis, start, end))
    return edges


def _clip_outside(axis, start, end, lo, hi) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parts of an axis-aligned segment that lie outside an open box."""
    fixed = [k for k in range(3) if k != axis]
    if not all(lo[k] < start[k] < hi[k] for k in fixed):
        return [(start, end)]
    a, b = start[axis], end[axis]
    pieces = []
    if lo[axis] > a:
        piece = end.copy()
        piece[axis] = min(lo[axis], b)
        pieces.append((start, piece))
    if hi[axis] < b:
        piece = start.copy()
        piece[axis] = max(hi[axis], a)
        pieces.append((piece, end))
    return [(p, q) for p, q in pieces if np.linalg.norm(q - p) > 1e-9]


def _aabb_union_curves(a, b, prefix: str = "") -> list[Curve]:
    """All edges of the union of two axis-aligned boxes in general position."""
    curves: list[Curve] = []
    for label, (lo, hi), (olo, ohi) in (("A", a, b), ("B", b, a)):
        for k, (axis, start, end) in enumerate(_box_edges(lo, hi)):
            for m, (p, q) in enumerate(_clip_outside(axis, start, end, olo, ohi)):
                curves.append(segment(p, q, f"{prefix}{label}{k}{'abc'[m]}"))
    (alo, ahi), (blo, bhi) = a, b
    for i in range(3):
        for av in (alo[i], ahi[i]):
            for j in range(3):
                if j == i:
                    continue
                for bv in (blo[j], bhi[j]):
                    if not (blo[i] < av < bhi[i] and alo[j] < bv < ahi[j]):
                        continue
                    k = 3 - i - j
                    lo_k, hi_k = max(alo[k], blo[k]), min(ahi[k], bhi[k])
                    if hi_k - lo_k <= 1e-9:
                        continue
                    p = np.zeros(3)
                    p[i], p[j], p[k] = av, bv, lo_k
                    q = p.copy()
                    q[k] = hi_k
                    sides = "-+"[int(av == ahi[i])] + "-+"[int(bv == bhi[j])]
                    curves.append(segment(p, q, f"{prefix}AB{i}{j}{sides}"))
    return curves


def _endpoints(curves: list[Curve]) -> np.ndarray:
    points = []
    for curve in curves:
        if curve.closed:
            continue
        pts = curve.samples(2)
        points.extend([pts[0], pts[-1]])
    if not points:
        return np.zeros((0, 3))
    unique = np.unique(np.round(np.asarray(points), 9), axis=0)
    return unique


# ── the cases ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    """One axiom: a scene and the B-rep it should derive.

    Attributes:
        name: File-safe identifier.
        build: Constructs a fresh scene (parameters are mutable state).
        lo, hi: The shape's bounding box; the grid adds :data:`MARGIN`.
        faces: Expected face count.
        face_kinds: Expected count per :attr:`BRepFace.kind`.
        edges: Expected edge count (open plus closed).
        closed_edges: How many of those are closed chains.
        vertices: Expected vertex count.
        euler: Expected Euler characteristic (2 for genus 0, 0 for genus 1).
        curves: Expected edge curves.
        corners: Expected vertex positions, shaped ``(v, 3)``.
        hard: Re-extract at every lattice offset and at 64 cells.
        note: What the right answer is and why, for degenerate cases.
        smoothness: Blend radius the scene was built with (0 for hard CSG).
        blend_tolerance_cells: Override of ``extract_brep``'s blend
            tolerance, in cells; ``None`` keeps the export default.
        sharp: For a blended case, the expected topology when the
            classifier rounds the fillet back to its edge (the overlay's
            one-cell rule), as ``(faces, face_kinds, edges, vertices)``.
    """

    name: str
    build: Callable[[], Any]
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    faces: int
    face_kinds: dict[str, int]
    edges: int
    closed_edges: int
    vertices: int
    euler: int
    curves: list[Curve] = field(default_factory=list)
    corners: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    hard: bool = False
    note: str = ""
    smoothness: float = 0.0
    blend_tolerance_cells: float | None = None
    sharp: tuple[int, dict[str, int], int, int] | None = None
    tags: tuple[str, ...] = ()


CASES: list[Case] = []


def case(**kwargs) -> Case:
    item = Case(**kwargs)
    CASES.append(item)
    return item


def _box(half, center=None):
    from cadjoint.sdf.primitives import Box
    from cadjoint.sdf.transforms import Translate

    box = Box(size=Vector(list(map(float, half))))
    if center is None:
        return box
    return Translate(box, Vector(list(map(float, center))))


def _cylinder(radius, half_height, center=None, axis=None):
    from cadjoint.sdf.primitives import Cylinder
    from cadjoint.sdf.transforms import Rotate, Translate

    node = Cylinder(radius=Scalar(float(radius)), height=Scalar(float(half_height)))
    if axis == "x":
        node = Rotate(node, "y", float(np.pi / 2))
    elif axis == "y":
        node = Rotate(node, "x", float(-np.pi / 2))
    if center is not None:
        node = Translate(node, Vector(list(map(float, center))))
    return node


def _union(*parts, smoothness=0.0):
    from cadjoint.sdf.boolean import Union

    return Union(tuple(parts), smoothness=float(smoothness))


def _difference(*parts, smoothness=0.0):
    from cadjoint.sdf.boolean import Difference

    return Difference(tuple(parts), smoothness=float(smoothness))


# 1. One box.
_A = _box_bounds((0.5, 0.4, 0.3))
_box_curves = [segment(s, e, f"e{k}") for k, (_, s, e) in enumerate(_box_edges(*_A))]
case(
    name="box",
    build=lambda: _box((0.5, 0.4, 0.3)),
    lo=(-0.5, -0.4, -0.3),
    hi=(0.5, 0.4, 0.3),
    faces=6,
    face_kinds={"plane": 6},
    edges=12,
    closed_edges=0,
    vertices=8,
    euler=2,
    curves=_box_curves,
    corners=_endpoints(_box_curves),
    note="The base case: six planes, twelve segments, eight corners.",
)

# 2. Two boxes overlapping in general position.
_B = _box_bounds((0.3, 0.35, 0.25), (0.35, 0.2, 0.15))
_overlap_curves = _aabb_union_curves(_A, _B)
case(
    name="boxes_overlap",
    build=lambda: _union(_box((0.5, 0.4, 0.3)), _box((0.3, 0.35, 0.25), (0.35, 0.2, 0.15))),
    lo=(-0.5, -0.4, -0.3),
    hi=(0.65, 0.55, 0.4),
    faces=12,
    face_kinds={"plane": 12},
    edges=30,
    closed_edges=0,
    vertices=20,
    euler=2,
    curves=_overlap_curves,
    corners=_endpoints(_overlap_curves),
    note=(
        "B pokes out of A's +x, +y and +z faces.  Three A faces are notched, "
        "three B faces are L-shaped, six new edges where A faces cross B faces."
    ),
)

# 3. Two boxes sharing a face plane (coplanar top and bottom).
_coplanar_top = [
    (-0.5, -0.4),
    (0.5, -0.4),
    (0.5, -0.3),
    (0.9, -0.3),
    (0.9, 0.3),
    (0.5, 0.3),
    (0.5, 0.4),
    (-0.5, 0.4),
]
_coplanar_curves: list[Curve] = []
for _z, _label in ((0.3, "top"), (-0.3, "bot")):
    for _k in range(8):
        _p = (*_coplanar_top[_k], _z)
        _q = (*_coplanar_top[(_k + 1) % 8], _z)
        _coplanar_curves.append(segment(_p, _q, f"{_label}{_k}"))
for _k, (_x, _y) in enumerate(_coplanar_top):
    _coplanar_curves.append(segment((_x, _y, -0.3), (_x, _y, 0.3), f"vert{_k}"))
case(
    name="boxes_coplanar",
    build=lambda: _union(_box((0.5, 0.4, 0.3)), _box((0.3, 0.3, 0.3), (0.6, 0.0, 0.0))),
    lo=(-0.5, -0.4, -0.3),
    hi=(0.9, 0.4, 0.3),
    faces=10,
    face_kinds={"plane": 10},
    edges=24,
    closed_edges=0,
    vertices=16,
    euler=2,
    curves=_coplanar_curves,
    corners=_endpoints(_coplanar_curves),
    hard=True,
    tags=("coplanar",),
    note=(
        "B's top and bottom lie in A's top and bottom planes.  The right answer "
        "merges each coplanar pair into ONE planar face (an 8-gon), with no edge "
        "across the join: 10 faces, 24 edges, 16 vertices.  Two patches share a "
        "zero set here, so ownership is a tie and any seam between them has a "
        "rank-deficient Gram."
    ),
)

# 4. Box with a cylinder standing on it.
_slab = _box_bounds((0.5, 0.5, 0.2))
_slab_curves = [segment(s, e, f"e{k}") for k, (_, s, e) in enumerate(_box_edges(*_slab))]
case(
    name="box_cyl_standing",
    build=lambda: _union(_box((0.5, 0.5, 0.2)), _cylinder(0.25, 0.3, (0.0, 0.0, 0.4))),
    lo=(-0.5, -0.5, -0.2),
    hi=(0.5, 0.5, 0.7),
    faces=8,
    face_kinds={"plane": 7, "cylinder": 1},
    edges=14,
    closed_edges=2,
    vertices=8,
    euler=2,
    curves=_slab_curves
    + [
        circle((0, 0, 0.2), (0, 0, 1), 0.25, "rim"),
        circle((0, 0, 0.7), (0, 0, 1), 0.25, "cap"),
    ],
    corners=_endpoints(_slab_curves),
    note="The slab top keeps its hole as a second loop; two full rim circles.",
)

# 5. Box with a cylinder through it (along y).
case(
    name="box_cyl_through",
    build=lambda: _union(_box((0.5, 0.4, 0.3)), _cylinder(0.25, 0.8, axis="y")),
    lo=(-0.5, -0.8, -0.3),
    hi=(0.5, 0.8, 0.3),
    faces=10,
    face_kinds={"plane": 8, "cylinder": 2},
    edges=16,
    closed_edges=4,
    vertices=8,
    euler=2,
    curves=_box_curves
    + [
        circle((0, 0.4, 0), (0, 1, 0), 0.25, "rim+y"),
        circle((0, -0.4, 0), (0, 1, 0), 0.25, "rim-y"),
        circle((0, 0.8, 0), (0, 1, 0), 0.25, "cap+y"),
        circle((0, -0.8, 0), (0, 1, 0), 0.25, "cap-y"),
    ],
    corners=_endpoints(_box_curves),
    note="Two stubs, two rims on the box, two cap rims.",
)

# 6. Plate with a bore (genus 1).
_plate = _box_bounds((0.6, 0.6, 0.4))
_plate_curves = [segment(s, e, f"e{k}") for k, (_, s, e) in enumerate(_box_edges(*_plate))]
case(
    name="plate_bore",
    build=lambda: _difference(_box((0.6, 0.6, 0.4)), _cylinder(0.25, 0.9)),
    lo=(-0.6, -0.6, -0.4),
    hi=(0.6, 0.6, 0.4),
    faces=7,
    face_kinds={"plane": 6, "cylinder": 1},
    edges=14,
    closed_edges=2,
    vertices=8,
    euler=0,
    curves=_plate_curves
    + [
        circle((0, 0, 0.4), (0, 0, 1), 0.25, "rim+z"),
        circle((0, 0, -0.4), (0, 0, 1), 0.25, "rim-z"),
    ],
    corners=_endpoints(_plate_curves),
    note="A through bore: genus 1, so chi is 0, not 2.",
)

# 7. Sphere on a box: one circle of intersection.
_sphere_box = _box_bounds((0.5, 0.5, 0.3))
_sphere_box_curves = [
    segment(s, e, f"e{k}") for k, (_, s, e) in enumerate(_box_edges(*_sphere_box))
]
case(
    name="sphere_box",
    build=lambda: _union(_box((0.5, 0.5, 0.3)), _sphere(0.4, (0.0, 0.0, 0.15))),
    lo=(-0.5, -0.5, -0.3),
    hi=(0.5, 0.5, 0.55),
    faces=7,
    face_kinds={"plane": 6, "sphere": 1},
    edges=13,
    closed_edges=1,
    vertices=8,
    euler=2,
    curves=_sphere_box_curves
    + [circle((0, 0, 0.3), (0, 0, 1), float(np.sqrt(0.4**2 - 0.15**2)), "rim")],
    corners=_endpoints(_sphere_box_curves),
    note="A sphere centred below the top face and crossing it transversally.",
)


def _sphere(radius, center=None):
    from cadjoint.sdf.primitives import Sphere
    from cadjoint.sdf.transforms import Translate

    node = Sphere(radius=Scalar(float(radius)))
    if center is not None:
        node = Translate(node, Vector(list(map(float, center))))
    return node


# 8. Two cylinders crossing at right angles, unequal radii (Steinmetz).
def _steinmetz_curve(sign: float, ra: float, rb: float):
    def fn(t):
        y = rb * np.sin(t)
        return (rb * np.cos(t), y, sign * np.sqrt(ra * ra - y * y))

    return fn


case(
    name="steinmetz",
    build=lambda: _union(_cylinder(0.3, 0.7, axis="x"), _cylinder(0.2, 0.7)),
    lo=(-0.7, -0.3, -0.7),
    hi=(0.7, 0.3, 0.7),
    faces=7,
    face_kinds={"plane": 4, "cylinder": 3},
    edges=6,
    closed_edges=6,
    vertices=0,
    euler=2,
    curves=[
        circle((0.7, 0, 0), (1, 0, 0), 0.3, "rimA+"),
        circle((-0.7, 0, 0), (1, 0, 0), 0.3, "rimA-"),
        circle((0, 0, 0.7), (0, 0, 1), 0.2, "rimB+"),
        circle((0, 0, -0.7), (0, 0, 1), 0.2, "rimB-"),
        param(_steinmetz_curve(1.0, 0.3, 0.2), True, "steinmetz+"),
        param(_steinmetz_curve(-1.0, 0.3, 0.2), True, "steinmetz-"),
    ],
    note=(
        "r=0.3 along x, r=0.2 along z: the small cylinder passes through the big "
        "one, leaving two closed quartic space curves and no vertex at all."
    ),
)

# 9. Two cylinders crossing at right angles, equal radii: two ellipses that
#    cross where the surfaces are tangent.
case(
    name="steinmetz_equal",
    build=lambda: _union(_cylinder(0.25, 0.7, axis="x"), _cylinder(0.25, 0.7)),
    lo=(-0.7, -0.25, -0.7),
    hi=(0.7, 0.25, 0.7),
    faces=8,
    face_kinds={"plane": 4, "cylinder": 4},
    edges=8,
    closed_edges=4,
    vertices=2,
    euler=2,
    curves=[
        circle((0.7, 0, 0), (1, 0, 0), 0.25, "rimA+"),
        circle((-0.7, 0, 0), (1, 0, 0), 0.25, "rimA-"),
        circle((0, 0, 0.7), (0, 0, 1), 0.25, "rimB+"),
        circle((0, 0, -0.7), (0, 0, 1), 0.25, "rimB-"),
        # The two ellipses x=z and x=-z cross at the tangent points; the
        # B-rep chains them as the closed curve around each stub, x=|z|
        # (top) and x=-|z| (bottom), each a corner at (0, +-r, 0).
        param(lambda t: (0.25 * np.cos(t), 0.25 * np.sin(t), abs(0.25 * np.cos(t))), True, "x=|z|"),
        param(
            lambda t: (0.25 * np.cos(t), 0.25 * np.sin(t), -abs(0.25 * np.cos(t))), True, "x=-|z|"
        ),
    ],
    corners=np.array([[0.0, 0.25, 0.0], [0.0, -0.25, 0.0]]),
    hard=True,
    tags=("tangent",),
    note=(
        "Equal radii: the intersection is two planar ellipses (x=z, x=-z) "
        "crossing at (0, +-r, 0), where both cylinders share the normal (0, +-1, 0). "
        "Each cylinder splits in two, so 4 curved faces, 4 arcs, 2 vertices with "
        "FOUR incident faces each."
    ),
)

# 10. Two parallel cylinders tangent along a line.
case(
    name="cyl_tangent",
    build=lambda: _union(
        _cylinder(0.3, 0.5, (-0.3, 0.0, 0.0)), _cylinder(0.3, 0.35, (0.3, 0.0, 0.0))
    ),
    lo=(-0.6, -0.3, -0.5),
    hi=(0.6, 0.3, 0.5),
    faces=6,
    face_kinds={"plane": 4, "cylinder": 2},
    edges=5,
    closed_edges=4,
    vertices=2,
    euler=2,
    curves=[
        circle((-0.3, 0, 0.5), (0, 0, 1), 0.3, "rimA+"),
        circle((-0.3, 0, -0.5), (0, 0, 1), 0.3, "rimA-"),
        circle((0.3, 0, 0.35), (0, 0, 1), 0.3, "rimB+"),
        circle((0.3, 0, -0.35), (0, 0, 1), 0.3, "rimB-"),
        segment((0, 0, -0.35), (0, 0, 0.35), "seam", tag="tangent"),
    ],
    corners=np.array([[0.0, 0.0, 0.35], [0.0, 0.0, -0.35]]),
    hard=True,
    tags=("tangent",),
    note=(
        "Two r=0.3 cylinders touching along the z axis: a figure-8 section.  The "
        "solid is NOT a manifold there — the surfaces meet with antiparallel "
        "normals, so the cross product of normals vanishes along the whole seam. "
        "The right answer is two cylinder faces and four rims, with the seam "
        "reported as a tangent contact (no transversal 2-field solution), never as "
        "an analytic edge.  Different heights keep the caps from being coplanar."
    ),
)

# 11. Sphere tangent to a plane face from inside.
_stf = _box_bounds((0.5, 0.5, 0.3))
_stf_curves = [segment(s, e, f"e{k}") for k, (_, s, e) in enumerate(_box_edges(*_stf))]
case(
    name="sphere_tangent_face",
    build=lambda: _union(_box((0.5, 0.5, 0.3)), _sphere(0.35, (0.15, 0.0, 0.08))),
    lo=(-0.5, -0.5, -0.3),
    hi=(0.5, 0.5, 0.43),
    faces=7,
    face_kinds={"plane": 6, "sphere": 1},
    edges=13,
    closed_edges=1,
    vertices=8,
    euler=2,
    curves=_stf_curves
    + [circle((0.15, 0, 0.3), (0, 0, 1), float(np.sqrt(0.35**2 - 0.22**2)), "rim")],
    corners=_endpoints(_stf_curves),
    hard=True,
    tags=("tangent",),
    note=(
        "The sphere pokes out of the top (rim circle r=0.272, its far side 1.5 "
        "cells short of the +x edge) and touches the +x face from inside at "
        "exactly (0.5, 0, 0.08), 4 cells below the top.  A point of tangency is "
        "not a feature: the right answer is the sphere_box topology, 7/13/8, with "
        "no island on the +x face."
    ),
)

# 12. A wall on a slab, the four base creases rounded by a smooth union at
#     several smoothness values relative to the cell.
_wall = _box_bounds((0.15, 0.3, 0.3), (0.1, 0.05, 0.4))
_bracket_curves = [
    (
        Curve(c.name, c.kind, c.closed, c.sampler, "crease")
        if abs(float(c.samples(2)[:, 2].mean()) - 0.2) < 1e-9
        and abs(float(c.samples(2)[0, 2]) - float(c.samples(2)[1, 2])) < 1e-9
        and np.all(np.abs(c.samples(2)[:, 0] - 0.5) > 1e-9)
        and np.all(np.abs(c.samples(2)[:, 0] + 0.5) > 1e-9)
        and np.all(np.abs(c.samples(2)[:, 1] - 0.5) > 1e-9)
        and np.all(np.abs(c.samples(2)[:, 1] + 0.5) > 1e-9)
        else c
    )
    for c in _aabb_union_curves(_slab, _wall)
]
assert sum(1 for c in _bracket_curves if c.tag == "crease") == 4
_bracket_corners = _endpoints(_bracket_curves)
#: The twelve corners a fillet leaves alone: everything but the four where
#: the wall's vertical edges met the slab top.
_bracket_corners_away = _bracket_corners[np.abs(_bracket_corners[:, 2] - 0.2) > 1e-9]
_bracket_corners_away = np.concatenate(
    [
        _bracket_corners_away,
        _bracket_corners[
            (np.abs(_bracket_corners[:, 2] - 0.2) < 1e-9)
            & (np.abs(np.abs(_bracket_corners[:, 0]) - 0.5) < 1e-9)
        ],
    ]
)
assert _bracket_corners_away.shape[0] == 12
_bracket_size = 1.0 + 2 * MARGIN
# smooth_min(a, b, k) = min(a, b) - h^2/(16k), h = max(4k - |a - b|, 0): the
# surface leaves the crease by exactly k and the band spans |a - b| < 4k.
# The graph splits a blend band per LEAF (the quad keeps the nearest leaf and
# loses its patch), so one fillet ring is TWO blend faces with a closed
# mid-band edge between them.  The wall's faces are 0.15 (+y), 0.25 (-y),
# 0.25 (+x) and 0.45 (-x) from the slab's edges and the wall stands 0.5 above
# the slab, so as 4k grows the ring reaches, in turn, the +y edge (4k > 0.15),
# then -y and +x (4k > 0.25), and at 4k = 0.83 nothing is left analytic.
#
#   band < 0.15   : F 13 = 11 plane + 2 blend, E 26 (2 closed), V 16
#   0.15 < 4k < .25: the ring touches the +y edge along a span: the slab top's
#                    hole merges into its outer loop, the +y top edge splits,
#                    one blend/+y-face edge, two vertices: F 13, E 28 (1
#                    closed), V 18
#   0.25 < 4k < .45: three touches cut the slab top into three disks:
#                    F 15 = 13 plane + 2 blend, E 34 (1 closed), V 22
#   4k > 0.83      : every point is within the band: F 2 (blend), E 1 (the
#                    closed mid-band), V 0
_FILLET_EXPECTED = {
    0.2: (13, {"plane": 11, "blend": 2}, 26, 2, 16),
    0.5: (13, {"plane": 11, "blend": 2}, 26, 2, 16),
    1.0: (13, {"plane": 11, "blend": 2}, 28, 1, 18),
    2.0: (15, {"plane": 13, "blend": 2}, 34, 1, 22),
    4.0: (2, {"blend": 2}, 1, 1, 0),
}
for _cells in (0.0, 0.2, 0.5, 1.0, 2.0, 4.0):
    _k = _cells * _bracket_size / TEST_RESOLUTION
    _blend = _cells > 0
    _f, _kinds, _e, _c, _v = _FILLET_EXPECTED.get(_cells, (11, {"plane": 11}, 24, 0, 16))
    case(
        name=f"fillet_{_cells:g}cell" if _blend else "bracket_sharp",
        build=(
            lambda k=_k: _union(
                _box((0.5, 0.5, 0.2)), _box((0.15, 0.3, 0.3), (0.1, 0.05, 0.4)), smoothness=k
            )
        ),
        lo=(-0.5, -0.5, -0.2),
        hi=(0.5, 0.5, 0.7),
        faces=_f,
        face_kinds=_kinds,
        edges=_e,
        closed_edges=_c,
        vertices=_v,
        euler=2,
        curves=_bracket_curves,
        corners=_bracket_corners_away if _blend else _bracket_corners,
        hard=_cells in (0.5, 2.0),
        smoothness=_k,
        sharp=(11, {"plane": 11}, 24, 16) if _blend else None,
        tags=("blend",) if _blend else (),
        note=(
            f"Smoothness k = {_k:.4f} = {_cells:g} cells at {TEST_RESOLUTION} cells, "
            f"band 4k = {4 * _k:.3f} = {4 * _cells:g} cells.  Under the export "
            f"tolerance the textbook answer is F {_f} {_kinds}, E {_e} ({_c} closed), "
            f"V {_v} (see _FILLET_EXPECTED); under the overlay's one-cell rule a "
            "sub-cell fillet must come back as the sharp bracket, 11/24/16."
            if _blend
            else "The hard bracket every fillet case is measured against: 11/24/16."
        ),
    )

# 13. An extruded L profile: a concave corner.
_L = [(-0.3, -0.3), (0.3, -0.3), (0.3, -0.05), (-0.05, -0.05), (-0.05, 0.3), (-0.3, 0.3)]
_L_curves: list[Curve] = []
for _k in range(6):
    _p, _q = _L[_k], _L[(_k + 1) % 6]
    _L_curves.append(segment((*_p, 0.25), (*_q, 0.25), f"top{_k}"))
    _L_curves.append(segment((*_p, -0.25), (*_q, -0.25), f"bot{_k}"))
    _L_curves.append(segment((*_p, -0.25), (*_p, 0.25), f"vert{_k}"))


def _extruded_L():
    from cadjoint.sdf.primitives import ExtrudedPolygon

    return ExtrudedPolygon([Vector2([float(x), float(y)]) for x, y in _L], depth=Scalar(0.5))


case(
    name="extruded_concave",
    build=_extruded_L,
    lo=(-0.3, -0.3, -0.25),
    hi=(0.3, 0.3, 0.25),
    faces=8,
    face_kinds={"plane": 8},
    edges=18,
    closed_edges=0,
    vertices=12,
    euler=2,
    curves=_L_curves,
    corners=_endpoints(_L_curves),
    note="Six walls and two caps; the concave corner at (-0.05, -0.05).",
)

# 14. Oblique cut: box minus a box rotated 0.5 rad about y.
_beta = 0.5
_n = np.array([np.sin(_beta), 0.0, np.cos(_beta)])
_t = np.array([0.6, 0.0, 0.6])
_d = float(_n @ _t - 0.4)
_z1 = (_d - 0.5 * _n[0]) / _n[2]
_x1 = (_d - 0.3 * _n[2]) / _n[0]
_oblique_curves: list[Curve] = []
for _k, (_axis, _s, _e) in enumerate(_box_edges(*_A)):
    if abs(_s[0] - 0.5) < 1e-9 and abs(_s[2] - 0.3) < 1e-9 and _axis == 1:
        continue  # the +x+z edge is cut away entirely
    _s, _e = _s.copy(), _e.copy()
    if _axis == 2 and abs(_s[0] - 0.5) < 1e-9:
        _e[2] = _z1
    if _axis == 0 and abs(_s[2] - 0.3) < 1e-9:
        _e[0] = _x1
    _oblique_curves.append(segment(_s, _e, f"e{_k}"))
_oblique_curves += [
    segment((0.5, -0.4, _z1), (0.5, 0.4, _z1), "cut+x"),
    segment((_x1, -0.4, 0.3), (_x1, 0.4, 0.3), "cut+z"),
    segment((0.5, -0.4, _z1), (_x1, -0.4, 0.3), "cut-y"),
    segment((0.5, 0.4, _z1), (_x1, 0.4, 0.3), "cut+y"),
]


def _oblique():
    from cadjoint.sdf.primitives import Box
    from cadjoint.sdf.transforms import Rotate, Translate

    cutter = Translate(
        Rotate(Box(size=Vector([1.0, 1.0, 0.4])), "y", float(_beta)),
        Vector([float(v) for v in _t]),
    )
    return _difference(_box((0.5, 0.4, 0.3)), cutter)


case(
    name="oblique_cut",
    build=_oblique,
    lo=(-0.5, -0.4, -0.3),
    hi=(0.5, 0.4, 0.3),
    faces=7,
    face_kinds={"plane": 7},
    edges=15,
    closed_edges=0,
    vertices=10,
    euler=2,
    curves=_oblique_curves,
    corners=_endpoints(_oblique_curves),
    hard=True,
    tags=("oblique",),
    note=(
        f"One face of a rotated box slices off the +x+z edge (plane n.p = {_d:.4f}, "
        f"n = ({_n[0]:.4f}, 0, {_n[2]:.4f})).  The cut face is a rectangle whose "
        "edges are not lattice-aligned."
    ),
)


def by_name(name: str) -> Case:
    for item in CASES:
        if item.name == name:
            return item
    raise KeyError(name)


# ── grids ────────────────────────────────────────────────────────────────────


def grid_for(item: Case, resolution: int = TEST_RESOLUTION, offset: float = 0.0) -> GridSpec:
    """A cubic grid around the case's bounding box, shifted by ``offset`` cells.

    The margin is chosen so no lattice plane lands on a face at any of the
    offsets used here, which keeps the extraction from ever having to place
    a vertex on a bit-exact zero.
    """
    lo = np.asarray(item.lo, dtype=np.float64)
    hi = np.asarray(item.hi, dtype=np.float64)
    centre = 0.5 * (lo + hi)
    size = float((hi - lo).max()) + 2 * MARGIN
    spacing = size / resolution
    origin = centre - 0.5 * size + offset * spacing
    return GridSpec.from_bounds(tuple(map(float, origin)), (size, size, size), resolution)


# ── measurement ──────────────────────────────────────────────────────────────


@dataclass
class Measurement:
    """Everything the tests assert and the gallery draws, for one extraction."""

    case: Case
    resolution: int
    offset: float
    blend_tolerance: float | None
    cell: float
    grid: GridSpec
    brep: Any
    faces: int
    face_kinds: dict[str, int]
    edges: int
    closed_edges: int
    open_edges: int
    vertices: int
    ambiguous_vertices: int
    euler: int | None
    non_simple_faces: int
    nan_points: int
    curve_error: dict[str, float]
    curve_coverage: dict[str, float]
    unmatched_edges: int
    vertex_error: float
    spurious_vertices: int
    verdicts: dict[str, int]
    edge_verdicts: list[dict[str, Any]]
    min_normal_sin: float
    t_mesh: float
    t_graph: float

    def summary(self) -> dict[str, Any]:
        worst_err = max(self.curve_error.values(), default=float("nan"))
        worst_cov = min(self.curve_coverage.values(), default=float("nan"))
        return {
            "case": self.case.name,
            "res": self.resolution,
            "offset": self.offset,
            "F": f"{self.faces}/{self.case.faces}",
            "E": f"{self.edges}/{self.case.edges}",
            "V": f"{self.vertices}/{self.case.vertices}",
            "chi": f"{self.euler}/{self.case.euler}",
            "kinds": self.face_kinds,
            "edge_err_cells": worst_err / self.cell,
            "coverage_min": worst_cov,
            "vertex_err_cells": self.vertex_error / self.cell,
            "spurious_v": self.spurious_vertices,
            "unmatched_e": self.unmatched_edges,
            "verdicts": self.verdicts,
            "min_sin": self.min_normal_sin,
            "nan": self.nan_points,
            "t_mesh": self.t_mesh,
            "t_graph": self.t_graph,
        }


def _polyline_segments(brep, edge) -> np.ndarray:
    """An edge as ``(k, 2, 3)`` segments, closed through its vertices."""
    pts = np.asarray(edge.polyline, dtype=np.float64)
    if edge.closed:
        if pts.shape[0] < 2:
            return np.zeros((0, 2, 3))
        ring = np.concatenate([pts, pts[:1]])
        return np.stack([ring[:-1], ring[1:]], axis=1)
    start, stop = edge.vertices
    chain = [pts]
    if start >= 0:
        chain.insert(0, brep.vertices[start].point[None])
    if stop >= 0:
        chain.append(brep.vertices[stop].point[None])
    pts = np.concatenate(chain)
    if pts.shape[0] < 2:
        return np.zeros((0, 2, 3))
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _point_segment_distance(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Min distance from every point to any segment, ``(p,)``."""
    if segments.shape[0] == 0 or points.shape[0] == 0:
        return np.full(points.shape[0], np.inf)
    a0 = segments[:, 0]
    direction = segments[:, 1] - segments[:, 0]
    squared = np.maximum(np.einsum("ij,ij->i", direction, direction), 1e-18)
    best = np.full(points.shape[0], np.inf)
    for start in range(0, points.shape[0], 1024):
        block = points[start : start + 1024]
        t = np.clip(
            (np.einsum("pi,si->ps", block, direction) - np.einsum("si,si->s", a0, direction))
            / squared,
            0.0,
            1.0,
        )
        closest = a0[None] + t[:, :, None] * direction[None]
        best[start : start + 1024] = np.linalg.norm(block[:, None] - closest, axis=-1).min(axis=1)
    return best


def _curve_segments(curve: Curve, count: int = 2048) -> np.ndarray:
    """A true curve as ``(k, 2, 3)`` chords: exact for a segment, sagitta
    ``L^2 / (8 r k^2)`` — below 1e-7 here — for a circle."""
    pts = curve.samples(2 if curve.kind == "segment" else count)
    if curve.closed:
        pts = np.concatenate([pts, pts[:1]])
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _point_cloud_distance(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    best = np.full(points.shape[0], np.inf)
    for start in range(0, points.shape[0], 512):
        block = points[start : start + 512]
        best[start : start + 512] = np.linalg.norm(block[:, None] - cloud[None], axis=-1).min(1)
    return best


def _face_loop_count(brep, face) -> int | None:
    """Loops of a face: the graph's, or 0 for a face with no boundary at all."""
    if face.loops:
        return len(face.loops)
    quads = np.asarray(brep.mesh.quads, dtype=np.int64)[face.quads]
    usage: dict[tuple[int, int], int] = {}
    for row in quads:
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
            a, b = int(row[i]), int(row[j])
            key = (a, b) if a < b else (b, a)
            usage[key] = usage.get(key, 0) + 1
    boundary = sum(1 for count in usage.values() if count == 1)
    return 0 if boundary == 0 else None


def euler_characteristic(brep) -> int | None:
    """``sum_f (2 - loops_f) - open_edges + vertices``; ``None`` if a face is non-simple."""
    total = 0
    for face in brep.faces:
        loops = _face_loop_count(brep, face)
        if loops is None:
            return None
        total += 2 - loops
    total -= sum(1 for edge in brep.edges if not edge.closed)
    total += len(brep.vertices)
    return total


def normal_crossing(brep, edge, points: np.ndarray) -> float:
    """``min |n_a x n_b|`` along an edge — the user's marching direction, measured.

    One where the two patches cross at right angles, zero where they are
    tangent.  Belongs next to ``transversal()`` in ``cadjoint/brep/project.py``.
    """
    import jax
    import jax.numpy as jnp

    a, b = edge.patches
    if a < 0 or b < 0 or points.shape[0] == 0:
        return float("nan")
    probes = jnp.asarray(points, dtype=jnp.float32)

    def unit_grad(f):
        g = jax.vmap(jax.grad(lambda p: jnp.asarray(f(p)).reshape(())))(probes)
        return np.asarray(g, dtype=np.float64)

    ga = unit_grad(brep.patches[a].field)
    gb = unit_grad(brep.patches[b].field)
    na = ga / np.maximum(np.linalg.norm(ga, axis=1, keepdims=True), 1e-12)
    nb = gb / np.maximum(np.linalg.norm(gb, axis=1, keepdims=True), 1e-12)
    return float(np.linalg.norm(np.cross(na, nb), axis=1).min())


def measure(
    item: Case,
    resolution: int = TEST_RESOLUTION,
    offset: float = 0.0,
    blend_tolerance_cells: float | None = None,
    mesh: Any = None,
) -> Measurement:
    """Run the real extraction on a case and compare it with the known answer.

    Args:
        item: The case.
        resolution: Cells per axis.
        offset: Lattice shift in cells.
        blend_tolerance_cells: ``extract_brep``'s blend tolerance in cells;
            ``None`` uses the case's own override or the export default.
        mesh: A dual-contour mesh already extracted on this grid (the fillet
            cases share one between two tolerances).
    """
    from cadjoint.brep import extract_brep
    from cadjoint.meshing.dual_contouring import extract_mesh

    names = [curve.name for curve in item.curves]
    assert len(set(names)) == len(names), f"{item.name}: duplicate curve names"
    grid = grid_for(item, resolution, offset)
    cell = float(grid.spacing[0])
    scene = item.build()
    cells = (
        blend_tolerance_cells if blend_tolerance_cells is not None else item.blend_tolerance_cells
    )
    tolerance = None if cells is None else cells * cell

    t0 = time.perf_counter()
    if mesh is None:
        mesh = extract_mesh(scene, grid)
    t1 = time.perf_counter()
    brep = extract_brep(scene, grid, mesh=mesh, blend_tolerance=tolerance)
    t2 = time.perf_counter()

    kinds: dict[str, int] = {}
    for face in brep.faces:
        kinds[face.kind] = kinds.get(face.kind, 0) + 1

    nan_points = int(np.isnan(brep.points).any(axis=1).sum())
    nan_points += sum(int(np.isnan(e.polyline).any()) for e in brep.edges)
    nan_points += sum(int(np.isnan(v.point).any()) for v in brep.vertices)

    # Curves: each extracted analytic edge is matched to the nearest known
    # curve; its error is the largest distance of its points from that
    # curve.  Coverage asks the other way round: how much of the true curve
    # lies within half a cell of some extracted segment.
    dense = {curve.name: _curve_segments(curve) for curve in item.curves}
    cover = {curve.name: curve.samples(512) for curve in item.curves}
    curve_error = {curve.name: 0.0 for curve in item.curves}
    matched = {curve.name: [] for curve in item.curves}
    edge_verdicts: list[dict[str, Any]] = []
    unmatched = 0
    min_sin = float("inf")
    for edge in brep.edges:
        segs = _polyline_segments(brep, edge)
        pts = segs.reshape(-1, 3) if segs.shape[0] else np.asarray(edge.polyline)
        verdict = (
            "blend"
            if not edge.analytic
            else ("refused" if edge.residual > REFUSE_CELLS * cell else "analytic")
        )
        sin = (
            normal_crossing(brep, edge, np.asarray(edge.polyline))
            if edge.analytic
            else float("nan")
        )
        if np.isfinite(sin):
            min_sin = min(min_sin, sin)
        best_name, best_dist = None, float("inf")
        if pts.shape[0] and dense:
            for name, samples in dense.items():
                d = _point_segment_distance(pts, samples)
                mean = float(d.mean())
                if mean < best_dist:
                    best_name, best_dist = name, mean
        if best_name is not None and best_dist <= 1.0 * cell:
            d = _point_segment_distance(pts, dense[best_name])
            curve_error[best_name] = max(curve_error[best_name], float(d.max()))
            matched[best_name].append(edge.index)
        else:
            best_name = None
            unmatched += 1
        edge_verdicts.append(
            {
                "edge": edge.index,
                "patches": edge.patches,
                "faces": edge.faces,
                "closed": edge.closed,
                "verdict": verdict,
                "residual": edge.residual,
                "points": int(np.asarray(edge.polyline).shape[0]),
                "curve": best_name,
                "min_sin": sin,
            }
        )
    all_segments = np.concatenate(
        [_polyline_segments(brep, e) for e in brep.edges] or [np.zeros((0, 2, 3))]
    )
    curve_coverage = {}
    for curve in item.curves:
        d = _point_segment_distance(cover[curve.name], all_segments)
        curve_coverage[curve.name] = float((d <= COVER_CELLS * cell).mean())
    verdicts: dict[str, int] = {}
    for row in edge_verdicts:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1

    vertex_points = (
        np.stack([v.point for v in brep.vertices]) if brep.vertices else np.zeros((0, 3))
    )
    if item.corners.shape[0] and vertex_points.shape[0]:
        vertex_error = float(_point_cloud_distance(item.corners, vertex_points).max())
        spurious = int((_point_cloud_distance(vertex_points, item.corners) > 0.5 * cell).sum())
    elif item.corners.shape[0]:
        vertex_error = float("inf")
        spurious = 0
    else:
        vertex_error = 0.0
        spurious = int(vertex_points.shape[0])

    return Measurement(
        case=item,
        resolution=resolution,
        offset=offset,
        blend_tolerance=tolerance,
        cell=cell,
        grid=grid,
        brep=brep,
        faces=len(brep.faces),
        face_kinds=kinds,
        edges=len(brep.edges),
        closed_edges=sum(1 for e in brep.edges if e.closed),
        open_edges=sum(1 for e in brep.edges if not e.closed),
        vertices=len(brep.vertices),
        ambiguous_vertices=int(brep.stats.get("ambiguous_vertices", 0)),
        euler=euler_characteristic(brep),
        non_simple_faces=int(brep.stats.get("non_simple_faces", 0)),
        nan_points=nan_points,
        curve_error=curve_error,
        curve_coverage=curve_coverage,
        unmatched_edges=unmatched,
        vertex_error=vertex_error,
        spurious_vertices=spurious,
        verdicts=verdicts,
        edge_verdicts=edge_verdicts,
        min_normal_sin=min_sin if np.isfinite(min_sin) else float("nan"),
        t_mesh=t1 - t0,
        t_graph=t2 - t1,
    )


# ── gallery ──────────────────────────────────────────────────────────────────


def _wire_segments(brep) -> np.ndarray:
    quads = np.asarray(brep.mesh.quads, dtype=np.int64)
    pts = np.asarray(brep.points, dtype=np.float64)
    pairs = np.concatenate([quads[:, [i, j]] for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return pts[pairs]


def drawable(m: Measurement) -> dict[str, Any]:
    """Everything :func:`draw` needs, as plain arrays — picklable, so a gallery
    can be re-rendered without re-extracting."""
    edges = []
    for row, edge in zip(m.edge_verdicts, m.brep.edges):
        segs = _polyline_segments(m.brep, edge)
        if segs.shape[0]:
            edges.append((row["verdict"], segs))
    vertices = np.stack([x.point for x in m.brep.vertices]) if m.brep.vertices else np.zeros((0, 3))
    ambiguous = np.array([len(x.faces) != 3 or not x.analytic for x in m.brep.vertices], dtype=bool)
    return {
        "case": m.case.name,
        "caption": _caption(m),
        "summary": m.summary(),
        "wire": _wire_segments(m.brep),
        "edges": edges,
        "vertices": vertices,
        "ambiguous": ambiguous,
        "curves": [(c.tag, c.samples(256), c.closed) for c in m.case.curves],
        "corners": m.case.corners,
        "lo": np.asarray(m.case.lo),
        "hi": np.asarray(m.case.hi),
    }


def draw(ax, d: dict[str, Any], elev: float = 28.0, azim: float = -55.0, wire: bool = True):
    """The extracted graph over the DC wireframe, true curves behind it."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if wire:
        ax.add_collection3d(Line3DCollection(d["wire"], colors="#c8c8c8", linewidths=0.25))
    for tag, pts, closed in d["curves"]:
        if closed:
            pts = np.concatenate([pts, pts[:1]])
        colour = {"analytic": "#2ca25f", "crease": "#9ecae1", "tangent": "#c51b8a"}[tag]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colour, lw=2.2, alpha=0.55, zorder=1)
    palette = {"analytic": "#08306b", "refused": "#d62728", "blend": "#ff7f0e"}
    for verdict, segs in d["edges"]:
        ax.add_collection3d(
            Line3DCollection(segs, colors=palette[verdict], linewidths=1.2, zorder=3)
        )
    v, amb = d["vertices"], d["ambiguous"]
    if v.shape[0]:
        ax.scatter(v[~amb, 0], v[~amb, 1], v[~amb, 2], s=14, c="#08306b", depthshade=False)
        if amb.any():
            ax.scatter(v[amb, 0], v[amb, 1], v[amb, 2], s=22, c="#d62728", marker="x")
    c = d["corners"]
    if c.shape[0]:
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=40, facecolors="none", edgecolors="#2ca25f")
    lo, hi = d["lo"], d["hi"]
    centre, half = 0.5 * (lo + hi), 0.55 * float((hi - lo).max())
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _caption(m: Measurement) -> str:
    s = m.summary()
    err = "n/a" if not m.curve_error else f"{s['edge_err_cells']:.2f} cell"
    cov = "n/a" if not m.curve_coverage else f"{s['coverage_min']:.2f}"
    return (
        f"{m.case.name}  {m.resolution}³ off {m.offset:g}\n"
        f"F {s['F']}  E {s['E']}  V {s['V']}  χ {s['chi']}\n"
        f"edge err {err}  min cov {cov}  verdicts {m.verdicts}"
    )


def render_case(d: dict[str, Any], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5.6))
    views = ((28, -55, "iso"), (90, -90, "top"), (0, -90, "front"))
    for k, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        draw(ax, d, elev, azim, wire=True)
        ax.set_title(label, fontsize=9, y=0.98)
    fig.suptitle(d["caption"], fontsize=9, family="monospace")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.04, top=0.86, wspace=0.0)
    fig.text(
        0.01,
        0.01,
        "grey: DC wire   navy: analytic edge   red: refused   orange: blend-adjacent   "
        "green: true curve   pink: tangent seam   light blue: crease under blend   "
        "o: true corner   x: ambiguous vertex",
        fontsize=7,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_sheet(drawables: list[dict[str, Any]], path, columns: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(drawables) / columns))
    fig = plt.figure(figsize=(4.2 * columns, 4.4 * rows))
    for k, d in enumerate(drawables):
        ax = fig.add_subplot(rows, columns, k + 1, projection="3d")
        draw(ax, d, wire=True)
        ax.set_title(d["caption"], fontsize=6.5, family="monospace", y=0.97)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.97, wspace=0.0, hspace=0.12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import pickle
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Render the axiom battery.")
    parser.add_argument("--out", default="research/brep-axioms")
    parser.add_argument("--resolution", type=int, nargs="*", default=[32, 64])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--no-offsets", action="store_true")
    parser.add_argument(
        "--rerender", action="store_true", help="redraw from the cached .pkl files only"
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    sheet: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def emit(m: Measurement, stem: str, tolerance: float | None = None) -> dict[str, Any]:
        d = drawable(m)
        row = m.summary()
        if tolerance is not None:
            row["blend_tolerance_cells"] = tolerance
        d["summary"] = row
        with open(cache / f"{stem}.pkl", "wb") as handle:
            pickle.dump(d, handle)
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
        render_case(d, out / f"{stem}.png")
        return d

    if args.rerender:
        for path in sorted(cache.glob("*.pkl")):
            with open(path, "rb") as handle:
                d = pickle.load(handle)
            rows.append(d["summary"])
            render_case(d, out / f"{path.stem}.png")
            if d["summary"]["offset"] == 0.0 and d["summary"]["res"] == max(args.resolution):
                if "blend_tolerance_cells" not in d["summary"]:
                    sheet.append(d)
    else:
        for resolution in args.resolution:
            for item in CASES:
                if args.only and item.name not in args.only:
                    continue
                offsets = OFFSETS if (item.hard and not args.no_offsets) else (0.0,)
                for offset in offsets:
                    m = measure(item, resolution, offset)
                    suffix = f"_{resolution}" + (f"_off{offset:g}" if offset else "")
                    d = emit(m, f"{item.name}{suffix}")
                    if offset == 0.0 and resolution == max(args.resolution):
                        sheet.append(d)
                    if item.tags and "blend" in item.tags and offset == 0.0:
                        m1 = measure(
                            item, resolution, offset, blend_tolerance_cells=1.0, mesh=m.brep.mesh
                        )
                        emit(m1, f"{item.name}{suffix}_tol1cell", 1.0)
    if sheet:
        sheet.sort(key=lambda d: [c.name for c in CASES].index(d["case"]))
        render_sheet(sheet, out / "gallery.png")
    (out / "measurements.json").write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


# ── idea (b): scatter, descend, see where the points snap ────────────────────


def snap_census(item: Case, seeds: int = 30000, seed: int = 0, reach: float = 2e-3) -> dict:
    """Descend uniformly scattered points onto the surface and count where they land.

    A distance field's gradient carries an exterior point to its nearest
    surface point, and the exterior points whose nearest point is a convex
    edge fill a wedge of positive volume — so convex edges attract seeds
    from outside, concave creases from inside, and a tangent seam or a
    coplanar join attracts (almost) nothing.  Returns, per side, the number
    of seeds, how many converged, how many landed within ``reach`` of some
    known curve, and that count per curve.
    """
    import jax
    import jax.numpy as jnp
    from cadjoint.brep.project import project_fields

    rng = np.random.default_rng(seed)
    scene = item.build()
    lo, hi = np.asarray(item.lo) - 0.2, np.asarray(item.hi) + 0.2
    points = rng.uniform(lo, hi, size=(seeds, 3))
    values = np.asarray(jax.vmap(scene)(jnp.asarray(points, dtype=jnp.float32)))
    out: dict = {}
    for side, block in (("outside", points[values > 0]), ("inside", points[values < 0])):
        landed = np.asarray(project_fields([scene], block, max_step=0.05, steps=60))
        residual = np.abs(np.asarray(jax.vmap(scene)(jnp.asarray(landed, dtype=jnp.float32))))
        landed = landed[residual < 1e-4]
        per_curve = {
            c.name: int((_point_segment_distance(landed, _curve_segments(c)) < reach).sum())
            for c in item.curves
        }
        stacked = np.stack(
            [_point_segment_distance(landed, _curve_segments(c)) for c in item.curves]
        )
        out[side] = {
            "seeds": int(block.shape[0]),
            "converged": int(landed.shape[0]),
            "on_edge": int((stacked.min(axis=0) < reach).sum()) if item.curves else 0,
            "per_curve": per_curve,
        }
    return out
