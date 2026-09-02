"""The axiom battery, pinned.

Every case in :mod:`tests.brep.axioms` is extracted once at 32 cells and
compared with its textbook B-rep: counts, Euler characteristic, every
analytic curve's error and coverage, every corner's position.  The hard
cases are re-extracted at two more lattice offsets and must give the same
topology.  What the extraction currently gets wrong is marked
``xfail(strict=True)`` with the measured failure in the reason — nothing is
loosened to pass.

Measured on 2026-09-02 (jax 0.8.2, CPU, float32, 32 cells over the shape
plus a 0.33 margin, cell 0.052 on a unit part); the reasons quote those
numbers.  ``python -m tests.brep.axioms`` regenerates the gallery in
``research/brep-axioms/`` from the same list; ``research/brep-axioms.md``
reads the whole battery.
"""

from __future__ import annotations

from functools import cache

import pytest

from tests.brep import axioms
from tests.brep.axioms import CASES, OFFSETS, Measurement

#: Edge points must lie on their true curve to this fraction of a cell.
#: Measured: below 1.2e-5 cell on every case that passes (float32 ulps on a
#: circle re-solved against a cylinder and a plane).
EDGE_TOLERANCE_CELLS = 1e-3
#: Corners likewise; measured below 1e-6 cell where they are right.
VERTEX_TOLERANCE_CELLS = 1e-3
#: A fillet rounded back to its edge under the overlay's one-cell rule is
#: allowed to sit within the fillet's own band of the sharp crease.
CREASE_TOLERANCE_CELLS = 0.5

# ── measured failures ────────────────────────────────────────────────────────
# Keyed by (test, case[, offset]).  Strict: a case that starts passing must be
# taken out of here, and a reason must say what was measured, not what is
# tolerable.

_COPLANAR = (
    "coplanar patches: B's top and bottom lie in A's top and bottom planes, "
    "and each shared plane comes out as TWO faces (A's patch, B's patch) with a "
    "spurious seam between them along x = 0.5.  Measured F 12/10, E 26/24, "
    "V 16/16 at every offset and at 64 cells.  The seam's two-field residual "
    "is exactly 0 (both fields vanish on the whole shared plane), so the "
    "residual gate calls it analytic; |n_a x n_b| along it is 0.0."
)
_STEINMETZ_EQUAL = (
    "equal radii: the two cylinders are tangent at (0, +-r, 0), and nothing "
    "splits cylinder A there.  Measured F 7/8 (A stays ONE face with 4 loops, "
    "B splits in two), E 6/8 (the two V-shaped seams x = +-|z| are single closed "
    "chains), V 0/2.  The seams miss the tangent corners by 0.36 cell at "
    "offset 0 (coverage 0.996), 5e-6 cell at the other offsets; min "
    "|n_a x n_b| along them is 0.07-0.13."
)
_CYL_TANGENT = (
    "tangent cylinders: the DC mesh bridges the sub-cell wedge along the "
    "contact line with a strip of quads, so the seam comes out TWICE, once each "
    "side of the wedge at y = +-0.015 (0.28 cell), and both copies pass the "
    "residual gate (residual 4e-4 = 0.008 cell) as analytic edges with "
    "|n_a x n_b| = 0.07-0.10.  Cylinder B's rims are each split by a one-point "
    "chain where the rim touches cylinder A.  Measured F 6/6, E 8/5, V 4/2; "
    "vertex error 0.25 / 0.50 / 0.22 cell and 0 / 4 / 0 spurious vertices at "
    "offsets 0 / 0.37 / 0.71."
)
_FILLET = {
    "fillet_0.2cell": (
        "k = 0.2 cell, band 0.8 cell: the ring is thinner than a cell and "
        "fragments into 6 blend faces (1-11 quads each) with 17 blend-adjacent "
        "edges and 11 ambiguous vertices.  Measured F 17/13, E 37/26, V 23/16; "
        "the wall's vertical edges start 1 cell above the slab (z 0.249 vs 0.2)."
    ),
    "fillet_0.5cell": (
        "k = 0.5 cell, band 2 cells: measured F 14/13 (2 blend faces plus a "
        "sliver), E 27/26, V 17/16, 2 non-simple faces so chi is undefined; at "
        "offsets 0.37 / 0.71 the topology changes to 13/26/16 and 14/28/18."
    ),
    "fillet_1cell": (
        "k = 1 cell, band 4 cells = 0.21 > the 0.15 gap to the slab's +y edge: "
        "measured F 15/13 (two 2-quad slivers of the wall's -x plane inside the "
        "band), E 33/28, V 22/18; 10 ambiguous vertices."
    ),
    "fillet_2cell": (
        "k = 2 cells, band 8 cells = 0.41: measured F 18/15, E 36/34, V 22/22 at "
        "offset 0 and 16/33/20, 25/48/29 at the other offsets — the count of "
        "sliver faces inside the band depends on the lattice."
    ),
    "fillet_4cell": (
        "k = 4 cells, band 16 cells = 0.83 covers the whole part: the surface "
        "is two blend faces (one per leaf), but 24 one-to-eighteen-quad 'plane' "
        "islands appear wherever a patch's INFINITE plane crosses the displaced "
        "surface (|f_patch| < tolerance there without the patch owning the "
        "point).  Measured F 26/2, E 28/1, V 4/0."
    ),
}
_SHARP = {
    "fillet_0.5cell": (
        "under the one-cell rule a 0.5-cell fillet should be the sharp bracket "
        "11/24/16; measured F 14 (3 slivers, 1 freeform), E 31, V 20, 2 "
        "ambiguous vertices — the band is 2 cells wide and ownership flickers "
        "across it."
    ),
    "fillet_1cell": "measured 17/40/26 under the one-cell rule (band 4 cells).",
    "fillet_2cell": "measured 27/67/44, 8 refused edges (band 8 cells).",
    "fillet_4cell": "measured 67/182/118, 15 refused edges (band 16 cells).",
}

XFAIL: dict[tuple, str] = {
    ("topology", "boxes_coplanar"): _COPLANAR,
    ("edges", "boxes_coplanar"): _COPLANAR + "  2 extracted edges match no curve.",
    ("topology", "steinmetz_equal"): _STEINMETZ_EQUAL,
    ("edges", "steinmetz_equal"): _STEINMETZ_EQUAL,
    ("vertices", "steinmetz_equal"): _STEINMETZ_EQUAL,
    ("topology", "cyl_tangent"): _CYL_TANGENT,
    ("edges", "cyl_tangent"): _CYL_TANGENT,
    ("vertices", "cyl_tangent"): _CYL_TANGENT,
    ("offsets", "cyl_tangent", 0.37): _CYL_TANGENT,
    ("offsets", "cyl_tangent", 0.71): _CYL_TANGENT
    + "  The counts agree across offsets, but cylinder B's rims sit 0.07 cell off "
    "their circles at offset 0.71 (0.006 at offset 0) where the one-point contact "
    "chain breaks them.",
    ("offsets", "fillet_0.5cell", 0.37): _FILLET["fillet_0.5cell"],
    ("offsets", "fillet_0.5cell", 0.71): _FILLET["fillet_0.5cell"],
    ("offsets", "fillet_2cell", 0.37): _FILLET["fillet_2cell"],
    ("offsets", "fillet_2cell", 0.71): _FILLET["fillet_2cell"],
}
for _name, _reason in _FILLET.items():
    XFAIL[("topology", _name)] = _reason
    XFAIL[("blend", _name)] = _reason
    XFAIL[("edges", _name)] = _reason + "  The creases are blend-adjacent chains, not curves."
    XFAIL[("vertices", _name)] = _reason
for _name, _reason in _SHARP.items():
    XFAIL[("sharp", _name)] = _reason


def _mark(test: str, name: str, *extra):
    reason = XFAIL.get((test, name, *extra))
    marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
    return pytest.param(name, *extra, marks=marks, id="-".join(map(str, (name, *extra))))


@cache
def measured(name: str, offset: float = 0.0, tolerance_cells: float | None = None) -> Measurement:
    item = axioms.by_name(name)
    mesh = None
    if tolerance_cells is not None:
        # Share the dual-contour mesh with the export-tolerance run.
        mesh = measured(name, offset).brep.mesh
    return axioms.measure(item, offset=offset, blend_tolerance_cells=tolerance_cells, mesh=mesh)


NAMES = [item.name for item in CASES]
HARD = [item.name for item in CASES if item.hard]
BLENDED = [item.name for item in CASES if item.sharp is not None]


def _report(m: Measurement) -> str:
    return (
        f"{m.case.name}: F {m.faces}/{m.case.faces} {m.face_kinds}, "
        f"E {m.edges}/{m.case.edges} (closed {m.closed_edges}/{m.case.closed_edges}), "
        f"V {m.vertices}/{m.case.vertices}, chi {m.euler}/{m.case.euler}, "
        f"ambiguous {m.ambiguous_vertices}, non-simple {m.non_simple_faces}"
    )


def _assert_topology(m: Measurement, faces, kinds, edges, vertices, closed=None, euler=None):
    report = _report(m)
    assert m.nan_points == 0, report
    assert (m.faces, m.face_kinds) == (faces, kinds), report
    assert m.edges == edges, report
    if closed is not None:
        assert m.closed_edges == closed, report
    assert m.vertices == vertices, report
    if euler is not None:
        assert m.euler == euler, report


@pytest.mark.parametrize("name", [_mark("topology", n) for n in NAMES])
def test_topology_is_the_textbook_one(name):
    m = measured(name)
    c = m.case
    _assert_topology(m, c.faces, c.face_kinds, c.edges, c.vertices, c.closed_edges, c.euler)


@pytest.mark.parametrize("name", [_mark("edges", n) for n in NAMES])
def test_every_analytic_edge_is_on_its_curve_and_whole(name):
    m = measured(name)
    analytic = [c for c in m.case.curves if c.tag == "analytic"]
    if not analytic:
        pytest.skip("no closed-form edges")
    worst = {c.name: m.curve_error[c.name] / m.cell for c in analytic}
    cover = {c.name: m.curve_coverage[c.name] for c in analytic}
    bad_err = {k: v for k, v in worst.items() if v > EDGE_TOLERANCE_CELLS}
    bad_cov = {k: v for k, v in cover.items() if v < 1.0}
    assert not bad_err, f"edge error above {EDGE_TOLERANCE_CELLS} cell: {bad_err}"
    assert not bad_cov, f"true curves not fully covered: {bad_cov}"
    assert m.unmatched_edges == 0, f"{m.unmatched_edges} extracted edges match no known curve"
    assert m.verdicts == {"analytic": m.edges}, m.verdicts


@pytest.mark.parametrize("name", [_mark("vertices", n) for n in NAMES])
def test_every_corner_is_solved_exactly(name):
    m = measured(name)
    if m.case.corners.shape[0] == 0:
        assert m.vertices == 0, f"{m.vertices} vertices on a solid with no corner"
        return
    assert m.vertex_error / m.cell < VERTEX_TOLERANCE_CELLS, m.vertex_error / m.cell
    assert m.spurious_vertices == 0, f"{m.spurious_vertices} vertices off every corner"


@pytest.mark.parametrize(
    "name,offset",
    [_mark("offsets", n, o) for n in HARD for o in OFFSETS if o],
)
def test_the_topology_does_not_depend_on_the_lattice(name, offset):
    """Same counts at every offset — whether or not they are the right ones."""
    base = measured(name)
    m = measured(name, offset)
    key = lambda x: (x.faces, x.face_kinds, x.edges, x.closed_edges, x.vertices, x.euler)  # noqa: E731
    assert key(m) == key(base), f"offset {offset}: {_report(m)}  vs  {_report(base)}"
    if m.case.smoothness == 0 and any(c.tag == "analytic" for c in m.case.curves):
        worst = max(m.curve_error[c.name] for c in m.case.curves if c.tag == "analytic")
        assert worst / m.cell < EDGE_TOLERANCE_CELLS, worst / m.cell


@pytest.mark.parametrize("name", [_mark("blend", n) for n in BLENDED])
def test_a_fillet_is_one_ring_of_two_blend_faces_under_the_export_tolerance(name):
    m = measured(name)
    c = m.case
    _assert_topology(m, c.faces, c.face_kinds, c.edges, c.vertices, c.closed_edges, c.euler)


@pytest.mark.parametrize("name", [_mark("sharp", n) for n in BLENDED])
def test_a_sub_cell_fillet_is_its_edge_under_the_overlay_rule(name):
    item = axioms.by_name(name)
    m = measured(name, 0.0, 1.0)
    if item.smoothness / m.cell >= 1.0:
        # Above a cell the overlay keeps the fillet as curvature: the export
        # answer applies.
        _assert_topology(m, item.faces, item.face_kinds, item.edges, item.vertices)
        return
    faces, kinds, edges, vertices = item.sharp
    _assert_topology(m, faces, kinds, edges, vertices, closed=0, euler=2)
    creases = {c.name: m.curve_coverage[c.name] for c in item.curves if c.tag == "crease"}
    assert all(v == 1.0 for v in creases.values()), creases
    worst = max(m.curve_error[c.name] for c in item.curves if c.tag == "crease") / m.cell
    assert worst < CREASE_TOLERANCE_CELLS, worst


def test_the_euler_helper_agrees_with_the_hand_count():
    """The count used above, on a graph with holes and closed edges."""
    assert axioms.euler_characteristic(measured("plate_bore").brep) == 0
    assert axioms.euler_characteristic(measured("box").brep) == 2
    assert axioms.euler_characteristic(measured("steinmetz").brep) == 2


def test_the_battery_stays_inside_its_budget():
    """Every extraction the suite ran, summed: measured ~2 minutes."""
    total = sum(
        m.t_mesh + m.t_graph for (name, *_), m in list(measured.cache_info() and _cache_items())
    )
    assert total < 240, total


def _cache_items():
    # lru_cache has no public iteration; re-derive the keys the suite used.
    items = []
    for name in NAMES:
        try:
            items.append(((name,), measured(name)))
        except Exception:  # noqa: BLE001 — a crashing case fails its own tests
            pass
    return items
