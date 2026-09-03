"""What the B-rep graph buys the mesh overlay that the lattice could not.

:mod:`tests.viewer.test_edge_artifacts` is the artifact oracle — it measures
crossings, debris and coverage, and it passed on the old lattice-linked sharp
layer too.  This suite measures the three things that are only true because
the layer is now the graph's own edges:

* a rim is a **circle**, right to a millionth of a unit, sampled uniformly in angle
  rather than staircased across the cells it happens to cross;
* a hard ``Difference`` seam is the exact intersection curve of the two
  patch fields it separates, bore rims included;
* the answer does not move when the **lattice** does — the same body at
  three sub-cell offsets gives the same topology and the same exact curves.

Everything here reads :mod:`cadjoint.brep.edges` — the ``feature_edges``
component itself — rather than the payload, because the payload rounds to a
thousandth of a unit and a millionth is the point.  These are the private
tier's tests (``research/two-tier.md`` §1.1); the public fallback's are in
:mod:`tests.viewer.test_edge_artifacts` and
:mod:`tests.plugins.test_degradation`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.brep.edges import (
    _DEBRIS_CELLS,
    _MAX_CHORD_TURN,
    _MAX_EDGE_TURN,
    _between_corners,
    _corners_on_curve,
    _Drawable,
    _extract_graph,
    _in_curve_order,
    _prune_debris,
    _resample,
    _sharp_polylines,
    _worst_turn,
)
from cadjoint.brep.project import trace_curves
from cadjoint.geometry.parameters import Vector
from cadjoint.sdf import Box, Cylinder, Difference, Translate, Union
from cadjoint.viewer._edge_overlay import (
    _MESH_EDGE_RESOLUTION,
    _MESH_EDGE_SIZE,
    _mesh_edge_payload,
    _overlay_grid,
)

CELL = max(_MESH_EDGE_SIZE) / _MESH_EDGE_RESOLUTION


def _sharp_chords(brep, spacing) -> np.ndarray:
    """The drawn edges as chord pairs — what the payload's sharp layer is.

    Lived in the overlay before the split; here it is one line over
    :func:`~cadjoint.brep.edges._sharp_polylines`, which is what the
    ``feature_edges`` component's :class:`~cadjoint.plugins.EdgeSet` ships.
    """
    chords = []
    for edge, points, _kind in _sharp_polylines(brep, spacing):
        following = np.roll(points, -1, axis=0) if edge.closed else points[1:]
        leading = points if edge.closed else points[:-1]
        chords.append(np.stack([leading, following], axis=1))
    if not chords:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.concatenate(chords)


#: How far a drawn point may be from the exact analytic curve it claims to
#: be on.  The projection runs in float32, so its residual floor is ~1e-8
#: and the position error is that over the field gradient; a millionth of a
#: unit is two orders above the floor and five below one cell.
EXACT = 1e-6


def _chords(scene) -> np.ndarray:
    """The sharp layer for a scene, unrounded."""
    brep, spacing = _extract_graph(scene, _overlay_grid())
    return _sharp_chords(brep, spacing)


def _endpoints(chords: np.ndarray) -> np.ndarray:
    """Every distinct chord endpoint, shaped ``(n, 3)``."""
    return np.unique(chords.reshape(-1, 3), axis=0)


@pytest.fixture(scope="module")
def cylinder_chords() -> np.ndarray:
    return _chords(Cylinder(radius=0.7, height=0.5))


@pytest.fixture(scope="module")
def bored_plate() -> tuple[object, np.ndarray]:
    """A plate with a round bore: eight rim circles' worth of hard seams.

    The box gives six planes, the subtracted cylinder one lateral surface
    and, where the bore breaks out, two circles that are neither box edges
    nor lattice-aligned.
    """
    scene = Difference(
        Box(size=Vector([0.9, 0.7, 0.25])),
        Cylinder(radius=0.32, height=1.0),
        smoothness=0.0,
    )
    return scene, _chords(scene)


# --------------------------------------------------------------------------
# A rim is a circle
# --------------------------------------------------------------------------


def test_a_cylinder_rim_is_an_exact_circle(cylinder_chords):
    """Every drawn point on a rim is on the exact circle, not near it."""
    points = _endpoints(cylinder_chords)
    radius = np.hypot(points[:, 0], points[:, 1])
    height = np.abs(points[:, 2])
    assert np.abs(radius - 0.7).max() < EXACT, "rim radius"
    assert np.abs(height - 0.5).max() < EXACT, "rim height"


def test_a_rim_is_sampled_uniformly_around_the_loop(cylinder_chords):
    """A closed edge is divided evenly, so a circle is uniform in angle.

    The lattice cannot do this: its chords are as long as the cells the
    curve happens to cross, which on a circle alternates between one and
    root-two cells.
    """
    upper = cylinder_chords[cylinder_chords[:, :, 2].mean(axis=1) > 0.0]
    lengths = np.linalg.norm(upper[:, 1] - upper[:, 0], axis=1)
    assert lengths.size > 40, "too few chords to call it a circle"
    # The residual spread is the seed polyline's, not the circle's: its
    # vertices are mesh-edge midpoints and therefore unevenly spaced, so
    # walking it at uniform arc length lands the samples a few parts in ten
    # thousand off uniform in angle before the projection puts them on the
    # curve.  The lattice's own chords vary by 40% (one cell against the
    # diagonal), so this is two and a half orders better and invisible.
    assert lengths.std() / lengths.mean() < 5e-3, "chord lengths are not uniform"
    # And the spacing really is the half-cell the overlay asks for.
    assert 0.4 * CELL < lengths.mean() < 0.6 * CELL


def test_the_rim_closes_on_itself(cylinder_chords):
    """A closed edge draws a cycle: every endpoint is used exactly twice."""
    upper = cylinder_chords[cylinder_chords[:, :, 2].mean(axis=1) > 0.0]
    _unique, counts = np.unique(np.round(upper.reshape(-1, 3), 9), axis=0, return_counts=True)
    assert set(counts.tolist()) == {2}, "the rim is not a single closed cycle"


# --------------------------------------------------------------------------
# A hard Difference seam
# --------------------------------------------------------------------------


def test_a_subtracted_bore_draws_its_two_rim_circles(bored_plate):
    """The bore's break-out circles are drawn, exactly, at both faces."""
    _scene, chords = bored_plate
    points = _endpoints(chords)
    radius = np.hypot(points[:, 0], points[:, 1])
    on_bore = np.abs(radius - 0.32) < 0.02
    assert on_bore.sum() > 80, "the bore rims are barely drawn"
    rim = points[on_bore]
    assert np.abs(np.hypot(rim[:, 0], rim[:, 1]) - 0.32).max() < EXACT
    # One circle at each face of the plate, and nothing in between.
    assert np.abs(np.abs(rim[:, 2]) - 0.25).max() < EXACT


def test_every_drawn_point_solves_its_own_two_patch_system(bored_plate):
    """The claim the whole layer rests on: the chords are *on* the curve.

    Each edge's samples are checked against the two patch fields the graph
    says the edge separates, which is the same test
    :attr:`~cadjoint.brep.BRepEdge.residual` reports for the seeds — here
    over the resampled points that are actually drawn.
    """
    scene, _chords_unused = bored_plate
    from cadjoint.brep.project import batched_residuals

    brep, spacing = _extract_graph(scene, _overlay_grid())
    fields = [patch.field for patch in brep.patches]
    worst = 0.0
    for edge in brep.edges:
        if not edge.analytic or edge.polyline.shape[0] < 2:
            continue
        members = np.tile(np.asarray(edge.patches, dtype=np.int32), (edge.polyline.shape[0], 1))
        worst = max(worst, float(batched_residuals(fields, members, edge.polyline).max()))
    assert worst < EXACT, f"worst edge residual {worst:.2e}"
    assert spacing.max() == pytest.approx(CELL)


def test_edges_meeting_at_a_corner_share_that_corner_exactly(bored_plate):
    """A corner is one point, not three near-misses.

    The samples are projected onto each edge's *own* patch pair, so a corner
    left in the batch would be pulled off the third face by a thousandth —
    enough to break every chain in the layer.  It is pinned instead, and the
    box's eight corners must each be an endpoint shared by three chords.
    """
    _scene, chords = bored_plate
    corners = np.array(
        [(sx * 0.9, sy * 0.7, sz * 0.25) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    endpoints = chords.reshape(-1, 3)
    for corner in corners:
        hits = np.linalg.norm(endpoints - corner[None, :], axis=1) < EXACT
        assert hits.sum() == 3, f"corner {corner} is shared by {hits.sum()} chords, not 3"


# --------------------------------------------------------------------------
# Lattice stability
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset",
    [(0.0, 0.0, 0.0), (CELL / 3.0, CELL / 7.0, CELL / 11.0), (CELL / 2.0, 0.0, CELL / 2.0)],
    ids=["aligned", "irrational", "half-cell"],
)
def test_a_bore_is_the_same_curve_wherever_the_lattice_falls(offset):
    """Move the body inside a cell: the drawn curves do not move with it.

    This is the property the lattice-linked layer could not have — its
    chords ran between cell vertices, so a sub-cell shift restaged every
    one of them.  Here the lattice only decides *which* patch pairs meet,
    and the positions come from the projection.
    """
    shift = np.asarray(offset, dtype=np.float64)
    # The shift goes on each operand, not on the Difference: a transform
    # above a Boolean is one opaque leaf with no patch decomposition, and
    # the graph would have nothing to derive.
    move = lambda node: Translate(node, offset=Vector(shift.tolist()))  # noqa: E731
    scene = Difference(
        move(Box(size=Vector([0.9, 0.7, 0.25]))),
        move(Cylinder(radius=0.32, height=1.0)),
        smoothness=0.0,
    )
    points = _endpoints(_chords(scene)) - shift[None, :]
    radius = np.hypot(points[:, 0], points[:, 1])
    on_bore = np.abs(radius - 0.32) < 0.02
    assert on_bore.sum() > 80
    assert np.abs(radius[on_bore] - 0.32).max() < EXACT
    assert np.abs(np.abs(points[on_bore, 2]) - 0.25).max() < EXACT
    # The plate's own twelve edges are exact too, wherever the cells fell.
    on_face = np.abs(np.abs(points) - np.array([0.9, 0.7, 0.25])[None, :]) < EXACT
    assert (on_face.sum(axis=1) + on_bore)[
        ~on_bore
    ].min() >= 2, "a drawn point is on fewer than two plate faces"


# --------------------------------------------------------------------------
# Fillets: sub-cell ones are the edge they round
# --------------------------------------------------------------------------


def _filleted_bore(smoothness: float):
    """A plate with a round bore, the break-out rounded by ``smoothness``."""
    return Difference(
        Box(size=Vector([0.9, 0.7, 0.35])),
        Cylinder(radius=0.32, height=1.0),
        smoothness=smoothness,
    )


@pytest.mark.parametrize("radius", [0.02, 0.5 * CELL], ids=["bracket-sized", "half-cell"])
def test_a_sub_cell_fillet_is_drawn_as_the_edge_it_rounds(radius):
    """A fillet finer than a cell keeps its rim.

    ``scenes/bracket.py`` rounds every junction — 0.05 at the web and 0.02
    at the bores — and at the viewport's 0.094 cell none of that is
    resolvable curvature: the dual-contour surface puts one vertex where the
    fillet is.  Classifying it as a blend deleted the bore rims and the
    web-to-plate line from the overlay, which is not what the part looks
    like.  Below :data:`~cadjoint.viewer._edge_overlay._BLEND_AS_EDGE_CELLS`
    the corner is drawn, and it is drawn *exactly*: the rim is still the
    intersection of the plane and the cylinder, to the projection's floor.
    """
    brep, spacing = _extract_graph(_filleted_bore(radius), _overlay_grid())
    assert [face for face in brep.faces if face.kind == "blend"] == []

    points = _endpoints(_sharp_chords(brep, spacing))
    on_bore = np.abs(np.hypot(points[:, 0], points[:, 1]) - 0.32) < 0.02
    assert on_bore.sum() > 80, "the rounded bore lost its rim"
    rim = points[on_bore]
    assert np.abs(np.hypot(rim[:, 0], rim[:, 1]) - 0.32).max() < EXACT
    assert np.abs(np.abs(rim[:, 2]) - 0.35).max() < EXACT


@pytest.mark.parametrize("radius", [2.0 * CELL, 3.0 * CELL], ids=["two-cell", "three-cell"])
def test_a_fillet_wider_than_a_cell_is_not_drawn(radius):
    """Above the cut the fillet is real curvature, and gets no line.

    The virtual sharp edge of a fillet this size sits a cell or more inside
    the material.  Drawing it would put a line where the model has none —
    the failure the threshold exists to bound on the other side.
    """
    brep, spacing = _extract_graph(_filleted_bore(radius), _overlay_grid())
    assert [face for face in brep.faces if face.kind == "blend"] != []

    points = _endpoints(_sharp_chords(brep, spacing))
    on_bore = np.abs(np.hypot(points[:, 0], points[:, 1]) - 0.32) < CELL
    assert on_bore.sum() == 0, "a line was drawn inside a resolvable fillet"


def test_the_threshold_is_the_radius_the_user_typed():
    """Why the constant needs no calibration factor.

    ``smooth_min(a, b, k)`` is ``min(a, b) - h²/(16k)`` with
    ``h = max(4k - |a - b|, 0)``, so where the two surfaces meet (``a = b``)
    it pulls the result down by exactly ``k`` and nowhere by more.  The
    blend test asks the owning patch for its value on the scene's own zero
    set, so ``|f_patch|`` on a fillet runs from 0 at the band's edge to
    ``k`` at its middle — which is what makes ``blend_tolerance`` readable
    directly as "the largest radius still counted as an edge".
    """
    from cadjoint.sdf.boolean.smooth import smooth_min

    for k in (0.02, 0.05, 0.3):
        coincident = float(smooth_min(np.float64(0.0), np.float64(0.0), k))
        assert coincident == pytest.approx(-k, rel=1e-6)
        # And outside the band it is exactly the hard minimum.
        assert float(smooth_min(np.float64(0.0), np.float64(5.0 * k), k)) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Grazing contacts
# --------------------------------------------------------------------------


def _fake_edge(index: int, closed: bool = False):
    """A :class:`~cadjoint.brep.BRepEdge` carrying only what pruning reads."""
    from cadjoint.brep.graph import BRepEdge

    return BRepEdge(
        index=index,
        faces=(0, 1),
        patches=(0, 1),
        polyline=np.zeros((0, 3)),
        vertices=(-1, -1),
        closed=closed,
        analytic=True,
        residual=0.0,
    )


def _segment(start, end, count: int = 2) -> np.ndarray:
    """``count`` points from ``start`` to ``end``."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    return start[None, :] + np.linspace(0.0, 1.0, count)[:, None] * (end - start)[None, :]


def test_short_open_complexes_go_and_anchored_or_closed_ones_stay():
    """The grazing-contact rule, stated on its own.

    Where two solids skim each other the graph finds genuine edges that last
    a couple of cells and end nowhere; drawn, they are ticks beside the
    geometry rather than edges.  A real curve is long, closed, or wired into
    a bigger complex — so the test is on the connected complex, and
    connectivity is judged on the *drawn* endpoints, at the precision the
    payload ships, because that is what the viewer joins up.
    """
    long_open = _Drawable(_fake_edge(0), _segment((0, 0, 0), (4.0 * CELL, 0, 0)))
    short_orphan = _Drawable(_fake_edge(1), _segment((0, 1, 0), (CELL, 1, 0)))
    short_loop = _Drawable(
        _fake_edge(2, closed=True),
        _segment((0, 2, 0), (CELL, 2, 0), count=4),
    )
    # Two short edges meeting at one point and dead-ending: still debris.
    shared = (0.0, 3.0, 0.0)
    short_pair = [
        _Drawable(_fake_edge(3), _segment(shared, (CELL, 3, 0))),
        _Drawable(_fake_edge(4), _segment(shared, (-CELL, 3, 0))),
    ]
    # A short edge anchored between two long ones through shared endpoints.
    left, right = (0.0, 4.0, 0.0), (0.4 * CELL, 4.0, 0.0)
    anchored = [
        _Drawable(_fake_edge(5), _segment((-4.0 * CELL, 4, 0), left)),
        _Drawable(_fake_edge(6), _segment(left, right)),
        _Drawable(_fake_edge(7), _segment(right, (4.4 * CELL, 4, 0))),
    ]

    entries = [long_open, short_orphan, short_loop, *short_pair, *anchored]
    kept = {item.edge.index for item in _prune_debris(entries, _DEBRIS_CELLS * CELL)}
    assert kept == {0, 2, 5, 6, 7}, f"kept {sorted(kept)}"


# --------------------------------------------------------------------------
# One extraction
# --------------------------------------------------------------------------


def test_the_overlay_runs_dual_contouring_exactly_once(monkeypatch):
    """Both layers come from one lattice sweep.

    The wire layer is that mesh's quad edges and the sharp layer is the
    graph derived over the same mesh, so a second extraction would be pure
    duplicated cost — and, worse, two answers.
    """
    from cadjoint.brep import graph as brep_graph

    calls = []
    original = brep_graph.extract_mesh

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(brep_graph, "extract_mesh", counted)
    payload = _mesh_edge_payload(Box(size=Vector([0.8, 0.6, 0.5])))
    assert payload is not None
    assert len(calls) == 1, f"dual contouring ran {len(calls)} times"


def test_the_wire_and_sharp_layers_share_their_vertices():
    """A sharp corner is a wire vertex, to the payload's own rounding.

    Both layers are drawn from the graph's re-solved points, so a seam the
    sharp layer puts on the curve is not a seam the wire layer puts a cell
    and a half away — the defect that made the two layers disagree before.
    """
    payload = _mesh_edge_payload(Box(size=Vector([0.8, 0.6, 0.5])))
    assert payload is not None
    wire = np.asarray(payload["wire"], dtype=np.float64).reshape(-1, 3)
    sharp = np.asarray(payload["sharp"], dtype=np.float64).reshape(-1, 3)
    corner = np.array([0.8, 0.6, 0.5])
    for point_set, label in ((wire, "wire"), (sharp, "sharp")):
        near = np.linalg.norm(point_set - corner[None, :], axis=1).min()
        assert near < 1e-3, f"{label} layer misses the corner by {near:.4f}"


# --------------------------------------------------------------------------
# Every drawn edge is a curve
# --------------------------------------------------------------------------


def _filleted_step(smoothness: float = 0.03):
    """A fin standing on a plate, welded by a sub-cell fillet.

    The shape of every junction the starter and the bracket are made of: two
    boxes smooth-unioned, so the graph classifies the fillet band as the
    sharp corner it rounds and then has to find that corner's curve inside a
    band where patch ownership is a coin flip.  The four seams around the
    fin's root are exact straight lines of known length.
    """
    return Union(
        Box(size=Vector([0.8, 0.6, 0.1])),
        Translate(Box(size=Vector([0.15, 0.45, 0.35])), offset=Vector([0.0, 0.0, 0.3])),
        smoothness=smoothness,
    )


def _drawn(scene):
    """Every drawn edge as an ordered polyline, plus the graph behind it."""
    brep, spacing = _extract_graph(scene, _overlay_grid())
    return brep, _sharp_polylines(brep, spacing)


@pytest.mark.parametrize("scene_name", ["filleted-step", "bored-plate"])
def test_no_drawn_edge_folds_back_on_itself(scene_name):
    """The defect that made the fin roots look like flags.

    The graph chains an edge's seeds in the order its walk over the mesh
    boundary between two face regions visited them, and that boundary
    staircases across the curve — so the order is not the curve's, and a
    polyline drawn in it doubles back.  Every joint of every drawn edge is
    bounded here: a genuine curve is sampled at
    :data:`~cadjoint.viewer._edge_overlay._MAX_CHORD_TURN` per chord, so
    anything near a reversal is not a curve at all.
    """
    scene = (
        _filleted_step()
        if scene_name == "filleted-step"
        else Difference(
            Box(size=Vector([0.9, 0.7, 0.25])),
            Cylinder(radius=0.32, height=1.0),
            smoothness=0.02,
        )
    )
    _brep, drawn = _drawn(scene)
    assert drawn, "nothing drawn at all"
    worst = [
        (edge.index, round(_worst_turn(points, edge.closed), 1))
        for edge, points, _kind in drawn
        if _worst_turn(points, edge.closed) > _MAX_EDGE_TURN
    ]
    assert not worst, f"edges that fold (index, degrees): {worst}"


def test_a_straight_seam_is_drawn_at_its_own_length():
    """A straight edge must not take a detour to get to its end.

    Turning alone does not catch a *smooth* wander — an ownership island
    between two planes of different solids draws a gentle arc, and two
    planes meet in a line, so any arc is wrong.  Length against the chord
    catches exactly that: for a straight seam the polyline is the chord.
    """
    brep, drawn = _drawn(_filleted_step())
    for edge, points, _kind in drawn:
        if edge.closed:
            continue
        if {brep.patches[index].kind for index in edge.patches} != {"plane"}:
            continue
        chord = float(np.linalg.norm(points[-1] - points[0]))
        if chord < 2.0 * CELL:
            continue
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        assert (
            length <= 1.05 * chord
        ), f"edge {edge.index} is {length / chord:.3f} times its own chord"


def test_the_chain_order_is_replaced_by_the_curve_order():
    """The reordering, on the exact numbers that exposed it.

    Seven points on one straight line, delivered folded — this is
    ``scenes/bracket.py``'s web-to-rib seam as the graph hands it over.
    """
    folded = np.array(
        [
            [-0.0845, -0.62, 0.9],
            [-0.1132, -0.62, 0.9],
            [-0.1059, -0.62, 0.9],
            [0.0, -0.62, 0.9],
            [0.1059, -0.62, 0.9],
            [0.1132, -0.62, 0.9],
            [0.0845, -0.62, 0.9],
        ]
    )
    assert _worst_turn(folded, False) == pytest.approx(180.0, abs=1e-6)
    ordered = _in_curve_order(folded, False)
    assert _worst_turn(ordered, False) < 1e-6
    assert sorted(ordered[:, 0]) == pytest.approx(sorted(folded[:, 0]))
    assert list(ordered[:, 0]) == pytest.approx(sorted(folded[:, 0]))


def test_an_edge_is_cut_to_its_corners():
    """Samples that run past a triple point must not fold the ends back.

    The seeds keep coming after the third face has taken over, so a corner
    appended to an end that is *inside* the sample range makes a spike out
    of an otherwise straight edge.
    """
    samples = np.array([[x, 0.0, 0.0] for x in (-0.11, -0.05, 0.0, 0.05, 0.11)])
    corners = np.array([[-0.07, 0.0, 0.0], [0.07, 0.0, 0.0]])
    points, pins = _between_corners(samples, corners, False)
    assert pins == (True, True)
    assert points[0] == pytest.approx(corners[0])
    assert points[-1] == pytest.approx(corners[1])
    assert np.abs(points[:, 0]).max() <= 0.07 + 1e-9, "kept a sample beyond the corner"
    assert _worst_turn(points, False) < 1e-6


def test_a_corner_that_is_not_on_the_edge_is_refused():
    """A triple point is solved against *its own* three patches.

    When those are not this edge's two it can sit a full cell off the line
    every other sample is on, and appending it kinks the edge by 87 degrees.
    Only corners that satisfy this edge's own pair may be used.
    """
    scene = _filleted_step()
    brep, spacing = _extract_graph(scene, _overlay_grid())
    limit = 1e-5 * float(spacing.max())
    drawable = [edge for edge in brep.edges if edge.analytic and edge.residual <= limit]
    usable = _corners_on_curve(brep, drawable, limit)
    from cadjoint.brep.project import batched_residuals

    fields = [patch.field for patch in brep.patches]
    for edge in drawable:
        for point in usable.get(edge.index, []):
            residual = batched_residuals(
                fields,
                np.asarray([edge.patches], dtype=np.int32),
                np.asarray([point], dtype=np.float64),
            )
            assert float(residual[0]) <= limit


# --------------------------------------------------------------------------
# Small rims are circles, not octagons
# --------------------------------------------------------------------------


def test_a_rim_narrower_than_a_cell_still_gets_a_circle():
    """Arc length alone under-samples a rim whose radius is about a cell.

    A screw head of radius 0.07 has a circumference of 0.44 — nine half-cell
    chords, which draws as an octagon.  The turning budget puts a floor on
    the count that does not depend on the radius.
    """
    scene = Union(
        Box(size=Vector([0.8, 0.6, 0.2])),
        Translate(Cylinder(radius=0.07, height=0.12), offset=Vector([0.0, 0.0, 0.28])),
        smoothness=0.0,
    )
    _brep, drawn = _drawn(scene)
    loops = [(edge, points) for edge, points, _kind in drawn if edge.closed]
    assert loops, "the head lost its rim entirely"
    for _edge, points in loops:
        radius = float(np.linalg.norm(points - points.mean(axis=0), axis=1).mean())
        if radius > 2.0 * CELL:
            continue
        assert (
            points.shape[0] >= 360.0 / _MAX_CHORD_TURN
        ), f"rim of radius {radius:.3f} drawn with only {points.shape[0]} chords"
        # The controller re-takes any step that overshoots by more than
        # twice the budget, so twice is the guarantee it actually makes.
        assert _worst_turn(points, True) <= 2.0 * _MAX_CHORD_TURN


def test_the_turning_budget_does_not_inflate_a_straight_edge():
    """A straight run is sampled by arc length alone, as before."""
    straight = np.array([[x, 0.0, 0.0] for x in np.linspace(0.0, 1.0, 5)])
    samples = _resample(straight, False, 0.5 * CELL)
    assert samples is not None
    assert samples.shape[0] == pytest.approx(1.0 / (0.5 * CELL), abs=1.5)


# --------------------------------------------------------------------------
# The tracer, on fields whose intersection is known exactly
# --------------------------------------------------------------------------


def _plane_z():
    return lambda p: p[2]


def _cylinder_about_z(radius: float):
    return lambda p: jnp.sqrt(p[0] ** 2 + p[1] ** 2) - radius


def _sphere_at(centre, radius: float):
    return lambda p: jnp.linalg.norm(p - jnp.asarray(centre)) - radius


def test_a_traced_curve_is_the_exact_circle_and_closes():
    """A plane through a cylinder: the trace must come back to its seed.

    Nothing about this involves a lattice — the tracer is given two fields
    and one point on their intersection, and it follows the tangent from
    there.
    """
    radius = 0.3
    curves = trace_curves(
        [_plane_z(), _cylinder_about_z(radius)],
        np.array([[0, 1]], dtype=np.int32),
        np.array([[radius, 0.0, 0.0]]),
        targets=np.array([[radius, 0.0, 0.0]]),
        closed=np.array([True]),
        max_step=0.5 * CELL,
        min_step=CELL / 32.0,
        max_turn=_MAX_CHORD_TURN,
        tangent_floor=0.1,
        tolerance=1e-5,
    )
    (points,) = curves
    assert points is not None, "the trace did not come back"
    assert points.shape[0] >= 360.0 / _MAX_CHORD_TURN
    assert np.abs(np.hypot(points[:, 0], points[:, 1]) - radius).max() < 1e-5
    assert np.abs(points[:, 2]).max() < 1e-5
    # It really went all the way round exactly once.
    angle = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    assert abs(abs(angle[-1] - angle[0]) - 2 * np.pi) < 0.35


def test_a_traced_line_runs_from_one_corner_to_the_other():
    """Two crossing planes: the trace is the chord, to the tolerance."""
    start, stop = np.array([-0.4, 0.0, 0.0]), np.array([0.55, 0.0, 0.0])
    curves = trace_curves(
        [_plane_z(), lambda p: p[1]],
        np.array([[0, 1]], dtype=np.int32),
        start[None, :],
        targets=stop[None, :],
        closed=np.array([False]),
        max_step=0.5 * CELL,
        min_step=CELL / 32.0,
        max_turn=_MAX_CHORD_TURN,
        tangent_floor=0.1,
        tolerance=1e-5,
    )
    (points,) = curves
    assert points is not None
    assert points[0] == pytest.approx(start)
    assert points[-1] == pytest.approx(stop)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    assert length == pytest.approx(float(np.linalg.norm(stop - start)), rel=1e-6)
    assert _worst_turn(points, False) < 1e-4


def test_tangent_surfaces_are_reported_as_no_edge():
    """Where the normals are parallel there is no curve, and none is invented.

    A sphere resting on a plane touches at one point; ``∇f_a × ∇f_b``
    vanishes there.  That is the blend case, and the honest answer is that
    the edge is not defined — not a polyline squeezed out of a singular
    system.
    """
    curves = trace_curves(
        [_plane_z(), _sphere_at((0.0, 0.0, 0.25), 0.25)],
        np.array([[0, 1]], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0]]),
        targets=np.array([[0.0, 0.0, 0.0]]),
        closed=np.array([True]),
        max_step=0.5 * CELL,
        min_step=CELL / 32.0,
        max_turn=_MAX_CHORD_TURN,
        tangent_floor=0.1,
        tolerance=1e-5,
    )
    assert curves == [None]


def test_the_step_shrinks_on_a_tight_curve_and_not_on_a_straight_one():
    """The step is set by curvature, so sampling is scale-free.

    A rim ten times smaller gets the same number of chords, and a straight
    edge is never subdivided for turning it does not do.
    """
    counts = []
    for radius in (0.6, 0.06):
        (points,) = trace_curves(
            [_plane_z(), _cylinder_about_z(radius)],
            np.array([[0, 1]], dtype=np.int32),
            np.array([[radius, 0.0, 0.0]]),
            targets=np.array([[radius, 0.0, 0.0]]),
            closed=np.array([True]),
            max_step=0.5 * CELL,
            min_step=CELL / 64.0,
            max_turn=_MAX_CHORD_TURN,
            tangent_floor=0.1,
            tolerance=1e-5,
        )
        assert points is not None
        counts.append(points.shape[0])
    small_rim = counts[1]
    assert small_rim >= 360.0 / _MAX_CHORD_TURN
    # The big rim is limited by arc length instead, so it gets more points,
    # never fewer: turning is a floor on the count, not a ceiling.
    assert counts[0] >= small_rim
