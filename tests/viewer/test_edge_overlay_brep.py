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

Everything here reads :func:`~cadjoint.viewer._edge_overlay._sharp_chords`
rather than the payload, because the payload rounds to a thousandth of a
unit and a millionth is the point.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.geometry.parameters import Vector
from cadjoint.sdf import Box, Cylinder, Difference, Translate
from cadjoint.viewer._edge_overlay import (
    _MESH_EDGE_RESOLUTION,
    _MESH_EDGE_SIZE,
    _extract_graph,
    _mesh_edge_payload,
    _sharp_chords,
)

CELL = max(_MESH_EDGE_SIZE) / _MESH_EDGE_RESOLUTION

#: How far a drawn point may be from the exact analytic curve it claims to
#: be on.  The projection runs in float32, so its residual floor is ~1e-8
#: and the position error is that over the field gradient; a millionth of a
#: unit is two orders above the floor and five below one cell.
EXACT = 1e-6


def _chords(scene) -> np.ndarray:
    """The sharp layer for a scene, unrounded."""
    brep, spacing = _extract_graph(scene)
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

    brep, spacing = _extract_graph(scene)
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
# Grazing contacts
# --------------------------------------------------------------------------


def _fake_edge(index: int, corners: tuple[int, int], closed: bool = False):
    """A :class:`~cadjoint.brep.BRepEdge` carrying only what pruning reads."""
    from cadjoint.brep import BRepEdge

    return BRepEdge(
        index=index,
        faces=(0, 1),
        patches=(0, 1),
        polyline=np.zeros((0, 3)),
        vertices=corners,
        closed=closed,
        analytic=True,
        residual=0.0,
    )


def _run(length: float, count: int = 2) -> np.ndarray:
    """``count`` samples spanning ``length`` along x."""
    return np.stack([np.linspace(0.0, length, count), np.zeros(count), np.zeros(count)], axis=1)


def test_short_open_complexes_go_and_anchored_or_closed_ones_stay():
    """The grazing-contact rule, stated on its own.

    Where two solids skim each other the graph finds genuine edges that last
    a couple of cells and end nowhere; drawn, they are ticks beside the
    geometry rather than edges.  A real curve is long, closed, or wired into
    a bigger complex through its corners — so the test is on the complex.
    """
    from cadjoint.viewer._edge_overlay import _DEBRIS_CELLS, _prune_debris

    limit = _DEBRIS_CELLS * CELL
    long_open = (_fake_edge(0, (-1, -1)), _run(4.0 * CELL))
    short_orphan = (_fake_edge(1, (-1, -1)), _run(1.0 * CELL))
    short_loop = (_fake_edge(2, (-1, -1), closed=True), _run(1.0 * CELL))
    # Two short edges that meet at corner 7 and dead-end: still debris.
    short_pair = [(_fake_edge(3, (7, -1)), _run(CELL)), (_fake_edge(4, (7, -1)), _run(CELL))]
    # A short edge anchored into the long complex through shared corners.
    anchored = [
        (_fake_edge(5, (2, 3)), _run(4.0 * CELL)),
        (_fake_edge(6, (3, 4)), _run(0.4 * CELL)),
        (_fake_edge(7, (4, 2)), _run(4.0 * CELL)),
    ]

    entries = [long_open, short_orphan, short_loop, *short_pair, *anchored]
    kept = {edge.index for edge, _samples in _prune_debris(entries, limit)}
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
