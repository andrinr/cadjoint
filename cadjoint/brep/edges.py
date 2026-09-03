"""The feature curves of a scene, from the derived B-rep — the ``feature_edges`` kind.

A feature curve is where two patch zero sets meet, and the graph in
:mod:`cadjoint.brep.graph` already finds exactly that: one dual-contouring
pass discovers which patch owns which region of the surface, the boundary
chain between two regions is an edge, and every point on it is placed by
the projection kernel rather than by the lattice.  So the viewer's overlay
runs *one* extraction and reads its answer, instead of re-deriving feature
cells from normal spreads and then spending most of its work undoing the
lattice's artifacts — the staircase, the X-lattices between near-parallel
rails, the orphan ticks.  Each analytic edge is emitted as a polyline
resampled at half the grid spacing and re-projected, so the chords are on
the true curve to projection precision and a rim really is a circle.

This is the private tier's half of the overlay (``research/two-tier.md``
§1.1).  Public cadjoint reaches it only through the ``feature_edges``
plugin kind and the :class:`~cadjoint.plugins.contracts.FeatureEdges`
Protocol; without it the overlay falls back to its own lattice classifier,
which is what :mod:`cadjoint.viewer._edge_overlay` still carries.

:func:`feature_edges` is the contract's entry point.  It returns an
:class:`~cadjoint.plugins.contracts.EdgeSet` and puts the extraction's own
dual-contour mesh in ``stats["mesh"]``, so the caller's wire layer draws
the same re-solved vertices this pass produced rather than running a second
one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from cadjoint.plugins.contracts import EdgeSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cadjoint.brep.graph import BRep, BRepEdge

__all__ = ["feature_edges"]


class _Drawable(NamedTuple):
    """One edge on its way to the payload.

    Attributes:
        edge: The graph edge it came from.
        samples: Its points, ordered along the curve, shaped ``(n, 3)``.
        kind: How the points were produced — ``"traced"`` along
            ``∇f_a × ∇f_b``, or ``"sampled"`` from the lattice seeds when
            the trace had no certified end to start from.
    """

    edge: BRepEdge
    samples: np.ndarray
    kind: str = "traced"


# Newton iterations per projection.  The graph's own default is eight,
# which converges from a cold seed anywhere in its cell; every seed here is
# a mesh-edge midpoint or a chord of an already-solved polyline, so it
# starts within a fraction of a cell and four iterations reach the same
# float32 floor (~1e-8 on the artifact battery) for half the cost.
_PROJECTION_STEPS = 4

# Arc-length spacing of the emitted samples, as a fraction of the grid
# spacing.  Half a cell is twice the density the lattice could ever offer,
# which is what makes a resampled rim read as a circle rather than as a
# 60-gon; a closed edge is divided uniformly, so a circle's samples are
# uniform in angle.
_SHARP_SAMPLE_FRACTION = 0.5

# Largest fillet radius, in cells, that is still drawn as a sharp edge.
#
# ``smooth_min(a, b, k)`` displaces the surface from the sharp corner by
# exactly ``k`` at the midline of the blend band and by nothing at its
# edges, so ``|f_patch|`` on a fillet runs from 0 up to ``k`` — which makes
# the graph's ``blend_tolerance`` *directly* the radius above which a
# rounded corner stops counting as an edge.  No other calibration is
# needed; the tolerance is in the same units as the radius the user typed.
#
# One cell is the cut because that is where the overlay stops being able to
# tell the difference.  A fillet finer than a cell cannot be *shown* as
# curvature on a 64-cell grid — the dual-contour surface rounds it into one
# vertex — so drawing its virtual sharp edge puts a line within a cell of
# the surface, where the viewport reads it as the edge a CAD user expects.
# Above a cell the fillet is a feature the viewport genuinely renders, and
# a line buried inside it would be a line that is not on the model.
#
# Real parts sit well under the cut and read as edges: ``scenes/bracket.py``
# blends at 0.05 and 0.02, ``scenes/starter.py`` at 0.03 and 0.005, against
# a 0.094 cell.  The graph's own default (a thousandth of the grid diagonal,
# 0.0104 here, about a ninth of a cell) is calibrated for *export*, where
# any rounding at all must be honoured; it drew almost nothing on the
# bracket, which is what this replaces.
_BLEND_AS_EDGE_CELLS = 1.0

# Largest ``|f|`` an edge may still carry and be drawn, as a fraction of the
# grid spacing.
#
# The two populations are separated by five orders of magnitude, not by a
# judgement call.  A genuine two-patch curve converges to the float32 floor:
# every real edge of the starter, the bracket and the end-cap lands under
# 1e-7.  Anything that is *not* a curve fails to converge and keeps a
# residual of order whatever it is really sitting on — the gap between two
# operands that never meet, or the width of the fillet band an ownership
# island lives in — and those start at 1e-5.  Nothing at all falls in
# between, on any of the three scenes.
#
# So the gate goes in the empty band, a decade clear of each side.  It has
# to be tight: an island between two planes of *different* solids draws a
# smooth arc that the turning test cannot fault (two planes meet in a line,
# so a smooth arc between them is impossible, but nothing local says so) and
# its residual, 8.6e-4, is the only thing that gives it away.
_EDGE_RESIDUAL_FRACTION = 1e-5

# Sharpest bend a drawn edge may contain, in degrees.
#
# This is the acceptance test for "is this a curve at all", and it is the
# one filter that separates the two things a face-pair boundary can be.
#
# Ownership is decided per quad by ``argmin |f_patch|``, and *inside* a
# fillet band the two patches that meet there are equidistant by
# construction, so which one wins is a coin flip at the sub-quad level.
# The band breaks into little islands, and an island's boundary is a closed
# loop between two patches whose zero sets meet in a *line* — a closed curve
# cannot lie on a line, so the projection strews the loop along it and the
# polyline folds back on itself.  That fold is the flag at the starter's fin
# roots, and it is 180 degrees.
#
# A real curve cannot do that: :func:`_resample` puts a sample every
# :data:`_MAX_CHORD_TURN` of turning, so a genuine edge is bounded near that
# by construction — measured, the worst joint on any real edge of the
# starter is 13 degrees and of the bracket 15.  Forty-five is three times
# the sampling law and a quarter of a fold, so it separates the two
# populations with room to spare and needs no per-scene tuning.
#
# Judging the *drawn* polyline rather than the face it came from is what
# keeps the small rims: a screw head's rim sits on a one-quad disc and would
# lose to any face-size rule, yet it draws as a clean circle at 20 degrees
# per joint.
_MAX_EDGE_TURN = 45.0

# Direction change one drawn chord should make, in degrees.
#
# Arc length alone is the wrong sampling law for a small closed rim: a
# screw head of radius 0.07 against a 0.094 cell gets its whole
# circumference in nine half-cell chords and draws as an octagon.  Turning
# is the scale-free law — a circle of any radius gets about
# ``360 / _MAX_CHORD_TURN`` chords — and it costs nothing on long straight
# edges, where the budget is never the binding constraint.
#
# It is the tracer's step controller target
# (:func:`cadjoint.brep.project.trace_curves`, which guarantees twice this
# per accepted chord) and, in the sampled fallback, a floor on the sample
# count alongside the arc-length one.
_MAX_CHORD_TURN = 15.0

# Smallest sine of the angle between the two patch normals that still
# counts as a crossing.
#
# The traced tangent is ``∇f_a × ∇f_b``, whose length is exactly that sine.
# Where it vanishes the surfaces are tangent rather than crossing and there
# is no curve to trace — which is the blend case, so it is answered with
# "no edge here" rather than pushed through.  A tenth is about six degrees,
# the same floor the retired seam projection used to call a cross product
# reliable.
_TANGENT_FLOOR = 0.1

# How far the tracer may shorten its step, as a fraction of the longest.
# The step falls where a curve bends; below this it gives up rather than
# grinding, which bounds the work per curve at a few hundred points.
_MIN_STEP_FRACTION = 1.0 / 16.0

# Shortest open link complex that is still a feature curve, in cells.
# See :func:`_prune_debris`; the value is the artifact battery's own
# debris threshold (three cells) with a margin.
_DEBRIS_CELLS = 3.25


def _extract_graph(
    scene: Any, grid: Any, *, blend_tolerance: float | None = None, steps: int = _PROJECTION_STEPS
) -> tuple[BRep, np.ndarray]:
    """Derive the B-rep on ``grid``: one dual-contouring pass.

    The extraction owns the pass — :func:`~cadjoint.brep.extract_brep` runs
    the mesher itself and the graph is built over that one mesh, so the
    overlay's two layers and every projected point come from a single
    lattice sweep.

    Surface fitting is off: the overlay wants the graph's topology and its
    edge curves, never a face's closed form, and fitting is a projection
    program plus a gradient program per face.

    The default blend tolerance is the overlay's own, not the graph's
    export-grade one: a fillet finer than one cell is classified as the
    sharp corner it rounds, and so gets drawn.  See
    :data:`_BLEND_AS_EDGE_CELLS`.

    Args:
        scene: Root SDF node.
        grid: The :class:`~cadjoint.meshing.GridSpec`.
        blend_tolerance: Override the default bar.
        steps: Newton iterations per projection.

    Returns:
        ``(brep, spacing)`` — the derived B-rep and the grid spacing.
    """
    import warnings

    from cadjoint.brep.graph import extract_brep

    if blend_tolerance is None:
        blend_tolerance = _BLEND_AS_EDGE_CELLS * float(max(grid.spacing))
    with warnings.catch_warnings():
        # The overlay's grid is a fixed view volume the scene is *expected*
        # to leave — a ground plane spans it edge to edge — so the mesher's
        # open-boundary warning would fire on every compile and say nothing
        # a viewer user can act on.  Clipping to the view is the point.
        warnings.filterwarnings("ignore", message="The isosurface crosses the extraction boundary")
        brep = extract_brep(
            scene,
            grid,
            steps=steps,
            fit_surfaces=False,
            blend_tolerance=float(blend_tolerance),
        )
    return brep, np.asarray(grid.spacing, dtype=np.float64)


def _design_patch_mask(brep: BRep, design_leaves: np.ndarray | None) -> np.ndarray | None:
    """Lift the caller's design-leaf selection to the graph's patch table.

    The rule and its motivation are the public
    :func:`cadjoint.viewer._edge_overlay._design_leaves`'; this only lifts
    it from leaves to the patch table the graph's edges name.  The leaf
    ordering is
    :func:`~cadjoint.meshing.patch_fields.world_frame_leaves`', which the
    graph's decomposition shares, so the indices mean the same thing on
    both sides of the seam.

    Args:
        brep: The derived B-rep.
        design_leaves: World-frame leaf indices, or ``None`` for all.

    Returns:
        A boolean mask over :attr:`~cadjoint.brep.BRep.patches`, or ``None``
            when the caller restricted nothing.
    """
    if design_leaves is None:
        return None
    leaves = np.asarray(design_leaves, dtype=np.int64).reshape(-1)
    return np.isin(np.asarray([patch.leaf for patch in brep.patches], dtype=np.int64), leaves)


def _turning(points: np.ndarray, closed: bool) -> np.ndarray:
    """Direction change at every joint of a polyline, in degrees.

    Zero-length steps are dropped rather than producing a NaN direction, so
    a repeated point neither hides a reversal nor invents one.

    Args:
        points: Ordered points, shaped ``(k, 3)``.
        closed: Whether the polyline closes on itself.

    Returns:
        The turn at each joint, shaped ``(j,)``; empty when there are fewer
            than two usable steps.
    """
    ring = np.concatenate([points, points[:2]]) if closed else points
    steps = np.diff(ring, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    usable = steps[lengths > 1e-12] / lengths[lengths > 1e-12][:, None]
    if usable.shape[0] < 2:
        return np.zeros(0)
    cosine = np.clip(np.einsum("ij,ij->i", usable[:-1], usable[1:]), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _worst_turn(points: np.ndarray, closed: bool) -> float:
    """The sharpest joint in a polyline, in degrees (0 when there is none)."""
    angles = _turning(points, closed)
    return float(angles.max()) if angles.size else 0.0


def _planar_parameter(points: np.ndarray) -> np.ndarray | None:
    """A monotone parameter along a planar curve, or ``None``.

    Fits the points' own plane, then a circle within it.  A real circle gets
    its angle (unwrapped across the widest empty sector, which is where an
    open arc's two ends are and where a closed loop's seam may fall); a
    straight or barely-curved run gets its position along the principal
    axis, which is the same parameter in the infinite-radius limit.

    Args:
        points: Points known to lie on one curve, shaped ``(k, 3)``.

    Returns:
        One parameter per point, or ``None`` when the points are not planar
            enough to trust either fit.
    """
    if points.shape[0] < 2:
        return None
    centred = points - points.mean(axis=0)
    # full_matrices keeps all three right singular vectors even for a batch
    # of two points, so ``basis[2]`` is always a plane normal.
    _values, _spectrum, basis = np.linalg.svd(centred, full_matrices=True)
    spread = float(np.linalg.norm(centred, axis=1).max())
    if spread <= 1e-12:
        return None
    if float(np.abs(centred @ basis[2]).max()) > 0.05 * spread:
        return None  # genuinely non-planar (two cylinders, say): no fit here
    plane = centred @ basis[:2].T

    # Kasa circle fit: minimise |p|^2 - 2 c.p - r^2 + |c|^2 in the plane.
    design = np.column_stack([2.0 * plane, np.ones(plane.shape[0])])
    target = np.sum(plane**2, axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    centre = solution[:2]
    squared = solution[2] + float(centre @ centre)
    radius = float(np.sqrt(squared)) if squared > 0.0 else 0.0
    offsets = plane - centre[None, :]
    deviation = float(np.abs(np.linalg.norm(offsets, axis=1) - radius).max())
    if radius <= 0.0 or radius > 50.0 * spread or deviation > 0.05 * radius:
        return plane[:, 0]  # straight, or too flat to call a circle

    angle = np.arctan2(offsets[:, 1], offsets[:, 0])
    order = np.argsort(angle)
    gaps = np.diff(np.concatenate([angle[order], angle[order][:1] + 2.0 * np.pi]))
    cut = angle[order][int(np.argmax(gaps))]
    return np.mod(angle - cut, 2.0 * np.pi)


def _in_curve_order(points: np.ndarray, closed: bool) -> np.ndarray:
    """Put an edge's solved samples into order *along the curve*.

    The graph hands its seeds over in the order its walk over the mesh edges
    between two face regions visited them, and **that is not the curve's
    order**.  The walk is a clean unbranched path over the mesh, but the
    boundary it follows staircases across the curve, so projecting its
    midpoints onto the curve is many-to-one in a way that scrambles the
    parameter: one bracket edge arrives as
    ``-0.084, -0.113, -0.106, 0, 0.106, 0.113, 0.084`` — seven points all
    exactly on one straight line, ordered as a fold.  Drawn, that is the
    spike the fin roots and web junctions show.

    The order has to come from the curve instead, and the curve is known:
    every point is on it to 4e-8, and a hard CSG intersection of planes,
    cylinders and spheres is planar — a line, an arc, a circle or an
    ellipse.  So fit the plane and parameterise within it
    (:func:`_planar_parameter`).  Nearest-neighbour chaining is *not* a
    substitute: these seeds are midpoints of mesh edges, so consecutive
    gaps range over an order of magnitude (0.007 to 0.085 on that same
    edge) and the walk hops to a near cluster and folds.

    The re-order is kept only if it is actually straighter, so a curve the
    fit cannot describe is left exactly as it arrived.

    Args:
        points: The edge's solved samples, shaped ``(k, 3)``.
        closed: Whether the edge closes on itself.

    Returns:
        The same points, reordered, shaped ``(k, 3)``.
    """
    if points.shape[0] < 4:
        return points
    parameter = _planar_parameter(points)
    if parameter is None:
        return points
    candidate = points[np.argsort(parameter)]
    return candidate if _worst_turn(candidate, closed) < _worst_turn(points, closed) else points


def _corners_on_curve(
    brep: BRep, edges: list[BRepEdge], limit: float
) -> dict[int, list[np.ndarray]]:
    """Per edge, the triple points that actually lie on its own curve.

    A :class:`~cadjoint.brep.BRepVertex` is solved against *its own* three
    patches, and an edge's chain can end on one whose three are not this
    edge's two — an ambiguous corner, or one whose mesh vertex was owned by
    a different set.  Appending such a point drags the polyline a full cell
    off the line every other sample is on: on one bracket edge the interior
    runs along ``y = 0.311`` and the corner sits at ``y = 0.216``, which is
    an 87-degree kink at each end of an otherwise perfectly straight edge.

    So a corner has to earn its place by the same test the edge itself
    passed — its residual against *this* pair of patch fields — and all of
    them are measured in one program, for the reason
    :func:`cadjoint.brep.project.project_batched` documents.

    Args:
        brep: The derived B-rep.
        edges: The edges that are going to be drawn.
        limit: Largest residual a corner may carry, in world units.

    Returns:
        Edge index to the corner points it may use, in no particular order.
    """
    from cadjoint.brep.project import batched_residuals

    points: list[np.ndarray] = []
    members: list[tuple[int, int]] = []
    owners: list[int] = []
    for edge in edges:
        for index in edge.vertices:
            if index < 0:
                continue
            points.append(brep.vertices[index].point)
            members.append(edge.patches)
            owners.append(edge.index)
    if not points:
        return {}
    residual = batched_residuals(
        [patch.field for patch in brep.patches],
        np.asarray(members, dtype=np.int32),
        np.asarray(points, dtype=np.float64),
    )
    usable: dict[int, list[np.ndarray]] = {}
    for owner, point, value in zip(owners, points, residual):
        if value <= limit:
            usable.setdefault(owner, []).append(point)
    return usable


def _between_corners(
    points: np.ndarray, corners: np.ndarray, closed: bool
) -> tuple[np.ndarray, tuple[bool, bool]]:
    """Order the samples along the curve and cut them to the corner span.

    An edge runs *between* its triple points, but its seeds do not stop
    there: the region boundary the graph chained wanders past the corner and
    keeps producing midpoints, which project onto the same intersection line
    beyond where the third face takes over.  On one bracket edge the corners
    sit at ``x = -0.074`` and ``0.074`` while the seeds reach ``-0.113`` to
    ``0.113``, so appending a corner to an end that is *inside* the sample
    range folds the polyline back on itself — a spike, even once the order
    is right.

    Parameterising the samples and the corners together puts all of them on
    one axis, and the edge is then simply the closed interval between the
    corners.  The result is kept only when it is at least as straight as
    leaving the chain alone, so a curve the planar fit cannot describe is
    never mangled by it.

    Args:
        points: The edge's solved samples, shaped ``(k, 3)``.
        corners: Its one or two triple points, shaped ``(c, 3)``.
        closed: Whether the edge closes on itself (always ``False`` here).

    Returns:
        ``(polyline, pins)`` — the points from corner to corner, and which
            ends are triple points.
    """
    combined = np.concatenate([points, corners])
    is_corner = np.zeros(combined.shape[0], dtype=bool)
    is_corner[points.shape[0] :] = True
    parameter = _planar_parameter(combined)
    fallback = _attach_corners(_in_curve_order(points, closed), corners)
    if parameter is None:
        return fallback
    order = np.argsort(parameter)
    ordered, flags = combined[order], is_corner[order]
    marks = np.flatnonzero(flags)
    if marks.size == 2:
        span = ordered[marks[0] : marks[-1] + 1]
        pins = (True, True)
    else:
        # One corner: the free end is whichever side carries more of the
        # curve, so the corner cuts the other side off.
        mark = int(marks[0])
        leading = mark >= ordered.shape[0] - 1 - mark
        span = ordered[: mark + 1] if leading else ordered[mark:]
        pins = (False, True) if leading else (True, False)
    if span.shape[0] < 2 or _worst_turn(span, closed) > _worst_turn(fallback[0], closed):
        return fallback
    return span, pins


def _attach_corners(
    points: np.ndarray, corners: np.ndarray
) -> tuple[np.ndarray, tuple[bool, bool]]:
    """Put each corner at whichever end of an ordered polyline it is nearer."""
    if corners.shape[0] == 2:
        head, tail = corners[0], corners[1]
        if np.linalg.norm(points[0] - head) > np.linalg.norm(points[0] - tail):
            head, tail = tail, head
        return np.concatenate([head[None, :], points, tail[None, :]]), (True, True)
    corner = corners[0]
    if np.linalg.norm(points[0] - corner) <= np.linalg.norm(points[-1] - corner):
        return np.concatenate([corner[None, :], points]), (True, False)
    return np.concatenate([points, corner[None, :]]), (False, True)


def _edge_polyline(
    edge: BRepEdge, corners: list[np.ndarray]
) -> tuple[np.ndarray, tuple[bool, bool]]:
    """An edge's solved points, in curve order and cut to its corners.

    The chain's seeds are mesh-edge *midpoints*, so the polyline stops half
    a cell short of each end.  Where the chain ended on a
    :class:`~cadjoint.brep.BRepVertex` that corner is the exact endpoint, and
    appending it both completes the curve and makes the edges meeting there
    share one endpoint exactly — which is what turns the sharp layer into
    connected chains rather than a set of near-touching arcs.

    Two things have to happen before that append is safe, and both are about
    the seeds not being in the curve's order to begin with:
    :func:`_in_curve_order` for the order, :func:`_between_corners` for the
    extent.

    Args:
        edge: The edge to read.
        corners: Its triple points that are certified to lie on *this*
            edge's curve — see :func:`_corners_on_curve`.

    Returns:
        ``(polyline, pins)`` — the points, shaped ``(k, 3)`` and without a
            repeated first point when closed, and whether each end is a
            triple point.
    """
    if edge.closed or not corners:
        return _in_curve_order(edge.polyline, edge.closed), (False, False)
    return _between_corners(edge.polyline, np.asarray(corners, dtype=np.float64), edge.closed)


def _resample(polyline: np.ndarray, closed: bool, spacing: float) -> np.ndarray | None:
    """Samples along a polyline, evenly in arc length and bounded in turning.

    The count is chosen so the samples divide the curve *evenly*: a closed
    edge is cut into equal arcs with no short closing piece, which for a rim
    is uniform sampling in angle.

    Arc length alone is not enough on a *small* rim.  A screw head of radius
    0.065 has a circumference of 0.41 — nine half-cell chords — and draws as
    an octagon however exactly each of its nine corners is placed.  So the
    curve's own turning sets a second floor on the count
    (:data:`_MAX_CHORD_TURN`), which is scale-free: any circle gets at least
    ``360 / _MAX_CHORD_TURN`` chords, and a straight edge is unaffected
    because its turning budget is never binding.

    Args:
        polyline: Ordered points, shaped ``(k, 3)``.
        closed: Whether the polyline closes on itself.
        spacing: Target arc length between consecutive samples.

    Returns:
        The samples, shaped ``(n, 3)``, or ``None`` for a polyline with no
            length to walk along.
    """
    points = np.asarray(polyline, dtype=np.float64)
    if closed and points.shape[0] >= 2:
        points = np.concatenate([points, points[:1]])
    if points.shape[0] < 2:
        return None
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(steps.sum())
    if total <= 0.0:
        return None
    knots = np.concatenate([[0.0], np.cumsum(steps)])
    count = max(
        int(round(total / spacing)),
        int(np.ceil(float(_turning(polyline, closed).sum()) / _MAX_CHORD_TURN)),
        1,
    )
    stations = np.linspace(0.0, total, count + 1)
    if closed:
        stations = stations[:-1]
    return np.stack([np.interp(stations, knots, points[:, axis]) for axis in range(3)], axis=1)


def _prune_debris(entries: list[_Drawable], limit: float) -> list[_Drawable]:
    """Drop link complexes that are open and shorter than *limit*.

    Where two solids graze — the artifact battery's ring skimming a roof —
    the graph finds short genuine edges around the contact: curves that
    exist, last a cell or two, and end nowhere.  Drawn, they read as ticks
    beside the geometry rather than as edges.  A real feature curve is long,
    or closed, or wired into a bigger complex through its corners, so the
    test is on the connected complex and not on the single edge.

    Connectivity is judged on the **drawn endpoints**, at the precision the
    payload ships, because that is what the viewer actually joins up.  Two
    edges that nominally meet at the same triple point but whose polylines
    stop at different places are two fragments, whatever the graph's indices
    say — and an edge that ends on a seed proxy rather than a certified
    corner is exactly that case.

    Args:
        entries: The drawable edges and their points.
        limit: Shortest total arc length an open complex may have.

    Returns:
        The surviving entries, in the order given.
    """
    parent: dict[tuple[float, ...], tuple[float, ...]] = {}

    def find(node: tuple[float, ...]) -> tuple[float, ...]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def key(point: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(point, 3))

    ends: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    degree: dict[tuple[float, ...], int] = {}
    for edge, samples, _kind in entries:
        pair = (key(samples[0]), key(samples[-1]))
        ends.append(pair)
        for node in pair:
            parent.setdefault(node, node)
        if not edge.closed:
            for node in pair:
                degree[node] = degree.get(node, 0) + 1
    for left, right in ends:
        if find(left) != find(right):
            parent[find(left)] = find(right)

    lengths: dict[tuple[float, ...], float] = {}
    open_roots: set[tuple[float, ...]] = set()
    for (edge, samples, _kind), pair in zip(entries, ends):
        root = find(pair[0])
        ring = np.concatenate([samples, samples[:1]]) if edge.closed else samples
        lengths[root] = lengths.get(root, 0.0) + float(
            np.linalg.norm(np.diff(ring, axis=0), axis=1).sum()
        )
        if not edge.closed and any(degree.get(node, 0) < 2 for node in pair):
            open_roots.add(root)
    return [
        entry
        for entry, pair in zip(entries, ends)
        if find(pair[0]) not in open_roots or lengths[find(pair[0])] >= limit
    ]


def _traced_polylines(
    brep: BRep,
    spacing: np.ndarray,
    edges: list[BRepEdge],
    corners: dict[int, list[np.ndarray]],
) -> tuple[dict[int, np.ndarray], list[BRepEdge]]:
    """Trace each edge along ``∇f_a × ∇f_b`` instead of ordering its seeds.

    The lattice is a poor witness for where an edge's points are: the seeds
    are midpoints of the mesh boundary between two face regions, that
    boundary staircases across the curve, and the order it is visited in is
    not the curve's — which is what drew the starter's fin roots as flags.
    But the lattice is a *fine* witness for where an edge starts and which
    two patches it belongs to, and that is all a trace needs
    (:func:`cadjoint.brep.project.trace_curves`).

    Seeds come from the graph's own triple points where it has them, so a
    traced edge begins and ends exactly on the corners it shares with its
    neighbours; a closed edge starts anywhere on itself and stops when it
    comes back.  An edge with no certified corner cannot say where it ends,
    so it is handed back for the sampled fallback.

    Args:
        brep: The derived B-rep.
        spacing: The grid spacing, shaped ``(3,)``.
        edges: The edges to draw.
        corners: Per edge, its certified triple points.

    Returns:
        ``(traced, remaining)`` — the polylines that came back, by edge
            index, and the edges the tracer could not take.
    """
    from cadjoint.brep.project import trace_curves

    step = _SHARP_SAMPLE_FRACTION * float(spacing.min())
    limit = _EDGE_RESIDUAL_FRACTION * float(spacing.max())
    plans: list[tuple[BRepEdge, np.ndarray, np.ndarray, bool]] = []
    remaining: list[BRepEdge] = []
    for edge in edges:
        own = corners.get(edge.index, [])
        if edge.closed and edge.polyline.shape[0]:
            seed = edge.polyline[0]
            plans.append((edge, seed, seed, True))
        elif len(own) == 2:
            plans.append((edge, own[0], own[1], False))
        elif len(own) == 1 and edge.polyline.shape[0]:
            # One corner: the far end is unknown, so aim at the seed
            # farthest from it and let the trace stop when it gets there.
            far = edge.polyline[int(np.argmax(np.linalg.norm(edge.polyline - own[0], axis=1)))]
            plans.append((edge, own[0], far, False))
        else:
            remaining.append(edge)
    if not plans:
        return {}, remaining

    curves = trace_curves(
        [patch.field for patch in brep.patches],
        np.asarray([edge.patches for edge, _s, _t, _c in plans], dtype=np.int32),
        np.asarray([seed for _e, seed, _t, _c in plans], dtype=np.float64),
        targets=np.asarray([target for _e, _s, target, _c in plans], dtype=np.float64),
        closed=np.asarray([closed for _e, _s, _t, closed in plans], dtype=bool),
        max_step=step,
        min_step=_MIN_STEP_FRACTION * step,
        max_turn=_MAX_CHORD_TURN,
        tangent_floor=_TANGENT_FLOOR,
        tolerance=limit,
    )
    traced: dict[int, np.ndarray] = {}
    for (edge, _seed, _target, _closed), curve in zip(plans, curves):
        if curve is None:
            remaining.append(edge)
        else:
            traced[edge.index] = curve
    return traced, remaining


def _sampled_polylines(
    brep: BRep,
    spacing: np.ndarray,
    edges: list[BRepEdge],
    corners: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    """The fallback: order the graph's own seeds and re-project them.

    Used only where a trace cannot start — an edge with no certified triple
    point has no end to stop at.  Its seeds are put in the curve's order
    (:func:`_in_curve_order`), resampled, and projected in one batched call;
    a corner, being already the three-patch solution, is pinned rather than
    projected onto a two-patch subset that would drag it off the third face.

    Args:
        brep: The derived B-rep.
        spacing: The grid spacing, shaped ``(3,)``.
        edges: The edges to draw this way.
        corners: Per edge, its certified triple points.

    Returns:
        The polylines, by edge index.
    """
    from cadjoint.brep.project import project_batched

    entries: list[tuple[BRepEdge, np.ndarray, tuple[bool, bool]]] = []
    for edge in edges:
        polyline, pins = _edge_polyline(edge, corners.get(edge.index, []))
        samples = _resample(polyline, edge.closed, _SHARP_SAMPLE_FRACTION * float(spacing.min()))
        if samples is not None and samples.shape[0] >= 2:
            entries.append((edge, samples, pins))
    if not entries:
        return {}

    seeds = np.concatenate([samples for _edge, samples, _pins in entries])
    solved = project_batched(
        [patch.field for patch in brep.patches],
        np.concatenate(
            [
                np.tile(np.asarray(edge.patches, dtype=np.int32), (samples.shape[0], 1))
                for edge, samples, _pins in entries
            ]
        ),
        seeds,
        max_step=0.5 * float(np.linalg.norm(spacing)),
        steps=_PROJECTION_STEPS,
    )
    pinned = np.concatenate(
        [
            np.asarray([pins[0]] + [False] * (samples.shape[0] - 2) + [pins[1]], dtype=bool)
            if samples.shape[0] >= 2
            else np.zeros(samples.shape[0], dtype=bool)
            for _edge, samples, pins in entries
        ]
    )
    solved = np.where(pinned[:, None], seeds, solved)

    out: dict[int, np.ndarray] = {}
    cursor = 0
    for edge, samples, _pins in entries:
        stop = cursor + samples.shape[0]
        out[edge.index] = solved[cursor:stop]
        cursor = stop
    return out


def _sharp_polylines(
    brep: BRep, spacing: np.ndarray, design: np.ndarray | None = None
) -> list[_Drawable]:
    """The design's analytic edges, each as one ordered polyline.

    Three things have to be true before a face-pair boundary is a drawable
    curve, and each is checked against the thing it actually bounds:

    * its seeds are *on* both patches — :data:`_EDGE_RESIDUAL_FRACTION`;
    * the two patches genuinely cross there rather than touching, so a
      tangent exists to follow — :data:`_TANGENT_FLOOR`, inside the trace;
    * and what comes out does not fold — :data:`_MAX_EDGE_TURN`, which is
      the last word for anything the fallback produced.

    Args:
        brep: The derived B-rep.
        spacing: The grid spacing, shaped ``(3,)``.
        design: A patch mask from :func:`_design_patch_mask`, or ``None``.

    Returns:
        One :class:`_Drawable` per drawn edge; a closed edge's points do not
            repeat the first.
    """
    limit = _EDGE_RESIDUAL_FRACTION * float(spacing.max())
    drawable = [
        edge
        for edge in brep.edges
        if edge.analytic
        and edge.residual <= limit
        # Both neighbours must be the design's: a curve that is half on
        # scenery is not a crease the user is authoring, and drawing it
        # would put sharp links on the board the heat sink sits on.
        and (design is None or (design[edge.patches[0]] and design[edge.patches[1]]))
    ]
    corners = _corners_on_curve(brep, drawable, limit)
    polylines, remaining = _traced_polylines(brep, spacing, drawable, corners)
    traced = set(polylines)
    polylines.update(_sampled_polylines(brep, spacing, remaining, corners))

    # The turning test first: an edge it rejects must not still be counted
    # as the anchor that keeps a short neighbour's complex alive.
    entries = [
        _Drawable(
            edge,
            polylines[edge.index],
            "traced" if edge.index in traced else "sampled",
        )
        for edge in drawable
        if edge.index in polylines
        and _worst_turn(polylines[edge.index], edge.closed) <= _MAX_EDGE_TURN
    ]
    return list(_prune_debris(entries, _DEBRIS_CELLS * float(spacing.max())))


def feature_edges(
    scene: Any,
    grid: Any,
    *,
    design_leaves: np.ndarray | None = None,
    blend_tolerance: float | None = None,
    steps: int = _PROJECTION_STEPS,
) -> EdgeSet:
    """The curves where two patch zero sets cross, resampled and re-projected.

    The ``feature_edges`` contract
    (:class:`cadjoint.plugins.contracts.FeatureEdges`).

    Args:
        scene: Root SDF node.
        grid: The :class:`~cadjoint.meshing.GridSpec` to discover on.
        design_leaves: World-frame leaf indices whose curves are drawn —
            both patches of a curve must belong (see
            :func:`cadjoint.viewer._edge_overlay._design_leaves` for the
            rule).  ``None`` draws every curve.
        blend_tolerance: ``|f_patch|`` above which a surface is a blend
            rather than the patch it rounds; defaults to
            :data:`_BLEND_AS_EDGE_CELLS` times the coarsest spacing.
        steps: Newton iterations per projection.

    Returns:
        The :class:`~cadjoint.plugins.contracts.EdgeSet`.  ``stats["mesh"]``
            is ``(points, quads)`` of the extraction's own dual-contour
            pass, so the caller need not run a second one.
    """
    brep, spacing = _extract_graph(scene, grid, blend_tolerance=blend_tolerance, steps=steps)
    patches = _design_patch_mask(brep, design_leaves)
    drawn = _sharp_polylines(brep, spacing, patches)
    polylines = tuple(np.asarray(item.samples, dtype=np.float64) for item in drawn)
    edges = [item.edge for item in drawn]
    return EdgeSet(
        polylines=polylines,
        closed=np.array([bool(edge.closed) for edge in edges], dtype=bool),
        patches=np.array(
            [[int(edge.patches[0]), int(edge.patches[1])] for edge in edges],
            dtype=np.int32,
        ).reshape(-1, 2),
        kind=tuple(item.kind for item in drawn),
        residual=np.array([float(edge.residual) for edge in edges]),
        vertices=np.array(
            [[int(edge.vertices[0]), int(edge.vertices[1])] for edge in edges],
            dtype=np.int32,
        ).reshape(-1, 2),
        stats={
            "curves": len(edges),
            "graph_edges": len(brep.edges),
            "mesh": (
                np.asarray(brep.points, dtype=np.float64),
                np.asarray(brep.mesh.quads, dtype=np.int64),
            ),
        },
    )
