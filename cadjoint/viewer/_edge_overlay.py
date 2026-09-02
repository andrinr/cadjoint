"""Feature-edge overlay for the viewer's mesh view.

Everything behind ``mode="mesh"``: dual-contour the scene on the viewer's
own grid and split the result into the two overlay layers the frontend
draws — the mesh's native quad edges ("wire") and the chords of its true
feature curves ("sharp").

**The sharp layer is the derived B-rep's edges, not the mesh's.**  A feature
curve is where two patch zero sets meet, and :mod:`cadjoint.brep` already
finds exactly that: one dual-contouring pass discovers which patch owns
which region of the surface, the boundary chain between two regions is an
edge, and every point on it is placed by the projection kernel rather than
by the lattice.  So the overlay runs *one* extraction and reads its answer,
instead of re-deriving feature cells from normal spreads and then spending
most of its work undoing the lattice's artifacts — the staircase, the
X-lattices between near-parallel rails, the orphan ticks.  Each analytic
edge is emitted as a polyline resampled at half the grid spacing and
re-projected, so the chords are on the true curve to projection precision
and a rim really is a circle.

The wire layer stays the dual-contour quad edges, drawn on the same
re-solved vertices (:attr:`~cadjoint.brep.BRep.points`) so the two layers
agree everywhere.

Only the extraction lives here.  The worker mode that calls it (executing
the user's program first) stays in :mod:`cadjoint.viewer._compile_worker`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cadjoint.meshing.patch_fields import world_frame_leaves

#: The overlay's name for the leaf split it shares with the mesher and the
#: graph: the design-subtree rule below is defined over exactly these
#: leaves, and :attr:`cadjoint.brep.BRep.decomposition` carries the same
#: list in the same order.
_world_frame_leaves = world_frame_leaves

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cadjoint.brep import BRep, BRepEdge

# Mesh-edge view settings.  The grid matches the raymarcher's view volume.
# Detection stays dense rather than Lipschitz-pruned: user-written fields
# can exceed any assumed gradient bound, and a hole in the viewer is worse
# than the ~100 ms this costs.
_MESH_EDGE_BOUNDS = (-3.0, -3.0, -3.0)
_MESH_EDGE_SIZE = (6.0, 6.0, 6.0)
_MESH_EDGE_RESOLUTION = 64

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

# Largest ``|f|`` an edge may still carry and be drawn, as a fraction of the
# grid spacing.  A genuine two-patch curve converges to ~1e-8; a pair of
# patches that never actually meet (the parallel faces of a sub-cell slab,
# two operands passing within a cell of each other) is left unmoved by the
# projection's rank guard and keeps a residual of order the gap.  This is
# the same tenth-of-a-cell test the leaf-level seam projection used.
_EDGE_RESIDUAL_FRACTION = 0.1

# Shortest open link complex that is still a feature curve, in cells.
# See :func:`_prune_debris`; the value is the artifact battery's own
# debris threshold (three cells) with a margin.
_DEBRIS_CELLS = 3.25


def _carries_construction(node: Any) -> bool:
    """Whether *node*'s subtree carries a construction-layer face set."""
    from cadjoint.construction.faces import FaceSet

    if isinstance(getattr(node, "faces", None), FaceSet):
        return True
    children = getattr(node, "children", None)
    return callable(children) and any(_carries_construction(child) for child in children())


def _design_leaves(leaves: list[Any]) -> np.ndarray | None:
    """Which world-frame leaves the construction tree owns, or ``None``.

    **The design-subtree rule.**  Sharp feature edges are the design's own
    creases and CSG seams.  Context primitives dropped in around it — a
    board, a die, screw heads, decoupling caps — are scenery: the user is
    not authoring their edges, and drawing feature curves on them clutters
    the overlay with geometry the physics never sees.

    No new flag is needed, because the scene already draws the line.  Every
    construction generator attaches a :class:`FaceSet` to the SDF it returns
    — ``extrude``, ``revolve`` and ``loft`` through ``attach_faces``, and the
    ``Solid.*`` primitive mirrors through the same call — so a leaf belongs
    to the design exactly when its subtree carries one.  A leaf assembled by
    hand out of raw SDF primitives carries nothing, and is scenery.

    Returns ``None`` when *no* leaf carries a mirror: a scene written
    directly as an SDF expression is all design, and restricting it to
    nothing would delete its sharp layer entirely.

    **Exactly one thing is restricted**, and it is the sharp layer's: which
    of the graph's edges are drawn (:func:`_design_patches` lifts the rule
    from leaves to patches, and an edge is drawn when *both* its patches
    pass — a curve half on scenery is not a crease the user is authoring).
    Everything else stays whole-scene.  In particular the graph is derived
    over every leaf, so the wire layer keeps its full mesh wireframe with
    every vertex re-solved, and a vertex on scenery is classified as scenery
    rather than handed to whichever design operand happens to be nearest.

    The net effect: scenery keeps its full mesh wireframe, solved exactly as
    the design's is, and grows no sharp edges.
    """
    marked = np.array([_carries_construction(leaf) for leaf in leaves], dtype=bool)
    return np.flatnonzero(marked) if marked.any() else None


# ── the retired leaf-level seam projection ───────────────────────────────────
#
# Before the sharp layer became the B-rep graph's edges, the overlay found
# CSG seams itself: group the mesh vertices whose surface ownership flips by
# the *operand set* they sit between, then Newton-project each group onto
# the common zero set of those operands' whole SDFs.  The graph does the
# same thing one level down — on patch fields rather than leaf fields, which
# is what makes a box's twelve edges twelve curves instead of one ownership
# flip — so nothing below is on the payload path any more.
#
# The four functions stay because the pair at the end of them is a *proved
# equivalence*, and `tests/viewer/test_edge_artifacts.py` is where the proof
# lives: one all-leaf program extracts exactly what one program per group
# did.  That is the trick :func:`cadjoint.brep.project.project_batched` is
# built on, and this is the only place it is written out plainly enough to
# check.  Delete them when that test is retired, not before.


def _project_to_seam(fields: list[Any], points: np.ndarray, max_step: float) -> np.ndarray:
    """Newton-project points onto the common zero set of two or more fields.

    Two fields define a seam curve, three a corner point (triple junction).
    Points whose gradients are rank-deficient (tangent or coincident
    surfaces — the system is singular and there is no transversal
    intersection to project onto) are returned unchanged.
    """
    import jax
    import jax.numpy as jnp

    evaluators = [
        jax.vmap(jax.value_and_grad(lambda p, f=field: jnp.asarray(f(p)))) for field in fields
    ]
    count = len(fields)
    start = jnp.asarray(points, dtype=jnp.float32)

    def system(x):
        values, gradients = zip(*(evaluate(x) for evaluate in evaluators))
        jacobian = jnp.stack(gradients, axis=1)
        gram = jnp.einsum("sij,skj->sik", jacobian, jacobian)
        return jnp.stack(values, axis=-1), jacobian, gram

    _, _, gram0 = system(start)
    eigenvalues = jnp.linalg.eigvalsh(gram0)
    trace = jnp.trace(gram0, axis1=-2, axis2=-1)
    transversal = eigenvalues[..., 0] > 1e-2 * trace / count

    x = start
    for _ in range(4):
        residual, jacobian, gram = system(x)
        # Regularize at a float32-meaningful scale; smaller epsilons
        # underflow against unit-gradient Gram entries.
        trace = jnp.trace(gram, axis1=-2, axis2=-1)
        gram = gram + (1e-4 * trace + 1e-12)[..., None, None] * jnp.eye(count, dtype=gram.dtype)
        multipliers = jnp.linalg.solve(gram, residual[..., None])[..., 0]
        step = jnp.einsum("sij,si->sj", jacobian, multipliers)
        length = jnp.linalg.norm(step, axis=-1, keepdims=True)
        step = step * jnp.minimum(1.0, max_step / jnp.maximum(length, 1e-9))
        x = x - step
    x = jnp.where(transversal[:, None], x, start)
    return np.asarray(x, dtype=np.float64)


def _seam_residual(fields: list[Any], points: np.ndarray) -> np.ndarray:
    """How far *points* still are from every field's zero set (the max |f|).

    The acceptance test for a projected seam group: a genuine seam vertex
    lands on every operand's zero set, a near-miss between disjoint
    surfaces keeps a residual of order the gap.
    """
    import jax
    import jax.numpy as jnp

    probes = jnp.asarray(points, dtype=jnp.float32)
    return np.max(
        np.stack(
            [
                np.abs(
                    np.asarray(
                        jax.vmap(lambda p, f=field: jnp.asarray(f(p)))(probes), dtype=np.float64
                    )
                )
                for field in fields
            ]
        ),
        axis=0,
    )


def _project_seam_groups_reference(
    leaves: list[Any],
    groups: list[tuple[np.ndarray, tuple[int, ...]]],
    vertices: np.ndarray,
    max_step: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One :func:`_project_to_seam` program per group — the plain reading.

    This is what :func:`_project_seam_groups` replaces and the oracle it is
    tested against: same inputs, same ``(projected, residual)`` per group,
    one JAX program per group instead of one for all of them.
    """
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for rows, operands in groups:
        fields = [leaves[index] for index in operands]
        projected = _project_to_seam(fields, vertices[rows], max_step)
        results.append((projected, _seam_residual(fields, projected)))
    return results


def _project_seam_groups(
    leaves: list[Any],
    groups: list[tuple[np.ndarray, tuple[int, ...]]],
    vertices: np.ndarray,
    max_step: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Project every seam group at once, in one all-leaf program.

    Retired with the rest of this section; see its banner.

    :func:`_project_to_seam` is exact but pays a fixed cost per call —
    building the ``vmap(value_and_grad)`` evaluators and dispatching four
    Newton iterations op-by-op — that is independent of how many points it
    projects; measured, one point costs as much as three hundred.  A scene
    with context geometry produces a seam group per operand *set*, so the
    per-call cost was what the whole mesh overlay used to be made of
    (measured on the starter: 15 groups, 108 points, 5.6 s).

    So evaluate every world-frame leaf at every seam point instead, in one
    program, and let each point pick its own operands out of the result by
    gather.  Groups shorter than the widest are padded with a repeat of
    their first operand and masked to zero, which is exact rather than
    approximate: a zero Jacobian row makes the padded Gram row and column
    zero, the regularizer puts ``eps`` on its diagonal, and the solve
    returns a zero multiplier there, so the Newton step is the one the
    unpadded system would have taken.  The two places the padding *is*
    visible are handled explicitly — the rank test's smallest eigenvalue
    (a padded row contributes a spurious zero, so those rows are lifted
    above the block's spectrum first) and the ``1e-2 * trace / count``
    threshold, which uses each point's own operand count.

    Returns one ``(projected, residual)`` pair per group, in the order the
    groups were given.
    """
    import jax
    import jax.numpy as jnp

    # Only the operands some group actually meets: a scene's other leaves
    # would be evaluated at every seam point for nothing.
    used = sorted({index for _rows, operands in groups for index in operands})
    slot = {index: position for position, index in enumerate(used)}
    evaluators = [
        jax.vmap(jax.value_and_grad(lambda p, f=leaves[index]: jnp.asarray(f(p)))) for index in used
    ]
    width = max(len(operands) for _rows, operands in groups)
    member_blocks, valid_blocks, point_blocks = [], [], []
    for rows, operands in groups:
        slots = [slot[index] for index in operands]
        padded = slots + [slots[0]] * (width - len(slots))
        member_blocks.append(np.tile(np.asarray(padded, dtype=np.int32), (rows.size, 1)))
        mask = np.zeros(width, dtype=np.float32)
        mask[: len(operands)] = 1.0
        valid_blocks.append(np.tile(mask, (rows.size, 1)))
        point_blocks.append(vertices[rows])
    members = jnp.asarray(np.concatenate(member_blocks))
    valid = jnp.asarray(np.concatenate(valid_blocks))
    counts = jnp.sum(valid, axis=1)
    start = jnp.asarray(np.concatenate(point_blocks), dtype=jnp.float32)
    picker = jnp.arange(start.shape[0])[:, None]

    def system(x):
        values, gradients = zip(*(evaluate(x) for evaluate in evaluators))
        jacobian = jnp.stack(gradients, axis=1)[picker, members] * valid[..., None]
        gram = jnp.einsum("sij,skj->sik", jacobian, jacobian)
        return jnp.stack(values, axis=-1)[picker, members] * valid, jacobian, gram

    identity = jnp.eye(width, dtype=start.dtype)
    _, _, gram0 = system(start)
    trace = jnp.trace(gram0, axis1=-2, axis2=-1)
    # A padded row is exactly zero, so it would contribute the smallest
    # eigenvalue and fail every point with a short operand set.  Lift the
    # padded diagonal above the real block's spectrum (bounded by its
    # trace) so the minimum is the real block's own.
    lifted = gram0 + ((trace[:, None] + 1.0) * (1.0 - valid))[..., None] * identity
    transversal = jnp.linalg.eigvalsh(lifted)[..., 0] > 1e-2 * trace / counts

    x = start
    for _ in range(4):
        residual, jacobian, gram = system(x)
        # Regularize at a float32-meaningful scale; smaller epsilons
        # underflow against unit-gradient Gram entries.
        trace = jnp.trace(gram, axis1=-2, axis2=-1)
        gram = gram + (1e-4 * trace + 1e-12)[..., None, None] * identity
        multipliers = jnp.linalg.solve(gram, residual[..., None])[..., 0]
        step = jnp.einsum("sij,si->sj", jacobian, multipliers)
        length = jnp.linalg.norm(step, axis=-1, keepdims=True)
        step = step * jnp.minimum(1.0, max_step / jnp.maximum(length, 1e-9))
        x = x - step
    x = jnp.where(transversal[:, None], x, start)

    values, _gradients = zip(*(evaluate(x) for evaluate in evaluators))
    # Padded slots are masked to zero and |f| >= 0, so they never win the max.
    residual = jnp.max(jnp.abs(jnp.stack(values, axis=-1)[picker, members]) * valid, axis=1)
    projected = np.asarray(x, dtype=np.float64)
    residuals = np.asarray(residual, dtype=np.float64)

    results: list[tuple[np.ndarray, np.ndarray]] = []
    offset = 0
    for rows, _operands in groups:
        stop = offset + rows.size
        results.append((projected[offset:stop], residuals[offset:stop]))
        offset = stop
    return results


def _extract_graph(scene: Any) -> tuple[BRep, np.ndarray]:
    """Derive the B-rep on the overlay's grid: one dual-contouring pass.

    The extraction owns the pass — :func:`~cadjoint.brep.extract_brep` runs
    the mesher itself and the graph is built over that one mesh, so the
    overlay's two layers and every projected point come from a single
    lattice sweep.

    Surface fitting is off: the overlay wants the graph's topology and its
    edge curves, never a face's closed form, and fitting is a projection
    program plus a gradient program per face.

    Args:
        scene: Root SDF node.

    Returns:
        ``(brep, spacing)`` — the derived B-rep and the grid spacing.
    """
    import warnings

    from cadjoint.brep import extract_brep
    from cadjoint.meshing import GridSpec

    grid = GridSpec.from_bounds(_MESH_EDGE_BOUNDS, _MESH_EDGE_SIZE, _MESH_EDGE_RESOLUTION)
    with warnings.catch_warnings():
        # The overlay's grid is a fixed view volume the scene is *expected*
        # to leave — a ground plane spans it edge to edge — so the mesher's
        # open-boundary warning would fire on every compile and say nothing
        # a viewer user can act on.  Clipping to the view is the point.
        warnings.filterwarnings("ignore", message="The isosurface crosses the extraction boundary")
        brep = extract_brep(scene, grid, steps=_PROJECTION_STEPS, fit_surfaces=False)
    return brep, np.asarray(grid.spacing, dtype=np.float64)


def _design_patches(brep: BRep) -> np.ndarray | None:
    """Which global patches belong to the design subtree, or ``None``.

    The rule and its motivation are :func:`_design_leaves`'; this only lifts
    it from leaves to the patch table the graph's edges name.

    Args:
        brep: The derived B-rep.

    Returns:
        A boolean mask over :attr:`~cadjoint.brep.BRep.patches`, or ``None``
            when no leaf carries a construction mirror.
    """
    design = _design_leaves(list(brep.decomposition.leaves))
    if design is None:
        return None
    return np.isin(np.asarray([patch.leaf for patch in brep.patches], dtype=np.int64), design)


def _edge_polyline(brep: BRep, edge: BRepEdge) -> np.ndarray:
    """An edge's solved points, extended to its triple-point corners.

    The chain's seeds are mesh-edge *midpoints*, so the polyline stops half
    a cell short of each end.  Where the chain ended on a
    :class:`~cadjoint.brep.BRepVertex` that corner is the exact endpoint, and
    appending it both completes the curve and makes the edges meeting there
    share one endpoint exactly — which is what turns the sharp layer into
    connected chains rather than a set of near-touching arcs.

    Args:
        brep: The derived B-rep.
        edge: The edge to read.

    Returns:
        The polyline, shaped ``(k, 3)``; for a closed edge, without a
            repeated first point.
    """
    if edge.closed:
        return edge.polyline
    start, stop = edge.vertices
    pieces = [edge.polyline]
    if start >= 0:
        pieces.insert(0, brep.vertices[start].point[None, :])
    if stop >= 0:
        pieces.append(brep.vertices[stop].point[None, :])
    return np.concatenate(pieces) if len(pieces) > 1 else edge.polyline


def _resample(polyline: np.ndarray, closed: bool, spacing: float) -> np.ndarray | None:
    """Uniform arc-length samples along a polyline.

    The count is chosen so the samples divide the curve *evenly*: a closed
    edge is cut into equal arcs with no short closing piece, which for a rim
    is uniform sampling in angle.

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
    count = max(int(round(total / spacing)), 1)
    stations = np.linspace(0.0, total, count + 1)
    if closed:
        stations = stations[:-1]
    return np.stack([np.interp(stations, knots, points[:, axis]) for axis in range(3)], axis=1)


def _prune_debris(
    entries: list[tuple[BRepEdge, np.ndarray]], limit: float
) -> list[tuple[BRepEdge, np.ndarray]]:
    """Drop link complexes that are open and shorter than *limit*.

    Where two solids graze — the starter's ring skimming the roof — the
    graph finds short genuine edges around the contact: a curve that exists,
    lasts a couple of cells, and ends nowhere.  Drawn, they read as ticks
    hovering beside the geometry rather than as edges, which is the artifact
    battery's ``debris`` metric.  A real feature curve is long, or closed, or
    anchored into a larger complex through its corners, so the test is on the
    connected complex and not on the single edge: components are joined at
    shared :class:`~cadjoint.brep.BRepVertex` corners, a component with a
    free end is open, and an open component under a few cells long goes.

    Args:
        entries: The drawable edges and their resampled points.
        limit: Shortest total arc length an open complex may have.

    Returns:
        The surviving entries, in the order given.
    """
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def node_of(edge_index: int, corner: int, side: int) -> int:
        # A corner is shared by every edge meeting there; a free end (-1) is
        # this edge's alone, so it gets a node no other edge can reach.
        return corner if corner >= 0 else -1 - (2 * edge_index + side)

    ends: list[tuple[int, int]] = []
    for position, (edge, _samples) in enumerate(entries):
        corners = (-1, -1) if edge.closed else edge.vertices
        pair = (node_of(position, corners[0], 0), node_of(position, corners[1], 1))
        ends.append(pair)
        for node in pair:
            parent.setdefault(node, node)
    degree: dict[int, int] = {}
    for position, (edge, _samples) in enumerate(entries):
        left, right = ends[position]
        if find(left) != find(right):
            parent[find(left)] = find(right)
        if not edge.closed:
            for node in ends[position]:
                degree[node] = degree.get(node, 0) + 1

    lengths: dict[int, float] = {}
    open_roots: set[int] = set()
    for position, (edge, samples) in enumerate(entries):
        root = find(ends[position][0])
        span = np.linalg.norm(np.diff(samples, axis=0), axis=1).sum()
        lengths[root] = lengths.get(root, 0.0) + float(span)
        if edge.closed:
            continue
        if any(degree.get(node, 0) < 2 for node in ends[position]):
            open_roots.add(root)
    return [
        entry
        for position, entry in enumerate(entries)
        if find(ends[position][0]) not in open_roots or lengths[find(ends[position][0])] >= limit
    ]


def _pinned_corners(entries: list[tuple[BRepEdge, np.ndarray]]) -> np.ndarray:
    """Rows of the stacked samples that are a triple point and must not move.

    :func:`_edge_polyline` puts each open edge's corners first and last, and
    :func:`_resample` keeps both stations, so a corner is always the first
    or last row of its edge's block.

    Args:
        entries: The drawable edges and their resampled points.

    Returns:
        A boolean mask over the concatenated sample rows.
    """
    mask = np.concatenate([np.zeros(samples.shape[0], dtype=bool) for _edge, samples in entries])
    cursor = 0
    for edge, samples in entries:
        if not edge.closed:
            mask[cursor] = edge.vertices[0] >= 0
            mask[cursor + samples.shape[0] - 1] = edge.vertices[1] >= 0
        cursor += samples.shape[0]
    return mask


def _sharp_chords(brep: BRep, spacing: np.ndarray) -> np.ndarray:
    """The design's analytic edges, resampled and re-projected, as chords.

    Every sample of every edge is projected in *one* batched call onto its
    own patch pair: the cost of an eager JAX program is per call and not per
    point, so a call per edge would be the whole cost of the overlay (the
    lesson :func:`cadjoint.brep.project.project_batched` documents).

    Args:
        brep: The derived B-rep.
        spacing: The grid spacing, shaped ``(3,)``.

    Returns:
        Chord endpoints, shaped ``(n, 2, 3)``.
    """
    from cadjoint.brep.project import project_batched

    design = _design_patches(brep)
    limit = _EDGE_RESIDUAL_FRACTION * float(spacing.max())
    entries: list[tuple[BRepEdge, np.ndarray]] = []
    for edge in brep.edges:
        if not edge.analytic or edge.residual > limit:
            continue
        # Both neighbours must be the design's: a curve that is half on
        # scenery is not a crease the user is authoring, and drawing it
        # would put sharp links on the board the heat sink sits on.
        if design is not None and not (design[edge.patches[0]] and design[edge.patches[1]]):
            continue
        samples = _resample(
            _edge_polyline(brep, edge),
            edge.closed,
            _SHARP_SAMPLE_FRACTION * float(spacing.min()),
        )
        if samples is None or samples.shape[0] < 2:
            continue
        entries.append((edge, samples))
    entries = _prune_debris(entries, _DEBRIS_CELLS * float(spacing.max()))
    if not entries:
        return np.empty((0, 2, 3), dtype=np.float64)

    seeds = np.concatenate([samples for _edge, samples in entries])
    solved = project_batched(
        [patch.field for patch in brep.patches],
        np.concatenate(
            [
                np.tile(np.asarray(edge.patches, dtype=np.int32), (samples.shape[0], 1))
                for edge, samples in entries
            ]
        ),
        seeds,
        max_step=0.5 * float(np.linalg.norm(spacing)),
        steps=_PROJECTION_STEPS,
    )
    # A corner is already the *three*-patch solution, so projecting it onto
    # the two patches of one of the edges meeting there could only drag it
    # off the third face — by a few thousandths, enough that the three edges
    # would no longer share one endpoint and the layer would stop being
    # connected chains.  Pin them instead.
    solved = np.where(_pinned_corners(entries)[:, None], seeds, solved)
    chords: list[np.ndarray] = []
    cursor = 0
    for edge, samples in entries:
        stop = cursor + samples.shape[0]
        points = solved[cursor:stop]
        cursor = stop
        following = np.roll(points, -1, axis=0) if edge.closed else points[1:]
        leading = points if edge.closed else points[:-1]
        chords.append(np.stack([leading, following], axis=1))
    return np.concatenate(chords)


def _mesh_edge_payload(scene: Any) -> dict[str, Any] | None:
    """Extract the overlay's two layers from one derived B-rep.

    The wire layer is the dual-contour mesh's native quad edges
    (triangulation diagonals carry no surface information), drawn on the
    graph's re-solved vertex positions so a vertex on a CSG seam sits on the
    seam rather than up to a cell and a half off it.  The sharp layer is the
    graph's analytic edges: each is the exact intersection curve of two
    patch zero sets, resampled at half a cell and re-projected.

    Optional viewer data: any failure prints a note (captured into the
    compile output) and returns ``None`` rather than failing the compile.
    """
    try:
        brep, spacing = _extract_graph(scene)
        quads = np.asarray(brep.mesh.quads, dtype=np.int64)
        quad_edges = np.concatenate(
            [quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]
        )
        wire_edges = np.unique(np.sort(quad_edges, axis=1), axis=0)

        def segments(pairs: np.ndarray) -> list[list[list[float]]]:
            return [
                [[round(float(value), 3) for value in point] for point in pair] for pair in pairs
            ]

        return {
            "wire": segments(brep.points[wire_edges]),
            "sharp": segments(_sharp_chords(brep, spacing)),
            "resolution": _MESH_EDGE_RESOLUTION,
        }
    except Exception as error:  # noqa: BLE001 - viewer extra must never break compiles
        print(f"mesh edge view unavailable: {error}")
        return None
