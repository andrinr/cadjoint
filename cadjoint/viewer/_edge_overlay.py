"""Feature-edge overlay for the viewer's mesh view.

Everything behind ``mode="mesh"``: dual-contour the scene on the viewer's
own grid and split the result into the two overlay layers the frontend
draws — the mesh's native quad edges ("wire") and the chords of its true
feature curves ("sharp").

**Where the sharp layer comes from depends on the tier.**  A feature curve
is where two patch zero sets meet, and the derived B-rep finds exactly
that: one dual-contouring pass discovers which patch owns which region of
the surface, the boundary chain between two regions is an edge, and every
point on it is placed by the projection kernel rather than by the lattice.
That is the private tier's, reached here through the ``feature_edges``
plugin kind (:class:`cadjoint.plugins.contracts.FeatureEdges`), and it is
what :func:`_graph_layers` asks for.

Without it — public cadjoint alone — :func:`_lattice_layers` reads the
curves off the lattice instead: feature cells are classified from normal
spreads and exact ``min``/``max`` CSG seams, the seam vertices are
Newton-projected onto the common zero set of the operands meeting there
(:func:`_project_seam_groups`, one program for every group at once), and
the surviving cell links are the chords.  It is the same layer the overlay
shipped before the graph existed, and it is honest about what it is: the
staircase, the near-parallel rails and the orphan ticks are fought with
tangent, chain-degree and debris tests rather than solved.  A rim is a
polygon of cell links, not a circle.  :mod:`cadjoint.tier` is where the
difference is reported.

The wire layer is the dual-contour quad edges either way, drawn on the
re-solved vertex positions of whichever pass produced them so the two
layers agree everywhere.

Only the extraction lives here.  The worker mode that calls it (executing
the user's program first) stays in :mod:`cadjoint.viewer._compile_worker`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadjoint.enums import PluginKind
from cadjoint.meshing.patch_fields import world_frame_leaves

#: The overlay's name for the leaf split it shares with the mesher and the
#: graph: the design-subtree rule below is defined over exactly these
#: leaves, and the derived B-rep's decomposition carries the same list in
#: the same order.
_world_frame_leaves = world_frame_leaves


# Mesh-edge view settings.  The grid matches the raymarcher's view volume.
# Detection stays dense rather than Lipschitz-pruned: user-written fields
# can exceed any assumed gradient bound, and a hole in the viewer is worse
# than the ~100 ms this costs.
_MESH_EDGE_BOUNDS = (-3.0, -3.0, -3.0)
_MESH_EDGE_SIZE = (6.0, 6.0, 6.0)
_MESH_EDGE_RESOLUTION = 64

# Newton iterations per projection asked of the ``feature_edges`` component.
# Its own default is eight, which converges from a cold seed anywhere in its
# cell; every seed the overlay produces is a mesh-edge midpoint or a chord of
# an already-solved polyline, so it starts within a fraction of a cell and
# four iterations reach the same float32 floor for half the cost.
_PROJECTION_STEPS = 4


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
# did.  That is the trick the private tier's batched projection kernel is
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


def _lattice_layers(scene: Any, grid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The overlay's two layers read off the lattice — the public fallback.

    What the overlay drew before the derived B-rep existed, and what it
    draws again when the ``feature_edges`` kind is unfilled.  The wire layer
    is the dual-contour mesh's native quad edges (triangulation diagonals
    carry no surface information).  The sharp layer is *not* mesh edges: a
    feature curve crosses grid cells diagonally, so its cells are usually
    not mesh-adjacent and mesh edges would trace a staircase around it.
    Instead, feature cells — normal-spread creases and corners, plus exact
    ``min``/``max`` CSG seams — are linked to their lattice neighbours, and
    because feature-aware placement puts each of their vertices exactly on
    the feature curve, those links are chords of the true curve.

    Everything downstream of that is the cost of the lattice being the only
    witness: the on-surface subgradient re-check, identity grouping so two
    curves within a cell of each other cannot cross-link, the tangent
    alignment test, greedy chain building to hold every vertex at degree
    two, and debris pruning.  The graph path needs none of it — which is
    the point of the seam.

    Args:
        scene: Root SDF node.
        grid: The :class:`~cadjoint.meshing.GridSpec` to extract on.

    Returns:
        ``(vertices, quad_edges, sharp_chords)`` — the re-solved dual
            vertices, the quad edge pairs indexing them, and the sharp
            chords as ``(n, 2, 3)`` point pairs.
    """
    import jax
    import jax.numpy as jnp

    from cadjoint.meshing import (
        CORNER,
        FACE,
        classify_feature_cells,
        dual_faces,
        edge_hermite_data,
        feature_cell_links,
        find_crossing_edges,
        manifold_cell_incidence,
        sample_grid,
        sharp_qef_vertices,
    )

    sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
    values = sample_grid(sdf, grid)
    edges = find_crossing_edges(values)
    if edges.count == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, np.empty((0, 2), dtype=np.int64), np.empty((0, 2, 3), dtype=np.float64)
    # Manifold incidence: one row per inside-corner component, so cells
    # crossed by two surface sheets keep the sheets on separate vertices.
    incidence = manifold_cell_incidence(edges, grid, values < 0.0)
    hermite = edge_hermite_data(sdf, grid, edges)
    vertices = sharp_qef_vertices(hermite, incidence, grid)
    quads, _triangles, _skipped = dual_faces(edges, incidence, grid, vertices)
    quad_edge_list = np.concatenate(
        [quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]
    )

    features = classify_feature_cells(hermite, incidence)
    geometric_mask = features.classes != FACE
    junctions = features.classes == CORNER

    # On-surface subgradient verification.  Root refinement converges
    # every crossing onto the surface to float32 precision, and several
    # fields (the polygon among them) have degenerate subgradients
    # exactly on their boundary: the recorded gradient can come out
    # zero or exactly negated.  A cell whose crossings hit that locus
    # classifies as a maximally sharp crease in the middle of a flat
    # face and sheds orphan tick links.  True normals live a hair off
    # the surface, so re-evaluate the gradient at +-h/4 along each
    # incident edge and demote feature cells whose verified normal
    # spread is face-like.  Genuine creases keep their fan: each
    # crossing edge pierces one smooth facet, and different edges
    # pierce different facets.
    feature_rows = np.flatnonzero(geometric_mask)
    if feature_rows.size:
        cell_edges = incidence.edge_ids[feature_rows]
        used = np.unique(cell_edges[cell_edges >= 0])
        axes = np.asarray(edges.axis)[used]
        offsets = (0.25 * np.asarray(grid.spacing, dtype=np.float64)[axes])[:, None] * np.eye(3)[
            axes
        ]
        base = np.asarray(hermite.points, dtype=np.float64)[used]
        probes = jnp.asarray(np.concatenate([base - offsets, base + offsets]), jnp.float32)
        gradients = np.asarray(jax.vmap(jax.grad(sdf))(probes), dtype=np.float64)
        scale = np.linalg.norm(gradients, axis=1, keepdims=True)
        probe_normals = np.where(scale > 1e-9, gradients / np.maximum(scale, 1e-12), 0.0)
        lookup = np.zeros(int(used.max()) + 1, dtype=np.int64)
        lookup[used] = np.arange(used.size)
        slots = lookup[np.maximum(cell_edges, 0)]
        samples = np.concatenate([probe_normals[slots], probe_normals[slots + used.size]], axis=1)
        sample_valid = np.concatenate([cell_edges >= 0] * 2, axis=1) & (
            np.linalg.norm(samples, axis=-1) > 0.5
        )
        sample_counts = np.maximum(sample_valid.sum(axis=1), 1)
        sample_mean = (samples * sample_valid[..., None]).sum(axis=1) / sample_counts[:, None]
        centered = (samples - sample_mean[:, None, :]) * sample_valid[..., None]
        spread = np.einsum("cei,cej->cij", centered, centered) / sample_counts[:, None, None]
        singular = np.sqrt(np.clip(np.linalg.eigvalsh(spread), 0.0, None))[:, ::-1]
        geometric_mask[feature_rows[singular[:, 0] <= 0.25]] = False
        junctions[feature_rows[singular[:, 1] <= 0.25]] = False

    # Feature curves must never cross-link: two distinct curves (a crease
    # and a nearby seam, or seams of different operand pairs) can run
    # within one cell of each other, and linking across them weaves an
    # X-lattice between the curves.  Cells are therefore grouped by
    # feature identity — the owning CSG operand for geometric creases,
    # the operand pair for seams — and linked only within their group.
    leaves = _world_frame_leaves(scene)
    groups: list[np.ndarray] = []
    seam_groups: list[tuple[np.ndarray, tuple[int, int]]] = []
    if len(leaves) >= 2 and quads.shape[0] > 0:
        points = jnp.asarray(vertices, dtype=jnp.float32)
        magnitudes = np.stack(
            [
                np.abs(
                    np.asarray(
                        jax.vmap(lambda p, field=leaf: jnp.asarray(field(p)))(points),
                        dtype=np.float64,
                    )
                )
                for leaf in leaves
            ]
        )
        owners = np.argmin(magnitudes, axis=0)
        # The sharp layer is the DESIGN's feature curves: a dual vertex
        # belongs to the design when the operand owning it does (see
        # _design_leaves for the rule and why it needs no new flag).
        # Ownership is decided against the whole scene either way, so a
        # vertex on a context primitive is recognised as context rather
        # than mis-assigned to whichever design operand is nearest.
        design = _design_leaves(leaves)
        if design is None:
            designed = np.ones(owners.shape, dtype=bool)
        else:
            designed = np.isin(owners, design)
            geometric_mask &= designed
            junctions &= designed
        mismatched = quad_edge_list[owners[quad_edge_list[:, 0]] != owners[quad_edge_list[:, 1]]]
        seam_rows = np.unique(mismatched)
        structural = np.zeros_like(geometric_mask)
        # Seam vertices are Newton-projected onto the exact common zero
        # set of every operand meeting there — a curve for two, a triple
        # point for three: min/max structure makes seams a solvable
        # system, not a mesh approximation.  Grouping by the full
        # operand set projects each vertex exactly once.
        #
        # Ownership flips alone are not proof of a seam: they also fire
        # along the near-miss band between DISJOINT surfaces (staggered
        # thin slabs, a ring hovering just above a roof), where there is
        # no intersection curve at all.  The operand fields are signed
        # distances, so the projection doubles as the test — a genuine
        # seam vertex converges onto every operand's zero set (residual
        # far below a tenth of a cell), while a near-miss row's residual
        # stays at half the surface gap or worse, even when a
        # sub-resolution gap merged both sheets into one dual vertex
        # sitting between them.  Rows that fail keep their original
        # vertex and return to their owner's geometric group instead of
        # being dragged onto a phantom curve and suppressed as seams.
        if seam_rows.size:
            incident: dict[int, set[int]] = {}
            for u, v in mismatched:
                for row in (int(u), int(v)):
                    incident.setdefault(row, set()).update((int(owners[u]), int(owners[v])))
            by_operands: dict[frozenset[int], list[int]] = {}
            for row, operand_set in incident.items():
                by_operands.setdefault(frozenset(operand_set), []).append(row)
            max_step = 2.0 * float(max(grid.spacing))  # ~2 cells
            pending = [
                (np.asarray(sorted(rows), dtype=np.int64), tuple(sorted(operand_set)))
                for operand_set, rows in by_operands.items()
            ]
            # One program for every group at once: the projection's cost
            # is per call, not per point (see _project_seam_groups).  The
            # groups partition the seam rows, so reading every group's
            # start position before any of them is written is the same
            # computation as projecting them one at a time.
            for (selected, operand_set), (projected, residual) in zip(
                pending, _project_seam_groups(leaves, pending, vertices, max_step)
            ):
                genuine = residual < 0.1 * max(grid.spacing)
                selected = selected[genuine]
                if selected.size == 0:
                    continue
                # Every accepted group moves its vertices, scenery
                # included: the wire layer draws these same vertices, and
                # a seam it did not snap wobbles off the curve by up to a
                # cell and a half.  Only the SHARP layer is the design's
                # (see _design_leaves), so the design filter lands here,
                # on which rows may grow links, and not on the projection.
                vertices[selected] = projected[genuine]
                structural[selected] = True
                selected = selected[designed[selected]]
                if selected.size == 0:
                    continue
                group_mask = np.zeros_like(geometric_mask)
                group_mask[selected] = True
                groups.append(group_mask)
                if len(operand_set) == 2:
                    seam_groups.append((selected, operand_set))
        geometric_only = geometric_mask & ~structural
        for owner in np.unique(owners[geometric_only]):
            groups.append(geometric_only & (owners == owner))
    else:
        groups.append(geometric_mask)

    link_groups = [
        # The junction-shortcut pruning normally done by
        # ``junction_mask`` happens below instead, where tangents are
        # available to tell a genuine corner shortcut from a straight
        # rail that merely passes beside someone else's corner.
        feature_cell_links(group, incidence, grid)
        for group in groups
    ]
    links = np.concatenate(link_groups) if link_groups else np.empty((0, 2), dtype=np.int32)

    # Two parallel creases of one thin face share an operand, so identity
    # grouping cannot separate them; rail-to-rail cross-links would weave
    # an X-band between them.  A genuine chord runs along its curve: the
    # local tangent is the least-varying direction of the cell's incident
    # normals (they fan across a crease, stay constant along it), and a
    # link must align with the tangent at both non-junction endpoints.
    # The threshold must be well above 1/2: a diagonal hop to a rail up
    # to two cells away still has slightly more than half its length
    # along the rail (alignment h / sqrt(h² + s²) ≈ 0.5 for rail
    # separation s ≈ 1.7 h), while genuine chords sit on the curve
    # itself and align above 0.9 even where the curve bends.
    if links.shape[0]:
        feature_rows = np.flatnonzero(np.any(groups, axis=0))
        unit = np.asarray(hermite.unit_normals(), dtype=np.float64)
        gathered = unit[np.maximum(incidence.edge_ids[feature_rows], 0)]
        valid = (incidence.edge_ids[feature_rows] >= 0) & (np.sum(gathered**2, axis=-1) > 0.25)
        counts = np.maximum(valid.sum(axis=1), 1)
        mean = (gathered * valid[..., None]).sum(axis=1) / counts[:, None]
        centered = (gathered - mean[:, None, :]) * valid[..., None]
        covariance = np.einsum("cei,cej->cij", centered, centered)
        _, eigenvectors = np.linalg.eigh(covariance)
        # The centered covariance of a straight crease is rank one, so
        # its nullspace is a plane; the tangent is instead perpendicular
        # to both the mean normal and the principal spread direction.
        raw = np.cross(mean, eigenvectors[:, :, 2])
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        tangents = np.full((incidence.count, 3), np.nan)
        tangents[feature_rows] = np.where(norms > 1e-6, raw / np.maximum(norms, 1e-12), np.nan)
        suppressed = np.zeros(incidence.count, dtype=bool)

        # Seam cells get the EXACT tangent cross(grad_a, grad_b) instead
        # of the covariance estimate, which degrades where two seam
        # curves of the same operand pair converge (near-tangent
        # contact).  Tiny cross products mean tangential contact — no
        # reliable direction, so leave those undefined (short links
        # only).  Seam cells also lose the junction exemption: a seam
        # is a curve, not a corner fan.
        for seam_rows, (index_a, index_b) in seam_groups:
            points = jnp.asarray(vertices[seam_rows], dtype=jnp.float32)
            grad_a = np.asarray(
                jax.vmap(jax.grad(lambda p, f=leaves[index_a]: jnp.asarray(f(p))))(points),
                dtype=np.float64,
            )
            grad_b = np.asarray(
                jax.vmap(jax.grad(lambda p, f=leaves[index_b]: jnp.asarray(f(p))))(points),
                dtype=np.float64,
            )
            cross = np.cross(grad_a, grad_b)
            cross_norms = np.linalg.norm(cross, axis=1, keepdims=True)
            scale = np.linalg.norm(grad_a, axis=1, keepdims=True) * np.linalg.norm(
                grad_b, axis=1, keepdims=True
            )
            reliable = cross_norms > 0.1 * np.maximum(scale, 1e-12)
            tangents[seam_rows] = np.where(reliable, cross / np.maximum(cross_norms, 1e-12), np.nan)
            junctions = junctions.copy()
            junctions[seam_rows] = False
            # Tangential contact has no seam CURVE at all — the operands
            # touch over a region.  Unlike sub-resolution creases (where
            # short rungs honestly show a thin band), drawing anything
            # there is noise: suppress these cells' links completely.
            suppressed[seam_rows[~reliable[:, 0]]] = True

        # Junction shortcuts: a link between two non-junction cells
        # that share a junction neighbor usually cuts a corner — the
        # true feature path runs through the junction, and the direct
        # chord would double-draw it.  A shared junction neighbor is
        # not proof, though: where another feature complex's corner
        # sits one cell beside a straight rail (a slab rim passing
        # under a second solid's corner), the rail's own links must
        # survive.  Drop the direct link only when the detour through
        # the junction is actually drawable — each replacement leg
        # aligned with the tangent at its non-junction end.  (An
        # undefined tangent cannot veto, matching the unconditional
        # drop this replaces.)
        adjacency: dict[int, set[int]] = {}
        for a, b in links:
            adjacency.setdefault(int(a), set()).add(int(b))
            adjacency.setdefault(int(b), set()).add(int(a))

        def drawable(row: int, junction_row: int) -> bool:
            tangent = tangents[row]
            if not np.isfinite(tangent).all():
                return True
            chord = vertices[junction_row] - vertices[row]
            length = float(np.linalg.norm(chord))
            return length < 1e-9 or abs(float(chord @ tangent)) / length > 0.9

        shortcut_keep = np.ones(links.shape[0], dtype=bool)
        for index, (a, b) in enumerate(links):
            a, b = int(a), int(b)
            if junctions[a] or junctions[b]:
                continue
            for common in adjacency[a] & adjacency[b]:
                if junctions[common] and drawable(a, common) and drawable(b, common):
                    shortcut_keep[index] = False
                    break
        links = links[shortcut_keep]

        directions = vertices[links[:, 1]] - vertices[links[:, 0]]
        lengths = np.maximum(np.linalg.norm(directions, axis=1), 1e-12)
        keep = np.ones(links.shape[0], dtype=bool)
        confirmed = np.zeros(links.shape[0], dtype=bool)
        for column in (0, 1):
            rows = links[:, column]
            tangent = tangents[rows]
            defined = np.isfinite(tangent).all(axis=1) & ~junctions[rows]
            alignment = np.abs(np.einsum("li,li->l", directions, tangent)) / lengths
            # Genuine on-curve chords align above ~0.98 even where the
            # curve turns sharply per cell; rail-to-rail diagonals on
            # shallow double seams reach ~0.8, so the cut sits at 0.9.
            keep &= ~defined | (alignment > 0.9)
            confirmed |= defined & (alignment > 0.9)
        # Sub-resolution strips classify whole rows of cells as corners
        # or leave tangents undefined, bypassing the alignment test; a
        # link no endpoint positively confirmed must stay within one
        # cell, so rail-to-rail diagonals (~1.4 cells) cannot weave an
        # X-band through the exemptions while diagonal chords of real
        # curves — whose endpoints do confirm — keep their full reach.
        keep &= confirmed | (lengths <= 1.2 * max(grid.spacing))
        keep &= ~(suppressed[links[:, 0]] | suppressed[links[:, 1]])
        links = links[keep]

        # Chain building: a curve visits each cell once, so every
        # non-junction vertex accepts at most one link per tangent side
        # (two links total where its tangent is undefined); junction
        # (corner) cells are exempt, since several chains legitimately
        # meet there.  Links are considered best-first — shortest,
        # discounted by endpoint alignment — and accepted only while
        # every non-junction endpoint has a free slot.  Greedy
        # acceptance keeps degree <= 2, so X-lattices stay structurally
        # impossible however thin a double rail gets, and it cannot dash
        # real edges the way independent per-slot competition with
        # mutual veto could: on a straight chain the direct link and its
        # cell-skipping chord tie at alignment 1.0, float noise then
        # picked slot winners the endpoints disagreed on, and BOTH links
        # died.  Here a link is only ever rejected because an accepted
        # link already continues the same chain through that slot.
        if links.shape[0]:
            directions = vertices[links[:, 1]] - vertices[links[:, 0]]
            lengths = np.maximum(np.linalg.norm(directions, axis=1), 1e-12)
            cost = lengths.copy()
            for column in (0, 1):
                rows = links[:, column]
                tangent = np.nan_to_num(tangents[rows])
                defined = np.isfinite(tangents[rows]).all(axis=1) & ~junctions[rows]
                cosine = np.abs(np.einsum("li,li->l", directions, tangent)) / lengths
                cost = np.where(defined, cost * (2.0 - cosine), cost)
            taken: set[tuple[int, int]] = set()
            budget: dict[int, int] = {}
            chain_keep = np.zeros(links.shape[0], dtype=bool)
            for index in np.argsort(cost, kind="stable"):
                directional: list[tuple[int, int]] = []
                loose: list[int] = []
                for column in (0, 1):
                    row = int(links[index, column])
                    if junctions[row]:
                        continue
                    tangent = tangents[row]
                    if np.isfinite(tangent).all():
                        outward = directions[index] * (1.0 if column == 0 else -1.0)
                        directional.append((row, 0 if float(outward @ tangent) >= 0.0 else 1))
                    else:
                        loose.append(row)
                if any(key in taken for key in directional) or any(
                    budget.get(row, 0) >= 2 for row in loose
                ):
                    continue
                chain_keep[index] = True
                taken.update(directional)
                for row in loose:
                    budget[row] = budget.get(row, 0) + 1
            links = links[chain_keep]

        # Debris pruning: a component of the link graph that is open (it
        # has a vertex of degree one) yet spans under about three cells
        # is an orphan fragment, not a curve — real chains are long,
        # closed, or anchored in a larger junction network, and honest
        # sub-resolution bands are long strips of short rungs.  Such
        # fragments survive the per-link tests near tangential contact,
        # where a handful of borderline cells sits between the
        # suppressed zone and clean geometry, and render as ticks.
        if links.shape[0]:
            parent = np.arange(incidence.count, dtype=np.int64)

            def find(row: int) -> int:
                while parent[row] != row:
                    parent[row] = parent[parent[row]]
                    row = int(parent[row])
                return row

            lengths = np.linalg.norm(vertices[links[:, 1]] - vertices[links[:, 0]], axis=1)
            for a, b in links:
                root_a, root_b = find(int(a)), find(int(b))
                if root_a != root_b:
                    parent[root_a] = root_b
            degree = np.bincount(links.ravel(), minlength=incidence.count)
            component_length: dict[int, float] = {}
            component_open: set[int] = set()
            for (a, _b), length in zip(links, lengths):
                root = find(int(a))
                component_length[root] = component_length.get(root, 0.0) + float(length)
            for row in np.unique(links.ravel()):
                if degree[row] == 1:
                    component_open.add(find(int(row)))
            limit = 3.25 * max(grid.spacing)
            keep_component = np.array(
                [
                    find(int(a)) not in component_open or component_length[find(int(a))] >= limit
                    for a, _b in links
                ],
                dtype=bool,
            )
            links = links[keep_component]

    return vertices, quad_edge_list, vertices[links]


def _overlay_grid() -> Any:
    """The viewer's own extraction grid — the raymarcher's view volume."""
    from cadjoint.meshing import GridSpec

    return GridSpec.from_bounds(_MESH_EDGE_BOUNDS, _MESH_EDGE_SIZE, _MESH_EDGE_RESOLUTION)


def _graph_layers(
    component: Any, scene: Any, grid: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two layers from the ``feature_edges`` kind — the private tier's.

    One extraction serves both layers: the component runs its own
    dual-contouring pass and hands back the mesh it built the graph over in
    ``stats["mesh"]``, so the wire layer draws that pass's re-solved
    vertices rather than paying for a second sweep.  A component that
    reports no mesh is not an error — the caller runs the public pass for
    the wire layer instead.

    Args:
        component: The object filling the ``feature_edges`` kind.
        scene: Root SDF node.
        grid: The overlay's grid.

    Returns:
        ``(vertices, quad_edges, sharp_chords)`` — see
            :func:`_lattice_layers`.
    """
    import warnings

    with warnings.catch_warnings():
        # The overlay's grid is a fixed view volume the scene is *expected*
        # to leave — a ground plane spans it edge to edge — so the mesher's
        # open-boundary warning would fire on every compile and say nothing
        # a viewer user can act on.  Clipping to the view is the point.
        warnings.filterwarnings("ignore", message="The isosurface crosses the extraction boundary")
        edges = component.feature_edges(
            scene,
            grid,
            design_leaves=_design_leaves(_world_frame_leaves(scene)),
            steps=_PROJECTION_STEPS,
        )
    mesh = edges.stats.get("mesh")
    if mesh is None:
        vertices, quad_edges, _sharp = _lattice_layers(scene, grid)
    else:
        points, quads = mesh
        vertices = np.asarray(points, dtype=np.float64)
        quads = np.asarray(quads, dtype=np.int64)
        quad_edges = np.concatenate(
            [quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]
        )
    return vertices, quad_edges, edges.chords()


def _mesh_edge_payload(scene: Any) -> dict[str, Any] | None:
    """Extract the overlay's two layers, from the graph where it is installed.

    Asks for the ``feature_edges`` kind and falls back to
    :func:`_lattice_layers` when nothing fills it, so the overlay is always
    drawn and ``payload["edges"]`` says which layer produced the sharp
    chords (``"graph"`` or ``"lattice"``).

    Optional viewer data: any failure prints a note (captured into the
    compile output) and returns ``None`` rather than failing the compile.
    """
    from cadjoint import tier

    try:
        grid = _overlay_grid()
        component = tier.component(PluginKind.FEATURE_EDGES.value)
        source = "lattice" if component is None else "graph"
        if component is None:
            vertices, quad_edges, sharp = _lattice_layers(scene, grid)
        else:
            vertices, quad_edges, sharp = _graph_layers(component, scene, grid)
        wire_edges = np.unique(np.sort(quad_edges, axis=1), axis=0)

        def segments(pairs: np.ndarray) -> list[list[list[float]]]:
            return [
                [[round(float(value), 3) for value in point] for point in pair] for pair in pairs
            ]

        return {
            "wire": segments(vertices[wire_edges]),
            "sharp": segments(sharp),
            "resolution": _MESH_EDGE_RESOLUTION,
            "edges": source,
        }
    except Exception as error:  # noqa: BLE001 - viewer extra must never break compiles
        print(f"mesh edge view unavailable: {error}")
        return None
