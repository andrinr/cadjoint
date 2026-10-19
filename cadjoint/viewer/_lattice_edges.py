"""The overlay's public fallback: feature edges read off the lattice.

What :mod:`cadjoint.viewer._edge_overlay` draws when nothing fills the
``feature_edges`` plugin kind.  Feature cells are classified from normal
spreads and exact ``min``/``max`` CSG seams, the seam vertices are
Newton-projected onto the common zero set of the operands meeting there
(:func:`_project_seam_groups`, one program for every group at once), and
the surviving cell links are the chords.  The module docstring of
:mod:`cadjoint.viewer._edge_overlay` says what this tier is and is not.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadjoint.viewer._design_leaves import _design_leaves, _world_frame_leaves

# ── leaf-level seam projection ───────────────────────────────────────────────
#
# The lattice sharp layer finds CSG seams from the leaves: group the mesh
# vertices whose surface ownership flips by the *operand set* they sit
# between, then Newton-project each group onto the common zero set of those
# operands' whole SDFs.  `_project_seam_groups` does that for every group in
# one program and is what `_lattice_layers` calls.  `_project_to_seam` and
# `_project_seam_groups_reference` are the one-program-per-group reading it
# replaced; they stay because `tests/viewer/test_edge_artifacts.py` proves
# the two extract the same edges, and the batched form is only trusted
# because of that proof.  The private tier's B-rep edges do the same thing
# one level down, on patch fields rather than leaf fields.


def _project_to_seam(fields: list[Any], points: np.ndarray, max_step: float) -> np.ndarray:
    """Newton-project points onto the common zero set of two or more fields.

    :func:`cadjoint.zeroset.project.project` on the seam's fields: the
    minimum-norm step, a relative regularisation of the Gram matrix, and a
    refusal for points whose fields meet tangentially — which stay put.
    """
    from cadjoint.zeroset.project import project

    return np.asarray(
        project(fields, np.asarray(points, dtype=np.float64), steps=4, max_step=max_step),
        dtype=np.float64,
    )


def _seam_residual(fields: list[Any], points: np.ndarray) -> np.ndarray:
    """How far *points* still are from every field's zero set (the max |f|).

    The acceptance test for a projected seam group: a genuine seam vertex
    lands on every operand's zero set, a near-miss between disjoint
    surfaces keeps a residual of order the gap.
    """
    import jax
    import jax.numpy as jnp

    probes = jnp.asarray(points, dtype=jnp.float32)
    # One program over every field, not an eager walk of each: see
    # `_project_seam_groups`.
    program = jax.jit(
        lambda p: jnp.max(
            jnp.abs(
                jnp.stack([jax.vmap(lambda q, f=field: jnp.asarray(f(q)))(p) for field in fields])
            ),
            axis=0,
        )
    )
    return np.asarray(program(probes), dtype=np.float64)


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

    :func:`_project_to_seam` is exact but pays a fixed cost per call —
    building the ``vmap(value_and_grad)`` evaluators and dispatching four
    Newton iterations — that is independent of how many points it
    projects; measured, one point costs as much as three hundred.  A scene
    with context geometry produces a seam group per operand *set*, so the
    per-call cost was what the whole mesh overlay used to be made of
    (measured on the starter: 15 groups, 108 points, 5.6 s).

    So every group goes through :func:`cadjoint.zeroset.project.project_groups`,
    the padded multi-group form of the one projection kernel: one program,
    each point gathering its own operands.  The seam points are solved in
    float32, as the residual test's thresholds assume.

    Returns one ``(projected, residual)`` pair per group, in the order the
    groups were given.
    """
    from cadjoint.zeroset.project import project_groups

    results = project_groups(
        leaves, groups, np.asarray(vertices, dtype=np.float32), steps=4, max_step=max_step
    )
    return [
        (np.asarray(projected, dtype=np.float64), np.asarray(residual, dtype=np.float64))
        for projected, residual in results
    ]


#: Leaves whose ``|f|`` at a dual vertex is within this fraction of a cell
#: of the smallest are all nearest to it (see :func:`_nearest_leaves`).
#: Float32 leaf values carry ~1e-7 of noise; geometry that is only *nearly*
#: symmetric differs by far more than 1e-5 cell.
_OWNER_TIE = 1e-5

#: A vertex within this fraction of a cell of its nearest leaves lies *on*
#: them (see :func:`_nearest_leaves`).  Crease vertices sit at the QEF's
#: float32 floor, ~1e-6 cell; vertices merely equidistant from two walls
#: sit at half their gap, a few hundredths of a cell and up.
_ON_SURFACE = 1e-3


def _nearest_leaves(magnitudes: np.ndarray, grid: Any) -> np.ndarray:
    """Which leaves own each dual vertex, with ties decided by geometry, not noise.

    Ownership used to be one ``argmin``, and an argmin between equal values
    is decided by float noise.  Geometry produces equal values all the
    time: a crease vertex of ``min(a, b)`` lies on both operands at once,
    a vertex in a gap below the grid's resolution sits midway between its
    two walls, and coincident faces are equally far from both.  The seam
    test compares owners across quad edges, so a flipped tie added or
    removed a seam edge, changed which group its vertices were projected
    in, and moved them by up to 0.028 between two evaluations of end_cap
    whose Hermite data agree to 1e-7.

    Values within :data:`_OWNER_TIE` of a cell are therefore read as
    equal, and what a tie means depends on where it is.  A vertex *on* its
    nearest leaves (within :data:`_ON_SURFACE` of a cell) belongs to all
    of them: it sits on their crease, and its seam is theirs.  A vertex
    merely equidistant from several belongs to the lowest-numbered one:
    it is on none of them, and putting every one into its seam's operand
    set asks the projection for a point on surfaces it is nowhere near.

    Args:
        magnitudes: ``|f|`` of every leaf at every vertex, ``(leaves, vertices)``.
        grid: The extraction grid (its spacing scales the tolerance).

    Returns:
        A ``(leaves, vertices)`` mask of each vertex's owners; every column
            holds at least one.
    """
    tolerance = _OWNER_TIE * float(max(grid.spacing))
    nearest = magnitudes.min(axis=0, keepdims=True)
    tied = magnitudes <= nearest + tolerance
    on_surface = nearest <= _ON_SURFACE * float(max(grid.spacing))
    lowest = np.arange(magnitudes.shape[0])[:, None] == np.argmax(tied, axis=0)[None, :]
    return np.where(on_surface, tied, lowest)


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
    # the surface, so re-evaluate the gradient h/4 off each
    # incident crossing and demote feature cells whose verified normal
    # spread is face-like.  Genuine creases keep their fan: each
    # crossing edge pierces one smooth facet, and different edges
    # pierce different facets.
    feature_rows = np.flatnonzero(geometric_mask)
    if feature_rows.size:
        cell_edges = incidence.edge_ids[feature_rows]
        used = np.unique(cell_edges[cell_edges >= 0])
        # A quarter cell off every crossing along all three axes, not only
        # along the crossing's own edge.  A crease that lies exactly in a
        # lattice plane (the house's roof ridge at x = 0) puts most of its
        # crossings *on* the crease, where the field's two facets tie; a
        # probe along an edge that runs in that plane stays on the tie, and
        # the gradient there is whichever facet the evaluation order picks
        # -- the same one every time.  The cell then sees one facet's normal
        # several times over the other's, its spread falls under the bar
        # and a real crease is demoted.  Stepping off along every axis puts
        # a probe on each side of any crease through the crossing.
        offsets = 0.25 * np.asarray(grid.spacing, dtype=np.float64)[:, None] * np.eye(3)
        base = np.asarray(hermite.points, dtype=np.float64)[used]
        probes = jnp.asarray(
            np.concatenate([base + sign * offset for offset in offsets for sign in (-1.0, 1.0)]),
            jnp.float32,
        )
        gradients = np.asarray(jax.jit(jax.vmap(jax.grad(sdf)))(probes), dtype=np.float64)
        scale = np.linalg.norm(gradients, axis=1, keepdims=True)
        probe_normals = np.where(scale > 1e-9, gradients / np.maximum(scale, 1e-12), 0.0)
        lookup = np.zeros(int(used.max()) + 1, dtype=np.int64)
        lookup[used] = np.arange(used.size)
        slots = lookup[np.maximum(cell_edges, 0)]
        samples = np.concatenate(
            [probe_normals[slots + block * used.size] for block in range(6)], axis=1
        )
        sample_valid = np.concatenate([cell_edges >= 0] * 6, axis=1) & (
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
        # Every leaf in one program: an eager `vmap` per leaf dispatches
        # one XLA program per primitive of that leaf's whole SDF tree.
        magnitudes = np.abs(
            np.asarray(
                jax.jit(
                    lambda p: jnp.stack(
                        [jax.vmap(lambda q, f=leaf: jnp.asarray(f(q)))(p) for leaf in leaves]
                    )
                )(points),
                dtype=np.float64,
            )
        )
        nearest = _nearest_leaves(magnitudes, grid)
        # One owner per vertex for the design filter and the geometric
        # groups: the lowest-numbered of its owners, a choice that does not
        # depend on noise.  The seam test below reads the whole set.
        owners = np.argmax(nearest, axis=0)
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
        # A seam edge joins vertices with different owners; a vertex on a
        # crease, owned by both leaves meeting there, differs from each
        # side's.
        mismatched = quad_edge_list[
            np.any(nearest[:, quad_edge_list[:, 0]] != nearest[:, quad_edge_list[:, 1]], axis=0)
        ]
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
                meeting = np.flatnonzero(nearest[:, u] | nearest[:, v]).tolist()
                for row in (int(u), int(v)):
                    incident.setdefault(row, set()).update(meeting)
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
            # Both operands' gradients in one program, for the reason
            # `magnitudes` above is one program.
            pair = jax.jit(
                lambda p, a=leaves[index_a], b=leaves[index_b]: (
                    jax.vmap(jax.grad(lambda q: jnp.asarray(a(q))))(p),
                    jax.vmap(jax.grad(lambda q: jnp.asarray(b(q))))(p),
                )
            )
            raw_a, raw_b = pair(points)
            grad_a = np.asarray(raw_a, dtype=np.float64)
            grad_b = np.asarray(raw_b, dtype=np.float64)
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
