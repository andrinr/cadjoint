"""Child process used by the local browser playground compiler."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from typing import Any

import numpy as np

from cadjoint.backends.wgsl import compile_scene_to_wgsl
from cadjoint.constraints.solve import capture_constraint_solves
from cadjoint.viewer._pathtracer import (
    WGSL_PRESENT_TEMPLATE,
    build_path_tracer_shader,
)
from cadjoint.viewer._source_map import (
    PLAYGROUND_FILENAME,
    build_construction_payload,
    build_construction_relations,
    build_material_payload,
    capture_profiles,
    locate_mesh_statements,
    locate_optimization_statements,
    locate_study_statements,
)
from cadjoint.viewer._webgpu import build_viewer_shader

# Mesh-edge view settings.  The grid matches the raymarcher's view volume.
# Detection stays dense rather than Lipschitz-pruned: user-written fields
# can exceed any assumed gradient bound, and a hole in the viewer is worse
# than the ~100 ms this costs.
_MESH_EDGE_BOUNDS = (-3.0, -3.0, -3.0)
_MESH_EDGE_SIZE = (6.0, 6.0, 6.0)
_MESH_EDGE_RESOLUTION = 64


def _world_frame_leaves(node: Any) -> list[Any]:
    """Maximal world-frame subtrees below the scene's Boolean structure.

    Hard CSG is built from ``min``/``max``, so the exact seam between two
    operands is where surface ownership switches between them — no angular
    threshold involved.  Descend only through Boolean nodes: their children
    share the parent's coordinate frame and stay callable in world space,
    while anything else (a transformed subtree) becomes one opaque leaf.
    """
    from cadjoint.sdf.boolean.base import BooleanOp

    if isinstance(node, BooleanOp):
        leaves: list[Any] = []
        for child in node.children():
            leaves.extend(_world_frame_leaves(child))
        return leaves
    return [node]


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


def _mesh_edge_payload(scene: Any) -> dict[str, Any] | None:
    """Extract the dual-contour mesh and split its edges into wire and sharp.

    The wire layer is the mesh's native quad edges (triangulation diagonals
    carry no surface information).  The sharp layer is *not* mesh edges: a
    feature curve crosses grid cells diagonally, so its cells are usually
    not mesh-adjacent and mesh edges would trace a staircase around it.
    Instead, feature cells — normal-spread creases and corners, plus exact
    ``min``/``max`` CSG seams — are linked to their lattice neighbors, and
    because feature-aware placement puts each of their vertices exactly on
    the feature curve, those links are chords of the true curve.

    Optional viewer data: any failure prints a note (captured into the
    compile output) and returns ``None`` rather than failing the compile.
    """
    try:
        import jax
        import jax.numpy as jnp

        from cadjoint.meshing import (
            CORNER,
            FACE,
            GridSpec,
            classify_feature_cells,
            dual_faces,
            edge_hermite_data,
            feature_cell_links,
            find_crossing_edges,
            manifold_cell_incidence,
            sample_grid,
            sharp_qef_vertices,
        )
        from cadjoint.meshing.native import native_available

        # The Rust core is a bit-identical drop-in for the heavy stages
        # (detection, incidence, QEF placement, faces); the SDF-evaluating
        # stages stay in JAX either way.  Fall back to the reference Python
        # pipeline when the native library is not built.
        native = native_available()
        if native:
            from cadjoint.meshing.native import (
                dual_faces_native as dual_faces,
            )
            from cadjoint.meshing.native import (
                find_crossing_edges_native as find_crossing_edges,
            )
            from cadjoint.meshing.native import (
                manifold_cell_incidence_native as manifold_cell_incidence,
            )
            from cadjoint.meshing.native import (
                sharp_qef_vertices_native as sharp_qef_vertices,
            )

        grid = GridSpec.from_bounds(_MESH_EDGE_BOUNDS, _MESH_EDGE_SIZE, _MESH_EDGE_RESOLUTION)
        sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
        values = sample_grid(sdf, grid)
        edges = find_crossing_edges(values)
        if edges.count == 0:
            return None
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
            offsets = (0.25 * np.asarray(grid.spacing, dtype=np.float64)[axes])[:, None] * np.eye(
                3
            )[axes]
            base = np.asarray(hermite.points, dtype=np.float64)[used]
            probes = jnp.asarray(np.concatenate([base - offsets, base + offsets]), jnp.float32)
            gradients = np.asarray(jax.vmap(jax.grad(sdf))(probes), dtype=np.float64)
            scale = np.linalg.norm(gradients, axis=1, keepdims=True)
            probe_normals = np.where(scale > 1e-9, gradients / np.maximum(scale, 1e-12), 0.0)
            lookup = np.zeros(int(used.max()) + 1, dtype=np.int64)
            lookup[used] = np.arange(used.size)
            slots = lookup[np.maximum(cell_edges, 0)]
            samples = np.concatenate(
                [probe_normals[slots], probe_normals[slots + used.size]], axis=1
            )
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
            mismatched = quad_edge_list[
                owners[quad_edge_list[:, 0]] != owners[quad_edge_list[:, 1]]
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
                    for row in (int(u), int(v)):
                        incident.setdefault(row, set()).update((int(owners[u]), int(owners[v])))
                by_operands: dict[frozenset[int], list[int]] = {}
                for row, operand_set in incident.items():
                    by_operands.setdefault(frozenset(operand_set), []).append(row)
                max_step = 2.0 * max(_MESH_EDGE_SIZE) / _MESH_EDGE_RESOLUTION  # ~2 cells
                for operand_set, rows in by_operands.items():
                    selected = np.asarray(sorted(rows), dtype=np.int64)
                    fields = [leaves[index] for index in sorted(operand_set)]
                    projected = _project_to_seam(fields, vertices[selected], max_step)
                    residual = np.max(
                        np.stack(
                            [
                                np.abs(
                                    np.asarray(
                                        jax.vmap(lambda p, f=field: jnp.asarray(f(p)))(
                                            jnp.asarray(projected, dtype=jnp.float32)
                                        ),
                                        dtype=np.float64,
                                    )
                                )
                                for field in fields
                            ]
                        ),
                        axis=0,
                    )
                    genuine = residual < 0.1 * max(grid.spacing)
                    selected = selected[genuine]
                    if selected.size == 0:
                        continue
                    vertices[selected] = projected[genuine]
                    structural[selected] = True
                    group_mask = np.zeros_like(geometric_mask)
                    group_mask[selected] = True
                    groups.append(group_mask)
                    if len(operand_set) == 2:
                        seam_groups.append((selected, tuple(sorted(operand_set))))
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
                tangents[seam_rows] = np.where(
                    reliable, cross / np.maximum(cross_norms, 1e-12), np.nan
                )
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
                        find(int(a)) not in component_open
                        or component_length[find(int(a))] >= limit
                        for a, _b in links
                    ],
                    dtype=bool,
                )
                links = links[keep_component]

        def segments(pairs: np.ndarray) -> list[list[list[float]]]:
            return [
                [[round(float(value), 3) for value in point] for point in pair] for pair in pairs
            ]

        wire_edges = np.unique(np.sort(quad_edge_list, axis=1), axis=0)
        return {
            "wire": segments(vertices[wire_edges]),
            "sharp": segments(vertices[links]),
            "resolution": _MESH_EDGE_RESOLUTION,
            "native": native,
        }
    except Exception as error:  # noqa: BLE001 - viewer extra must never break compiles
        print(f"mesh edge view unavailable: {error}")
        return None


def _study_entries(studies: list[Any], source: str) -> list[dict[str, Any]]:
    """Serialize declared studies for the viewer, with their source locations.

    Mirrors how constraints flow into the payload: each entry is the study's
    ``describe()`` dict plus a stable ``index``, the statement's ``line`` and
    the constructor call's character ``span``, an ``editable`` flag, and a
    per-BC ``serializable`` flag (false only for predicate selections) with
    the BC argument's character ``span``.

    Studies are matched to source statements positionally: top-level
    declarations execute in source order, so the alignment holds exactly when
    the counts and kinds agree.  Anything else (studies built in loops, from
    helper functions) still renders but is marked non-editable.
    """
    statements = locate_study_statements(source) or []
    aligned = len(statements) == len(studies) and all(
        statement.kind == study.describe()["kind"] for statement, study in zip(statements, studies)
    )
    entries: list[dict[str, Any]] = []
    for index, study in enumerate(studies):
        described = study.describe()
        statement = statements[index] if aligned else None
        bc_spans: tuple[Any, ...] = ()
        if statement is not None and len(statement.bc_spans) == len(study.bcs):
            bc_spans = statement.bc_spans
        entries.append(
            {
                **described,
                "index": index,
                "line": statement.statement.lineno if statement is not None else None,
                "span": list(statement.call_span) if statement is not None else None,
                "editable": statement is not None,
                "mesh_span": list(statement.mesh_span)
                if statement is not None and statement.mesh_span is not None
                else None,
                "domain_span": list(statement.domain_span)
                if statement is not None and statement.domain_span is not None
                else None,
                "bcs": [
                    {
                        **bc.describe(),
                        "serializable": bc.nodes.serializable,
                        "span": list(bc_spans[position]) if bc_spans else None,
                    }
                    for position, bc in enumerate(study.bcs)
                ],
            }
        )
    return entries


def _mesh_entries(sim_meshes: list[Any], source: str) -> list[dict[str, Any]]:
    """Serialize declared simulation meshes for the viewer, with locations.

    Mirrors :func:`_study_entries`: each entry is the mesh's ``describe()``
    dict plus a stable ``index``, the statement's ``line`` and the
    constructor call's character ``span``, and an ``editable`` flag.  Meshes
    are matched to source statements positionally; a count mismatch (meshes
    built in loops or helpers) or a literal-name mismatch marks every entry
    non-editable.  Declaration only: nothing is built here.
    """
    statements = locate_mesh_statements(source) or []
    aligned = len(statements) == len(sim_meshes) and all(
        statement.name is None or statement.name == mesh.name
        for statement, mesh in zip(statements, sim_meshes)
    )
    entries: list[dict[str, Any]] = []
    for index, mesh in enumerate(sim_meshes):
        statement = statements[index] if aligned else None
        entries.append(
            {
                **mesh.describe(),
                "index": index,
                "line": statement.statement.lineno if statement is not None else None,
                "span": list(statement.call_span) if statement is not None else None,
                "editable": statement is not None,
            }
        )
    return entries


def _optimization_entries(optimizations: list[Any], source: str) -> list[dict[str, Any]]:
    """Serialize declared optimizations for the viewer, with locations.

    Mirrors :func:`_mesh_entries`: each entry is the optimization's
    ``describe()`` dict plus a stable ``index``, the statement's ``line``
    and the constructor call's character ``span``, an ``editable`` flag,
    and the ``steps``/``learning_rate`` argument-value spans.
    Optimizations are matched to source statements positionally; a count
    mismatch (declarations built in loops or helpers) or a literal-name
    mismatch marks every entry non-editable.  Declaration only: nothing is
    optimized here.
    """
    statements = locate_optimization_statements(source) or []
    aligned = len(statements) == len(optimizations) and all(
        statement.name is None or statement.name == optimization.name
        for statement, optimization in zip(statements, optimizations)
    )
    entries: list[dict[str, Any]] = []
    for index, optimization in enumerate(optimizations):
        statement = statements[index] if aligned else None

        def span(value) -> list[int] | None:
            return list(value) if value is not None else None

        entries.append(
            {
                **optimization.describe(),
                "index": index,
                "line": statement.statement.lineno if statement is not None else None,
                "span": span(statement.call_span) if statement is not None else None,
                "editable": statement is not None,
                "steps_span": span(statement.steps_span) if statement is not None else None,
                "learning_rate_span": span(statement.learning_rate_span)
                if statement is not None
                else None,
            }
        )
    return entries


def _execute_scene(source: str) -> dict[str, Any]:
    """Run playground source and return its namespace (the scene lives inside).

    The exec always happens inside :func:`capture_sim_meshes` +
    :func:`capture_studies` + :func:`capture_optimizations` registries: a
    scene program that references a declared mesh by name (``mesh="..."``)
    can only resolve it through an active capture context, so every worker
    mode needs them, whether or not it looks at the captured lists
    afterwards.
    """
    from cadjoint.fem.simmesh import capture_sim_meshes
    from cadjoint.fem.study import capture_studies
    from cadjoint.optimize import capture_optimizations

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__cadjoint_playground__",
    }
    with (
        capture_sim_meshes() as sim_meshes,
        capture_studies() as studies,
        capture_optimizations() as optimizations,
    ):
        exec(compile(source, PLAYGROUND_FILENAME, "exec"), namespace, namespace)
    if "scene" not in namespace:
        raise ValueError("Your program must assign the SDF to a variable named `scene`.")
    namespace["__sim_meshes__"] = sim_meshes
    namespace["__studies__"] = studies
    namespace["__optimizations__"] = optimizations
    return namespace


def _mesh_source(source: str) -> dict[str, Any]:
    """Extract only the dual-contour mesh edges for the viewer.

    The mesh view is optional and expensive (it re-runs dual contouring on a
    dense grid), so the playground requests it lazily through ``/api/mesh``
    instead of paying for it on every compile.
    """
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(source)
        mesh_edges = _mesh_edge_payload(namespace["scene"])
    return {
        "ok": True,
        "mesh_edges": mesh_edges,
        "output": captured.getvalue()[-8_000:],
    }


_FEM_UNAVAILABLE_MESSAGE = (
    "FEM simulation needs the 'fem' extra (jax-fem). Install it with: pip install cadjoint[fem]"
)


def _named_study(studies: list[Any], name: Any) -> Any:
    """The one declared study called *name* (or raise, listing the others)."""
    matches = [study for study in studies if study.name == name]
    if not matches:
        declared = ", ".join(repr(study.name) for study in studies) or "none"
        raise ValueError(f"The program declares no study named {name!r} (declared: {declared}).")
    if len(matches) > 1:
        raise ValueError(f"The program declares more than one study named {name!r}.")
    return matches[0]


def _boundary_vertex_nodes(mesh: Any) -> np.ndarray:
    """Node indices behind the compacted boundary vertex list.

    Must mirror the compaction in
    :func:`cadjoint.fem.render_payload.boundary_render_payload`: quads are
    gathered group by group (sorted by group id) and their node ids
    deduplicated with ``np.unique``, so position *i* of the render payload's
    vertex arrays corresponds to mesh node ``result[i]``.
    """
    quads = np.concatenate(
        [mesh.boundary_faces[group_id].nodes for group_id in sorted(mesh.boundary_faces)],
        axis=0,
    )
    return np.unique(quads.reshape(-1))


def _finite_range(values: np.ndarray) -> list[float]:
    """``[min, max]`` over the finite entries, like the render payload's range."""
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    return [round(low, 6), round(high, 6)]


def _result_field_payload(result: Any, payload: dict[str, Any]) -> None:
    """Attach the full per-vertex field catalog to a render payload.

    The base payload carries one display scalar per vertex; inspection
    wants every solved field.  Thermal results expose ``temperature``
    (identical to the display scalars, kept for a uniform shape); elastic
    results expose ``von_mises`` (the display scalars) plus
    ``displacement_magnitude``, and the raw per-vertex ``displacements``
    so the viewer can draw a warped surface.  ``ranges`` maps each field
    to its finite ``[lo, hi]``.  Mapping happens here, viewer-side, from
    the concrete SimulationResult arrays.
    """
    if result.kind == "thermal":
        payload["fields"] = {"temperature": list(payload["scalars"])}
    else:
        used = _boundary_vertex_nodes(result.mesh)
        displacement = np.asarray(result.solution.displacement, dtype=np.float64)[used]
        magnitude = np.linalg.norm(displacement, axis=-1)
        payload["fields"] = {
            "von_mises": list(payload["scalars"]),
            "displacement_magnitude": [round(float(value), 6) for value in magnitude],
        }
        payload["displacements"] = [
            [round(float(component), 6) for component in row] for row in displacement
        ]
    payload["ranges"] = {
        name: _finite_range(np.asarray(values, dtype=np.float64))
        for name, values in payload["fields"].items()
    }


def _simulate_study(scene: Any, studies: list[Any], request: dict[str, Any]) -> dict[str, Any]:
    """Solve one study the scene program declared, by name.

    The study is the source of truth: mesh, material, and boundary
    conditions all come from its declaration — the request only picks which
    one to run.  Non-serializable (predicate) selections solve fine here
    since the declared objects are used directly.

    With ``cached=True`` the study's ``last_result`` is served without
    re-solving when it exists.  The cache lives on the study object, so over
    the HTTP API — where every request runs a fresh worker process — it only
    ever hits when the scene program itself called ``solve()`` while it
    executed; it is a per-worker-process cache, not a server-side one.
    """
    import jax.numpy as jnp

    from cadjoint.fem.render_payload import boundary_render_payload

    study = _named_study(studies, request.get("name"))

    try:
        import jax_fem  # noqa: F401
    except ImportError:
        return {"ok": False, "error_kind": "fem_unavailable", "error": _FEM_UNAVAILABLE_MESSAGE}

    sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
    cached = bool(request.get("cached")) and study.last_result is not None
    result = study.last_result if cached else study.solve(sdf)
    scalar = np.asarray(result.nodal_scalar(), dtype=np.float64)
    described = study.describe()
    described["bcs"] = [
        {**bc.describe(), "serializable": bc.nodes.serializable} for bc in study.bcs
    ]
    render_payload = boundary_render_payload(result.mesh, scalar)
    _result_field_payload(result, render_payload)
    return {
        "ok": True,
        "kind": "study",
        "study": described,
        "field": result.field,
        "mesh": render_payload,
        "result": result.describe(),
        "mesh_info": result.sim_mesh.inspect(sdf) if result.sim_mesh is not None else None,
        "cached": cached,
    }


def _simulate_source(request: dict[str, Any]) -> dict[str, Any]:
    """Run the FEM simulation mode: exec scene -> declared study -> payload."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(request["source"])
        result = _simulate_study(namespace["scene"], namespace["__studies__"], request)
    result["output"] = captured.getvalue()[-8_000:]
    return result


# The server validates requested step counts, but the declaration itself may
# ask for more than one HTTP-bounded run should pay for; the worker caps both.
OPTIMIZE_STEP_LIMIT = 200


def _named_optimization(optimizations: list[Any], name: Any) -> Any:
    """The one declared optimization called *name* (or raise, listing them)."""
    matches = [optimization for optimization in optimizations if optimization.name == name]
    if not matches:
        declared = ", ".join(repr(optimization.name) for optimization in optimizations) or "none"
        raise ValueError(
            f"The program declares no optimization named {name!r} (declared: {declared})."
        )
    if len(matches) > 1:
        raise ValueError(f"The program declares more than one optimization named {name!r}.")
    return matches[0]


def _run_optimization(
    source: str, optimizations: list[Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Run one declared optimization by name and patch its result into source.

    The optimizer is a patch layer: the optimized free-parameter values are
    written back into the program text through the same exact-repr patch
    machinery the viewer's other edits use, and the client adopts the
    returned ``source`` and recompiles — code parity, like ``/patch``.
    """
    from cadjoint.viewer._patch import set_parameter_values

    optimization = _named_optimization(optimizations, request.get("name"))
    steps = request.get("steps")
    steps = optimization.steps if steps is None else int(steps)
    run = optimization.run(steps=min(steps, OPTIMIZE_STEP_LIMIT))
    patched = set_parameter_values(source, run.parameters)
    return {
        "ok": True,
        "kind": "optimize",
        "name": optimization.name,
        "method": run.method,
        "steps": run.steps,
        "source": patched,
        "history": run.history,
        "trajectory": run.trajectory,
        "parameters": run.parameters,
        "initial": run.initial,
    }


def _optimize_source(request: dict[str, Any]) -> dict[str, Any]:
    """Run the optimize mode: exec scene -> declared optimization -> patch."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(request["source"])
        result = _run_optimization(request["source"], namespace["__optimizations__"], request)
    result["output"] = captured.getvalue()[-8_000:]
    return result


def _inspect_mesh(
    scene: Any, sim_meshes: list[Any], studies: list[Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Build one declared (or study-implicit) SimMesh and describe it.

    Resolution of the ``name`` request field:

    * a declared ``SimMesh`` name wins;
    * otherwise a declared study's name selects that study's mesh (its
      ``mesh=`` SimMesh, or the anonymous mesh implied by its
      resolution/bounds/size/domain);
    * with no name at all, a single declared mesh — or, failing that, a
      single declared study — is used.

    Returns the JSON inspection summary plus a renderable boundary surface
    whose scalars are the per-vertex scaled-jacobian quality field (each
    element's quality mapped to its 8 corners, min-combined), so the
    viewer can show a quality heatmap before anything is solved.
    """
    import jax.numpy as jnp

    from cadjoint.fem.hexmesh import scaled_jacobians
    from cadjoint.fem.render_payload import boundary_render_payload
    from cadjoint.fem.study import _solve_mesh

    sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
    name = request.get("name")

    def implicit(study: Any) -> Any:
        if study.mesh is not None:
            return study.mesh
        sim_mesh, _ = _solve_mesh(study, sdf, None)
        return sim_mesh

    declared = ", ".join(repr(mesh.name) for mesh in sim_meshes) or "none"
    if name is not None:
        matches = [mesh for mesh in sim_meshes if mesh.name == name]
        if len(matches) > 1:
            raise ValueError(f"The program declares more than one mesh named {name!r}.")
        if matches:
            target = matches[0]
        else:
            study_matches = [study for study in studies if study.name == name]
            if len(study_matches) > 1:
                raise ValueError(f"The program declares more than one study named {name!r}.")
            if not study_matches:
                studies_declared = ", ".join(repr(study.name) for study in studies) or "none"
                raise ValueError(
                    f"No declared mesh or study named {name!r} "
                    f"(meshes: {declared}; studies: {studies_declared})."
                )
            target = implicit(study_matches[0])
    elif len(sim_meshes) == 1:
        target = sim_meshes[0]
    elif not sim_meshes and len(studies) == 1:
        target = implicit(studies[0])
    else:
        raise ValueError(
            f"Pass `name` to pick a mesh: the program declares meshes {declared} "
            f"and {len(studies)} studies."
        )

    hex_mesh = target.build(sdf)
    quality = scaled_jacobians(hex_mesh.points, hex_mesh.cells)
    node_quality = np.full(hex_mesh.num_points, np.inf, dtype=np.float64)
    np.minimum.at(node_quality, hex_mesh.cells.reshape(-1), np.repeat(quality, 8))
    node_quality = np.where(np.isfinite(node_quality), node_quality, 1.0)
    payload = boundary_render_payload(hex_mesh, node_quality)
    return {
        "ok": True,
        "kind": "mesh_inspect",
        "name": target.name,
        "field": "scaled_jacobian",
        "info": target.inspect(sdf),
        "mesh": payload,
        "quality_scalars": payload["scalars"],
    }


def _mesh_inspect_source(request: dict[str, Any]) -> dict[str, Any]:
    """Run the mesh-inspection mode: exec scene -> build SimMesh -> payload."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(request["source"])
        result = _inspect_mesh(
            namespace["scene"], namespace["__sim_meshes__"], namespace["__studies__"], request
        )
    result["output"] = captured.getvalue()[-8_000:]
    return result


def _compile_source(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__cadjoint_playground__",
    }
    from cadjoint.fem.simmesh import capture_sim_meshes
    from cadjoint.fem.study import capture_studies
    from cadjoint.optimize import capture_optimizations

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        with (
            capture_constraint_solves() as solver_reports,
            capture_profiles(PLAYGROUND_FILENAME) as profiles,
            capture_sim_meshes() as sim_meshes,
            capture_studies() as studies,
            capture_optimizations() as optimizations,
        ):
            exec(compile(source, PLAYGROUND_FILENAME, "exec"), namespace, namespace)
        if "scene" not in namespace:
            raise ValueError("Your program must assign the SDF to a variable named `scene`.")
        scene_code = compile_scene_to_wgsl(namespace["scene"])
        preview_shader = build_viewer_shader(scene_code)
        path_shader = build_path_tracer_shader(scene_code)
        construction = build_construction_payload(profiles, source)
        relations = build_construction_relations(profiles)
        materials = build_material_payload(namespace, source)
        # Declaration only: studies, meshes, and optimizations are serialized
        # from their describe() payloads — no meshing, solving, or descending
        # happens at compile time.
        studies_payload = _study_entries(studies, source)
        sim_meshes_payload = _mesh_entries(sim_meshes, source)
        optimizations_payload = _optimization_entries(optimizations, source)
        node_ids = {
            id(obj): (f"{obj.kind}_{index}" if hasattr(obj, "kind") else f"profile_{index}")
            for index, (obj, _) in enumerate(profiles)
        }
        solver_runs = [
            {
                "node": node_ids.get(report["target_id"]),
                "method": report["method"],
                "iterations": report["iterations"],
                "losses": report["losses"],
            }
            for report in solver_reports
        ]

    return {
        "ok": True,
        "sdf": scene_code,
        "shader": preview_shader,
        "scene_wgsl": scene_code,
        "preview_shader": preview_shader,
        "path_shader": path_shader,
        "present_shader": WGSL_PRESENT_TEMPLATE,
        "construction": construction,
        "relations": relations,
        "materials": materials,
        "studies": studies_payload,
        "sim_meshes": sim_meshes_payload,
        "optimizations": optimizations_payload,
        # The mesh-edge view is requested lazily via `mode: "mesh"` — computing
        # it here used to dominate the compile round-trip.
        "mesh_edges": None,
        "solver_runs": solver_runs,
        "output": captured.getvalue()[-8_000:],
    }


def main() -> None:
    try:
        request = json.load(sys.stdin)
        source = request.get("source")
        if not isinstance(source, str):
            raise TypeError("The compile request must contain a string `source` field.")
        mode = request.get("mode", "compile")
        if mode == "mesh":
            result = _mesh_source(source)
        elif mode == "simulate":
            result = _simulate_source(request)
        elif mode == "mesh_inspect":
            result = _mesh_inspect_source(request)
        elif mode == "optimize":
            result = _optimize_source(request)
        elif mode == "compile":
            result = _compile_source(source)
        else:
            raise ValueError(f"Unknown compile worker mode: {mode!r}.")
    except Exception:
        result = {"ok": False, "error": traceback.format_exc()}
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
