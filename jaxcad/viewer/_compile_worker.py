"""Child process used by the local browser playground compiler."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import traceback
from typing import Any

import numpy as np

from jaxcad.backends.wgsl import compile_scene_to_wgsl
from jaxcad.constraints.solve import capture_constraint_solves
from jaxcad.viewer._pathtracer import (
    WGSL_PRESENT_TEMPLATE,
    build_path_tracer_shader,
)
from jaxcad.viewer._source_map import (
    PLAYGROUND_FILENAME,
    build_construction_payload,
    build_construction_relations,
    build_material_payload,
    capture_profiles,
)
from jaxcad.viewer._webgpu import build_viewer_shader


def _differentiability_payload(namespace: dict[str, Any]) -> dict[str, Any] | None:
    """Serialize an optional, source-computed autodiff demonstration.

    The program computes the derivative itself, keeping the proof transparent
    and editable. A malformed optional demo never prevents its scene from
    compiling.
    """
    demo = namespace.get("differentiability_demo")
    if not isinstance(demo, dict):
        return None
    try:
        value = float(demo["value"])
        parameter_count = int(demo["parameter_count"])
        sensitivities = [
            {
                "parameter": str(item["parameter"]),
                "value": float(item["value"]),
            }
            for item in demo["sensitivities"]
            if isinstance(item, dict)
        ]
        if (
            not math.isfinite(value)
            or parameter_count < 1
            or not sensitivities
            or any(not math.isfinite(item["value"]) for item in sensitivities)
        ):
            return None
        return {
            "pipeline": str(demo["pipeline"]),
            "metric": str(demo["metric"]),
            "value": value,
            "parameter_count": parameter_count,
            "sensitivities": sensitivities,
        }
    except (KeyError, TypeError, ValueError):
        return None


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
    from jaxcad.sdf.boolean.base import BooleanOp

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

        from jaxcad.meshing import (
            CORNER,
            FACE,
            GridSpec,
            cell_edge_incidence,
            classify_feature_cells,
            dual_faces,
            edge_hermite_data,
            feature_cell_links,
            find_crossing_edges,
            sample_grid,
            sharp_qef_vertices,
        )

        grid = GridSpec.from_bounds(_MESH_EDGE_BOUNDS, _MESH_EDGE_SIZE, _MESH_EDGE_RESOLUTION)
        sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
        edges = find_crossing_edges(sample_grid(sdf, grid))
        if edges.count == 0:
            return None
        incidence = cell_edge_incidence(edges, grid)
        hermite = edge_hermite_data(sdf, grid, edges)
        vertices = sharp_qef_vertices(hermite, incidence, grid)
        quads, _triangles, _skipped = dual_faces(edges, incidence, grid, vertices)
        quad_edge_list = np.concatenate(
            [quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]
        )

        features = classify_feature_cells(hermite, incidence)
        geometric_mask = features.classes != FACE
        junctions = features.classes == CORNER

        # Feature curves must never cross-link: two distinct curves (a crease
        # and a nearby seam, or seams of different operand pairs) can run
        # within one cell of each other, and linking across them weaves an
        # X-lattice between the curves.  Cells are therefore grouped by
        # feature identity — the owning CSG operand for geometric creases,
        # the operand pair for seams — and linked only within their group.
        leaves = _world_frame_leaves(scene)
        groups: list[np.ndarray] = []
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
            structural[seam_rows] = True
            geometric_only = geometric_mask & ~structural
            for owner in np.unique(owners[geometric_only]):
                groups.append(geometric_only & (owners == owner))
            # Seam vertices are Newton-projected onto the exact common zero
            # set of every operand meeting there — a curve for two, a triple
            # point for three: min/max structure makes seams a solvable
            # system, not a mesh approximation.  Grouping by the full
            # operand set projects each vertex exactly once.
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
                    vertices[selected] = _project_to_seam(
                        [leaves[index] for index in sorted(operand_set)],
                        vertices[selected],
                        max_step,
                    )
                    group_mask = np.zeros_like(geometric_mask)
                    group_mask[selected] = True
                    groups.append(group_mask)
        else:
            groups.append(geometric_mask)

        link_groups = [
            feature_cell_links(group, incidence, grid, junction_mask=junctions) for group in groups
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

            directions = vertices[links[:, 1]] - vertices[links[:, 0]]
            lengths = np.maximum(np.linalg.norm(directions, axis=1), 1e-12)
            keep = np.ones(links.shape[0], dtype=bool)
            for column in (0, 1):
                rows = links[:, column]
                tangent = tangents[rows]
                defined = np.isfinite(tangent).all(axis=1) & ~junctions[rows]
                alignment = np.abs(np.einsum("li,li->l", directions, tangent)) / lengths
                keep &= ~defined | (alignment > 0.75)
            links = links[keep]

        def segments(pairs: np.ndarray) -> list[list[list[float]]]:
            return [
                [[round(float(value), 3) for value in point] for point in pair] for pair in pairs
            ]

        wire_edges = np.unique(np.sort(quad_edge_list, axis=1), axis=0)
        return {
            "wire": segments(vertices[wire_edges]),
            "sharp": segments(vertices[links]),
            "resolution": _MESH_EDGE_RESOLUTION,
        }
    except Exception as error:  # noqa: BLE001 - viewer extra must never break compiles
        print(f"mesh edge view unavailable: {error}")
        return None


def _compile_source(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__jaxcad_playground__",
    }
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        with (
            capture_constraint_solves() as solver_reports,
            capture_profiles(PLAYGROUND_FILENAME) as profiles,
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
        differentiability = _differentiability_payload(namespace)
        mesh_edges = _mesh_edge_payload(namespace["scene"])
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
        "differentiability": differentiability,
        "mesh_edges": mesh_edges,
        "solver_runs": solver_runs,
        "output": captured.getvalue()[-8_000:],
    }


def main() -> None:
    try:
        request = json.load(sys.stdin)
        source = request.get("source")
        if not isinstance(source, str):
            raise TypeError("The compile request must contain a string `source` field.")
        result = _compile_source(source)
    except Exception:
        result = {"ok": False, "error": traceback.format_exc()}
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
