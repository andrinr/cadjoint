"""Child process used by the local browser playground compiler."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import traceback
import warnings
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
# The sharpness threshold must sit above the faceting angle of the smallest
# curved feature the resolution can carry (about 2*asin(spacing / 2r)), or
# ordinary curvature misreads as creases; resolution 48 keeps even a
# 0.18-radius tube's faceting near 41 degrees, under the 45-degree
# threshold, while real CAD creases (rims, ridges, eaves) sit at 55-90
# degrees.  The structural CSG-seam check below has no such tuning: it is
# exact.
_MESH_EDGE_BOUNDS = (-3.0, -3.0, -3.0)
_MESH_EDGE_SIZE = (6.0, 6.0, 6.0)
_MESH_EDGE_RESOLUTION = 48
_SHARP_DIHEDRAL_DEGREES = 45.0


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


def _mesh_edge_payload(scene: Any) -> dict[str, Any] | None:
    """Extract the dual-contour mesh and split its edges into wire and sharp.

    Only native quad edges are reported — triangulation diagonals double the
    visual density without carrying any surface information.  An edge is
    sharp when the dihedral between its two triangles exceeds the threshold
    OR when its endpoints belong to different CSG operands (exact ``min``/
    ``max`` seam detection, independent of any angle).

    Optional viewer data: any failure prints a note (captured into the
    compile output) and returns ``None`` rather than failing the compile.
    """
    try:
        import jax.numpy as jnp

        from jaxcad.meshing import GridSpec, extract_mesh

        grid = GridSpec.from_bounds(_MESH_EDGE_BOUNDS, _MESH_EDGE_SIZE, _MESH_EDGE_RESOLUTION)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # an open mesh is fine for edge display
            mesh = extract_mesh(lambda p: jnp.asarray(scene(p)), grid)
        quads = mesh.quads
        if quads.shape[0] == 0:
            return None
        vertices = np.asarray(mesh.vertices, dtype=np.float64)

        # Per-quad normals via the diagonal cross product: smoother than
        # triangle normals, whose arbitrary diagonal split inflates the
        # apparent dihedral on curved surfaces.
        corners = vertices[quads]
        normals = np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(lengths, 1e-30)[:, None]

        directed = np.concatenate(
            [quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]
        )
        quad_ids = np.tile(np.arange(quads.shape[0]), 4)
        undirected, inverse = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
        # Up to two quads meet at each undirected edge of a manifold mesh.
        first = np.full(undirected.shape[0], -1, dtype=np.int64)
        second = np.full(undirected.shape[0], -1, dtype=np.int64)
        for edge_row, quad in zip(inverse, quad_ids):
            if first[edge_row] < 0:
                first[edge_row] = quad
            elif second[edge_row] < 0:
                second[edge_row] = quad
        interior = second >= 0
        cosine = np.einsum(
            "ij,ij->i",
            normals[np.maximum(first, 0)],
            normals[np.maximum(second, 0)],
        )
        threshold = math.cos(math.radians(_SHARP_DIHEDRAL_DEGREES))
        sharp_mask = interior & (cosine < threshold)

        # Exact structural seams: each vertex is owned by the CSG operand
        # whose field magnitude vanishes there; an edge crossing owners is a
        # seam even when the dihedral is shallow.
        leaves = _world_frame_leaves(scene)
        if len(leaves) >= 2:
            import jax

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
            sharp_mask |= owners[undirected[:, 0]] != owners[undirected[:, 1]]

        def segments(mask: np.ndarray) -> list[list[list[float]]]:
            pairs = vertices[undirected[mask]]
            return [
                [[round(float(value), 3) for value in point] for point in pair] for pair in pairs
            ]

        return {
            "wire": segments(~sharp_mask),
            "sharp": segments(sharp_mask),
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
