"""Child process used by the local browser playground compiler.

The worker's entry point and stdin/stdout protocol: it reads one JSON
request object from stdin, dispatches on its ``mode``, and writes the JSON
response to stdout.  ``mode="optimize"`` additionally streams NDJSON
progress lines to that same stdout before the final response object (see
:mod:`cadjoint.viewer._worker_optimize`).  Any exception becomes an
``{"ok": false, "error": <traceback>}`` response.

``mode="compile"`` — the worker's namesake, implemented here — runs the
program and builds the shaders, construction tree, and declaration entries
the viewer opens with, then checks the whole payload against
:mod:`cadjoint.viewer.schema.payloads` before sending it: the models are
what the frontend's generated types are emitted from, so a payload that
does not match them would be a type the browser was promised and did not
get.  The other modes live beside it:
:mod:`._edge_overlay` (``mesh``), :mod:`._worker_fem` (``simulate``,
``mesh_inspect``), :mod:`._worker_optimize` (``optimize``), and
:mod:`._export` (``export``).
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from typing import Any

from cadjoint.backends.wgsl import compile_scene_to_wgsl
from cadjoint.cache import enable_compilation_cache
from cadjoint.constraints.solve import capture_constraint_solves
from cadjoint.viewer._edge_overlay import (  # noqa: F401 - re-exported for callers
    _MESH_EDGE_RESOLUTION,
    _MESH_EDGE_SIZE,
    _mesh_edge_payload,
)
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
    describe_identities,
)
from cadjoint.viewer._webgpu import build_viewer_shader
from cadjoint.viewer._worker_declarations import (
    _mesh_entries,
    _optimization_entries,
    _study_entries,
)
from cadjoint.viewer._worker_fem import (  # noqa: F401 - re-exported for callers
    _inspect_mesh,
    _mesh_inspect_source,
    _simulate_source,
    _simulate_study,
)
from cadjoint.viewer._worker_optimize import _optimize_source
from cadjoint.viewer._worker_scene import _execute_scene
from cadjoint.viewer.schema.payloads import validate_compile_payload


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


def _compile_source(source: str) -> dict[str, Any]:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(
            source,
            capture=(
                ("__solver_reports__", capture_constraint_solves),
                ("__profiles__", lambda: capture_profiles(PLAYGROUND_FILENAME)),
            ),
        )
        solver_reports = namespace["__solver_reports__"]
        profiles = namespace["__profiles__"]
        sim_meshes = namespace["__sim_meshes__"]
        studies = namespace["__studies__"]
        optimizations = namespace["__optimizations__"]
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
        optimizations_payload = _optimization_entries(optimizations, source, namespace["scene"])
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

    return validate_compile_payload(
        {
            "ok": True,
            "sdf": scene_code,
            "shader": preview_shader,
            "scene_wgsl": scene_code,
            "preview_shader": preview_shader,
            "path_shader": path_shader,
            "present_shader": WGSL_PRESENT_TEMPLATE,
            "construction": construction,
            # Every stable id the text declares, so the viewer can name anything
            # the payload reports only by line.
            "identities": describe_identities(source),
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
    )


def main() -> None:
    # Every request runs in a fresh process, so without this each edit
    # recompiles the same XLA programs from scratch (see cadjoint.cache).
    enable_compilation_cache()
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
        elif mode == "export":
            from cadjoint.viewer._export import export_scene

            result = export_scene(request)
        elif mode == "compile":
            result = _compile_source(source)
        else:
            raise ValueError(f"Unknown compile worker mode: {mode!r}.")
    except Exception:
        result = {"ok": False, "error": traceback.format_exc()}
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
