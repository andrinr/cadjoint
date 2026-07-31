"""Child process used by the local browser playground compiler."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import traceback
from typing import Any

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
