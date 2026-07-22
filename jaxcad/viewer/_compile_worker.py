"""Child process used by the local browser playground compiler."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from typing import Any

from jaxcad.backends.wgsl import compile_sdf_to_wgsl
from jaxcad.viewer._webgpu import build_viewer_shader


def _compile_source(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__jaxcad_playground__",
    }
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        exec(compile(source, "<jaxcad-playground>", "exec"), namespace, namespace)
        if "scene" not in namespace:
            raise ValueError("Your program must assign the SDF to a variable named `scene`.")
        sdf_code = compile_sdf_to_wgsl(namespace["scene"])

    return {
        "ok": True,
        "sdf": sdf_code,
        "shader": build_viewer_shader(sdf_code),
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
