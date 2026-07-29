from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from jaxcad.viewer._pathtracer import (
    WGSL_PRESENT_TEMPLATE,
    build_path_tracer_shader,
)
from jaxcad.viewer._webgpu import build_viewer_shader, ensure_material_wgsl
from jaxcad.viewer.playground import (
    EXAMPLE_SOURCE,
    compile_source,
    create_server,
    patch_source,
    resolve_static,
)


def test_example_scene_compiles_to_complete_webgpu_shader():
    result = compile_source(EXAMPLE_SOURCE)

    assert result["ok"] is True
    assert "fn sdf(" in result["sdf"]
    assert "@vertex" in result["shader"]
    assert "@fragment" in result["shader"]
    assert "fn material_base(" in result["scene_wgsl"]
    assert "fn material_optics(" in result["scene_wgsl"]
    assert "fn fs_path_trace(" in result["path_shader"]
    assert "fn fs_present(" in result["present_shader"]


def test_compile_source_reports_missing_scene():
    result = compile_source("answer = 42")

    assert result["ok"] is False
    assert "variable named `scene`" in result["error"]


def test_compile_source_enforces_timeout():
    result = compile_source("while True:\n    pass", timeout=0.1)

    assert result == {"ok": False, "error": "Compilation exceeded the 0.1-second timeout."}


def test_shader_builder_rejects_reserved_marker():
    with pytest.raises(ValueError, match="reserved marker"):
        build_viewer_shader("fn sdf() {} // __JAXCAD_SDF_CODE__")


def test_preview_builder_supplies_default_material_for_plain_sdf():
    shader = build_viewer_shader("fn sdf(p: vec3<f32>) -> f32 { return length(p); }")

    assert "fn material_base(" in shader
    assert "fn material_optics(" in shader
    assert "fn shade_material(" in shader
    assert "fn aces_tone_map(" in shader


def test_preview_builder_rejects_partial_material_interface():
    with pytest.raises(ValueError, match="both material WGSL functions"):
        ensure_material_wgsl(
            """
fn sdf(p: vec3<f32>) -> f32 { return length(p); }
fn material_base(p: vec3<f32>) -> vec4<f32> { return vec4<f32>(1.0); }
"""
        )


def test_path_tracer_builder_embeds_scene_transport_functions():
    scene_code = """
fn sdf(p: vec3<f32>) -> f32 { return length(p) - 1.0; }
fn material_base(p: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(0.8, 0.2, 0.1, 0.4);
}
fn material_optics(p: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(0.0, 1.0, 1.5, 0.0);
}
"""

    shader = build_path_tracer_shader(scene_code)

    assert scene_code in shader
    assert "fn trace_path(" in shader
    assert "fn refine_sign_crossing(" in shader
    assert "fn signs_differ(" in shader
    assert "fn sample_opaque_bsdf(" in shader
    assert "MAX_SHADOW_SAMPLES" in shader
    assert "configured_shadow_samples" in shader
    assert "previous_accumulation: texture_2d<f32>" in shader
    assert "@fragment\nfn fs_path_trace(" in shader
    assert "fn fs_present(" in WGSL_PRESENT_TEMPLATE


def test_path_tracer_builder_rejects_reserved_marker():
    with pytest.raises(ValueError, match="reserved marker"):
        build_path_tracer_shader("// __JAXCAD_SCENE_CODE__")


def test_example_scene_reports_its_construction_for_the_viewer():
    result = compile_source(EXAMPLE_SOURCE)

    assert result["ok"] is True
    nodes = {node["kind"]: node for node in result["construction"]}
    assert set(nodes) == {"profile", "sphere"}

    profile = nodes["profile"]
    assert profile["editable"] is True
    assert profile["name"] == "house"
    assert len(profile["vertices"]) == 5
    # Every vertex carries the span the viewer highlights and patches.
    for vertex in profile["vertices"]:
        start, end = vertex["span"]
        assert EXAMPLE_SOURCE[start:end].startswith("[")

    ball = nodes["sphere"]
    assert ball["editable"] is True
    assert ball["transform"]["position"] == pytest.approx([1.95, -0.25, 0.0], abs=1e-6)
    # A wireframe the viewer can draw without knowing the shape's topology.
    assert len(ball["edges"]) > 0
    start, end = ball["spans"]["position"]
    assert EXAMPLE_SOURCE[start:end] == "[1.95, -0.25, 0.0]"

    assert "fn fs_main_depth(" in result["preview_shader"]


def test_patch_source_round_trips_through_compile():
    edited = patch_source(
        {
            "source": EXAMPLE_SOURCE,
            "op": "set_vertex",
            "line": 10,
            "index": 0,
            "xy": [-1.4, -0.9],
        }
    )
    assert edited["ok"] is True

    result = compile_source(edited["source"])
    assert result["ok"] is True
    assert result["construction"][0]["vertices"][0]["uv"] == pytest.approx([-1.4, -0.9], abs=1e-6)


@pytest.mark.parametrize(
    ("request_body", "message"),
    [
        ({"op": "set_vertex", "line": 1, "index": 0}, "string `source`"),
        ({"source": "x = 1", "line": 1, "index": 0}, "string `op`"),
        ({"source": "x = 1", "op": "set_vertex", "index": 0}, "integer `line`"),
        ({"source": "x = 1", "op": "set_vertex", "line": 1, "index": 0}, "`xy` as two numbers"),
    ],
)
def test_patch_source_validates_its_request(request_body, message):
    result = patch_source(request_body)

    assert result["ok"] is False
    assert message in result["error"]


def test_resolve_static_refuses_paths_outside_the_static_root():
    assert resolve_static("/../../pyproject.toml") is None
    assert resolve_static("/../_webgpu.py") is None


@contextmanager
def running_server():
    server = create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def post(base: str, path: str, payload: dict, token: str | None = None) -> Request:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Jaxcad-Token"] = token
    return Request(base + path, data=json.dumps(payload).encode(), headers=headers, method="POST")


def test_session_endpoint_hands_out_a_token_and_the_example():
    with running_server() as base:
        with urlopen(f"{base}/api/session") as response:
            session = json.loads(response.read())

    assert session["ok"] is True
    assert len(session["token"]) > 20
    assert session["example"] == EXAMPLE_SOURCE


@pytest.mark.parametrize("path", ["/compile", "/patch"])
def test_write_endpoints_require_the_session_token(path):
    with running_server() as base:
        with pytest.raises(HTTPError) as error:
            urlopen(post(base, path, {"source": "scene = None"}))
        assert error.value.code == 403


def test_patch_endpoint_edits_the_program_text():
    with running_server() as base:
        with urlopen(f"{base}/api/session") as response:
            token = json.loads(response.read())["token"]
        with urlopen(
            post(
                base,
                "/patch",
                {
                    "source": EXAMPLE_SOURCE,
                    "op": "insert_vertex",
                    "line": 10,
                    "index": 5,
                    "xy": [0.4, 0.8],
                },
                token,
            )
        ) as response:
            result = json.loads(response.read())

    assert result["ok"] is True
    assert "[0.4, 0.8]" in result["source"]


def test_patch_endpoint_reports_failures_without_crashing():
    with running_server() as base:
        with urlopen(f"{base}/api/session") as response:
            token = json.loads(response.read())["token"]
        with pytest.raises(HTTPError) as error:
            urlopen(
                post(
                    base,
                    "/patch",
                    {
                        "source": EXAMPLE_SOURCE,
                        "op": "set_vertex",
                        "line": 1,
                        "index": 0,
                        "xy": [0, 0],
                    },
                    token,
                )
            )
        assert error.value.code == 422
        assert "No editable PolygonProfile" in json.loads(error.value.read())["error"]


def test_unknown_host_header_is_rejected():
    with running_server() as base:
        request = Request(f"{base}/api/session", headers={"Host": "evil.example.com"})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
