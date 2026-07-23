from __future__ import annotations

import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from jaxcad.viewer._pathtracer import (
    WGSL_PRESENT_TEMPLATE,
    build_path_tracer_shader,
)
from jaxcad.viewer._webgpu import build_viewer_shader, ensure_material_wgsl
from jaxcad.viewer.playground import EXAMPLE_SOURCE, compile_source, create_server


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
    assert "fn sample_opaque_bsdf(" in shader
    assert "previous_accumulation: texture_2d<f32>" in shader
    assert "@fragment\nfn fs_path_trace(" in shader
    assert "fn fs_present(" in WGSL_PRESENT_TEMPLATE


def test_path_tracer_builder_rejects_reserved_marker():
    with pytest.raises(ValueError, match="reserved marker"):
        build_path_tracer_shader("// __JAXCAD_SCENE_CODE__")


def test_playground_serves_page_and_rejects_requests_without_token():
    server = create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urlopen(f"http://{host}:{port}/") as response:
            page = response.read().decode("utf-8")
        assert "JAXCAD Playground" in page
        assert EXAMPLE_SOURCE.splitlines()[0] in page

        request = Request(
            f"http://{host}:{port}/compile",
            data=b'{"source":"scene = None"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
