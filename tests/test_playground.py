from __future__ import annotations

import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from jaxcad.viewer._webgpu import build_viewer_shader
from jaxcad.viewer.playground import EXAMPLE_SOURCE, compile_source, create_server


def test_example_scene_compiles_to_complete_webgpu_shader():
    result = compile_source(EXAMPLE_SOURCE)

    assert result["ok"] is True
    assert "fn sdf(" in result["sdf"]
    assert "@vertex" in result["shader"]
    assert "@fragment" in result["shader"]


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
