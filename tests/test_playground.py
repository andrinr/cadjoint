from __future__ import annotations

import ast
import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
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
    list_scenes,
    load_scene,
    mesh_source,
    patch_source,
    resolve_static,
    sanitize_scene_name,
    save_scene,
)


def call_line(source: str, name: str) -> int:
    """Line of the first bare or attribute-qualified call in a test program."""
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) == name or getattr(node.func, "attr", None) == name)
    ]
    assert calls
    return min(calls, key=lambda call: call.lineno).lineno


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


def test_compile_no_longer_pays_for_mesh_edges():
    # Mesh extraction used to dominate the compile round-trip; it now runs
    # only through the lazy `/api/mesh` request.
    result = compile_source(EXAMPLE_SOURCE)

    assert result["ok"] is True
    assert result["mesh_edges"] is None


def test_mesh_endpoint_reports_mesh_edges_for_the_viewer():
    result = mesh_source(EXAMPLE_SOURCE)

    assert result["ok"] is True
    mesh_edges = result["mesh_edges"]
    assert mesh_edges is not None
    assert mesh_edges["resolution"] >= 8
    for group in ("wire", "sharp"):
        assert len(mesh_edges[group]) > 0
        for segment in mesh_edges[group][:16]:
            assert len(segment) == 2
            for point in segment:
                assert len(point) == 3
                assert all(isinstance(value, float) for value in point)
                assert all(-3.0 <= value <= 3.0 for value in point)


def test_thin_slab_sharp_edges_stay_on_their_own_rim():
    # A slab thinner than two grid cells puts its top and bottom crease
    # rails in vertically adjacent cells, so neighbor linking offers
    # diagonal rail-to-rail links; the tangent-alignment filter must drop
    # them.  Only the four short true corner edges of the box may remain
    # near-vertical, so the cross-rail fraction stays tiny.
    source = (
        "from jaxcad.geometry import Vector\n"
        "from jaxcad.sdf.primitives import Box\n"
        "\n"
        "scene = Box(size=Vector([0.9, 0.9, 0.08]))\n"
    )

    result = mesh_source(source)

    assert result["ok"] is True
    sharp = np.asarray(result["mesh_edges"]["sharp"], dtype=float)
    assert sharp.shape[0] > 0
    directions = sharp[:, 1] - sharp[:, 0]
    lengths = np.maximum(np.linalg.norm(directions, axis=1), 1e-9)
    cross_rail = np.abs(directions[:, 2]) / lengths > 0.7
    assert cross_rail.mean() < 0.05


def test_compile_source_reports_missing_scene():
    result = compile_source("answer = 42")

    assert result["ok"] is False
    assert "variable named `scene`" in result["error"]


def test_compile_source_reports_profile_solver_loss_history():
    source = (
        "from jaxcad.construction import PolygonProfile, extrude\n"
        "from jaxcad.constraints import DistanceConstraint, satisfy_constraints\n"
        "profile = PolygonProfile([[0, 0], [2, 0], [0, 1]], name='solved')\n"
        "DistanceConstraint(profile.vertices[0], profile.vertices[1], 1.0)\n"
        "satisfy_constraints(profile, method='newton', steps=2)\n"
        "scene = extrude(profile, depth=0.5)\n"
    )

    result = compile_source(source)

    assert result["ok"] is True
    run = result["solver_runs"][0]
    assert run["node"] == "profile_0"
    assert run["method"] == "newton"
    assert run["iterations"] == 2
    assert len(run["losses"]) == 3
    assert run["losses"][-1] < run["losses"][0]


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
    assert "display_flag(DISPLAY_HIDE_SOLID)" in shader
    assert "display_flag(DISPLAY_FLAT)" in shader
    assert "display_flag(DISPLAY_REFLECTIONS)" in shader
    assert "let xray = clamp(u.display.w" in shader
    assert "previous_accumulation: texture_2d<f32>" in shader
    assert "@fragment\nfn fs_path_trace(" in shader
    assert "fn fs_present(" in WGSL_PRESENT_TEMPLATE


def test_path_tracer_builder_rejects_reserved_marker():
    with pytest.raises(ValueError, match="reserved marker"):
        build_path_tracer_shader("// __JAXCAD_SCENE_CODE__")


def test_example_scene_reports_its_construction_for_the_viewer():
    result = compile_source(EXAMPLE_SOURCE)

    assert result["ok"] is True
    profiles = {node["name"]: node for node in result["construction"] if node["kind"] == "profile"}
    nodes = {node["kind"]: node for node in result["construction"] if node["kind"] != "profile"}
    # Extrude and revolve sketches sit alongside two editable primitives.
    assert set(profiles) == {"house", "revolve section"}
    assert set(nodes) == {"sphere", "cylinder"}

    profile = profiles["house"]
    assert profile["editable"] is True
    assert profile["name"] == "house"
    assert len(profile["vertices"]) == 5
    # Every vertex carries the span the viewer highlights and patches.
    for vertex in profile["vertices"]:
        start, end = vertex["span"]
        assert EXAMPLE_SOURCE[start:end].startswith("[")
    assert [item["kind"] for item in profile["constraints"]] == [
        "fixed",
        "distance",
        "distance",
    ]
    assert profile["material"] == "clay"

    revolve_profile = profiles["revolve section"]
    assert revolve_profile["operators"] == [
        {"kind": "revolve", "line": call_line(EXAMPLE_SOURCE, "revolve")}
    ]
    assert revolve_profile["material"] == "polished_copper"
    assert revolve_profile["transform"]["position"] == pytest.approx([0.0, 1.65, 0.15], abs=1e-6)

    glass = nodes["sphere"]
    assert glass["editable"] is True
    metal_position = nodes["cylinder"]["transform"]["position"]
    distance = (
        sum((glass["transform"]["position"][axis] - metal_position[axis]) ** 2 for axis in range(3))
        ** 0.5
    )
    assert distance == pytest.approx(3.8, abs=1e-5)
    assert glass["transform"]["position"] != pytest.approx([2.2, -0.3, 0.35], abs=1e-3)
    assert glass["material"] == "glass_material"
    # A wireframe the viewer can draw without knowing the shape's topology.
    assert len(glass["edges"]) > 0
    start, end = glass["spans"]["position"]
    assert EXAMPLE_SOURCE[start:end] == "[2.2, -0.3, 0.35]"

    metal = nodes["cylinder"]
    assert metal["transform"]["rotation"] == pytest.approx([1.5708, 0.0, 0.0], abs=1e-5)
    assert "rotation" in metal["spans"]
    assert metal["material"] == "brass"

    materials = {material["name"]: material for material in result["materials"]}
    assert set(materials) == {
        "clay",
        "glass_material",
        "brass",
        "polished_copper",
        "ground",
    }
    assert materials["brass"]["metallic"] == pytest.approx(1.0)
    assert materials["glass_material"]["opacity"] == pytest.approx(0.18)
    assert materials["polished_copper"]["metallic"] == pytest.approx(0.92)

    assert result["solver_runs"] == [
        {
            "node": None,
            "method": "newton",
            "iterations": 2,
            "losses": pytest.approx([0.0135935191, 0.0, 0.0], abs=1e-6),
        }
    ]

    autodiff = result["differentiability"]
    assert autodiff["pipeline"] == "Profile -> Extrude -> SDF"
    assert autodiff["metric"] == "two-probe clearance"
    assert autodiff["parameter_count"] == 6
    assert autodiff["value"] == pytest.approx(0.85)
    assert autodiff["sensitivities"] == [
        {"parameter": "eave_right.x", "value": pytest.approx(-0.7)},
        {"parameter": "body_depth", "value": pytest.approx(-0.5)},
    ]

    assert result["relations"] == [
        {
            "kind": "distance",
            "nodes": ["cylinder_2", "sphere_1"],
            "value": pytest.approx(3.8),
        },
        {
            "kind": "fixed",
            "nodes": ["cylinder_2"],
            "value": pytest.approx([-1.9, -0.65, 0.0]),
        },
    ]

    assert "fn fs_main_depth(" in result["preview_shader"]


def test_patch_source_round_trips_through_compile():
    edited = patch_source(
        {
            "source": EXAMPLE_SOURCE,
            "op": "set_vertex",
            "line": call_line(EXAMPLE_SOURCE, "PolygonProfile"),
            "index": 3,
            "xy": [-0.2, 1.25],
        }
    )
    assert edited["ok"] is True

    result = compile_source(edited["source"])
    assert result["ok"] is True
    assert result["construction"][0]["vertices"][3]["uv"] == pytest.approx([-0.2, 1.25], abs=1e-6)


def test_material_patches_create_and_assign_source_definitions():
    created = patch_source(
        {
            "source": EXAMPLE_SOURCE,
            "op": "add_material",
            "color": [0.2, 0.4, 0.8],
            "roughness": 0.25,
        }
    )
    assert created["ok"] is True
    assert "material1 = Material(" in created["source"]

    assigned = patch_source(
        {
            "source": created["source"],
            "op": "assign_material",
            "line": call_line(created["source"], "sphere"),
            "material": "material1",
        }
    )
    assert assigned["ok"] is True
    assert "material=material1" in assigned["source"]

    result = compile_source(assigned["source"])
    assert result["ok"] is True
    sphere = next(node for node in result["construction"] if node["kind"] == "sphere")
    assert sphere["material"] == "material1"


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


@pytest.mark.parametrize(
    "path", ["/compile", "/patch", "/api/mesh", "/api/scenes/load", "/api/scenes/save"]
)
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
                    "line": call_line(EXAMPLE_SOURCE, "PolygonProfile"),
                    "index": 5,
                    "xy": [0.4, 0.8],
                },
                token,
            )
        ) as response:
            result = json.loads(response.read())

    assert result["ok"] is True
    assert "[0.4, 0.8]" in result["source"]


def test_solver_patch_endpoint_forwards_optimizer_controls():
    result = patch_source(
        {
            "source": EXAMPLE_SOURCE,
            "op": "solve_sketch",
            "line": call_line(EXAMPLE_SOURCE, "PolygonProfile"),
            "method": "adam",
            "iterations": 24,
        }
    )

    assert result["ok"] is True
    assert "satisfy_constraints(profile, method='adam', steps=24)" in result["source"]


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


def test_scene_files_round_trip_in_the_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # The directory is created lazily, so listing before any save is empty.
    assert list_scenes() == {"ok": True, "files": []}

    saved = save_scene({"name": "bracket.py", "source": "scene = None\n"})
    assert saved == {"ok": True, "name": "bracket.py"}
    assert (tmp_path / "scenes" / "bracket.py").read_text() == "scene = None\n"
    assert list_scenes() == {"ok": True, "files": ["bracket.py"]}

    loaded = load_scene({"name": "bracket.py"})
    assert loaded["ok"] is True
    assert loaded["source"] == "scene = None\n"


@pytest.mark.parametrize(
    "name",
    [
        "../evil.py",
        "..\\evil.py",
        "/etc/passwd.py",
        "nested/evil.py",
        ".hidden.py",
        "..py",
        "evil.txt",
        "",
        7,
        None,
    ],
)
def test_scene_requests_reject_bad_names(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)

    assert sanitize_scene_name(name) is None
    assert save_scene({"name": name, "source": "scene = None\n"})["ok"] is False
    assert load_scene({"name": name})["ok"] is False
    # Nothing may leak outside the workspace, and the lazy directory itself
    # is only created by a valid save.
    assert not (tmp_path / "scenes").exists()
    assert not (tmp_path / "evil.py").exists()


def test_scene_load_reports_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = load_scene({"name": "missing.py"})
    assert result["ok"] is False
    assert "missing.py" in result["error"]


def test_scene_save_rejects_oversized_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = save_scene({"name": "big.py", "source": "x" * 200_000})
    assert result["ok"] is False
    assert "limit" in result["error"]


def test_scene_endpoints_round_trip_over_http(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with running_server() as base:
        with urlopen(f"{base}/api/session") as response:
            token = json.loads(response.read())["token"]

        with urlopen(
            post(base, "/api/scenes/save", {"name": "part.py", "source": "scene = None\n"}, token)
        ) as response:
            assert json.loads(response.read()) == {"ok": True, "name": "part.py"}

        with urlopen(f"{base}/api/scenes") as response:
            assert json.loads(response.read()) == {"ok": True, "files": ["part.py"]}

        with urlopen(post(base, "/api/scenes/load", {"name": "part.py"}, token)) as response:
            loaded = json.loads(response.read())
        assert loaded["ok"] is True
        assert loaded["source"] == "scene = None\n"

        # Path traversal is refused with a client error, not a write.
        with pytest.raises(HTTPError) as error:
            urlopen(post(base, "/api/scenes/save", {"name": "../evil.py", "source": "x"}, token))
        assert 400 <= error.value.code < 500
    assert not (tmp_path / "evil.py").exists()


def test_patch_rejects_an_operation_this_server_does_not_know():
    # A browser running newer assets than the server used to get a confusing
    # complaint about a missing `line` instead of the real problem.
    result = patch_source({"source": "x = 1", "op": "teleport", "line": 1, "index": 0})

    assert result["ok"] is False
    assert "does not support the patch operation" in result["error"]
    assert "restart" in result["error"]


# ── Simulation studies as first-class code citizens ─────────────────────────

STUDY_SOURCE = """from jaxcad.fem import Dirichlet, Nodes, ThermalStudy
from jaxcad.geometry import Vector
from jaxcad.sdf.primitives import Box

scene = Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))
heat = ThermalStudy(
    name="bar-conduction",
    resolution=10,
    conductivity=2.0,
    bcs=[Dirichlet(Nodes.side("-x"), value=1.0), Dirichlet(Nodes.side("+x"), value=0.0)],
)
"""


def test_compile_reports_declared_studies_for_the_viewer():
    result = compile_source(STUDY_SOURCE)

    assert result["ok"] is True
    studies = result["studies"]
    assert len(studies) == 1
    study = studies[0]
    assert study["name"] == "bar-conduction"
    assert study["kind"] == "thermal"
    assert study["index"] == 0
    assert study["editable"] is True
    assert study["line"] == call_line(STUDY_SOURCE, "ThermalStudy")
    assert study["resolution"] == 10
    assert study["material"] == {"conductivity": 2.0}
    start, end = study["span"]
    assert STUDY_SOURCE[start:end].startswith("ThermalStudy(")
    assert [bc["type"] for bc in study["bcs"]] == ["dirichlet", "dirichlet"]
    for bc in study["bcs"]:
        assert bc["serializable"] is True
        assert bc["nodes"]["kind"] == "side"
        start, end = bc["span"]
        assert STUDY_SOURCE[start:end].startswith("Dirichlet(")


def test_compile_reports_a_scene_without_studies_as_an_empty_list():
    result = compile_source("from jaxcad.sdf.primitives import Sphere\nscene = Sphere(1.0)\n")

    assert result["ok"] is True
    assert result["studies"] == []


def test_study_patches_round_trip_through_compile():
    added = patch_source(
        {
            "source": STUDY_SOURCE,
            "op": "add_study_bc",
            "study": "bar-conduction",
            "bc_type": "dirichlet",
            "selection": {
                "kind": "box",
                "min_corner": [0.0, 0.0, 0.0],
                "max_corner": [1.0, 1.0, 1.0],
            },
            "value": 300.0,
        }
    )
    assert added["ok"] is True
    assert "Dirichlet(Nodes.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), value=300.0)" in added["source"]

    result = compile_source(added["source"])
    assert result["ok"] is True
    bcs = result["studies"][0]["bcs"]
    assert len(bcs) == 3
    assert bcs[2]["nodes"] == {
        "kind": "box",
        "min_corner": [0.0, 0.0, 0.0],
        "max_corner": [1.0, 1.0, 1.0],
    }
    assert bcs[2]["value"] == 300.0


@pytest.mark.parametrize(
    ("request_body", "message"),
    [
        ({"op": "add_study", "kind": "modal"}, "thermal"),
        ({"op": "add_study", "kind": "thermal", "name": "  "}, "non-empty"),
        ({"op": "delete_study"}, "`study`"),
        ({"op": "delete_study", "study": -1}, "`study`"),
        ({"op": "delete_study_bc", "study": 0, "bc": "first"}, "`bc`"),
        ({"op": "add_study_bc", "study": 0, "bc_type": "neumann"}, "bc_type"),
        (
            {"op": "add_study_bc", "study": 0, "bc_type": "dirichlet", "selection": "box"},
            "description object",
        ),
        (
            {
                "op": "add_study_bc",
                "study": 0,
                "bc_type": "dirichlet",
                "selection": {"kind": "side", "side": "-x"},
                "value": "hot",
            },
            "numeric `value`",
        ),
        (
            {
                "op": "add_study_bc",
                "study": 0,
                "bc_type": "traction",
                "selection": {"kind": "side", "side": "-x"},
                "value": 1.0,
            },
            "three numbers",
        ),
        ({"op": "set_study_value", "study": 0, "value": 1.0}, "exactly one"),
        (
            {"op": "set_study_value", "study": 0, "bc": 0, "argument": "source", "value": 1.0},
            "exactly one",
        ),
        ({"op": "set_study_value", "study": 0, "bc": 0, "value": "hot"}, "`value`"),
    ],
)
def test_patch_source_validates_study_requests(request_body, message):
    result = patch_source({"source": STUDY_SOURCE, **request_body})

    assert result["ok"] is False
    assert message in result["error"]


def test_simulate_study_requires_a_declared_study_name():
    from jaxcad.viewer.playground import simulate_source

    result = simulate_source({"source": STUDY_SOURCE, "kind": "study"})
    assert result["ok"] is False
    assert "name" in result["error"]

    result = simulate_source({"source": STUDY_SOURCE, "kind": "study", "name": "   "})
    assert result["ok"] is False
    assert "name" in result["error"]
