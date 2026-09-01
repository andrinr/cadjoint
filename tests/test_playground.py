from __future__ import annotations

import ast
import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from cadjoint.viewer._pathtracer import (
    WGSL_PRESENT_TEMPLATE,
    build_path_tracer_shader,
)
from cadjoint.viewer._webgpu import build_viewer_shader, ensure_material_wgsl
from cadjoint.viewer.playground import (
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
    # Observability: which dual-contouring backend produced the edges.
    assert isinstance(mesh_edges["native"], bool)
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
        "from cadjoint.geometry import Vector\n"
        "from cadjoint.sdf.primitives import Box\n"
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
        "from cadjoint.construction import PolygonProfile, extrude\n"
        "from cadjoint.constraints import DistanceConstraint, satisfy_constraints\n"
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
        build_viewer_shader("fn sdf() {} // __CADJOINT_SDF_CODE__")


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
        build_path_tracer_shader("// __CADJOINT_SCENE_CODE__")


def test_example_scene_reports_its_construction_for_the_viewer():
    result = compile_source(EXAMPLE_SOURCE)

    assert result["ok"] is True
    profiles = {node["name"]: node for node in result["construction"] if node["kind"] == "profile"}
    nodes = {node["name"]: node for node in result["construction"] if node["kind"] != "profile"}
    # Extrude and revolve sketches sit alongside the two bushing cylinders.
    assert set(profiles) == {"fin comb", "slug section"}
    assert set(nodes) == {"bush_a", "bush_b"}

    profile = profiles["fin comb"]
    assert profile["editable"] is True
    assert profile["name"] == "fin comb"
    assert len(profile["vertices"]) == 16
    # Every vertex carries the span the viewer highlights and patches.
    for vertex in profile["vertices"]:
        start, end = vertex["span"]
        assert EXAMPLE_SOURCE[start:end].startswith("[")
    # The comb is dimension-driven: the full realistic constraint set rides
    # along, in statement order, so the viewer can chip and overlay it.
    from collections import Counter

    kinds = Counter(item["kind"] for item in profile["constraints"])
    assert kinds == {
        "fixed": 1,
        "distance": 1,
        "horizontal": 11,
        "vertical": 8,
        "equal_length": 2,
    }
    # The one distance constraint carries its driving-dimension value (the
    # viewport overlay); relational constraints carry value: None.
    distances = [item for item in profile["constraints"] if item["kind"] == "distance"]
    assert [item["value"] for item in distances] == pytest.approx([1.8])
    assert all(
        item["value"] is None
        for item in profile["constraints"]
        if item["kind"] in {"horizontal", "vertical", "equal_length"}
    )
    assert profile["material"] == "aluminum"

    slug_profile = profiles["slug section"]
    assert slug_profile["operators"] == [
        {"kind": "revolve", "line": call_line(EXAMPLE_SOURCE, "revolve")}
    ]
    assert slug_profile["material"] == "copper"
    assert slug_profile["transform"]["position"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    bush_a = nodes["bush_a"]
    assert bush_a["kind"] == "cylinder"
    assert bush_a["editable"] is True
    # The bushing pattern is constraint-driven: one pinned, the other held at
    # the named spacing.
    assert bush_a["transform"]["position"] == pytest.approx([0.78, 0.0, 0.1], abs=1e-5)
    bush_b_position = nodes["bush_b"]["transform"]["position"]
    distance = (
        sum(
            (bush_a["transform"]["position"][axis] - bush_b_position[axis]) ** 2
            for axis in range(3)
        )
        ** 0.5
    )
    assert distance == pytest.approx(1.56, abs=1e-5)
    assert bush_a["material"] == "steel"
    # A wireframe the viewer can draw without knowing the shape's topology.
    assert len(bush_a["edges"]) > 0
    # Named-parameter indirection: the position span points at the Vector literal.
    start, end = bush_a["spans"]["position"]
    assert EXAMPLE_SOURCE[start:end] == "[0.78, 0.0, 0.1]"

    materials = {material["name"]: material for material in result["materials"]}
    assert set(materials) == {"aluminum", "copper", "steel"}
    assert materials["aluminum"]["metallic"] == pytest.approx(0.9)
    assert materials["copper"]["roughness"] == pytest.approx(0.18)
    assert materials["steel"]["metallic"] == pytest.approx(0.85)

    # The starter's constraints are satisfied by construction, so the solve
    # converges instantly with a flat loss history.
    assert result["solver_runs"] == [
        {
            "node": None,
            "method": "newton",
            "iterations": 2,
            "losses": pytest.approx([0.0, 0.0, 0.0], abs=1e-9),
        }
    ]

    # The starter declares ONE optimization — the study-backed showcase:
    # the declared thermal study is the objective, with the material volume
    # riding along as regularizer.  Nothing descends at compile time.
    assert "differentiability" not in result
    optimizations = result["optimizations"]
    assert len(optimizations) == 1
    cool = optimizations[0]
    assert cool["kind"] == "optimization"
    assert cool["name"] == "cool-sink"
    assert cool["objective"] == "max(sink-conduction)"
    assert cool["study"] == "sink-conduction"
    assert cool["metric"] == "max"
    assert cool["remesh_every"] == 6
    assert cool["regularizer"] == "material_volume"
    assert cool["regularizer_weight"] == pytest.approx(0.4)
    assert cool["steps"] == 12
    assert cool["learning_rate"] == pytest.approx(0.004)
    assert cool["method"] == "adam"
    assert cool["index"] == 0
    assert cool["editable"] is True
    # No declared domain: the whole scene's free parameters are the design.
    assert "fin_depth" in cool["parameters"]
    assert len(cool["parameters"]) > 17
    start, end = cool["span"]
    assert EXAMPLE_SOURCE[start:end].startswith("Optimization(")
    start, end = cool["steps_span"]
    assert EXAMPLE_SOURCE[start:end] == "12"
    start, end = cool["learning_rate_span"]
    assert EXAMPLE_SOURCE[start:end] == "0.004"

    assert result["relations"] == [
        {
            "kind": "fixed",
            "nodes": ["cylinder_2"],
            "value": pytest.approx([0.78, 0.0, 0.1], abs=1e-6),
        },
        {
            "kind": "distance",
            "nodes": ["cylinder_2", "cylinder_3"],
            "value": pytest.approx(1.56, abs=1e-6),
        },
    ]

    # The starter declares its thermal study on an explicit named SimMesh —
    # the boundary-conforming quadratic tet path.
    studies = result["studies"]
    assert len(studies) == 1
    assert studies[0]["name"] == "sink-conduction"
    assert studies[0]["kind"] == "thermal"
    assert studies[0]["editable"] is True
    assert studies[0]["resolution"] == [18, 13, 11]
    assert studies[0]["mesh"] == "sink-mesh"
    meshes = result["sim_meshes"]
    assert len(meshes) == 1
    assert meshes[0]["name"] == "sink-mesh"
    assert meshes[0]["method"] == "tet10"
    assert meshes[0]["resolution"] == [18, 13, 11]
    assert meshes[0]["editable"] is True

    assert "fn fs_main_depth(" in result["preview_shader"]


def test_patch_source_round_trips_through_compile():
    # The slug profile is the unconstrained sketch: a patched vertex must
    # round-trip exactly. (The comb profile is dimension-driven — a moved
    # vertex there gets projected back onto its constraints on execution.)
    slug_line = max(
        node.lineno
        for node in ast.walk(ast.parse(EXAMPLE_SOURCE))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "PolygonProfile"
    )
    edited = patch_source(
        {
            "source": EXAMPLE_SOURCE,
            "op": "set_vertex",
            "line": slug_line,
            "index": 2,
            "xy": [0.3, 0.1],
        }
    )
    assert edited["ok"] is True

    result = compile_source(edited["source"])
    assert result["ok"] is True
    assert result["construction"][1]["vertices"][2]["uv"] == pytest.approx([0.3, 0.1], abs=1e-6)


def test_comb_vertex_patches_project_back_onto_the_constraints():
    # Dragging a constrained comb vertex is allowed — the program's
    # satisfy_constraints() call projects it back onto the sketch system on
    # the next compile, so the deck level survives the edit.
    edited = patch_source(
        {
            "source": EXAMPLE_SOURCE,
            "op": "set_vertex",
            "line": call_line(EXAMPLE_SOURCE, "PolygonProfile"),
            "index": 3,
            "xy": [0.66, 0.3],
        }
    )
    assert edited["ok"] is True
    assert "[0.66, 0.3]" in edited["source"]

    result = compile_source(edited["source"])
    assert result["ok"] is True
    # fin1_root_r (vertex 3) is pulled back to the deck line by projection.
    uv = result["construction"][0]["vertices"][3]["uv"]
    assert uv[1] == pytest.approx(result["construction"][0]["vertices"][2]["uv"][1], abs=5e-2)


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
            "line": call_line(created["source"], "cylinder"),
            "material": "material1",
        }
    )
    assert assigned["ok"] is True
    assert "material=material1" in assigned["source"]

    result = compile_source(assigned["source"])
    assert result["ok"] is True
    bush = next(node for node in result["construction"] if node.get("name") == "bush_a")
    assert bush["material"] == "material1"


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
        headers["X-Cadjoint-Token"] = token
    return Request(base + path, data=json.dumps(payload).encode(), headers=headers, method="POST")


def test_session_endpoint_hands_out_a_token_and_the_example():
    with running_server() as base:
        with urlopen(f"{base}/api/session") as response:
            session = json.loads(response.read())

    assert session["ok"] is True
    assert len(session["token"]) > 20
    assert session["example"] == EXAMPLE_SOURCE


@pytest.mark.parametrize(
    "path",
    ["/compile", "/patch", "/api/mesh", "/api/optimize", "/api/scenes/load", "/api/scenes/save"],
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
    assert "satisfy_constraints(comb_profile, method='adam', steps=24)" in result["source"]


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

STUDY_SOURCE = """from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
from cadjoint.geometry import Vector
from cadjoint.sdf.primitives import Box

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
    result = compile_source("from cadjoint.sdf.primitives import Sphere\nscene = Sphere(1.0)\n")

    assert result["ok"] is True
    assert result["studies"] == []
    assert result["sim_meshes"] == []


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
    from cadjoint.viewer.playground import simulate_source

    result = simulate_source({"source": STUDY_SOURCE, "kind": "study"})
    assert result["ok"] is False
    assert "name" in result["error"]

    result = simulate_source({"source": STUDY_SOURCE, "kind": "study", "name": "   "})
    assert result["ok"] is False
    assert "name" in result["error"]


# ── Simulation meshes as first-class code citizens ──────────────────────────

MESH_SOURCE = """from cadjoint.fem import Dirichlet, Nodes, SimMesh, ThermalStudy
from cadjoint.geometry import Vector
from cadjoint.sdf.primitives import Box

scene = Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))
grid = SimMesh(name="box-grid", resolution=[12, 8, 8], bounds=[-1.0, -0.75, -0.75],
               size=[2.0, 1.5, 1.5], padding=0.2)
heat = ThermalStudy(
    name="bar-conduction",
    conductivity=2.0,
    mesh="box-grid",
    bcs=[Dirichlet(Nodes.side("-x"), value=1.0), Dirichlet(Nodes.side("+x"), value=0.0)],
)
"""


def test_mesh_endpoint_handles_scenes_that_reference_meshes_by_name():
    # `mesh="..."` resolves through the capture registry at construction
    # time, so every worker mode must exec inside it — not just compiles.
    result = mesh_source(MESH_SOURCE)

    assert result["ok"] is True
    assert result["mesh_edges"] is not None


def test_compile_reports_declared_meshes_for_the_viewer():
    result = compile_source(MESH_SOURCE)

    assert result["ok"] is True
    meshes = result["sim_meshes"]
    assert len(meshes) == 1
    mesh = meshes[0]
    assert mesh["kind"] == "mesh"
    assert mesh["name"] == "box-grid"
    assert mesh["index"] == 0
    assert mesh["editable"] is True
    assert mesh["line"] == call_line(MESH_SOURCE, "SimMesh")
    assert mesh["resolution"] == [12, 8, 8]
    assert mesh["bounds"] == [-1.0, -0.75, -0.75]
    assert mesh["size"] == [2.0, 1.5, 1.5]
    assert mesh["padding"] == 0.2
    assert mesh["domain"] is None
    start, end = mesh["span"]
    assert MESH_SOURCE[start:end].startswith("SimMesh(")

    # The study references the mesh; its mesh= span is editable.
    study = result["studies"][0]
    assert study["mesh"] == "box-grid"
    assert study["resolution"] == [12, 8, 8]
    start, end = study["mesh_span"]
    assert MESH_SOURCE[start:end] == '"box-grid"'
    assert study["domain_span"] is None


def test_mesh_patches_round_trip_through_compile():
    added = patch_source({"source": STUDY_SOURCE, "op": "add_mesh", "name": "grid"})
    assert added["ok"] is True
    assert "mesh1 = SimMesh(name='grid', resolution=20)" in added["source"]
    # Declared before the study so the study can reference it by name.
    assert added["source"].index("SimMesh(") < added["source"].index("ThermalStudy(")

    sized = patch_source(
        {
            "source": added["source"],
            "op": "set_mesh_value",
            "mesh": "grid",
            "argument": "resolution",
            "value": [10, 6, 6],
        }
    )
    assert sized["ok"] is True
    assert "resolution=[10, 6, 6]" in sized["source"]

    wired = patch_source(
        {
            "source": sized["source"],
            "op": "set_study_value",
            "study": "bar-conduction",
            "argument": "mesh",
            "value": "grid",
        }
    )
    assert wired["ok"] is True
    assert "mesh='grid'" in wired["source"]

    result = compile_source(wired["source"])
    assert result["ok"] is True
    assert [mesh["name"] for mesh in result["sim_meshes"]] == ["grid"]
    assert result["studies"][0]["mesh"] == "grid"
    assert result["studies"][0]["resolution"] == [10, 6, 6]

    # The referenced mesh cannot be deleted until the study lets go of it.
    refused = patch_source({"source": wired["source"], "op": "delete_mesh", "mesh": "grid"})
    assert refused["ok"] is False
    assert "referenced" in refused["error"]

    deleted = patch_source({"source": sized["source"], "op": "delete_mesh", "mesh": "grid"})
    assert deleted["ok"] is True
    assert "SimMesh(" not in deleted["source"].split("import")[-1].split("\n", 1)[1]


@pytest.mark.parametrize(
    ("request_body", "message"),
    [
        ({"op": "add_mesh", "name": "  "}, "non-empty"),
        ({"op": "delete_mesh"}, "`mesh`"),
        ({"op": "delete_mesh", "mesh": -1}, "`mesh`"),
        ({"op": "set_mesh_value", "mesh": 0, "argument": 7, "value": 1.0}, "`argument`"),
        ({"op": "set_mesh_value", "mesh": 0, "argument": "resolution", "value": "ten"}, "`value`"),
        (
            {"op": "set_mesh_value", "mesh": 0, "argument": "domain", "value": 3.0},
            "`domain` name",
        ),
        (
            {"op": "set_mesh_value", "mesh": 0, "argument": "method", "value": "voxel"},
            "hex, tet4, tet10",
        ),
        (
            {"op": "set_study_value", "study": 0, "argument": "mesh", "value": 3.0},
            "`mesh` name",
        ),
        (
            {"op": "set_study_value", "study": 0, "argument": "domain", "value": ""},
            "`domain` name",
        ),
    ],
)
def test_patch_source_validates_mesh_requests(request_body, message):
    result = patch_source({"source": MESH_SOURCE, **request_body})

    assert result["ok"] is False
    assert message in result["error"]


def test_mesh_method_patches_write_a_string_literal():
    # Text surgery only: the declared enum is validated here, the literal is
    # written as a keyword, and nothing executes.
    patched = patch_source(
        {
            "source": MESH_SOURCE,
            "op": "set_mesh_value",
            "mesh": "box-grid",
            "argument": "method",
            "value": "tet4",
        }
    )

    assert patched["ok"] is True
    assert "method='tet4'" in patched["source"]


# ── Optimizations as first-class code citizens ──────────────────────────────

OPTIMIZE_SOURCE = """from cadjoint.geometry import Scalar
from cadjoint.optimize import Optimization
from cadjoint.sdf.primitives import Sphere

radius = Scalar(0.8, free=True, name="radius")
scene = Sphere(radius)


def fit(params):
    return (params["radius"] - 0.25) ** 2


shrink = Optimization(name="fit-radius", objective=fit, of=scene, steps=4, learning_rate=0.1)
"""


def test_compile_reports_declared_optimizations_for_the_viewer():
    result = compile_source(OPTIMIZE_SOURCE)

    assert result["ok"] is True
    optimizations = result["optimizations"]
    assert len(optimizations) == 1
    optimization = optimizations[0]
    assert optimization["kind"] == "optimization"
    assert optimization["name"] == "fit-radius"
    assert optimization["objective"] == "fit"
    assert optimization["parameters"] == ["radius"]
    assert optimization["steps"] == 4
    assert optimization["learning_rate"] == pytest.approx(0.1)
    assert optimization["editable"] is True
    assert optimization["line"] == call_line(OPTIMIZE_SOURCE, "Optimization")
    start, end = optimization["span"]
    assert OPTIMIZE_SOURCE[start:end].startswith("Optimization(")


def test_compile_reports_a_scene_without_optimizations_as_an_empty_list():
    result = compile_source("from cadjoint.sdf.primitives import Sphere\nscene = Sphere(1.0)\n")

    assert result["ok"] is True
    assert result["optimizations"] == []


def test_optimize_endpoint_descends_and_patches_the_source_back():
    from cadjoint.viewer.playground import optimize_source

    result = optimize_source({"source": OPTIMIZE_SOURCE, "name": "fit-radius"})

    assert result["ok"] is True
    assert result["kind"] == "optimize"
    assert result["name"] == "fit-radius"
    assert result["steps"] == 4
    assert "output" in result

    # The differentiable path descends: four steps, monotone here.
    assert len(result["history"]) == 4
    assert result["history"][-1]["objective"] < result["history"][0]["objective"]
    assert all(record["grad_norm"] >= 0.0 for record in result["history"])

    # The trajectory replays the parameter path from the initial state.
    assert len(result["trajectory"]) == 5
    assert result["trajectory"][0]["step"] == 0
    assert result["trajectory"][0]["parameters"] == result["initial"]
    assert result["trajectory"][-1]["parameters"] == result["parameters"]

    assert result["initial"]["radius"] == pytest.approx(0.8, abs=1e-6)
    optimized = result["parameters"]["radius"]
    assert abs(optimized - 0.25) < abs(0.8 - 0.25)

    # Writeback is literal text surgery: the declaration now carries the
    # optimized value with exact repr, and the rest of the file is intact.
    assert "radius = Scalar(0.8, free=True" not in result["source"]
    assert f'radius = Scalar({optimized!r}, free=True, name="radius")' in result["source"]
    assert 'shrink = Optimization(name="fit-radius"' in result["source"]
    recompiled = compile_source(result["source"])
    assert recompiled["ok"] is True
    assert recompiled["optimizations"][0]["editable"] is True


def test_optimize_endpoint_honors_a_step_override():
    from cadjoint.viewer.playground import optimize_source

    result = optimize_source({"source": OPTIMIZE_SOURCE, "name": "fit-radius", "steps": 2})

    assert result["ok"] is True
    assert result["steps"] == 2
    assert len(result["history"]) == 2
    assert len(result["trajectory"]) == 3


def test_optimize_endpoint_reports_an_unknown_optimization():
    from cadjoint.viewer.playground import optimize_source

    result = optimize_source({"source": OPTIMIZE_SOURCE, "name": "nope"})

    assert result["ok"] is False
    assert "declares no optimization named 'nope'" in result["error"]
    assert "'fit-radius'" in result["error"]


@pytest.mark.parametrize(
    ("request_body", "message"),
    [
        ({}, "needs `name`"),
        ({"name": "   "}, "needs `name`"),
        ({"name": "fit-radius", "steps": 0}, "positive integer"),
        ({"name": "fit-radius", "steps": True}, "positive integer"),
        ({"name": "fit-radius", "steps": 2.5}, "positive integer"),
        ({"name": "fit-radius", "steps": 500}, "capped at 200"),
    ],
)
def test_optimize_endpoint_validates_its_request(request_body, message):
    from cadjoint.viewer.playground import optimize_source

    result = optimize_source({"source": OPTIMIZE_SOURCE, **request_body})

    assert result["ok"] is False
    assert message in result["error"]


def test_optimization_patches_round_trip_through_compile():
    stepped = patch_source(
        {
            "source": OPTIMIZE_SOURCE,
            "op": "set_optimization_value",
            "optimization": "fit-radius",
            "argument": "steps",
            "value": 6,
        }
    )
    assert stepped["ok"] is True
    assert "steps=6" in stepped["source"]

    tuned = patch_source(
        {
            "source": stepped["source"],
            "op": "set_optimization_value",
            "optimization": 0,
            "argument": "learning_rate",
            "value": 0.02,
        }
    )
    assert tuned["ok"] is True
    assert "learning_rate=0.02" in tuned["source"]

    result = compile_source(tuned["source"])
    assert result["ok"] is True
    assert result["optimizations"][0]["steps"] == 6
    assert result["optimizations"][0]["learning_rate"] == pytest.approx(0.02)

    deleted = patch_source(
        {"source": tuned["source"], "op": "delete_optimization", "optimization": "fit-radius"}
    )
    assert deleted["ok"] is True
    assert "shrink = " not in deleted["source"]


@pytest.mark.parametrize(
    ("request_body", "message"),
    [
        ({"op": "delete_optimization"}, "`optimization`"),
        ({"op": "delete_optimization", "optimization": -1}, "`optimization`"),
        ({"op": "delete_optimization", "optimization": "  "}, "`optimization`"),
        (
            {"op": "set_optimization_value", "optimization": 0, "argument": "method", "value": 1},
            "`steps` or `learning_rate`",
        ),
        (
            {
                "op": "set_optimization_value",
                "optimization": 0,
                "argument": "steps",
                "value": "many",
            },
            "numeric `value`",
        ),
    ],
)
def test_patch_source_validates_optimization_requests(request_body, message):
    result = patch_source({"source": OPTIMIZE_SOURCE, **request_body})

    assert result["ok"] is False
    assert message in result["error"]


# ── Study-backed optimizations (end-to-end differentiable simulation) ────────

STUDY_OPTIMIZE_SOURCE = """import jax.numpy as jnp

from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
from cadjoint.geometry import Vector
from cadjoint.optimize import Optimization
from cadjoint.sdf.primitives import Box

size = Vector([0.8, 0.5, 0.5], free=True, name="size")
scene = Box(size)

heat = ThermalStudy(
    name="bar-conduction",
    resolution=8,
    conductivity=1.0,
    bcs=[Dirichlet(Nodes.side("-x"), value=0.0), Dirichlet(Nodes.side("+x"), value=100.0)],
    bounds=(-1.2, -0.9, -0.9),
    size=(2.4, 1.8, 1.8),
)


def volume(params):
    return jnp.prod(2.0 * params["size"])


cool = Optimization(
    name="cool-bar",
    study="bar-conduction",
    metric="mean",
    regularizer=volume,
    regularizer_weight=5.0,
    remesh_every=0,
    steps=2,
    learning_rate=0.05,
)
"""


def test_compile_reports_a_study_backed_optimization():
    result = compile_source(STUDY_OPTIMIZE_SOURCE)

    assert result["ok"] is True
    (optimization,) = result["optimizations"]
    assert optimization["name"] == "cool-bar"
    assert optimization["objective"] == "mean(bar-conduction)"
    assert optimization["study"] == "bar-conduction"
    assert optimization["metric"] == "mean"
    assert optimization["remesh_every"] == 0
    assert optimization["regularizer"] == "volume"
    assert optimization["regularizer_weight"] == pytest.approx(5.0)
    assert optimization["parameters"] == ["size"]
    assert optimization["editable"] is True


def test_study_backed_optimize_returns_the_simulate_block():
    pytest.importorskip("jax_fem", reason="study-backed runs need the fem extra")
    from cadjoint.viewer.playground import optimize_source

    result = optimize_source({"source": STUDY_OPTIMIZE_SOURCE, "name": "cool-bar"})

    assert result["ok"] is True, result.get("error")
    assert result["kind"] == "optimize"
    assert result["name"] == "cool-bar"
    assert result["steps"] == 2

    # The descent bookkeeping is exactly the objective form's.
    assert len(result["history"]) == 2
    assert result["history"][-1]["objective"] < result["history"][0]["objective"]
    assert len(result["trajectory"]) == 3
    assert result["trajectory"][0]["parameters"] == result["initial"]
    assert result["trajectory"][-1]["parameters"] == result["parameters"]
    optimized = result["parameters"]["size"]
    assert all(after < before for after, before in zip(optimized, [0.8, 0.5, 0.5]))

    # Writeback is literal text surgery, same as the objective form.
    assert 'Vector([0.8, 0.5, 0.5], free=True, name="size")' not in result["source"]
    assert 'name="size"' in result["source"]
    assert 'cool = Optimization(\n    name="cool-bar"' in result["source"]

    # The ADDITIVE simulate block: the optimized design solved on a fresh
    # mesh, in the exact shapes /api/simulate responses carry.
    simulate = result["simulate"]
    assert set(simulate) == {"field", "mesh", "result", "mesh_info"}
    assert simulate["field"] == "temperature"
    mesh = simulate["mesh"]
    assert set(mesh) >= {
        "positions",
        "scalars",
        "indices",
        "groups",
        "range",
        "vertex_count",
        "edges",
        "fields",
        "ranges",
    }
    assert len(mesh["scalars"]) == mesh["vertex_count"]
    assert mesh["fields"]["temperature"] == mesh["scalars"]
    assert mesh["ranges"]["temperature"] == pytest.approx(mesh["range"])
    # Element edges are index pairs into the compacted vertex list.
    assert len(mesh["edges"]) % 2 == 0 and len(mesh["edges"]) > 0
    assert all(0 <= index < mesh["vertex_count"] for index in mesh["edges"])
    summary = simulate["result"]
    assert summary["name"] == "bar-conduction"
    assert summary["kind"] == "thermal"
    info = simulate["mesh_info"]
    assert info["nodes"] == summary["nodes"]
    assert info["elements"] == summary["elements"]

    # The optimized program recompiles cleanly.
    recompiled = compile_source(result["source"])
    assert recompiled["ok"] is True
    assert recompiled["optimizations"][0]["editable"] is True


def test_objective_form_optimize_carries_no_simulate_block():
    from cadjoint.viewer.playground import optimize_source

    result = optimize_source({"source": OPTIMIZE_SOURCE, "name": "fit-radius", "steps": 2})

    assert result["ok"] is True
    assert "simulate" not in result


# ── /api/optimize streams progress as chunked NDJSON ─────────────────────────


def test_optimize_events_stream_progress_then_done():
    from cadjoint.viewer.playground import optimize_source_events

    events = list(optimize_source_events({"source": OPTIMIZE_SOURCE, "name": "fit-radius"}))

    progress = [event for event in events if event["event"] == "progress"]
    assert len(progress) == 4  # one per declared optimizer step
    assert [event["step"] for event in progress] == [1, 2, 3, 4]
    assert all(event["steps"] == 4 for event in progress)
    assert all(event["elapsed"] >= 0.0 for event in progress)
    objectives = [event["objective"] for event in progress]
    assert objectives[-1] < objectives[0]
    assert all(event["grad_norm"] >= 0.0 for event in progress)

    done = events[-1]
    assert done["event"] == "done"
    assert done["ok"] is True
    assert len(done["history"]) == 4
    assert "source" in done and "trajectory" in done


def test_optimize_events_report_validation_errors_as_done():
    from cadjoint.viewer.playground import optimize_source_events

    events = list(optimize_source_events({"source": OPTIMIZE_SOURCE, "name": "   "}))
    assert len(events) == 1
    assert events[0]["event"] == "done"
    assert events[0]["ok"] is False
    assert "name" in events[0]["error"]


def test_optimize_events_report_a_timeout_as_a_clear_error():
    from cadjoint.viewer.playground import optimize_source_events

    events = list(
        optimize_source_events({"source": OPTIMIZE_SOURCE, "name": "fit-radius"}, timeout=0.1)
    )
    assert events[-1]["event"] == "done"
    assert events[-1]["ok"] is False
    assert "Optimization exceeded the 0.1-second timeout." == events[-1]["error"]


def test_optimize_endpoint_streams_ndjson_over_http():
    import threading
    from urllib.request import urlopen

    from cadjoint.viewer.playground import create_server

    server = create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        base = f"http://{host}:{port}"
        with urlopen(f"{base}/api/session") as response:
            token = json.loads(response.read())["token"]
        request = post(
            base, "/api/optimize", {"source": OPTIMIZE_SOURCE, "name": "fit-radius"}, token
        )
        with urlopen(request) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("application/x-ndjson")
            events = [json.loads(line) for line in response if line.strip()]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    kinds = [event["event"] for event in events]
    assert kinds == ["progress"] * 4 + ["done"]
    assert events[-1]["ok"] is True
    assert events[-1]["parameters"]["radius"] == pytest.approx(
        events[-1]["trajectory"][-1]["parameters"]["radius"]
    )
