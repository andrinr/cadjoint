"""The browser's WGSL must be the worker's WGSL.

``frontend/src/scene`` builds the viewer's shaders from a ``cadjoint-scene``
v2 graph, with no Python in the path. That makes it a second implementation
of :mod:`cadjoint.backends.wgsl.direct`, and the Python module is its oracle:

* the module text is byte for byte the same, in the uniform form the viewer
  ships (and in the literal form without culling), so every image and every
  field value the two produce are the same;
* the parameter slots have the same names, sizes and values, except the
  culling boxes, which JAX computes in float32 with its own sum order and
  transcendental functions and which therefore agree to an ulp or two;
* on a device both modules, each with its own buffer, give identical
  distances and materials;
* the six shader fields and the program contract of the compile payload are
  the ones the worker assembles.

The TypeScript runs under Node (22.6 or later strips the types itself).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cadjoint.backends.jaxjs import graph_sdf, scene_graph
from cadjoint.backends.wgsl.direct import UnsupportedNode, compile_sdf_direct
from cadjoint.geometry.parameters import Scalar, Vector, Vector2
from cadjoint.render.material import Material
from cadjoint.sdf.boolean import Difference, Intersection, Union
from cadjoint.sdf.boolean.xor import Xor
from cadjoint.sdf.primitives import Box, Capsule, Cylinder, Plane, RoundBox, Sphere, Torus
from cadjoint.sdf.primitives.loft import LoftedPolygon
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon, RevolvedPolygon
from cadjoint.sdf.transforms import Translate
from cadjoint.sdf.transforms.affine import Rotate, Scale
from cadjoint.sdf.transforms.deformations import Twist
from cadjoint.sdf.transforms.fields import Mirror, Offset, Shell
from cadjoint.sdf.transforms.patterns import LinearPattern, PolarPattern

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "frontend" / "src" / "scene" / "cli.ts"
TEMPLATES = ROOT / "frontend" / "src" / "scene" / "templates.generated.ts"

#: The shipped scenes that load stand-alone (``starter`` needs a declared study).
SHIPPED = ["bracket", "cylinder_block", "duct_fairing", "duct_sink", "end_cap", "motor_shield"]


def _node_version() -> tuple[int, ...] | None:
    node = shutil.which("node")
    if node is None:
        return None
    out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout
    return tuple(int(x) for x in out.strip().lstrip("v").split(".")[:2])


needs_node = pytest.mark.skipif(
    (_node_version() or (0,)) < (22, 6), reason="the TypeScript emitter runs under Node >= 22.6"
)


def _typescript(request: dict) -> dict:
    out = subprocess.run(
        ["node", str(CLI)], input=json.dumps(request), capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def _ts_direct(scene, *, uniforms: bool, culling: bool) -> dict:
    graph = scene_graph(scene, version=2)
    return _typescript({"graph": graph, "mode": "direct", "uniforms": uniforms, "culling": culling})


def _shipped(stem: str):
    from cadjoint.viewer.worker.scene import _execute_scene

    return _execute_scene((ROOT / "scenes" / f"{stem}.py").read_text())["scene"]


def _ring(count: int, radius: float, free: str | None = None) -> list:
    return [
        Vector2(
            [radius * np.cos(2 * np.pi * i / count), radius * np.sin(2 * np.pi * i / count)],
            free=free is not None and i == 1,
            name=f"{free}{i}" if free is not None and i == 1 else None,
        )
        for i in range(count)
    ]


def _covered_scene():
    """Every node type the direct backend covers, with free parameters,
    materials and a subtree reached twice."""
    steel = Material(color=[0.6, 0.62, 0.65], roughness=0.3, metallic=1.0)
    paint = Material(color=[0.8, 0.2, 0.1], roughness=0.6, reflectivity=0.04)
    radius = Scalar(0.3, free=True, name="radius")
    shared = Translate(Sphere(radius, material=paint), Vector([0.0, 0.0, 0.5]))
    square = [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]]
    small = [[-0.15, -0.15], [0.15, -0.15], [0.15, 0.15], [-0.15, 0.15]]
    parts = [
        Translate(Sphere(radius), Vector([1.0, 0.0, 0.0], free=True, name="shift")),
        Rotate(Box([0.3, 0.2, 0.25], material=steel), [0.0, 1.0, 0.0], 0.4),
        Rotate(Cylinder(0.2, 0.5), [1.0, 1.0, 0.3], -1.1),
        Capsule(Scalar(0.15, free=True, name="capsule_r"), 0.4),
        Mirror(Translate(Torus(0.3, 0.08), [0.5, 0.5, 0.0]), "x"),
        Offset(ExtrudedPolygon(_ring(6, 0.4, free="hex"), Scalar(0.3, free=True, name="d")), 0.02),
        Shell(ExtrudedPolygon(_ring(7, 0.4), 0.3, draft=5.0, twist=30.0), 0.05),
        RevolvedPolygon([[0.2, -0.2], [0.35, -0.2], [0.35, 0.2], [0.2, 0.2]], 0.1),
        LoftedPolygon(square, small, 0.6),
        LinearPattern(Sphere(0.1), [1.0, 1.0, 0.0], 5, 0.3, skip=(2,)),
        PolarPattern(Translate(Box([0.05, 0.05, 0.3]), [0.6, 0.0, 0.0]), 6, skip=(4,)),
        shared,
        Mirror(shared, "z"),
    ]
    body = Union(*parts, smoothness=0.05)
    return Difference(
        Intersection(body, Box([1.2, 1.2, 0.8]), smoothness=0.0),
        Translate(Sphere(0.2), [0.0, 0.0, 0.7]),
        smoothness=0.02,
    )


def _scenes():
    yield "covered", _covered_scene()
    for stem in SHIPPED:
        yield stem, _shipped(stem)


_SCENES = dict(_scenes())

_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?")


def _assert_same_up_to_box_literals(python: str, typescript: str) -> None:
    """Identical text except numbers inside `cull_box(...)` tests, which agree to 1e-6."""
    assert _NUMBER.sub("#", python) == _NUMBER.sub("#", typescript)
    for a, b in zip(python.splitlines(), typescript.splitlines()):
        if a == b:
            continue
        assert "cull_box(" in a, (a, b)
        np.testing.assert_allclose(
            [float(x) for x in _NUMBER.findall(a)],
            [float(x) for x in _NUMBER.findall(b)],
            rtol=1e-6,
            atol=1e-7,
        )


def _assert_same_slots(python, typescript: dict) -> None:
    assert [(n, c) for n, c, _ in python.parameters] == [
        (n, c) for n, c, _ in typescript["parameters"]
    ]
    derived = set(python.derived)
    for (name, _, a), (_, _, b) in zip(python.parameters, typescript["parameters"]):
        if name in derived:
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7, err_msg=name)
        else:
            assert list(a) == list(b), name
    assert list(python.derived) == typescript["derived"]
    assert {k: list(v) for k, v in python.depends.items()} == typescript["depends"]


# ── the graph ────────────────────────────────────────────────────────────────


def test_version_1_stays_the_default_and_unchanged():
    graph = scene_graph(_covered_scene())
    assert graph["version"] == 1

    def keys(node):
        yield set(node)
        for child in node["children"]:
            yield from keys(child)

    assert all(k == {"type", "params", "children"} for k in keys(graph["root"]))


def test_version_2_evaluates_like_the_object():
    scene = _covered_scene()
    points = np.random.default_rng(0).uniform(-1.2, 1.2, size=(2000, 3)).astype(np.float32)
    v2 = graph_sdf(scene_graph(scene, version=2))
    np.testing.assert_array_equal(np.asarray(v2(points)), np.asarray(scene(points)))


def test_version_2_names_free_parameters_and_shares_ids():
    graph = scene_graph(_covered_scene(), version=2)
    nodes = []

    def walk(node):
        nodes.append(node)
        for child in node["children"]:
            walk(child)

    walk(graph["root"])
    free = [n["free"] for n in nodes if "free" in n]
    assert {"radius": "radius"} in free
    assert {"offset": "shift"} in free
    assert any(f.get("vertices", [None, None])[1] == "hex1" for f in free)
    ids = [n["id"] for n in nodes]
    # The shared translate and its sphere appear twice under one id each.
    assert len(ids) - len(set(ids)) == 2
    assert all("material" in n for n in nodes if not n["children"])


# ── the emitted module ───────────────────────────────────────────────────────


@needs_node
def test_the_templates_are_up_to_date():
    sys.path.insert(0, str(ROOT / "frontend" / "tools"))
    try:
        from emit_wgsl_templates import templates_ts
    finally:
        sys.path.pop(0)
    assert TEMPLATES.read_text() == templates_ts(), (
        "frontend/src/scene/templates.generated.ts is stale: "
        "run `python frontend/tools/emit_wgsl_templates.py`"
    )


@needs_node
@pytest.mark.parametrize("label", list(_SCENES))
@pytest.mark.parametrize(
    ("uniforms", "culling"), [(True, True), (True, False), (False, False), (False, True)]
)
def test_the_typescript_module_is_the_python_one(label, uniforms, culling):
    scene = _SCENES[label]
    python = compile_sdf_direct(scene, uniforms=uniforms, culling=culling)
    typescript = _ts_direct(scene, uniforms=uniforms, culling=culling)
    if uniforms or not culling:
        # Boxes are buffer slots in the uniform form, so nothing rounded
        # differently reaches the text.
        assert typescript["wgsl"] == python.wgsl
    else:
        _assert_same_up_to_box_literals(python.wgsl, typescript["wgsl"])
    _assert_same_slots(python, typescript)
    np.testing.assert_array_equal(
        np.asarray(typescript["vertices"], dtype=np.float32).reshape(-1, 2), python.vertices
    )


@needs_node
@pytest.mark.parametrize("label", list(_SCENES))
def test_the_payload_shaders_are_the_workers(label):
    from cadjoint.viewer._pathtracer import WGSL_PRESENT_TEMPLATE, build_path_tracer_shader
    from cadjoint.viewer._webgpu import build_viewer_shader
    from cadjoint.viewer.worker.main import _direct_shader

    scene = _SCENES[label]
    source, program = _direct_shader(scene)
    graph = scene_graph(scene, version=2)
    shaders = _typescript({"graph": graph, "mode": "shaders", "culling": True})
    preview = build_viewer_shader(source)
    assert shaders["sdf"] == shaders["scene_wgsl"] == source
    assert shaders["shader"] == shaders["preview_shader"] == preview
    assert shaders["path_shader"] == build_path_tracer_shader(source)
    assert shaders["present_shader"] == WGSL_PRESENT_TEMPLATE
    import hashlib

    expected_hash = hashlib.sha256(
        preview.encode() + b"\0" + build_path_tracer_shader(source).encode()
    ).hexdigest()
    assert shaders["shader_hash"] == expected_hash
    ours = shaders["program"]
    for key in ("group", "binding", "buffer_bytes", "nan_offset", "cull_margin_offset"):
        assert ours[key] == program[key], key
    for a, b in zip(program["parameters"], ours["parameters"], strict=True):
        assert {k: v for k, v in a.items() if k != "value"} == {
            k: v for k, v in b.items() if k != "value"
        }
        if a["free"]:
            assert a["value"] == b["value"], a["name"]
        else:
            np.testing.assert_allclose(a["value"], b["value"], rtol=1e-6, atol=1e-7)


@needs_node
@pytest.mark.parametrize(
    ("label", "scene"),
    [
        ("round_box", RoundBox([0.3, 0.3, 0.3], 0.05)),
        ("plane", Plane(0.2)),
        ("twist", Twist(Capsule(0.15, 0.4), 0.8)),
        ("xor", Xor(Sphere(0.4), Translate(Sphere(0.4), [0.3, 0.0, 0.0]))),
    ],
)
def test_a_node_without_a_kernel_is_refused_on_both_sides(label, scene):
    with pytest.raises(UnsupportedNode):
        compile_sdf_direct(scene, uniforms=True, culling=True)
    assert "unsupported" in _ts_direct(scene, uniforms=True, culling=True)


@needs_node
def test_scale_is_refused_in_typescript():
    """The Python direct backend reads a ``factor`` that ``Scale`` does not
    have and fails with a ``KeyError``; this side refuses it outright."""
    scene = Scale(Sphere(0.3), 1.5)
    assert "unsupported" in _ts_direct(scene, uniforms=True, culling=True)


# ── on a device ──────────────────────────────────────────────────────────────


def _device_points(scene) -> np.ndarray:
    from cadjoint.backends.wgsl.direct import _Emitter

    bounds = _Emitter()._node_bounds(scene)
    center = np.zeros(3) if bounds is None else np.asarray(bounds.center)
    half = np.full(3, 2.0) if bounds is None else np.asarray(bounds.half) * 1.2 + 1e-3
    rng = np.random.default_rng(20261019)
    return (center + half * rng.uniform(-1.0, 1.0, size=(4096, 3))).astype(np.float32)


@needs_node
@pytest.mark.parametrize("label", list(_SCENES))
def test_both_modules_give_identical_fields_on_a_device(label):
    pytest.importorskip("wgpu", reason="the comparison runs both modules on a device")
    from tests.backends.test_wgsl_direct import _MATERIAL_HARNESS, _evaluate_on_device

    scene = _SCENES[label]
    python = compile_sdf_direct(scene, uniforms=True, culling=True)
    typescript = _ts_direct(scene, uniforms=True, culling=True)
    points = _device_points(scene)
    for harness in (None, _MATERIAL_HARNESS):
        kwargs = {} if harness is None else {"harness": harness}
        a = _evaluate_on_device(
            python.wgsl, points, uniform_values=list(python.parameters), **kwargs
        )
        b = _evaluate_on_device(
            typescript["wgsl"],
            points,
            uniform_values=[tuple(s) for s in typescript["parameters"]],
            **kwargs,
        )
        np.testing.assert_array_equal(a, b)


# ── the frontend's own fixture ───────────────────────────────────────────────

FIXTURE = ROOT / "frontend" / "test" / "fixtures" / "scene-direct.json"


def _fixture() -> dict:
    """What `frontend/test/sceneDirect.test.ts` checks the emitter against
    without Python: graphs and the Python module for each, in two forms."""
    cases = []
    for label in ("covered", "bracket"):
        scene = _SCENES[label]
        graph = scene_graph(scene, version=2)
        for uniforms, culling in ((True, True), (False, False)):
            program = compile_sdf_direct(scene, uniforms=uniforms, culling=culling)
            cases.append(
                {
                    "label": label,
                    "graph": graph,
                    "uniforms": uniforms,
                    "culling": culling,
                    "wgsl": program.wgsl,
                    "parameters": [[n, c, list(v)] for n, c, v in program.parameters],
                    "derived": list(program.derived),
                    "depends": {k: list(v) for k, v in program.depends.items()},
                }
            )
    return {"cases": cases}


def test_the_frontend_fixture_is_up_to_date():
    """Rewrite it with ``CADJOINT_WRITE_FIXTURES=1 pytest -k frontend_fixture``."""
    import os

    text = json.dumps(_fixture(), indent=1) + "\n"
    if os.environ.get("CADJOINT_WRITE_FIXTURES"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(text)
    assert FIXTURE.read_text() == text, (
        f"{FIXTURE.relative_to(ROOT)} is stale: "
        "run `CADJOINT_WRITE_FIXTURES=1 pytest tests/backends/test_wgsl_frontend.py -k frontend_fixture`"
    )


@needs_node
@pytest.mark.parametrize("stem", ["duct_sink", "end_cap"])
def test_the_compile_payload_graph_rebuilds_its_shaders(stem):
    """End to end through the worker: the payload's ``scene_graph`` gives the
    payload's own shaders and hash in TypeScript."""
    from cadjoint.viewer.worker.main import _compile_source

    payload = _compile_source((ROOT / "scenes" / f"{stem}.py").read_text())
    assert payload["scene_graph"]["version"] == 2
    shaders = _typescript({"graph": payload["scene_graph"], "mode": "shaders", "culling": True})
    assert shaders["scene_wgsl"] == payload["scene_wgsl"]
    assert shaders["shader_hash"] == payload["shader_hash"]
