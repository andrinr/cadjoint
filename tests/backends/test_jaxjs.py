"""The jax-js backend: every node type emits, graphs lose nothing, TS matches Python.

The last test runs the generated TypeScript on jax-js (Wasm, under Node) and
is skipped when ``web-spike/node_modules`` is not installed
(``cd web-spike && npm install``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.backends.jaxjs import NODE_TYPES, UnsupportedNode, graph_sdf, scene_graph
from cadjoint.backends.jaxjs.kernels import kernel_sets
from cadjoint.sdf.boolean import Difference, Intersection, Union, Xor
from cadjoint.sdf.primitives import Box, Capsule, Cylinder, Plane, RoundBox, Sphere, Torus
from cadjoint.sdf.primitives.loft import LoftedPolygon
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon, RevolvedPolygon
from cadjoint.sdf.transforms.affine.rotate import Rotate
from cadjoint.sdf.transforms.affine.scale import Scale
from cadjoint.sdf.transforms.affine.translate import Translate
from cadjoint.sdf.transforms.deformations.twist import Twist
from cadjoint.sdf.transforms.fields import Mirror, Offset, Shell
from cadjoint.sdf.transforms.patterns import LinearPattern, PolarPattern

SPIKE = Path(__file__).parents[2] / "web-spike"


def _every_node_type():
    """One scene using all node types, each with non-trivial parameters."""
    hexagon = [[np.cos(a) * 0.4, np.sin(a) * 0.4] for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
    ring = [[0.2, -0.2], [0.35, -0.2], [0.35, 0.2], [0.2, 0.2]]
    square = [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]]
    small = [[-0.15, -0.15], [0.15, -0.15], [0.15, 0.15], [-0.15, 0.15]]
    parts = [
        Translate(Sphere(0.3), [1.0, 0.0, 0.0]),
        Rotate(Box([0.3, 0.2, 0.25]), [0.0, 1.0, 0.0], 0.4),
        Scale(RoundBox([0.2, 0.2, 0.2], 0.05), 1.5),
        Scale(Cylinder(0.2, 0.5), [1.0, 2.0, 1.0]),
        Twist(Capsule(0.15, 0.4), 0.8),
        Mirror(Translate(Torus(0.3, 0.08), [0.5, 0.5, 0.0]), "x"),
        Offset(ExtrudedPolygon(hexagon, 0.3), 0.02),
        Shell(ExtrudedPolygon(hexagon, 0.3, draft=5.0, twist=30.0), 0.05),
        RevolvedPolygon(ring, 0.1),
        LoftedPolygon(square, small, 0.6),
        LinearPattern(Sphere(0.1), [1.0, 1.0, 0.0], 5, 0.3, skip=(2,)),
        PolarPattern(Translate(Box([0.05, 0.05, 0.3]), [0.6, 0.0, 0.0]), 6, skip=(4,)),
        Xor(Sphere(0.4), Translate(Sphere(0.4), [0.3, 0.0, 0.0])),
    ]
    body = Union(*parts, smoothness=0.05)
    return Difference(
        Intersection(body, Translate(Plane(0.8), [0.0, 0.0, 0.0]), smoothness=0.0),
        Translate(Sphere(0.2), [0.0, 0.0, 0.7]),
        smoothness=0.02,
    )


def _points(n=3000):
    return np.random.default_rng(0).uniform(-1.2, 1.2, size=(n, 3)).astype(np.float32)


def test_the_scene_uses_every_node_type():
    graph = scene_graph(_every_node_type())

    def types(node):
        yield node["type"]
        for child in node["children"]:
            yield from types(child)

    assert set(types(graph["root"])) == set(NODE_TYPES)


def test_every_node_type_emits_typescript():
    sets = kernel_sets()
    assert {ks.node.name for ks in sets} == set(NODE_TYPES)
    for ks in sets:
        assert ks.fns, ks.node.name


def test_graph_evaluates_like_the_object():
    scene = _every_node_type()
    graph = json.loads(json.dumps(scene_graph(scene)))  # through JSON, as the browser gets it
    p = jnp.asarray(_points())
    np.testing.assert_array_equal(np.asarray(graph_sdf(graph)(p)), np.asarray(scene(p)))


def test_plain_callable_children_are_refused():
    with pytest.raises(UnsupportedNode):
        scene_graph(Shell(lambda p: jnp.linalg.norm(p, axis=-1) - 1.0, 0.1))


@pytest.mark.skipif(
    shutil.which("npx") is None or not (SPIKE / "node_modules" / "@jax-js").exists(),
    reason="needs Node and `npm install` in web-spike/",
)
@pytest.mark.parametrize("mode", ["fused", "per-node"])
def test_generated_typescript_matches_python(tmp_path, mode):
    scene = _every_node_type()
    points = _points()
    expected = np.asarray(scene(jnp.asarray(points)))
    (tmp_path / "in.json").write_text(
        json.dumps({"graph": scene_graph(scene), "points": points.ravel().tolist()})
    )
    args = ["npx", "tsx", "eval.mts", str(tmp_path / "in.json")]
    if mode == "per-node":
        args.append("--per-node")
    run = subprocess.run(args, cwd=SPIKE, capture_output=True, text=True, timeout=600, check=True)
    values = np.asarray(json.loads(run.stdout)["values"])
    np.testing.assert_allclose(values, expected, atol=2e-5)
