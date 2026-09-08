"""The WGSL emitted from a table is the same function JAX evaluates from it.

Run on the GPU with the harness the direct backend's tests use, at random
points, for every node kind and every shipped scene.  With θ bound as a
storage buffer the module is compiled once and the design is a buffer
write, which is the property the viewer's drag depends on.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from cadjoint.zeroset import lower
from cadjoint.zeroset.evaluate import field
from cadjoint.zeroset.wgsl import emit
from tests.backends.test_wgsl_direct import _evaluate_on_device
from tests.zeroset.test_lowering import NODES, SCENES

wgpu = pytest.importorskip("wgpu")


def _gpu_field(model, pts):
    return _evaluate_on_device(emit(model), pts.astype(np.float32))


@pytest.mark.parametrize("label", list(NODES))
def test_each_node_kind_emits_the_field_jax_evaluates(label):
    model = lower(NODES[label]())
    pts = np.random.default_rng(1).uniform(-1.5, 1.5, size=(512, 3))
    expected = np.asarray(field(model)(np.asarray(model.theta, np.float32), pts.astype(np.float32)))
    np.testing.assert_allclose(_gpu_field(model, pts), expected, rtol=2e-4, atol=2e-5)


# Deliberately not marked slow: the bracket case measures 43 s and every
# other scene under nine, which is the emitter's one compilation landing on
# whichever case runs first — marking it would move that cost, not remove it.
@pytest.mark.parametrize("scene", SCENES, ids=[s.stem for s in SCENES])
def test_each_shipped_scene_emits_the_field_jax_evaluates(scene):
    from cadjoint.viewer.worker.scene import _execute_scene

    model = lower(_execute_scene(scene.read_text())["scene"])
    pts = np.random.default_rng(2).uniform(-2.5, 2.5, size=(1024, 3))
    expected = np.asarray(field(model)(np.asarray(model.theta, np.float32), pts.astype(np.float32)))
    got = _gpu_field(model, pts)
    # The polygon's even-odd sign is decided by comparisons against vertex
    # heights; a point that nearly ties one is resolved by float32 evaluation
    # order, which XLA and WGSL do differently, and the rule is discontinuous
    # there. Everywhere else the two must agree to float32 precision.
    close = np.isclose(got, expected, rtol=5e-4, atol=5e-5)
    assert close.mean() > 0.99, f"{(~close).sum()} of {len(pts)} points disagree"
    np.testing.assert_allclose(got[close], expected[close], rtol=5e-4, atol=5e-5)


def test_the_module_names_one_function_per_shape_node_and_no_more():
    from cadjoint.geometry.parameters import Scalar
    from cadjoint.sdf.primitives import Cylinder
    from cadjoint.sdf.transforms import Translate
    from cadjoint.sdf.transforms.patterns import PolarPattern

    model = lower(
        PolarPattern(
            Translate(Cylinder(radius=Scalar(0.1, name="r"), height=0.4), [0.7, 0, 0]),
            count=6,
            axis="z",
        )
    )
    source = emit(model)
    shapes = sum(1 for n in model.nodes if next(iter(n)) in ("patch", "warp", "combine"))
    assert len(re.findall(r"^fn s\d+\(", source, re.M)) == shapes
    assert "fn sdf(p: vec3<f32>) -> f32" in source


def test_theta_can_be_a_buffer_instead_of_literals():
    from cadjoint.geometry.parameters import Scalar
    from cadjoint.sdf.primitives import Sphere

    model = lower(Sphere(radius=Scalar(0.8, free=True, name="r")))
    baked, bound = emit(model), emit(model, theta_binding=(2, 0))
    assert "0.8" in baked and "theta[0u]" in bound
    assert "@group(2) @binding(0) var<storage, read> theta: array<f32>;" in bound
