"""The table is a faithful lowering: its JAX evaluation is cadjoint's own field.

Every node kind the lowering handles is checked against the kernel it
lowers — the kernel is normative — on values and on the derivative in the
design, at random points inside and outside.  The shipped scenes are
checked whole, which is where compositions (patterns of extrusions under
smooth booleans) would show a mistake no single node does.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint import extract_parameters, functionalize
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.boolean import Difference, Intersection, Union
from cadjoint.sdf.primitives import Box, Capsule, Cylinder, Sphere, Torus
from cadjoint.sdf.transforms import Rotate, Scale, Translate
from cadjoint.sdf.transforms.fields import Offset, Shell
from cadjoint.zeroset import lower
from cadjoint.zeroset.evaluate import field, surfaces

SCENES = sorted(Path(__file__).resolve().parents[2].joinpath("scenes").glob("*.py"))


def _reference(root):
    """cadjoint's field as (theta flat, points) -> values, in `lower`'s parameter order."""
    free, fixed, _ = extract_parameters(root)
    fn, names = functionalize(root), list(free.keys())

    def f(theta, pts):
        fp, i = {}, 0
        for n in names:
            size = np.asarray(free[n]).size
            fp[n] = theta[i : i + size].reshape(np.asarray(free[n]).shape) if size > 1 else theta[i]
            i += size
        return jax.vmap(fn(fp, fixed))(pts)

    return f


def _agree(root, extent=1.5, n=300, rtol=2e-5):
    model = lower(root)
    theta = jnp.asarray(model.theta)
    pts = jnp.asarray(np.random.default_rng(0).uniform(-extent, extent, size=(n, 3)))
    ours, theirs = field(model), _reference(root)
    np.testing.assert_allclose(ours(theta, pts), theirs(theta, pts), rtol=rtol, atol=1e-5)
    if len(model.theta):
        np.testing.assert_allclose(
            jax.jacfwd(ours)(theta, pts), jax.jacfwd(theirs)(theta, pts), rtol=1e-4, atol=1e-4
        )
    return model


NODES = {
    "sphere": lambda: Sphere(radius=Scalar(0.8, free=True, name="r")),
    "box": lambda: Box(size=Vector([0.6, 0.4, 0.9], free=True, name="size")),
    "cylinder": lambda: Cylinder(radius=Scalar(0.5, free=True, name="r"), height=0.7),
    "torus": lambda: Torus(major_radius=Scalar(1.0, free=True, name="R"), minor_radius=0.25),
    "capsule": lambda: Capsule(radius=Scalar(0.3, free=True, name="r"), height=0.7),
    "translate": lambda: Translate(
        Sphere(radius=0.7), Vector([0.4, -0.3, 0.2], free=True, name="o")
    ),
    "rotate": lambda: Rotate(
        Box(size=[0.6, 0.3, 0.2]), [0.3, 1.0, 0.2], Scalar(0.7, free=True, name="a")
    ),
    "scale": lambda: Scale(Sphere(radius=0.5), [1.5, 1.5, 1.5]),
    "offset": lambda: Offset(Box(size=[0.5, 0.5, 0.5]), Scalar(0.1, free=True, name="d")),
    "shell": lambda: Shell(Sphere(radius=0.8), Scalar(0.2, free=True, name="t")),
    "hard union": lambda: Union(Sphere(radius=0.8), Box(size=[0.5, 0.5, 0.5]), smoothness=0.0),
    "smooth union": lambda: Union(
        Sphere(radius=Scalar(0.8, free=True, name="a")),
        Translate(Box(size=[0.5, 0.5, 0.5]), [0.6, 0, 0]),
        smoothness=Scalar(0.1, free=True, name="k"),
    ),
    "smooth intersection": lambda: Intersection(
        Sphere(radius=0.8), Box(size=[0.6, 0.6, 0.6]), smoothness=0.05
    ),
    "smooth difference": lambda: Difference(
        Box(size=[0.7, 0.7, 0.7]), Sphere(radius=0.8), smoothness=0.05
    ),
}


@pytest.mark.parametrize("label", list(NODES))
def test_each_node_kind_lowers_to_its_kernel(label):
    _agree(NODES[label]())


@pytest.mark.parametrize("scene", SCENES, ids=[s.stem for s in SCENES])
def test_each_shipped_scene_lowers_whole(scene):
    from cadjoint.viewer.worker.scene import _execute_scene

    root = _execute_scene(scene.read_text())["scene"]
    model = _agree(root, extent=2.5, n=200, rtol=5e-5)
    assert len(model.names) == len(model.theta)
    assert len(model.nodes) < 20_000


def test_a_pattern_stores_its_child_once():
    from cadjoint.sdf.transforms.patterns import PolarPattern

    single = lower(Cylinder(radius=0.1, height=0.4))
    patterned = lower(
        PolarPattern(
            Translate(Cylinder(radius=0.1, height=0.4), [0.7, 0, 0]),
            count=8,
            axis="z",
        )
    )
    # eight copies are eight warps of one child: the cylinder's three patches appear once
    patches = lambda m: sum(1 for n in m.nodes if "patch" in n)  # noqa: E731
    assert patches(patterned) == patches(single) == 3


def test_the_census_names_a_box_by_its_six_faces_and_one_corner_blend():
    kinds = [k for k, _ in surfaces(lower(Box(size=[1, 1, 1])))]
    assert kinds == ["band", "rim"] + ["patch"] * 6


def test_surfaces_vanish_on_the_boundary_they_own():
    model = lower(Translate(Sphere(radius=0.5), [0.2, 0.0, 0.0]))
    ((kind, f),) = surfaces(model)
    assert kind == "patch"
    on = jnp.asarray([[0.7, 0.0, 0.0], [0.2, 0.5, 0.0], [0.2, 0.0, -0.5]])
    np.testing.assert_allclose(f(jnp.asarray(model.theta), on), 0.0, atol=1e-6)
