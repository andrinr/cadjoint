"""Bounding-box culling: the field it computes is the field without it.

:mod:`cadjoint.backends.wgsl._culling` makes the shader *skip* a boolean's
operand when the operand's bounding box is far enough away that it provably
cannot change the running result.  Skipping work is only worth anything if
the picture is unchanged, so these tests attack the two things that could
make it change:

1. The **bound** (``cadjoint.sdf._lowering.node_bounds``) might not be a
   lower bound — if a node's field dips below the distance to its box, a
   skip fires while the operand still mattered, and geometry disappears.
2. The **threshold** might be too tight for the smooth minimum's blend band.

Both are checked the same way: against the flat evaluation, at points drawn
from the whole scene volume, for every shipped scene and for a battery of
node types the scenes do not all cover.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.backends.wgsl._culling import culled_scene_sdf, scene_bounds
from cadjoint.extraction import extract_parameters
from cadjoint.functionalize import functionalize_scene
from cadjoint.geometry.parameters import Scalar
from cadjoint.render import Material
from cadjoint.sdf import (
    Box,
    Capsule,
    Cylinder,
    Difference,
    Intersection,
    LinearPattern,
    Mirror,
    Offset,
    PolarPattern,
    Rotate,
    RoundBox,
    Scale,
    Shell,
    Sphere,
    Torus,
    Translate,
    Union,
)
from cadjoint.sdf._lowering import Bounds, box_distance, node_bounds, scalar_lowering

#: Float32 sphere tracing works to about 1e-6 of the scene's extent; the
#: contract in the brief is that culling stays an order below that.
TOLERANCE = 1e-6

SCENES = Path(__file__).resolve().parents[2] / "scenes"


def _cases() -> list:
    """One scene per node family the bounds rules distinguish."""
    ball = Sphere(0.4, material=Material(color=[0.8, 0.2, 0.2]))
    bar = Box([0.6, 0.2, 0.2])
    return [
        ("hard union", Union((ball, Translate(bar, [0.9, 0.0, 0.0])), smoothness=0.0)),
        ("smooth union", Union((ball, Translate(bar, [0.9, 0.0, 0.0])), smoothness=0.08)),
        (
            "deep union",
            Union(
                tuple(Translate(Sphere(0.25), [0.7 * i - 1.4, 0.0, 0.0]) for i in range(5)),
                smoothness=0.05,
            ),
        ),
        ("difference", Difference((Box([0.7, 0.7, 0.3]), Sphere(0.5)), smoothness=0.0)),
        ("smooth difference", Difference((Box([0.7, 0.7, 0.3]), Sphere(0.5)), smoothness=0.06)),
        ("intersection", Intersection((Box([0.7, 0.7, 0.3]), Sphere(0.8)))),
        ("rotate", Union((Rotate(bar, [0.3, 1.0, 0.4], 0.7), ball), smoothness=0.03)),
        ("mirror", Union((Mirror(Translate(bar, [0.8, 0.0, 0.0]), "x"), ball))),
        ("uniform scale", Union((Scale(bar, [1.7, 1.7, 1.7]), ball), smoothness=0.02)),
        ("non-uniform scale", Union((Scale(bar, [1.9, 0.6, 1.2]), ball), smoothness=0.02)),
        ("shell", Union((Shell(Sphere(0.6), 0.08), Translate(ball, [1.2, 0.0, 0.0])))),
        ("offset", Union((Offset(bar, 0.12), Translate(ball, [1.2, 0.0, 0.0])))),
        (
            "linear pattern",
            Union(
                (LinearPattern(Sphere(0.2), direction=[1.0, 0.0, 0.0], count=6, spacing=0.5), ball)
            ),
        ),
        (
            "polar pattern",
            Union(
                (
                    PolarPattern(Translate(Sphere(0.18), [0.7, 0.0, 0.0]), count=7, axis="z"),
                    ball,
                )
            ),
        ),
        # A pattern may leave instances out. Culling unrolls the instances
        # itself, so it has to unroll the *kept* ones: emitting the whole run
        # puts back exactly the copies the scene asked to omit, and the
        # difference only shows where the skipped copy would have been.
        (
            "linear pattern with a gap",
            Union(
                (
                    LinearPattern(
                        Sphere(0.2),
                        direction=[1.0, 0.0, 0.0],
                        count=6,
                        spacing=0.5,
                        skip=(2, 4),
                    ),
                    ball,
                )
            ),
        ),
        (
            "polar pattern with a gap",
            Union(
                (
                    PolarPattern(
                        Translate(Sphere(0.18), [0.7, 0.0, 0.0]),
                        count=8,
                        axis="z",
                        skip=(3,),
                    ),
                    ball,
                )
            ),
        ),
        ("torus", Union((Torus(0.6, 0.15), Translate(ball, [0.0, 0.0, 0.7])))),
        ("capsule", Union((Capsule(0.15, 0.5), Translate(ball, [0.9, 0.0, 0.0])))),
        ("cylinder", Union((Cylinder(0.35, 0.4), Translate(ball, [0.9, 0.0, 0.0])))),
        ("round box", Union((RoundBox([0.4, 0.3, 0.2], 0.06), Translate(ball, [1.1, 0.0, 0.0])))),
        (
            "nested",
            Difference(
                (
                    Union(
                        (Box([0.8, 0.5, 0.3]), Translate(Sphere(0.4), [0.8, 0.0, 0.0])),
                        smoothness=0.05,
                    ),
                    LinearPattern(
                        Cylinder(0.1, 0.6), direction=[1.0, 0.0, 0.0], count=4, spacing=0.4
                    ),
                ),
                smoothness=0.0,
            ),
        ),
    ]


def _shipped_scenes() -> list:
    """Every scene the playground ships, executed the way the worker does."""
    from cadjoint.fem.simmesh import capture_sim_meshes
    from cadjoint.fem.study import capture_studies
    from cadjoint.optimize import capture_optimizations

    found = []
    for path in sorted(SCENES.glob("*.py")):
        namespace: dict = {"__builtins__": __builtins__, "__name__": "__cadjoint_playground__"}
        with capture_sim_meshes(), capture_studies(), capture_optimizations():
            exec(compile(path.read_text(), str(path), "exec"), namespace, namespace)
        found.append((path.stem, namespace["scene"]))
    return found


def _sample(scene, count: int, seed: int) -> np.ndarray:
    """Points filling the scene's bounds plus a margin, so both sides are hit."""
    free, fixed, _ = extract_parameters(scene)
    bounds = scene_bounds(scene, free, fixed)
    if bounds is None:
        low, high = np.full(3, -3.0), np.full(3, 3.0)
    else:
        extent = np.asarray(bounds.half) * 2.0 + 0.5
        low = np.asarray(bounds.center) - extent
        high = np.asarray(bounds.center) + extent
    return np.random.default_rng(seed).uniform(low, high, size=(count, 3)).astype(np.float32)


def _deviation(scene, points) -> float:
    """Largest |culled - flat| over ``points``."""
    free, fixed, _ = extract_parameters(scene)
    with scalar_lowering():
        flat = jax.jit(jax.vmap(functionalize_scene(scene)(free, fixed)[0]))
        culled = jax.jit(jax.vmap(culled_scene_sdf(scene)(free, fixed)))
        return float(np.max(np.abs(np.asarray(flat(points)) - np.asarray(culled(points)))))


@pytest.mark.parametrize("name,scene", _cases(), ids=[name for name, _ in _cases()])
def test_culling_reproduces_the_flat_field(name, scene):
    """Every node family: the culled field equals the one that skips nothing."""
    assert _deviation(scene, _sample(scene, 20_000, seed=7)) <= TOLERANCE


@pytest.mark.parametrize("name,scene", _shipped_scenes(), ids=[n for n, _ in _shipped_scenes()])
def test_culling_reproduces_the_flat_field_in_shipped_scenes(name, scene):
    """The real scenes, at the 100k points the brief asks for."""
    assert _deviation(scene, _sample(scene, 100_000, seed=11)) <= TOLERANCE


@pytest.mark.parametrize("name,scene", _cases(), ids=[name for name, _ in _cases()])
def test_the_box_distance_is_a_lower_bound(name, scene):
    """(★): outside a node's box, its field is at least the distance to it.

    This is the property every skip rests on, tested at the root — which is
    the strongest statement, because a root bound is built from its
    children's and inherits any error they made.
    """
    free, fixed, _ = extract_parameters(scene)
    bounds = scene_bounds(scene, free, fixed)
    assert bounds is not None, f"{name} reported no bounds"
    points = _sample(scene, 40_000, seed=13)
    with scalar_lowering():
        field = np.asarray(jax.jit(jax.vmap(functionalize_scene(scene)(free, fixed)[0]))(points))
        distance = np.asarray(jax.jit(jax.vmap(lambda p: box_distance(p, bounds)))(points))
    outside = distance > 0
    assert outside.sum() > 100, f"{name}: too few points outside the box to be a test"
    assert np.min(field[outside] - distance[outside]) >= -TOLERANCE


def test_an_unbounded_node_reports_no_bounds():
    """A node that cannot promise (★) says so, and is then never skipped."""
    from cadjoint.sdf.primitives import ExtrudedPolygon

    profile = ExtrudedPolygon([[0, 0], [1, 0], [1, 1]], depth=0.5, draft=0.2)
    assert node_bounds(profile, {"draft": 0.2, "depth": 0.5, "v0": jnp.zeros(2)}, []) is None


def test_the_bound_follows_a_parameter_edit():
    """A radius slider moves the box, so a skip cannot outlive its geometry.

    The whole point of the uniform form is that a value can change without
    recompiling; a bound baked from the compile-time value would let the
    grown sphere fall outside its own box and vanish.
    """
    radius = Scalar(0.3, free=True, name="radius")
    scene = Union((Sphere(radius), Translate(Sphere(0.2), [2.0, 0.0, 0.0])), smoothness=0.0)
    free, fixed, _ = extract_parameters(scene)
    small = scene_bounds(scene, free, fixed)
    grown = scene_bounds(scene, {**free, "radius": jnp.asarray(1.1)}, fixed)
    assert float(np.max(np.asarray(grown.half))) > float(np.max(np.asarray(small.half)))


def test_box_distance_is_zero_inside_and_euclidean_outside():
    box = Bounds(center=jnp.zeros(3), half=jnp.ones(3) * 0.5)
    assert float(box_distance(jnp.zeros(3), box)) == 0.0
    assert float(box_distance(jnp.array([0.5, 0.0, 0.0]), box)) == 0.0
    assert float(box_distance(jnp.array([1.5, 0.0, 0.0]), box)) == pytest.approx(1.0)
    assert float(box_distance(jnp.array([1.5, 1.5, 0.5]), box)) == pytest.approx(np.sqrt(2.0))


# ── Culling as a render toggle ───────────────────────────────────────────────
# The viewer offers culling as a switch, and the switch must not be a
# recompile. It is not: the margin every skip test is compared against is an
# *argument*, bound to a reserved slot in the scene shader's own parameter
# buffer, so turning culling off is a buffer write like any other render
# setting. An infinite margin makes every test false, which is the flat field.


@pytest.mark.parametrize("name,scene", _cases(), ids=[name for name, _ in _cases()])
def test_an_infinite_margin_is_the_flat_field(name, scene):
    """Culling off computes exactly what skipping nothing computes."""
    from cadjoint.backends.wgsl.codegen import CULL_DISABLED_MARGIN

    free, fixed, _ = extract_parameters(scene)
    points = _sample(scene, 20_000, seed=17)
    with scalar_lowering():
        flat = jax.jit(jax.vmap(functionalize_scene(scene)(free, fixed)[0]))(points)
        off = jax.jit(
            jax.vmap(culled_scene_sdf(scene)(free, fixed, jnp.float32(CULL_DISABLED_MARGIN)))
        )(points)
    assert float(np.max(np.abs(np.asarray(flat) - np.asarray(off)))) <= TOLERANCE


@pytest.mark.parametrize("stem", ["starter", "end_cap"])
def test_both_switch_positions_agree_on_a_shipped_scene(stem):
    """On and off are the same field, at 100k points of a real part."""
    from cadjoint.backends.wgsl._culling import CULL_MARGIN
    from cadjoint.backends.wgsl.codegen import CULL_DISABLED_MARGIN

    scene = dict(_shipped_scenes())[stem]
    free, fixed, _ = extract_parameters(scene)
    points = _sample(scene, 100_000, seed=19)
    with scalar_lowering():
        build = culled_scene_sdf(scene)
        on = jax.jit(jax.vmap(build(free, fixed, jnp.float32(CULL_MARGIN))))(points)
        off = jax.jit(jax.vmap(build(free, fixed, jnp.float32(CULL_DISABLED_MARGIN))))(points)
    assert float(np.max(np.abs(np.asarray(on) - np.asarray(off)))) <= TOLERANCE


def test_the_margin_reaches_every_skip_test():
    """A margin bound after the tree is built must not miss a node.

    The margin is read late, out of the enclosing scope, rather than threaded
    through the outlining as a fourth argument. That works because the skip
    tests run at trace time — but if any test had captured the *value*
    instead, it would keep culling with the old margin and this would catch
    it: an infinite margin that reached only some tests would still skip at
    the others, and the two positions would disagree somewhere.
    """
    from cadjoint.backends.wgsl.codegen import CULL_DISABLED_MARGIN

    scene = dict(_shipped_scenes())["end_cap"]
    free, fixed, _ = extract_parameters(scene)
    points = _sample(scene, 40_000, seed=23)
    with scalar_lowering():
        flat = jax.jit(jax.vmap(functionalize_scene(scene)(free, fixed)[0]))(points)
        off = jax.jit(
            jax.vmap(culled_scene_sdf(scene)(free, fixed, jnp.float32(CULL_DISABLED_MARGIN)))
        )(points)
    assert float(np.max(np.abs(np.asarray(flat) - np.asarray(off)))) <= TOLERANCE


def test_the_program_reserves_a_slot_for_the_margin():
    """The buffer layout the viewer writes: parameters, NaN, margin."""
    from cadjoint.backends.wgsl import PARAMETER_SLOT_BYTES, compile_scene_with_uniforms
    from cadjoint.backends.wgsl._culling import CULL_MARGIN

    program = compile_scene_with_uniforms(
        Union((Sphere(Scalar(0.6, free=True, name="r")), Sphere(0.4)), smoothness=0.05)
    )
    slots = len(program.parameters)
    assert program.nan_offset == slots * PARAMETER_SLOT_BYTES
    assert program.cull_margin_offset == (slots + 1) * PARAMETER_SLOT_BYTES
    assert program.buffer_bytes == (slots + 2) * PARAMETER_SLOT_BYTES
    assert program.as_dict()["cull_margin_offset"] == program.cull_margin_offset
    # The default the packer writes is culling on.
    assert program.buffer()[program.cull_margin_offset // 4] == np.float32(CULL_MARGIN)


def test_the_uncompiled_form_still_takes_its_margin_from_the_default():
    """The literal form has no buffer, so its margin stays a constant."""
    from cadjoint.backends.wgsl import compile_scene_to_wgsl

    code = compile_scene_to_wgsl(Union((Sphere(0.6), Sphere(0.4)), smoothness=0.05))
    assert isinstance(code, str)
    assert "sdf_parameters" not in code
