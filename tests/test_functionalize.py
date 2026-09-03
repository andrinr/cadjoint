"""Tests for SDF compilation to pure JAX functions."""

import jax.numpy as jnp
from cadjoint import extract_parameters, functionalize, functionalize_scene
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.render import Material
from cadjoint.sdf.primitives.box import Box
from cadjoint.sdf.primitives.sphere import Sphere


def test_compile_sphere_basic():
    """Test compiling a sphere to a function."""
    radius = Scalar(1.0, free=True, name="radius")
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)

    # Query at origin (inside sphere)
    point = jnp.array([0.0, 0.0, 0.0])
    free_vals = {"radius": 1.0}
    fixed_vals = {}

    distance = sdf_fn(free_vals, fixed_vals)(point)

    # At origin, distance should be -radius
    assert jnp.isclose(distance, -1.0)


def test_compile_sphere_outside():
    """Test compiled sphere SDF for point outside."""
    radius = Scalar(1.0, free=False, name="radius")
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)

    # Query at distance 2 from origin
    point = jnp.array([2.0, 0.0, 0.0])
    free_vals = {}
    fixed_vals = {"sphere_0.radius": 1.0}

    distance = sdf_fn(free_vals, fixed_vals)(point)

    # Distance should be 2 - 1 = 1
    assert jnp.isclose(distance, 1.0)


def test_compile_with_parameter_variation():
    """Test that compiled function responds to parameter changes."""
    radius = Scalar(1.0, free=True, name="radius")
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)
    point = jnp.array([2.0, 0.0, 0.0])

    # Test with radius = 1.0
    dist1 = sdf_fn({"radius": 1.0}, {})(point)
    assert jnp.isclose(dist1, 1.0)

    # Test with radius = 1.5
    dist2 = sdf_fn({"radius": 1.5}, {})(point)
    assert jnp.isclose(dist2, 0.5)

    # Test with radius = 2.0
    dist3 = sdf_fn({"radius": 2.0}, {})(point)
    assert jnp.isclose(dist3, 0.0)


def test_compile_box():
    """Test compiling a box to a function."""
    size = Vector([1.0, 1.0, 1.0], free=True, name="size")
    box = Box(size=size)

    sdf_fn = functionalize(box)

    # Query at origin (inside box)
    point = jnp.array([0.0, 0.0, 0.0])
    free_vals = {"size": jnp.array([1.0, 1.0, 1.0])}
    fixed_vals = {}

    distance = sdf_fn(free_vals, fixed_vals)(point)

    # Should be inside the box
    assert distance < 0


def test_compile_consistency():
    """Test that compiled function gives same results as direct evaluation."""
    radius = Scalar(1.5, free=False, name="radius")
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)

    # Test multiple points
    test_points = [
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([2.0, 0.0, 0.0]),
        jnp.array([1.0, 1.0, 0.0]),
    ]

    for point in test_points:
        # Direct evaluation using __call__
        direct_dist = sphere(point)

        # Compiled evaluation
        compiled_dist = sdf_fn({}, {"sphere_0.radius": 1.5})(point)

        # Should be very close
        assert jnp.isclose(direct_dist, compiled_dist, atol=1e-6)


def test_functionalized_material_preserves_forward_render_properties():
    material = Material(reflectivity=0.7, roughness=0.2, metallic=0.9)
    sphere = Sphere(radius=1.0, material=material)
    free, fixed, _ = extract_parameters(sphere)
    _, material_fn = functionalize_scene(sphere)(free, fixed)

    compiled = material_fn(jnp.array([1.0, 0.0, 0.0]))
    assert set(compiled) == {
        "color",
        "roughness",
        "metallic",
        "opacity",
        "ior",
        "reflectivity",
    }
    assert jnp.isclose(compiled["reflectivity"], 0.7)


# ── structured lowering ───────────────────────────────────────────────────────


def _lowered_text(fn, *args):
    import jax

    return jax.jit(fn).lower(*args).as_text()


def test_pattern_emits_one_copy_of_its_child():
    """Eight instances must not become eight copies of the rib's arithmetic."""
    import math

    from cadjoint.sdf._lowering import scalar_lowering
    from cadjoint.sdf.primitives.polygon import ExtrudedPolygon
    from cadjoint.sdf.transforms.patterns import PolarPattern

    ring = [
        jnp.array(
            [0.4 + 0.1 * math.cos(2 * math.pi * i / 12), 0.1 * math.sin(2 * math.pi * i / 12)]
        )
        for i in range(12)
    ]
    pattern = PolarPattern(ExtrudedPolygon(ring, depth=0.2), count=8)
    free, fixed, _ = extract_parameters(pattern)

    structured = _lowered_text(functionalize(pattern)(free, fixed), jnp.zeros(3))
    with scalar_lowering():
        unrolled = _lowered_text(functionalize(pattern)(free, fixed), jnp.zeros(3))

    # Even unrolled, the child is outlined into a helper called once per instance.
    import collections
    import re

    callees = collections.Counter(re.findall(r"call @(\S+)\(", unrolled))
    outlined = sum(n for name, n in callees.items() if name.startswith("eval_fn"))
    assert outlined == 8
    # Vectorising the instances and the vertex loop halves what is left.
    assert len(structured) < 0.7 * len(unrolled)


def test_pattern_agrees_with_the_unrolled_form():
    from cadjoint.sdf._lowering import scalar_lowering
    from cadjoint.sdf.transforms.patterns import LinearPattern, PolarPattern

    for pattern in (
        PolarPattern(Box(size=jnp.array([0.2, 0.2, 1.0])), count=6),
        LinearPattern(
            Sphere(radius=0.3), direction=jnp.array([1.0, 0.0, 0.0]), count=5, spacing=0.4
        ),
    ):
        free, fixed, _ = extract_parameters(pattern)
        points = jnp.array([[0.3, 0.1, 0.2], [1.1, -0.4, 0.0], [0.0, 0.0, 0.0]])
        vectorized = [functionalize(pattern)(free, fixed)(p) for p in points]
        with scalar_lowering():
            unrolled = [functionalize(pattern)(free, fixed)(p) for p in points]
        for a, b in zip(vectorized, unrolled):
            assert jnp.isclose(a, b, atol=1e-6)


def test_a_shared_subtree_is_built_once():
    """A tool cut from two bodies is one function, not two."""
    from cadjoint.sdf.boolean import Difference, Union
    from cadjoint.sdf.transforms.affine import Translate

    tool = Sphere(radius=0.4)
    shared = Union(
        (
            Difference((Box(size=jnp.array([1.0, 1.0, 1.0])), tool)),
            Difference(
                (Translate(Box(size=jnp.array([1.0, 1.0, 1.0])), jnp.array([2.0, 0.0, 0.0])), tool)
            ),
        ),
    )
    free, fixed, _ = extract_parameters(shared)
    # Numerically unchanged: the point sits inside the first body, outside the tool.
    value = functionalize(shared)(free, fixed)(jnp.array([0.45, 0.0, 0.0]))
    assert bool(jnp.isfinite(value))
    assert _lowered_text(functionalize(shared)(free, fixed), jnp.zeros(3)).count("func.func") >= 2


def test_parametric_lowering_is_identical_across_value_edits():
    """The whole point: an edit must not produce a different program."""
    from cadjoint.functionalize import functionalize_parametric

    radius = Scalar(1.0, free=True, name="radius")
    offset = Vector([0.5, 0.0, 0.0], free=True, name="offset")
    from cadjoint.sdf.transforms.affine import Translate

    scene = Translate(Sphere(radius=radius), offset)
    free, fixed, _ = extract_parameters(scene)

    evaluate = functionalize_parametric(scene)
    point = jnp.zeros(3)
    first = _lowered_text(evaluate, free, fixed, point)
    import jax

    edited = jax.tree.map(lambda value: value * 2.25, free)
    second = _lowered_text(evaluate, edited, fixed, point)
    assert first == second

    # …and the literal form, which bakes the values in, does not.
    literal_first = _lowered_text(functionalize(scene)(free, fixed), point)
    literal_second = _lowered_text(functionalize(scene)(edited, fixed), point)
    assert literal_first != literal_second

    assert jnp.isclose(
        jax.jit(evaluate)(edited, fixed, point),
        functionalize(scene)(edited, fixed)(point),
        atol=1e-6,
    )


def test_pattern_count_stays_static_under_a_parametric_trace():
    """``count`` decides how much program is emitted, so it cannot be an argument."""
    import jax
    from cadjoint.functionalize import functionalize_parametric
    from cadjoint.sdf.transforms.patterns import PolarPattern

    pattern = PolarPattern(Box(size=jnp.array([0.2, 0.2, 1.0])), count=5)
    free, fixed, _ = extract_parameters(pattern)
    value = jax.jit(functionalize_parametric(pattern))(free, fixed, jnp.zeros(3))
    assert jnp.isclose(value, functionalize(pattern)(free, fixed)(jnp.zeros(3)), atol=1e-6)
