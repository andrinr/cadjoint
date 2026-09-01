"""Tests for compiling distance and material point queries into one WGSL module."""

import re

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.backends.wgsl import compile_scene_to_wgsl, compile_sdf_to_wgsl
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import Sphere
from cadjoint.sdf.transforms.affine import Translate


def _material_scene():
    red = Translate(
        Sphere(
            0.75,
            material=Material(
                color=[1.0, 0.0, 0.0],
                roughness=0.2,
                metallic=0.1,
            ),
        ),
        jnp.array([-2.0, 0.0, 0.0]),
    )
    blue_glass = Translate(
        Sphere(
            0.75,
            material=Material(
                color=[0.0, 0.0, 1.0],
                roughness=0.05,
                metallic=0.8,
                opacity=0.4,
                ior=1.6,
                reflectivity=0.3,
            ),
        ),
        jnp.array([2.0, 0.0, 0.0]),
    )
    return Union(red, blue_glass, smoothness=0.0)


def test_scene_material_selection_follows_union_and_transform_coordinates():
    from cadjoint import extract_parameters, functionalize_scene

    scene = _material_scene()
    free_parameters, fixed_parameters, _ = extract_parameters(scene)
    _, material_at = functionalize_scene(scene)(free_parameters, fixed_parameters)

    left = material_at(jnp.array([-2.0, 0.0, 0.0]))
    right = material_at(jnp.array([2.0, 0.0, 0.0]))

    np.testing.assert_allclose(left["color"], [1.0, 0.0, 0.0])
    assert jnp.isclose(left["roughness"], 0.2)
    np.testing.assert_allclose(right["color"], [0.0, 0.0, 1.0])
    assert jnp.isclose(right["metallic"], 0.8)
    assert jnp.isclose(right["opacity"], 0.4)
    assert jnp.isclose(right["ior"], 1.6)
    assert jnp.isclose(right["reflectivity"], 0.3)


def test_scene_wgsl_has_stable_distance_and_packed_material_signatures():
    source = compile_scene_to_wgsl(_material_scene())

    assert "fn sdf(p: vec3<f32>) -> f32" in source
    assert "fn material_base(p: vec3<f32>) -> vec4<f32>" in source
    assert "fn material_optics(p: vec3<f32>) -> vec4<f32>" in source

    declarations = re.findall(r"^fn ([A-Za-z_][A-Za-z0-9_]*)\(", source, re.MULTILINE)
    assert len(declarations) == len(set(declarations))
    base_helpers = {
        name.removeprefix("material_base__")
        for name in declarations
        if name.startswith("material_base__")
    }
    optics_helpers = {
        name.removeprefix("material_optics__")
        for name in declarations
        if name.startswith("material_optics__")
    }
    assert base_helpers & optics_helpers
    for prefix, helpers in (
        ("material_base__", base_helpers),
        ("material_optics__", optics_helpers),
    ):
        assert all(source.count(f"{prefix}{helper}(") >= 2 for helper in helpers)


def test_constant_material_keeps_point_query_signatures():
    source = compile_scene_to_wgsl(
        Sphere(
            1.0,
            material=Material(
                color=[0.2, 0.4, 0.8],
                roughness=0.3,
                metallic=0.7,
                opacity=0.6,
                ior=1.5,
                reflectivity=0.2,
            ),
        )
    )

    assert "fn material_base(p: vec3<f32>) -> vec4<f32>" in source
    assert "fn material_optics(p: vec3<f32>) -> vec4<f32>" in source
    for value in ("0.200000", "0.400000", "0.800000", "0.300000", "0.700000", "0.600000"):
        assert value in source


def test_legacy_sdf_compiler_still_emits_only_the_sdf_contract():
    source = compile_sdf_to_wgsl(Sphere(1.0))

    assert "fn sdf(p: vec3<f32>) -> f32" in source
    assert "material_base" not in source
    assert "material_optics" not in source


def test_wgsl_emitter_validates_custom_entry_point_and_output():
    from cadjoint.backends.wgsl._wgsl_emitter import StableHLOToWGSL

    compiler = StableHLOToWGSL()
    point = jnp.zeros(3, dtype=jnp.float32)

    with pytest.raises(ValueError, match="valid identifier"):
        compiler.compile(lambda p: p[0], point, entry_point="invalid-name")

    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        compiler.compile(
            lambda p: p,
            point,
            entry_point="material_base",
            output_shape=(4,),
            output_description="float32 vector with shape (4,)",
        )
