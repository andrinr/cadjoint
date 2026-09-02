"""The uniform-backed shader: one module, every parameter value.

Inlining design parameters as float literals makes a slider drag a full
recompile of a hundred-kilobyte module to change three constants.  These tests
pin the alternative: identical source for every value, a documented buffer
layout, and the same field the literal form computes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.backends.wgsl import (
    PARAMETER_SLOT_BYTES,
    ShaderProgram,
    compile_scene_to_wgsl,
    compile_scene_with_uniforms,
)
from cadjoint.extraction import apply_parameters, extract_parameters
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import Sphere
from cadjoint.sdf.transforms.affine import Translate


def _scene():
    radius = Scalar(0.75, free=True, name="radius")
    offset = Vector([1.4, 0.0, 0.0], free=True, name="offset")
    return Union(
        (
            Sphere(radius, material=Material(color=[0.9, 0.3, 0.2])),
            Translate(Sphere(0.6, material=Material(color=[0.1, 0.4, 0.9])), offset),
        ),
        smoothness=0.05,
    )


def test_the_module_is_identical_across_parameter_edits():
    scene = _scene()
    before = compile_scene_with_uniforms(scene)

    free, _, _ = extract_parameters(scene)
    apply_parameters(scene, {name: value * 1.7 for name, value in free.items()})
    after = compile_scene_with_uniforms(scene)

    assert before.wgsl == after.wgsl
    assert [p.name for p in before.parameters] == [p.name for p in after.parameters]
    assert not np.array_equal(before.buffer(), after.buffer())


def test_the_literal_form_is_not():
    scene = _scene()
    before = compile_scene_to_wgsl(scene)
    free, _, _ = extract_parameters(scene)
    apply_parameters(scene, {name: value * 1.7 for name, value in free.items()})
    assert compile_scene_to_wgsl(scene) != before


def test_the_buffer_layout_is_one_vec4_slot_per_parameter():
    program = compile_scene_with_uniforms(_scene())

    assert isinstance(program, ShaderProgram)
    assert program.buffer_bytes == len(program.parameters) * PARAMETER_SLOT_BYTES
    for index, parameter in enumerate(program.parameters):
        assert parameter.offset == index * PARAMETER_SLOT_BYTES
        assert 1 <= parameter.components <= 4
        assert len(parameter.value) == parameter.components
    assert program.buffer().size == program.buffer_bytes // 4

    # The declared names are the ones extract_parameters uses.
    free, fixed, _ = extract_parameters(_scene())
    for parameter in program.parameters:
        assert parameter.name in (free if parameter.free else fixed)


def test_the_module_declares_the_uniform_it_reads():
    program = compile_scene_with_uniforms(_scene())
    slots = len(program.parameters)
    assert f"array<vec4<f32>, {slots}>" in program.wgsl
    assert (
        f"@group({program.group}) @binding({program.binding}) "
        "var<uniform> sdf_parameters: SdfParameters;" in program.wgsl
    )
    for entry_point in ("sdf", "material_base", "material_optics"):
        assert f"fn {entry_point}(p: vec3<f32>)" in program.wgsl


def test_the_flag_on_compile_scene_to_wgsl_selects_the_same_program():
    program = compile_scene_to_wgsl(_scene(), uniforms=True)
    assert isinstance(program, ShaderProgram)
    assert program.wgsl == compile_scene_with_uniforms(_scene()).wgsl


def test_the_uniform_module_compiles_on_a_real_gpu():
    wgpu = pytest.importorskip("wgpu", reason="wgpu is needed to validate WGSL")
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()
    except Exception as error:  # pragma: no cover - depends on the host
        pytest.skip(f"No usable WebGPU adapter: {error}")
    program = compile_scene_with_uniforms(_scene())
    device.create_shader_module(code=program.wgsl, label="uniform scene")


def test_the_uniform_form_evaluates_the_same_field():
    """The refactor must not move the surface, only where the numbers live."""
    from cadjoint.functionalize import functionalize_scene

    scene = _scene()
    free, fixed, _ = extract_parameters(scene)
    reference = functionalize_scene(scene)(free, fixed)[0]
    program = compile_scene_with_uniforms(scene)

    # Every slot's recorded value must match the tree it was read from.
    for parameter in program.parameters:
        source = (free if parameter.free else fixed)[parameter.name]
        np.testing.assert_allclose(
            np.asarray(source, dtype=np.float32).reshape(-1),
            np.asarray(parameter.value, dtype=np.float32),
            rtol=0,
            atol=0,
        )
    assert float(reference(jnp.zeros(3))) < 0.0
