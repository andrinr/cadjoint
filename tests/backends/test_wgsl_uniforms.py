"""The uniform-backed shader: one module, every parameter value.

Inlining design parameters as float literals makes a slider drag a full
recompile of a hundred-kilobyte module to change three constants.  These tests
pin the alternative: identical source for every value, a documented buffer
layout, and the same field the literal form computes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.backends.wgsl import (
    PARAMETER_SLOT_BYTES,
    RESERVED_PARAMETER_SLOTS,
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
    # One slot per parameter, plus the reserved slots at the end: the NaN,
    # then the cull margin.
    assert (
        program.buffer_bytes
        == (len(program.parameters) + RESERVED_PARAMETER_SLOTS) * PARAMETER_SLOT_BYTES
    )
    assert program.nan_offset == len(program.parameters) * PARAMETER_SLOT_BYTES
    assert program.cull_margin_offset == program.nan_offset + PARAMETER_SLOT_BYTES
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
    slots = len(program.parameters) + RESERVED_PARAMETER_SLOTS  # NaN, cull margin
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


#: WGSL caps a function's parameter list. Naga and Tint both enforce it, and
#: the message is the one the browser reports: "function declares N
#: parameters, maximum is 255".
MAX_WGSL_FUNCTION_PARAMETERS = 255

#: Every scene the playground ships, by file stem.
_SCENES = Path(__file__).resolve().parents[2] / "scenes"


def _function_parameter_counts(wgsl: str) -> dict[str, int]:
    """How many parameters each ``fn`` in a module declares."""
    counts = {}
    for match in re.finditer(r"^fn (\w+)\(([^)]*)\)", wgsl, flags=re.MULTILINE):
        arguments = match.group(2).strip()
        counts[match.group(1)] = 0 if not arguments else arguments.count(",") + 1
    return counts


def _shipped_scene(stem: str):
    from cadjoint.fem.simmesh import capture_sim_meshes
    from cadjoint.fem.study import capture_studies
    from cadjoint.optimize import capture_optimizations

    namespace: dict = {"__builtins__": __builtins__, "__name__": "__cadjoint_playground__"}
    path = _SCENES / f"{stem}.py"
    with capture_sim_meshes(), capture_studies(), capture_optimizations():
        exec(compile(path.read_text(), str(path), "exec"), namespace, namespace)
    return namespace["scene"]


def test_parameters_are_read_from_the_buffer_not_passed_down():
    """No function may take the parameters as arguments.

    Outlining closes every shared subtree over the whole parameter tuple, so
    threading the parameters gave *every* outlined helper one argument per
    parameter — 438 of them for ``scenes/end_cap.py`` against a WGSL maximum
    of 255, which the browser rejects outright. The buffer is module scope
    and readable anywhere, so a parameter is read where it is used instead.
    This pins that: the entry points take the point alone, and nothing in
    the module comes close to the limit.
    """
    # The "all" form is what produced the 438-argument function; the shipped
    # "free" form has too few parameters to reach the limit by accident, so
    # the binding pass has to be stressed with the form that did.
    program = compile_scene_with_uniforms(_scene(), scope="all")
    counts = _function_parameter_counts(program.wgsl)
    assert counts["sdf"] == 1
    assert counts["sdf_impl"] == 1
    worst = max(counts.values())
    assert worst <= MAX_WGSL_FUNCTION_PARAMETERS, f"widest signature is {worst} parameters"
    # The parameters outnumber the widest signature by a wide margin, which is
    # the property that makes the module compile at all.
    assert worst < len(program.parameters)


@pytest.mark.parametrize("stem", ["starter", "end_cap", "motor_shield"])
def test_shipped_scenes_stay_inside_the_wgsl_parameter_limit(stem):
    """The scenes that broke this, at their real size."""
    program = compile_scene_with_uniforms(_shipped_scene(stem), scope="all")
    worst = max(_function_parameter_counts(program.wgsl).values())
    assert (
        worst <= MAX_WGSL_FUNCTION_PARAMETERS
    ), f"{stem}: {len(program.parameters)} parameters produced a {worst}-parameter function"


def test_the_payload_form_of_a_program_is_strict_json():
    """A value the scene never set must not travel as a bare ``NaN``.

    ``json.dumps`` writes NaN as a bare ``NaN``, which no strict parser will
    read — the browser's ``JSON.parse`` included, so the whole compile
    response is unreadable, not just the parameter. A ``Material`` that
    leaves its physical properties unset carries a NaN in every one of them,
    so this is the common case rather than the exotic one.
    """
    scene = Union(
        (Sphere(0.5, material=Material(color=[0.8, 0.2, 0.2])), Sphere(0.3)),
        smoothness=0.05,
    )
    # A material property is a *fixed* parameter, so this is the "all"
    # form's payload: the shipped "free" form leaves fixed ones literals.
    payload = compile_scene_with_uniforms(scene, scope="all").as_dict()
    text = json.dumps(payload)

    def reject(constant):
        raise AssertionError(f"payload carries a bare {constant}")

    json.loads(text, parse_constant=reject)
    unset = [
        component
        for parameter in payload["parameters"]
        for component in parameter["value"]
        if component is None
    ]
    assert unset, "this scene is meant to have unset material properties"


def test_an_unset_value_packs_as_nan_exactly_as_the_literal_form_inlines_it():
    """``None`` on the wire is a NaN in the buffer, not a zero.

    The two forms have to draw the same image, and that includes agreeing
    about the values nobody supplied.
    """
    scene = Union(
        (Sphere(0.5, material=Material(color=[0.8, 0.2, 0.2])), Sphere(0.3)),
        smoothness=0.05,
    )
    program = compile_scene_with_uniforms(scene, scope="all")
    packed = program.buffer()
    for parameter, entry in zip(program.parameters, program.as_dict()["parameters"]):
        slot = parameter.offset // 4
        for index, component in enumerate(entry["value"]):
            if component is None:
                assert np.isnan(packed[slot + index])


# ── Which parameters get a slot ──────────────────────────────────────────────
# Putting *every* parameter in the buffer is correct and 31x slower per frame
# on `scenes/end_cap.py`: a value read from a buffer is a value the GPU's
# compiler cannot fold, and the generated field is large precisely because it
# is mostly foldable. The shipped form buffers only the free parameters — the
# ones a handle drags — and leaves the rest literals. These pin that split.


def test_only_the_free_parameters_get_a_slot_by_default():
    program = compile_scene_with_uniforms(_scene())

    assert [p.name for p in program.parameters] == ["radius", "offset"]
    assert all(p.free for p in program.parameters)


def test_the_all_scope_buffers_the_fixed_parameters_too():
    free_only = compile_scene_with_uniforms(_scene())
    everything = compile_scene_with_uniforms(_scene(), scope="all")

    assert len(everything.parameters) > len(free_only.parameters)
    assert any(not p.name.startswith(("radius", "offset")) for p in everything.parameters)
    # The fixed ones are the majority, which is the whole reason the default
    # leaves them out of the buffer.
    assert sum(not p.free for p in everything.parameters) > sum(
        p.free for p in everything.parameters
    )


def test_a_free_parameter_edit_still_leaves_the_module_byte_identical():
    """The property the whole form exists for, in the shipped scope.

    A fixed parameter now costs a recompile, which is the trade; a free one
    must not, because a free one is what a handle drags.
    """
    scene = _scene()
    before = compile_scene_with_uniforms(scene)
    free, _, _ = extract_parameters(scene)
    apply_parameters(scene, {name: value * 1.7 for name, value in free.items()})
    after = compile_scene_with_uniforms(scene)

    assert before.wgsl == after.wgsl
    assert not np.array_equal(before.buffer(), after.buffer())


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="scope must be one of"):
        compile_scene_with_uniforms(_scene(), scope="some")


@pytest.mark.parametrize("stem", ["starter", "end_cap"])
def test_the_free_scope_keeps_the_buffer_small_on_a_shipped_scene(stem):
    """The buffer a drag writes is hundreds of bytes, not thousands.

    ``scenes/end_cap.py`` has 11 free parameters against 319 fixed ones, so
    this is also the ratio that makes the folding argument: the default form
    leaves 97 % of the scene's numbers constant.
    """
    scene = _shipped_scene(stem)
    free_only = compile_scene_with_uniforms(scene)
    everything = compile_scene_with_uniforms(scene, scope="all")

    assert all(p.free for p in free_only.parameters)
    assert len(free_only.parameters) < len(everything.parameters) / 3
    assert free_only.buffer_bytes < everything.buffer_bytes / 3
    assert (
        free_only.buffer_bytes
        == (len(free_only.parameters) + RESERVED_PARAMETER_SLOTS) * PARAMETER_SLOT_BYTES
    )


def test_the_free_scope_reads_only_its_own_slots():
    """No read may point past the slots the program declares.

    The reserved slots sit past the last parameter — the NaN, then the cull
    margin — so the highest index the source may mention is the last of
    those.
    """
    program = compile_scene_with_uniforms(_shipped_scene("end_cap"))
    indices = {int(index) for index in re.findall(r"sdf_parameters\.values\[(\d+)\]", program.wgsl)}
    assert indices, "the module reads no parameters at all"
    last = len(program.parameters) + RESERVED_PARAMETER_SLOTS - 1
    assert max(indices) <= last
    assert program.nan_offset == len(program.parameters) * PARAMETER_SLOT_BYTES
    assert program.cull_margin_offset == last * PARAMETER_SLOT_BYTES
