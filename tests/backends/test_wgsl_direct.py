"""The direct WGSL backend must agree with the traced one, on the GPU.

`cadjoint.backends.wgsl.direct` writes WGSL by walking the SDF graph, with
no jaxpr in the path.  That makes it a *second implementation* of every
distance function it covers, and the only thing standing between a second
implementation and silent divergence is a test that runs both.

So this file does not compare source text or trust either emitter's own
account of itself.  It builds both shaders, dispatches each on the same
points on a real device, and requires the two fields to agree — and
requires both to agree with the Python the whole project differentiates,
which is the actual source of truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.backends.wgsl.codegen import compile_sdf_to_wgsl
from cadjoint.backends.wgsl.direct import UnsupportedNode, compile_sdf_direct
from cadjoint.construction import Axis
from cadjoint.geometry.parameters import Scalar, Vector, Vector2
from cadjoint.sdf.boolean import Difference, Intersection, Union
from cadjoint.sdf.primitives import Box, Capsule, Sphere, Torus
from cadjoint.sdf.primitives.loft import LoftedPolygon
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon, RevolvedPolygon
from cadjoint.sdf.transforms import Translate
from cadjoint.sdf.transforms.fields import Mirror, Offset, Shell
from cadjoint.sdf.transforms.patterns import LinearPattern, PolarPattern

wgpu = pytest.importorskip("wgpu", reason="the comparison runs both shaders on a device")

#: Points to evaluate both fields at. Deterministic, spanning inside,
#: outside and near-surface, because a backend can be wrong in only one.
_RNG = np.random.default_rng(20260906)
_POINTS = np.concatenate(
    [
        _RNG.uniform(-2.5, 2.5, size=(4096, 3)),
        _RNG.normal(0.0, 0.05, size=(512, 3)),  # clustered at the origin
    ]
).astype(np.float32)

_HARNESS = """
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = sdf(points[i].xyz);
}
"""


def _evaluate_on_device(wgsl, points: np.ndarray) -> np.ndarray:
    """Run `fn sdf(p) -> f32` from *wgsl* at every point, on the GPU.

    Accepts either plain source (the traced backend) or a `DirectProgram`,
    whose profile vertices are bound as a second group — the buffer its
    polygon kernel loops over instead of unrolling.
    """
    profile = getattr(wgsl, "vertices", None)
    source = str(wgsl)
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    module = device.create_shader_module(code=source + _HARNESS)

    # vec4 rather than vec3: WGSL pads a vec3 in a storage array to 16 bytes,
    # so a packed (N, 3) upload would be read misaligned.
    padded = np.zeros((len(points), 4), dtype=np.float32)
    padded[:, :3] = points
    usage = wgpu.BufferUsage
    inputs = device.create_buffer_with_data(data=padded.tobytes(), usage=usage.STORAGE)
    outputs = device.create_buffer(size=len(points) * 4, usage=usage.STORAGE | usage.COPY_SRC)
    layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
        ]
    )
    bind_group = device.create_bind_group(
        layout=layout,
        entries=[
            {"binding": 0, "resource": {"buffer": inputs, "offset": 0, "size": inputs.size}},
            {"binding": 1, "resource": {"buffer": outputs, "offset": 0, "size": outputs.size}},
        ],
    )
    layouts = [layout]
    groups = [bind_group]
    if profile is not None and len(profile):
        # vec2<f32> in a storage array packs at 8 bytes, so this one uploads
        # as-is rather than padded like the query points above.
        vertex_buffer = device.create_buffer_with_data(
            data=np.ascontiguousarray(profile, dtype=np.float32).tobytes(), usage=usage.STORAGE
        )
        profile_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                }
            ]
        )
        layouts.append(profile_layout)
        groups.append(
            device.create_bind_group(
                layout=profile_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": vertex_buffer,
                            "offset": 0,
                            "size": vertex_buffer.size,
                        },
                    }
                ],
            )
        )
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=layouts),
        compute={"module": module, "entry_point": "main"},
    )
    encoder = device.create_command_encoder()
    pass_ = encoder.begin_compute_pass()
    pass_.set_pipeline(pipeline)
    for index, group in enumerate(groups):
        pass_.set_bind_group(index, group)
    pass_.dispatch_workgroups((len(points) + 63) // 64)
    pass_.end()
    device.queue.submit([encoder.finish()])
    raw = device.queue.read_buffer(outputs)
    return np.frombuffer(raw, dtype=np.float32).copy()


def _python_field(scene, points: np.ndarray) -> np.ndarray:
    """The scene evaluated by the code the project differentiates."""
    import jax.numpy as jnp

    return np.asarray(jnp.asarray([scene(jnp.asarray(p)) for p in points]), dtype=np.float64)


def _ring(count: int, radius: float = 1.0) -> list:
    """A regular `count`-gon, the shape whose cost the exercise is about."""
    import math

    return [
        Vector2(
            [radius * math.cos(2 * math.pi * i / count), radius * math.sin(2 * math.pi * i / count)]
        )
        for i in range(count)
    ]


def _scenes() -> list[tuple[str, object]]:
    """Scenes built only from nodes the direct backend claims to cover."""
    return [
        ("sphere", Sphere(radius=Scalar(0.8, name="r"))),
        ("box", Box(size=Vector([0.6, 0.4, 0.9], name="size"))),
        ("torus", Torus(major_radius=Scalar(1.0, name="R"), minor_radius=Scalar(0.25, name="r"))),
        ("capsule", Capsule(radius=Scalar(0.3, name="r"), height=Scalar(0.7, name="h"))),
        ("translated", Translate(Sphere(radius=Scalar(0.7, name="r")), Vector([0.4, -0.3, 0.2]))),
        (
            "hard union",
            Union(
                Sphere(radius=Scalar(0.8, name="a")),
                Box(size=Vector([0.5, 0.5, 0.5])),
                smoothness=0.0,
            ),
        ),
        (
            "smooth union",
            Union(
                Sphere(radius=Scalar(0.8, name="a")),
                Box(size=Vector([0.5, 0.5, 0.5])),
                smoothness=0.2,
            ),
        ),
        (
            "intersection",
            Intersection(Sphere(radius=Scalar(0.9, name="a")), Box(size=Vector([0.6, 0.6, 0.6]))),
        ),
        (
            "difference",
            Difference(Box(size=Vector([0.7, 0.7, 0.7])), Sphere(radius=Scalar(0.85, name="a"))),
        ),
        ("profile 5", ExtrudedPolygon(_ring(5), depth=Scalar(0.6, name="d"))),
        ("profile 24", ExtrudedPolygon(_ring(24, 0.9), depth=Scalar(0.4, name="d"))),
        (
            "profile in a union",
            Union(
                ExtrudedPolygon(_ring(7, 0.8), depth=Scalar(0.5)),
                Translate(Sphere(radius=Scalar(0.5)), Vector([0.7, 0.0, 0.0])),
                smoothness=0.05,
            ),
        ),
        (
            "two profiles share one buffer",
            Union(
                ExtrudedPolygon(_ring(5, 0.7), depth=Scalar(0.5)),
                Translate(
                    ExtrudedPolygon(_ring(9, 0.5), depth=Scalar(0.9)),
                    Vector([0.9, 0.2, 0.0]),
                ),
                smoothness=0.0,
            ),
        ),
        ("revolve", RevolvedPolygon(_ring(6, 0.3), offset=Scalar(0.9, name="o"))),
        (
            "linear pattern",
            LinearPattern(
                Sphere(radius=Scalar(0.25)),
                direction=Vector([1.0, 0.0, 0.0]),
                count=5,
                spacing=Scalar(0.7),
            ),
        ),
        (
            "linear pattern of a profile, with a skip",
            LinearPattern(
                ExtrudedPolygon(_ring(6, 0.25), depth=Scalar(0.4)),
                direction=Vector([0.0, 1.0, 0.0]),
                count=6,
                spacing=Scalar(0.8),
                skip=(1, 3),
            ),
        ),
        (
            "polar pattern",
            PolarPattern(
                Translate(Box(size=Vector([0.12, 0.3, 0.12])), Vector([0.8, 0.0, 0.0])),
                count=8,
            ),
        ),
        (
            "polar pattern about a tilted axis, with a skip",
            PolarPattern(
                Translate(Sphere(radius=Scalar(0.2)), Vector([0.7, 0.1, 0.0])),
                count=6,
                axis=Axis(origin=[0.1, 0.0, -0.2], direction=[0.3, 0.2, 1.0]),
                skip=(2,),
            ),
        ),
        (
            "loft",
            LoftedPolygon(
                vertices_a=_ring(6, 0.6), vertices_b=_ring(6, 0.25), height=Scalar(0.9, name="h")
            ),
        ),
        (
            "mirror",
            Mirror(Translate(Sphere(radius=Scalar(0.4)), Vector([0.7, 0.1, 0.0])), axis="x"),
        ),
        ("shell", Shell(Sphere(radius=Scalar(0.8)), thickness=Scalar(0.15))),
        ("offset", Offset(Box(size=Vector([0.4, 0.5, 0.3])), distance=Scalar(0.12))),
        (
            "drafted extrusion",
            ExtrudedPolygon(_ring(7, 0.6), depth=Scalar(0.7), draft=Scalar(8.0)),
        ),
        (
            "nested",
            Union(
                Translate(Sphere(radius=Scalar(0.5, name="a")), Vector([0.6, 0.0, 0.0])),
                Intersection(
                    Box(size=Vector([0.6, 0.6, 0.6])),
                    Sphere(radius=Scalar(0.8, name="b")),
                ),
                smoothness=0.1,
            ),
        ),
    ]


@pytest.mark.parametrize(("label", "scene"), _scenes(), ids=[name for name, _ in _scenes()])
def test_both_backends_agree_with_the_python_field(label, scene):
    """The claim the direct backend has to earn, on a real device.

    Tolerance is f32: both shaders compute in single precision while the
    reference is f64, so the bound is set by the representation rather than
    by either backend's arithmetic.  A genuine divergence — a wrong sign, a
    missed clamp, a transform applied forwards — is orders of magnitude
    larger than this and is what the test is for.
    """
    traced = _evaluate_on_device(compile_sdf_to_wgsl(scene), _POINTS)
    direct = _evaluate_on_device(compile_sdf_direct(scene), _POINTS)
    reference = _python_field(scene, _POINTS)

    np.testing.assert_allclose(direct, traced, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(direct, reference, rtol=2e-5, atol=2e-6)


def test_a_repeated_subtree_is_emitted_once():
    """One function per node, so sharing is structural rather than textual.

    The traced path emits a shared subtree twice, because `scalar_lowering`
    unrolls and nothing downstream re-recognises the duplicate.
    """
    shared = ExtrudedPolygon(_ring(12, 0.4), depth=Scalar(0.5))
    scene = Union(shared, Translate(shared, Vector([1.0, 0.0, 0.0])), smoothness=0.0)
    wgsl = compile_sdf_direct(scene).wgsl
    # The profile kernel body appears once, and the node wrapping it once.
    assert wgsl.count("fn prim_extrudedpolygon(") == 1
    assert wgsl.count("prim_extrudedpolygon(p,") == 1


def test_a_pattern_loops_instead_of_unrolling():
    """The size claim for patterns, stated as a measurement.

    `LinearPattern.sdf` under `scalar_lowering` emits one copy of the child
    per kept instance; here the child is one function and the instances are
    an array the loop walks.
    """

    def pattern(count: int):
        return LinearPattern(
            ExtrudedPolygon(_ring(8, 0.3), depth=Scalar(0.4)),
            direction=Vector([1.0, 0.0, 0.0]),
            count=count,
            spacing=Scalar(0.7),
        )

    traced = [len(compile_sdf_to_wgsl(pattern(n))) for n in (2, 16)]
    direct = [len(compile_sdf_direct(pattern(n))) for n in (2, 16)]
    assert traced[1] > 6 * traced[0], "the traced form unrolls every instance"
    # Only the instance list grows: one short integer per kept copy.
    assert direct[1] < direct[0] + 100


#: Shipped scenes the direct backend can compile today. The rest name a node
#: it has no kernel for, which is the honest report of partial coverage.
_SHIPPED = ("starter", "bracket", "duct_sink", "end_cap")


@pytest.mark.parametrize("stem", _SHIPPED)
def test_a_shipped_scene_agrees_field_for_field(stem):
    """The comparison that matters: real geometry, not a synthetic ring.

    A scene assembled by hand exercises the combinations — a revolve inside
    a blend inside a pattern, a rotation about an arbitrary axis — that each
    kernel gets right alone and can still get wrong together.
    """
    from pathlib import Path

    from cadjoint.viewer._worker_scene import _execute_scene

    root = Path(__file__).resolve().parents[2]
    scene = _execute_scene((root / "scenes" / f"{stem}.py").read_text())["scene"]
    points = _POINTS[:1024]
    traced = _evaluate_on_device(compile_sdf_to_wgsl(scene), points)
    direct = _evaluate_on_device(compile_sdf_direct(scene), points)
    # Looser than the unit scenes: these fields span metres and the smooth
    # blends amplify f32 rounding, so the bound is relative to the values
    # rather than absolute. A wrong kernel is still orders of magnitude out.
    np.testing.assert_allclose(direct, traced, rtol=1e-4, atol=1e-5)


def test_an_unknown_node_is_refused_rather_than_guessed():
    """Partial coverage has to announce itself.

    A backend that silently emitted *something* for a node it does not know
    would be the exact failure this whole file exists to prevent. When the
    node named here gains a kernel, move this test to whatever is still
    missing rather than deleting it.
    """
    # A twisted extrusion rotates the query by an angle that varies with z,
    # which makes the field non-1-Lipschitz. It is the remaining gap, and it
    # is refused rather than approximated.
    with pytest.raises(UnsupportedNode, match="twist"):
        compile_sdf_direct(
            ExtrudedPolygon(_ring(5), depth=Scalar(0.5), twist=Scalar(15.0))
        )


def test_the_direct_module_is_smaller_than_the_traced_one():
    """Not a performance assertion — a statement about what each emits.

    The traced path writes one `let` per StableHLO op; the direct path
    writes a call per node and one shared body per primitive *type*. The gap
    is the point of the exercise, so it is worth failing if it inverts.
    """
    scene = _scenes()[-1][1]
    assert len(compile_sdf_direct(scene)) < len(compile_sdf_to_wgsl(scene))


@pytest.mark.parametrize("count", [4, 16, 64])
def test_a_profile_costs_the_same_code_whatever_its_vertex_count(count):
    """The claim the storage buffer exists to make.

    The traced backend holds `scalar_lowering` for its whole trace, which
    keeps every vertex a separate 2-vector and unrolls the edge loop, so its
    module grows with the profile. The direct backend emits one loop and
    moves the vertices into a buffer, so its module does not — the vertices
    become data, and data is not code.
    """
    program = compile_sdf_direct(ExtrudedPolygon(_ring(count), depth=Scalar(0.5)))
    assert program.vertices.shape == (count, 2)
    # Against a baseline rather than an absolute size, so that adding a
    # parameter to the kernel does not look like a regression here: only the
    # two `u32` literals at the call site vary with the count.
    baseline = compile_sdf_direct(ExtrudedPolygon(_ring(4), depth=Scalar(0.5)))
    assert len(program.wgsl) <= len(baseline.wgsl) + 8


def test_the_traced_module_grows_with_the_profile_and_the_direct_one_does_not():
    """State the contrast as a measurement rather than an assertion of faith."""
    small, large = 4, 64
    traced = [
        len(compile_sdf_to_wgsl(ExtrudedPolygon(_ring(n), depth=Scalar(0.5))))
        for n in (small, large)
    ]
    direct = [
        len(compile_sdf_direct(ExtrudedPolygon(_ring(n), depth=Scalar(0.5))))
        for n in (small, large)
    ]
    assert traced[1] > 8 * traced[0], "the traced form unrolls, so it must grow steeply"
    assert direct[1] < direct[0] + 32, "the direct form only widens two integer literals"
