"""Validate every WGSL shader the playground ships through a real GPU compiler.

Shader bugs otherwise surface only in a browser with WebGPU, which CI and
headless test runs usually lack. ``wgpu`` (wgpu-native/naga) compiles the same
WGSL a browser would, so syntax and type errors fail here instead.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu", reason="wgpu is needed to validate WGSL")

from cadjoint.backends.wgsl import (  # noqa: E402
    PARAMETER_SLOT_BYTES,
    compile_scene_to_wgsl,
)
from cadjoint.sdf.boolean import Union  # noqa: E402
from cadjoint.sdf.primitives import Sphere  # noqa: E402
from cadjoint.viewer._pathtracer import (  # noqa: E402
    WGSL_PRESENT_TEMPLATE,
    build_path_tracer_shader,
)
from cadjoint.viewer._webgpu import build_viewer_shader  # noqa: E402

OVERLAY_WGSL = Path(__file__).resolve().parents[2] / "frontend/src/viewer/overlay.wgsl"
SIMULATION_WGSL = Path(__file__).resolve().parents[2] / "frontend/src/viewer/simulation.wgsl"
GRATICULE_WGSL = Path(__file__).resolve().parents[2] / "frontend/src/viewer/graticule.wgsl"


@pytest.fixture(scope="module")
def device():
    """A real GPU device; skips when no adapter is available."""
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        return adapter.request_device_sync()
    except Exception as error:  # pragma: no cover - depends on the host
        pytest.skip(f"No usable WebGPU adapter: {error}")


@pytest.fixture(scope="module")
def scene_code():
    return compile_scene_to_wgsl(Union(Sphere(1.0), Sphere(0.6), smoothness=0.1))


def compile_wgsl(device, code: str, label: str) -> None:
    """Compile WGSL, surfacing validation errors as test failures."""
    device.create_shader_module(code=code, label=label)


def test_preview_shader_compiles(device, scene_code):
    compile_wgsl(device, build_viewer_shader(scene_code), "preview")


def test_path_tracer_shader_compiles(device, scene_code):
    compile_wgsl(device, build_path_tracer_shader(scene_code), "path tracer")


def test_present_shader_compiles(device):
    compile_wgsl(device, WGSL_PRESENT_TEMPLATE, "present")


@pytest.mark.skipif(not OVERLAY_WGSL.is_file(), reason="frontend sources not present")
def test_overlay_shader_compiles(device):
    compile_wgsl(device, OVERLAY_WGSL.read_text(), "overlay")


def test_sketch_scene_shader_compiles(device):
    """Extruded and revolved sketch polygons must survive shader lowering."""
    from cadjoint.construction import PolygonProfile, extrude, revolve

    profile = PolygonProfile([[0.0, 0.0], [2.0, 0.0], [1.5, 1.2], [0.2, 1.0]], name="p")
    section = PolygonProfile(
        [[0.7, -0.2], [1.0, -0.2], [1.0, 0.2], [0.7, 0.2]],
        name="section",
    )
    code = compile_scene_to_wgsl(Union(extrude(profile, depth=0.8), revolve(section)))
    compile_wgsl(device, build_viewer_shader(code), "sketch preview")
    compile_wgsl(device, build_path_tracer_shader(code), "sketch path tracer")


@pytest.mark.skipif(not SIMULATION_WGSL.is_file(), reason="frontend sources not present")
def test_simulation_shader_compiles(device):
    compile_wgsl(device, SIMULATION_WGSL.read_text(), "simulation")


@pytest.mark.skipif(not GRATICULE_WGSL.is_file(), reason="frontend sources not present")
def test_graticule_shader_compiles(device):
    """The construction grid rules three world planes, each with `fwidth`.

    Derivatives are only legal in uniform control flow, and the plane the grid
    lands on is chosen per draw — so the one thing that can go wrong here is
    invisible in a unit test and shows up as a compile failure in a browser.
    """
    compile_wgsl(device, GRATICULE_WGSL.read_text(), "graticule")


def test_invalid_wgsl_is_actually_rejected(device):
    """Guard against the validation silently passing everything."""
    with pytest.raises(wgpu.GPUValidationError):
        compile_wgsl(device, "fn broken( { not wgsl }", "invalid")


# ── Pipeline layouts ─────────────────────────────────────────────────────────
# Shader compilation alone would not catch a wrong binding number or vertex
# attribute offset, so the viewer's pipelines are built here the same way
# renderer.ts builds them.

COLOR_FORMAT = "bgra8unorm"
# The preview pass is drawn first and owns both colour and depth. It must not
# depth-test: a ray miss writes depth 1.0, which "less" rejects against the 1.0
# clear, discarding the whole background.
DEPTH_STATE = {
    "format": "depth32float",
    "depth_write_enabled": True,
    "depth_compare": "always",
}


def test_preview_depth_pipeline_matches_the_renderer(device, scene_code):
    module = device.create_shader_module(code=build_viewer_shader(scene_code))
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={"module": module, "entry_point": "vs_main"},
        fragment={
            "module": module,
            "entry_point": "fs_main_depth",
            "targets": [{"format": COLOR_FORMAT}],
        },
        primitive={"topology": "triangle-list"},
        depth_stencil=DEPTH_STATE,
    )
    layout = pipeline.get_bind_group_layout(0)
    scene_uniforms = device.create_buffer(size=112, usage=wgpu.BufferUsage.UNIFORM)
    view_uniforms = device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM)
    # Depth needs both the scene uniforms and the view-projection matrix.
    device.create_bind_group(
        layout=layout,
        entries=[
            {"binding": 0, "resource": {"buffer": scene_uniforms, "offset": 0, "size": 112}},
            {"binding": 2, "resource": {"buffer": view_uniforms, "offset": 0, "size": 64}},
        ],
    )


def test_widget_entry_point_does_not_require_the_view_binding(device, scene_code):
    """The Jupyter widget builds fs_main with a single 80-byte uniform buffer.

    Adding the view-projection binding for depth must not leak into that
    pipeline's derived layout, or the notebook viewer would break.
    """
    module = device.create_shader_module(code=build_viewer_shader(scene_code))
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={"module": module, "entry_point": "vs_main"},
        fragment={
            "module": module,
            "entry_point": "fs_main",
            "targets": [{"format": COLOR_FORMAT}],
        },
        primitive={"topology": "triangle-list"},
    )
    uniforms = device.create_buffer(size=112, usage=wgpu.BufferUsage.UNIFORM)
    device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": uniforms, "offset": 0, "size": 112}}],
    )


OVERLAY_BLEND = {
    "color": {
        "src_factor": "src-alpha",
        "dst_factor": "one-minus-src-alpha",
        "operation": "add",
    },
    "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
}


@pytest.mark.skipif(not OVERLAY_WGSL.is_file(), reason="frontend sources not present")
@pytest.mark.parametrize(
    ("shader", "entry_vertex", "entry_fragment", "stride", "step_mode", "blend", "attributes"),
    [
        (
            "overlay.wgsl",
            "vs_edge",
            "fs_edge",
            40,
            "instance",
            OVERLAY_BLEND,
            [
                {"shader_location": 0, "offset": 0, "format": "float32x3"},
                {"shader_location": 1, "offset": 12, "format": "float32x3"},
                {"shader_location": 2, "offset": 24, "format": "float32x4"},
            ],
        ),
        (
            "overlay.wgsl",
            "vs_handle",
            "fs_handle",
            # Centre, colour, emphasis, and the fill flag that draws a point
            # backed by a free design parameter as a disc and one the shader
            # holds as a constant as a ring.
            36,
            "instance",
            OVERLAY_BLEND,
            [
                {"shader_location": 0, "offset": 0, "format": "float32x3"},
                {"shader_location": 1, "offset": 12, "format": "float32x4"},
                {"shader_location": 2, "offset": 28, "format": "float32"},
                {"shader_location": 3, "offset": 32, "format": "float32"},
            ],
        ),
        # The simulation surface: interleaved position + scalar + overlay
        # vector per vertex, drawn indexed and opaque (no blend).
        (
            "simulation.wgsl",
            "vs_sim",
            "fs_sim",
            32,
            "vertex",
            None,
            [
                {"shader_location": 0, "offset": 0, "format": "float32x3"},
                {"shader_location": 1, "offset": 12, "format": "float32"},
                {"shader_location": 2, "offset": 16, "format": "float32x4"},
            ],
        ),
    ],
)
def test_overlay_pipelines_match_the_renderer(
    device, shader, entry_vertex, entry_fragment, stride, step_mode, blend, attributes
):
    path = OVERLAY_WGSL.parent / shader
    module = device.create_shader_module(code=path.read_text())
    target = {"format": COLOR_FORMAT}
    if blend is not None:
        target["blend"] = blend
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={
            "module": module,
            "entry_point": entry_vertex,
            "buffers": [{"array_stride": stride, "step_mode": step_mode, "attributes": attributes}],
        },
        fragment={
            "module": module,
            "entry_point": entry_fragment,
            "targets": [target],
        },
        primitive={"topology": "triangle-list"},
        depth_stencil={
            "format": "depth32float",
            "depth_write_enabled": True,
            "depth_compare": "less-equal",
        },
    )
    overlay_uniforms = device.create_buffer(size=112, usage=wgpu.BufferUsage.UNIFORM)
    device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": overlay_uniforms, "offset": 0, "size": 112}}
        ],
    )


def shared_layout(device, bindings):
    """Explicit layout shared by several pipelines, mirroring Renderer.sharedLayout."""
    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": binding,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }
            for binding in bindings
        ]
    )
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    return bind_group_layout, pipeline_layout


def parameter_pipeline_layout(device, scene_layout, program):
    """The scene layout plus the program's parameter block, mirroring the renderer.

    WebGPU wants every group up to the highest one spelled out, so the groups
    between the scene's own (0) and the parameters' (3) are declared empty.
    An empty layout binds nothing and needs no bind group set against it.

    Returns:
        ``(parameter bind group layout, pipeline layout)``.
    """
    parameter_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": program.binding,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }
        ]
    )
    layouts = [scene_layout]
    while len(layouts) < program.group:
        layouts.append(device.create_bind_group_layout(entries=[]))
    layouts.append(parameter_layout)
    return parameter_layout, device.create_pipeline_layout(bind_group_layouts=layouts)


def parameter_bind_group(device, layout, program, cull_margin=None):
    """The program's current values, packed and bound.

    ``cull_margin`` overrides the reserved cull-margin slot, which is how the
    viewer switches bounding-box culling off without a recompile: an infinite
    margin makes every skip test false.
    """
    packed = program.buffer()
    if cull_margin is not None:
        packed[program.cull_margin_offset // 4] = cull_margin
    buffer = device.create_buffer_with_data(data=packed.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    return device.create_bind_group(
        layout=layout,
        entries=[{"binding": program.binding, "resource": {"buffer": buffer}}],
    )


def test_preview_and_overlay_draw_in_one_pass(device, scene_code):
    """Record the frame the viewer actually submits.

    A bind group derived from one ``layout: "auto"`` pipeline cannot be bound to
    another pipeline, and that only fails at draw time — so this records real
    draw calls with the shared bind groups instead of just building pipelines.
    """
    size = (64, 64)
    color = device.create_texture(
        size=(*size, 1),
        format=COLOR_FORMAT,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )
    depth = device.create_texture(
        size=(*size, 1),
        format="depth32float",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )

    # Preview: colour and depth-only variants share one bind group.
    preview_module = device.create_shader_module(code=build_viewer_shader(scene_code))
    preview_bgl, preview_pl = shared_layout(device, [0, 2])

    def preview_pipeline(write_mask):
        return device.create_render_pipeline(
            layout=preview_pl,
            vertex={"module": preview_module, "entry_point": "vs_main"},
            fragment={
                "module": preview_module,
                "entry_point": "fs_main_depth",
                "targets": [{"format": COLOR_FORMAT, "write_mask": write_mask}],
            },
            primitive={"topology": "triangle-list"},
            depth_stencil=DEPTH_STATE,
        )

    color_pipeline = preview_pipeline(wgpu.ColorWrite.ALL)
    depth_only_pipeline = preview_pipeline(0)
    preview_bind_group = device.create_bind_group(
        layout=preview_bgl,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": device.create_buffer(size=112, usage=wgpu.BufferUsage.UNIFORM),
                    "offset": 0,
                    "size": 112,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM),
                    "offset": 0,
                    "size": 64,
                },
            },
        ],
    )

    # Overlay: edge and handle pipelines share one bind group.
    overlay_module = device.create_shader_module(code=OVERLAY_WGSL.read_text())
    overlay_bgl, overlay_pl = shared_layout(device, [0])
    blend = {
        "color": {
            "src_factor": "src-alpha",
            "dst_factor": "one-minus-src-alpha",
            "operation": "add",
        },
        "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
    }

    def overlay_pipeline(entry_vertex, entry_fragment, stride, attributes):
        return device.create_render_pipeline(
            layout=overlay_pl,
            vertex={
                "module": overlay_module,
                "entry_point": entry_vertex,
                "buffers": [
                    {"array_stride": stride, "step_mode": "instance", "attributes": attributes}
                ],
            },
            fragment={
                "module": overlay_module,
                "entry_point": entry_fragment,
                "targets": [{"format": COLOR_FORMAT, "blend": blend}],
            },
            primitive={"topology": "triangle-list"},
            depth_stencil={
                "format": "depth32float",
                "depth_write_enabled": True,
                "depth_compare": "less-equal",
            },
        )

    edge_pipeline = overlay_pipeline(
        "vs_edge",
        "fs_edge",
        40,
        [
            {"shader_location": 0, "offset": 0, "format": "float32x3"},
            {"shader_location": 1, "offset": 12, "format": "float32x3"},
            {"shader_location": 2, "offset": 24, "format": "float32x4"},
        ],
    )
    handle_pipeline = overlay_pipeline(
        "vs_handle",
        "fs_handle",
        36,
        [
            {"shader_location": 0, "offset": 0, "format": "float32x3"},
            {"shader_location": 1, "offset": 12, "format": "float32x4"},
            {"shader_location": 2, "offset": 28, "format": "float32"},
            {"shader_location": 3, "offset": 32, "format": "float32"},
        ],
    )
    overlay_bind_group = device.create_bind_group(
        layout=overlay_bgl,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": device.create_buffer(size=112, usage=wgpu.BufferUsage.UNIFORM),
                    "offset": 0,
                    "size": 112,
                },
            }
        ],
    )
    edges = device.create_buffer(size=40 * 4, usage=wgpu.BufferUsage.VERTEX)
    handles = device.create_buffer(size=36 * 4, usage=wgpu.BufferUsage.VERTEX)

    encoder = device.create_command_encoder()
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": color.create_view(),
                "clear_value": (0, 0, 0, 1),
                "load_op": "clear",
                "store_op": "store",
            }
        ],
        depth_stencil_attachment={
            "view": depth.create_view(),
            "depth_clear_value": 1.0,
            "depth_load_op": "clear",
            "depth_store_op": "store",
        },
    )
    render_pass.set_pipeline(color_pipeline)
    render_pass.set_bind_group(0, preview_bind_group)
    render_pass.draw(3)
    # Same bind group, different pipeline — the path-trace depth prepass.
    render_pass.set_pipeline(depth_only_pipeline)
    render_pass.set_bind_group(0, preview_bind_group)
    render_pass.draw(3)
    render_pass.set_pipeline(edge_pipeline)
    render_pass.set_bind_group(0, overlay_bind_group)
    render_pass.set_vertex_buffer(0, edges)
    render_pass.draw(6, 4)
    # Same bind group again, now on the handle pipeline.
    render_pass.set_pipeline(handle_pipeline)
    render_pass.set_bind_group(0, overlay_bind_group)
    render_pass.set_vertex_buffer(0, handles)
    render_pass.draw(6, 4)
    render_pass.end()
    device.queue.submit([encoder.finish()])


# Display uniform: projection, ortho height, flag bits, x-ray strength.
SHADOWS = 1
REFLECTIONS = 2
FLAT = 4
HIDE_SOLID = 8
PERSPECTIVE = (0.0, 0.0, float(SHADOWS | REFLECTIONS), 0.0)


# The six scalars the SDF views ride in — see `Uniforms` in `_webgpu.py`.
# (view, slice axis, plane coordinate, iso offset, contour spacing, march steps)
SOLID_VIEW = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def render_preview(
    device,
    scene_code,
    display=PERSPECTIVE,
    size=64,
    camera=(3.0, 2.0, 4.0),
    sdf=SOLID_VIEW,
    read="color",
    program=None,
    cull_margin=None,
):
    """Render one preview frame and read one of its attachments back.

    Args:
        read: ``"color"`` for the red channel, one byte per pixel; ``"depth"``
            for the depth attachment as floats in 0..1, which is what the
            floor grid and the construction overlays are tested against.
        program: The :class:`~cadjoint.backends.wgsl.ShaderProgram` the scene
            was compiled to, when it was built in the uniform form. Its
            buffer is filled from the program's own values and bound at the
            group the program names, exactly as ``renderer.ts`` does. ``None``
            is the literal form, which binds nothing extra.
    """
    color = device.create_texture(
        size=(size, size, 1),
        format="rgba8unorm",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    depth = device.create_texture(
        size=(size, size, 1),
        format="depth32float",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    built = (
        str(scene_code)
        if isinstance(scene_code, _SceneCodeOverride)
        else build_viewer_shader(scene_code)
    )
    module = device.create_shader_module(code=built)
    bind_group_layout, pipeline_layout = shared_layout(device, [0, 2])
    parameter_layout = None
    if program is not None:
        parameter_layout, pipeline_layout = parameter_pipeline_layout(
            device, bind_group_layout, program
        )
    pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={"module": module, "entry_point": "vs_main"},
        fragment={
            "module": module,
            "entry_point": "fs_main_depth",
            "targets": [{"format": "rgba8unorm"}],
        },
        primitive={"topology": "triangle-list"},
        depth_stencil=DEPTH_STATE,
    )

    # resolution | camera_pos | camera_target | light+intensity | bg | path | display
    view_mode, slice_axis, slice_at, iso_offset, spacing, march = sdf
    scene_buffer = device.create_buffer_with_data(
        data=struct.pack(
            "28f",
            size,
            size,
            view_mode,
            slice_axis,
            camera[0],
            camera[1],
            camera[2],
            slice_at,
            0,
            0,
            0,
            iso_offset,
            0.55,
            0.8,
            0.35,
            3.0,
            0.035,
            0.045,
            0.035,
            spacing,
            0,
            0,
            0,
            march,
            *display,
        ),
        usage=wgpu.BufferUsage.UNIFORM,
    )
    view_buffer = device.create_buffer_with_data(
        data=struct.pack("16f", *([0.0] * 16)), usage=wgpu.BufferUsage.UNIFORM
    )
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": scene_buffer, "offset": 0, "size": 112}},
            {"binding": 2, "resource": {"buffer": view_buffer, "offset": 0, "size": 64}},
        ],
    )

    encoder = device.create_command_encoder()
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": color.create_view(),
                "clear_value": (0, 0, 0, 1),
                "load_op": "clear",
                "store_op": "store",
            }
        ],
        depth_stencil_attachment={
            "view": depth.create_view(),
            "depth_clear_value": 1.0,
            "depth_load_op": "clear",
            "depth_store_op": "store",
        },
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, bind_group)
    if parameter_layout is not None:
        render_pass.set_bind_group(
            program.group,
            parameter_bind_group(device, parameter_layout, program, cull_margin),
        )
    render_pass.draw(3)
    render_pass.end()
    device.queue.submit([encoder.finish()])

    if read == "depth":
        raw = device.queue.read_texture(
            {"texture": depth, "aspect": wgpu.TextureAspect.depth_only},
            {"offset": 0, "bytes_per_row": size * 4, "rows_per_image": size},
            (size, size, 1),
        )
        return list(struct.unpack(f"{size * size}f", bytes(raw)))

    pixels = device.queue.read_texture(
        {"texture": color},
        {"offset": 0, "bytes_per_row": size * 4, "rows_per_image": size},
        (size, size, 1),
    )
    return list(bytes(pixels))[::4]


def assert_rendered(channel, size=64, label="frame"):
    """A rendered frame must be neither black nor a flat fill."""
    lit = sum(1 for value in channel if value > 8)
    assert lit > size * size * 0.5, f"{label} rendered an almost entirely black image"
    assert len(set(channel)) > 4, f"{label} rendered a flat image with no shading"


def test_preview_pass_actually_shades_pixels(device, scene_code):
    """Render the preview and read it back — a black frame is a failure.

    Pipeline construction succeeds even when every fragment is later discarded,
    so this checks the image itself: both the environment background and the
    lit surface have to survive the depth configuration.
    """
    assert_rendered(render_preview(device, scene_code), label="perspective preview")


def test_orthographic_projection_renders(device, scene_code):
    """The orthographic branch of primary_ray must produce an image too."""
    ortho = (1.0, 6.0, float(SHADOWS | REFLECTIONS), 0.0)
    channel = render_preview(device, scene_code, display=ortho)
    assert_rendered(channel, label="orthographic preview")
    # Parallel rays frame the scene differently from a perspective camera.
    assert channel != render_preview(device, scene_code)


@pytest.mark.parametrize(
    ("label", "camera"),
    [("top", (0.0, 0.0, 6.0)), ("bottom", (0.0, 0.0, -6.0))],
)
def test_pole_views_do_not_degenerate(device, scene_code, label, camera):
    """Looking straight down or up is where cross(forward, +Z) collapses.

    The world is Z-up, so the poles are on the Z axis. Without the
    reference-vector fallback in camera_basis the frame is NaN and the image
    comes out blank, so this renders from directly overhead.
    """
    channel = render_preview(
        device,
        scene_code,
        display=(1.0, 6.0, float(SHADOWS | REFLECTIONS), 0.0),
        camera=camera,
    )
    assert_rendered(channel, label=f"{label} view")


@pytest.mark.parametrize(
    ("label", "display"),
    [
        ("flat shading", (0.0, 0.0, float(FLAT), 0.0)),
        ("no shadows", (0.0, 0.0, float(REFLECTIONS), 0.0)),
        ("x-ray", (0.0, 0.0, float(SHADOWS | REFLECTIONS), 1.0)),
    ],
)
def test_display_modes_render(device, scene_code, label, display):
    assert_rendered(render_preview(device, scene_code, display=display), label=label)


def test_hide_solid_leaves_only_the_environment(device, scene_code):
    """The hide-solid flag skips tracing, so only the background remains."""
    hidden = render_preview(device, scene_code, display=(0.0, 0.0, float(HIDE_SOLID), 0.0))
    shaded = render_preview(device, scene_code)
    assert hidden != shaded
    # A pure environment gradient has far fewer distinct values than a shaded solid.
    assert len(set(hidden)) < len(set(shaded))


def test_present_pipeline_can_share_the_overlay_depth_attachment(device):
    """Path-trace mode draws present, a depth prepass, and the overlay in one pass."""
    module = device.create_shader_module(code=WGSL_PRESENT_TEMPLATE)
    device.create_render_pipeline(
        layout="auto",
        vertex={"module": module, "entry_point": "vs_present"},
        fragment={
            "module": module,
            "entry_point": "fs_present",
            "targets": [{"format": COLOR_FORMAT}],
        },
        primitive={"topology": "triangle-list"},
        depth_stencil={
            "format": "depth32float",
            "depth_write_enabled": False,
            "depth_compare": "always",
        },
    )


# ── the distance-field views ────────────────────────────────────────────────

# One entry per `SDF_VIEW_*` constant in `_webgpu.py`, with the extra scalars
# each one reads. The contour spacing is the floor grid's rung, which is what
# the renderer writes; 0.5 against a unit sphere at this framing puts several
# major intervals across the card.
# `display.y` is the framed world height. The depth view maps its ramp across
# that, so it is the one view that needs a real one rather than the zero the
# perspective fixture leaves it at.
FRAMED = (0.0, 6.0, float(SHADOWS | REFLECTIONS), 0.0)

SDF_VIEWS = [
    ("slice", (1.0, 0.0, 0.0, 0.0, 0.5, 0.0), PERSPECTIVE),
    ("gradient", (2.0, 2.0, 0.1, 0.0, 0.5, 0.0), PERSPECTIVE),
    ("normals", (3.0, 0.0, 0.0, 0.0, 0.0, 0.0), PERSPECTIVE),
    ("depth", (4.0, 0.0, 0.0, 0.0, 0.0, 0.0), FRAMED),
]


@pytest.mark.parametrize("label, sdf, display", SDF_VIEWS, ids=[name for name, _, _ in SDF_VIEWS])
def test_distance_field_views_render(device, scene_code, label, sdf, display):
    """Every view flag has to produce a real image, not a flat fill.

    The branch is chosen by a uniform, so compiling the module proves nothing
    about any individual view: a typo inside one arm compiles perfectly and
    renders a blank card. Each one is drawn and read back instead.
    """
    channel = render_preview(device, scene_code, display=display, sdf=sdf)
    assert len(set(channel)) > 4, f"{label} rendered a flat image"


def test_the_slice_shows_the_interior_of_the_solid(device, scene_code):
    """A plane through the middle of a solid must show its negative field.

    This is the defect the view shipped with: the plane was composited by depth
    against the raymarched surface, so inside a part — where the surface is
    always nearer than the plane — the interior was never drawn at all. The
    scene fixture is a sphere of radius 1 at the origin, so a cut at x = 0
    passes through its middle, and the inside hue is the ramp's violet, which
    is far redder than the ochre outside it.
    """
    through = render_preview(device, scene_code, sdf=(1.0, 0.0, 0.0, 0.0, 0.5, 0.0))
    outside = render_preview(device, scene_code, sdf=(1.0, 0.0, 1.9, 0.0, 0.5, 0.0))
    # Violet is 0.39 red where the surrounding ochre is 0.54: the interior
    # shows up as a population of darker-red pixels the outside cut lacks.
    interior = sum(1 for value in through if value < 120)
    assert interior > 64, "a cut through the solid showed no interior field"
    assert interior > sum(1 for value in outside if value < 120)


def test_an_x_rayed_solid_still_writes_its_depth(device, scene_code):
    """The floor grid must be occluded by the part, X-Ray preset included.

    The complaint: "lets put that floor grid behind all objects" — on the
    default preset the ground grid came straight through the heat sink, so the
    drawing read as a grid laid over a model rather than a model standing on
    the ground.

    The cause was here. The grid is a fullscreen pass at z = 1 that depth-tests
    ``less-equal`` against whatever the scene wrote, and an x-rayed solid used
    to clear ``occludes`` — so ``fs_main_depth`` wrote 1.0, the depth buffer
    said the ray had missed, and the grid drew through the part. Construction
    geometry got its see-through from that, and it gets it from its own end
    now (``depthCompare: "always"`` on the overlay pipelines while the solid is
    x-rayed, which is what the gizmos have always used).

    So: the same silhouette, at the same depths, x-rayed or not.
    """
    opaque = render_preview(device, scene_code, read="depth")
    xrayed = render_preview(
        device,
        scene_code,
        display=(0.0, 0.0, float(SHADOWS | REFLECTIONS), 1.0),
        read="depth",
    )
    covered = [index for index, value in enumerate(opaque) if value < 1.0]
    assert len(covered) > 64, "the fixture sphere covered almost none of the frame"
    assert [index for index, value in enumerate(xrayed) if value < 1.0] == covered, (
        "an x-rayed solid reported a different silhouette to the depth buffer"
    )
    for index in covered:
        assert xrayed[index] == pytest.approx(opaque[index], abs=1e-6), (
            "an x-rayed solid wrote a different depth than the solid it is"
        )

    # And the hidden-solid flag is the one case that legitimately writes no
    # depth: there is nothing traced to be in front of anything.
    hidden = render_preview(
        device, scene_code, display=(0.0, 0.0, float(HIDE_SOLID), 0.0), read="depth"
    )
    assert all(value == 1.0 for value in hidden)


def test_the_march_budget_changes_the_image(device, scene_code):
    """`path_settings.w` has to reach the marcher, not just the struct."""
    starved = render_preview(device, scene_code, sdf=(0.0, 0.0, 0.0, 0.0, 0.0, 2.0))
    generous = render_preview(device, scene_code, sdf=(0.0, 0.0, 0.0, 0.0, 0.0, 192.0))
    assert starved != generous, "the step budget made no difference to the frame"


def test_the_isosurface_offset_moves_the_surface(device, scene_code):
    """`f = c` must be traced, shaded and shadowed as a surface in its own right."""
    surface = render_preview(device, scene_code)
    dilated = render_preview(device, scene_code, sdf=(0.0, 0.0, 0.0, 0.35, 0.0, 0.0))
    eroded = render_preview(device, scene_code, sdf=(0.0, 0.0, 0.0, -0.35, 0.0, 0.0))
    assert surface != dilated
    assert surface != eroded
    # A dilated solid covers more of the frame than an eroded one.
    assert_rendered(dilated, label="dilated solid")


# ── The two shader forms draw the same image ─────────────────────────────────
# The viewer compiles a scene one of two ways. In the *literal* form every
# design parameter is a float constant in the WGSL; in the *uniform* form the
# free ones are read from a `@group(3)` buffer, which is what lets a handle
# drag be a buffer write instead of a recompile. That is only a legitimate
# choice if the two draw the same picture, so this renders both and compares
# them pixel by pixel. See `compile_scene_with_uniforms` for why only the
# *free* parameters move into the buffer.


@pytest.fixture(scope="module")
def parametric_scene():
    """A scene with free parameters, so the uniform form has slots to fill."""
    from cadjoint.geometry.parameters import Scalar, Vector
    from cadjoint.sdf.transforms.affine import Translate

    radius = Scalar(0.8, free=True, name="radius")
    offset = Vector([0.9, 0.2, 0.0], free=True, name="offset")
    return Union(
        (Sphere(radius), Translate(Sphere(0.55), offset)),
        smoothness=0.12,
    )


#: The two forms are the same arithmetic in a different order, so a handful of
#: units of last-place difference on a 0-255 channel is expected; a moved
#: surface is not, and shows up as whole-silhouette disagreement.
PIXEL_TOLERANCE = 2


def _max_deviation(a, b):
    assert len(a) == len(b)
    return max(abs(int(x) - int(y)) for x, y in zip(a, b))


@pytest.mark.parametrize(
    ("label", "display"),
    [
        ("perspective", PERSPECTIVE),
        ("flat shading", (0.0, 0.0, float(FLAT), 0.0)),
        ("x-ray", (0.0, 0.0, float(SHADOWS | REFLECTIONS), 1.0)),
    ],
)
def test_both_shader_forms_render_the_same_pixels(device, parametric_scene, label, display):
    from cadjoint.backends.wgsl import compile_scene_with_uniforms

    literal = compile_scene_to_wgsl(parametric_scene)
    program = compile_scene_with_uniforms(parametric_scene)

    from_literal = render_preview(device, literal, display=display)
    from_uniform = render_preview(device, program.wgsl, display=display, program=program)

    assert_rendered(from_literal, label=f"{label} literal")
    assert_rendered(from_uniform, label=f"{label} uniform")
    deviation = _max_deviation(from_literal, from_uniform)
    assert deviation <= PIXEL_TOLERANCE, (
        f"{label}: the uniform form moved a pixel by {deviation} levels"
    )


def test_both_shader_forms_agree_on_depth(parametric_scene, device):
    """Colour can agree while the surface sits somewhere else; depth cannot."""
    from cadjoint.backends.wgsl import compile_scene_with_uniforms

    program = compile_scene_with_uniforms(parametric_scene)
    literal = render_preview(device, compile_scene_to_wgsl(parametric_scene), read="depth")
    uniform = render_preview(device, program.wgsl, read="depth", program=program)

    assert max(abs(a - b) for a, b in zip(literal, uniform)) <= 1e-5


def test_the_uniform_form_follows_its_buffer(device, parametric_scene):
    """The buffer has to be what the image depends on, not a decoration.

    Without this the parity tests above would still pass on a shader that
    ignored the buffer and used stale constants.
    """
    from cadjoint.backends.wgsl import compile_scene_with_uniforms
    from cadjoint.extraction import apply_parameters, extract_parameters

    program = compile_scene_with_uniforms(parametric_scene)
    before = render_preview(device, program.wgsl, program=program)

    free, _, _ = extract_parameters(parametric_scene)
    apply_parameters(parametric_scene, {"radius": free["radius"] * 0.5})
    moved = compile_scene_with_uniforms(parametric_scene)
    apply_parameters(parametric_scene, {"radius": free["radius"]})

    assert moved.wgsl == program.wgsl, "only the values were meant to change"
    after = render_preview(device, moved.wgsl, program=moved)
    assert before != after, "halving the radius did not change the image"


# ── The march settings ───────────────────────────────────────────────────────
# Three render settings reach the shader without a recompile: the step budget
# (`path_settings.w`), hit refinement (a bit in `display.z`) and bounds
# culling (a reserved slot in the scene's own parameter buffer). Two rules
# govern all three, and both are checked here: the defaults must draw exactly
# the image the viewer drew before they existed, and a setting that claims to
# change nothing visible must actually change nothing.

#: The refinement bit, matching DISPLAY_REFINE_HIT in `_webgpu.py` and
#: `DISPLAY.refineHit` in `display.ts`.
REFINE_HIT = 32

#: What the frontend writes to switch culling off — see `CULL_MARGIN_OFF`.
CULL_OFF = float("inf")


def _without_refinement(code: str) -> str:
    """The same shader with the refinement branch forced off.

    Stands in for the shader as it was before refinement existed. The
    substitution is one line and the compiler folds the branch away, so what
    this renders is the old `trace` — including the absence of the
    `t_prev`/`d_prev` bookkeeping, which is the thing that could have moved
    the default image without anyone noticing.
    """
    marker = "let refine = display_flag(DISPLAY_REFINE_HIT);"
    assert marker in code, "the refinement branch is not where this test expects it"
    return code.replace(marker, "let refine = false;")


@pytest.mark.parametrize(
    ("label", "display"),
    [
        ("perspective", PERSPECTIVE),
        ("flat shading", (0.0, 0.0, float(FLAT), 0.0)),
        ("x-ray", (0.0, 0.0, float(SHADOWS | REFLECTIONS), 1.0)),
    ],
)
def test_the_default_settings_draw_the_shader_that_has_no_refinement(
    device, scene_code, label, display
):
    """The defaults must reproduce today's image, to the byte.

    Adding refinement put two extra `var`s in the primary march and a branch
    on its exit. With the flag clear none of it may reach the picture — not
    approximately, exactly, because "the default is unchanged" is the whole
    licence for adding the feature at all.
    """
    current = build_viewer_shader(scene_code)
    before = _without_refinement(current)
    assert render_preview(device, scene_code, display=display) == render_preview(
        device, _SceneCodeOverride(before), display=display
    )


class _SceneCodeOverride(str):
    """A prebuilt viewer shader passed where scene code is expected.

    `render_preview` wraps whatever it is given in `build_viewer_shader`; this
    marks a string that is already wrapped, so the two shaders under test
    differ in exactly one line and nothing else.
    """


def test_refinement_moves_the_grazing_pixels_and_no_others(device, scene_code):
    """On is a different image, and only along the silhouette.

    A refinement that changed nothing would be a knob with no effect; one that
    changed a large fraction of the frame would be moving the surface rather
    than resolving it. Both are failures, and this brackets the real
    behaviour between them.
    """
    plain = render_preview(device, scene_code)
    refined = render_preview(
        device, scene_code, display=(0.0, 0.0, float(SHADOWS | REFLECTIONS | REFINE_HIT), 0.0)
    )
    changed = sum(1 for a, b in zip(plain, refined) if a != b)
    assert changed > 0, "refinement changed nothing at all"
    # Measured at 0.1-2.1 % of the frame on the shipped scenes; a tenth of the
    # image is far past anything a sub-epsilon correction can justify.
    assert changed < len(plain) * 0.1, f"refinement moved {changed} of {len(plain)} pixels"


def test_bounds_culling_is_invisible(device, parametric_scene):
    """Culling off must draw the identical frame, at 2x the cost.

    The whole argument for skipping a leaf is that the skip is only taken
    where the exact value is provably what the result already is. If that
    holds, the toggle cannot be seen; if it does not, this is where it shows.
    """
    from cadjoint.backends.wgsl import compile_scene_with_uniforms

    program = compile_scene_with_uniforms(parametric_scene)
    culled = render_preview(device, program.wgsl, program=program)
    flat = render_preview(device, program.wgsl, program=program, cull_margin=CULL_OFF)
    assert culled == flat


def test_the_cull_margin_slot_is_its_own(device, parametric_scene):
    """Switching culling off must not disturb a parameter or the NaN slot."""
    from cadjoint.backends.wgsl import compile_scene_with_uniforms

    program = compile_scene_with_uniforms(parametric_scene)
    assert program.cull_margin_offset == program.nan_offset + PARAMETER_SLOT_BYTES
    assert program.buffer_bytes == program.cull_margin_offset + PARAMETER_SLOT_BYTES
    packed = program.buffer()
    assert packed[program.cull_margin_offset // 4] == np.float32(1e-4)
    assert np.isnan(packed[program.nan_offset // 4])


def test_the_step_budget_is_a_cap_not_a_cost(device, scene_code):
    """A budget above what the scene needs draws the same picture.

    `path_settings.w` caps the march; almost every ray converges long before
    it. Raising it must therefore be invisible, which is what makes it safe
    to expose — and lowering it far enough must not be, which is the other
    half of the same claim.
    """
    generous = render_preview(device, scene_code, sdf=(0.0, 0.0, 0.0, 0.0, 0.0, 512.0))
    enough = render_preview(device, scene_code, sdf=(0.0, 0.0, 0.0, 0.0, 0.0, 192.0))
    starved = render_preview(device, scene_code, sdf=(0.0, 0.0, 0.0, 0.0, 0.0, 4.0))
    assert generous == enough
    assert starved != enough


# ── The capped section ───────────────────────────────────────────────────────
# Clipping a solid at a plane and *capping* it are different pictures: a clip
# lets the ray through to hit the inside of the far shell, which is why a
# clipped part looks hollow. The cap is an intersection with a half-space, so
# the ray stops on the plane. These pin that it is an intersection and not an
# approximation of one.

#: The section flag, matching DISPLAY_SECTION in `_webgpu.py` and
#: `DISPLAY.section` in `display.ts`.
SECTION = 64

#: The SDF view code for world normals, from `SDF_VIEW_CODE` in `display.ts`.
NORMAL_VIEW = (3.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _section_display(base=None):
    """Display tuple with the section on."""
    flags = float(SHADOWS | REFLECTIONS | SECTION)
    return (0.0, 0.0, flags, 0.0) if base is None else base


def test_the_section_cuts_the_solid(device, scene_code):
    plain = render_preview(device, scene_code)
    cut = render_preview(
        device, scene_code, display=(0.0, 0.0, float(SHADOWS | REFLECTIONS | SECTION), 0.0)
    )
    assert plain != cut, "the section flag changed nothing"


def test_a_plane_outside_the_scene_is_exactly_no_section(device, scene_code):
    """The strongest statement available: it is an intersection, not a fudge.

    `traced_field` is `max(scene, halfspace)`. Push the half-space entirely
    clear of the geometry and the `max` can never select it, so the image must
    come back *identical* — not close. Anything else means the section is
    perturbing the field where it has no business to.

    The removed half is always the one the camera is in, so "removes
    nothing" means putting the plane *between the scene and the camera's far
    side*: the camera sits at x = 3 and the scene inside |x| <= 1, so a plane
    at x = 2.5 removes only x > 2.5, which is empty.
    """
    far = (0.0, 0.0, 2.5, 0.0, 0.0, 0.0)  # sdf view solid, plane axis X
    plain = render_preview(device, scene_code, sdf=far)
    cut = render_preview(
        device,
        scene_code,
        display=(0.0, 0.0, float(SHADOWS | REFLECTIONS | SECTION), 0.0),
        sdf=far,
        camera=(3.0, 2.0, 4.0),
    )
    assert plain == cut


def test_the_cap_carries_the_planes_own_normal(device, scene_code):
    """The cut face is flat, and the normal view is where that shows.

    `sdf_normal` takes central differences of the *sectioned* field, so on the
    cap the half-space is what the `max` selected and the differences return
    the plane's normal. A plane has exactly one normal, so capping introduces
    a large run of pixels sharing a single value — which is the signature this
    looks for, and which a hollow clip (showing the curved inside of the far
    shell) could not produce.
    """
    from collections import Counter

    plain = render_preview(device, scene_code, sdf=NORMAL_VIEW)
    cut = render_preview(
        device,
        scene_code,
        display=(0.0, 0.0, float(SHADOWS | REFLECTIONS | SECTION), 0.0),
        sdf=NORMAL_VIEW,
    )
    assert plain != cut
    flat_run = Counter(cut).most_common(1)[0][1] - Counter(plain).most_common(1)[0][1]
    assert flat_run > 0, "the section added no flat-normal region"


def test_the_section_leaves_the_data_path_on_the_true_field():
    """A data view must read the true field, not the sectioned one.

    `scene_field` is what the slice and gradient views sample; only the
    tracer goes through `traced_field`. If the section leaked into the data
    path, a |∇f| view would report the plane's own gradient and call the
    scene metric where it is not — the diagnostic would be diagnosing the
    diagnostic.

    Checked on the source rather than on pixels because the claim is about
    *which function is called where*, and the two fields agree everywhere the
    section is not cutting, so most images cannot tell them apart.
    """
    import re

    source = build_viewer_shader("fn sdf(p: vec3<f32>) -> f32 { return length(p) - 1.0; }")

    def body(name: str) -> str:
        start = source.index(f"fn {name}(")
        depth, i = 0, source.index("{", start)
        for j in range(i, len(source)):
            depth += (source[j] == "{") - (source[j] == "}")
            if depth == 0:
                return source[i : j + 1]
        raise AssertionError(f"unterminated {name}")

    # The diagnostics sample the field the scene actually has.
    for name in ("sdf_gradient_magnitude", "sdf_view_color"):
        assert "traced_field" not in body(name), f"{name} reads the sectioned field"
        assert "scene_field" in body(name)
    # The tracer, its refinement and both shadow marches see the section.
    for name in ("trace", "refine_hit", "hard_shadow", "soft_shadow", "sdf_normal"):
        assert "traced_field" in body(name), f"{name} misses the section"
    # And the section itself is exactly an intersection with a half-space.
    assert re.search(r"return max\(field, section_halfspace\(p\)\);", source)
