"""Validate every WGSL shader the playground ships through a real GPU compiler.

Shader bugs otherwise surface only in a browser with WebGPU, which CI and
headless test runs usually lack. ``wgpu`` (wgpu-native/naga) compiles the same
WGSL a browser would, so syntax and type errors fail here instead.
"""

import struct
from pathlib import Path

import pytest

wgpu = pytest.importorskip("wgpu", reason="wgpu is needed to validate WGSL")

from cadjoint.backends.wgsl import compile_scene_to_wgsl  # noqa: E402
from cadjoint.sdf.boolean import Union  # noqa: E402
from cadjoint.sdf.primitives import Sphere  # noqa: E402
from cadjoint.viewer._pathtracer import (  # noqa: E402
    WGSL_PRESENT_TEMPLATE,
    build_path_tracer_shader,
)
from cadjoint.viewer._webgpu import build_viewer_shader  # noqa: E402

OVERLAY_WGSL = Path(__file__).resolve().parents[2] / "frontend/src/viewer/overlay.wgsl"
SIMULATION_WGSL = Path(__file__).resolve().parents[2] / "frontend/src/viewer/simulation.wgsl"


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
            32,
            "instance",
            OVERLAY_BLEND,
            [
                {"shader_location": 0, "offset": 0, "format": "float32x3"},
                {"shader_location": 1, "offset": 12, "format": "float32x4"},
                {"shader_location": 2, "offset": 28, "format": "float32"},
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


def parameter_bind_group(device, layout, program):
    """The program's current values, packed and bound."""
    buffer = device.create_buffer_with_data(
        data=program.buffer().tobytes(), usage=wgpu.BufferUsage.UNIFORM
    )
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
        32,
        [
            {"shader_location": 0, "offset": 0, "format": "float32x3"},
            {"shader_location": 1, "offset": 12, "format": "float32x4"},
            {"shader_location": 2, "offset": 28, "format": "float32"},
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
    handles = device.create_buffer(size=32 * 4, usage=wgpu.BufferUsage.VERTEX)

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
    module = device.create_shader_module(code=build_viewer_shader(scene_code))
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
            program.group, parameter_bind_group(device, parameter_layout, program)
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
    assert [
        index for index, value in enumerate(xrayed) if value < 1.0
    ] == covered, "an x-rayed solid reported a different silhouette to the depth buffer"
    for index in covered:
        assert xrayed[index] == pytest.approx(
            opaque[index], abs=1e-6
        ), "an x-rayed solid wrote a different depth than the solid it is"

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
    assert (
        deviation <= PIXEL_TOLERANCE
    ), f"{label}: the uniform form moved a pixel by {deviation} levels"


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
