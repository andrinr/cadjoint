"""Validate every WGSL shader the playground ships through a real GPU compiler.

Shader bugs otherwise surface only in a browser with WebGPU, which CI and
headless test runs usually lack. ``wgpu`` (wgpu-native/naga) compiles the same
WGSL a browser would, so syntax and type errors fail here instead.
"""

import struct
from pathlib import Path

import pytest

wgpu = pytest.importorskip("wgpu", reason="wgpu is needed to validate WGSL")

from jaxcad.backends.wgsl import compile_scene_to_wgsl  # noqa: E402
from jaxcad.sdf.boolean import Union  # noqa: E402
from jaxcad.sdf.primitives import Sphere  # noqa: E402
from jaxcad.viewer._pathtracer import (  # noqa: E402
    WGSL_PRESENT_TEMPLATE,
    build_path_tracer_shader,
)
from jaxcad.viewer._webgpu import build_viewer_shader  # noqa: E402

OVERLAY_WGSL = Path(__file__).resolve().parents[2] / "frontend/src/viewer/overlay.wgsl"


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
    from jaxcad.construction import PolygonProfile, extrude, revolve

    profile = PolygonProfile([[0.0, 0.0], [2.0, 0.0], [1.5, 1.2], [0.2, 1.0]], name="p")
    section = PolygonProfile(
        [[0.7, -0.2], [1.0, -0.2], [1.0, 0.2], [0.7, 0.2]],
        name="section",
    )
    code = compile_scene_to_wgsl(Union(extrude(profile, depth=0.8), revolve(section)))
    compile_wgsl(device, build_viewer_shader(code), "sketch preview")
    compile_wgsl(device, build_path_tracer_shader(code), "sketch path tracer")


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


@pytest.mark.skipif(not OVERLAY_WGSL.is_file(), reason="frontend sources not present")
@pytest.mark.parametrize(
    ("entry_vertex", "entry_fragment", "stride", "attributes"),
    [
        (
            "vs_edge",
            "fs_edge",
            40,
            [
                {"shader_location": 0, "offset": 0, "format": "float32x3"},
                {"shader_location": 1, "offset": 12, "format": "float32x3"},
                {"shader_location": 2, "offset": 24, "format": "float32x4"},
            ],
        ),
        (
            "vs_handle",
            "fs_handle",
            32,
            [
                {"shader_location": 0, "offset": 0, "format": "float32x3"},
                {"shader_location": 1, "offset": 12, "format": "float32x4"},
                {"shader_location": 2, "offset": 28, "format": "float32"},
            ],
        ),
    ],
)
def test_overlay_pipelines_match_the_renderer(
    device, entry_vertex, entry_fragment, stride, attributes
):
    module = device.create_shader_module(code=OVERLAY_WGSL.read_text())
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={
            "module": module,
            "entry_point": entry_vertex,
            "buffers": [
                {"array_stride": stride, "step_mode": "instance", "attributes": attributes}
            ],
        },
        fragment={
            "module": module,
            "entry_point": entry_fragment,
            "targets": [
                {
                    "format": COLOR_FORMAT,
                    "blend": {
                        "color": {
                            "src_factor": "src-alpha",
                            "dst_factor": "one-minus-src-alpha",
                            "operation": "add",
                        },
                        "alpha": {
                            "src_factor": "one",
                            "dst_factor": "one-minus-src-alpha",
                            "operation": "add",
                        },
                    },
                }
            ],
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


def render_preview(device, scene_code, display=PERSPECTIVE, size=64, camera=(3.0, 2.0, 4.0)):
    """Render one preview frame and return its red channel, one byte per pixel."""
    color = device.create_texture(
        size=(size, size, 1),
        format="rgba8unorm",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    depth = device.create_texture(
        size=(size, size, 1),
        format="depth32float",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )
    module = device.create_shader_module(code=build_viewer_shader(scene_code))
    bind_group_layout, pipeline_layout = shared_layout(device, [0, 2])
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
    scene_buffer = device.create_buffer_with_data(
        data=struct.pack(
            "28f",
            size,
            size,
            0,
            0,
            camera[0],
            camera[1],
            camera[2],
            0,
            0,
            0,
            0,
            0,
            0.55,
            0.8,
            0.35,
            3.0,
            0.035,
            0.045,
            0.035,
            1.0,
            0,
            0,
            0,
            0,
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
    render_pass.draw(3)
    render_pass.end()
    device.queue.submit([encoder.finish()])

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
    [("top", (0.0, 6.0, 0.0)), ("bottom", (0.0, -6.0, 0.0))],
)
def test_pole_views_do_not_degenerate(device, scene_code, label, camera):
    """Looking straight down or up is where cross(forward, +Y) collapses.

    Without the reference-vector fallback in camera_basis the frame is NaN and
    the frame comes out blank, so this renders from directly overhead.
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
