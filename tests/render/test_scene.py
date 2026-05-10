"""Tests for Scene, Camera, and functionalize_render."""

import jax
import jax.numpy as jnp

from jaxcad.extraction import extract_parameters
from jaxcad.geometry.parameters import Scalar, Vector
from jaxcad.render import Camera, Scene, functionalize_render
from jaxcad.sdf.primitives import Sphere


def _make_scene(free_camera=False, free_lights=False, free_bg=False):
    """Helper: unit sphere scene with configurable free params."""
    sdf = Sphere(radius=1.0)
    cam = Camera(
        camera_pos=Vector([0.0, 0.0, 5.0], free=free_camera, name="camera_pos"),
        look_at=Vector([0.0, 0.0, 0.0], free=False, name="look_at"),
        fov=Scalar(0.6, free=False, name="fov"),
    )
    if free_lights:
        light_dirs = [Vector([0.5, 1.0, 0.3], free=True, name="light_dir_0")]
        light_colors = [Vector([1.0, 1.0, 1.0], free=True, name="light_color_0")]
    else:
        light_dirs = None
        light_colors = None
    bg = Vector([0.0, 0.0, 0.0], free=free_bg, name="bg_color") if free_bg else None
    return Scene(sdf, cam, light_dirs=light_dirs, light_colors=light_colors, bg_color=bg)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def test_camera_params_keys():
    cam = Camera(
        camera_pos=Vector([0.0, 0.0, 5.0], name="camera_pos"),
        look_at=Vector([0.0, 0.0, 0.0], name="look_at"),
        fov=Scalar(0.6, name="fov"),
    )
    assert set(cam.params.keys()) == {"camera_pos", "look_at", "fov"}


def test_camera_has_no_children():
    cam = Camera(
        camera_pos=Vector([0.0, 0.0, 5.0], name="camera_pos"),
        look_at=Vector([0.0, 0.0, 0.0], name="look_at"),
        fov=Scalar(0.6, name="fov"),
    )
    assert cam.children() == []


# ---------------------------------------------------------------------------
# Scene structure
# ---------------------------------------------------------------------------


def test_scene_params_has_bg_color():
    scene = _make_scene()
    assert "bg_color" in scene.params


def test_scene_children_include_camera():
    scene = _make_scene()
    assert scene.camera in scene.children()


def test_scene_fixed_lights_stored_as_arrays():
    scene = _make_scene(free_lights=False)
    assert not scene.free_lights
    assert hasattr(scene, "light_dirs")
    assert scene.light_dirs.shape[1] == 3


def test_scene_free_lights_stored_in_params():
    scene = _make_scene(free_lights=True)
    assert scene.free_lights
    assert "light_dir_0" in scene.params
    assert "light_color_0" in scene.params


def test_scene_default_light_dirs_normalised():
    scene = _make_scene(free_lights=False)
    norms = jnp.linalg.norm(scene.light_dirs, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# extract_parameters round-trip
# ---------------------------------------------------------------------------


def test_extract_parameters_fixed_scene():
    """With all params fixed, free_params should be empty."""
    scene = _make_scene(free_camera=False, free_lights=False, free_bg=False)
    free, _, _ = extract_parameters(scene)
    assert len(free) == 0


def test_extract_parameters_free_camera():
    scene = _make_scene(free_camera=True)
    free, _, _ = extract_parameters(scene)
    assert "camera_pos" in free
    assert free["camera_pos"].shape == (3,)


def test_extract_parameters_free_bg():
    scene = _make_scene(free_bg=True)
    free, _, _ = extract_parameters(scene)
    assert "bg_color" in free


def test_extract_parameters_free_lights():
    scene = _make_scene(free_lights=True)
    free, _, _ = extract_parameters(scene)
    assert "light_dir_0" in free
    assert "light_color_0" in free


# ---------------------------------------------------------------------------
# functionalize_render
# ---------------------------------------------------------------------------


def test_functionalize_render_returns_callable():
    scene = _make_scene()
    render_fn = functionalize_render(scene)
    assert callable(render_fn)


def test_functionalize_render_output_shape():
    scene = _make_scene()
    free, _, _ = extract_parameters(scene)
    render_fn = functionalize_render(scene)
    img = render_fn(free, resolution=(8, 12))
    assert img.shape == (8, 12, 3)


def test_functionalize_render_values_in_unit_range():
    scene = _make_scene()
    free, _, _ = extract_parameters(scene)
    render_fn = functionalize_render(scene)
    img = render_fn(free, resolution=(8, 8))
    assert float(img.min()) >= 0.0
    assert float(img.max()) <= 1.0


def test_functionalize_render_grad_wrt_camera_pos_finite():
    """Gradient of mean pixel brightness w.r.t. camera_pos should be finite."""
    scene = _make_scene(free_camera=True)
    free, _, _ = extract_parameters(scene)
    render_fn = functionalize_render(scene, fd_normals=True)

    def loss(fp):
        return render_fn(fp, resolution=(8, 8)).mean()

    grad = jax.grad(loss)(free)
    assert jnp.isfinite(grad["camera_pos"]).all()


def test_functionalize_render_grad_nonzero_for_free_camera():
    """Moving the camera changes the render, so the gradient must be non-zero."""
    scene = _make_scene(free_camera=True)
    free, _, _ = extract_parameters(scene)
    render_fn = functionalize_render(scene, fd_normals=True)

    def loss(fp):
        return render_fn(fp, resolution=(8, 8)).mean()

    grad = jax.grad(loss)(free)
    assert jnp.any(grad["camera_pos"] != 0.0)


def test_functionalize_render_free_lights_grad_finite():
    """Gradient w.r.t. free light direction should be finite."""
    scene = _make_scene(free_lights=True)
    free, _, _ = extract_parameters(scene)
    render_fn = functionalize_render(scene, fd_normals=True)

    def loss(fp):
        return render_fn(fp, resolution=(8, 8)).mean()

    grad = jax.grad(loss)(free)
    assert jnp.isfinite(grad["light_dir_0"]).all()
