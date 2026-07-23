"""Tests for the forward-rendering scene abstraction and quality presets."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcad.render import Camera, RenderSettings, Scene, raymarch, render_scene
from jaxcad.sdf.primitives import Sphere


def test_quality_presets_increase_work_and_fidelity():
    draft = RenderSettings.draft((32, 40))
    balanced = RenderSettings.balanced((32, 40))
    high = RenderSettings.high_quality((32, 40))

    assert draft.resolution == balanced.resolution == high.resolution
    assert draft.max_steps < balanced.max_steps < high.max_steps
    assert draft.shadow_steps < balanced.shadow_steps < high.shadow_steps
    assert draft.aa_samples < balanced.aa_samples < high.aa_samples
    assert draft.hit_epsilon > balanced.hit_epsilon > high.hit_epsilon


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resolution", (0, 10)),
        ("max_steps", 0),
        ("step_scale", 1.1),
        ("aa_samples", 0),
        ("silhouette_smoothing", -0.1),
        ("gamma", 0.0),
        ("exposure", 0.0),
        ("tone_mapping", "invalid"),
        ("ambient", -0.1),
        ("shadow_hardness", 0.0),
    ],
)
def test_render_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(RenderSettings(), **{field: value})


def test_camera_rejects_nonpositive_fov():
    with pytest.raises(ValueError, match="fov"):
        Camera(fov=0.0)


def test_renderer_rejects_zero_light_direction():
    with pytest.raises(ValueError, match="zero vectors"):
        raymarch(Sphere(1.0), light_dirs=(0.0, 0.0, 0.0), resolution=(4, 4))


def test_render_scene_uses_camera_and_settings():
    scene = Scene(
        Sphere(radius=1.0),
        camera=Camera(position=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0)),
        light_directions=((1.0, 0.0, 0.0),),
        background_color=(0.1, 0.2, 0.3),
    )
    image = render_scene(scene, RenderSettings.draft((12, 16)))
    assert image.shape == (12, 16, 3)
    assert np.isfinite(image).all()


def test_render_scene_matches_low_level_raymarch():
    geometry = Sphere(radius=1.0)
    camera = Camera(position=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0), fov=0.6)
    settings = RenderSettings.draft((12, 12))
    scene = Scene(
        geometry,
        camera=camera,
        light_directions=((0.5, 1.0, 0.3),),
    )
    structured = render_scene(scene, settings)
    direct = raymarch(
        geometry,
        camera_pos=camera.position,
        look_at=camera.target,
        fov=camera.fov,
        light_dirs=scene.light_directions,
        resolution=settings.resolution,
        max_steps=settings.max_steps,
        max_dist=settings.max_distance,
        hit_epsilon=settings.hit_epsilon,
        step_scale=settings.step_scale,
        normal_eps=settings.normal_epsilon,
        shadow_steps=settings.shadow_steps,
        shadow_distance=settings.shadow_distance,
        shadow_hardness=settings.shadow_hardness,
        ambient=settings.ambient,
        aa_samples=settings.aa_samples,
        silhouette_smoothing=settings.silhouette_smoothing,
        gamma=settings.gamma,
        exposure=settings.exposure,
        tone_mapping=settings.tone_mapping,
    )
    np.testing.assert_allclose(structured, direct, atol=1e-6)


def test_cached_renderer_uses_current_geometry_parameters():
    sphere = Sphere(radius=0.4)
    scene = Scene(
        sphere,
        camera=Camera(position=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0)),
    )
    settings = RenderSettings.draft((24, 24))

    small = render_scene(scene, settings)
    sphere.params["radius"].value = jnp.asarray(1.2)
    large = render_scene(scene, settings)

    small_coverage = np.count_nonzero(small.mean(axis=2) > 1e-3)
    large_coverage = np.count_nonzero(large.mean(axis=2) > 1e-3)
    assert large_coverage > small_coverage
