"""Behavioral tests for the forward sphere-tracing renderer."""

import jax.numpy as jnp
import numpy as np

from jaxcad.render.material import Material
from jaxcad.render.raymarch import (
    _ambient_occlusion,
    _camera_rays,
    _cast_shadow,
    _compute_normal,
    _normalize,
    _render_pixel,
    _shade_surface,
    _sphere_trace,
    raymarch,
)
from jaxcad.sdf.primitives import Sphere


def _sphere_sdf(radius=1.0):
    return lambda point: jnp.linalg.norm(point) - radius


def _default_material(**overrides):
    material = Material().as_dict()
    material.update(overrides)
    return material


def _pixel_options(**overrides):
    options = {
        "max_steps": 64,
        "max_distance": 20.0,
        "hit_epsilon": 1e-3,
        "step_scale": 0.9,
        "normal_epsilon": 1e-3,
        "shadow_steps": 12,
        "shadow_distance": 20.0,
        "shadow_hardness": 12.0,
        "ambient": 0.08,
        "ao_steps": 2,
        "ao_step_size": 0.08,
        "ao_strength": 1.0,
        "reflect_steps": 0,
        "refract_steps": 0,
    }
    options.update(overrides)
    return options


def test_normalize_known_vector():
    normal = _normalize(jnp.array([3.0, 4.0, 0.0]))
    np.testing.assert_allclose(normal, [0.6, 0.8, 0.0], atol=1e-6)


def test_camera_center_ray_aligns_with_target():
    camera = jnp.array([0.0, 0.0, 5.0])
    target = jnp.zeros(3)
    rays = _camera_rays(camera, target, (11, 11), fov=0.6)
    np.testing.assert_allclose(rays[60], _normalize(target - camera), atol=1e-5)


def test_camera_wider_fov_increases_corner_angle():
    camera = jnp.array([0.0, 0.0, 5.0])
    target = jnp.zeros(3)

    def corner_angle(fov):
        rays = _camera_rays(camera, target, (11, 11), fov=fov)
        return jnp.arccos(jnp.clip(jnp.dot(rays[60], rays[0]), -1.0, 1.0))

    assert corner_angle(0.9) > corner_angle(0.3)


def test_sphere_trace_hits_at_expected_distance_and_exits_early():
    result = _sphere_trace(
        _sphere_sdf(),
        jnp.array([5.0, 0.0, 0.0]),
        jnp.array([-1.0, 0.0, 0.0]),
        max_steps=64,
    )
    assert bool(result.hit)
    assert jnp.isclose(result.distance, 4.0, atol=2e-3)
    assert int(result.steps) < 8


def test_sphere_trace_miss_stops_at_view_distance():
    result = _sphere_trace(
        _sphere_sdf(),
        jnp.array([0.0, 5.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
        max_steps=64,
        max_distance=12.0,
    )
    assert not bool(result.hit)
    assert float(result.distance) == 12.0
    assert int(result.steps) < 64


def test_sphere_trace_respects_step_budget():
    result = _sphere_trace(
        _sphere_sdf(),
        jnp.array([1.1, 0.0, 5.0]),
        _normalize(jnp.array([0.0, 0.0, -1.0])),
        max_steps=2,
    )
    assert not bool(result.hit)
    assert int(result.steps) == 2


def test_finite_difference_normal_matches_sphere_normal():
    normal = _compute_normal(_sphere_sdf(), jnp.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=2e-3)


def test_ambient_occlusion_is_open_above_plane():
    def plane(point):
        return point[1]

    ao = _ambient_occlusion(
        plane,
        jnp.zeros(3),
        jnp.array([0.0, 1.0, 0.0]),
        steps=5,
        step_size=0.1,
        strength=1.0,
    )
    assert jnp.isclose(ao, 1.0)


def test_ambient_occlusion_darkens_blocked_samples():
    def blocked(_point):
        return jnp.asarray(0.0)

    ao = _ambient_occlusion(
        blocked,
        jnp.zeros(3),
        jnp.array([0.0, 1.0, 0.0]),
        steps=5,
        step_size=0.1,
        strength=1.0,
    )
    assert float(ao) < 0.5


def test_cast_shadow_exits_on_occluder():
    shadow = _cast_shadow(
        _sphere_sdf(),
        jnp.array([0.0, -2.0, 0.0]),
        jnp.array([0.0, -1.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
        steps=64,
        hardness=8.0,
    )
    assert float(shadow) < 0.05


def test_cast_shadow_hardness_changes_penumbra():
    position = jnp.array([1.1, -3.0, 0.0])
    normal = jnp.array([0.0, -1.0, 0.0])
    light = jnp.array([0.0, 1.0, 0.0])
    soft = _cast_shadow(_sphere_sdf(), position, normal, light, 64, 2.0)
    hard = _cast_shadow(_sphere_sdf(), position, normal, light, 64, 32.0)
    assert float(hard) > float(soft)


def test_backlit_surface_without_ambient_is_black():
    color = _shade_surface(
        _sphere_sdf(),
        _default_material(),
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([0.0, 0.0, -1.0]),
        jnp.asarray(1.0),
        jnp.array([[0.0, -1.0, 0.0]]),
        jnp.ones((1, 3)),
        shadow_steps=0,
        shadow_hardness=8.0,
        ambient=0.0,
    )
    np.testing.assert_allclose(color, 0.0, atol=1e-6)


def test_ggx_metallic_highlight_is_tinted_by_base_color():
    common = {
        "sdf": _sphere_sdf(),
        "position": jnp.array([1.0, 0.0, 0.0]),
        "normal": jnp.array([1.0, 0.0, 0.0]),
        "ray_direction": jnp.array([-1.0, 0.0, 0.0]),
        "ambient_occlusion": jnp.asarray(1.0),
        "light_directions": jnp.array([[1.0, 0.0, 0.0]]),
        "light_colors": jnp.ones((1, 3)),
        "shadow_steps": 0,
        "shadow_hardness": 8.0,
        "ambient": 0.0,
    }
    red = jnp.array([1.0, 0.0, 0.0])
    dielectric = _shade_surface(
        material=_default_material(
            color=red,
            metallic=jnp.asarray(0.0),
            roughness=jnp.asarray(0.2),
        ),
        **common,
    )
    metal = _shade_surface(
        material=_default_material(
            color=red,
            metallic=jnp.asarray(1.0),
            roughness=jnp.asarray(0.2),
        ),
        **common,
    )
    assert float(dielectric[1]) > float(metal[1])


def test_render_pixel_uses_hard_visibility_for_miss():
    background = jnp.array([0.2, 0.4, 0.6])
    pixel = _render_pixel(
        _sphere_sdf(),
        lambda _point: _default_material(),
        jnp.array([0.0, 0.0, 5.0]),
        _normalize(jnp.array([1.2, 0.0, -5.0])),
        jnp.array([[0.5, 1.0, 0.3]]),
        jnp.ones((1, 3)),
        background,
        **_pixel_options(),
    )
    np.testing.assert_array_equal(pixel, background)


def test_raymarch_lit_side_is_brighter_than_dark_side():
    image = raymarch(
        _sphere_sdf(),
        camera_pos=jnp.array([0.0, 0.0, 5.0]),
        look_at=jnp.zeros(3),
        light_dirs=jnp.array([1.0, 0.0, 0.0]),
        resolution=(32, 32),
        max_steps=64,
        shadow_steps=0,
        ao_steps=0,
        ambient=0.0,
    )
    middle = image.shape[0] // 2
    assert image[middle, middle + 4].mean() > image[middle, middle - 4].mean()


def test_raymarch_background_color_on_miss():
    background = jnp.array([0.2, 0.4, 0.6])
    gamma = 2.2
    image = raymarch(
        _sphere_sdf(radius=0.01),
        resolution=(8, 8),
        max_steps=8,
        background_color=background,
        gamma=gamma,
    )
    expected = np.asarray(background) ** (1.0 / gamma)
    np.testing.assert_allclose(image[0, 0], expected, atol=1e-4)


def test_single_light_vector_and_matrix_match():
    light = jnp.array([0.5, 1.0, 0.3])
    options = {"resolution": (16, 16), "max_steps": 32, "shadow_steps": 8, "ao_steps": 0}
    vector_image = raymarch(_sphere_sdf(), light_dirs=light, **options)
    matrix_image = raymarch(_sphere_sdf(), light_dirs=light[None], **options)
    np.testing.assert_allclose(vector_image, matrix_image, atol=1e-5)


def test_primitive_material_changes_render():
    options = {"resolution": (20, 20), "max_steps": 32, "shadow_steps": 8, "ao_steps": 0}
    plain = raymarch(_sphere_sdf(), **options)
    green = raymarch(
        Sphere(radius=1.0, material=Material(color=[0.2, 0.8, 0.2])),
        **options,
    )
    assert not np.allclose(plain, green, atol=1e-3)


def test_reflection_uses_background_on_secondary_miss():
    background = jnp.array([0.5, 0.2, 0.8])
    pixel = _render_pixel(
        _sphere_sdf(),
        lambda _point: _default_material(reflectivity=jnp.asarray(1.0)),
        jnp.array([0.0, 0.0, 5.0]),
        jnp.array([0.0, 0.0, -1.0]),
        jnp.array([[0.0, 1.0, 0.0]]),
        jnp.ones((1, 3)),
        background,
        **_pixel_options(reflect_steps=32, shadow_steps=0, ao_steps=0),
    )
    np.testing.assert_allclose(pixel, background, atol=0.02)


def test_refraction_produces_finite_image():
    glass = Sphere(
        radius=1.0,
        material=Material(color=[0.8, 0.95, 1.0], opacity=0.1, ior=1.5),
    )
    image = raymarch(
        glass,
        resolution=(16, 16),
        max_steps=48,
        shadow_steps=8,
        ao_steps=0,
        refract_steps=32,
    )
    assert np.isfinite(image).all()
