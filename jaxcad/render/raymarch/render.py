"""Forward SDF rendering pipeline and public convenience APIs."""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from jaxcad.render.raymarch.camera import _camera_rays
from jaxcad.render.raymarch.shade import (
    _compute_normal,
    _shade_surface,
)
from jaxcad.render.raymarch.surface import _offset_surface
from jaxcad.render.raymarch.trace import (
    _fresnel_schlick,
    _sphere_trace,
    _trace_through_glass,
)
from jaxcad.render.scene import Scene
from jaxcad.render.settings import RenderSettings, ToneMapping


def _default_material_at(_position: Array) -> dict:
    """Return the renderer's stable fallback material."""
    return {
        "color": jnp.ones(3),
        "roughness": jnp.asarray(0.5),
        "metallic": jnp.asarray(0.0),
        "opacity": jnp.asarray(1.0),
        "ior": jnp.asarray(1.5),
        "reflectivity": jnp.asarray(0.0),
    }


@lru_cache(maxsize=128)
def _parameterized_scene_factory(geometry):
    """Compile a stable geometry evaluator while keeping parameters dynamic."""
    from jaxcad.functionalize import functionalize_scene

    return functionalize_scene(geometry)


@lru_cache(maxsize=128)
def _callable_scene_factory(sdf):
    """Wrap a plain SDF callable in the same factory contract."""
    material_fn = getattr(sdf, "material_at", _default_material_at)

    def factory(_free_parameters: dict, _fixed_parameters: dict):
        return sdf, material_fn

    return factory


def _prepare_scene(sdf) -> tuple[Callable, dict, dict]:
    """Return a cached evaluator factory and current parameter values."""
    from jaxcad.fluent import Fluent

    if isinstance(sdf, Fluent):
        from jaxcad.extraction import extract_parameters

        free_parameters, fixed_parameters, _ = extract_parameters(sdf)
        return _parameterized_scene_factory(sdf), free_parameters, fixed_parameters
    return _callable_scene_factory(sdf), {}, {}


def _render_pixel(
    sdf: Callable[[Array], Array],
    material_fn: Callable[[Array], dict],
    ray_origin: Array,
    ray_direction: Array,
    light_directions: Array,
    light_colors: Array,
    background_color: Array,
    max_steps: int,
    max_distance: float,
    hit_epsilon: float,
    step_scale: float,
    normal_epsilon: float,
    shadow_steps: int,
    shadow_distance: float,
    shadow_hardness: float,
    ambient: float,
    edge_width: float,
    reflect_steps: int,
    refract_steps: int,
    env_fn: Callable[[Array], Array] | None = None,
) -> Array:
    """Trace and shade one ray with reconstructed silhouette coverage."""

    def background(direction: Array) -> Array:
        return env_fn(direction) if env_fn is not None else background_color

    def shade_hit(position: Array, direction: Array) -> tuple[Array, dict, Array]:
        normal = _compute_normal(sdf, position, normal_epsilon)
        material = material_fn(position)
        ambient_color = background(normal) if env_fn is not None else jnp.ones(3)
        color = _shade_surface(
            sdf,
            material,
            position,
            normal,
            direction,
            light_directions,
            light_colors,
            shadow_steps,
            shadow_hardness,
            ambient,
            ambient_color,
            shadow_distance,
            hit_epsilon,
            step_scale,
        )
        return color, material, normal

    primary = _sphere_trace(
        sdf,
        ray_origin,
        ray_direction,
        max_steps,
        max_distance,
        hit_epsilon,
        step_scale,
    )
    hit_position = ray_origin + primary.distance * ray_direction
    closest_position = ray_origin + primary.closest_distance * ray_direction
    sample_position = jnp.where(primary.hit, hit_position, closest_position)
    surface_color, material, normal = shade_hit(sample_position, ray_direction)

    def render_surface(_unused: None) -> Array:
        shaded_color = surface_color
        reflected_color = surface_color

        if reflect_steps > 0:
            reflected_direction = ray_direction - 2.0 * jnp.dot(ray_direction, normal) * normal
            reflected_origin = _offset_surface(sample_position, normal, hit_epsilon)
            reflection = _sphere_trace(
                sdf,
                reflected_origin,
                reflected_direction,
                reflect_steps,
                max_distance,
                hit_epsilon,
                step_scale,
            )
            reflected_position = reflected_origin + reflection.distance * reflected_direction
            reflected_color = jax.lax.cond(
                reflection.hit,
                lambda _: shade_hit(reflected_position, reflected_direction)[0],
                lambda _: background(reflected_direction),
                operand=None,
            )
            reflectivity = material.get("reflectivity", jnp.asarray(0.0))
            shaded_color = shaded_color * (1.0 - reflectivity) + reflected_color * reflectivity

        opacity = material.get("opacity", jnp.asarray(1.0))
        if refract_steps == 0:
            return shaded_color * opacity + background(ray_direction) * (1.0 - opacity)

        ior = material.get("ior", jnp.asarray(1.5))
        exit_position, exit_direction, exit_normal, exited = _trace_through_glass(
            sdf,
            sample_position,
            ray_direction,
            normal,
            ior,
            refract_steps,
            max_distance,
            hit_epsilon,
            step_scale,
            normal_epsilon,
        )

        def trace_transmission(_unused: None) -> Array:
            transmission_origin = _offset_surface(exit_position, exit_normal, hit_epsilon)
            transmission = _sphere_trace(
                sdf,
                transmission_origin,
                exit_direction,
                refract_steps,
                max_distance,
                hit_epsilon,
                step_scale,
            )
            transmission_position = transmission_origin + transmission.distance * exit_direction
            return jax.lax.cond(
                transmission.hit,
                lambda _: shade_hit(transmission_position, exit_direction)[0],
                lambda _: background(exit_direction),
                operand=None,
            )

        transmitted_color = jax.lax.cond(
            exited,
            trace_transmission,
            lambda _: background(ray_direction),
            operand=None,
        )
        transmitted_color = transmitted_color * material["color"]
        fresnel = _fresnel_schlick(jnp.dot(-ray_direction, normal), ior)
        dielectric_color = reflected_color * fresnel + transmitted_color * (1.0 - fresnel)
        return shaded_color * opacity + dielectric_color * (1.0 - opacity)

    def render_silhouette(_unused: None) -> Array:
        opacity = material.get("opacity", jnp.asarray(1.0))
        return surface_color * opacity + background(ray_direction) * (1.0 - opacity)

    safe_edge_width = jnp.maximum(edge_width, 1e-8)
    edge_proximity = jnp.clip(
        1.0 - primary.closest_surface_distance / safe_edge_width,
        0.0,
        1.0,
    )
    smooth_proximity = edge_proximity**2 * (3.0 - 2.0 * edge_proximity)
    miss_coverage = jnp.where(edge_width > 0.0, 0.5 * smooth_proximity, 0.0)
    coverage = jnp.where(primary.hit, 1.0, miss_coverage)
    surface_sample = jax.lax.cond(
        primary.hit,
        render_surface,
        render_silhouette,
        operand=None,
    )
    return surface_sample * coverage + background(ray_direction) * (1.0 - coverage)


@partial(
    jax.jit,
    static_argnames=(
        "scene_factory",
        "max_steps",
        "shadow_steps",
        "reflect_steps",
        "refract_steps",
    ),
)
def _render_image(
    scene_factory: Callable,
    free_parameters: dict,
    fixed_parameters: dict,
    camera_position: Array,
    rays: Array,
    light_directions: Array,
    light_colors: Array,
    background_color: Array,
    max_steps: int,
    max_distance: float,
    hit_epsilon: float,
    step_scale: float,
    normal_epsilon: float,
    shadow_steps: int,
    shadow_distance: float,
    shadow_hardness: float,
    ambient: float,
    edge_width: float,
    reflect_steps: int,
    refract_steps: int,
    env_map: Array | None,
) -> Array:
    """Render flat camera rays in linear color space.

    The module-level JIT is intentionally cached across public ``raymarch``
    calls. Repeated renders of the same scene and image shape reuse the
    compiled program instead of rebuilding a closure on every call.
    """
    sdf, material_fn = scene_factory(free_parameters, fixed_parameters)
    if env_map is not None:
        from jaxcad.render.raymarch.env import _sample_env_map

        def environment(direction: Array) -> Array:
            return _sample_env_map(env_map, direction)

    else:
        environment = None

    return jax.vmap(
        lambda direction: _render_pixel(
            sdf,
            material_fn,
            camera_position,
            direction,
            light_directions,
            light_colors,
            background_color,
            max_steps,
            max_distance,
            hit_epsilon,
            step_scale,
            normal_epsilon,
            shadow_steps,
            shadow_distance,
            shadow_hardness,
            ambient,
            edge_width,
            reflect_steps,
            refract_steps,
            environment,
        )
    )(rays)


def _render_with_settings(
    sdf: Callable[[Array], Array],
    camera_position: Array,
    camera_target: Array,
    camera_fov: float,
    light_directions: Array,
    light_colors: Array | None,
    background_color: Array,
    environment_map: Array | None,
    settings: RenderSettings,
) -> np.ndarray:
    height, width = settings.resolution
    camera_position = jnp.asarray(camera_position, dtype=jnp.float32)
    camera_target = jnp.asarray(camera_target, dtype=jnp.float32)
    if camera_position.shape != (3,) or camera_target.shape != (3,):
        raise ValueError("camera position and target must be 3D vectors")
    if camera_fov <= 0.0:
        raise ValueError("camera fov must be positive")
    if np.allclose(camera_position, camera_target):
        raise ValueError("camera position and target must be different")
    light_directions = jnp.atleast_2d(jnp.asarray(light_directions, dtype=jnp.float32))
    if light_directions.ndim != 2 or light_directions.shape[1] != 3:
        raise ValueError("light_directions must have shape (3,) or (N, 3)")
    light_norms = jnp.linalg.norm(
        light_directions,
        axis=1,
        keepdims=True,
    )
    if bool(jnp.any(light_norms <= 1e-8)):
        raise ValueError("light directions cannot be zero vectors")
    light_directions = light_directions / light_norms
    if light_colors is None:
        light_colors = jnp.ones_like(light_directions)
    else:
        light_colors = jnp.atleast_2d(jnp.asarray(light_colors, dtype=jnp.float32))
        if light_colors.shape != light_directions.shape:
            raise ValueError("light_colors must match light_directions")

    background_color = jnp.asarray(background_color, dtype=jnp.float32)
    if background_color.shape != (3,):
        raise ValueError("background_color must be an RGB vector")
    if environment_map is not None:
        environment_map = jnp.asarray(environment_map, dtype=jnp.float32)
        if environment_map.ndim != 3 or environment_map.shape[2] != 3:
            raise ValueError("environment_map must have shape (height, width, 3)")
    render_resolution = (
        height * settings.aa_samples,
        width * settings.aa_samples,
    )
    scene_distance = float(np.linalg.norm(np.asarray(camera_position - camera_target)))
    edge_width = (
        settings.silhouette_smoothing * 2.0 * camera_fov * scene_distance / min(render_resolution)
    )
    rays = _camera_rays(
        camera_position,
        camera_target,
        render_resolution,
        camera_fov,
    )
    scene_factory, free_parameters, fixed_parameters = _prepare_scene(sdf)
    pixels = _render_image(
        scene_factory=scene_factory,
        free_parameters=free_parameters,
        fixed_parameters=fixed_parameters,
        camera_position=camera_position,
        rays=rays,
        light_directions=light_directions,
        light_colors=light_colors,
        background_color=background_color,
        max_steps=settings.max_steps,
        max_distance=settings.max_distance,
        hit_epsilon=settings.hit_epsilon,
        step_scale=settings.step_scale,
        normal_epsilon=settings.normal_epsilon,
        shadow_steps=settings.shadow_steps,
        shadow_distance=settings.shadow_distance,
        shadow_hardness=settings.shadow_hardness,
        ambient=settings.ambient,
        edge_width=edge_width,
        reflect_steps=settings.reflect_steps,
        refract_steps=settings.refract_steps,
        env_map=environment_map,
    )

    render_height, render_width = render_resolution
    image = pixels.reshape(render_height, render_width, 3)
    if settings.aa_samples > 1:
        image = image.reshape(
            height,
            settings.aa_samples,
            width,
            settings.aa_samples,
            3,
        ).mean(axis=(1, 3))

    image = jnp.maximum(image * settings.exposure, 0.0)
    if settings.tone_mapping == "aces":
        # Narkowicz ACES fit: compresses high-energy GGX highlights instead of
        # clipping isolated pixels into white fireflies.
        image = jnp.clip(
            image * (2.51 * image + 0.03) / (image * (2.43 * image + 0.59) + 0.14),
            0.0,
            1.0,
        )
    image = image ** (1.0 / settings.gamma)
    return np.asarray(jnp.clip(image, 0.0, 1.0))


def render_scene(
    scene: Scene,
    settings: RenderSettings | None = None,
) -> np.ndarray:
    """Render a :class:`Scene` with an explicit quality preset or settings."""
    settings = settings or RenderSettings.balanced()
    return _render_with_settings(
        scene.geometry,
        scene.camera.position,
        scene.camera.target,
        scene.camera.fov,
        scene.light_directions,
        scene.light_colors,
        scene.background_color,
        scene.environment_map,
        settings,
    )


def raymarch(
    sdf: Callable[[Array], Array],
    camera_pos: Array = jnp.array([5.0, 5.0, 5.0]),
    look_at: Array = jnp.array([0.0, 0.0, 0.0]),
    light_dirs: Array = jnp.array([0.5, 1.0, 0.3]),
    light_colors: Array | None = None,
    resolution: tuple[int, int] = (200, 200),
    fov: float = 0.6,
    max_steps: int = 96,
    max_dist: float = 20.0,
    hit_epsilon: float = 1e-3,
    step_scale: float = 0.9,
    normal_eps: float = 1e-3,
    shadow_steps: int = 32,
    shadow_distance: float = 20.0,
    shadow_hardness: float = 12.0,
    ambient: float = 0.12,
    aa_samples: int = 1,
    silhouette_smoothing: float = 0.75,
    exposure: float = 1.0,
    tone_mapping: ToneMapping = "aces",
    gamma: float = 2.2,
    background_color: Array = jnp.array([0.0, 0.0, 0.0]),
    reflect_steps: int = 0,
    refract_steps: int = 0,
    env_map: Array | None = None,
) -> np.ndarray:
    """Render an SDF to an ``(height, width, 3)`` NumPy image.

    This compatibility-oriented convenience function exposes individual
    controls. New code can group the same values with :class:`RenderSettings`
    and call :func:`render_scene`.
    """
    settings = RenderSettings(
        resolution=resolution,
        max_steps=max_steps,
        max_distance=max_dist,
        hit_epsilon=hit_epsilon,
        step_scale=step_scale,
        normal_epsilon=normal_eps,
        shadow_steps=shadow_steps,
        shadow_distance=shadow_distance,
        shadow_hardness=shadow_hardness,
        ambient=ambient,
        aa_samples=aa_samples,
        silhouette_smoothing=silhouette_smoothing,
        exposure=exposure,
        tone_mapping=tone_mapping,
        gamma=gamma,
        reflect_steps=reflect_steps,
        refract_steps=refract_steps,
    )
    return _render_with_settings(
        sdf,
        camera_pos,
        look_at,
        fov,
        light_dirs,
        light_colors,
        background_color,
        env_map,
        settings,
    )


def render_raymarched(
    sdf: Callable[[Array], Array],
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    **render_options,
) -> plt.Axes:
    """Render an SDF with :func:`raymarch` and display it with matplotlib."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(raymarch(sdf, **render_options), vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(title or "Raymarched Render", fontsize=12)
    return ax
