"""Finite-difference normals, visibility, and physically based shading."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from cadjoint.render.raymarch._constants import (
    _MIN_MARCH_STEP,
    _NORMAL_ZERO_THRESHOLD,
    _SHADOW_T_START,
)
from cadjoint.render.raymarch.camera import _normalize
from cadjoint.render.raymarch.surface import _offset_surface

_PI = jnp.pi


def _normal_fd(
    sdf: Callable[[Array], Array],
    position: Array,
    epsilon: float = 1e-3,
) -> Array:
    """Estimate an SDF normal with symmetric central differences.

    Six samples cost slightly more than a tetrahedral estimate, but avoid its
    directional bias and produce substantially more stable specular highlights
    on edges and small curved surfaces.
    """
    offsets = jnp.eye(3, dtype=position.dtype) * epsilon
    positive = jax.vmap(lambda offset: sdf(position + offset))(offsets)
    negative = jax.vmap(lambda offset: sdf(position - offset))(offsets)
    return positive - negative


def _compute_normal(
    sdf: Callable[[Array], Array],
    position: Array,
    epsilon: float = 1e-3,
) -> Array:
    """Return a unit surface normal estimated with finite differences."""
    raw_normal = _normal_fd(sdf, position, epsilon)
    magnitude = jnp.linalg.norm(raw_normal)
    return raw_normal / jnp.maximum(magnitude, _NORMAL_ZERO_THRESHOLD)


def _cast_shadow(
    sdf: Callable[[Array], Array],
    position: Array,
    normal: Array,
    light_direction: Array,
    steps: int,
    hardness: float,
    max_distance: float = 20.0,
    hit_epsilon: float = 1e-3,
    step_scale: float = 0.9,
) -> Array:
    """Return an early-exit soft-shadow visibility factor in ``[0, 1]``."""
    origin = _offset_surface(position, normal, hit_epsilon)
    initial = (
        jnp.asarray(_SHADOW_T_START),
        jnp.asarray(1.0),
        jnp.asarray(jnp.inf),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def keep_marching(state: tuple[Array, Array, Array, Array]) -> Array:
        distance, visibility, _, count = state
        return (count < steps) & (distance < max_distance) & (visibility > 0.0)

    def march(state: tuple[Array, Array, Array, Array]):
        distance, visibility, previous_surface_distance, count = state
        surface_distance = sdf(origin + light_direction * distance)
        finite_distance = jnp.where(jnp.isfinite(surface_distance), surface_distance, max_distance)
        intersection = finite_distance <= hit_epsilon

        # Estimate the closest surface-to-ray distance between this sample and
        # the previous one. This avoids the step-shaped light leaks produced by
        # evaluating h / t only at march positions, especially around corners.
        has_previous_sample = jnp.isfinite(previous_surface_distance)
        y = jnp.where(
            has_previous_sample,
            finite_distance**2 / jnp.maximum(2.0 * previous_surface_distance, 1e-8),
            0.0,
        )
        valid_closest_estimate = (
            has_previous_sample
            & (previous_surface_distance > 0.0)
            & (finite_distance > 0.0)
            & (y < finite_distance)
        )
        closest_distance = jnp.where(
            valid_closest_estimate,
            jnp.sqrt(jnp.maximum(finite_distance**2 - y**2, 0.0)),
            jnp.abs(finite_distance),
        )
        distance_along_ray = jnp.where(valid_closest_estimate, distance - y, distance)
        distance_along_ray = jnp.maximum(distance_along_ray, _MIN_MARCH_STEP)
        penumbra = jnp.clip(
            hardness * closest_distance / distance_along_ray,
            0.0,
            1.0,
        )
        next_visibility = jnp.minimum(
            visibility,
            jnp.where(intersection, 0.0, penumbra),
        )
        advance = jnp.maximum(jnp.abs(finite_distance) * step_scale, _MIN_MARCH_STEP)
        return distance + advance, next_visibility, finite_distance, count + 1

    _, visibility, _, _ = jax.lax.while_loop(keep_marching, march, initial)
    return visibility


def _ggx_light(
    light_direction: Array,
    light_color: Array,
    normal: Array,
    view_direction: Array,
    base_color: Array,
    roughness: Array,
    metallic: Array,
    visibility: Array,
) -> Array:
    """Evaluate one Cook-Torrance GGX directional light."""
    half_vector = _normalize(view_direction + light_direction)
    n_dot_l = jnp.clip(jnp.dot(normal, light_direction), 0.0, 1.0)
    n_dot_v = jnp.clip(jnp.dot(normal, view_direction), 0.0, 1.0)
    n_dot_h = jnp.clip(jnp.dot(normal, half_vector), 0.0, 1.0)
    v_dot_h = jnp.clip(jnp.dot(view_direction, half_vector), 0.0, 1.0)

    perceptual_roughness = jnp.clip(roughness, 0.04, 1.0)
    alpha = perceptual_roughness**2
    alpha_squared = alpha**2
    distribution_denominator = n_dot_h**2 * (alpha_squared - 1.0) + 1.0
    distribution = alpha_squared / jnp.maximum(
        _PI * distribution_denominator**2,
        1e-6,
    )

    geometry_k = (perceptual_roughness + 1.0) ** 2 / 8.0

    def geometry_schlick(n_dot_direction: Array) -> Array:
        return n_dot_direction / jnp.maximum(
            n_dot_direction * (1.0 - geometry_k) + geometry_k,
            1e-6,
        )

    geometry = geometry_schlick(n_dot_l) * geometry_schlick(n_dot_v)
    f0 = 0.04 * (1.0 - metallic) + base_color * metallic
    fresnel = f0 + (1.0 - f0) * (1.0 - v_dot_h) ** 5
    specular = distribution * geometry * fresnel / jnp.maximum(4.0 * n_dot_v * n_dot_l, 1e-6)
    diffuse = (1.0 - fresnel) * (1.0 - metallic) * base_color / _PI
    return light_color * (diffuse + specular) * n_dot_l * visibility


def _shade_surface(
    sdf: Callable[[Array], Array],
    material: dict,
    position: Array,
    normal: Array,
    ray_direction: Array,
    light_directions: Array,
    light_colors: Array,
    shadow_steps: int,
    shadow_hardness: float,
    ambient: float,
    ambient_color: Array = jnp.ones(3),
    shadow_distance: float = 20.0,
    hit_epsilon: float = 1e-3,
    step_scale: float = 0.9,
) -> Array:
    """Shade a surface using Cook-Torrance GGX and directional lights."""
    base_color = material["color"]
    roughness = material["roughness"]
    metallic = material["metallic"]
    view_direction = -ray_direction

    def shade_light(light_direction: Array, light_color: Array) -> Array:
        visibility = jnp.asarray(1.0)
        if shadow_steps > 0:
            visibility = _cast_shadow(
                sdf,
                position,
                normal,
                light_direction,
                shadow_steps,
                shadow_hardness,
                shadow_distance,
                hit_epsilon,
                step_scale,
            )
        return _ggx_light(
            light_direction,
            light_color,
            normal,
            view_direction,
            base_color,
            roughness,
            metallic,
            visibility,
        )

    direct = jax.vmap(shade_light)(light_directions, light_colors).sum(axis=0)
    return direct + base_color * ambient * ambient_color
