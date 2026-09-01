"""Forward-only sphere tracing and geometric optics helpers."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cadjoint.render.raymarch._constants import _MIN_MARCH_STEP
from cadjoint.render.raymarch.surface import _offset_surface


class TraceResult(NamedTuple):
    """Result of a ray/surface intersection query.

    ``distance`` is the travelled distance along the ray. ``hit`` is a hard
    visibility result, and ``steps`` reports the amount of work performed.
    ``closest_distance`` and ``closest_surface_distance`` preserve the nearest
    approach for screen-space silhouette reconstruction on rays that narrowly
    miss a surface.
    """

    distance: Array
    hit: Array
    steps: Array
    closest_distance: Array
    closest_surface_distance: Array


def _sphere_trace(
    sdf: Callable[[Array], Array],
    origin: Array,
    direction: Array,
    max_steps: int,
    max_distance: float = 20.0,
    hit_epsilon: float = 1e-3,
    step_scale: float = 0.9,
) -> TraceResult:
    """Find the first SDF surface along a ray with early termination.

    The march stops as soon as it hits a surface, leaves the configured view
    distance, encounters a non-finite SDF value, or exhausts ``max_steps``.
    ``step_scale`` is a safety factor for approximate distance fields; exact
    SDFs can use ``1.0`` for the fewest steps.
    """

    initial_distance = sdf(origin)
    initial_abs_distance = jnp.where(
        jnp.isfinite(initial_distance),
        jnp.abs(initial_distance),
        jnp.asarray(jnp.inf),
    )
    initial = (
        jnp.asarray(0.0),
        initial_distance,
        jnp.asarray(0.0),
        initial_abs_distance,
        jnp.asarray(0, dtype=jnp.int32),
    )

    def keep_marching(state: tuple[Array, Array, Array, Array, Array]) -> Array:
        distance, surface_distance, _, _, steps = state
        return (
            (steps < max_steps)
            & (distance < max_distance)
            & jnp.isfinite(surface_distance)
            & (jnp.abs(surface_distance) > hit_epsilon)
        )

    def march(
        state: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        distance, surface_distance, closest_distance, closest_surface_distance, steps = state
        advance = jnp.maximum(jnp.abs(surface_distance) * step_scale, _MIN_MARCH_STEP)
        next_distance = jnp.minimum(distance + advance, max_distance)
        next_surface_distance = sdf(origin + next_distance * direction)
        next_abs_distance = jnp.where(
            jnp.isfinite(next_surface_distance),
            jnp.abs(next_surface_distance),
            jnp.asarray(jnp.inf),
        )
        closer = next_abs_distance < closest_surface_distance
        return (
            next_distance,
            next_surface_distance,
            jnp.where(closer, next_distance, closest_distance),
            jnp.minimum(next_abs_distance, closest_surface_distance),
            steps + 1,
        )

    distance, surface_distance, closest_distance, closest_surface_distance, steps = (
        jax.lax.while_loop(keep_marching, march, initial)
    )
    hit = (
        jnp.isfinite(surface_distance)
        & (jnp.abs(surface_distance) <= hit_epsilon)
        & (distance < max_distance)
    )
    return TraceResult(
        distance,
        hit,
        steps,
        closest_distance,
        closest_surface_distance,
    )


def _refract(direction: Array, normal: Array, eta: Array) -> Array:
    """Apply Snell's law, falling back to reflection for total internal reflection."""
    cos_incident = -jnp.dot(direction, normal)
    sin_transmitted_sq = eta**2 * (1.0 - cos_incident**2)
    cos_transmitted = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin_transmitted_sq))
    refracted = eta * direction + (eta * cos_incident - cos_transmitted) * normal
    reflected = direction - 2.0 * jnp.dot(direction, normal) * normal
    return jnp.where(sin_transmitted_sq >= 1.0, reflected, refracted)


def _fresnel_schlick(cos_theta: Array, ior: Array) -> Array:
    """Return Schlick's dielectric Fresnel approximation in ``[0, 1]``."""
    base_reflectance = ((1.0 - ior) / (1.0 + ior)) ** 2
    angle = 1.0 - jnp.clip(cos_theta, 0.0, 1.0)
    return base_reflectance + (1.0 - base_reflectance) * angle**5


def _trace_through_glass(
    sdf: Callable[[Array], Array],
    entry_position: Array,
    ray_direction: Array,
    entry_normal: Array,
    ior: Array,
    max_steps: int,
    max_distance: float,
    hit_epsilon: float,
    step_scale: float,
    normal_epsilon: float,
) -> tuple[Array, Array, Array, Array]:
    """Trace a ray from a glass entry surface to its exit surface."""
    from cadjoint.render.raymarch.shade import _compute_normal

    inside_direction = _refract(ray_direction, entry_normal, 1.0 / ior)
    inside_origin = _offset_surface(entry_position, -entry_normal, hit_epsilon)
    exit_trace = _sphere_trace(
        lambda point: -sdf(point),
        inside_origin,
        inside_direction,
        max_steps,
        max_distance,
        hit_epsilon,
        step_scale,
    )
    exit_position = inside_origin + exit_trace.distance * inside_direction
    exit_normal = _compute_normal(sdf, exit_position, normal_epsilon)
    outside_direction = _refract(inside_direction, -exit_normal, ior)
    return exit_position, outside_direction, exit_normal, exit_trace.hit
