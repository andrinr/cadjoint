"""Surface normals, lighting, and Blinn-Phong shading."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.render.raymarch._constants import (
    _AO_WEIGHT,
    _DIFFUSE_WEIGHT,
    _MIN_MARCH_STEP,
    _NORMAL_MAG_EPS,
    _NORMAL_ZERO_THRESHOLD,
    _ROUGHNESS_EPS,
    _SECONDARY_RAY_OFFSET,
    _SHADOW_T_START,
    _SPECULAR_WEIGHT,
)
from jaxcad.render.raymarch.camera import _normalize


def _normal_fd(
    sdf: Callable[[Array], Array],
    pos: Array,
    eps: float = 1e-4,
) -> tuple[Array, Array]:
    """Tetrahedral finite-difference surface normal (IQ trick).

    Uses 4 SDF evaluations instead of the 6 required by central differences.
    Avoids nested backward-mode AD when called inside jax.grad(loss).
    Reference: https://iquilezles.org/articles/normalsSDF/
    """
    # Tetrahedral sample directions
    k0 = jnp.array([1.0, -1.0, -1.0])
    k1 = jnp.array([-1.0, -1.0, 1.0])
    k2 = jnp.array([-1.0, 1.0, -1.0])
    k3 = jnp.array([1.0, 1.0, 1.0])
    raw = (
        k0 * sdf(pos + eps * k0)
        + k1 * sdf(pos + eps * k1)
        + k2 * sdf(pos + eps * k2)
        + k3 * sdf(pos + eps * k3)
    )
    return raw, jnp.sqrt(jnp.sum(raw**2) + _NORMAL_MAG_EPS)


def _compute_normal(
    sdf: Callable[[Array], Array],
    pos: Array,
    fd_normals: bool,
    eps: float,
) -> tuple[Array, Array]:
    """Return ``(unit_normal, raw_magnitude)`` at *pos*.

    Args:
        sdf: Signed-distance function.
        pos: Surface point, shape ``(3,)``.
        fd_normals: Use finite differences instead of autodiff.
        eps: Step size for finite-difference estimation.

    Returns:
        ``(unit_normal, raw_mag)`` where ``unit_normal`` is the normalized
        outward surface normal and ``raw_mag`` is the gradient magnitude
        (useful as an AO proxy or for Eikonal-deviation checks).
    """
    if fd_normals:
        raw, mag = _normal_fd(sdf, pos, eps)
    else:
        raw = jax.grad(sdf)(pos)
        # sqrt(sum+eps) avoids the 0/||0|| NaN gradient of linalg.norm at zero.
        mag = jnp.sqrt(jnp.sum(raw**2) + _NORMAL_MAG_EPS)
    return raw / jnp.where(mag > _NORMAL_ZERO_THRESHOLD, mag, 1.0), mag


def _cast_shadow(
    sdf: Callable[[Array], Array],
    pos: Array,
    normal: Array,
    light_dir: Array,
    steps: int,
    hardness: float,
) -> Array:
    """Soft shadow via secondary sphere tracing toward the light.

    Uses the Inigo Quilez penumbra technique: the minimum ratio of SDF value
    to ray distance approximates how much light is blocked.

    The shadow ray origin is offset along the surface normal to avoid
    self-intersection with the surface the ray was cast from.

    Args:
        sdf: Signed distance function, callable (3,) → scalar.
        pos: Surface point to shadow-test, shape (3,).
        normal: Outward surface normal at pos, shape (3,).
        light_dir: Unit vector toward the light source, shape (3,).
        steps: Number of shadow-ray march steps.
        hardness: Controls shadow sharpness; higher values give harder edges.

    Returns:
        Shadow factor in [0, 1]: 0 = fully shadowed, 1 = fully lit.
    """
    # Offset along the normal so the shadow ray starts clearly outside the surface
    origin = pos + normal * _SECONDARY_RAY_OFFSET

    def f(carry, _):
        t, shadow = carry
        h = sdf(origin + light_dir * t)
        # Clamp step to positive to prevent the ray going backwards inside geometry
        return (t + jnp.maximum(h, _MIN_MARCH_STEP), jnp.clip(hardness * h / t, 0.0, shadow)), None

    (_, shadow), _ = jax.lax.scan(
        f, (jnp.array(_SHADOW_T_START), jnp.array(1.0)), None, length=steps
    )
    return shadow


def _shade_one_light(
    ldir: Array,
    lcolor: Array,
    sdf: Callable,
    pos: Array,
    normal: Array,
    ray_dir: Array,
    ao: Array,
    base_color: Array,
    specular_color: Array,
    shininess: Array,
    shadow_steps: int,
    shadow_hardness: float,
) -> Array:
    if shadow_steps == 0:
        shadow = 1.0
    else:
        shadow = _cast_shadow(sdf, pos, normal, ldir, shadow_steps, shadow_hardness)

    diffuse = jnp.clip(jnp.dot(normal, ldir), 0.0, 1.0) * shadow
    halfway = _normalize(ldir - ray_dir)
    # Safe power: gradient of 0^s = 0^s*log(0) = NaN; replace base=0 with 1 so
    # the gradient stays finite, then mask the result to 0 via jnp.where.
    spec_cos = jnp.clip(jnp.dot(halfway, normal), 0.0, 1.0)
    safe_cos = jnp.where(spec_cos > 0, spec_cos, jnp.ones_like(spec_cos))
    specular = jnp.where(spec_cos > 0, safe_cos**shininess, 0.0) * shadow
    return lcolor * (
        base_color * (_AO_WEIGHT * ao + _DIFFUSE_WEIGHT * diffuse)
        + specular_color * _SPECULAR_WEIGHT * specular
    )


def _shade_surface(
    sdf: Callable,
    mat: dict,
    pos: Array,
    normal: Array,
    ray_dir: Array,
    ao: Array,
    light_dirs: Array,
    light_colors: Array,
    shadow_steps: int,
    shadow_hardness: float,
    ambient: float,
) -> Array:
    """Blinn-Phong shading for a surface hit point.

    Args:
        sdf: Signed distance function used for soft shadow rays.
        mat: Material dict with keys "color" (RGB Array), "roughness"
            (scalar in [0, 1]), and "metallic" (scalar in [0, 1]).
        pos: Surface hit position in world space, shape (3,).
        normal: Outward unit surface normal at pos, shape (3,).
        ray_dir: Incident ray direction (pointing toward surface), shape (3,).
        ao: Ambient occlusion factor in [0, 1]; 1.0 = fully lit.
        light_dirs: Unit vectors toward each light source, shape (N, 3).
        light_colors: RGB color/intensity of each light, shape (N, 3).
        shadow_steps: Number of ray march steps for soft shadow evaluation.
        shadow_hardness: Controls shadow sharpness; higher = harder shadows.
        ambient: Scalar ambient light intensity added as base_color * ambient.

    Returns:
        RGB color array of shape (3,), unweighted by opacity or Fresnel.
    """
    base_color = mat["color"]
    roughness = mat["roughness"]
    metallic = mat["metallic"]
    shininess = jnp.maximum(2.0 / (roughness**2 + _ROUGHNESS_EPS) - 2.0, 1.0)
    specular_color = jnp.ones(3) * (1.0 - metallic) + base_color * metallic
    per_light = jax.vmap(
        lambda ld, lc: _shade_one_light(
            ld,
            lc,
            sdf,
            pos,
            normal,
            ray_dir,
            ao,
            base_color,
            specular_color,
            shininess,
            shadow_steps,
            shadow_hardness,
        )
    )(light_dirs, light_colors)
    return per_light.sum(0) + base_color * ambient
