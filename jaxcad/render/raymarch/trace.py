"""Sphere tracing, optics (Snell's law, Fresnel), and glass traversal."""

from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.render.raymarch._constants import (
    _GLASS_MIN_STEP,
    _GLASS_SURFACE_OFFSET,
    _MIN_MARCH_STEP,
    _SDF_INF,
)

TraceMode = Literal["sphere", "bisection_refinement"]


def _sphere_trace(
    sdf: Callable, origin: Array, direction: Array, steps: int
) -> tuple[Array, Array]:
    """Sphere trace from origin along direction for ``steps`` iterations.

    Advances along the ray by the SDF value at each step (sphere tracing /
    ray marching).  Tracks the parameter value ``t`` at the closest approach
    and the minimum SDF distance seen, which together let callers detect hits
    and compute surface positions.

    Args:
        sdf: Signed-distance function mapping a 3-D point to a scalar distance.
        origin: Ray origin, shape ``(3,)``.
        direction: Unit ray direction, shape ``(3,)``.
        steps: Number of marching steps (fixed, executed via ``jax.lax.scan``).

    Returns:
        ``(t_hit, d_min)`` where ``t_hit`` is the ray parameter at the closest
        approach and ``d_min`` is the minimum SDF value seen along the ray.
        A small ``d_min`` (near zero) indicates a surface hit.
    """

    def march(carry, _):
        t, t_hit, d_min = carry
        d = sdf(origin + t * direction)
        d_safe = jnp.where(jnp.isfinite(d), d, jnp.array(_SDF_INF))
        t_next = t + jnp.maximum(d_safe, _MIN_MARCH_STEP)
        closer = d_safe < d_min
        return (t_next, jnp.where(closer, t, t_hit), jnp.minimum(d_safe, d_min)), None

    (_, t_hit, d_min), _ = jax.lax.scan(
        march, (jnp.array(0.0), jnp.array(0.0), jnp.array(_SDF_INF)), None, length=steps
    )
    return t_hit, d_min


def _sphere_trace_with_bracket(
    sdf: Callable, origin: Array, direction: Array, steps: int
) -> tuple[Array, Array, Array, Array]:
    """Sphere trace that also records a sign-change bracket for bisection.

    Same march as :func:`_sphere_trace` but additionally tracks the last ray
    position where SDF was non-negative (``t_lo``) and the first position where
    SDF went negative (``t_hi``).  The pair ``[t_lo, t_hi]`` brackets the surface
    zero-crossing and can be refined with :func:`_bisection_refine`.

    Args:
        sdf: Signed-distance function.
        origin: Ray origin, shape ``(3,)``.
        direction: Unit ray direction, shape ``(3,)``.
        steps: Number of marching steps.

    Returns:
        ``(t_hit, d_min, t_lo, t_hi)`` where ``t_lo``/``t_hi`` bound the surface
        crossing.  If the ray never crosses (``d`` never goes negative),
        ``t_hi`` is ``_SDF_INF`` and the bracket is invalid.
    """

    def march(carry, _):
        t, t_last_pos, t_hit, d_min, t_lo, t_hi = carry
        d = sdf(origin + t * direction)
        d_safe = jnp.where(jnp.isfinite(d), d, jnp.array(_SDF_INF))
        t_next = t + jnp.maximum(d_safe, _MIN_MARCH_STEP)
        closer = d_safe < d_min
        t_hit_new = jnp.where(closer, t, t_hit)
        d_min_new = jnp.minimum(d_safe, d_min)
        t_last_pos_new = jnp.where(d_safe >= 0.0, t, t_last_pos)
        crossed_first = (d_safe < 0.0) & (t_hi >= _SDF_INF)
        t_lo_new = jnp.where(crossed_first, t_last_pos, t_lo)
        t_hi_new = jnp.where(crossed_first, t, t_hi)
        return (t_next, t_last_pos_new, t_hit_new, d_min_new, t_lo_new, t_hi_new), None

    init = (
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(_SDF_INF),
        jnp.array(0.0),
        jnp.array(_SDF_INF),
    )
    (_, _, t_hit, d_min, t_lo, t_hi), _ = jax.lax.scan(march, init, None, length=steps)
    return t_hit, d_min, t_lo, t_hi


def _bisection_refine(
    sdf: Callable, origin: Array, direction: Array, t_lo: Array, t_hi: Array, steps: int
) -> Array:
    """Bisection search for the SDF zero-crossing between ``t_lo`` and ``t_hi``.

    Assumes ``sdf(origin + t_lo * direction) >= 0`` and
    ``sdf(origin + t_hi * direction) < 0``.

    Args:
        sdf: Signed-distance function.
        origin: Ray origin, shape ``(3,)``.
        direction: Unit ray direction, shape ``(3,)``.
        t_lo: Lower bracket (outside surface).
        t_hi: Upper bracket (inside surface).
        steps: Number of bisection iterations; each halves the interval.

    Returns:
        Refined ``t`` at the surface zero-crossing.
    """

    def bisect(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        d = sdf(origin + mid * direction)
        lo_new = jnp.where(d > 0.0, mid, lo)
        hi_new = jnp.where(d > 0.0, hi, mid)
        return (lo_new, hi_new), None

    (lo, hi), _ = jax.lax.scan(bisect, (t_lo, t_hi), None, length=steps)
    return 0.5 * (lo + hi)


def _trace(
    sdf: Callable,
    origin: Array,
    direction: Array,
    steps: int,
    trace_mode: TraceMode = "sphere",
    bisect_steps: int = 8,
) -> tuple[Array, Array]:
    """Unified ray-surface intersection: sphere trace or sphere + bisection refinement.

    Args:
        sdf: Signed-distance function.
        origin: Ray origin, shape ``(3,)``.
        direction: Unit ray direction, shape ``(3,)``.
        steps: Sphere-tracing iterations.
        trace_mode: ``"sphere"`` for standard sphere tracing (default) or
            ``"bisection_refinement"`` to follow the coarse march with a
            bisection pass that pins the hit to the SDF zero-crossing.
        bisect_steps: Bisection iterations when ``trace_mode="bisection_refinement"``.
            Each step halves the bracket; 8 gives ~256× precision improvement.

    Returns:
        ``(t_hit, d_min)`` — same contract as :func:`_sphere_trace`.
    """
    if trace_mode == "bisection_refinement":
        t_hit, d_min, t_lo, t_hi = _sphere_trace_with_bracket(sdf, origin, direction, steps)
        bracket_found = t_hi < _SDF_INF

        # Exact SDFs never go negative during sphere tracing (no overshoot), so the
        # sign-change bracket above is rarely set.  Use a Newton step to establish
        # one: the directional derivative of the SDF along the ray gives the distance
        # to the zero crossing.  A small nudge past that puts us inside the surface.
        p_hit = origin + t_hit * direction
        grad = jax.grad(sdf)(p_hit)  # surface normal (unit vec for exact SDFs)
        proj = jnp.dot(grad, direction)  # < 0 for a ray approaching the surface
        safe_proj = jnp.where(proj < -1e-6, proj, jnp.array(-1e-6))
        t_over = t_hit + (-d_min / safe_proj) + 1e-4  # Newton step + tiny overshoot
        d_over = sdf(origin + t_over * direction)
        sign_change = (d_over < 0.0) & ~bracket_found & (d_min < 1.0)
        t_lo = jnp.where(sign_change, t_hit, t_lo)
        t_hi = jnp.where(sign_change, t_over, t_hi)
        bracket_found = bracket_found | sign_change

        # Clamp t_hi so bisection always operates in a finite range
        t_hi_safe = jnp.where(bracket_found, t_hi, t_lo + 1.0)
        t_refined = _bisection_refine(sdf, origin, direction, t_lo, t_hi_safe, bisect_steps)
        t_final = jnp.where(bracket_found, t_refined, t_hit)
        d_final = jnp.where(bracket_found, jnp.abs(sdf(origin + t_final * direction)), d_min)
        return t_final, d_final
    return _sphere_trace(sdf, origin, direction, steps)


def _refract(d: Array, n: Array, eta: Array) -> Array:
    """Snell's-law refraction; falls back to reflection on total internal reflection.

    Args:
        d: Incident ray direction (unit vector, pointing toward surface).
        n: Surface normal (unit vector, pointing against incident ray).
        eta: Ratio of incident IOR to transmitted IOR (n1/n2).

    Returns:
        Refracted (or reflected on TIR) ray direction.
    """
    cos_i = -jnp.dot(d, n)
    sin2_t = eta**2 * (1.0 - cos_i**2)
    cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin2_t))
    refracted = eta * d + (eta * cos_i - cos_t) * n
    reflected = d - 2.0 * jnp.dot(d, n) * n  # TIR fallback
    return jnp.where(sin2_t >= 1.0, reflected, refracted)


def _fresnel_schlick(cos_theta: Array, ior: Array) -> Array:
    """Schlick approximation for dielectric reflectance.

    Approximates how much light reflects vs. refracts at a dielectric boundary
    (e.g. air -> glass). Reflectance is minimized at normal incidence and
    approaches 1.0 at grazing angles.

    The baseline reflectance at normal incidence is:
        r0 = ((1 - ior) / (1 + ior))^2

    Which is then interpolated toward 1.0 with a (1 - cos_theta)^5 term to
    fit the real Fresnel curve:
        F = r0 + (1 - r0) * (1 - cos_theta)^5

    cos_theta is clamped to [0, 1] to handle back-facing geometry or
    numerical drift that could otherwise produce reflectance > 1.

    Args:
        cos_theta: Cosine of the angle between the ray and the surface normal.
        ior: Index of refraction of the medium being entered.

    Returns:
        Fresnel reflectance in [0, 1].
    """
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    return r0 + (1.0 - r0) * (1.0 - jnp.maximum(cos_theta, 0.0)) ** 5


def _trace_through_glass(
    sdf: Callable[[Array], Array],
    entry_pos: Array,
    ray_dir: Array,
    entry_normal: Array,
    ior: float,
    refract_steps: int,
    fd_normals: bool,
    normal_eps: float,
    trace_mode: TraceMode = "sphere",
    bisect_steps: int = 8,
) -> tuple[Array, Array]:
    """Trace a ray through a glass volume (entry refraction → interior march → exit refraction).

    Args:
        sdf: Signed-distance function of the scene.
        entry_pos: Surface hit point where the ray enters the medium, shape ``(3,)``.
        ray_dir: Incident ray direction (unit vector), shape ``(3,)``.
        entry_normal: Outward surface normal at *entry_pos*, shape ``(3,)``.
        ior: Index of refraction of the medium.
        refract_steps: Sphere-tracing iterations inside the medium.
        fd_normals: Use finite differences for the exit normal.
        normal_eps: Step size for finite-difference normal estimation.
        trace_mode: Ray-surface intersection mode (passed to :func:`_trace`).
        bisect_steps: Bisection refinement iterations (passed to :func:`_trace`).

    Returns:
        ``(exit_pos, dir_out)``: the exit point on the back face and the
        refracted ray direction leaving the medium into air.
    """
    from jaxcad.render.raymarch.shade import _compute_normal

    # Bend into material (air → glass)
    dir_in = _refract(ray_dir, entry_normal, 1.0 / ior)
    # March interior using -sdf (interior is where sdf < 0)
    t_exit, _ = _trace(
        lambda p: jnp.maximum(-sdf(p), _GLASS_MIN_STEP),
        entry_pos + _GLASS_SURFACE_OFFSET * dir_in,
        dir_in,
        refract_steps,
        trace_mode,
        bisect_steps,
    )
    exit_pos = entry_pos + _GLASS_SURFACE_OFFSET * dir_in + t_exit * dir_in
    # Exit normal — outward gradient points out of glass, so flip inward for Snell's law
    exit_norm_out, _ = _compute_normal(sdf, exit_pos, fd_normals, normal_eps)
    # Bend back into air (glass → air)
    dir_out = _refract(dir_in, -exit_norm_out, ior)
    return exit_pos, dir_out
