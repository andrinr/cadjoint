"""Pixel-level render pipeline and public rendering API."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from jaxcad.render.raymarch._constants import (
    _GLASS_SURFACE_OFFSET,
    _HIT_THRESHOLD,
    _SECONDARY_RAY_OFFSET,
)
from jaxcad.render.raymarch.camera import _camera_rays
from jaxcad.render.raymarch.shade import _compute_normal, _shade_surface
from jaxcad.render.raymarch.trace import (
    TraceMode,
    _fresnel_schlick,
    _trace,
    _trace_through_glass,
)


def _render_pixel(
    sdf: Callable[[Array], Array],
    material_fn: Callable[[Array], dict],
    ray_origin: Array,
    ray_dir: Array,
    light_dirs: Array,
    light_colors: Array,
    max_steps: int,
    max_dist: float,
    shadow_steps: int,
    shadow_hardness: float,
    ambient: float,
    edge_width: float,
    background_color: Array,
    refract_steps: int,
    use_grad_ao: bool = True,
    fd_normals: bool = False,
    normal_eps: float = 1e-4,
    reflect_steps: int = 0,
    env_fn: Callable[[Array], Array] | None = None,
    trace_mode: TraceMode = "sphere",
    bisect_steps: int = 8,
) -> Array:
    """Trace one ray and return its shaded RGB color.

    Pipeline:
      1. Sphere trace to find surface hit (closest-approach tracking).
      2. Compute surface normal via autodiff gradient (or finite differences).
      3. Query material at hit point.
      4. For each light: cast soft shadow, compute Blinn-Phong diffuse + specular,
         weight by light color and material properties, accumulate into RGB.
      5. If reflect_steps > 0: trace reflected ray, blend by ``reflectivity``.
      6. If refract_steps > 0: bend ray through material using Snell's law,
         march interior, exit, shade background, blend via Fresnel + opacity.
      7. Apply smooth edge coverage blending with background.

    Args:
        sdf: Signed-distance function mapping a 3-D point to a scalar distance.
        material_fn: Maps a 3-D surface point to a material property dict
            (keys: ``color``, ``roughness``, ``metallic``, ``opacity``,
            ``ior``, ``reflectivity``).
        ray_origin: Ray origin in world space, shape ``(3,)``.
        ray_dir: Unit ray direction, shape ``(3,)``.
        light_dirs: Unit vectors toward each light source, shape ``(L, 3)``.
        light_colors: RGB intensity of each light, shape ``(L, 3)``.
        max_steps: Sphere-tracing iterations for the primary ray.
        max_dist: Distance threshold beyond which a ray is considered a miss.
        shadow_steps: Sphere-tracing iterations for shadow rays.
        shadow_hardness: Controls shadow softness; higher values give harder edges.
        ambient: Minimum light level applied regardless of occlusion.
        edge_width: SDF distance threshold for smooth silhouette blending.
        background_color: Fallback RGB color for rays that miss all geometry.
            Ignored when ``env_fn`` is supplied.
        refract_steps: Sphere-tracing iterations inside the medium (0 disables).
        use_grad_ao: If ``True``, use gradient-magnitude as a cheap AO proxy.
        fd_normals: If ``True``, compute normals via finite differences.
        normal_eps: Step size used for finite-difference normal estimation.
        reflect_steps: Sphere-tracing iterations for the reflected ray
            (0 disables reflections).
        env_fn: Optional callable ``(direction: Array[3]) -> Array[3]`` that
            returns an environment-map RGB sample for any ray direction.
            When provided it replaces ``background_color`` for all misses,
            giving direction-dependent backgrounds and correct env reflections.

    Returns:
        Shaded RGB color for this pixel, shape ``(3,)``.
    """

    def _bg(d: Array) -> Array:
        """Return environment / background color for direction d."""
        if env_fn is not None:
            return env_fn(d)
        return background_color

    # Primary trace
    t_hit, d_min = _trace(sdf, ray_origin, ray_dir, max_steps, trace_mode, bisect_steps)
    pos = ray_origin + t_hit * ray_dir

    # Normal + optional gradient-magnitude AO proxy
    normal, normal_magnitude = _compute_normal(sdf, pos, fd_normals, normal_eps)
    ao = jnp.clip(normal_magnitude, 0.0, 1.0) if use_grad_ao else jnp.array(1.0)

    # Material at hit point
    mat = material_fn(pos)

    # Surface shading (opacity/Fresnel not yet applied)
    rgb_surface = _shade_surface(
        sdf,
        mat,
        pos,
        normal,
        ray_dir,
        ao,
        light_dirs,
        light_colors,
        shadow_steps,
        shadow_hardness,
        ambient,
    )

    hit = d_min < _HIT_THRESHOLD

    def _reflect() -> Array:
        reflect_dir = ray_dir - 2.0 * jnp.dot(ray_dir, normal) * normal
        # Offset avoids self-intersection: self-surface d_min stays ≥ _SECONDARY_RAY_OFFSET,
        # while a real hit converges to d_min < _HIT_THRESHOLD.
        reflect_origin = pos + _SECONDARY_RAY_OFFSET * normal
        t_refl, d_min_refl = _trace(
            sdf, reflect_origin, reflect_dir, reflect_steps, trace_mode, bisect_steps
        )
        refl_pos = reflect_origin + t_refl * reflect_dir
        refl_hit = d_min_refl < _HIT_THRESHOLD
        refl_norm, refl_mag = _compute_normal(sdf, refl_pos, fd_normals, normal_eps)
        refl_ao = jnp.clip(refl_mag, 0.0, 1.0) if use_grad_ao else jnp.array(1.0)
        refl_mat = material_fn(refl_pos)
        rgb_reflected = jnp.where(
            refl_hit,
            _shade_surface(
                sdf,
                refl_mat,
                refl_pos,
                refl_norm,
                reflect_dir,
                refl_ao,
                light_dirs,
                light_colors,
                shadow_steps,
                shadow_hardness,
                ambient,
            ),
            _bg(reflect_dir),
        )
        reflectivity = mat["reflectivity"]
        return rgb_surface * (1.0 - reflectivity) + rgb_reflected * reflectivity

    def _refract() -> Array:
        opacity = mat["opacity"]
        ior = mat["ior"]
        cos_entry_angle = jnp.maximum(0.0, jnp.dot(-ray_dir, normal))
        # Approximate reflection vs. refraction ratio
        fresnel = _fresnel_schlick(cos_entry_angle, ior)
        # Trace through glass: entry refraction → interior march → exit refraction
        exit_pos, dir_out = _trace_through_glass(
            sdf,
            pos,
            ray_dir,
            normal,
            ior,
            refract_steps,
            fd_normals,
            normal_eps,
            trace_mode,
            bisect_steps,
        )
        # March scene behind the glass
        bg_origin = exit_pos + _GLASS_SURFACE_OFFSET * dir_out
        t_bg, d_min_bg = _trace(sdf, bg_origin, dir_out, refract_steps, trace_mode, bisect_steps)
        bg_pos = bg_origin + t_bg * dir_out
        bg_hit = d_min_bg < _HIT_THRESHOLD
        bg_norm, bg_mag = _compute_normal(sdf, bg_pos, fd_normals, normal_eps)
        bg_ao = jnp.clip(bg_mag, 0.0, 1.0)
        bg_mat = material_fn(bg_pos)
        rgb_behind = jnp.where(
            bg_hit,
            _shade_surface(
                sdf,
                bg_mat,
                bg_pos,
                bg_norm,
                dir_out,
                bg_ao,
                light_dirs,
                light_colors,
                shadow_steps,
                shadow_hardness,
                ambient,
            ),
            _bg(dir_out),
        )
        # Tint transmitted light by glass colour
        rgb_transmitted = rgb_behind * mat["color"]
        # Blend:
        #   opacity=1          → fully opaque (surface only)
        #   opacity=0, head-on → mostly transmitted, small Fresnel highlight
        #   opacity=0, grazing → mostly Fresnel (TIR / strong reflection)
        return (opacity + (1.0 - opacity) * fresnel) * rgb_surface + (1.0 - fresnel) * (
            1.0 - opacity
        ) * rgb_transmitted

    if reflect_steps > 0:
        rgb_surface = jnp.where(hit, _reflect(), rgb_surface)

    if refract_steps > 0:
        rgb = jnp.where(hit, _refract(), _bg(ray_dir))
    else:
        # Opacity blends surface with background (no refraction)
        opacity = mat["opacity"]
        rgb = rgb_surface * opacity + _bg(ray_dir) * (1.0 - opacity)

    # Smooth edge: anti-alias contour by fading to background
    coverage = jnp.clip(1.0 - d_min / edge_width, 0.0, 1.0)
    hit_rgb = rgb * coverage + _bg(ray_dir) * (1.0 - coverage)
    return jnp.where(t_hit < max_dist, hit_rgb, _bg(ray_dir))


def _render_image(
    sdf: Callable[[Array], Array],
    material_fn: Callable[[Array], dict],
    camera_pos: Array,
    rays: Array,
    light_dirs: Array,
    light_colors: Array,
    background_color: Array,
    edge_width: float,
    max_steps: int,
    max_dist: float,
    shadow_steps: int,
    shadow_hardness: float,
    ambient: float,
    refract_steps: int,
    use_grad_ao: bool,
    fd_normals: bool,
    normal_eps: float,
    reflect_steps: int = 0,
    env_map: Array | None = None,
    trace_mode: TraceMode = "sphere",
    bisect_steps: int = 8,
) -> Array:
    """Core render loop: vmap ``_render_pixel`` over pre-computed camera rays.

    Returns a **linear** (pre-gamma) flat array of shape ``(N, 3)`` where
    ``N = H * W``.  Callers reshape, downsample (for AA), and apply gamma.
    """
    if env_map is not None:
        from jaxcad.render.raymarch.env import _sample_env_map

        def _env_fn(d: Array) -> Array:
            return _sample_env_map(env_map, d)
    else:
        _env_fn = None

    return jax.vmap(
        lambda ray_dir: _render_pixel(
            sdf,
            material_fn,
            camera_pos,
            ray_dir,
            light_dirs,
            light_colors,
            max_steps,
            max_dist,
            shadow_steps,
            shadow_hardness,
            ambient,
            edge_width,
            background_color,
            refract_steps,
            use_grad_ao,
            fd_normals,
            normal_eps,
            reflect_steps,
            _env_fn,
            trace_mode,
            bisect_steps,
        )
    )(rays)


def raymarch(
    sdf: Callable[[Array], Array],
    camera_pos: Array = jnp.array([5.0, 5.0, 5.0]),
    look_at: Array = jnp.array([0.0, 0.0, 0.0]),
    light_dirs: Array = jnp.array([0.5, 1.0, 0.3]),
    light_colors: Array | None = None,
    resolution: tuple[int, int] = (200, 200),
    fov: float = 0.6,
    max_steps: int = 64,
    max_dist: float = 20.0,
    shadow_steps: int = 24,
    shadow_hardness: float = 8.0,
    gamma: float = 2.2,
    ambient: float = 0.0,
    aa_samples: int = 1,
    background_color: Array = jnp.array([0.0, 0.0, 0.0]),
    refract_steps: int = 0,
    fd_normals: bool = False,
    normal_eps: float = 1e-4,
    reflect_steps: int = 0,
    env_map: Array | None = None,
    trace_mode: TraceMode = "sphere",
    bisect_steps: int = 8,
) -> np.ndarray:
    """Render an SDF via sphere tracing and return an RGB image array.

    Uses ``jax.vmap`` over pixels and ``jax.lax.scan`` for the inner march
    loop.  Supports multiple colored lights, transparency, refraction, and
    mirror reflections.

    Args:
        sdf: Signed distance function, callable ``(point: Array[3]) → Array[]``.
        camera_pos: Camera position in world space.
        look_at: Point the camera looks toward.
        light_dirs: Light direction(s) toward the light source(s).  Shape
            ``(3,)`` for a single light or ``(N, 3)`` for N lights.
        light_colors: RGB color(s) of the light(s), shape ``(3,)`` or
            ``(N, 3)``.  Defaults to white ``[1, 1, 1]`` for every light.
        resolution: Output image size as (height, width).
        fov: Half-width field-of-view parameter.
        max_steps: Sphere tracing iterations per ray.
        max_dist: Rays beyond this distance are treated as misses.
        shadow_steps: Iterations for the soft shadow ray.
        shadow_hardness: Shadow sharpness (higher = harder edges).
        gamma: Gamma correction exponent (1.5 default, 2.2 = sRGB standard).
        ambient: Constant ambient light added to all hit pixels (0 = fully
            black shadows, higher = lifted shadows).
        aa_samples: Super-sampling factor for antialiasing.  1 = no AA;
            2 = 2×2 SSAA (4× rays, box-filtered); 3 = 3×3, etc.
        background_color: RGB color returned for rays that miss all geometry.
            Ignored when ``env_map`` is provided.  Default is black.
        refract_steps: Number of interior march steps for glass refraction.
            0 disables refraction (legacy behaviour).  Try 32–64 for glass.
        fd_normals: Use central finite differences for surface normals instead
            of ``jax.grad``.  Eliminates 2nd-order AD overhead when rendering
            inside ``jax.grad(loss_fn)``.  Default False (AD normals).
        normal_eps: Step size for finite-difference normal estimation.
        reflect_steps: Sphere-tracing iterations for mirror reflections
            (0 disables).  Uses material ``reflectivity`` to blend.  Try 32.
        env_map: Optional equirectangular HDR environment map, shape
            ``(H, W, 3)``.  When set, miss rays and reflected rays that miss
            geometry sample this map instead of ``background_color``.  Load any
            ``.hdr`` or ``.exr`` file with ``imageio`` and pass through
            ``jnp.asarray``, or generate one with
            :func:`~jaxcad.render.raymarch.env.make_gradient_sky`.

    Returns:
        Float32 numpy array of shape (H, W, 3) with values in [0, 1].
    """
    h, w = resolution

    # Normalise light directions — accept (3,) for a single light
    light_dirs = jnp.atleast_2d(jnp.asarray(light_dirs, dtype=jnp.float32))  # (N, 3)
    light_dirs = light_dirs / jnp.linalg.norm(light_dirs, axis=1, keepdims=True)

    if light_colors is None:
        light_colors = jnp.ones_like(light_dirs)  # white for every light
    else:
        light_colors = jnp.atleast_2d(jnp.asarray(light_colors, dtype=jnp.float32))

    background_color = jnp.asarray(background_color, dtype=jnp.float32)

    # Super-sample: render at N× resolution, box-filter down after shading
    render_res = (h * aa_samples, w * aa_samples)
    rays = _camera_rays(camera_pos, look_at, render_res, fov)

    # Edge width scaled to the super-sampled pixel footprint
    scene_dist = float(jnp.linalg.norm(camera_pos - look_at))
    edge_width = 2.0 * fov / min(h * aa_samples, w * aa_samples) * scene_dist

    if hasattr(sdf, "material_at"):
        material_fn = sdf.material_at
    else:
        from jaxcad.render.material import Material

        def material_fn(_p):
            return Material().as_dict()

    # For approximate SDFs (is_exact=False, e.g. Twist) the gradient magnitude
    # is not a reliable AO proxy — use ao=1 (unoccluded) instead.
    use_grad_ao = getattr(sdf, "is_exact", True)

    # Capture all non-array config in a closure so jit only traces array inputs.
    pixels = jax.jit(
        lambda r: _render_image(
            sdf,
            material_fn,
            camera_pos,
            r,
            light_dirs,
            light_colors,
            background_color,
            edge_width,
            max_steps,
            max_dist,
            shadow_steps,
            shadow_hardness,
            ambient,
            refract_steps,
            use_grad_ao,
            fd_normals,
            normal_eps,
            reflect_steps,
            env_map,
            trace_mode,
            bisect_steps,
        )
    )(rays)

    rh, rw = render_res
    image = pixels.reshape(rh, rw, 3)

    # Box-filter downsample in linear space before gamma correction
    if aa_samples > 1:
        image = image.reshape(h, aa_samples, w, aa_samples, 3).mean(axis=(1, 3))

    image = jnp.clip(image ** (1.0 / gamma), 0.0, 1.0)
    return np.array(image)


def render_raymarched(
    sdf: Callable[[Array], Array],
    camera_pos: Array = jnp.array([5.0, 5.0, 5.0]),
    look_at: Array = jnp.array([0.0, 0.0, 0.0]),
    light_dirs: Array = jnp.array([0.5, 1.0, 0.3]),
    light_colors: Array | None = None,
    resolution: tuple[int, int] = (200, 200),
    fov: float = 0.6,
    max_steps: int = 48,
    max_dist: float = 20.0,
    shadow_steps: int = 32,
    shadow_hardness: float = 8.0,
    gamma: float = 2.2,
    ambient: float = 0.0,
    aa_samples: int = 1,
    background_color: Array = jnp.array([0.0, 0.0, 0.0]),
    refract_steps: int = 0,
    fd_normals: bool = False,
    normal_eps: float = 1e-4,
    reflect_steps: int = 0,
    env_map: Array | None = None,
    trace_mode: TraceMode = "sphere",
    bisect_steps: int = 8,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Render an SDF via sphere tracing and display with matplotlib.

    Wraps :func:`raymarch` and shows the result on a ``plt.Axes``.

    Args:
        sdf: Signed distance function.
        camera_pos: Camera position in world space.
        look_at: Point the camera looks toward.
        light_dirs: Light direction(s), shape ``(3,)`` or ``(N, 3)``.
        light_colors: Light color(s), shape ``(3,)`` or ``(N, 3)``.
            Defaults to white for every light.
        resolution: Output image size as (height, width).
        fov: Half-width field-of-view parameter.
        max_steps: Sphere tracing iterations per ray.
        max_dist: Miss threshold distance.
        shadow_steps: Soft shadow ray iterations.
        shadow_hardness: Shadow edge sharpness.
        gamma: Gamma correction exponent.
        ambient: Constant ambient term (0 = fully black shadows).
        aa_samples: Super-sampling anti-aliasing factor.
        background_color: RGB color for rays that miss all geometry.
        refract_steps: Interior march steps for glass refraction (0 = disabled).
        fd_normals: Use finite-difference normals (avoids 2nd-order AD overhead).
        normal_eps: Step size for finite-difference normal estimation.
        reflect_steps: Mirror reflection march steps (0 = disabled).
        env_map: Equirectangular HDR environment map ``(H, W, 3)`` used for
            miss-ray and reflection-miss coloring.
        ax: Existing matplotlib axes; creates new figure if None.
        title: Axes title.

    Returns:
        The matplotlib axes with the rendered image.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    image = raymarch(
        sdf,
        camera_pos=camera_pos,
        look_at=look_at,
        light_dirs=light_dirs,
        light_colors=light_colors,
        resolution=resolution,
        fov=fov,
        max_steps=max_steps,
        max_dist=max_dist,
        shadow_steps=shadow_steps,
        shadow_hardness=shadow_hardness,
        gamma=gamma,
        ambient=ambient,
        aa_samples=aa_samples,
        background_color=background_color,
        refract_steps=refract_steps,
        fd_normals=fd_normals,
        normal_eps=normal_eps,
        reflect_steps=reflect_steps,
        env_map=env_map,
        trace_mode=trace_mode,
        bisect_steps=bisect_steps,
    )

    ax.imshow(image, vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(title or "Raymarched Render", fontsize=12)
    return ax
