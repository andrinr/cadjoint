"""Compile a Scene to a differentiable render function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxcad.render.scene import Scene


def functionalize_render(
    scene: Scene,
    max_steps: int = 32,
    max_dist: float = 15.0,
    shadow_steps: int = 12,
    shadow_hardness: float = 6.0,
    gamma: float = 2.2,
    fd_normals: bool = False,
    normal_eps: float = 1e-4,
    reflect_steps: int = 0,
) -> Callable:
    """Compile a ``Scene`` to a differentiable render function.

    Fixed geometry params are extracted once at call time and baked in, so the
    returned function only needs ``free_params`` — the dict that changes each
    optimisation step::

        render_fn = functionalize_render(scene)
        image     = render_fn(free_params, resolution=(64, 96))

    ``image`` is a JAX ``(H, W, 3)`` float32 array fully differentiable
    w.r.t. ``free_params`` via ``jax.grad``.

    Args:
        scene: ``Scene`` with geometry, camera, and lighting.
        max_steps: Sphere-tracing iterations per primary ray.
        max_dist: Miss threshold distance.
        shadow_steps: Soft shadow ray iterations.
        shadow_hardness: Shadow edge sharpness.
        gamma: Gamma correction exponent applied to the final image.
        fd_normals: Use central finite differences for surface normals instead
            of ``jax.grad``.  Set to True when calling inside
            ``jax.grad(loss_fn)`` to avoid 2nd-order AD overhead.
        normal_eps: Step size for finite-difference normal estimation.

    Returns:
        ``(free_params, resolution=(H, W)) -> JAX image (H, W, 3)``
    """
    from jaxcad.extraction import extract_parameters
    from jaxcad.functionalize import _resolve, functionalize_scene
    from jaxcad.render.raymarch import _camera_rays, _render_image

    _, geo_fixed, _ = extract_parameters(scene.geometry)
    scene_fn = functionalize_scene(scene.geometry)

    @jax.jit(static_argnames=["resolution"])
    def render_fn(free_params: dict, resolution: tuple[int, int] = (64, 96)):
        sdf, material_fn = scene_fn(free_params, geo_fixed)

        camera_pos = _resolve(scene.camera.params["camera_pos"], free_params)
        look_at = _resolve(scene.camera.params["look_at"], free_params)
        fov = _resolve(scene.camera.params["fov"], free_params)
        bg = jax.nn.sigmoid(_resolve(scene.params["bg_color"], free_params))

        # Stop gradient through scene_dist: edge_width is a rendering-quality
        # scalar, not an optimisation target, so it shouldn't pull camera params.
        scene_dist = jax.lax.stop_gradient(jnp.linalg.norm(camera_pos - look_at))
        h, w = resolution
        edge_width = 2.0 * fov / min(h, w) * scene_dist

        if scene.free_lights:
            light_dirs = jnp.stack(
                [
                    _resolve(scene.params[f"light_dir_{i}"], free_params)
                    for i in range(scene.n_lights)
                ]
            )
            norms = jnp.sqrt(jnp.sum(light_dirs**2, axis=1, keepdims=True) + 1e-12)
            light_dirs = light_dirs / norms
            light_colors = jnp.stack(
                [
                    _resolve(scene.params[f"light_color_{i}"], free_params)
                    for i in range(scene.n_lights)
                ]
            )
        else:
            light_dirs = scene.light_dirs
            light_colors = scene.light_colors

        rays = _camera_rays(camera_pos, look_at, resolution, fov)
        pixels = _render_image(
            sdf,
            material_fn,
            camera_pos,
            rays,
            light_dirs,
            light_colors,
            bg,
            edge_width,
            max_steps,
            max_dist,
            shadow_steps,
            shadow_hardness,
            ambient=0.0,  # ambient handled via bg fallback
            refract_steps=0,
            use_grad_ao=True,
            fd_normals=fd_normals,
            normal_eps=normal_eps,
            reflect_steps=reflect_steps,
        )
        h, w = resolution
        image = pixels.reshape(h, w, 3)
        return jnp.clip(jnp.maximum(image, 0.0) ** (1.0 / gamma), 0.0, 1.0)

    return render_fn
