"""GLSLBackend: compile + optionally render JAX SDF functions via OpenGL."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

from ..base import ShaderBackend
from .codegen import compile_sdf_to_glsl


class GLSLBackend(ShaderBackend):
    """Compiles JAX SDF functions to GLSL and renders them with OpenGL.

    Compilation uses ``jax.export`` → StableHLO → GLSL and requires no GPU.
    Rendering is optional and requires ``moderngl`` (``pip install cadjoint[glsl]``).

    Example::

        backend = GLSLBackend()
        glsl_code = backend.compile_sdf(Sphere(1.0))
        print(glsl_code)  # readable GLSL

        img = backend.render(glsl_code,
                             camera_pos=np.array([3., 3., 3.]),
                             camera_target=np.zeros(3),
                             resolution=(256, 256))
    """

    @property
    def name(self) -> str:
        return "glsl"

    def compile_sdf(
        self,
        fn: Callable,
        example_point: jnp.ndarray | None = None,
    ) -> str:
        return compile_sdf_to_glsl(fn, example_point)

    def render(
        self,
        sdf_code: str,
        camera_pos: np.ndarray,
        camera_target: np.ndarray,
        resolution: tuple[int, int],
        light_dir: np.ndarray | None = None,
        bg_color: np.ndarray | None = None,
        max_steps: int = 64,
        max_dist: float = 100.0,
        surf_eps: float = 1e-3,
    ) -> np.ndarray:
        """Render *sdf_code* (output of :meth:`compile_sdf`) to an RGB image.

        Wraps the SDF in a full raymarching fragment shader and renders it
        offscreen via moderngl.

        Args:
            sdf_code: GLSL SDF source returned by :meth:`compile_sdf`.
            camera_pos: Camera position, shape ``(3,)``.
            camera_target: Camera look-at point, shape ``(3,)``.
            resolution: Output ``(height, width)``.
            light_dir: World-space light direction, defaults to ``[1, 2, 3]``.
            bg_color: Background RGB, defaults to black.
            max_steps: Sphere-tracing iteration limit.
            max_dist: Ray miss distance.
            surf_eps: Hit threshold.

        Returns:
            ``float32`` numpy array ``(H, W, 3)`` in ``[0, 1]``.
        """
        from .codegen import _build_fragment_shader_from_code
        from .renderer import GLSLRenderer

        fragment_shader = _build_fragment_shader_from_code(
            sdf_code, max_steps=max_steps, max_dist=max_dist, surf_eps=surf_eps
        )
        with GLSLRenderer() as renderer:
            return renderer.render(
                fragment_shader,
                camera_pos=np.asarray(camera_pos),
                camera_target=np.asarray(camera_target),
                resolution=resolution,
                light_dir=light_dir,
                bg_color=bg_color,
            )
