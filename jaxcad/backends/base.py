"""Abstract base class for shader-based SDF backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp
import numpy as np


class ShaderBackend(ABC):
    """Compile JAX SDF functions to GPU shader code and optionally render them.

    Each backend targets a specific shading language (GLSL, WGSL, HLSL, …).
    The minimal contract is :meth:`compile_sdf`; :meth:`render` is optional
    and raises ``NotImplementedError`` for backends that only emit source code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short backend identifier, e.g. ``'glsl'``, ``'wgsl'``."""

    @abstractmethod
    def compile_sdf(
        self,
        fn: Callable,
        example_point: jnp.ndarray | None = None,
    ) -> str:
        """Compile a JAX SDF function to backend shader source.

        The function is traced via ``jax.make_jaxpr`` and every JAX primitive
        is mapped to an equivalent shader expression.

        Args:
            fn: Callable ``(p: f32[3]) -> f32[]`` traceable by JAX.
            example_point: Example input used for shape inference.
                Defaults to ``jnp.zeros(3)``.

        Returns:
            Shader-language source string for the SDF function.
        """

    def render(
        self,
        sdf_code: str,
        camera_pos: np.ndarray,
        camera_target: np.ndarray,
        resolution: tuple[int, int],
        **kwargs,
    ) -> np.ndarray:
        """Render a scene from compiled SDF shader code.

        Args:
            sdf_code: SDF function source returned by :meth:`compile_sdf`.
            camera_pos: Camera position, shape ``(3,)``.
            camera_target: Camera look-at point, shape ``(3,)``.
            resolution: Output ``(height, width)``.

        Returns:
            ``float32`` array of shape ``(H, W, 3)`` in ``[0, 1]``.

        Raises:
            NotImplementedError: If this backend does not support rendering.
        """
        raise NotImplementedError(
            f"The '{self.name}' backend does not implement render(). "
            f"Use a backend that wraps a GPU context (e.g. GLSLBackend)."
        )
