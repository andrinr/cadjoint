"""WGSLBackend: compile JAX SDF functions to WGSL."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from ..base import ShaderBackend
from .codegen import compile_sdf_to_wgsl


class WGSLBackend(ShaderBackend):
    """Compiles JAX SDF functions to WebGPU Shading Language (WGSL).

    Compilation is CPU-only (no GPU required). The output can be used with
    :class:`jaxcad.viewer.SDFViewer` for interactive in-notebook rendering,
    or exported to any WebGPU application.
    """

    @property
    def name(self) -> str:
        return "wgsl"

    def compile_sdf(
        self,
        fn: Callable,
        example_point: jnp.ndarray | None = None,
    ) -> str:
        return compile_sdf_to_wgsl(fn, example_point)
