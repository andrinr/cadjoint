"""Public API: compile JAX SDF functions to WGSL."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from ._wgsl_emitter import StableHLOToWGSL


def compile_sdf_to_wgsl(
    fn: Callable,
    example_point: jnp.ndarray | None = None,
) -> str:
    """Compile a JAX SDF function to a WGSL function string.

    The emitted entry-point is always named ``sdf``.

    Args:
        fn: Callable ``(p: f32[3]) -> f32[]`` traceable by JAX.
        example_point: Example input for shape inference. Defaults to zeros.

    Returns:
        WGSL source for the SDF function(s) — no surrounding shader boilerplate.
    """
    if example_point is None:
        example_point = jnp.zeros(3, dtype=jnp.float32)
    return StableHLOToWGSL().compile(fn, example_point)
