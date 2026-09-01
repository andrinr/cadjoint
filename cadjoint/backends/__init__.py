"""Shader code generation: compile JAX SDF functions to WGSL."""

from .wgsl import (
    MATERIAL_BASE_ENTRY_POINT,
    MATERIAL_OPTICS_ENTRY_POINT,
    compile_scene_to_wgsl,
    compile_sdf_to_wgsl,
)

__all__ = [
    "MATERIAL_BASE_ENTRY_POINT",
    "MATERIAL_OPTICS_ENTRY_POINT",
    "compile_scene_to_wgsl",
    "compile_sdf_to_wgsl",
]
