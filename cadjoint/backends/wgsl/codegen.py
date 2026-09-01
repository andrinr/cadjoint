"""Public API: compile JAX SDF functions to WGSL."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from ._wgsl_emitter import StableHLOToWGSL

MATERIAL_BASE_ENTRY_POINT = "material_base"
MATERIAL_OPTICS_ENTRY_POINT = "material_optics"


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


def compile_scene_to_wgsl(
    geometry,
    example_point: jnp.ndarray | None = None,
) -> str:
    """Compile an SDF and its material evaluation to one WGSL module.

    The returned source has three stable point-query functions:

    - ``sdf(p) -> f32``
    - ``material_base(p) -> vec4<f32>`` containing RGB and roughness
    - ``material_optics(p) -> vec4<f32>`` containing metallic, opacity, IOR,
      and reflectivity

    All functions are built from the same functionalized geometry snapshot, so
    CSG material selection and transformed material coordinates match the
    distance field. Generated helper functions are namespaced per entry point,
    making the returned source safe to embed as one WGSL module.

    Args:
        geometry: Root CADJOINT SDF node with optional per-primitive materials.
        example_point: Example input for shape inference. Defaults to zeros.

    Returns:
        WGSL source containing the distance and two packed material functions.
    """
    from cadjoint.extraction import extract_parameters
    from cadjoint.functionalize import functionalize_scene

    if example_point is None:
        example_point = jnp.zeros(3, dtype=jnp.float32)

    free_parameters, fixed_parameters, _ = extract_parameters(geometry)
    sdf, material_at = functionalize_scene(geometry)(free_parameters, fixed_parameters)

    def material_base(point):
        material = material_at(point)
        color = jnp.reshape(jnp.asarray(material["color"], dtype=jnp.float32), (3,))
        roughness = jnp.reshape(
            jnp.asarray(material["roughness"], dtype=jnp.float32),
            (1,),
        )
        return jnp.concatenate((color, roughness))

    def material_optics(point):
        material = material_at(point)
        return jnp.concatenate(
            tuple(
                jnp.reshape(jnp.asarray(material[key], dtype=jnp.float32), (1,))
                for key in ("metallic", "opacity", "ior", "reflectivity")
            )
        )

    compiler = StableHLOToWGSL()
    sections = (
        compiler.compile(sdf, example_point),
        compiler.compile(
            material_base,
            example_point,
            entry_point=MATERIAL_BASE_ENTRY_POINT,
            output_shape=(4,),
            output_description="float32 vector with shape (4,)",
        ),
        compiler.compile(
            material_optics,
            example_point,
            entry_point=MATERIAL_OPTICS_ENTRY_POINT,
            output_shape=(4,),
            output_description="float32 vector with shape (4,)",
        ),
    )
    return "\n\n".join(sections)
