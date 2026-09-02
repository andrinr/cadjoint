"""Public API: compile JAX SDF functions to WGSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from cadjoint.sdf._lowering import scalar_lowering

from ._wgsl_emitter import StableHLOToWGSL

MATERIAL_BASE_ENTRY_POINT = "material_base"
MATERIAL_OPTICS_ENTRY_POINT = "material_optics"

#: Bind group and binding the generated parameter uniform declares itself at.
DEFAULT_PARAMETER_GROUP = 3
DEFAULT_PARAMETER_BINDING = 0

#: Every parameter occupies one ``vec4<f32>`` slot, which is the only element
#: type a WGSL uniform array can carry without per-field alignment padding.
PARAMETER_SLOT_BYTES = 16

_COMPONENT_SWIZZLE = {1: "x", 2: "xy", 3: "xyz", 4: "xyzw"}


@dataclass(frozen=True)
class ShaderParameter:
    """One design parameter's place in the generated uniform buffer.

    Attributes:
        name: The parameter's name — a free parameter's declared name, or a
            fixed parameter's ``node.attribute`` path. The same names
            :func:`cadjoint.extract_parameters` returns.
        offset: Byte offset into the uniform buffer. Always a multiple of
            :data:`PARAMETER_SLOT_BYTES`.
        components: How many floats the parameter occupies (1-4); the rest of
            its 16-byte slot is padding.
        value: The parameter's current value, as a list of floats.
        free: Whether it is a free (named, optimizable) parameter.
    """

    name: str
    offset: int
    components: int
    value: tuple[float, ...]
    free: bool

    def as_dict(self) -> dict:
        """This entry as a plain JSON-serializable dict."""
        return {
            "name": self.name,
            "offset": self.offset,
            "components": self.components,
            "value": list(self.value),
            "free": self.free,
        }


@dataclass(frozen=True)
class ShaderProgram:
    """WGSL source plus the parameter buffer its entry points read.

    Attributes:
        wgsl: The shader source, identical for every value of every parameter.
        parameters: Buffer layout, in binding order.
        buffer_bytes: Total size of the uniform buffer to allocate.
        group: Bind group index the uniform declares.
        binding: Binding index within that group.
    """

    wgsl: str
    parameters: tuple[ShaderParameter, ...]
    buffer_bytes: int
    group: int
    binding: int

    def buffer(self) -> np.ndarray:
        """The current parameter values, packed for upload.

        Returns:
            A ``float32`` array of ``buffer_bytes // 4`` elements, with each
                parameter written at its own slot and the padding left zero.
        """
        packed = np.zeros(self.buffer_bytes // 4, dtype=np.float32)
        for parameter in self.parameters:
            start = parameter.offset // 4
            packed[start : start + parameter.components] = parameter.value
        return packed


class _RecordingParams(dict):
    """A parameter dict that remembers which keys a trace actually read."""

    def __init__(self, source: dict):
        super().__init__(source)
        self.read: list = []

    def __getitem__(self, key):
        if key not in self.read:
            self.read.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self and key not in self.read:
            self.read.append(key)
        return super().get(key, default)


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
    with scalar_lowering():
        return StableHLOToWGSL().compile(fn, example_point)


def compile_scene_to_wgsl(
    geometry,
    example_point: jnp.ndarray | None = None,
    *,
    uniforms: bool = False,
) -> str | ShaderProgram:
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

    Design parameters are inlined as float literals by default, which is what
    the frontend consumes today: the whole module changes when any value does.
    Pass ``uniforms=True`` for the form that reads them out of a uniform buffer
    instead — see :func:`compile_scene_with_uniforms`, whose
    :class:`ShaderProgram` this then returns in place of the source string.

    Args:
        geometry: Root CADJOINT SDF node with optional per-primitive materials.
        example_point: Example input for shape inference. Defaults to zeros.
        uniforms: Emit parameters as a uniform buffer rather than as literals,
            returning a :class:`ShaderProgram` instead of a string.

    Returns:
        WGSL source containing the distance and two packed material functions,
            or a :class:`ShaderProgram` when ``uniforms`` is set.
    """
    from cadjoint.extraction import extract_parameters
    from cadjoint.functionalize import functionalize_scene

    if uniforms:
        return compile_scene_with_uniforms(geometry, example_point)

    if example_point is None:
        example_point = jnp.zeros(3, dtype=jnp.float32)

    free_parameters, fixed_parameters, _ = extract_parameters(geometry)

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
    with scalar_lowering():
        sdf, material_at = functionalize_scene(geometry)(free_parameters, fixed_parameters)
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


def _swizzle(components: int) -> str:
    """The WGSL swizzle that reads ``components`` floats out of a ``vec4``."""
    try:
        return _COMPONENT_SWIZZLE[components]
    except KeyError as error:
        raise ValueError(f"A shader parameter must hold 1 to 4 floats, got {components}") from error


def _parameter_components(value) -> int | None:
    """How many floats a parameter value occupies, or None if it cannot fit."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return 1
    if array.ndim == 1 and 1 <= array.size <= 4:
        return int(array.size)
    return None


def _record_parameter_reads(geometry, example_point) -> tuple[list, list]:
    """Which free names and fixed paths each of the three entry points reads.

    The recording runs eagerly on one concrete point, which is enough: the tree
    reads every parameter its shape depends on before it does any arithmetic
    with them.

    Args:
        geometry: Root SDF node.
        example_point: A concrete query point.

    Returns:
        ``(free_names, fixed_paths)``, each in first-read order.
    """
    from cadjoint.extraction import extract_parameters
    from cadjoint.functionalize import functionalize_scene

    free, fixed, _ = extract_parameters(geometry)
    recorded_free, recorded_fixed = _RecordingParams(free), _RecordingParams(fixed)
    sdf, material_at = functionalize_scene(geometry)(recorded_free, recorded_fixed)
    sdf(example_point)
    material_at(example_point)
    return recorded_free.read, recorded_fixed.read


def compile_scene_with_uniforms(
    geometry,
    example_point: jnp.ndarray | None = None,
    *,
    group: int = DEFAULT_PARAMETER_GROUP,
    binding: int = DEFAULT_PARAMETER_BINDING,
) -> ShaderProgram:
    """Compile a scene to WGSL whose parameters live in a uniform buffer.

    :func:`compile_scene_to_wgsl` folds every design parameter into the shader
    as a float literal, so moving one slider by 0.05 rewrites three lines of a
    139 kB module and forces the browser to recompile all of it. Here the same
    parameters are function arguments read out of a uniform block instead: the
    source is byte-identical for every value of every parameter, and an edit is
    a buffer write and a redraw.

    The buffer is an array of ``vec4<f32>`` — one 16-byte slot per parameter,
    which is the only element type a WGSL uniform array carries without
    per-field alignment rules — declared as::

        struct SdfParameters { values: array<vec4<f32>, N> };
        @group(G) @binding(B) var<uniform> sdf_parameters: SdfParameters;

    Parameters a pattern consumes structurally (a ``count``) keep their slot
    but are never read: the instance count decides how much shader gets
    emitted, so it cannot be edited without recompiling.

    Args:
        geometry: Root cadjoint SDF node with optional per-primitive materials.
        example_point: Example input for shape inference. Defaults to zeros.
        group: Bind group index for the uniform.
        binding: Binding index within that group.

    Returns:
        A :class:`ShaderProgram` carrying the source, the buffer layout and
            the parameters' current values.
    """
    from cadjoint.extraction import extract_parameters
    from cadjoint.functionalize import functionalize_scene

    if example_point is None:
        example_point = jnp.zeros(3, dtype=jnp.float32)

    free, fixed, metadata = extract_parameters(geometry)

    with scalar_lowering():
        free_names, fixed_paths = _record_parameter_reads(geometry, example_point)

        slots: list[ShaderParameter] = []
        values: list = []
        for is_free, key in [(True, name) for name in free_names] + [
            (False, path) for path in fixed_paths
        ]:
            raw = (free if is_free else fixed)[key]
            components = _parameter_components(raw)
            if components is None:
                # Wider than a vec4: it stays a literal rather than distorting
                # the layout, and the shader is still correct — just not
                # editable in that one parameter without a recompile.
                continue
            slots.append(
                ShaderParameter(
                    name=key,
                    offset=len(slots) * PARAMETER_SLOT_BYTES,
                    components=components,
                    value=tuple(np.asarray(raw, dtype=np.float32).reshape(-1).tolist()),
                    free=is_free,
                )
            )
            values.append(jnp.asarray(raw, dtype=jnp.float32))

        scene_fn = functionalize_scene(geometry)

        def bind(arguments):
            """Overlay the argument values onto the parameter dicts."""
            bound_free, bound_fixed = dict(free), dict(fixed)
            for slot, argument in zip(slots, arguments):
                (bound_free if slot.free else bound_fixed)[slot.name] = argument
            return scene_fn(bound_free, bound_fixed)

        def distance(point, *arguments):
            return bind(arguments)[0](point)

        def material_base(point, *arguments):
            material = bind(arguments)[1](point)
            color = jnp.reshape(jnp.asarray(material["color"], dtype=jnp.float32), (3,))
            roughness = jnp.reshape(
                jnp.asarray(material["roughness"], dtype=jnp.float32),
                (1,),
            )
            return jnp.concatenate((color, roughness))

        def material_optics(point, *arguments):
            material = bind(arguments)[1](point)
            return jnp.concatenate(
                tuple(
                    jnp.reshape(jnp.asarray(material[key], dtype=jnp.float32), (1,))
                    for key in ("metallic", "opacity", "ior", "reflectivity")
                )
            )

        compiler = StableHLOToWGSL()
        entries = (
            ("sdf", distance, (), "scalar float32 distance"),
            (
                MATERIAL_BASE_ENTRY_POINT,
                material_base,
                (4,),
                "float32 vector with shape (4,)",
            ),
            (
                MATERIAL_OPTICS_ENTRY_POINT,
                material_optics,
                (4,),
                "float32 vector with shape (4,)",
            ),
        )
        sections = [_uniform_block(slots, group, binding)]
        for entry_point, fn, output_shape, output_description in entries:
            sections.append(
                compiler.compile(
                    fn,
                    example_point,
                    *values,
                    entry_point=f"{entry_point}_impl",
                    output_shape=output_shape,
                    output_description=output_description,
                    extra_inputs=len(values),
                )
            )
            sections.append(_uniform_entry_point(entry_point, slots, output_shape))

    return ShaderProgram(
        wgsl="\n\n".join(sections),
        parameters=tuple(slots),
        buffer_bytes=max(len(slots), 1) * PARAMETER_SLOT_BYTES,
        group=group,
        binding=binding,
    )


def _uniform_block(slots: list, group: int, binding: int) -> str:
    """The uniform declaration the generated entry points read from."""
    count = max(len(slots), 1)
    lines = [f"// {len(slots)} design parameters, one vec4<f32> slot each."]
    for slot in slots:
        kind = "free" if slot.free else "fixed"
        lines.append(f"// {slot.offset:6d}  {slot.components}f  {kind:5s}  {slot.name}")
    lines.append("struct SdfParameters {")
    lines.append(f"    values: array<vec4<f32>, {count}>,")
    lines.append("};")
    lines.append(f"@group({group}) @binding({binding}) var<uniform> sdf_parameters: SdfParameters;")
    return "\n".join(lines)


def _uniform_entry_point(entry_point: str, slots: list, output_shape: tuple[int, ...]) -> str:
    """A ``point -> value`` wrapper that feeds the uniform into the impl."""
    arguments = ["p"] + [
        f"sdf_parameters.values[{slot.offset // PARAMETER_SLOT_BYTES}].{_swizzle(slot.components)}"
        for slot in slots
    ]
    return_type = "f32" if output_shape == () else f"vec{output_shape[0]}<f32>"
    body = ",\n        ".join(arguments)
    return (
        f"fn {entry_point}(p: vec3<f32>) -> {return_type} {{\n"
        f"    return {entry_point}_impl(\n        {body}\n    );\n}}"
    )
