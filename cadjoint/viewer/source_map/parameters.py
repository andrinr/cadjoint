"""Which named free parameter backs a value the viewer can drag.

The scene's two shaders read every *free* design parameter out of a uniform
buffer (``compile_scene_with_uniforms``), so moving one is a ``writeBuffer``
rather than a recompile.  The frontend can only take that path if it knows
*which slot* a handle drives, and the compile payload used to say only where a
value lives in the text — enough to rewrite the source, not enough to write a
buffer.

This module answers the missing half: given a captured construction object,
which of its draggable values are backed by a free ``Parameter``, and by what
name.  The names are the ones :func:`cadjoint.extract_parameters` returns and
the ones ``ShaderParameter.name`` carries, which is what lets the client join
the two halves of the payload.

Two rules, both about not guessing:

- **A value is bound only when every one of its components is.**  A rotation
  whose X angle is free and whose Y angle is a pinned ``Scalar`` reports no
  binding at all, because a drag that wrote half of it would leave the image
  disagreeing with the source.
- **A binding is a claim about the source, not about the shader.**  A
  parameter can be free and named and still have no slot — a zero rotation is
  never built into the SDF, so its angle folds away entirely.  The client
  checks each name against the program's slot table and falls back when it is
  missing; see ``dragBinding.ts``.
"""

from __future__ import annotations

from typing import Any

#: The rotation angles a primitive keeps, in the order the payload lists them.
ROTATION_KEYS = ("rx", "ry", "rz")


def _binding(parameter: Any, index: int | None = None) -> dict | None:
    """One free parameter's entry, or ``None`` if it cannot back a drag.

    Args:
        parameter: A ``Parameter`` (``Scalar``, ``Vector``, ``Vector2``), or
            anything else — a raw float reaches here when a value was never
            wrapped, and is not bindable.
        index: Which component of the payload's value this parameter drives,
            when it drives exactly one of several. ``None`` when it covers the
            whole value.

    Returns:
        ``{"name", "components", "index"}``, or ``None`` when the value is a
        fixed literal (not free, or free but unnamed — extraction would refuse
        it, so the viewer must not promise a slot for it).
    """
    if not getattr(parameter, "free", False):
        return None
    name = getattr(parameter, "name", None)
    if not name:
        return None
    value = getattr(parameter, "value", None)
    if value is None:
        return None
    components = int(getattr(value, "size", 1)) or 1
    return {"name": str(name), "components": components, "index": index}


def vertex_binding(vertex: Any) -> dict | None:
    """The free parameter backing a sketch vertex, if it has one.

    A profile vertex *is* the parameter — one ``Vector2`` per point — so the
    binding covers both components at once and carries no index.
    """
    return _binding(vertex)


def transform_bindings(primitive: Any) -> dict[str, list[dict]]:
    """The free parameters backing a primitive's draggable arguments.

    Keyed by the argument name a drag writes back (``position``, ``rotation``,
    and the kind's dimension keywords), so the client can look up the very
    argument it is about to patch. An argument only appears when *all* of it
    is bound; a partially free value is left out rather than half-applied.

    Args:
        primitive: A ``ConstructionPrimitive``.

    Returns:
        Argument name → the bindings covering its components, in order.
    """
    from cadjoint.construction.solid import DIMENSIONS

    params = getattr(primitive, "params", None)
    if not params:
        return {}

    bindings: dict[str, list[dict]] = {}
    position = _binding(params.get("position"))
    if position is not None:
        bindings["position"] = [position]

    # Three scalars behind one three-component payload value: each angle
    # names the component it drives.
    angles = [_binding(params.get(key), index) for index, key in enumerate(ROTATION_KEYS)]
    if all(angle is not None for angle in angles):
        bindings["rotation"] = [angle for angle in angles if angle is not None]

    for key in DIMENSIONS.get(getattr(primitive, "kind", ""), ()):
        dimension = _binding(params.get(key))
        if dimension is not None:
            bindings[key] = [dimension]
    return bindings
