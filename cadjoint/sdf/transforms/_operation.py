"""Shared plumbing for the transforms that wrap one callable shape.

`Shell`, `Offset`, `Mirror` and the two patterns are all `Transform`s over a
single child, and they need the same three things: a child list that tolerates
a plain lambda, a material lookup that follows the point through the
transform, and a guarded way to read a world direction off a parameter or a
geometric reference.  Keeping that here rather than in one of them stops
`patterns` importing privates from `fields` for no reason other than history.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from cadjoint.fluent import Fluent
from cadjoint.sdf.transforms.base import Transform

_MIRROR_NORMALS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}

# Below this squared length a direction carries no orientation; the guarded
# norm keeps value and derivative finite where a bare norm would give 0/0.
_MIN_SQUARED = 1e-12


def _unit(vector: Array) -> Array:
    """Normalize a 3-vector with a guarded norm, safe under tracing."""
    return vector / jnp.sqrt(jnp.maximum(jnp.sum(vector * vector), _MIN_SQUARED))


def _world_vector(value) -> Array:
    """A world 3-vector from a raw array or from a ``Vector`` parameter."""
    return jnp.asarray(getattr(value, "xyz", value), dtype=jnp.float32)


def _reference_line(reference, attribute: str) -> tuple[Array, Array] | None:
    """``(origin, unit direction)`` read off a geometric reference, or None.

    Duck-typed on purpose: ``cadjoint.sdf`` must not import
    ``cadjoint.construction`` — the dependency runs the other way — but the
    references a user naturally reaches for all expose the same two
    attributes. An :class:`~cadjoint.construction.faces.Axis` carries
    ``origin``/``direction``; a :class:`~cadjoint.construction.faces.Face` and
    a :class:`~cadjoint.construction.sketch.SketchPlane` carry
    ``origin``/``normal``, the plane's ``origin`` being a ``Vector``
    parameter rather than a bare array.

    Args:
        reference: The object to read.
        attribute: ``"direction"`` for a line, ``"normal"`` for a plane.

    Returns:
        The pair, or None when ``reference`` does not carry both attributes.
    """
    origin = getattr(reference, "origin", None)
    vector = getattr(reference, attribute, None)
    if origin is None or vector is None:
        return None
    return _world_vector(origin), _unit(_world_vector(vector))


class _Operation(Transform):
    """Shared plumbing for field operations wrapping one callable shape."""

    def children(self) -> list:
        # A plain lambda child cannot be walked as a Fluent tree node.
        return [self.sdf] if isinstance(self.sdf, Fluent) else []

    def material_at(self, p: Array) -> dict:
        if isinstance(self.sdf, Fluent):
            values = self._extract_param_values()
            return self.sdf.material_at(self.__class__._transform_point(p, **values))
        return super().material_at(p)
