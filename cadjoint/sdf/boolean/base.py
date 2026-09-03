"""Base class for boolean operation SDFs, and the material blend they share."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from cadjoint.sdf import SDF


def _mix(first: Any, second: Any, weight: Array) -> Array:
    """Lerp two property values, letting a specified one survive an unspecified one.

    Written as the *double* ``where`` that JAX's autodiff requires. Masking
    the result of an expression that saw a ``nan`` is not enough: the VJP of
    ``jnp.where`` multiplies the unselected branch's cotangent by zero, and
    ``0.0 * nan`` is ``nan``. So a single-``where`` version returns the right
    value and a ``nan`` *gradient*, which is worse than the bug it fixes —
    the value is checkable and the gradient reaches an optimiser as
    ``grad_norm=nan`` twenty minutes later. Both operands are made finite
    *before* the lerp; the second ``where`` puts the ``nan`` back where it
    belongs, over an expression that never carried one.
    """
    a, b = jnp.asarray(first), jnp.asarray(second)
    a_missing, b_missing = jnp.isnan(a), jnp.isnan(b)
    a_safe = jnp.where(a_missing, jnp.zeros_like(a), a)
    b_safe = jnp.where(b_missing, jnp.zeros_like(b), b)
    lerp = b_safe * (1.0 - weight) + a_safe * weight
    return jnp.where(a_missing, b, jnp.where(b_missing, a, lerp))


def blend_materials(first: dict, second: dict, weight: Array) -> dict:
    """Blend two material dicts across a boolean seam.

    ``weight`` is the fraction of *first*: 1 returns ``first``, 0 returns
    ``second``.

    This is :meth:`cadjoint.render.material.Material.blend` with one rule
    added, and the rule is load-bearing. A plain lerp is
    ``b * (1 - t) + a * t``, and IEEE 754 says ``nan * 0.0 == nan`` — so a
    child that leaves a property unspecified erases the other child's value
    even where its weight is *exactly zero*. Every cut tool in a real part is
    such a child: a ``Face.hole`` is geometry, not a substance, and it says
    nothing about conductivity. The consequence was that any solid with a
    single hole in it reported ``conductivity = nan`` at every interior point,
    and every ``FROM_MATERIAL`` study over it failed with "the scene's
    material field does not specify 'conductivity' for 3748 of 3748
    elements".

    Here a property is unspecified only when *neither* side specifies it. Two
    specified values still blend exactly as before, which is what makes a
    smooth CSG interface a smooth (and differentiable) property interface.

    Args:
        first: Material dict whose keys define the result's keys.
        second: The other material dict.
        weight: Blend factor, broadcastable against each property.

    Returns:
        A material dict with the same keys as ``first``.
    """
    return {key: _mix(first[key], second[key], weight) for key in first}


class BooleanOp(SDF):
    """Base class for boolean operation SDFs.

    Boolean operations combine one or more SDFs - union, intersection, difference, etc.

    Subclasses must implement:
    - @staticmethod def sdf(child_sdfs: tuple, p: Array, **params) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable

    Subclasses should store:
    - self.sdfs: Tuple of child SDFs
    - self.params: Dictionary of Parameter objects
    """

    # A boolean leaves its base operand's surface where it was — cutting a
    # hole in a plate does not move the plate's top face — so the base's
    # analytic face references still land on the result and are forwarded by
    # SDF.__getattr__. Two caveats live in that method's docstring: the
    # boundary polygon is the *uncut* outline, and a smoothness > 0 blend
    # rounds the surface near the seam while leaving it exact away from it.
    inherits_faces = True

    def children(self) -> list:
        return list(self.sdfs)
