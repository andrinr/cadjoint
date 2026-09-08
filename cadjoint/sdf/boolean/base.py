"""Base class for boolean operation SDFs, and the material blend they share."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from cadjoint.sdf import SDF
from cadjoint.sdf.base import _child_patch_fields


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


def _is_hard(smoothness) -> bool:
    """Whether a blend radius is a *known* zero, i.e. the node is sharp CSG.

    A traced radius reports ``False``: it could be anything at call time, and
    the decomposition below is only valid for the sharp composition, so the
    unknown case has to decline.

    Args:
        smoothness: The node's blend radius — a number or a tracer.

    Returns:
        True when the value is concrete and exactly zero.
    """
    try:
        return float(smoothness) == 0.0
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return False


def _operand_patch_fields(node: BooleanOp, negate_tools: bool = False):
    """The operands' patch fields concatenated, operand-major.

    A hard boolean is a ``min``/``max`` over its operands, so its surface is
    made of pieces of the operands' surfaces and nothing else: the node's
    decomposition is exactly the operands' decompositions, laid end to end.
    The order is **operand-major** — all of ``sdfs[0]``'s fields, then all of
    ``sdfs[1]``'s, and so on — and stable, so a consumer that wants to know
    which operand a patch came from can recover it by counting.  Nothing here
    marks it: the consumer's ownership is a two-stage ``argmin`` (nearest
    leaf, then nearest patch within it), it assumes no operand is convex or
    even connected, and a declared patch that ends up bounding nothing is
    dropped by its own boundary probe.  A marker would be a fifth wheel.

    If any operand has no decomposition, neither does the node: the protocol's
    contract is that the fields cover the whole surface, and a partial cover
    would hide the missing operand's surface rather than fall back to the
    single opaque patch a ``None`` earns.

    Where this stops being exact is smoothness.  A blended boolean rounds the
    seam, and the rounded fillet is **not** on any operand's zero set — it is
    a surface the blend invents, which no operand field vanishes on.  So a
    node with a non-zero (or traced, hence unknown) blend radius declines
    rather than declaring a decomposition with a hole in it where the fillet
    is.  Sharp nodes, where ``smooth_min``/``smooth_max`` degenerate to plain
    ``min``/``max``, are exact.

    Args:
        node: The boolean node, read for its ``smoothness`` and ``sdfs``.
        negate_tools: Negate every operand after the first, which is what
            :class:`~cadjoint.sdf.boolean.difference.Difference` needs so its
            tools' fields still read positive outside the *result*.

    Returns:
        The concatenated fields, or ``None`` when the node is blended or any
        operand declares nothing.
    """
    if not _is_hard(node.params["smoothness"].value):
        return None
    fields: list = []
    for index, child in enumerate(node.sdfs):
        child_fields = _child_patch_fields(child)
        if child_fields is None:
            return None
        if negate_tools and index > 0:
            fields.extend((lambda p, f=field: -f(p)) for field in child_fields)
        else:
            fields.extend(child_fields)
    return fields


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
