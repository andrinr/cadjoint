"""Smooth blending functions for boolean operations.

**These shift the surface, and by how much is worth knowing before using
them.**  Both are Inigo Quilez's polynomial blend, and the blend is paid for
by displacing the field, not only near the seam.  Setting ``a == b`` in
:func:`smooth_min` collapses it to ``a - k``, so wherever the two operands
agree the result is the operand lowered by the full smoothness, and
:func:`smooth_max` raises it by the same amount.  Between those extremes the
shift falls off over a band of width ``4k``.

Two consequences bite in practice.

The shift **accumulates**: each blended operation moves the field again, so a
part cut by eight tools can sit up to ``8k`` away from where its author drew
it.  And the worst case is not a rare coincidence but the thing a draughtsman
naturally draws — two **coincident faces**.  Operands that are equal along a
whole plane are equal *everywhere on it*, so the entire face moves by ``k``
rather than a seam rounding by ``k``.  A cavity ceiling drawn exactly on the
solid's floor will lift the whole floor out of the solid.  Overlap such faces
deliberately instead of matching them.

**The blend is not a rolling ball, and calling it a fillet oversells it.**
Setting the smooth minimum to zero cancels its cross terms exactly and
leaves ``sqrt(u) + sqrt(v) = sqrt(4k)``, so the blend surface is a parabola
swept along the operands' intersection.  Against a best-fit circle across
the band it is wrong by about 6.6 % of ``k``, and its curvature radius
varies by a factor of 2.828 from end to end.  For a cosmetic rounding that
is immaterial; for anything that has to match a drawn radius, or for a
downstream tool that expects a constant-radius surface, it is not.  Only
two operand pairs have an exact surface — two planes, and a coaxial pair
with straight meridians — and every other pair has a profile of genus one
or two, so no NURBS represents it exactly at any degree.

``k`` is in model units and knows nothing about the part.  The default of
0.1 was chosen for the unit-sized geometry in the shipped scenes, where it is
a tenth of the smallest feature and reads as a fillet.  On a scene whose unit
stands for something larger it is whatever 0.1 of that unit happens to be,
which on a normalised engine block came to 20 mm and removed the water
jacket.  Set it explicitly whenever the scene's unit is not roughly the scale
of its smallest feature.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def smooth_min(a: Array, b: Array, k: float = 0.1) -> Array:
    """Smooth minimum function for blending SDFs (Inigo Quilez's version).

    Args:
        a: First SDF value(s)
        b: Second SDF value(s)
        k: Smoothness parameter (larger = smoother blend)

    Returns:
        Smoothly blended minimum value
    """
    k_scaled = jnp.maximum(k * 4.0, 1e-10)
    h = jnp.maximum(k_scaled - jnp.abs(a - b), 0.0)
    return jnp.minimum(a, b) - h * h * 0.25 / k_scaled


def smooth_max(a: Array, b: Array, k: float = 0.1) -> Array:
    """Smooth maximum function for blending SDFs.

    Args:
        a: First SDF value(s)
        b: Second SDF value(s)
        k: Smoothness parameter (larger = smoother blend)

    Returns:
        Smoothly blended maximum value
    """
    return -smooth_min(-a, -b, k)
