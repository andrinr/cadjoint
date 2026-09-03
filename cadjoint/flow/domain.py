"""From a signed distance field to a penalisation field, on a fixed grid.

This is the whole coupling between cadjoint's geometry and this solver, and
it is two lines of arithmetic: sample the scene's SDF at the cell centres of
a fixed grid, and squash the distances through a smooth profile.

``chi`` is 1 deep inside the solid, 0 well outside it, and passes smoothly
through 1/2 on the surface.  Nothing here contours, meshes, or decides which
cells are "in" -- there is no discrete membership anywhere in the chain, so
``d chi / d theta`` exists for every design parameter and is exactly what
:func:`jax.grad` of the SDF gives.  That is why a fixed grid is not a
limitation here: the grid never moves, the *field on it* does.

**The profile has to have compact support, and that is not a detail.**  The
obvious choice is ``sigmoid(-f(x)/eps)``, which is what ``scenes/starter.py``
uses for its material-volume regulariser.  It is the wrong choice *here*,
because this field is not integrated -- it is multiplied by ``alpha_max``
and fed to a momentum sink.  A sigmoid never reaches zero, so a cell in the
middle of an open fin channel still carries ``chi ~ 1e-2``, and
``alpha_max`` multiplies that tail along with everything else.  Raising
``alpha_max`` to make the *solid* impermeable therefore makes the *fluid*
porous at the same rate, and the answer never converges to the no-slip one
it is supposed to approximate.

Measured on the starter sink at 32x64x32, pressure drop against
``alpha_max``:

=========  ==========  =============
alpha_max  sigmoid     compact
=========  ==========  =============
1          2.12e-2     5.63e-3
5          5.02e-2     6.24e-3
20         1.22e-1     6.49e-3
100        diverged    6.64e-3
400        diverged    6.72e-3
=========  ==========  =============

The sigmoid column grows without bound and then blows up; the compact
column converges, changing 1.1% over the last factor of four.  In the cells
the geometry calls clearly open (``f > 2 eps``) the sigmoid leaves ``chi``
as high as 0.12 -- a drag of 12 at ``alpha_max = 100``, against a kinematic
viscosity of 0.0064.

Clamping to exact 0 and 1 costs nothing in differentiability that matters:
the derivative outside the band is genuinely zero, because moving a surface
that is already more than ``epsilon`` away does not change that cell's
occupancy.  Every cell the design can actually influence sits in the band,
and there the profile is smooth.

**Which compact profile, though.**  Clamping puts a join at each band edge,
and how smooth that join is decides how well a *finite-difference check*
behaves -- not how correct the adjoint is, but how easily it can be
confirmed.  The cubic ``"smoothstep"`` is only C1 there, so a central
difference through it converges at first order; the quintic
``"smootherstep"`` matches second derivatives too and restores the expected
second order.  Measured on a box obstacle whose faces land exactly on cell
centres (the worst case -- many cells sit precisely on a join), relative
error between a central difference and the adjoint:

========  ==============  ==============  ==============
h         smootherstep    smoothstep      sigmoid
========  ==============  ==============  ==============
1e-2      2.57e-3         1.64e-1         2.57e-6
1e-3      2.59e-5         1.93e-2         2.43e-8
1e-4      2.60e-7         1.97e-3         1.11e-9
1e-5      2.90e-9         1.97e-4         1.29e-9
========  ==============  ==============  ==============

The smoothstep column falls by ten per ten and the smootherstep column by a
hundred, which is the whole difference between a first- and a second-order
truncation.  The adjoint itself is exact in every column -- confirmed
independently against a converged unrolled tape, which agrees with it to
5.6e-11 (:mod:`cadjoint.flow.steady`).  ``"smootherstep"`` is the default
because it costs one extra multiply and makes the gradient checkable.

``epsilon`` is the interface width in world units.  Too sharp and the
gradient localises onto a single cell layer and gets noisy; too soft and the
solid bleeds into the flow and thin fins never reach ``chi = 1``.
:meth:`FlowGrid.suggested_epsilon` returns half a cell, which puts the
transition across about two of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FlowGrid:
    """A fixed axis-aligned lattice of cell centres in world coordinates.

    Attributes:
        shape: ``(NX, NY, NZ)`` cell counts.  Flow runs along ``+Y``.
        origin: World coordinates of the box's minimum corner.
        size: World extent of the box along each axis.
    """

    shape: tuple[int, int, int]
    origin: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def spacing(self) -> tuple[float, float, float]:
        """World size of one cell along each axis."""
        return tuple(s / n for s, n in zip(self.size, self.shape))

    @property
    def cells(self) -> int:
        """Total number of cells."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def cell_volume(self) -> float:
        """World volume of one cell."""
        dx, dy, dz = self.spacing
        return dx * dy * dz

    def suggested_epsilon(self) -> float:
        """Half a cell, which puts the interface across about two of them.

        ``sigmoid(-d/eps)`` runs from 0.12 to 0.88 over ``|d| < 2 eps``, so
        the transition band is four ``epsilon`` wide: half a cell here means
        two cells of smear.  That is the narrowest band the lattice still
        resolves, and it matters that it *is* narrow — a wider one never
        lets ``chi`` reach 1 inside a thin fin, so the Brinkman drag never
        reaches ``alpha_max`` and the fin leaks.  (Measured on the starter
        sink at 32x64x32: ``max chi`` is 0.85 at one cell diagonal and 0.99
        here.)

        Returns:
            The recommended ``epsilon`` for :func:`solid_fraction`.
        """
        dx, dy, dz = self.spacing
        return 0.5 * (dx * dy * dz) ** (1.0 / 3.0)

    def centers(self) -> jax.Array:
        """Cell-centre coordinates.

        Returns:
            ``(NX, NY, NZ, 3)`` world coordinates.
        """
        axes = [
            self.origin[axis] + (jnp.arange(self.shape[axis]) + 0.5) * self.spacing[axis]
            for axis in range(3)
        ]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

    def flat_centers(self) -> jax.Array:
        """Cell-centre coordinates as a point list.

        Returns:
            ``(NX*NY*NZ, 3)`` world coordinates.
        """
        return self.centers().reshape(-1, 3)


#: The profiles :func:`solid_fraction` accepts.
PROFILES = ("smootherstep", "smoothstep", "sigmoid")


def solid_fraction(distance: jax.Array, epsilon: float, profile: str = "smootherstep") -> jax.Array:
    """Smooth solid indicator from signed distances.

    Args:
        distance: Signed distance, negative inside the solid, any shape.
        epsilon: Interface half-width in the same units as ``distance``.
        profile: ``"smootherstep"`` -- Perlin's quintic blend, exactly 1
            below ``-epsilon`` and exactly 0 above ``+epsilon``, with both
            first and second derivatives vanishing at the join.
            ``"smoothstep"`` -- the cubic Hermite blend, same support but
            only C1.  ``"sigmoid"`` -- ``sigmoid(-distance/epsilon)``,
            which never reaches either end; see this module's docstring for
            why that is a poor choice for a penalisation field and a fine
            one for a volume integral.

    Returns:
        Solid fraction in ``[0, 1]``, the shape of ``distance``.

    Raises:
        ValueError: On an unknown ``profile``.
    """
    if profile == "sigmoid":
        return jax.nn.sigmoid(-distance / epsilon)
    t = jnp.clip(0.5 - distance / (2.0 * epsilon), 0.0, 1.0)
    if profile == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if profile == "smootherstep":
        return t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    raise ValueError(f"profile must be one of {PROFILES}; got {profile!r}.")


def sample_solid_fraction(
    sdf: Callable[[jax.Array], jax.Array],
    grid: FlowGrid,
    epsilon: float | None = None,
    profile: str = "smootherstep",
) -> jax.Array:
    """Evaluate a scene SDF on the grid and squash it to a solid fraction.

    Args:
        sdf: A callable on ``(..., 3)`` points, as
            :func:`cadjoint.functionalize` produces.
        grid: The lattice to sample.
        epsilon: Interface half-width; ``None`` uses
            :meth:`FlowGrid.suggested_epsilon`.
        profile: Passed to :func:`solid_fraction`.

    Returns:
        ``(NX, NY, NZ)`` solid fraction.
    """
    width = grid.suggested_epsilon() if epsilon is None else epsilon
    distance = sdf(grid.flat_centers())
    return solid_fraction(jnp.reshape(distance, grid.shape), width, profile)
