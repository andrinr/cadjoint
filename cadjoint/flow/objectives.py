"""Scalar read-outs of a converged flow, and what each one is good for.

Both quantities here are integrals of the converged fields, so both are
smooth in the design through :mod:`cadjoint.flow.steady`'s adjoint.  They
pull in opposite directions, which is the point: a heat sink that maximises
surface contact with moving air by filling the duct with metal also
strangles the fan driving it, and an optimiser needs both terms to find the
fin pitch in between.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cadjoint.flow.lattice import CS2
from cadjoint.flow.lbm import macroscopic


def fields(f: jax.Array, chi: jax.Array, alpha_max: float) -> tuple[jax.Array, jax.Array]:
    """Density and velocity from converged populations.

    Args:
        f: Converged populations, ``(19, NX, NY, NZ)``.
        chi: Solid fraction, ``(NX, NY, NZ)``.
        alpha_max: Drag coefficient at ``chi = 1``.

    Returns:
        ``(rho, u)`` with shapes ``(NX, NY, NZ)`` and ``(3, NX, NY, NZ)``.
    """
    return macroscopic(f, alpha_max * chi)


def pressure(rho: jax.Array) -> jax.Array:
    """Lattice pressure from density: ``p = rho cs^2``.

    Args:
        rho: Density, ``(NX, NY, NZ)``.

    Returns:
        Pressure in lattice units, ``(NX, NY, NZ)``.
    """
    return rho * CS2


def pressure_drop(rho: jax.Array, margin: int = 1) -> jax.Array:
    """Mean pressure across the inlet plane minus across the outlet plane.

    This is the fan's share of the bargain: what it costs to push air
    through whatever the design has left of the duct.  The planes are taken
    ``margin`` cells inside the ends so the inlet's imposed equilibrium and
    the outlet's copied plane do not enter the number.

    Args:
        rho: Density, ``(NX, NY, NZ)``.
        margin: How many cells in from each end to measure.

    Returns:
        Scalar pressure drop in lattice units, positive for flow that
        resists.
    """
    upstream = jnp.mean(pressure(rho)[:, margin, :])
    downstream = jnp.mean(pressure(rho)[:, -1 - margin, :])
    return upstream - downstream


def heat_transfer_proxy(chi: jax.Array, u: jax.Array, cell_volume: float = 1.0) -> jax.Array:
    """``int chi |u|`` -- air in motion where the metal is.

    A stand-in for the surface heat-transfer coefficient, and a cheap one:
    it needs no surface extraction, only the two fields already in hand.
    Deep inside the solid it contributes nothing (the drag has killed ``u``
    there) and far outside it contributes nothing (``chi`` is zero), so what
    it actually measures is flow speed in the interface band -- which is
    where convective cooling happens.

    Its known weakness is that it is a volume integral over a band whose
    thickness is set by ``epsilon``, so comparing two designs at different
    grid resolutions needs the same ``epsilon`` in world units.  The
    surface-weighted variant ``int |grad chi| |u|`` removes that dependence
    at the cost of a finite-difference gradient of ``chi``; see
    ``research/flow-solver.md``.

    Args:
        chi: Solid fraction, ``(NX, NY, NZ)``.
        u: Velocity, ``(3, NX, NY, NZ)``.
        cell_volume: World volume of one cell, to make the integral extensive.

    Returns:
        Scalar proxy, larger is better cooled.
    """
    speed = jnp.sqrt(jnp.sum(u * u, axis=0) + 1e-30)
    return cell_volume * jnp.sum(chi * speed)
