"""The assembled solver: a penalisation field in, scalar read-outs out.

:func:`solve` is the whole prototype in one call.  It is a pure function of
the solid fraction ``chi`` and the inlet velocity, differentiable in both,
with the fixed-point adjoint of :mod:`cadjoint.flow.steady` underneath -- so
``jax.grad(lambda chi: solve(chi, config).pressure_drop)`` is a valid and
tape-free thing to write.

Everything static lives in :class:`FlowConfig`, which is frozen and
hashable so the step closure can be memoised on it.  That matters more than
it looks: :func:`~cadjoint.flow.steady.steady_populations` takes its step
function as a ``nondiff_argnum``, which JAX hashes by *identity*, so a
freshly built closure on every call would recompile the entire scan every
time.  :func:`step_for` is the cache that stops that happening.

**Units.**  The solver works in lattice units throughout: one cell, one
step, reference density 1.  ``inlet_speed`` is the inlet velocity in cells
per step and should stay well under the lattice sound speed
(``1/sqrt(3) = 0.577``) -- 0.05 or below keeps the compressibility error
under a percent.  The Reynolds number fixes the viscosity, and hence the
relaxation rate, against a characteristic length quoted in *cells*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

import jax
import jax.numpy as jnp

from cadjoint.flow import objectives
from cadjoint.flow.lattice import omega_from_viscosity
from cadjoint.flow.lbm import bounce_back_mask, duct_walls, initial_populations, step
from cadjoint.flow.steady import SteadyOptions, residual_history, steady_populations

#: The largest ``omega`` this solver will accept.  BGK is nominally stable
#: for any ``omega < 2``, but the Brinkman drag couples a very fast mode
#: into the collision and the real limit is lower.  Measured on the starter
#: sink, ``inlet_speed = 0.02``, ``alpha_max = 200``: 24x48x24 converges at
#: ``omega = 1.9440`` and 20x40x20 returns NaN at ``omega = 1.9531``.  The
#: ceiling sits between them, and a config that trips it fails immediately
#: with a message rather than after a minute of marching to NaN.
OMEGA_CEILING = 1.95


@dataclass(frozen=True)
class FlowConfig:
    """Static settings for one flow problem.

    Attributes:
        shape: ``(NX, NY, NZ)`` grid, with the duct axis along ``+Y``.
        inlet_speed: Inlet velocity in cells per step (lattice units).
        reynolds: Reynolds number, which together with ``inlet_speed`` and
            ``characteristic_cells`` fixes the viscosity.
        characteristic_cells: The length scale of the Reynolds number, in
            cells.  ``None`` uses the duct's ``NZ``.
        alpha_max: Brinkman drag at ``chi = 1``.  Large enough that the
            solid is effectively impermeable, small enough that the
            linearised operator stays well conditioned; see
            :func:`recommended_alpha_max` for the sweep that picks it.
            Note that it multiplies ``chi`` *everywhere*, so it is only
            safe with a compactly supported profile -- see
            :mod:`cadjoint.flow.domain`.
        steady: Convergence settings for the forward and adjoint solves.

    Raises:
        ValueError: If the implied ``omega`` exceeds :data:`OMEGA_CEILING`.
    """

    shape: tuple[int, int, int] = (32, 64, 32)
    inlet_speed: float = 0.05
    reynolds: float = 100.0
    characteristic_cells: int | None = None
    alpha_max: float = 200.0
    steady: SteadyOptions = field(default_factory=SteadyOptions)

    @property
    def length_scale(self) -> int:
        """The characteristic length in cells."""
        return self.characteristic_cells or self.shape[2]

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity in lattice units from the Reynolds number."""
        return self.inlet_speed * self.length_scale / self.reynolds

    @property
    def omega(self) -> float:
        """The BGK relaxation rate this configuration implies."""
        return omega_from_viscosity(self.viscosity)

    @property
    def mach(self) -> float:
        """Inlet Mach number; compressibility error grows with its square."""
        return self.inlet_speed * (3.0**0.5)

    def __post_init__(self) -> None:
        if self.omega > OMEGA_CEILING:
            raise ValueError(
                f"shape={self.shape}, inlet_speed={self.inlet_speed}, "
                f"reynolds={self.reynolds} gives viscosity {self.viscosity:.5g} and "
                f"omega {self.omega:.4f}, above the stable ceiling {OMEGA_CEILING}. "
                "BGK loses stability as omega approaches 2, and the penalised solid "
                "brings it on sooner. Any of these fixes it: lower the Reynolds "
                "number, lower inlet_speed, or refine the grid "
                "(characteristic_cells scales the viscosity with resolution)."
            )


#: Drag at ``chi = 1``.  A pure lattice-units rate (velocity surviving one
#: step is ``1/(1 + alpha/2)``), so unlike the viscosity it does not scale
#: with the Reynolds number or the inlet speed -- which is why
#: :func:`recommended_alpha_max` returns a constant rather than a formula.
DEFAULT_ALPHA_MAX = 200.0


def recommended_alpha_max(config: FlowConfig) -> float:
    """A drag strong enough that the solid reads as impermeable.

    Measured on the starter sink at 32x32x64, ``inlet_speed = 0.02``,
    ``Re = 100``, with the ``"smoothstep"`` profile -- leak is mean ``|u|``
    where ``chi > 0.8``, as a fraction of the inlet speed:

    =========  ==========  ==========
    alpha_max  drop        leak
    =========  ==========  ==========
    1          5.63e-3     2.2e-2
    5          6.24e-3     6.1e-3
    20         6.49e-3     1.9e-3
    100        6.64e-3     5.0e-4
    400        6.72e-3     2.0e-4
    2000       6.89e-3     4.3e-5
    50000      7.66e-3     2.7e-6
    =========  ==========  ==========

    Leak falls like ``1/alpha``; the pressure drop converges much more
    slowly, because the penalised wall sits a penetration depth
    ``sqrt(nu/alpha)`` inside the true one and that error only falls like
    ``1/sqrt(alpha)``.  Chasing the last percent is therefore not worth the
    stiffness it puts into the adjoint, and :data:`DEFAULT_ALPHA_MAX` sits
    where leak is under 3e-4 and the drop is within about 3% of its value
    at fifty times the drag.

    The march stays finite to ``alpha_max = 5e4`` at least, so this is a
    working point rather than a stability limit -- but see
    :mod:`cadjoint.flow.domain` on why it is only a working point with a
    *compactly supported* ``chi``.

    Args:
        config: The flow configuration.

    Returns:
        A suggested ``alpha_max``.
    """
    del config
    return DEFAULT_ALPHA_MAX


@lru_cache(maxsize=32)
def step_for(config: FlowConfig) -> Callable[[jax.Array, dict], jax.Array]:
    """The memoised one-step closure for a configuration.

    Cached on ``config`` so that repeated solves reuse one compiled scan;
    see this module's docstring for why identity matters here.

    Args:
        config: The flow configuration.

    Returns:
        ``step_fn(f, theta) -> f`` where ``theta`` has keys ``"chi"`` and
        ``"inlet_velocity"``.
    """
    wall = duct_walls(config.shape)
    incoming = bounce_back_mask(wall)
    omega = config.omega
    alpha_max = config.alpha_max

    def step_fn(f: jax.Array, theta: dict) -> jax.Array:
        return step(
            f,
            theta["chi"],
            theta["inlet_velocity"],
            omega=omega,
            alpha_max=alpha_max,
            wall=wall,
            incoming=incoming,
        )

    return step_fn


def _theta(chi: jax.Array, config: FlowConfig, inlet_velocity: Any) -> dict:
    """Assemble the differentiable parameter pytree."""
    if inlet_velocity is None:
        inlet_velocity = jnp.array([0.0, config.inlet_speed, 0.0], dtype=chi.dtype)
    return {"chi": chi, "inlet_velocity": jnp.asarray(inlet_velocity, dtype=chi.dtype)}


@dataclass(frozen=True)
class FlowResult:
    """The converged flow and the scalars read off it.

    Attributes:
        populations: Converged populations, ``(19, NX, NY, NZ)``.
        density: Density, ``(NX, NY, NZ)``.
        velocity: Velocity, ``(3, NX, NY, NZ)``.
        pressure_drop: Inlet-to-outlet pressure difference, lattice units.
        heat_transfer: The ``int chi |u|`` cooling proxy.
    """

    populations: jax.Array
    density: jax.Array
    velocity: jax.Array
    pressure_drop: jax.Array
    heat_transfer: jax.Array


def solve(
    chi: jax.Array,
    config: FlowConfig,
    inlet_velocity: Any = None,
    cell_volume: float = 1.0,
) -> FlowResult:
    """Converge the penalised flow and read off both objectives.

    Args:
        chi: Solid fraction, ``(NX, NY, NZ)``, matching ``config.shape``.
        config: The flow configuration.
        inlet_velocity: ``(3,)`` inlet velocity; ``None`` uses
            ``config.inlet_speed`` along ``+Y``.
        cell_volume: World volume of one cell, for the heat-transfer
            integral.

    Returns:
        The :class:`FlowResult`.

    Raises:
        ValueError: If ``chi`` does not match ``config.shape``.
    """
    if tuple(chi.shape) != tuple(config.shape):
        raise ValueError(f"chi has shape {tuple(chi.shape)}, expected {tuple(config.shape)}.")
    step_fn = step_for(config)
    theta = _theta(chi, config, inlet_velocity)
    f0 = initial_populations(config.shape, theta["inlet_velocity"])
    f_star = steady_populations(step_fn, config.steady, theta, f0)
    density, velocity = objectives.fields(f_star, chi, config.alpha_max)
    return FlowResult(
        populations=f_star,
        density=density,
        velocity=velocity,
        pressure_drop=objectives.pressure_drop(density),
        heat_transfer=objectives.heat_transfer_proxy(chi, velocity, cell_volume),
    )


def convergence(
    chi: jax.Array, config: FlowConfig, inlet_velocity: Any = None
) -> tuple[jax.Array, jax.Array]:
    """Run the march without early exit and return its residual history.

    Args:
        chi: Solid fraction, ``(NX, NY, NZ)``.
        config: The flow configuration.
        inlet_velocity: ``(3,)`` inlet velocity, or ``None``.

    Returns:
        ``(f, history)`` -- the final populations and the per-step relative
        change recorded once per ``config.steady.check_every`` steps.
    """
    step_fn = step_for(config)
    theta = _theta(chi, config, inlet_velocity)
    f0 = initial_populations(config.shape, theta["inlet_velocity"])
    return residual_history(step_fn, theta, f0, config.steady)
