"""Conjugate heat transfer on the flow lattice: one energy equation, two phases.

The momentum solve next door leaves a velocity field on a fixed grid and a
solid fraction ``chi`` that says where the metal is.  This module solves the
steady energy equation on that same grid:

    div(k grad T)  =  (rho cp)_f  u . grad T  -  q

with the conductivity interpolated by ``chi`` between the air's and the
solid's.  Inside the metal ``u`` is zero (the Brinkman drag killed it), so
the equation degenerates to pure conduction there; in the channels ``k`` is
the air's and advection dominates.  **There is no interface condition to
impose**, and that is the whole reason for solving it this way: a
single-domain formulation with a variable coefficient satisfies continuity
of temperature (one unknown field, no jump to allow) and continuity of heat
flux (a conservative finite-volume flux shared by the two cells either side
of a face) *by construction*.  A two-domain formulation would have to find
the interface, mesh it, and match fluxes across it — the three things this
project is trying not to do.

**The discretisation is finite volume, not lattice Boltzmann.**  A second
distribution function would reuse the machinery in :mod:`cadjoint.flow.lbm`,
but the steady energy equation is *linear in T*, so marching it in
pseudo-time to a fixed point buys nothing and costs the stiffness of a
conductivity ratio of several thousand between metal and air.  One
matrix-free Krylov solve gets the same answer in one shot, and its adjoint
is :func:`jax.lax.custom_linear_solve`'s — the transposed solve, exact and
implicit, which composes with the flow's fixed-point adjoint without either
solver knowing about the other.

**The face coefficients are Patankar's.**  For each face, a conductance
``D = k_face / h`` and a convective flux ``F = (rho cp) u_face . n``; the
neighbour coefficient is

    a_nb = D * A(|F/D|) + max(-F, 0)

and the diagonal is ``a_P = sum(a_nb) + sum(F_out)``.  Two choices in there
carry weight:

* ``k_face`` is the **harmonic** mean of the two cell conductivities, not
  the arithmetic one.  Across a face separating air from aluminium the
  harmonic mean is the exact series resistance, so a step change in ``k``
  that lands on a face is reproduced to machine precision; the arithmetic
  mean would smear it over a cell and report a flux that is wrong by the
  conductivity ratio.  This is the discrete form of "continuity of heat
  flux at the interface", and ``tests/flow/test_energy.py`` checks it
  against the two-layer analytic answer.
* ``A(|P|)`` is by default the **exponential** blend ``|P|/(e^|P| - 1)``,
  which makes the scheme nodally exact for one-dimensional constant-
  coefficient advection-diffusion at any cell Peclet number.  Plain
  ``"upwind"`` (``A = 1``) is offered because it is what most codes do and
  because the difference between them *is* the false diffusion: upwind adds
  a numerical conductivity ``|u| h / 2`` along the flow, which at the cell
  Peclet numbers a duct at Re = 100 actually runs (about 3) more than
  doubles the air's effective conductivity.  The measurements are in
  ``research/flow-solver.md``.

**The linear solve is GMRES, and bicgstab was tried first.**  The operator
is non-symmetric whenever the flow moves, which rules out CG; bicgstab is
the obvious cheap alternative and it is the wrong one here.  JAX's bicgstab
carries no breakdown guard, and on a *pure advection* column -- an inlet
temperature at one end, a held temperature at the other, no source -- the
``rho`` recurrence collapses and the routine returns ``NaN`` at every
tolerance from 1e-8 to 1e-12, silently, with the same convergence flag it
returns on success.  Restarted GMRES on the same system reaches 4.2e-16 of
a dense reference.  On the conjugate problems where bicgstab does converge
it is no more accurate (7.2e-11 against GMRES's 9.0e-14 at a conductivity
ratio of 200) and no faster, so there is nothing to trade away.

**The convective flux is the mass flux ``rho u``, not the velocity.**  The
energy equation transports ``rho cp T``, so its flux is ``rho cp u`` and the
``rho`` is not a refinement -- it is the difference between a scheme that
conserves energy and one that does not.  Lattice Boltzmann conserves *mass*
exactly in its streaming step, so ``rho u`` (which, with Guo forcing and a
Brinkman drag, is exactly ``sum_i f_i c_i - alpha u / 2``, the quantity
:func:`~cadjoint.flow.lbm.macroscopic` is built to return) satisfies a
discrete continuity equation that ``u`` alone does not.  Measured on a
blocked duct, the global energy balance as a fraction of the injected
power:

========  ==========  ==========
lattice   flux = u    flux = rho u
========  ==========  ==========
8x14x8    3.6e-2      8.9e-16
10x18x10  9.4e-3      1.6e-6
========  ==========  ==========

Four to thirteen orders, for one multiply.  What is left with ``u`` alone
is not round-off; it is the lattice's compressibility error appearing as
energy that the duct creates or destroys.

The ``sum(F_out)`` term in the diagonal is not decoration either.  It is
what keeps the scheme transportive when the mass flux is not exactly
divergence-free -- and it is only divergence-free to the momentum solve's
own truncation.  Without it a cell with a small mass imbalance creates or
destroys energy in proportion.

**Units.**  Lattice units throughout, matching the momentum solve: one cell,
one step, and ``(rho cp)`` of the *fluid* equal to one.  The fluid
conductivity is then numerically the fluid's thermal diffusivity
``alpha_f = nu / Pr``, and the solid enters through one dimensionless
number, ``conductivity_ratio = k_solid / k_fluid``.  Nothing else about the
solid's material matters at steady state: its heat capacity multiplies a
velocity that is zero.  Temperature is affine-free — only differences from
the inlet mean anything — so ``inlet_temperature`` is conventionally 0 and
every read-out here is a rise above it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "SCHEMES",
    "EnergyConfig",
    "bulk_outlet_temperature",
    "conductivity",
    "energy_imbalance",
    "mean_temperature",
    "peak_temperature",
    "solve_temperature",
    "thermal_resistance",
]

#: The convection-diffusion blends :class:`EnergyConfig` accepts.
SCHEMES = ("exponential", "upwind")

#: Prandtl number of air at 300 K, the default working fluid.
AIR_PRANDTL = 0.71


@dataclass(frozen=True)
class EnergyConfig:
    """Static settings for one conjugate energy solve.

    Attributes:
        shape: ``(NX, NY, NZ)`` lattice, duct axis along ``+Y``, matching
            the momentum solve's grid exactly.
        fluid_diffusivity: The air's thermal diffusivity in lattice units,
            ``nu / Pr``.  Numerically also its conductivity, because
            ``(rho cp)_f`` is the unit of this system.
        conductivity_ratio: ``k_solid / k_fluid``.  Aluminium in air is
            about 8000; a ratio that large is physically right and
            numerically punishing (see the module docstring on
            conditioning), so a study that only needs "much better than
            air" is better served by a few hundred.
        inlet_temperature: Temperature carried in at the inlet plane.
        wall_temperature: Temperature of the duct's lateral walls, or
            ``None`` (the default) for adiabatic walls — the right choice
            for a duct that is not itself a heat exchanger.
        scheme: ``"exponential"`` or ``"upwind"``; see the module docstring.
        tol: Relative residual the linear solve stops at.
        max_steps: Iteration cap for the linear solve.
        restart: Krylov subspace size before GMRES restarts, and the one
            setting here that can silently return a wrong answer.
            Restarted GMRES *stagnates* when the subspace is too small for
            the operator's spectrum, and the conductivity ratio is what
            sets that spectrum: on a 14x26x14 duct, ``restart = 30``
            converges to 2.9e-12 at a ratio of 50 and stalls at a relative
            residual of **0.23** at a ratio of 200 -- reporting a peak
            temperature of 0.014 where the answer is 0.602, with no error
            and no warning.  60 converges at both.  It costs ``restart``
            stored fields, which is the reason not to simply set it very
            high on a large lattice.  :func:`energy_imbalance` is the
            check that catches a stall, and
            :meth:`~cadjoint.flow.FlowStudy.warnings` runs it.

    Raises:
        ValueError: On a non-positive diffusivity or conductivity ratio, or
            an unknown scheme.
    """

    shape: tuple[int, int, int]
    fluid_diffusivity: float
    conductivity_ratio: float = 200.0
    inlet_temperature: float = 0.0
    wall_temperature: float | None = None
    scheme: str = "exponential"
    tol: float = 1e-10
    max_steps: int = 4000
    restart: int = 60

    def __post_init__(self) -> None:
        if not self.fluid_diffusivity > 0.0:
            raise ValueError(
                f"fluid_diffusivity must be positive, got {self.fluid_diffusivity}. "
                "It is nu / Pr in lattice units; a zero would make the air a perfect "
                "insulator and the energy equation pure advection."
            )
        if not self.conductivity_ratio > 0.0:
            raise ValueError(
                f"conductivity_ratio must be positive, got {self.conductivity_ratio}; "
                "it is k_solid / k_fluid."
            )
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}; got {self.scheme!r}.")

    @property
    def solid_conductivity(self) -> float:
        """The solid's conductivity in the same lattice units."""
        return self.fluid_diffusivity * self.conductivity_ratio


def conductivity(chi: jax.Array, config: EnergyConfig) -> jax.Array:
    """Cell conductivity, interpolated linearly between air and solid.

    Linear in ``chi`` rather than harmonic or power-law: this is the
    material *mixture* inside a cell of the interface band, and the band is
    two cells wide by construction (:mod:`cadjoint.flow.domain`), so the
    interpolation rule only ever acts where the geometry genuinely is part
    metal.  The rule that decides how a *face* between two such cells
    conducts is the harmonic mean, and that one is not a choice — see the
    module docstring.

    Args:
        chi: Solid fraction in ``[0, 1]``, ``(NX, NY, NZ)``.
        config: The energy configuration.

    Returns:
        ``(NX, NY, NZ)`` conductivity in lattice units.
    """
    fluid = config.fluid_diffusivity
    return fluid + (config.solid_conductivity - fluid) * chi


def _blend(peclet: jax.Array, scheme: str) -> jax.Array:
    """Patankar's ``A(|P|)``, the diffusive weight at cell Peclet ``P``.

    Args:
        peclet: Face Peclet number ``F / D``, any shape.
        scheme: ``"exponential"`` or ``"upwind"``.

    Returns:
        The weight, same shape.
    """
    if scheme == "upwind":
        return jnp.ones_like(peclet)
    magnitude = jnp.abs(peclet)
    # Two `where`s rather than one: the branch not taken must still be
    # finite, or reverse mode propagates a NaN through it.
    safe = jnp.where(magnitude < 1e-6, 1.0, magnitude)
    return jnp.where(magnitude < 1e-6, 1.0 - 0.5 * magnitude, safe / jnp.expm1(safe))


def _face_coefficients(
    k: jax.Array, flux: jax.Array, scheme: str
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Neighbour coefficients and flux for a plane of faces.

    Args:
        k: Face conductance ``D``, positive.
        flux: Convective flux ``F`` through the face, positive along the
            axis (i.e. outward from the lower cell).
        scheme: The convection blend.

    Returns:
        ``(a_upper, a_lower, flux)`` — the coefficient the lower cell's
        equation gives its upper neighbour, the one the upper cell gives its
        lower neighbour, and ``flux`` unchanged for the diagonal sums.
    """
    diffusive = k * _blend(flux / k, scheme)
    return diffusive + jnp.maximum(-flux, 0.0), diffusive + jnp.maximum(flux, 0.0), flux


def _inlet_plane(shape: tuple[int, int, int]) -> np.ndarray:
    """The ``y = 0`` cells, where the inlet temperature is imposed."""
    plane = np.zeros(shape, dtype=bool)
    plane[:, 0, :] = True
    return plane


def _inlet_terms(
    k: jax.Array, mass_flux: jax.Array, config: EnergyConfig, active: np.ndarray
) -> tuple[jax.Array, jax.Array]:
    """The inlet plane's neighbour coefficient and its outward flux.

    The prescribed temperature sits half a cell upstream of the first cell
    centre, so the conductance is ``2k``.  The outward flux there is
    ``-rho u_y``, and the ``a_nb + F_out`` sum cancels the advective part
    exactly.

    Shared by :func:`_assemble` and :func:`energy_imbalance` rather than
    written twice, because the balance is only a check on the solver if it
    is built from the *same* coefficients the solver used.  Written twice,
    a change to one would show up as a physics result in the other.

    Args:
        k: Cell conductivity, ``(NX, NY, NZ)``.
        mass_flux: ``rho u``, ``(3, NX, NY, NZ)``.
        config: The energy configuration.
        active: Boolean mask of cells that are not duct wall.

    Returns:
        ``(a_inlet, inlet_flux)``, both zero away from the inlet plane.
    """
    mask = jnp.asarray(active & _inlet_plane(config.shape), dtype=k.dtype)
    flux = -mass_flux[1] * mask
    coefficient, _, _ = _face_coefficients(2.0 * k, flux, config.scheme)
    return coefficient * mask, flux


def _shift(array: jax.Array, axis: int, offset: int) -> jax.Array:
    """Shift along ``axis`` by ``offset`` cells, filling the vacated end with 0."""
    padded = jnp.zeros_like(array)
    if offset > 0:
        return padded.at[_slice(axis, slice(offset, None))].set(
            array[_slice(axis, slice(None, -offset))]
        )
    return padded.at[_slice(axis, slice(None, offset))].set(
        array[_slice(axis, slice(-offset, None))]
    )


def _slice(axis: int, key: slice) -> tuple:
    """An index tuple selecting ``key`` along ``axis`` and everything else."""
    return tuple(key if index == axis else slice(None) for index in range(3))


@dataclass(frozen=True)
class _Stencil:
    """The assembled seven-point operator, as plain arrays.

    Attributes:
        upper: Per-axis coefficient on the ``+axis`` neighbour, ``(3, ...)``.
        lower: Per-axis coefficient on the ``-axis`` neighbour, ``(3, ...)``.
        diagonal: ``a_P``, strictly positive on every solved cell.
        rhs: The right-hand side, sources plus Dirichlet contributions.
        solved: Boolean mask of the cells this system actually solves for.
    """

    upper: tuple[jax.Array, jax.Array, jax.Array]
    lower: tuple[jax.Array, jax.Array, jax.Array]
    diagonal: jax.Array
    rhs: jax.Array
    solved: np.ndarray


def _assemble(
    chi: jax.Array,
    mass_flux: jax.Array,
    config: EnergyConfig,
    wall: np.ndarray,
    source: jax.Array,
    fixed_mask: np.ndarray,
    fixed_value: jax.Array,
) -> _Stencil:
    """Build the finite-volume operator for one design and one flow field.

    Every face of every active cell falls into exactly one of four cases:
    an interior face shared with another active cell, the inlet plane, the
    outlet plane, or a wall face (against a duct-wall cell or the lattice's
    own edge).  The first three carry flux; the fourth is adiabatic unless
    the configuration names a wall temperature.

    Args:
        chi: Solid fraction, ``(NX, NY, NZ)``.
        mass_flux: Converged ``rho u``, ``(3, NX, NY, NZ)``, lattice units.
        config: The energy configuration.
        wall: Boolean duct-wall mask, ``(NX, NY, NZ)``, concrete.
        source: Volumetric power per cell, ``(NX, NY, NZ)``.
        fixed_mask: Cells held at a prescribed temperature, concrete.
        fixed_value: The prescribed temperatures, ``(NX, NY, NZ)``.

    Returns:
        The assembled :class:`_Stencil`.
    """
    k = conductivity(chi, config)
    active = ~wall
    solved = active & ~fixed_mask

    diagonal = jnp.zeros(config.shape, dtype=k.dtype)
    rhs = jnp.where(jnp.asarray(active), source, 0.0)
    upper: list[jax.Array] = []
    lower: list[jax.Array] = []

    for axis in range(3):
        low = _slice(axis, slice(None, -1))
        high = _slice(axis, slice(1, None))
        # Harmonic mean of the two cell conductivities: the exact series
        # resistance across the face, and the reason a k-jump landing on a
        # face is reproduced rather than smeared.
        k_low, k_high = k[low], k[high]
        face_k = 2.0 * k_low * k_high / (k_low + k_high)
        face_flux = 0.5 * (mass_flux[axis][low] + mass_flux[axis][high])
        shared = jnp.asarray(active[low] & active[high], dtype=k.dtype)
        a_up, a_down, flux = _face_coefficients(face_k, face_flux, config.scheme)
        a_up, a_down, flux = a_up * shared, a_down * shared, flux * shared

        pad = jnp.zeros(config.shape, dtype=k.dtype)
        upper.append(pad.at[low].set(a_up))
        lower.append(pad.at[high].set(a_down))
        diagonal = diagonal + pad.at[low].set(a_up + flux) + pad.at[high].set(a_down - flux)

    inlet = _inlet_plane(config.shape)
    outlet = np.zeros(config.shape, dtype=bool)
    outlet[:, -1, :] = True

    a_inlet, inlet_flux = _inlet_terms(k, mass_flux, config, active)
    diagonal = diagonal + a_inlet + inlet_flux
    rhs = rhs + a_inlet * config.inlet_temperature

    # Outlet plane: zero diffusive gradient, and the convected temperature
    # is the cell's own whichever way the flow crosses the plane, so the
    # face contributes exactly its outward flux to the diagonal.  That
    # stays stable under backflow, which a naive outflow condition does not.
    diagonal = diagonal + mass_flux[1] * jnp.asarray(active & outlet, dtype=k.dtype)

    # Wall faces: every face of an active cell whose neighbour is a wall
    # cell or off the lattice, minus the inlet and outlet planes.
    wall_faces = jnp.zeros(config.shape, dtype=k.dtype)
    for axis in range(3):
        for offset in (1, -1):
            neighbour = np.roll(active, -offset, axis=axis)
            edge = np.zeros(config.shape, dtype=bool)
            edge[_slice(axis, slice(-1, None) if offset > 0 else slice(None, 1))] = True
            outside = active & (edge | ~neighbour)
            if axis == 1:
                outside &= ~(inlet | outlet)
            wall_faces = wall_faces + jnp.asarray(outside, dtype=k.dtype)
    if config.wall_temperature is not None:
        conductance = 2.0 * k * wall_faces
        diagonal = diagonal + conductance
        rhs = rhs + conductance * config.wall_temperature

    solved_f = jnp.asarray(solved, dtype=k.dtype)
    diagonal = jnp.where(jnp.asarray(solved), diagonal, jnp.ones_like(diagonal))
    rhs = jnp.where(jnp.asarray(fixed_mask), fixed_value, rhs * solved_f)
    return _Stencil(
        upper=tuple(coefficient * solved_f for coefficient in upper),
        lower=tuple(coefficient * solved_f for coefficient in lower),
        diagonal=diagonal,
        rhs=rhs,
        solved=solved,
    )


def _apply(stencil: _Stencil, temperature: jax.Array) -> jax.Array:
    """Apply the seven-point operator (identity on cells it does not solve)."""
    result = stencil.diagonal * temperature
    for axis in range(3):
        result = result - stencil.upper[axis] * _shift(temperature, axis, -1)
        result = result - stencil.lower[axis] * _shift(temperature, axis, 1)
    return result


def solve_temperature(
    chi: jax.Array,
    velocity: jax.Array,
    config: EnergyConfig,
    *,
    density: Any = 1.0,
    wall: np.ndarray | None = None,
    source: Any = 0.0,
    fixed_mask: np.ndarray | None = None,
    fixed_value: Any = 0.0,
) -> jax.Array:
    """Solve the conjugate energy equation on the lattice.

    Differentiable in ``chi``, ``velocity``, ``source`` and ``fixed_value``:
    the linear solve carries :func:`jax.lax.custom_linear_solve`'s implicit
    adjoint, so reverse mode is one transposed solve rather than a tape
    through the Krylov iterations.  That is what lets this compose with the
    momentum solve's fixed-point adjoint without either knowing the other
    exists.

    Args:
        chi: Solid fraction, ``(NX, NY, NZ)``.
        velocity: Velocity in lattice units, ``(3, NX, NY, NZ)``.  Pass
            zeros for a pure-conduction solve.
        config: The energy configuration.
        density: Density, ``(NX, NY, NZ)`` or a scalar.  Multiplied into
            ``velocity`` to form the convective flux; the default 1.0 is
            the incompressible reading, and a solve fed a lattice
            Boltzmann field should pass that field's own density so the
            flux is the one the momentum solve actually conserves (see the
            module docstring).
        wall: Boolean duct-wall mask; ``None`` builds
            :func:`~cadjoint.flow.lbm.duct_walls` for the shape.
        source: Volumetric power per cell — a scalar or an
            ``(NX, NY, NZ)`` array.
        fixed_mask: Cells held at a prescribed temperature (concrete
            boolean array), or ``None`` for none.
        fixed_value: The prescribed temperatures, scalar or
            ``(NX, NY, NZ)``.

    Returns:
        ``(NX, NY, NZ)`` temperature, zero on duct-wall cells (which the
        system does not solve for), or all-``NaN`` when ``velocity`` is not
        finite -- see the note in the body on why that is better than the
        zeros GMRES would otherwise return.

    Raises:
        ValueError: If ``chi`` or ``velocity`` does not match the
            configured shape.
    """
    from cadjoint.flow.lbm import duct_walls

    shape = tuple(config.shape)
    if tuple(chi.shape) != shape:
        raise ValueError(f"chi has shape {tuple(chi.shape)}, expected {shape}.")
    if tuple(velocity.shape) != (3, *shape):
        raise ValueError(f"velocity has shape {tuple(velocity.shape)}, expected {(3, *shape)}.")
    wall = duct_walls(shape) if wall is None else np.asarray(wall, dtype=bool)
    fixed = (
        np.zeros(shape, dtype=bool) if fixed_mask is None else np.asarray(fixed_mask, dtype=bool)
    )
    mass_flux = velocity * jnp.broadcast_to(jnp.asarray(density, dtype=chi.dtype), shape)
    stencil = _assemble(
        chi,
        mass_flux,
        config,
        wall,
        jnp.broadcast_to(jnp.asarray(source, dtype=chi.dtype), shape),
        fixed & ~wall,
        jnp.broadcast_to(jnp.asarray(fixed_value, dtype=chi.dtype), shape),
    )

    def operator(field: jax.Array) -> jax.Array:
        return _apply(stencil, field)

    def preconditioner(field: jax.Array) -> jax.Array:
        return field / stencil.diagonal

    # A diverged momentum march hands this function a velocity full of NaN,
    # and GMRES answers a NaN operator with its *initial guess* -- zeros --
    # under a success flag.  A study would then report a heat sink at
    # exactly ambient everywhere, which reads like a converged answer and is
    # not one.  Poisoning the result deliberately is the honest failure: a
    # NaN objective stops an optimizer, a plausible zero steers it.
    usable = jnp.all(jnp.isfinite(mass_flux))
    temperature, _ = jax.scipy.sparse.linalg.gmres(
        operator,
        stencil.rhs,
        M=preconditioner,
        tol=config.tol,
        atol=0.0,
        restart=config.restart,
        maxiter=max(1, config.max_steps // config.restart),
        solve_method="batched",
    )
    temperature = jnp.where(usable, temperature, jnp.nan)
    return jnp.where(jnp.asarray(wall), 0.0, temperature)


def peak_temperature(temperature: jax.Array, chi: jax.Array, threshold: float = 0.5) -> jax.Array:
    """The hottest cell in the solid.

    Restricted to the solid because the objective a designer means by "peak
    temperature" is the junction's, not a stagnant pocket of air's.  ``max``
    is only piecewise smooth in the design — its derivative is the
    derivative at whichever cell currently wins — which is the same bargain
    ``Optimization(metric="max")`` already makes on the FEM path.

    Args:
        temperature: ``(NX, NY, NZ)`` temperature field.
        chi: Solid fraction, ``(NX, NY, NZ)``.
        threshold: Solid fraction above which a cell counts as solid.

    Returns:
        Scalar peak temperature, or ``-inf`` if the design has no solid.
    """
    return jnp.max(jnp.where(chi > threshold, temperature, -jnp.inf))


def mean_temperature(temperature: jax.Array, chi: jax.Array) -> jax.Array:
    """Solid-fraction-weighted mean temperature: ``int chi T / int chi``.

    The smooth counterpart of :func:`peak_temperature`, and the one to
    check a gradient against — it has no argmax to jump between cells.

    Args:
        temperature: ``(NX, NY, NZ)`` temperature field.
        chi: Solid fraction, ``(NX, NY, NZ)``.

    Returns:
        Scalar mean temperature over the solid.
    """
    return jnp.sum(chi * temperature) / (jnp.sum(chi) + 1e-30)


def bulk_outlet_temperature(
    temperature: jax.Array,
    velocity: jax.Array,
    wall: np.ndarray | None = None,
    density: Any = 1.0,
) -> jax.Array:
    """Flux-weighted mean temperature leaving the duct.

    ``int u_y T dA / int u_y dA`` over the outlet plane — the mixing-cup
    temperature, which is what an energy balance closes against.

    Args:
        temperature: ``(NX, NY, NZ)`` temperature field.
        velocity: ``(3, NX, NY, NZ)`` velocity field.
        wall: Boolean duct-wall mask, or ``None`` to build the default.
        density: Density field or scalar, weighting the average by mass
            flux rather than by velocity.

    Returns:
        Scalar bulk temperature at the outlet.
    """
    from cadjoint.flow.lbm import duct_walls

    shape = tuple(temperature.shape)
    wall = duct_walls(shape) if wall is None else np.asarray(wall, dtype=bool)
    open_plane = jnp.asarray(~wall[:, -1, :], dtype=temperature.dtype)
    rho = jnp.broadcast_to(jnp.asarray(density, dtype=temperature.dtype), shape)
    flux = velocity[1][:, -1, :] * rho[:, -1, :] * open_plane
    return jnp.sum(flux * temperature[:, -1, :]) / (jnp.sum(flux) + 1e-30)


def thermal_resistance(
    temperature: jax.Array, chi: jax.Array, power: Any, inlet_temperature: float = 0.0
) -> jax.Array:
    """``(T_peak - T_inlet) / power`` — the number a data sheet quotes.

    Args:
        temperature: ``(NX, NY, NZ)`` temperature field.
        chi: Solid fraction, ``(NX, NY, NZ)``.
        power: Total power injected, in the same units as ``source``.
        inlet_temperature: The reference the rise is measured from.

    Returns:
        Scalar thermal resistance in lattice units.
    """
    return (peak_temperature(temperature, chi) - inlet_temperature) / power


def energy_imbalance(
    temperature: jax.Array,
    chi: jax.Array,
    velocity: jax.Array,
    power: Any,
    config: EnergyConfig,
    wall: np.ndarray | None = None,
    density: Any = 1.0,
) -> jax.Array:
    """What the steady solve fails to conserve, as a fraction of the power.

    At steady state every watt put into the solid must leave through a
    boundary.  This is the strongest check available without an analytic
    answer, because it is independent of the geometry and of the flow
    field's accuracy: summing the discrete equations over every solved cell
    makes each interior face cancel against itself *exactly* -- the
    coefficient ``P`` gives ``E`` and the flux ``P`` sends ``E`` telescope
    whatever the mass flux is doing -- so what remains is the inlet plane,
    the outlet plane, and the source.  A conservative scheme therefore
    satisfies this to the linear solver's tolerance for **any** design and
    any velocity field, and a residual above that means the assembly is
    wrong, not that the flow is.

    Both ends of the duct have to be counted in full, and that is the part
    it is easy to get wrong.  The outlet carries enthalpy out at the cell's
    own temperature.  The inlet does two things: it carries enthalpy *in*
    at the first cell's temperature, and it conducts heat back out
    upstream, through the same ``2k`` half-cell conductance the assembly
    uses.  Counting only "mass flux times inlet temperature" ignores both
    corrections and reports a couple of percent of spurious imbalance on a
    duct where the scheme is in fact conserving to round-off.

    Args:
        temperature: ``(NX, NY, NZ)`` temperature field.
        chi: Solid fraction, ``(NX, NY, NZ)`` -- the conductivity the inlet
            conductance is built from.
        velocity: ``(3, NX, NY, NZ)`` velocity field.
        power: Total power injected.
        config: The energy configuration.
        wall: Boolean duct-wall mask, or ``None`` to build the default.
        density: Density field or scalar, forming the mass flux.  Must be
            the one the solve was assembled with.

    Returns:
        Scalar ``(what leaves - what enters - power) / power``.
    """
    from cadjoint.flow.lbm import duct_walls

    shape = tuple(temperature.shape)
    wall = duct_walls(shape) if wall is None else np.asarray(wall, dtype=bool)
    active = ~wall
    rho = jnp.broadcast_to(jnp.asarray(density, dtype=temperature.dtype), shape)
    mass_flux = velocity * rho[None]

    a_inlet, inlet_flux = _inlet_terms(conductivity(chi, config), mass_flux, config, active)
    # Net heat leaving through the inlet plane: conduction upstream against
    # the prescribed temperature, plus the (negative) enthalpy the incoming
    # air carries.  `inlet_flux` is already the outward flux there.
    leaving_inlet = jnp.sum(
        a_inlet * (temperature - config.inlet_temperature) + inlet_flux * temperature
    )
    outlet_open = jnp.asarray(active[:, -1, :], dtype=temperature.dtype)
    leaving_outlet = jnp.sum(mass_flux[1][:, -1, :] * outlet_open * temperature[:, -1, :])
    return (leaving_inlet + leaving_outlet - power) / power
