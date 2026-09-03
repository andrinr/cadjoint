"""D3Q19 BGK lattice Boltzmann with a Brinkman drag term, in pure JAX.

One function, :func:`step`, advances the populations one lattice time; every
other function here is a piece of it exposed for testing.  Nothing in this
module loops in Python over time -- the caller composes ``step`` under
:func:`jax.lax.scan` (:mod:`cadjoint.flow.steady` does), so a thousand
iterations compile once rather than unrolling into a thousand copies.

**Why LBM rather than a pressure-projection Navier-Stokes solver.**  The
design enters this solver as a *smooth volume fraction*, not as a mesh.  A
lattice Boltzmann step is entirely local (collide) plus a shift (stream),
so a per-cell drag coefficient is the whole of the solid-fluid coupling:
there is no matrix to reassemble, no mesh to regenerate, and no
non-differentiable remeshing step between the design parameters and the
answer.  That is the property this prototype exists to exploit.

**The Brinkman term.**  Solid is not meshed away; it is *penalised*.  Each
cell carries a drag coefficient ``alpha = alpha_max * chi``, where ``chi``
runs smoothly from 0 in fluid to 1 in solid, and the momentum equation
gains a body force ``F = -alpha u``.  As ``alpha_max`` grows, the velocity
in the solid is driven towards zero and the penalised solution converges to
the no-slip one; because ``chi`` is a smooth function of the signed
distance, ``d(answer)/d(design)`` exists everywhere -- which a bounce-back
staircase boundary, being a discrete set membership, cannot offer.

The drag is applied *implicitly in the velocity*, which is what makes large
``alpha_max`` stable.  Guo's forcing scheme defines the fluid velocity as
``rho u = sum_i f_i c_i + F/2``; substituting ``F = -alpha u`` and solving
for ``u`` gives

    u = (sum_i f_i c_i) / (rho + alpha/2)

in closed form (:func:`macroscopic`).  No iteration, no explicit
``-alpha u`` update that would go unstable once ``alpha`` exceeds ``2/dt``,
and the expression is smooth in ``alpha`` -- hence in the design.  The
march stays finite to ``alpha_max = 5e4``, five orders above where the
solid stops leaking, so the drag is not what limits this solver.

What *is* limited is the field it multiplies.  ``alpha_max`` scales every
cell's drag, including whatever ``chi`` leaves in cells that are supposed
to be open, so a profile with tails turns the fluid porous exactly as fast
as it turns the solid solid -- and then the duct plugs, density climbs
without bound, and the march diverges.  That is a property of ``chi``, not
of the forcing, and :mod:`cadjoint.flow.domain` is where it is dealt with.

**Boundary conditions.**  Streaming is periodic (a ``roll``); the planes
that must not be periodic are overwritten afterwards.  The duct's lateral
walls are *halfway* bounce-back, and that word is not decoration.  Fullway
bounce-back returns a population to its origin over two time steps, which
puts an eigenvalue at exactly ``-1`` in the step operator: the march then
settles onto a period-2 limit cycle rather than a fixed point, and a solver
built on ``f* = T(f*)`` has nothing to converge to.  (Measured on a
12x20x12 duct: ``||T(f) - f|| / ||f|| = 4.1e-2`` while
``||T^2(f) - f|| / ||f|| = 4e-16``.)  Halfway bounce-back reflects within
one step, has a genuine fixed point, and places the wall halfway between
the last fluid node and the first solid one, which is second-order accurate
instead of first.  It is a gather through
:data:`~cadjoint.flow.lattice.OPP` and so differentiable like everything
else.  Bounce-back is used only for the *fixed* duct walls, never for the
design -- the design is always the penalisation field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint.flow.lattice import CS2, OPP, C, Q, W


def equilibrium(rho: jax.Array, u: jax.Array) -> jax.Array:
    """The discrete Maxwellian, expanded to second order in velocity.

    Args:
        rho: Density, ``(NX, NY, NZ)``.
        u: Velocity, ``(3, NX, NY, NZ)``.

    Returns:
        Equilibrium populations, ``(19, NX, NY, NZ)``.
    """
    c = jnp.asarray(C, dtype=rho.dtype)
    w = jnp.asarray(W, dtype=rho.dtype)
    # cu[q] = c_q . u, shaped (19, NX, NY, NZ).
    cu = jnp.tensordot(c, u, axes=(1, 0))
    usq = jnp.sum(u * u, axis=0)
    expansion = 1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * usq / CS2
    return w[:, None, None, None] * rho[None] * expansion


def macroscopic(f: jax.Array, alpha: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Density and Brinkman-corrected velocity from the populations.

    The velocity solves ``rho u = sum_i f_i c_i - alpha u / 2`` exactly, so
    the drag is implicit and stays stable for any ``alpha >= 0``.

    Args:
        f: Populations, ``(19, NX, NY, NZ)``.
        alpha: Per-cell drag coefficient, ``(NX, NY, NZ)``.

    Returns:
        ``(rho, u)`` with shapes ``(NX, NY, NZ)`` and ``(3, NX, NY, NZ)``.
    """
    c = jnp.asarray(C, dtype=f.dtype)
    rho = jnp.sum(f, axis=0)
    momentum = jnp.tensordot(c.T, f, axes=(1, 0))
    return rho, momentum / (rho + 0.5 * alpha)[None]


def _guo_force(u: jax.Array, force: jax.Array, omega: float) -> jax.Array:
    """Guo's forcing source term for a body force.

    Args:
        u: Velocity, ``(3, NX, NY, NZ)``.
        force: Body force density, ``(3, NX, NY, NZ)``.
        omega: BGK relaxation rate.

    Returns:
        The source term added after collision, ``(19, NX, NY, NZ)``.
    """
    c = jnp.asarray(C, dtype=u.dtype)
    w = jnp.asarray(W, dtype=u.dtype)
    cu = jnp.tensordot(c, u, axes=(1, 0))
    cf = jnp.tensordot(c, force, axes=(1, 0))
    uf = jnp.sum(u * force, axis=0)
    bracket = (cf - uf[None]) / CS2 + cu * cf / CS2**2
    return (1.0 - 0.5 * omega) * w[:, None, None, None] * bracket


def stream(f: jax.Array) -> jax.Array:
    """Shift each population one cell along its own velocity (periodic).

    Args:
        f: Post-collision populations, ``(19, NX, NY, NZ)``.

    Returns:
        Streamed populations, ``(19, NX, NY, NZ)``.
    """
    # A per-q roll over the three spatial axes.  Written as a Python loop
    # over the 19 static directions, which unrolls at trace time into 19
    # shifts -- not a traced loop.
    return jnp.stack(
        [jnp.roll(f[q], shift=tuple(int(v) for v in C[q]), axis=(0, 1, 2)) for q in range(Q)]
    )


def step(
    f: jax.Array,
    chi: jax.Array,
    inlet_velocity: jax.Array,
    *,
    omega: float,
    alpha_max: float,
    wall: np.ndarray,
    incoming: np.ndarray,
) -> jax.Array:
    """Advance the populations one lattice time.

    Collide with the Brinkman drag, bounce back off the duct walls, stream,
    then impose the inlet and outlet planes.  Flow runs along ``+Y``: the
    inlet is the ``y = 0`` plane and the outlet the ``y = NY-1`` plane.

    Args:
        f: Populations, ``(19, NX, NY, NZ)``.
        chi: Solid volume fraction in ``[0, 1]``, ``(NX, NY, NZ)``.  This is
            the only design-dependent input.
        inlet_velocity: Prescribed inlet velocity, ``(3,)`` in lattice units.
        omega: BGK relaxation rate, in ``(0, 2)``.
        alpha_max: Drag coefficient at ``chi = 1``.
        wall: Boolean duct-wall mask, ``(NX, NY, NZ)``.  Concrete, not
            traced -- see :func:`duct_walls`.
        incoming: Bounce-back mask from :func:`bounce_back_mask`,
            ``(19, NX, NY, NZ)``.

    Returns:
        The populations one step later, ``(19, NX, NY, NZ)``.
    """
    alpha = alpha_max * chi
    rho, u = macroscopic(f, alpha)
    force = -alpha[None] * u
    collided = f - omega * (f - equilibrium(rho, u)) + _guo_force(u, force, omega)
    # Wall nodes are inert: their populations are never read by a fluid
    # node (the bounce-back below replaces exactly the directions that
    # would have come from them), so pin them at rest to keep them bounded.
    collided = jnp.where(wall[None], equilibrium(jnp.ones_like(rho), jnp.zeros_like(u)), collided)
    streamed = stream(collided)
    # Halfway bounce-back: where a population would have arrived from a wall
    # node, it is instead this node's own post-collision opposite, reflected
    # off the wall face halfway between the two.
    streamed = jnp.where(incoming, collided[OPP], streamed)
    return _apply_inlet_outlet(streamed, inlet_velocity, alpha)


def _apply_inlet_outlet(f: jax.Array, inlet_velocity: jax.Array, alpha: jax.Array) -> jax.Array:
    """Overwrite the two non-periodic planes after streaming.

    Both planes use Guo's non-equilibrium extrapolation: take the neighbour
    plane's populations, swap in whichever macroscopic quantity is being
    prescribed, and carry the neighbour's *non-equilibrium* part across
    unchanged.  That keeps the viscous stress at the boundary instead of
    flattening it, which a pure equilibrium condition would.

    The inlet prescribes velocity and extrapolates density.  The outlet
    prescribes density (``rho = 1``) and extrapolates velocity, and that
    choice is load-bearing: with a velocity inlet and a zero-gradient
    outlet the mean density is an undamped free mode -- nothing in the
    problem fixes it -- and the march then spends thousands of steps
    drifting towards an inflated density instead of converging.  Anchoring
    pressure at one end removes the mode, and makes the pressure drop a
    reading against a fixed reference rather than against a moving one.

    Both planes are written across their whole cross-section, including
    the cells that belong to the duct wall.  That is harmless and left
    deliberately simple: :func:`step` pins every wall node to rest before
    streaming, and bounce-back replaces exactly the populations that would
    have arrived from one, so no fluid node ever reads a wall cell.  The
    only trace is cosmetic -- the velocity *reported* on the inlet plane's
    wall ring is the prescribed inlet speed rather than zero.

    Args:
        f: Streamed populations, ``(19, NX, NY, NZ)``.
        inlet_velocity: Prescribed inlet velocity, ``(3,)``.
        alpha: Per-cell drag coefficient, ``(NX, NY, NZ)``.

    Returns:
        Populations with both planes imposed, ``(19, NX, NY, NZ)``.
    """
    inlet_neighbour = f[:, :, 1:2, :]
    rho_inlet, u_inlet_neighbour = macroscopic(inlet_neighbour, alpha[:, 1:2, :])
    prescribed = jnp.broadcast_to(
        inlet_velocity.astype(f.dtype)[:, None, None, None], u_inlet_neighbour.shape
    )
    inlet = inlet_neighbour + (
        equilibrium(rho_inlet, prescribed) - equilibrium(rho_inlet, u_inlet_neighbour)
    )

    outlet_neighbour = f[:, :, -2:-1, :]
    rho_outlet_neighbour, u_outlet = macroscopic(outlet_neighbour, alpha[:, -2:-1, :])
    reference = jnp.ones_like(rho_outlet_neighbour)
    outlet = outlet_neighbour + (
        equilibrium(reference, u_outlet) - equilibrium(rho_outlet_neighbour, u_outlet)
    )

    return f.at[:, :, 0:1, :].set(inlet).at[:, :, -1:, :].set(outlet)


def duct_walls(shape: tuple[int, int, int]) -> np.ndarray:
    """The lateral walls of a duct whose axis is ``+Y``.

    The ``x`` and ``z`` extremes are solid; the ``y`` extremes are left open
    for the inlet and outlet.

    Built in NumPy, and that is deliberate rather than incidental: the mask
    is fixed geometry that nothing differentiates, and
    :func:`~cadjoint.flow.solver.step_for` caches it on the configuration.
    A ``jnp`` array built during someone's first (traced) call would be a
    *tracer*, and the cache would hand that tracer to every later call --
    which JAX reports as a leak.  A concrete array cannot leak.

    Args:
        shape: ``(NX, NY, NZ)`` grid shape.

    Returns:
        Boolean wall mask, ``(NX, NY, NZ)``.
    """
    nx, _, nz = shape
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = mask[nx - 1, :, :] = True
    mask[:, :, 0] = mask[:, :, nz - 1] = True
    return mask


def bounce_back_mask(wall: np.ndarray) -> np.ndarray:
    """Which arriving populations came from a wall node, per direction.

    ``incoming[q, x]`` is true when the node ``x - c_q`` -- the one whose
    population would stream into ``x`` along direction ``q`` -- is a wall.
    Those are precisely the populations halfway bounce-back replaces.

    In NumPy for the same reason as :func:`duct_walls`.

    Args:
        wall: Boolean wall mask, ``(NX, NY, NZ)``.

    Returns:
        Boolean mask, ``(19, NX, NY, NZ)``.
    """
    wall = np.asarray(wall)
    shifted = [np.roll(wall, shift=tuple(int(v) for v in C[q]), axis=(0, 1, 2)) for q in range(Q)]
    return np.stack(shifted) & ~wall[None]


def initial_populations(shape: tuple[int, int, int], inlet_velocity: jax.Array) -> jax.Array:
    """Populations at rest density moving uniformly at the inlet velocity.

    Args:
        shape: ``(NX, NY, NZ)`` grid shape.
        inlet_velocity: ``(3,)`` lattice velocity.

    Returns:
        Equilibrium populations, ``(19, NX, NY, NZ)``.
    """
    dtype = jnp.asarray(inlet_velocity).dtype
    rho = jnp.ones(shape, dtype=dtype)
    u = jnp.broadcast_to(jnp.asarray(inlet_velocity, dtype=dtype)[:, None, None, None], (3, *shape))
    return equilibrium(rho, u)
