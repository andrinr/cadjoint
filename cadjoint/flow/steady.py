"""Steady state by pseudo-time, and its gradient by implicit differentiation.

The forward problem is a fixed point: iterate the lattice Boltzmann step
until it stops moving, ``f* = T(f*; theta)``.  The gradient problem is where
the design decision lives, and there are two ways to take it.

**The tape.**  Differentiate the iteration as written.  Reverse mode then
stores every intermediate ``f``, so the memory is ``O(steps x cells x 19)``
-- at 64x32x32 that is 10 MB per stored step, and a converged run is
thousands of steps.  It also differentiates the *path* to the answer, so the
gradient depends on the initial guess and on how many iterations were spent,
which is not a property the answer itself has.

**The implicit function theorem.**  Differentiate the *equation the answer
satisfies* instead.  With ``R(f, theta) = T(f, theta) - f = 0`` and
``A = dT/df``, the converged state obeys ``(I - A) df*/dtheta = dT/dtheta``,
so for a scalar objective the reverse-mode rule is: solve the adjoint system

    (I - A)^T lambda = g,          g = dJ/df*

and then ``dJ/dtheta = lambda^T dT/dtheta``.  Memory is ``O(cells x 19)``
-- one state, not a trajectory -- independent of how long convergence took,
and the gradient is a property of the converged solution alone.  This module
implements the second, behind :func:`jax.custom_vjp`, and never builds
``A``: both the forward operator and the parameter pullback come from a
single :func:`jax.vjp` of one step.

Two adjoint solvers are offered because they fail differently.
``"gmres"`` is :func:`jax.scipy.sparse.linalg.gmres` on ``v -> v - A^T v``,
which converges fastest when it converges.  ``"fixed_point"`` is Richardson
iteration ``lambda <- g + A^T lambda``, which is the adjoint of the forward
iteration itself: it converges exactly when the forward does, at the same
rate, and cannot stall the way a restarted Krylov method can.  When they
disagree, the forward run had not actually converged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class SteadyOptions:
    """Convergence settings for the forward iteration and the adjoint solve.

    Attributes:
        max_steps: Cap on pseudo-time steps.  Reaching it is not an error;
            :func:`residual_history` is how a caller checks convergence.
        tol: Relative change in the populations, per step, below which the
            iteration stops.
        check_every: Steps run between residual checks.  The check costs a
            full reduction, so amortising it over a chunk is worth more than
            stopping a few steps earlier.
        adjoint_solver: ``"gmres"`` or ``"fixed_point"``.
        adjoint_tol: Convergence tolerance for the adjoint solve.
        adjoint_max_steps: Iteration cap for the adjoint solve.
        adjoint_restart: Krylov subspace size for ``"gmres"``.
    """

    max_steps: int = 20000
    tol: float = 1e-7
    check_every: int = 50
    adjoint_solver: str = "gmres"
    adjoint_tol: float = 1e-8
    adjoint_max_steps: int = 400
    adjoint_restart: int = 30


def _chunk(step_fn: Callable, theta: Any, f: jax.Array, length: int) -> jax.Array:
    """Run ``length`` steps under ``scan`` (compiled once, not unrolled).

    Args:
        step_fn: ``(f, theta) -> f``.
        theta: The differentiable parameters.
        f: Starting populations.
        length: Number of steps.

    Returns:
        The populations after ``length`` steps.
    """

    def body(state: jax.Array, _: Any) -> tuple[jax.Array, None]:
        return step_fn(state, theta), None

    out, _ = lax.scan(body, f, None, length=length)
    return out


def _relative_change(new: jax.Array, old: jax.Array) -> jax.Array:
    """Per-step relative L2 change, the convergence measure used throughout.

    Args:
        new: Populations after a chunk.
        old: Populations before it.

    Returns:
        Scalar relative change, normalised by the chunk length's worth of
        steps by the caller.
    """
    return jnp.linalg.norm(new - old) / (jnp.linalg.norm(old) + 1e-300)


def iterate_to_steady(
    step_fn: Callable, theta: Any, f0: jax.Array, options: SteadyOptions
) -> jax.Array:
    """Pseudo-time march to the fixed point, without building a tape.

    Args:
        step_fn: ``(f, theta) -> f``.
        theta: The differentiable parameters.
        f0: Initial populations.
        options: Convergence settings.

    Returns:
        The converged populations.
    """
    chunk = max(1, options.check_every)
    chunks = max(1, options.max_steps // chunk)

    def cond(state: tuple) -> jax.Array:
        _, index, change = state
        return jnp.logical_and(index < chunks, change > options.tol * chunk)

    def body(state: tuple) -> tuple:
        f, index, _ = state
        advanced = _chunk(step_fn, theta, f, chunk)
        return advanced, index + 1, _relative_change(advanced, f)

    initial = (f0, jnp.asarray(0), jnp.asarray(jnp.inf, dtype=f0.dtype))
    converged, _, _ = lax.while_loop(cond, body, initial)
    return converged


def residual_history(
    step_fn: Callable, theta: Any, f0: jax.Array, options: SteadyOptions
) -> tuple[jax.Array, jax.Array]:
    """The same march, reporting the residual after every chunk.

    Runs the full ``max_steps`` without early exit so the history has a
    fixed length and the whole thing stays one compiled computation.  This
    is a diagnostic, not the path gradients are taken through.

    Args:
        step_fn: ``(f, theta) -> f``.
        theta: The differentiable parameters.
        f0: Initial populations.
        options: Convergence settings.

    Returns:
        ``(f, history)`` -- the final populations and the ``(chunks,)``
        per-step relative change recorded after each chunk.
    """
    chunk = max(1, options.check_every)
    chunks = max(1, options.max_steps // chunk)

    def body(f: jax.Array, _: Any) -> tuple[jax.Array, jax.Array]:
        advanced = _chunk(step_fn, theta, f, chunk)
        return advanced, _relative_change(advanced, f) / chunk

    return lax.scan(body, f0, None, length=chunks)


def _solve_adjoint(
    operator: Callable[[jax.Array], jax.Array], rhs: jax.Array, options: SteadyOptions
) -> jax.Array:
    """Solve ``(I - A^T) lambda = rhs`` matrix-free.

    Args:
        operator: ``v -> v - A^T v``, one transposed step per call.
        rhs: The objective's cotangent on the converged state.
        options: Which solver, and how tightly.

    Returns:
        The adjoint state ``lambda``.

    Raises:
        ValueError: On an unknown ``adjoint_solver``.
    """
    if options.adjoint_solver == "gmres":
        solution, _ = jax.scipy.sparse.linalg.gmres(
            operator,
            rhs,
            tol=options.adjoint_tol,
            atol=0.0,
            restart=options.adjoint_restart,
            maxiter=max(1, options.adjoint_max_steps // options.adjoint_restart),
            solve_method="batched",
        )
        return solution
    if options.adjoint_solver == "fixed_point":
        # lambda <- rhs + A^T lambda, i.e. Richardson on the same system:
        # operator(v) = v - A^T v, so A^T v = v - operator(v).
        def body(state: tuple) -> tuple:
            current, index, _ = state
            updated = rhs + (current - operator(current))
            change = jnp.linalg.norm(updated - current) / (jnp.linalg.norm(updated) + 1e-300)
            return updated, index + 1, change

        def cond(state: tuple) -> jax.Array:
            _, index, change = state
            return jnp.logical_and(index < options.adjoint_max_steps, change > options.adjoint_tol)

        initial = (rhs, jnp.asarray(0), jnp.asarray(jnp.inf, dtype=rhs.dtype))
        solution, _, _ = lax.while_loop(cond, body, initial)
        return solution
    raise ValueError(
        f"adjoint_solver must be 'gmres' or 'fixed_point'; got {options.adjoint_solver!r}."
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def steady_populations(
    step_fn: Callable, options: SteadyOptions, theta: Any, f0: jax.Array
) -> jax.Array:
    """The converged populations, differentiated through the fixed point.

    Forward, this is :func:`iterate_to_steady`.  Backward, it is the adjoint
    system of the *converged equation* -- the initial guess ``f0`` therefore
    receives a zero cotangent, which is the honest answer: move the guess
    and the converged state does not move.

    Args:
        step_fn: ``(f, theta) -> f``, one lattice time.  Static; passed as a
            ``nondiff_argnum``, so a fresh closure per call defeats caching.
        options: Convergence settings, static and hashable.
        theta: The differentiable parameters (any pytree ``step_fn`` accepts).
        f0: Initial populations.

    Returns:
        The converged populations, ``(19, NX, NY, NZ)``.
    """
    return iterate_to_steady(step_fn, theta, f0, options)


def _steady_fwd(
    step_fn: Callable, options: SteadyOptions, theta: Any, f0: jax.Array
) -> tuple[jax.Array, tuple]:
    """Forward pass: converge, and keep only the converged state."""
    f_star = iterate_to_steady(step_fn, theta, f0, options)
    return f_star, (theta, f_star)


def _steady_bwd(step_fn: Callable, options: SteadyOptions, residuals: tuple, g: jax.Array) -> tuple:
    """Backward pass: one adjoint solve, no trajectory.

    ``jax.vjp`` of a *single* step at the converged state gives both
    ``A^T`` (the state pullback) and ``dT/dtheta^T`` (the parameter
    pullback); the first drives the linear solve and the second turns its
    answer into the parameter gradient.
    """
    theta, f_star = residuals
    _, pullback = jax.vjp(lambda params, state: step_fn(state, params), theta, f_star)

    def operator(v: jax.Array) -> jax.Array:
        return v - pullback(v)[1]

    adjoint = _solve_adjoint(operator, g, options)
    theta_bar = pullback(adjoint)[0]
    return theta_bar, jnp.zeros_like(f_star)


steady_populations.defvjp(_steady_fwd, _steady_bwd)


def unrolled_populations(step_fn: Callable, theta: Any, f0: jax.Array, steps: int) -> jax.Array:
    """The same march with the tape left on, for cross-checking the adjoint.

    Reverse mode through this stores all ``steps`` intermediate states.  It
    exists to prove :func:`steady_populations` right at a size where the
    tape still fits, and to measure what it costs -- not to be used.

    Args:
        step_fn: ``(f, theta) -> f``.
        theta: The differentiable parameters.
        f0: Initial populations.
        steps: Exactly how many steps to run.

    Returns:
        The populations after ``steps`` steps.
    """
    return _chunk(step_fn, theta, f0, steps)
