"""Constraint solving via Optimistix, Newton projection, and Optax."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jax import Array

from cadjoint.constraints.residual import (
    _collect_constraints,
    build_residual_fn,
    compute_param_vector,
    pack_param_dict,
    unpack_param_vector,
)
from cadjoint.enums import ConstraintSolveMethod, ConstraintSolveMethodLike, listed, parse
from cadjoint.fluent import Fluent
from cadjoint.geometry.parameters import Parameter

_CAPTURED_SOLVES: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "cadjoint_captured_constraint_solves",
    default=None,
)


@contextmanager
def capture_constraint_solves() -> Iterator[list[dict[str, Any]]]:
    """Capture diagnostics from ``satisfy_constraints`` calls in this context."""
    reports: list[dict[str, Any]] = []
    token = _CAPTURED_SOLVES.set(reports)
    try:
        yield reports
    finally:
        _CAPTURED_SOLVES.reset(token)


def _loss(flat_fn, values: Array) -> Array:
    residual = flat_fn(values)
    return jnp.mean(jnp.square(residual))


def _newton_projection(flat_fn, values: Array, steps: int) -> tuple[Array, list[float]]:
    """Minimum-norm Newton corrections plus their residual loss history.

    The step is ``Δ = Jᵀ(JJᵀ)⁻¹ c(p)``, solved as a least-squares problem
    rather than an exact inverse. The two agree to rounding whenever ``JJᵀ``
    is invertible, and they differ in exactly the case that matters: a sketch
    carrying a *redundant* constraint — one implied by the others, such as a
    perpendicularity that a horizontal and a vertical already force — makes
    ``J`` rank-deficient and ``JJᵀ`` singular. ``jnp.linalg.solve`` answers
    that with NaN, which then propagates into every free parameter in the
    program and turns the whole model into NaN geometry, silently.

    Worse, it did so only *sometimes*: in float32 the redundancy is masked by
    roundoff and the solve succeeds, while the float64 the FEM path enables
    makes the singularity exact. A scene could therefore render correctly and
    become NaN the moment it was meshed.

    A redundant-but-consistent constraint set is a perfectly ordinary thing to
    draw and every CAD sketcher tolerates it, so the least-squares step is
    also the semantically right answer: it takes the minimum-norm correction
    on the constraints that are actually independent.
    """
    x = values
    losses = [float(_loss(flat_fn, x))]
    for _ in range(steps):
        residual = flat_fn(x)
        jacobian = jax.jacobian(flat_fn)(x)
        correction = jnp.linalg.lstsq(jacobian @ jacobian.T, residual)[0]
        x = x - jacobian.T @ correction
        losses.append(float(_loss(flat_fn, x)))
    return x, losses


def _gradient_projection(
    flat_fn,
    values: Array,
    steps: int,
    method: ConstraintSolveMethod,
) -> tuple[Array, list[float]]:
    """Minimize squared constraint residuals with an Optax optimizer."""
    optimizer = optax.adam(0.05) if method == ConstraintSolveMethod.ADAM else optax.sgd(0.15)
    state = optimizer.init(values)
    x = values
    losses = [float(_loss(flat_fn, x))]
    loss_and_grad = jax.value_and_grad(lambda current: _loss(flat_fn, current))
    for _ in range(steps):
        _, gradient = loss_and_grad(x)
        updates, state = optimizer.update(gradient, state, x)
        x = optax.apply_updates(x, updates)
        losses.append(float(_loss(flat_fn, x)))
    return x, losses


def solve_constraints(
    sdf: Fluent,
    *,
    tol: float = 1e-6,
    max_steps: int = 256,
) -> dict[str, Any]:
    """Solve geometric constraints attached to a scene's free parameters.

    Extracts free parameters from the SDF tree, discovers all constraints
    registered on them, checks that the system is exactly constrained, then
    runs Levenberg-Marquardt (via optimistix) to find parameter values that
    satisfy all constraints.

    Args:
        sdf: The SDF tree whose free parameters should be solved.
        tol: Convergence tolerance (used as both rtol and atol).
        max_steps: Maximum solver iterations.

    Returns:
        A free_params dict (same format as extract_parameters) with solved
        parameter values — ready to pass to functionalize.

    Raises:
        ValueError: If the scene is under- or over-constrained.
        RuntimeError: If the solver does not converge.

    Example:
        ```python
        solved_params = solve_constraints(scene)
        sdf_fn = functionalize(scene)(solved_params, fixed_params)
        ```
    """
    # Local import avoids a circular dependency: constraints.solve → cadjoint → constraints
    from cadjoint.extraction import extract_parameters

    free_params, _, metadata = extract_parameters(sdf)

    constraints = _collect_constraints(metadata)

    # DOF check
    total_dof = sum(p.value.size for p in metadata.values())
    constraint_dof = sum(c.dof_reduction() for c in constraints)
    remaining = total_dof - constraint_dof
    if remaining > 0:
        raise ValueError(
            f"Under-constrained: {remaining} DOF remaining. "
            f"({total_dof} parameter DOF, {constraint_dof} constraint equations)"
        )
    if remaining < 0:
        raise ValueError(f"Over-constrained by {-remaining} equations.")

    flat_fn = build_residual_fn(constraints, metadata)
    residual_fn = lambda x, _: flat_fn(x)  # noqa: E731
    x0 = compute_param_vector(metadata)

    solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
    sol = optx.least_squares(residual_fn, solver, x0, max_steps=max_steps)

    return unpack_param_vector(sol.value, metadata)


def project_to_manifold(
    free_params: dict[str, Array],
    metadata: dict[str, Parameter],
    *,
    steps: int = 1,
) -> dict[str, Array]:
    """Project free_params onto the constraint manifold via Newton correction(s).

    Applies ``steps`` Newton steps: ``Δ = Jᵀ(JJᵀ)⁻¹ c(p)``, ``p ← p − Δ``.
    One step is exact for linear constraints and first-order accurate for
    nonlinear ones (e.g. sphere). Increase ``steps`` for tighter satisfaction.

    Args:
        free_params: Current name-keyed parameter arrays (may be off-manifold).
        metadata: Name-keyed Parameter objects (carries constraint info).
        steps: Number of Newton corrections to apply (default 1).

    Returns:
        Projected ``dict[str, Array]`` satisfying constraints to first order.
    """
    constraints = _collect_constraints(metadata)
    if not constraints:
        return free_params

    flat_fn = build_residual_fn(constraints, metadata)
    x = pack_param_dict(free_params, metadata)

    x, _ = _newton_projection(flat_fn, x, steps)

    return unpack_param_vector(x, metadata)


def satisfy_constraints(
    root: Fluent,
    *,
    steps: int = 8,
    method: ConstraintSolveMethodLike = ConstraintSolveMethod.NEWTON,
) -> dict[str, Array]:
    """Project a construction/SDF tree onto its attached constraints in place.

    Unlike :func:`solve_constraints`, this is also useful for under-constrained
    sketches: unconstrained degrees of freedom stay near their current values
    while the selected method satisfies the relationships that do exist.

    The returned mapping is also applied to ``root`` so construction overlays
    and generated solids sharing those Parameter objects update together.

    Args:
        root: Construction or SDF tree whose attached constraints are solved.
        steps: Number of optimizer updates.
        method: A :class:`~cadjoint.enums.ConstraintSolveMethod` or its
            plain string spelling — ``"newton"`` for minimum-norm manifold
            projection, or ``"adam"``/``"sgd"`` for Optax residual
            minimization.

    Returns:
        The solved free-parameter mapping, after applying it to ``root``.
    """
    from cadjoint.extraction import apply_parameters, extract_parameters

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    method = parse(
        ConstraintSolveMethod, method, f"method must be one of: {listed(ConstraintSolveMethod)}"
    )

    free, _, metadata = extract_parameters(root)
    constraints = _collect_constraints(metadata)
    if constraints:
        flat_fn = build_residual_fn(constraints, metadata)
        initial = pack_param_dict(free, metadata)
        if method == ConstraintSolveMethod.NEWTON:
            values, losses = _newton_projection(flat_fn, initial, steps)
        else:
            values, losses = _gradient_projection(flat_fn, initial, steps, method)
        solved = unpack_param_vector(values, metadata)
    else:
        solved = free
        losses = [0.0]
    apply_parameters(root, solved)

    reports = _CAPTURED_SOLVES.get()
    if reports is not None:
        reports.append(
            {
                "target_id": id(root),
                "method": method,
                "iterations": steps,
                "losses": losses,
            }
        )
    return solved


def constraint_residuals(
    free_params: dict[str, Array],
    metadata: dict[str, Parameter],
) -> Array:
    """Evaluate the constraint residuals at free_params.

    Args:
        free_params: Name-keyed parameter arrays.
        metadata: Name-keyed Parameter objects (carries constraint info).

    Returns:
        Flat residual array (length = total constraint equations). Empty array
        if there are no constraints.
    """
    constraints = _collect_constraints(metadata)
    if not constraints:
        return jnp.array([])
    flat_fn = build_residual_fn(constraints, metadata)
    return flat_fn(pack_param_dict(free_params, metadata))


def project_bounds(
    free_params: dict[str, Array],
    metadata: dict[str, Parameter],
) -> dict[str, Array]:
    """Clamp free_params to per-parameter bounds declared in metadata.

    Reads the ``bounds`` field of each Parameter (set at construction time as
    ``Scalar(v, bounds=(lo, hi))`` or ``Vector(v, bounds=(lo, hi))``).
    Either bound may be ``None`` to leave that side unconstrained.

    Args:
        free_params: Current name-keyed parameter arrays.
        metadata: Name-keyed Parameter objects carrying ``bounds`` info.

    Returns:
        A new dict with each array clipped to its declared bounds.
        Parameters without bounds are returned unchanged.
    """
    result = {}
    for k, v in free_params.items():
        param = metadata.get(k)
        if param is not None and param.bounds is not None:
            lo, hi = param.bounds
            lo = -jnp.inf if lo is None else lo
            hi = jnp.inf if hi is None else hi
            result[k] = jnp.clip(v, lo, hi)
        else:
            result[k] = v
    return result


def make_bounds_projection(
    metadata: dict[str, Parameter],
) -> optax.GradientTransformationExtraArgs:
    """Return an optax transform that clamps params to their declared bounds.

    Reads ``bounds`` from each Parameter in metadata and enforces them after
    every gradient step.  Chain after the main optimizer:

        optimizer = optax.chain(
            optax.adamw(lr),
            make_bounds_projection(metadata),
        )

    Args:
        metadata: Name-keyed Parameter objects carrying ``bounds`` info.

    Returns:
        An optax.GradientTransformationExtraArgs that clamps updates.
        Requires ``params`` to be passed to ``update``.
    """
    # Pre-compute which params actually have bounds to avoid repeated checks
    bounded = {
        k: (
            -jnp.inf if p.bounds[0] is None else p.bounds[0],
            jnp.inf if p.bounds[1] is None else p.bounds[1],
        )
        for k, p in metadata.items()
        if p.bounds is not None
    }

    def init_fn(_params):
        return ()

    def update_fn(updates, state, params=None, **_):
        if params is None or not bounded:
            return updates, state
        new_params = optax.apply_updates(params, updates)
        projected = {
            k: jnp.clip(v, *bounded[k]) if k in bounded else v for k, v in new_params.items()
        }
        corrected = jax.tree.map(lambda a, b: b - a, params, projected)
        return corrected, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def make_manifold_projection(
    metadata: dict[str, Parameter],
    *,
    steps: int = 1,
) -> optax.GradientTransformationExtraArgs:
    """Return an optax transform that projects params onto the constraint manifold.

    Chain after an optimizer to enforce constraints after each gradient step:

        optimizer = optax.chain(optax.adam(0.05), make_manifold_projection(metadata))
        state = optimizer.init(free_params)
        updates, state = optimizer.update(grads, state, free_params)
        free_params = optax.apply_updates(free_params, updates)

    Args:
        metadata: Name-keyed Parameter objects (carries constraint info).
        steps: Number of Newton corrections to apply (default 1).

    Returns:
        An optax.GradientTransformationExtraArgs that projects updates onto the
        constraint manifold. Requires ``params`` to be passed to ``update``.
    """

    def init_fn(_params):
        return ()

    def update_fn(updates, state, params=None, **_):
        if params is None:
            return updates, state
        new_params = optax.apply_updates(params, updates)
        projected = project_to_manifold(new_params, metadata, steps=steps)
        corrected = jax.tree.map(lambda a, b: b - a, params, projected)
        return corrected, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
