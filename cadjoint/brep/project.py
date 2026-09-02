"""The one projection kernel: Newton onto the intersection of 1, 2 or 3 zero sets.

Every position in a derived B-rep is the solution of the same problem at a
different arity:

- a **face** point solves ``f_a(x) = 0`` — one patch field,
- an **edge** point solves ``f_a(x) = f_b(x) = 0`` — a curve,
- a **vertex** solves ``f_a(x) = f_b(x) = f_c(x) = 0`` — a point.

So there is one kernel, :func:`project`, and it takes a stack of ``m ≤ 3``
fields.  Its Newton step is the minimum-norm Gauss-Newton correction
``x ← x − Jᵀ(JJᵀ)⁻¹ f(x)``, which at ``m = 1`` is exactly the step
:func:`cadjoint.fem.motion.project_points` takes (``f ∇f / |∇f|²``) and at
``m = 2`` exactly the step the viewer's seam projection takes.  The three
arities were three implementations; they are one here.

**The adjoint is the implicit-function theorem, not the unrolled loop.**  The
iteration runs on ``stop_gradient`` values and a :func:`jax.custom_vjp`
attaches the exact derivative of the *converged* map.  Writing the projection
as "displace the start point inside the normal space until every field
vanishes" — ``x* = x₀ + Jᵀλ`` with ``f(x*) = 0`` — and differentiating that
system gives

    ``dx* = P dx₀ − Jᵀ(JJᵀ)⁻¹ (∂f/∂θ) dθ``,  ``P = I − Jᵀ(JJᵀ)⁻¹J``

with ``P`` the tangential projector of the intersection.  The parameter term
is the pseudo-inverse pull every arity shares; the ``P dx₀`` term is what
makes a seed point's *tangential* placement irrelevant to the answer, which
is precisely the frozen-topology contract: the seed decides which branch of
the solution, never where on it.

**The guard is the repo's double-``where``.**  A rank test on the Gram matrix
``G = JJᵀ`` decides whether the intersection is transversal at all —
``λ_min(G) > 1e-2 · tr(G)/m`` (the viewer's seam test) together with
``tr(G) > 1e-8`` (:func:`~cadjoint.fem.motion.project_points`'s dead-gradient
floor, which at ``m = 1`` is ``|∇f|² > 1e-8``).  Where it fails the step is
suppressed in *both* passes: the point stays put and carries no parameter
derivative, rather than moving on float noise and amplifying a floored
denominator into the cotangent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "batched_residuals",
    "field_residuals",
    "project",
    "project_batched",
    "project_fields",
    "stacked_fields",
    "transversal",
]

#: Below this Gram trace the fields carry no usable direction at all.  At
#: ``m = 1`` the trace *is* ``|∇f|²``, so this is bit-for-bit the floor
#: :data:`cadjoint.fem.motion._MIN_GRADIENT_SQUARED` uses.
_MIN_TRACE = 1e-8

#: Relative smallest-eigenvalue floor for calling an intersection
#: transversal — the viewer's seam test (``_project_to_seam``), which rejects
#: tangent or coincident patches whose Gram matrix is rank deficient.
_RANK_TOLERANCE = 1e-2


def stacked_fields(fields: Sequence[Callable[[Array], Array]]) -> Callable[[Any, Array], Array]:
    """Turn concrete world-frame patch fields into a kernel ``field_fn``.

    Args:
        fields: 1 to 3 callables ``f(p)`` on points shaped ``(3,)``.

    Returns:
        ``field_fn(params, p) -> (m,)`` ignoring ``params`` — the shape
            :func:`project` expects for fields that close over concrete
            values instead of a traced parameter pytree.
    """
    field_list = list(fields)

    def field_fn(_params: Any, point: Array) -> Array:
        return jnp.stack([jnp.asarray(field(point)).reshape(()) for field in field_list])

    return field_fn


def _evaluate(field_fn: Callable[[Any, Array], Array], params: Any, x: Array) -> Array:
    single = lambda p: jnp.atleast_1d(jnp.asarray(field_fn(params, p)))  # noqa: E731
    return jax.vmap(single)(x)


def _system(
    field_fn: Callable[[Any, Array], Array], params: Any, x: Array
) -> tuple[Array, Array, Array]:
    """Values ``(n, m)``, Jacobian ``(n, m, 3)`` and Gram ``(n, m, m)`` at ``x``."""
    single = lambda p: jnp.atleast_1d(jnp.asarray(field_fn(params, p)))  # noqa: E731
    values = jax.vmap(single)(x)
    jacobian = jax.vmap(jax.jacfwd(single))(x)
    gram = jnp.einsum("nij,nkj->nik", jacobian, jacobian)
    return values, jacobian, gram


def _usable(gram: Array) -> Array:
    """Transversality mask ``(n,)`` for a batch of Gram matrices.

    A rank-deficient Gram means the patches are tangent or coincident and
    there is no transversal intersection to project onto; a vanishing trace
    means no field carries a usable gradient at all.
    """
    arity = gram.shape[-1]
    trace = jnp.trace(gram, axis1=-2, axis2=-1)
    smallest = jnp.linalg.eigvalsh(gram)[..., 0]
    return (smallest > _RANK_TOLERANCE * trace / arity) & (trace > _MIN_TRACE)


def _solve_masked(gram: Array, rhs: Array, usable: Array) -> Array:
    """``G⁻¹ rhs`` where ``usable``, zero elsewhere, with a finite denominator.

    The inner ``where`` swaps the identity in for the singular blocks so the
    solve itself never sees them; the outer one zeroes the answer.  Merely
    flooring ``G`` would keep the forward value finite while leaving the
    Jacobian of the solve scaled by the floor — the failure mode
    :func:`cadjoint.fem.motion.project_points` documents.
    """
    identity = jnp.broadcast_to(jnp.eye(gram.shape[-1], dtype=gram.dtype), gram.shape)
    safe = jnp.where(usable[:, None, None], gram, identity)
    solved = jnp.linalg.solve(safe, rhs[..., None])[..., 0]
    return jnp.where(usable[:, None], solved, 0.0)


def _newton(
    field_fn: Callable[[Any, Array], Array],
    params: Any,
    start: Array,
    max_step: float,
    steps: int,
) -> tuple[Array, Array]:
    """Run the fixed-count Newton iteration; return ``(x, usable)``."""
    x = start
    usable = jnp.ones(start.shape[:1], dtype=bool)
    for _ in range(steps):
        values, jacobian, gram = _system(field_fn, params, x)
        usable = _usable(gram)
        multipliers = _solve_masked(gram, values, usable)
        x = x - jnp.einsum("nij,ni->nj", jacobian, multipliers)
        displacement = x - start
        # Guarded norm: at zero displacement a bare norm's gradient is 0/0
        # and the NaN would leak through minimum() (cf. project_points).
        squared = jnp.sum(displacement * displacement, axis=-1, keepdims=True)
        length = jnp.sqrt(jnp.maximum(squared, 1e-24))
        x = start + displacement * jnp.minimum(1.0, max_step / length)
    return x, usable


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def _projected(
    field_fn: Callable[[Any, Array], Array],
    params: Any,
    points: Array,
    max_step: float,
    steps: int,
) -> Array:
    x, _ = _newton(field_fn, params, points, max_step, steps)
    return x


def _projected_fwd(field_fn, params, points, max_step, steps):
    x, _ = _newton(
        field_fn, jax.lax.stop_gradient(params), jax.lax.stop_gradient(points), max_step, steps
    )
    return x, (params, x)


def _projected_bwd(field_fn, max_step, steps, residuals, cotangent):  # noqa: ARG001
    params, x = residuals

    def stacked(candidate: Any) -> Array:
        return _evaluate(field_fn, candidate, x)

    _values, pullback = jax.vjp(stacked, params)
    _, jacobian, gram = _system(field_fn, params, x)
    usable = _usable(gram)

    # ``J ḡ`` is the cotangent expressed in field space; ``G⁻¹`` turns it
    # into the multipliers of the IFT adjoint.
    projected_cotangent = jnp.einsum("nij,nj->ni", jacobian, cotangent)
    multipliers = _solve_masked(gram, projected_cotangent, usable)
    (params_bar,) = pullback(-multipliers)
    # ``P ḡ`` — the start point only influences the answer tangentially.
    correction = jnp.einsum("nij,ni->nj", jacobian, multipliers)
    points_bar = cotangent - jnp.where(usable[:, None], correction, 0.0)
    return params_bar, points_bar


_projected.defvjp(_projected_fwd, _projected_bwd)


def project(
    field_fn: Callable[[Any, Array], Array],
    params: Any,
    points: Array,
    *,
    max_step: float,
    steps: int = 8,
) -> Array:
    """Project points onto the common zero set of 1, 2 or 3 fields.

    Args:
        field_fn: ``field_fn(params, p) -> (m,)`` for a point shaped ``(3,)``
            and ``1 <= m <= 3``.  Fields that close over concrete values
            instead of a parameter pytree can be wrapped with
            :func:`stacked_fields`.
        params: Parameter pytree handed to ``field_fn``; the quantity the
            returned positions are differentiable with respect to.
        points: Seed positions shaped ``(n, 3)``.
        max_step: Maximum total displacement from the seed.
        steps: Newton iterations (fixed count, traceable).

    Returns:
        Projected positions shaped ``(n, 3)``.  Differentiable with respect
            to ``params`` and ``points`` through the implicit-function
            adjoint described in the module docstring; points at a
            non-transversal intersection stay at their seed and carry no
            parameter derivative.
    """
    array = jnp.asarray(points)
    if array.ndim != 2 or array.shape[-1] != 3:
        raise ValueError(f"points must be shaped (n, 3); got {array.shape}.")
    if not max_step > 0:
        raise ValueError("max_step must be positive.")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    return _projected(field_fn, params, array, float(max_step), int(steps))


def project_fields(
    fields: Sequence[Callable[[Array], Array]],
    points: Array,
    *,
    max_step: float,
    steps: int = 8,
) -> np.ndarray:
    """Concrete projection onto 1-3 world-frame patch fields.

    The forward-only convenience wrapper around :func:`project` for fields
    that already close over their parameter values (the graph's own seeds).

    Args:
        fields: 1 to 3 callables ``f(p)``.
        points: Seed positions shaped ``(n, 3)``.
        max_step: Maximum total displacement from the seed.
        steps: Newton iterations.

    Returns:
        Projected positions as a float64 NumPy array shaped ``(n, 3)``.
    """
    if not 1 <= len(fields) <= 3:
        raise ValueError(f"project takes 1 to 3 fields; got {len(fields)}.")
    seeds = jnp.asarray(np.asarray(points, dtype=np.float64).reshape(-1, 3), dtype=jnp.float32)
    result = project(stacked_fields(fields), (), seeds, max_step=max_step, steps=steps)
    return np.asarray(result, dtype=np.float64)


def field_residuals(fields: Sequence[Callable[[Array], Array]], points: Array) -> np.ndarray:
    """Largest ``|f_i|`` over the fields at each point — the acceptance test.

    A genuine intersection point lands on every field's zero set; a near
    miss between disjoint surfaces keeps a residual of order the gap.

    Args:
        fields: The patch fields the points are supposed to satisfy.
        points: Positions shaped ``(n, 3)``.

    Returns:
        Residual per point, shaped ``(n,)``.
    """
    probes = jnp.asarray(np.asarray(points, dtype=np.float64).reshape(-1, 3), dtype=jnp.float32)
    values = _evaluate(stacked_fields(fields), (), probes)
    return np.max(np.abs(np.asarray(values, dtype=np.float64)), axis=-1)


def transversal(fields: Sequence[Callable[[Array], Array]], points: Array) -> np.ndarray:
    """Whether the fields meet transversally at each point.

    Args:
        fields: The patch fields.
        points: Positions shaped ``(n, 3)``.

    Returns:
        Boolean mask shaped ``(n,)``; ``False`` marks tangent or coincident
            patches, where :func:`project` leaves the point at its seed.
    """
    probes = jnp.asarray(np.asarray(points, dtype=np.float64).reshape(-1, 3), dtype=jnp.float32)
    _values, _jacobian, gram = _system(stacked_fields(fields), (), probes)
    return np.asarray(_usable(gram))


def project_batched(
    fields: Sequence[Callable[[Array], Array]],
    members: np.ndarray,
    points: Array,
    *,
    max_step: float,
    steps: int = 8,
) -> np.ndarray:
    """Project many points onto *their own* subsets of a shared field table.

    Forward-only sibling of :func:`project` for the graph's own bookkeeping,
    where thousands of points each belong to a different one, two or three
    of a scene's patches.  Calling :func:`project` once per subset is what
    the naive reading does and it is ruinous: the cost of a JAX program in
    eager mode is per *call*, not per point (``research/performance.md`` §6.2
    measured one point costing as much as three hundred), and a scene has
    hundreds of distinct subsets.

    So evaluate every field at every point once per iteration and let each
    point gather its own rows out of the result — one program for the whole
    batch, exactly the trick
    :func:`cadjoint.viewer._edge_overlay._project_seam_groups` plays for seam
    groups.  Rows must all have the same arity; group by arity and call once
    per group.

    Args:
        fields: The scene's whole patch-field table.
        members: Indices into ``fields``, shaped ``(n, m)`` with ``m <= 3``.
        points: Seed positions shaped ``(n, 3)``.
        max_step: Maximum total displacement from the seed.
        steps: Newton iterations.

    Returns:
        Projected positions as a float64 NumPy array shaped ``(n, 3)``.
    """
    seeds = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    member_array = np.asarray(members, dtype=np.int32).reshape(seeds.shape[0], -1)
    arity = member_array.shape[1]
    if not 1 <= arity <= 3:
        raise ValueError(f"project takes 1 to 3 fields per point; got {arity}.")
    if seeds.shape[0] == 0:
        return seeds

    evaluators = [
        jax.vmap(jax.value_and_grad(lambda p, f=field: jnp.asarray(f(p)).reshape(())))
        for field in fields
    ]
    start = jnp.asarray(seeds, dtype=jnp.float32)
    picker = jnp.arange(start.shape[0])[:, None]
    member_ids = jnp.asarray(member_array)

    def gathered(x: Array) -> tuple[Array, Array]:
        values, gradients = zip(*(evaluate(x) for evaluate in evaluators))
        value = jnp.stack(values, axis=-1)[picker, member_ids]
        jacobian = jnp.stack(gradients, axis=1)[picker, member_ids]
        return value, jacobian

    x = start
    for _ in range(steps):
        value, jacobian = gathered(x)
        gram = jnp.einsum("nij,nkj->nik", jacobian, jacobian)
        usable = _usable(gram)
        multipliers = _solve_masked(gram, value, usable)
        x = x - jnp.einsum("nij,ni->nj", jacobian, multipliers)
        displacement = x - start
        length = jnp.sqrt(jnp.maximum(jnp.sum(displacement**2, axis=-1, keepdims=True), 1e-24))
        x = start + displacement * jnp.minimum(1.0, max_step / length)
    return np.asarray(x, dtype=np.float64)


def batched_residuals(
    fields: Sequence[Callable[[Array], Array]], members: np.ndarray, points: Array
) -> np.ndarray:
    """Largest ``|f_i|`` over each point's own field subset, in one program.

    Args:
        fields: The scene's whole patch-field table.
        members: Indices into ``fields``, shaped ``(n, m)``.
        points: Positions shaped ``(n, 3)``.

    Returns:
        Residual per point, shaped ``(n,)``.
    """
    probes = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    member_array = np.asarray(members, dtype=np.int32).reshape(probes.shape[0], -1)
    if probes.shape[0] == 0:
        return np.zeros(0)
    x = jnp.asarray(probes, dtype=jnp.float32)
    stacked = jnp.stack([jax.vmap(field)(x) for field in fields], axis=-1)
    picker = jnp.arange(x.shape[0])[:, None]
    selected = stacked[picker, jnp.asarray(member_array)]
    return np.max(np.abs(np.asarray(selected, dtype=np.float64)), axis=-1)
