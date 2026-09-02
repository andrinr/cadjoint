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
    "trace_curves",
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
    stacked = stacked_fields(fields)
    # Compiled, for the reason :func:`project_batched` gives: this wrapper is
    # forward-only, and op-by-op the iteration re-traces the whole field stack
    # once per Newton step.
    run = jax.jit(lambda seed: project(stacked, (), seed, max_step=max_step, steps=steps))
    return np.asarray(run(seeds), dtype=np.float64)


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


def _restrict(
    fields: Sequence[Callable[[Array], Array]], members: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The field indices a batch actually gathers, and members renumbered onto them.

    A batch names its fields by index into the scene's whole patch table, but
    a given batch usually touches a fraction of it — and the cost of an eager
    JAX program is per *call*, so evaluating a field no point gathers is pure
    waste repeated once per Newton step.  Evaluate the used subset instead and
    renumber ``members`` onto it; the gather is identical either way.

    Args:
        fields: The scene's whole patch-field table.
        members: Indices into ``fields``, shaped ``(n, m)``.

    Returns:
        ``(used, renumbered)`` — the sorted distinct field indices, and
            ``members`` expressed as positions within ``used``.
    """
    used, inverse = np.unique(np.asarray(members), return_inverse=True)
    if used.size and (used.min() < 0 or used.max() >= len(fields)):
        raise ValueError("members index outside the field table.")
    return used, inverse.reshape(np.shape(members)).astype(np.int32)


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

    used, member_array = _restrict(fields, member_array)
    evaluators = [
        jax.vmap(jax.value_and_grad(lambda p, f=fields[index]: jnp.asarray(f(p)).reshape(())))
        for index in used
    ]
    start = jnp.asarray(seeds, dtype=jnp.float32)
    picker = jnp.arange(start.shape[0])[:, None]
    member_ids = jnp.asarray(member_array)

    def gathered(x: Array) -> tuple[Array, Array]:
        values, gradients = zip(*(evaluate(x) for evaluate in evaluators))
        value = jnp.stack(values, axis=-1)[picker, member_ids]
        jacobian = jnp.stack(gradients, axis=1)[picker, member_ids]
        return value, jacobian

    # The whole unrolled iteration is *one* compiled program.  Op-by-op, JAX
    # re-traces every ``vmap(value_and_grad(f))`` on every call, so a table of
    # fifty patches over four steps is two hundred traces of which a hundred
    # and fifty are redundant — and that tracing, not the arithmetic, is what
    # the viewer's cold extraction is made of (measured on the playground
    # starter: 5.7 s of the 19 s, down to 1.4 s here).
    @jax.jit
    def iterate(x: Array) -> Array:
        for _ in range(steps):
            value, jacobian = gathered(x)
            gram = jnp.einsum("nij,nkj->nik", jacobian, jacobian)
            usable = _usable(gram)
            multipliers = _solve_masked(gram, value, usable)
            x = x - jnp.einsum("nij,ni->nj", jacobian, multipliers)
            displacement = x - start
            length = jnp.sqrt(jnp.maximum(jnp.sum(displacement**2, axis=-1, keepdims=True), 1e-24))
            x = start + displacement * jnp.minimum(1.0, max_step / length)
        return x

    return np.asarray(iterate(start), dtype=np.float64)


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
    used, member_array = _restrict(fields, member_array)
    evaluators = [fields[index] for index in used]

    @jax.jit
    def worst(x: Array, member_ids: Array) -> Array:
        stacked = jnp.stack([jax.vmap(field)(x) for field in evaluators], axis=-1)
        picker = jnp.arange(x.shape[0])[:, None]
        return jnp.max(jnp.abs(stacked[picker, member_ids]), axis=-1)

    residual = worst(jnp.asarray(probes, dtype=jnp.float32), jnp.asarray(member_array))
    return np.asarray(residual, dtype=np.float64)


def trace_curves(
    fields: Sequence[Callable[[Array], Array]],
    members: np.ndarray,
    starts: np.ndarray,
    *,
    targets: np.ndarray,
    closed: np.ndarray,
    max_step: float,
    min_step: float,
    max_turn: float,
    tangent_floor: float,
    tolerance: float,
    max_points: int = 512,
) -> list[np.ndarray | None]:
    """Trace each curve where two patch zero sets cross, instead of sampling it.

    **The tangent is known in closed form.** Where ``f_a = f_b = 0`` the curve
    runs along ``∇f_a × ∇f_b``, so a point on it can be *continued*: step
    along that tangent (the predictor), then pull back onto both zero sets
    with the same Gauss-Newton corrector :func:`project_batched` uses.  The
    lattice is then only asked where a curve *starts* and which two patches
    it belongs to — never where its points are or what order they come in,
    which is what a set of scattered mesh-edge midpoints cannot say
    reliably.

    **The step is set by curvature, not by the grid.**  After each step the
    tangent has turned by some angle; the controller drives that towards
    ``max_turn`` and re-takes any step that turns by more than twice it, so
    ``2 · max_turn`` is the guarantee every accepted chord carries.  A
    straight edge runs at ``max_step`` and a rim of radius ``r`` settles at
    ``r · max_turn``, so a circle gets the same number of chords whether it
    is ten cells across or one.

    **A vanishing cross product means there is no edge.**  Parallel normals
    are tangent surfaces or a blend, where the two zero sets do not cross
    transversally and the "curve" is not defined; those are reported as
    ``None`` rather than pushed through, which is the honest answer and the
    same rank test :func:`_usable` applies one arity down.

    Every curve advances together, one batched program per step, for the
    reason :func:`project_batched` documents: the cost of an eager JAX
    program is per call, not per point.

    A step is re-taken, at half the length, when the tangent turns by more
    than twice ``max_turn`` or when the corrector has to pull the predicted
    point back by more than half the step — the latter meaning it landed on
    a different branch of the same pair's intersection rather than
    continuing this one.

    Args:
        fields: The scene's whole patch-field table.
        members: The two field indices per curve, shaped ``(n, 2)``.
        starts: A point on each curve, shaped ``(n, 3)``.
        targets: Where each open curve ends, shaped ``(n, 3)``; ignored
            wherever ``closed`` is set.
        closed: Whether each curve closes on itself, shaped ``(n,)``.
        max_step: Longest predictor step, in world units.
        min_step: Shortest predictor step; the trace gives up below it.
        max_turn: Target direction change per chord, in degrees.
        tangent_floor: Smallest ``|∇f_a × ∇f_b| / (|∇f_a| |∇f_b|)`` — the
            sine of the angle between the two normals — that still counts
            as a transversal crossing.
        tolerance: Largest ``|f|`` a traced point may keep.
        max_points: Cap on the points in any one curve.

    Returns:
        One polyline per curve, ordered along it and shaped ``(k, 3)``, or
            ``None`` where the curve is not traceable.
    """
    seeds = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    count = seeds.shape[0]
    if count == 0:
        return []
    member_array = np.asarray(members, dtype=np.int32).reshape(count, 2)
    ends = np.asarray(targets, dtype=np.float64).reshape(count, 3)
    is_closed = np.asarray(closed, dtype=bool).reshape(count)

    used, member_ids = _restrict(fields, member_array)
    evaluators = [
        jax.vmap(jax.value_and_grad(lambda p, f=fields[index]: jnp.asarray(f(p)).reshape(())))
        for index in used
    ]
    picker = jnp.arange(count)[:, None]
    ids = jnp.asarray(member_ids)

    def system(x: Array) -> tuple[Array, Array]:
        values, gradients = zip(*(evaluate(x) for evaluate in evaluators))
        return (
            jnp.stack(values, axis=-1)[picker, ids],
            jnp.stack(gradients, axis=1)[picker, ids],
        )

    @jax.jit
    def advance(points: Array, step: Array) -> tuple[Array, Array, Array, Array]:
        _values, jacobian = system(points)
        cross = jnp.cross(jacobian[:, 0], jacobian[:, 1])
        length = jnp.linalg.norm(cross, axis=-1)
        scale = jnp.linalg.norm(jacobian[:, 0], axis=-1) * jnp.linalg.norm(jacobian[:, 1], axis=-1)
        transversal = length > tangent_floor * jnp.maximum(scale, 1e-12)
        tangent = cross / jnp.maximum(length, 1e-12)[:, None]
        start = points + step[:, None] * tangent
        x = start
        for _ in range(4):
            value, jac = system(x)
            gram = jnp.einsum("nij,nkj->nik", jac, jac)
            multipliers = _solve_masked(gram, value, _usable(gram))
            x = x - jnp.einsum("nij,ni->nj", jac, multipliers)
            drift = x - start
            span = jnp.sqrt(jnp.maximum(jnp.sum(drift**2, axis=-1, keepdims=True), 1e-24))
            x = start + drift * jnp.minimum(1.0, jnp.abs(step)[:, None] / span)
        value, _jac = system(x)
        # How far the corrector had to pull the predicted point back.  A
        # correction of order the step itself means the corrector landed on
        # a *different* branch of the same pair's intersection rather than
        # continuing this one — the branch-jump guard of [AG90].
        drift = jnp.linalg.norm(x - start, axis=-1)
        return x, tangent, transversal, jnp.max(jnp.abs(value), axis=-1), drift

    turn_limit = np.radians(max_turn)
    current = jnp.asarray(seeds, dtype=jnp.float32)
    step = np.full(count, max_step, dtype=np.float64)

    # Which way to set off: along the tangent for a loop, towards the far
    # end for an open curve.
    _initial, tangent0, transversal0, _residual0, _drift0 = advance(
        current, jnp.zeros(count, jnp.float32)
    )
    tangent = np.asarray(tangent0, dtype=np.float64)
    heading = np.einsum("ij,ij->i", tangent, ends - seeds)
    sign = np.where(is_closed | (heading >= 0.0), 1.0, -1.0)

    alive = np.array(transversal0, dtype=bool)
    traced: list[list[np.ndarray]] = [[point] for point in seeds]
    failed = ~alive
    for _iteration in range(2 * max_points):
        if not alive.any():
            break
        moved, tangent_new, transversal, residual, drift = advance(
            current, jnp.asarray(sign * step, dtype=jnp.float32)
        )
        points = np.asarray(moved, dtype=np.float64)
        directions = np.asarray(tangent_new, dtype=np.float64)
        cosine = np.clip(np.einsum("ij,ij->i", directions, tangent), -1.0, 1.0)
        turned = np.arccos(cosine)

        # Overshot a bend, or the corrector pulled the point back so far
        # that it landed on another branch of the same pair: halve the step
        # and take it again.
        overshot = (turned > 2.0 * turn_limit) | (np.asarray(drift) > 0.5 * step)
        retry = alive & overshot & (step > min_step)
        step = np.where(retry, np.maximum(0.5 * step, min_step), step)
        accept = alive & ~retry
        if accept.any():
            for row in np.flatnonzero(accept):
                traced[row].append(points[row])
            current = jnp.where(
                jnp.asarray(accept)[:, None], jnp.asarray(points, jnp.float32), current
            )
            tangent = np.where(accept[:, None], directions, tangent)
            # Drive the turn towards the budget for the next step.
            gain = np.clip(turn_limit / np.maximum(turned, 1e-6), 0.5, 2.0)
            step = np.where(accept, np.clip(step * gain, min_step, max_step), step)

        lengths = np.asarray([len(rows) for rows in traced])
        home = np.linalg.norm(points - seeds, axis=1)
        away = np.linalg.norm(points - ends, axis=1)
        looped = accept & is_closed & (lengths >= 6) & (home <= np.abs(step))
        arrived = accept & ~is_closed & (away <= np.abs(step))
        broken = accept & (~np.asarray(transversal) | (np.asarray(residual) > tolerance))
        overrun = alive & (lengths >= max_points)
        for row in np.flatnonzero(arrived):
            traced[row].append(ends[row])
        failed |= broken | overrun
        alive &= ~(looped | arrived | broken | overrun)

    return [
        None if bad or len(rows) < 2 else np.asarray(rows, dtype=np.float64)
        for rows, bad in zip(traced, failed)
    ]
