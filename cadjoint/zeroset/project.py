"""The one projection: Newton onto the common zero set of one, two or three fields.

Every place that put a point *on* the model used to carry its own Newton
loop — the mesh node motion (one field), the edge overlay's seams (two or
three leaf fields at once), the cut-cell surface, the GPU solve.  They
are the same computation, and they disagreed only where it mattered: at a
crease, one field's Newton steps between the faces that meet there while
the multi-field one lands on their intersection.  This is that
computation, once, differentiable, and the CPU counterpart of
:class:`~cadjoint.zeroset.gpu.Projector`'s kernel.

Minimum-norm gauge: with the k field gradients as rows of J, the step is
``−Jᵀ (J Jᵀ)⁻¹ f``, the smallest move that cancels every residual.  The
Gram matrix is regularised at a scale relative to its trace so a near
tangency does not blow the step up, and a point whose fields are not
transversal at the start (the smallest Gram eigenvalue under a hundredth of
the mean) is refused: it stays where it was.  A displacement clamp keeps a
stray point from leaving its basin, the same clamp the meshes always had.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["classify", "group_by_incidence", "project", "project_table", "theta_array"]

#: Below this squared gradient a field says nothing about where its zero
#: set is; the point stays put and contributes no derivative there.
_MIN_GRADIENT_SQUARED = 1e-12


def project(
    fields: Sequence[Callable[[Any], Any]],
    points: Any,
    *,
    steps: int = 8,
    max_step: float | None = None,
    transversality: float = 1e-2,
) -> Any:
    """Newton-project ``points`` onto the common zero set of ``fields``.

    Args:
        fields: One to three scalar fields, each callable on a ``(3,)`` point
            (they are ``vmap``-ed here); a field may close over traced
            parameters, and the result is differentiable through them.
        points: ``(N, 3)`` starting points, inside the basin.
        steps: Newton iterations, fixed so the program is traceable.
        max_step: Total displacement clamp per point, or None for none.
        transversality: A point whose Gram matrix has an eigenvalue below
            this fraction of the mean at the start is refused — its fields
            meet tangentially there, and the intersection is not a point.

    Returns:
        The projected points as a JAX array, ``(N, 3)``; a refused point is
        returned where it started.
    """
    import jax
    import jax.numpy as jnp

    if not 1 <= len(fields) <= 3:
        raise ValueError(f"a point lies on one to three surfaces, not {len(fields)}")
    system = _system(fields)

    def run(start):
        return _iterate(system, start, len(fields), steps, max_step, transversality)

    # One compiled program, not the nine eager walks of the field this used
    # to dispatch primitive by primitive (one per Newton step plus the
    # transversality probe).  Fresh per call rather than cached: `fields`
    # are opaque closures, and one may capture either an outer trace's
    # tracers or a scene whose parameters are mutated in place between
    # calls — a cached program would hold the first as a dead tracer and
    # bake the second in as a stale constant.  `project_table`, the caller
    # that runs this in a loop, takes the design as an *argument* and so
    # can and does cache its program (:func:`_table_program`).
    return jax.jit(run)(jnp.asarray(points))


def _system(fields: Sequence[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """``x -> (values, jacobian, gram)`` for the fields at every point.

    Args:
        fields: One to three scalar fields, each callable on a ``(3,)``
            point; they are ``vmap``-ed here.

    Returns:
        A callable taking ``(N, 3)`` points to ``values`` ``(N, k)``,
            ``jacobian`` ``(N, k, 3)`` whose rows are the field gradients,
            and ``gram`` ``(N, k, k)``.
    """
    import jax
    import jax.numpy as jnp

    evaluators = [
        jax.vmap(jax.value_and_grad(lambda p, f=field: jnp.asarray(f(p)).reshape(())))
        for field in fields
    ]

    def system(x):
        values, gradients = zip(*(evaluate(x) for evaluate in evaluators))
        jacobian = jnp.stack(gradients, axis=1)  # (N, k, 3): rows are the gradients
        gram = jnp.einsum("nij,nkj->nik", jacobian, jacobian)
        return jnp.stack(values, axis=-1), jacobian, gram

    return system


def _iterate(
    system: Callable[[Any], Any],
    start: Any,
    count: int,
    steps: int,
    max_step: float | None,
    transversality: float,
) -> Any:
    """The Newton iteration itself — the one kernel, traced once.

    Split out of :func:`project` so the closure form and the node-table
    form (:func:`project_table`) run the *same* arithmetic rather than two
    copies that could drift apart.  The sweep is a
    :func:`jax.lax.fori_loop` with a static trip count, so the traced
    program holds one body instead of ``steps`` copies of it; a static
    count lowers to a ``scan``, which differentiates in both modes.

    Args:
        system: ``x -> (values, jacobian, gram)``, from :func:`_system`.
        start: Starting positions, ``(N, 3)``.
        count: Number of fields ``k`` (their Gram is ``(N, k, k)``).
        steps: Newton iterations.
        max_step: Total displacement clamp per point, or None for none.
        transversality: Refusal threshold on the smallest Gram eigenvalue.

    Returns:
        The projected points, ``(N, 3)``; a refused point is returned
            where it started.
    """
    import jax
    import jax.numpy as jnp

    eye = jnp.eye(count, dtype=start.dtype)
    _, _, gram0 = system(start)
    trace0 = jnp.trace(gram0, axis1=-2, axis2=-1)
    if count == 1:
        transversal = gram0[:, 0, 0] > _MIN_GRADIENT_SQUARED
    else:
        transversal = jnp.linalg.eigvalsh(jax.lax.stop_gradient(gram0))[..., 0] > (
            transversality * trace0 / count
        )

    def sweep(_iteration, x):
        residual, jacobian, gram = system(x)
        trace = jnp.trace(gram, axis1=-2, axis2=-1)
        # Relative regularisation: a unit-gradient Gram entry is O(1), and
        # the same epsilon has to be meaningful in float32 and float64.
        regularised = gram + (1e-4 * trace + 1e-12)[..., None, None] * eye
        multipliers = jnp.linalg.solve(regularised, residual[..., None])[..., 0]
        step = jnp.einsum("nij,ni->nj", jacobian, multipliers)
        # Every iteration, not only the first: a point can walk into a flat
        # spot of the field (the far interior of a difference, the centre of
        # a ball), where the linearisation says nothing and the step is
        # 0/0.  There it stays put, and contributes no derivative.
        usable = jax.lax.stop_gradient(trace) > _MIN_GRADIENT_SQUARED
        step = jnp.where(usable[:, None] & jnp.isfinite(step), step, 0.0)
        x = x - step
        if max_step is not None:
            displacement = x - start
            length = jnp.sqrt(
                jnp.maximum(jnp.sum(displacement * displacement, axis=-1, keepdims=True), 1e-24)
            )
            x = start + displacement * jnp.minimum(1.0, max_step / length)
        return x

    x = jax.lax.fori_loop(0, steps, sweep, start)
    return jnp.where(transversal[:, None], x, start)


def theta_array(model: Any, values: dict[str, Any]) -> Any:
    """The table's θ with ``values`` (by free-parameter name) written in, as a JAX array.

    The traced counterpart of :func:`cadjoint.zeroset.refresh.theta_from_values`:
    an entry the values name comes from them (and may be traced), any other
    keeps the table's own.
    """
    import jax.numpy as jnp

    slots: dict[str, list[int]] = {}
    for i, name in enumerate(model.names):
        slots.setdefault(name.split("[", 1)[0], []).append(i)
    entries = [jnp.asarray(v, dtype=jnp.result_type(float)) for v in model.theta]
    for name, value in values.items():
        indices = slots.get(name)
        if indices is None:
            continue
        flat = jnp.asarray(value).reshape(-1)
        for k, i in enumerate(indices):
            entries[i] = flat[k]
    return jnp.stack(entries) if entries else jnp.zeros(0)


def group_by_incidence(incidence: Sequence[Sequence[int]]) -> dict[tuple[int, ...], np.ndarray]:
    """Point indices grouped by the (sorted) set of surfaces they lie on."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, surfaces in enumerate(incidence):
        groups.setdefault(tuple(sorted(int(s) for s in surfaces)), []).append(index)
    return {key: np.asarray(rows, dtype=np.int64) for key, rows in groups.items()}


def project_table(
    model: Any,
    theta: Any,
    points: Any,
    incidence: Sequence[Sequence[int]],
    *,
    steps: int = 8,
    max_step: float | None = None,
) -> Any:
    """Project every point onto its census surfaces of a node table at ``theta``.

    The design derivative flows through ``theta``: the surfaces' fields are
    :func:`cadjoint.zeroset.evaluate.surfaces` closed over it.  Points on
    no surface stay where they are.

    Args:
        model: A :class:`~cadjoint.zeroset.table.Model`.
        theta: The design, an array in the table's parameter order.
        points: ``(N, 3)``.
        incidence: Per point, the ids of the census surfaces it lies on
            (one to three; empty leaves it fixed) — from :func:`classify`
            or :meth:`cadjoint.zeroset.gpu.Projector.classify`.
    """
    import jax.numpy as jnp

    groups = tuple((key, rows) for key, rows in group_by_incidence(incidence).items() if key)
    program = _table_program(model, groups, steps, max_step)
    return program(jnp.asarray(theta), jnp.asarray(points))


def _arity_classes(groups: tuple) -> list[tuple[int, np.ndarray, np.ndarray, list[int]]]:
    """The incidence groups collapsed to one class per surface count.

    Groups of the same arity differ only in data, so they are merged into
    one batch and the surface identity becomes a column index.  The rows of
    a class are its groups' rows concatenated, and they stay disjoint
    across classes because a point has one incidence, so writing the
    classes back one after another is the same as writing the groups back
    one after another.

    Args:
        groups: ``((surface ids, point rows), ...)``, non-empty keys only.

    Returns:
        ``(count, rows, columns, wanted)`` per arity present, ascending:
            ``rows`` the class's point indices ``(M,)``; ``wanted`` the
            census ids the class needs, ascending; ``columns`` ``(M, count)``
            the index into ``wanted`` of each point's surfaces, in the same
            ascending-id order the ungrouped solve used them in.
    """
    out = []
    for count in (1, 2, 3):
        members = [(ids, rows) for ids, rows in groups if len(ids) == count]
        if not members:
            continue
        wanted = sorted({s for ids, _ in members for s in ids})
        column = {s: c for c, s in enumerate(wanted)}
        rows = np.concatenate([rows for _, rows in members])
        columns = np.concatenate(
            [np.tile([column[s] for s in ids], (len(rows), 1)) for ids, rows in members]
        )
        out.append((count, rows, columns.astype(np.int64), wanted))
    return out


def _selected(
    values: Callable[[Any, Any], Any], theta: Any, points: Any, columns: Any, count: int
) -> tuple[Any, Any, Any]:
    """``(values, jacobian, gram)`` for the surfaces each point was classified on.

    The same triple :func:`_system` builds, from a gathered evaluation
    instead of one closure per field: the class's surfaces are computed by
    a single walk of the table and each point takes the ``count`` columns
    that are its own.  Its gradients come from reverse passes over that one
    walk, seeded one column at a time, which is ``jax.value_and_grad`` of
    that column's field and nothing else — the other columns' seeds are
    exact zeros.  Reverse mode, and not the three forward tangents that
    would also serve, because reverse is what the per-field form used and
    the last bits have to agree with it; measured, forward mode is also the
    larger program (118 096 equations against 52 507).

    The ``count`` seeds are pushed through **one** ``vmap``-ed reverse pass
    rather than ``count`` separate ones.  They run the same arithmetic on
    the same graph and differ only in the seed, so batching them costs a
    leading axis and saves two copies of the backward pass — which decides
    whether XLA can compile the three-surface class at all.

    Args:
        values: ``(theta, points) -> (N, S)`` from
            :func:`~cadjoint.zeroset.evaluate.gathered`.
        theta: The design.
        points: ``(N, 3)``.
        columns: ``(N, count)`` column of ``values`` per point per slot.
        count: Surfaces per point, 1 to 3.

    Returns:
        ``values`` ``(N, count)``, ``jacobian`` ``(N, count, 3)`` whose rows
            are the field gradients, and ``gram`` ``(N, count, count)``.
    """
    import jax
    import jax.numpy as jnp

    selected, pull = jax.vjp(
        lambda p: jnp.take_along_axis(values(theta, p), columns, axis=1), points
    )
    seeds = jnp.broadcast_to(
        jnp.eye(count, dtype=selected.dtype)[:, None, :], (count, *selected.shape)
    )
    gradients = jax.vmap(lambda seed: pull(seed)[0])(seeds)  # (count, N, 3)
    jacobian = jnp.moveaxis(gradients, 0, 1)  # (N, count, 3): rows are the gradients
    return selected, jacobian, jnp.einsum("nij,nkj->nik", jacobian, jacobian)


#: Compiled :func:`project_table` programs, newest last.  Small: an entry
#: pins its model alive (see :func:`_table_program`), and a scene edit
#: lowers a new one, so this is a working set and not a registry.
_TABLE_PROGRAMS: OrderedDict[tuple, tuple[Any, Any]] = OrderedDict()
_TABLE_PROGRAMS_LIMIT = 8


def _table_program(
    model: Any, groups: tuple, steps: int, max_step: float | None
) -> Callable[[Any, Any], Any]:
    """The compiled ``(theta, points) -> points`` solve for one incidence structure.

    **One program per arity, not one per group.**  Groups that name the
    same number of surfaces run structurally identical arithmetic and
    differ only in *which* surfaces and which rows, so the 147 groups a
    real part produces collapse to the three classes — on one surface, on
    two, on three — and the surface choice becomes a column index into a
    gathered evaluation rather than traced structure.  Each class's
    surfaces are evaluated by a single :func:`~cadjoint.zeroset.evaluate.gathered`
    walk of the table (:func:`_selected`), so the traced program is the
    size of the table and **does not grow with the number of groups**.

    That is the whole point.  The unrolled form this replaces put a
    complete Newton kernel — a ``fori_loop`` carrying a linear solve, an
    einsum and an ``eigvalsh`` — in the jaxpr once per group, and on
    ``scenes/motor_shield.py`` (147 groups, 6 645 table nodes) that jaxpr
    reached 15.2 GB of RSS after 24 minutes of tracing without finishing,
    which killed every CI run in this repository's history.  The same
    structure here traces in seconds (``research/performance.md`` §18).

    **One jitted program per class, chained, rather than one for all three.**
    The classes are disjoint — a point has one incidence — so each reads and
    writes only its own rows and running them one after another is running
    them together.  Keeping them apart is what bounds what XLA is handed at
    once: its compile cost is steeply superlinear in program size (7.6 GB
    for the shield's one-surface class alone), so a single program holding
    all three costs far more than three costing one each, and each also
    caches, and is served from the persistent cache, on its own.  Under an
    outer ``jit`` they inline back into one, which is the caller's choice
    and not this function's.

    The design arrives as an **argument**, not as a captured constant, so
    the program is keyed on structure alone and every design reuses it —
    which is what makes :meth:`~cadjoint.fem.tetmesh.TetMesh.moved` cheap
    inside an optimizer, and what lets the four fixed-point passes in
    :func:`~cadjoint.fem.hexmesh.with_table` share one compile.

    Args:
        model: A :class:`~cadjoint.zeroset.table.Model`.
        groups: ``((surface ids, point rows), ...)``, non-empty keys only.
        steps: Newton iterations.
        max_step: Total displacement clamp per point, or None for none.

    Returns:
        A ``(theta, points) -> points`` callable over jitted per-class solves.
    """
    import jax
    import jax.numpy as jnp

    from cadjoint.zeroset.evaluate import gathered

    for ids, _rows in groups:
        if not 1 <= len(ids) <= 3:
            raise ValueError(f"a point lies on one to three surfaces, not {len(ids)}")
    key = (id(model), tuple((ids, rows.tobytes()) for ids, rows in groups), steps, max_step)
    cached = _TABLE_PROGRAMS.get(key)
    # The model is held in the entry, so its id cannot have been recycled
    # under us; the identity check is the belt to that braces.
    if cached is not None and cached[0] is model:
        _TABLE_PROGRAMS.move_to_end(key)
        return cached[1]

    def one_class(count: int, rows: Any, columns: Any, wanted: list[int]):
        values = gathered(model, wanted)

        def solve(theta, points):
            index, column = jnp.asarray(rows), jnp.asarray(columns)

            def system(y):
                return _selected(values, theta, y, column, count)

            moved = _iterate(system, points[index], count, steps, max_step, 1e-2)
            return points.at[index].set(moved)

        return jax.jit(solve)

    passes = [one_class(*c) for c in _arity_classes(groups)]

    def program(theta, points):
        for run in passes:
            points = run(theta, points)
        return points

    _TABLE_PROGRAMS[key] = (model, program)
    while len(_TABLE_PROGRAMS) > _TABLE_PROGRAMS_LIMIT:
        _TABLE_PROGRAMS.popitem(last=False)
    return program


def classify(
    model: Any, theta: Any, points: Any, *, tolerance: float, ownership: float = 1e-6
) -> list[list[int]]:
    """Which census surfaces each boundary point lies on, by ownership, on the CPU.

    The same rule as :meth:`cadjoint.zeroset.gpu.Projector.classify` and
    :class:`cadjoint.zeroset.refresh.Overlay`: a patch where its value is
    the model's, a band strictly inside its rim, a rim or fold only with a
    band that owns some point of the set.  Up to three, closest first.
    """
    import jax.numpy as jnp

    from cadjoint.zeroset.evaluate import census, field, surfaces

    theta = jnp.asarray(theta)
    pts = jnp.asarray(points)
    kinds = [kind for kind, _w, _x in census(model)]
    whole = np.asarray(field(model)(theta, pts))
    values = np.stack([np.asarray(f(theta, pts)) for _, f in surfaces(model)], axis=1)  # (N, S)
    band_of: list[int | None] = []
    band = None
    for kind in kinds:
        if kind == "band":
            band = len(band_of)
        band_of.append(band)
    close = np.abs(values) <= tolerance
    owns = np.abs(values - whole[:, None]) <= ownership
    candidate = np.zeros_like(close)
    for s, kind in enumerate(kinds):
        if kind == "patch":
            candidate[:, s] = close[:, s] & owns[:, s]
        elif kind == "band":
            inside = np.ones(len(pts), bool)
            if s + 1 < len(kinds) and kinds[s + 1] == "rim":
                inside = values[:, s + 1] > ownership
            candidate[:, s] = close[:, s] & owns[:, s] & inside
    real_bands = {s for s, kind in enumerate(kinds) if kind == "band" and candidate[:, s].any()}
    for s, kind in enumerate(kinds):
        if kind in ("rim", "fold") and band_of[s] in real_bands:
            candidate[:, s] = close[:, s] & owns[:, band_of[s]]
    out: list[list[int]] = []
    for n in range(len(pts)):
        ids = np.nonzero(candidate[n])[0]
        out.append([int(s) for s in ids[np.argsort(np.abs(values[n, ids]))][:3]])
    return out
