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
    evaluators = [
        jax.vmap(jax.value_and_grad(lambda p, f=field: jnp.asarray(f(p)).reshape(())))
        for field in fields
    ]
    count = len(fields)
    start = jnp.asarray(points)
    eye = jnp.eye(count, dtype=start.dtype)

    def system(x):
        values, gradients = zip(*(evaluate(x) for evaluate in evaluators))
        jacobian = jnp.stack(gradients, axis=1)  # (N, k, 3): rows are the gradients
        gram = jnp.einsum("nij,nkj->nik", jacobian, jacobian)
        return jnp.stack(values, axis=-1), jacobian, gram

    _, _, gram0 = system(start)
    trace0 = jnp.trace(gram0, axis1=-2, axis2=-1)
    if count == 1:
        transversal = gram0[:, 0, 0] > _MIN_GRADIENT_SQUARED
    else:
        transversal = jnp.linalg.eigvalsh(jax.lax.stop_gradient(gram0))[..., 0] > (
            transversality * trace0 / count
        )
    x = start
    for _ in range(steps):
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

    from cadjoint.zeroset.evaluate import surfaces

    fields = [field for _, field in surfaces(model)]
    theta = jnp.asarray(theta)
    x = jnp.asarray(points)
    for key, rows in group_by_incidence(incidence).items():
        if not key:
            continue
        per_point = [lambda p, f=fields[s]: f(theta, p[None])[0] for s in key]
        x = x.at[jnp.asarray(rows)].set(project(per_point, x[rows], steps=steps, max_step=max_step))
    return x


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
