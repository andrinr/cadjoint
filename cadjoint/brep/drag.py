"""Direct manipulation: drag a B-rep handle, solve for the design that puts it there.

A stored B-rep lets you drag a vertex because the vertex *is* the geometry.
Here the vertex is a consequence — the solution of ``f_a = f_b = f_c = 0`` for
three patch fields that belong to a sketch, an extrusion depth, a radius.  So
dragging is an inverse problem: find the parameter update that moves the
solution to where the pointer went, without breaking the sketch's own
constraints.

That inverse problem is small and linear at each step.  The handle position
is differentiable in the design parameters through
:func:`cadjoint.brep.project.project`'s implicit-function adjoint, so one
Gauss-Newton step solves

    ``[J_h; J_c] Δθ = [target − h(θ); −c(θ)]``

for the minimum-norm ``Δθ`` — the handle row asks for the motion, the
constraint rows keep the sketch legal, and least squares picks the smallest
change that does both.  That is the same
``Δ = Jᵀ(JJᵀ)⁻¹c`` manifold projection
:func:`cadjoint.constraints.solve.project_to_manifold` applies after every
optimizer step, with the drag stacked on top of it; a final projection
restores the constraints exactly.

**Topology is not solved for.**  A vertex exists only while its three patches
still bound the solid there.  Drag a corner far enough and it is cut away by
another operand, or two edges cross and the graph is a different graph — and
no parameter update can express that, because the topology was frozen at
extraction.  The test is exact and needs no re-extraction: solve the drag,
then ask whether the moved handle still lies on the *scene's* zero set.  A
handle that has gone inside the solid is a corner that no longer exists.  The
drag is reported, not applied.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cadjoint.brep.graph import BRep
from cadjoint.brep.project import project

__all__ = ["DragResult", "drag_handle", "handle_position", "patch_field_fn"]


@dataclass(frozen=True)
class DragResult:
    """What a drag did, and why.

    Attributes:
        handle: The handle that was dragged, as ``"vertex:3"`` or
            ``"edge:12@0.50"``.
        target: The requested position.
        achieved: Where the handle actually ended up.
        error: ``|achieved − target|``.
        parameters: The full solved free-parameter mapping.
        delta: Per-parameter change from the starting design.
        moved: ``(name, magnitude)`` for every parameter that moved, largest
            first — the answer to "what did my drag actually edit?".
        constraint_residual: Largest ``|c(θ)|`` after the solve.
        topology_changed: Whether the drag would take the handle off the
            solid's boundary, which the frozen graph cannot represent.
        applied: Whether ``parameters`` was written back into the scene.
        reason: A one-line explanation when the drag was refused.
        iterations: Gauss-Newton steps taken.
    """

    handle: str
    target: np.ndarray
    achieved: np.ndarray
    error: float
    parameters: dict[str, Any]
    delta: dict[str, np.ndarray]
    moved: list[tuple[str, float]]
    constraint_residual: float
    topology_changed: bool
    applied: bool
    reason: str
    iterations: int


@contextmanager
def _parameters_set(metadata: dict, values: dict):
    """Temporarily write ``values`` into a tree's ``Parameter`` objects.

    Patch fields close over parameter *values* when they are built (a box
    reads its half-extents, an extrusion its depth), so building them while
    traced values are in the tree is what makes them differentiable.  The
    originals go back on the way out, traced or not, so the scene a caller
    holds is never left with a tracer in it.
    """
    saved = {name: parameter.value for name, parameter in metadata.items()}
    try:
        for name, parameter in metadata.items():
            if name in values:
                parameter.value = values[name]
        yield
    finally:
        for name, parameter in metadata.items():
            parameter.value = saved[name]


def patch_field_fn(scene: Any, patch_indices: Sequence[int]):
    """A ``field_fn(params, point)`` for named patches of a scene.

    The graph's patch fields close over concrete parameter values, which is
    what makes them fast and what makes them useless for a drag.  This
    rebuilds the same fields with whatever values the caller passes, so the
    projection kernel can differentiate through them.

    Args:
        scene: Root SDF node.
        patch_indices: Global patch indices, as numbered by
            :func:`~cadjoint.brep.graph.extract_brep`.

    Returns:
        ``field_fn(params, p) -> (m,)`` where ``params`` is a free-parameter
            mapping (partial mappings are merged over the scene's current
            values).
    """
    from cadjoint.extraction import extract_parameters
    from cadjoint.meshing.patch_fields import scene_patch_fields

    baseline, _fixed, metadata = extract_parameters(scene)
    indices = list(patch_indices)

    def field_fn(params: dict, point: Array) -> Array:
        merged = {**baseline, **(params or {})}
        with _parameters_set(metadata, merged):
            decomposition = scene_patch_fields(scene)
            flat = [field for fields in decomposition.fields for field in fields]
            return jnp.stack([jnp.asarray(flat[index](point)).reshape(()) for index in indices])

    return field_fn


def handle_position(
    scene: Any,
    patch_indices: Sequence[int],
    seed: np.ndarray,
    params: dict,
    *,
    max_step: float,
    steps: int = 8,
) -> Array:
    """Where a handle sits for a given design, differentiably.

    Args:
        scene: Root SDF node.
        patch_indices: The one, two or three patches the handle belongs to.
        seed: Its position at the nominal design, shaped ``(3,)``.
        params: Free-parameter mapping to evaluate at.
        max_step: Displacement clamp for the projection.
        steps: Newton iterations.

    Returns:
        The solved position, shaped ``(3,)``.
    """
    field_fn = patch_field_fn(scene, patch_indices)
    seeds = jnp.asarray(np.asarray(seed, dtype=np.float64).reshape(1, 3), dtype=jnp.float32)
    return project(field_fn, params, seeds, max_step=max_step, steps=steps)[0]


def _patch_values(scene: Any, patch_indices: Sequence[int], point: np.ndarray) -> np.ndarray:
    """The patch fields' values at one point, for the tree's current design."""
    from cadjoint.meshing.patch_fields import scene_patch_fields

    probe = jnp.asarray(point, dtype=jnp.float32)
    flat = [field for fields in scene_patch_fields(scene).fields for field in fields]
    return np.asarray(
        [float(jnp.asarray(flat[index](probe)).reshape(())) for index in patch_indices]
    )


def _patch_jacobian(scene: Any, patch_indices: Sequence[int], point: np.ndarray) -> np.ndarray:
    """``∂f/∂x`` of the handle's patch fields, shaped ``(m, 3)``.

    Exact: the point is the only traced argument, so nothing discrete in the
    primitives' own construction is in the way.
    """
    from cadjoint.meshing.patch_fields import scene_patch_fields

    flat = [field for fields in scene_patch_fields(scene).fields for field in fields]

    def stacked(q: Array) -> Array:
        return jnp.stack([jnp.asarray(flat[index](q)).reshape(()) for index in patch_indices])

    return np.asarray(jax.jacfwd(stacked)(jnp.asarray(point, dtype=jnp.float32)), dtype=np.float64)


def _handle_jacobian(
    scene: Any,
    metadata: dict,
    names: list[str],
    values: dict,
    patch_indices: Sequence[int],
    point: np.ndarray,
    *,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """``∂x/∂θ`` of a handle, shaped ``(3, dof)``, by the implicit-function theorem.

    The two halves of ``dx = −Jᵀ(JJᵀ)⁻¹ (∂f/∂θ)`` are obtained differently
    and on purpose:

    - ``J = ∂f/∂x`` is autodiff, exact, with the *point* as the only traced
      argument.
    - ``∂f/∂θ`` is central differences on the field *values* at the fixed
      point — three scalars per perturbed design, no Newton iteration.

    Differencing the parameters rather than tracing them is not a shortcut,
    it is forced and it is right.  A primitive's patch decomposition makes
    discrete decisions while it is built — an :class:`ExtrudedPolygon` reads
    its profile's shoelace *sign* with ``float()`` to orient every wall — and
    those decisions cannot be traced, exactly because they are the frozen
    topology.  Rebuilding the fields at a perturbed design re-derives them
    concretely, which is what a design step is supposed to do; the expensive
    part, the projection itself, is not repeated at all.
    """
    jacobian = _patch_jacobian(scene, patch_indices, point)
    gram = jacobian @ jacobian.T
    columns = []
    for name in names:
        base = np.atleast_1d(np.asarray(values[name], dtype=np.float64)).ravel()
        shape = metadata[name].value.shape
        for component in range(base.size):
            step = epsilon * max(1.0, abs(float(base[component])))
            sensitivities = []
            for sign in (1.0, -1.0):
                shifted = base.copy()
                shifted[component] += sign * step
                perturbed = dict(values)
                perturbed[name] = jnp.asarray(shifted.reshape(shape) if shape else shifted[0])
                with _parameters_set(metadata, perturbed):
                    sensitivities.append(_patch_values(scene, patch_indices, point))
            columns.append((sensitivities[0] - sensitivities[1]) / (2.0 * step))
    if not columns:
        return np.zeros((3, 0))
    field_sensitivity = np.stack(columns, axis=1)
    multipliers = np.linalg.solve(gram, field_sensitivity)
    return -jacobian.T @ multipliers


def _pack(values: dict, names: list[str]) -> np.ndarray:
    return np.concatenate(
        [np.atleast_1d(np.asarray(values[name], dtype=np.float64)).ravel() for name in names]
    )


def _unpack(vector: np.ndarray, metadata: dict, names: list[str]) -> dict:
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name in names:
        size = metadata[name].value.size
        block = vector[offset : offset + size]
        shape = metadata[name].value.shape
        result[name] = block.reshape(shape) if shape else block[0]
        offset += size
    return result


def drag_handle(
    scene: Any,
    brep: BRep,
    handle: str,
    index: int,
    target: Array,
    *,
    position: float = 0.5,
    parameters: Sequence[str] | None = None,
    steps: int = 4,
    apply: bool = True,
    max_step: float | None = None,
    surface_tolerance: float | None = None,
) -> DragResult:
    """Move a B-rep vertex or edge point by solving for the design that fits.

    Args:
        scene: Root SDF node whose free parameters may move.
        brep: The extracted graph the handle belongs to.
        handle: ``"vertex"`` or ``"edge"``.
        index: Index into :attr:`BRep.vertices` or :attr:`BRep.edges`.
        target: Desired world position, shaped ``(3,)``.
        position: For an edge handle, the fractional station along its
            polyline to grab.
        parameters: Names of the free parameters the drag may edit; all of
            them when omitted.
        steps: Gauss-Newton iterations.
        apply: Write the solved parameters back into ``scene`` when the drag
            is accepted.  A drag that would change topology is never applied.
        max_step: Projection displacement clamp; defaults to the graph's own
            (half the cell diagonal).
        surface_tolerance: How far off the scene's zero set the moved handle
            may be before the drag is called a topology change; defaults to
            a tenth of the cell diagonal.

    Returns:
        The :class:`DragResult`.

    Raises:
        ValueError: If the handle is unknown, not analytic, or the scene has
            no free parameters to move.
    """
    from cadjoint.constraints.solve import constraint_residuals, project_to_manifold
    from cadjoint.extraction import apply_parameters, extract_parameters

    spacing = np.asarray(brep.grid.spacing, dtype=np.float64)
    cell_diagonal = float(np.linalg.norm(spacing))
    if max_step is None:
        max_step = 0.5 * cell_diagonal
    if surface_tolerance is None:
        surface_tolerance = 0.1 * cell_diagonal

    if handle == "vertex":
        element = brep.vertices[index]
        if not element.analytic:
            raise ValueError(
                f"Vertex {index} is not a clean triple point "
                f"(faces {element.faces}, patches {element.patches}); it has no analytic solve."
            )
        patch_indices = list(element.patches)
        seed = np.asarray(element.point, dtype=np.float64)
        label = f"vertex:{index}"
    elif handle == "edge":
        element = brep.edges[index]
        if not element.analytic or element.polyline.shape[0] == 0:
            raise ValueError(f"Edge {index} borders a blend face; it has no analytic solve.")
        patch_indices = [p for p in element.patches if p >= 0]
        station = int(round(np.clip(position, 0.0, 1.0) * (element.polyline.shape[0] - 1)))
        seed = np.asarray(element.polyline[station], dtype=np.float64)
        label = f"edge:{index}@{position:.2f}"
    else:
        raise ValueError(f"handle must be 'vertex' or 'edge'; got {handle!r}.")

    free, _fixed, metadata = extract_parameters(scene)
    names = list(metadata)
    if not names:
        raise ValueError("The scene has no free parameters; nothing to solve for.")
    movable = [name for name in names if parameters is None or name in parameters]
    if not movable:
        raise ValueError(f"None of {list(parameters or [])} is a free parameter of the scene.")
    columns = []
    offset = 0
    for name in names:
        size = metadata[name].value.size
        if name in movable:
            columns.extend(range(offset, offset + size))
        offset += size
    columns = np.asarray(columns, dtype=np.int64)

    target = np.asarray(target, dtype=np.float64).reshape(3)
    current = {name: jnp.asarray(free[name]) for name in names}

    def position_of(params: dict) -> Array:
        return handle_position(scene, patch_indices, seed, params, max_step=max_step)

    residual_fn = lambda vector: _constraint_vector(vector, metadata, names)  # noqa: E731

    achieved = np.asarray(position_of(current), dtype=np.float64)
    taken = 0
    for _ in range(steps):
        with _parameters_set(metadata, current):
            jacobian = _handle_jacobian(scene, metadata, names, current, patch_indices, achieved)
        packed = _pack(current, names)
        constraint = np.asarray(residual_fn(packed), dtype=np.float64)
        rows = [jacobian[:, columns]]
        right = [target - achieved]
        if constraint.size:
            constraint_jacobian = np.asarray(
                jax.jacobian(residual_fn)(jnp.asarray(packed)), dtype=np.float64
            )
            rows.append(constraint_jacobian[:, columns])
            right.append(-constraint)
        system = np.concatenate(rows, axis=0)
        rhs = np.concatenate(right)
        # Minimum-norm least squares: the smallest edit that both moves the
        # handle and keeps the sketch legal, exactly the step
        # ``_newton_projection`` takes with the drag row stacked on.
        step, *_ = np.linalg.lstsq(system, rhs, rcond=None)
        update = np.zeros(packed.shape[0])
        update[columns] = step
        current = {
            name: jnp.asarray(value)
            for name, value in _unpack(packed + update, metadata, names).items()
        }
        achieved = np.asarray(position_of(current), dtype=np.float64)
        taken += 1
        if float(np.linalg.norm(achieved - target)) < 1e-12:
            break

    # Restore the constraints exactly, then re-solve the handle on the result.
    solved = project_to_manifold(current, metadata, steps=2)
    achieved = np.asarray(position_of(solved), dtype=np.float64)
    constraint = np.asarray(constraint_residuals(solved, metadata), dtype=np.float64)

    with _parameters_set(metadata, solved):
        surface = float(
            abs(np.asarray(scene(jnp.asarray(achieved, dtype=jnp.float32)), dtype=np.float64))
        )
    topology_changed = surface > surface_tolerance
    reason = (
        f"the handle left the solid's boundary (|sdf| = {surface:.3g} > {surface_tolerance:.3g}); "
        "the frozen graph cannot represent the new topology — re-extract instead"
        if topology_changed
        else ""
    )

    delta = {
        name: np.asarray(solved[name], dtype=np.float64) - np.asarray(free[name], dtype=np.float64)
        for name in names
    }
    moved = sorted(
        ((name, float(np.linalg.norm(value))) for name, value in delta.items() if np.any(value)),
        key=lambda item: -item[1],
    )
    applied = bool(apply and not topology_changed)
    if applied:
        apply_parameters(scene, solved)

    return DragResult(
        handle=label,
        target=target,
        achieved=achieved,
        error=float(np.linalg.norm(achieved - target)),
        parameters={name: np.asarray(value) for name, value in solved.items()},
        delta=delta,
        moved=moved,
        constraint_residual=float(np.abs(constraint).max()) if constraint.size else 0.0,
        topology_changed=topology_changed,
        applied=applied,
        reason=reason,
        iterations=taken,
    )


def _constraint_vector(vector: Array, metadata: dict, names: list[str]) -> Array:
    """Constraint residuals as a function of the packed parameter vector."""
    from cadjoint.constraints.residual import _collect_constraints, build_residual_fn

    constraints = _collect_constraints(metadata)
    if not constraints:
        return jnp.zeros(0)
    ordered = {name: metadata[name] for name in names}
    return build_residual_fn(constraints, ordered)(vector)
