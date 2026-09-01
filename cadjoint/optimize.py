"""Declarative, code-first parameter optimizations.

Optimizations are first-class citizens of the scene program, exactly like
studies: declared in code (the source of truth), serializable for the
viewer via :meth:`Optimization.describe`, and runnable directly by scripts
and the playground via :meth:`Optimization.run`.  Constructing an
optimization inside a :func:`capture_optimizations` context registers it
automatically, so the compile worker can collect the optimizations a user
program declares — mirroring ``capture_studies`` in
:mod:`cadjoint.fem.study`.

The optimized variables are the FREE parameters of one scene object: the
named :class:`~cadjoint.geometry.parameters.Scalar` /
:class:`~cadjoint.geometry.parameters.Vector2` /
:class:`~cadjoint.geometry.parameters.Vector` values that
:func:`cadjoint.extract_parameters` collects from the target.  Every
gradient is a real reverse-mode derivative (:func:`jax.value_and_grad`);
there is no finite-difference path.  Two declaration forms exist, mutually
exclusive:

* **Objective form** (``objective=``/``of=``): an ordinary Python function
  ``params -> scalar`` over the free-parameter dict of ``of`` — the same
  signature the starter scene's ``material_volume`` has.

* **Study form** (``study=``/``metric=``): the objective is a declared
  :class:`~cadjoint.fem.study.ThermalStudy` /
  :class:`~cadjoint.fem.study.ElasticStudy` solved *inside* the
  differentiable loop.  Per step, the study's frozen-topology mesh keeps
  its connectivity while the node positions are recomputed through the
  traced SDF at the current parameters
  (:func:`~cadjoint.fem.hexmesh.recompute_points` /
  :func:`~cadjoint.fem.tetmesh.recompute_tet_points`), the study solves on
  those traced points, and the ``metric`` — ``"mean"``/``"max"`` of the
  result's objective scalar, or ``"compliance"`` (the work of the applied
  tractions, twice the strain energy) for elastic studies — plus
  ``regularizer_weight * regularizer(params)`` is minimized.  Topology is
  re-extracted at the current design every ``remesh_every`` steps and once
  more for the final evaluation, following the frozen-topology doctrine of
  ``examples/fem_bracket_optimization.py``.  The optimized free parameters
  come from the study's meshed domain when it declares one, and from the
  ``scene`` handed to :meth:`Optimization.run` otherwise.

Constrained sketches stay constrained: when the target's free parameters
carry declared constraints (fixed anchors, horizontal/vertical edges,
equal lengths, named distance dimensions), every optimizer update is
chained with a Newton projection back onto the constraint manifold —
:func:`cadjoint.constraints.make_manifold_projection`, the same machinery
``satisfy_constraints`` runs for interactive edits — so descent explores
only the sketch's genuine degrees of freedom.  A ``DistanceConstraint``'s
scalar target (``fin_height``-style driving dimensions) is a constant of
the projection, never an optimizer variable: the objective has no gradient
path into it, so re-dimensioning stays a source edit.

Example::

    sink_parameters, sink_fixed, _ = extract_parameters(sink)
    sink_sdf = functionalize(sink)

    def material_volume(parameters):
        sdf = sink_sdf(parameters, sink_fixed)
        return cell_volume * jnp.sum(jax.nn.sigmoid(-sdf(cells) / 0.03))

    optimize = Optimization(
        name="min-aluminum",
        objective=material_volume,
        of=sink,
        steps=25,
        learning_rate=0.03,
    )
    run = optimize.run()
    run.parameters["fin_depth"]  # optimized value

    cool = Optimization(
        name="cool-sink",
        study="sink-conduction",       # a declared ThermalStudy
        metric="mean",                 # minimize the mean temperature
        regularizer=material_volume,   # ... against material use
        regularizer_weight=0.5,
        steps=12,
        learning_rate=0.02,
    )
    run = cool.run(scene=scene)
    run.result.describe()              # final design solved on a fresh mesh
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["Optimization", "OptimizationRun", "capture_optimizations"]

METHODS = ("adam", "sgd")
METRICS = ("mean", "max", "compliance")
TRAJECTORY_LIMIT = 100

_CAPTURED_OPTIMIZATIONS: ContextVar[list[Optimization] | None] = ContextVar(
    "cadjoint_captured_optimizations",
    default=None,
)


@contextmanager
def capture_optimizations() -> Iterator[list[Optimization]]:
    """Collect every optimization constructed inside this context.

    Mirrors ``capture_studies``: the compile worker wraps user program
    execution in this context and receives the declared optimizations in
    construction order.
    """
    optimizations: list[Optimization] = []
    token = _CAPTURED_OPTIMIZATIONS.set(optimizations)
    try:
        yield optimizations
    finally:
        _CAPTURED_OPTIMIZATIONS.reset(token)


def _register(optimization: Optimization) -> None:
    captured = _CAPTURED_OPTIMIZATIONS.get()
    if captured is not None:
        captured.append(optimization)


def _plain(value: Any) -> float | list[float]:
    """One parameter value as JSON-ready plain numbers (float or [floats])."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return [float(component) for component in array.reshape(-1)]


def _serialize(params: dict[str, Any]) -> dict[str, float | list[float]]:
    return {name: _plain(value) for name, value in params.items()}


def _subsample(trajectory: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Thin a trajectory to at most *limit* evenly spaced entries.

    The first (step 0) and last entries always survive, so an animation
    player still replays the full parameter path end to end.
    """
    if len(trajectory) <= limit:
        return trajectory
    positions = np.linspace(0, len(trajectory) - 1, limit).round().astype(int)
    return [trajectory[position] for position in dict.fromkeys(positions.tolist())]


def _resolve_study(study: Any) -> Any:
    """Turn a ``study=`` argument into a study instance (resolving names).

    Mirrors ``_resolve_mesh_reference`` in :mod:`cadjoint.fem.study`: a
    name resolves against the studies captured in the same program, so a
    declaration can say ``study="sink-conduction"``.
    """
    from cadjoint.fem.study import _CAPTURED_STUDIES, ElasticStudy, ThermalStudy

    if isinstance(study, (ThermalStudy, ElasticStudy)):
        return study
    if isinstance(study, str):
        captured = _CAPTURED_STUDIES.get()
        declared = [candidate for candidate in (captured or []) if candidate.name == study]
        if len(declared) == 1:
            return declared[0]
        if len(declared) > 1:
            raise ValueError(f"The program declares more than one study named {study!r}.")
        names = ", ".join(repr(candidate.name) for candidate in (captured or [])) or "none"
        raise ValueError(
            f"No declared study named {study!r} (declared: {names}). "
            "Declare one before the optimization, or pass the study instance itself."
        )
    raise ValueError(
        "study must be a ThermalStudy/ElasticStudy or the name of a declared one, "
        f"got {type(study).__name__}."
    )


def _unresolvable_bc(study: Any, mesh: Any) -> str | None:
    """Describe the first boundary condition that fails to resolve on *mesh*.

    Selections are anchored in space, so a re-meshed design can move a
    loaded surface out of its selection.  Returns a human-readable
    ``"boundary condition <type> <selection> ..."`` summary for the first
    failing BC, or ``None`` when every selection resolves (node-valued
    conditions need nodes, area-integrated ones a complete boundary face).
    """
    from cadjoint.fem.hexmesh import faces_from_nodes
    from cadjoint.fem.study import HeatFlux, Traction
    from cadjoint.fem.tetmesh import TetMesh, tet_faces_from_nodes

    for bc in study.bcs:
        label = f"boundary condition {type(bc).__name__} {bc.nodes.describe()}"
        try:
            indices = bc.nodes.resolve(mesh)
        except ValueError:
            return f"{label} matched no surface nodes"
        if isinstance(bc, (HeatFlux, Traction)):
            if isinstance(mesh, TetMesh):
                spanned = int(tet_faces_from_nodes(mesh, indices).shape[0])
            else:
                spanned = int(faces_from_nodes(mesh, indices).nodes.shape[0])
            if spanned == 0:
                return f"{label} spans no complete boundary face"
    return None


def _compliance(study: Any, result: Any, mesh: Any, points: Any) -> Any:
    """Classical compliance: the work of the study's tractions, ``f . u``.

    Twice the strain energy under the applied loads — the strain-energy
    objective the flagship bracket example minimizes.  Differentiable in
    both the (possibly traced) ``points`` and the displacement, so
    ``jax.grad`` flows through the load surface as it moves with the
    design.  ``points=None`` evaluates on the mesh's own (concrete) nodes.
    """
    import jax.numpy as jnp

    from cadjoint.fem.hexmesh import faces_from_nodes
    from cadjoint.fem.study import Traction
    from cadjoint.fem.tetmesh import (
        TetMesh,
        load_work_quads,
        load_work_tris,
        tet_faces_from_nodes,
    )

    tractions = [bc for bc in study.bcs if isinstance(bc, Traction)]
    if not tractions:
        raise ValueError(
            f"metric='compliance' needs at least one Traction BC on study {study.name!r}: "
            "compliance is the work of the applied loads."
        )
    positions = jnp.asarray(mesh.points) if points is None else points
    displacement = result.displacement
    total = jnp.zeros(())
    for bc in tractions:
        indices = bc.nodes.resolve(mesh)
        vector = jnp.asarray(list(bc.vector), dtype=positions.dtype)
        if isinstance(mesh, TetMesh):
            faces = tet_faces_from_nodes(mesh, indices)
            if getattr(mesh, "edge_parents", None) is not None:
                from cadjoint.fem.tetmesh import load_work_tri6, tet10_face_midsides

                faces6 = np.concatenate([faces, tet10_face_midsides(mesh, faces)], axis=1)
                total = total + load_work_tri6(positions, displacement, faces6, vector)
            else:
                total = total + load_work_tris(positions, displacement, faces, vector)
        else:
            quads = faces_from_nodes(mesh, indices).nodes
            total = total + load_work_quads(positions, displacement, quads, vector)
    return total


@dataclass(frozen=True)
class OptimizationRun:
    """One finished optimization: its descent history and both endpoints.

    Attributes:
        name: The optimization's name.
        method: Optimizer actually used (``"adam"``/``"sgd"``, or
            ``"gradient-descent"`` when optax is unavailable).
        steps: Number of optimizer steps executed.
        learning_rate: Step size the run used.
        history: One record per step —
            ``{"step", "objective", "grad_norm"}`` — evaluated at the
            parameters *before* that step's update.
        trajectory: Parameter path for animation — one
            ``{"step", "objective", "parameters"}`` entry per step
            including step 0 (the initial state) and the final state,
            evenly subsampled to at most ``TRAJECTORY_LIMIT`` entries.
        parameters: Final free-parameter values (name → float | [floats]).
        initial: The values the run started from (same shape).
        result: For study-backed runs, the final design's concrete
            :class:`~cadjoint.fem.result.SimulationResult` — solved on a
            freshly extracted mesh, ready for
            ``describe()``/``nodal_scalar()``/rendering.  ``None`` for
            objective-form runs.
    """

    name: str
    method: str
    steps: int
    learning_rate: float
    history: list[dict[str, float]]
    trajectory: list[dict[str, Any]]
    parameters: dict[str, float | list[float]]
    initial: dict[str, float | list[float]]
    result: Any = field(default=None, repr=False, compare=False)

    @property
    def objective(self) -> float:
        """The final objective value (the last trajectory entry's)."""
        return float(self.trajectory[-1]["objective"])


@dataclass
class Optimization:
    """Declarative gradient-based optimization of a scene's free parameters.

    Attributes:
        name: Optimization identifier (unique within a scene program).
        objective: Callable ``(params: dict) -> scalar`` — a JAX-traceable
            function of the free-parameter dict, minimized by :meth:`run`
            (objective form; leave None for the study form).
        of: The scene object whose FREE parameters are optimized
            (anything :func:`cadjoint.extract_parameters` accepts;
            objective form only).
        study: A declared :class:`~cadjoint.fem.study.ThermalStudy` /
            :class:`~cadjoint.fem.study.ElasticStudy` (or its name) whose
            solved field the run minimizes (study form; mutually exclusive
            with ``objective``/``of``).
        metric: Study-form objective — ``"mean"`` or ``"max"`` of the
            result's objective scalar (temperature / displacement
            magnitude), or ``"compliance"`` (traction work, twice the
            strain energy; elastic studies only).
        regularizer: Optional callable ``(params: dict) -> scalar`` added
            to the study metric as ``regularizer_weight * regularizer``,
            e.g. a smoothed material volume (study form, keyword-only).
        regularizer_weight: Weight of the regularizer term (keyword-only).
        remesh_every: Study form: re-extract the frozen mesh topology at
            the current design every this many steps (0: never; default
            6).  In between, only node positions move — differentiably.
        gradient_path: Study form: how the design->points derivative is
            carried per step (keyword-only).  ``"direct"`` (default) is
            the validated frozen-topology path — node positions Newton
            re-projected onto the true SDF, solved in-process.
            ``"tesseract"`` runs the packaged two-tesseract chain instead
            (lattice samples -> mesher tesseract with its
            surface-interpolation VJP -> solver tesseract adjoint); it
            meshes the *trilinear interpolant* of the samples, so it can
            need a finer lattice than the direct path and its gradient
            carries only normal boundary motion (measured in
            ``research/tet-vs-hex.md``).  Requires the ``tesseract``
            extra.  The final reported result is always evaluated on the
            direct path.
        steps: Default number of optimizer steps (keyword-only).
        learning_rate: Optimizer step size (keyword-only).
        method: ``"adam"`` (default) or ``"sgd"`` (keyword-only).  Runs
            through optax; plain gradient descent when optax is missing.
    """

    name: str
    objective: Callable[[dict[str, Any]], Any] | None = None
    of: Any = None
    study: Any = None
    metric: str | None = None
    _: KW_ONLY
    regularizer: Callable[[dict[str, Any]], Any] | None = None
    regularizer_weight: float = 0.0
    remesh_every: int | None = None
    gradient_path: str = "direct"
    steps: int = 30
    learning_rate: float = 0.05
    method: str = "adam"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Optimization needs a non-empty name.")
        if self.study is not None:
            self._validate_study_form()
        else:
            self._validate_objective_form()
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer.")
        rate = self.learning_rate
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not rate > 0.0:
            raise ValueError("learning_rate must be a positive number.")
        self.learning_rate = float(rate)
        if self.method not in METHODS:
            raise ValueError(f"method must be one of: {', '.join(METHODS)}.")
        _register(self)

    def _validate_objective_form(self) -> None:
        if self.objective is None:
            raise ValueError(
                "Optimization needs either objective=/of= (a callable objective over a "
                "scene object) or study=/metric= (a declared study and a metric)."
            )
        if not callable(self.objective):
            raise ValueError(
                "objective must be a callable (params: dict) -> scalar, got "
                f"{type(self.objective).__name__}."
            )
        if not hasattr(self.of, "children"):
            raise ValueError(
                "of must be a scene object (a construction node or SDF) whose "
                f"parameters can be extracted, got {type(self.of).__name__}."
            )
        extras = [
            label
            for label, value in (
                ("metric", self.metric),
                ("regularizer", self.regularizer),
                ("remesh_every", self.remesh_every),
                ("regularizer_weight", self.regularizer_weight or None),
                ("gradient_path", None if self.gradient_path == "direct" else self.gradient_path),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                f"{', '.join(extras)} belong to the study form (study=/metric=); "
                "the objective=/of= form and the study form are mutually exclusive."
            )

    def _validate_study_form(self) -> None:
        if self.objective is not None or self.of is not None:
            raise ValueError(
                "objective=/of= and study=/metric= are mutually exclusive: declare "
                "either a Python objective over a scene object or a study metric."
            )
        self.study = _resolve_study(self.study)
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of: {', '.join(METRICS)} (got {self.metric!r}).")
        if self.metric == "compliance" and self._study_kind() != "elastic":
            raise ValueError(
                f"metric='compliance' needs an elastic study; {self.study.name!r} "
                f"is {self._study_kind()}."
            )
        if self.regularizer is not None and not callable(self.regularizer):
            raise ValueError(
                "regularizer must be a callable (params: dict) -> scalar, got "
                f"{type(self.regularizer).__name__}."
            )
        weight = self.regularizer_weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise ValueError("regularizer_weight must be a finite non-negative number.")
        self.regularizer_weight = float(weight)
        if self.remesh_every is None:
            self.remesh_every = 6
        if (
            isinstance(self.remesh_every, bool)
            or not isinstance(self.remesh_every, int)
            or self.remesh_every < 0
        ):
            raise ValueError("remesh_every must be a non-negative integer (0: never remesh).")
        if self.gradient_path not in ("direct", "tesseract"):
            raise ValueError(
                f"gradient_path must be 'direct' or 'tesseract' (got {self.gradient_path!r})."
            )

    def _study_kind(self) -> str:
        from cadjoint.fem.study import ThermalStudy

        return "thermal" if isinstance(self.study, ThermalStudy) else "elastic"

    def _study_target(self, scene: Any = None) -> Any:
        """The object whose free parameters a study-backed run optimizes.

        The study's meshed domain when it declares one (on its SimMesh or
        on the study itself); otherwise the ``scene`` the caller supplies —
        the same field the study would mesh at solve time.  ``None`` when
        neither is available (describe degrades, run raises).
        """
        study = self.study
        mesh = getattr(study, "mesh", None)
        domain = mesh.domain if mesh is not None and mesh.domain is not None else study.domain
        if domain is not None:
            return domain
        return scene

    def _free_parameters(self, target: Any) -> dict[str, Any]:
        from cadjoint.extraction import extract_parameters

        free, _, _ = extract_parameters(target)
        return free

    def describe(self, scene: Any = None) -> dict[str, Any]:
        """JSON-ready payload: everything the viewer needs to display it.

        Args:
            scene: Optional scene object; lets a study-backed optimization
                whose study meshes the whole scene report the scene's free
                parameters (the compile worker passes it).
        """
        if self.study is not None:
            target = self._study_target(scene)
            parameters = (
                list(self._free_parameters(target))
                if target is not None and hasattr(target, "children")
                else []
            )
            objective = f"{self.metric}({self.study.name})"
        else:
            parameters = list(self._free_parameters(self.of))
            objective = getattr(self.objective, "__name__", type(self.objective).__name__)
        return {
            "kind": "optimization",
            "name": self.name,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "method": self.method,
            "parameters": parameters,
            "objective": objective,
            "study": self.study.name if self.study is not None else None,
            "metric": self.metric,
            "remesh_every": self.remesh_every,
            "regularizer": (
                getattr(self.regularizer, "__name__", type(self.regularizer).__name__)
                if self.regularizer is not None
                else None
            ),
            "regularizer_weight": self.regularizer_weight,
        }

    def _updater(self, metadata: dict[str, Any] | None = None):
        """``(method, init, step)`` — optax when available, plain GD otherwise.

        When *metadata* (name-keyed Parameters, as returned by
        ``extract_parameters``) carries constrained parameters, every update
        is chained with a Newton projection back onto the constraint
        manifold (:func:`cadjoint.constraints.make_manifold_projection`) —
        the same projection ``satisfy_constraints`` applies to interactive
        edits — so descent steps respect the sketch's declared
        relationships.  Named driving dimensions (a ``DistanceConstraint``'s
        scalar target) are treated as constants by the constraint system:
        they steer the projection but are never optimizer variables.
        """
        constrained = False
        if metadata is not None:
            from cadjoint.constraints import constraint_residuals

            values = {name: parameter.value for name, parameter in metadata.items()}
            constrained = constraint_residuals(values, metadata).size > 0
        try:
            import optax
        except ImportError:
            import jax

            rate = self.learning_rate
            if constrained:
                from cadjoint.constraints import project_to_manifold

            def descend(params, grads, state):
                updated = jax.tree_util.tree_map(lambda p, g: p - rate * g, params, grads)
                if constrained:
                    updated = project_to_manifold(updated, metadata, steps=2)
                return updated, state

            return "gradient-descent", (lambda _params: None), descend

        transform = (
            optax.adam(self.learning_rate)
            if self.method == "adam"
            else optax.sgd(self.learning_rate)
        )
        if constrained:
            from cadjoint.constraints import make_manifold_projection

            transform = optax.chain(transform, make_manifold_projection(metadata, steps=2))

        def apply(params, grads, state):
            updates, state = transform.update(grads, state, params)
            return optax.apply_updates(params, updates), state

        return self.method, transform.init, apply

    def run(self, steps: int | None = None, callback=None, *, scene: Any = None) -> OptimizationRun:
        """Minimize the objective over the target's free parameters.

        Pure reverse-mode differentiation (:func:`jax.value_and_grad`)
        through the objective.  Objective-form runs never touch the scene
        object; study-backed runs restore the target's original parameter
        values before returning (topology refreezes write the candidate
        design into the target while extracting) — the returned run
        carries the optimized values either way.

        Args:
            steps: Number of optimizer steps (default: the declared
                ``steps``).
            callback: Optional ``callback(record)`` invoked with each
                history record as it is produced.
            scene: Study form only — the scene object the study meshes,
                required when neither the study nor its SimMesh declares a
                ``domain`` (keyword-only; ignored by the objective form).

        Returns:
            The finished :class:`OptimizationRun`.  Study-backed runs also
            carry the final design's concrete ``result``, solved on a
            freshly extracted mesh.

        Raises:
            ValueError: When the target has no free parameters, the
                objective (or its gradient) leaves the finite range, or a
                study-backed run cannot resolve its target.
        """
        count = self.steps if steps is None else steps
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("steps must be a positive integer.")
        if self.study is not None:
            return self._run_study(count, callback, scene)
        return self._run_objective(count, callback)

    def _checked_params(self, target: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """The target's free parameters as JAX arrays, plus their metadata.

        The metadata (name-keyed Parameter objects) carries the constraints
        attached to the free parameters; :meth:`_updater` uses it to project
        descent steps back onto the constraint manifold.  Raises when the
        target declares no free parameters.
        """
        import jax.numpy as jnp

        from cadjoint.extraction import extract_parameters

        free, _, metadata = extract_parameters(target)
        if not free:
            raise ValueError(
                f"Optimization {self.name!r} has nothing to optimize: {type(target).__name__} "
                "declares no free parameters (mark Scalars/Vector2s with free=True)."
            )
        return {name: jnp.asarray(value) for name, value in free.items()}, metadata

    def _record_step(self, step, value, grads, params, history, trajectory, callback) -> None:
        """Finite-check one evaluation and append it to history/trajectory."""
        import jax.numpy as jnp

        objective_value = float(value)
        grad_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(grad)) for grad in grads.values())))
        if not (math.isfinite(objective_value) and math.isfinite(grad_norm)):
            raise ValueError(
                f"Optimization {self.name!r} left the finite range at step {step} "
                f"(objective={objective_value}, grad_norm={grad_norm}); "
                "lower the learning rate or rescale the objective."
            )
        record = {"step": step, "objective": objective_value, "grad_norm": grad_norm}
        history.append(record)
        trajectory.append(
            {"step": step, "objective": objective_value, "parameters": _serialize(params)}
        )
        if callback is not None:
            callback(record)

    def _finished(
        self, count, method, history, trajectory, params, initial, result=None
    ) -> OptimizationRun:
        return OptimizationRun(
            name=self.name,
            method=method,
            steps=count,
            learning_rate=self.learning_rate,
            history=history,
            trajectory=_subsample(trajectory, TRAJECTORY_LIMIT),
            parameters=_serialize(params),
            initial=initial,
            result=result,
        )

    def _run_objective(self, count: int, callback) -> OptimizationRun:
        """The objective-form descent loop (one fixed differentiable objective)."""
        import jax
        import jax.numpy as jnp

        params, metadata = self._checked_params(self.of)
        objective = self.objective
        value_and_grad = jax.value_and_grad(lambda p: jnp.asarray(objective(p)))
        method, init, step_fn = self._updater(metadata)

        initial = _serialize(params)
        history: list[dict[str, float]] = []
        trajectory: list[dict[str, Any]] = []
        state = init(params)
        for step in range(count):
            value, grads = value_and_grad(params)
            self._record_step(step, value, grads, params, history, trajectory, callback)
            params, state = step_fn(params, grads, state)

        final_value = float(jnp.asarray(objective(params)))
        if not math.isfinite(final_value):
            raise ValueError(
                f"Optimization {self.name!r} left the finite range after its last step "
                f"(objective={final_value}); lower the learning rate."
            )
        trajectory.append(
            {"step": count, "objective": final_value, "parameters": _serialize(params)}
        )
        return self._finished(count, method, history, trajectory, params, initial)

    # ── study form ──────────────────────────────────────────────────────────

    def _study_sim_mesh(self) -> Any:
        """The study's SimMesh — its declared one, or the implicit wrap.

        Creates the study's anonymous implicit mesh exactly like
        ``cadjoint.fem.study._solve_mesh`` does, so run-time refreezes and
        later ``study.solve`` calls share one mesh (and its cache).
        """
        from cadjoint.fem.simmesh import _anonymous

        study = self.study
        if study.mesh is not None:
            return study.mesh
        if study._implicit_mesh is None:
            study._implicit_mesh = _anonymous(
                name=f"{study.name}::mesh",
                resolution=study.resolution,
                domain=study.domain,
                bounds=study.bounds,
                size=study.size,
            )
        return study._implicit_mesh

    def _metric_value(self, result: Any, mesh: Any, points: Any) -> Any:
        """The study metric as a (possibly traced) JAX scalar."""
        if self.metric == "compliance":
            return _compliance(self.study, result, mesh, points)
        return result.mean() if self.metric == "mean" else result.max()

    def _run_study(self, count: int, callback, scene: Any) -> OptimizationRun:
        """The study-form descent loop: frozen-topology solves per step.

        Follows the flagship bracket example's doctrine: extract the
        study's mesh once (and every ``remesh_every`` steps at the current
        design, plus once more for the final evaluation), resolve BC node
        sets on the frozen topology's nominal points, and per step
        recompute only the node positions through the traced SDF —
        design -> mesh points -> solve -> metric stays one reverse-mode
        chain.  x64 is enabled for the duration (the FEM adjoints require
        float64) and the caller's setting restored afterwards.
        """
        import jax
        import jax.numpy as jnp

        from cadjoint.extraction import apply_parameters, extract_parameters
        from cadjoint.fem.backends import _x64_scope
        from cadjoint.fem.hexmesh import recompute_points
        from cadjoint.fem.tetmesh import TetMesh, recompute_tet_points
        from cadjoint.functionalize import functionalize

        study = self.study
        target = self._study_target(scene)
        if target is None:
            raise ValueError(
                f"Optimization {self.name!r} is study-backed and study {study.name!r} "
                "meshes the scene itself; pass the scene object via run(scene=...)."
            )
        if not hasattr(target, "children"):
            raise ValueError(
                f"Optimization {self.name!r} needs a scene object to differentiate; "
                f"study {study.name!r} meshes a plain callable field "
                f"({type(target).__name__}), which has no extractable parameters."
            )

        free, fixed, metadata = extract_parameters(target)
        if not free:
            raise ValueError(
                f"Optimization {self.name!r} has nothing to optimize: {type(target).__name__} "
                "declares no free parameters (mark Scalars/Vector2s with free=True)."
            )
        fn = functionalize(target)
        sim_mesh = self._study_sim_mesh()
        regularizer = self.regularizer
        weight = self.regularizer_weight

        def concrete(values: dict[str, Any]) -> dict[str, np.ndarray]:
            return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}

        def field_at(values: dict[str, Any]):
            inner = fn(values, fixed)
            return lambda p: jnp.asarray(inner(p))

        # GRADIENT-PATH SEAM.  This is the one place the design->points
        # derivative path is chosen, per gradient_path:
        # - "direct" (default): the frozen-topology path — node positions
        #   re-projected onto the true SDF (recompute_points /
        #   recompute_tet_points), validated on crease-heavy geometry.  Tet
        #   meshes smooth the boundary displacement into the interior so
        #   frozen Steiner tets stay well shaped (and the solve well
        #   conditioned) as the design moves between refreezes.
        # - "tesseract": the packaged two-tesseract chain — lattice samples
        #   -> mesher tesseract (surface-interpolation VJP) -> solver
        #   tesseract adjoint (cadjoint.fem.tesseracts.chain).  Validated
        #   descending and sign-consistent with the direct path on the
        #   crease-heavy starter heat sink (research/tet-vs-hex.md), but it
        #   meshes the interpolant (can need a finer lattice) and drops
        #   tangential vertex motion — kept opt-in until cleared as default.
        use_tesseract = self.gradient_path == "tesseract"
        if use_tesseract:
            from cadjoint.fem.tesseracts.chain import freeze_study_chain

        def recompute(field: Any, mesh: Any) -> Any:
            if isinstance(mesh, TetMesh):
                return recompute_tet_points(field, mesh, smooth_passes=2)
            return recompute_points(field, mesh)

        def objective_on(mesh: Any, chain: Any = None):
            def objective(params):
                if chain is not None:
                    samples = field_at(params)(jnp.asarray(chain.lattice))
                    value = chain.metric_value(samples, self.metric)
                else:
                    points = recompute(field_at(params), mesh)
                    result = study.solve(mesh=mesh, points=points)
                    value = jnp.asarray(self._metric_value(result, mesh, points))
                if regularizer is not None:
                    value = value + weight * jnp.asarray(regularizer(params))
                return value

            return objective

        def refreeze(values: dict[str, Any], step: int, current: Any):
            """Re-extract topology at *values*; fall back when a BC empties.

            A boundary-condition selection is anchored in space; a design
            that moves its surface out of the selection would make the new
            mesh unsolvable.  The same goes for a re-extraction the mesher
            itself rejects (TetGen refuses some surfaces at unlucky
            designs).  At step 0 both are clear declaration errors; mid-run
            the previous frozen topology is kept (with a printed warning)
            so the run completes instead of raising.
            """
            held = concrete(values)
            apply_parameters(target, held)
            chain = None
            try:
                if use_tesseract:
                    chain = freeze_study_chain(study, sim_mesh, field_at(held))
                    mesh = chain.mesh
                else:
                    mesh = sim_mesh.build(field_at(held), rebuild=True)
            except RuntimeError as error:
                if current is None:
                    raise ValueError(
                        f"Optimization {self.name!r} cannot start: meshing study "
                        f"{study.name!r} failed at the initial design ({error})."
                    ) from error
                print(
                    f"warning: re-meshing at step {step} failed ({error}); "
                    "keeping the previous frozen topology."
                )
                return current
            problem = _unresolvable_bc(study, mesh)
            if problem is None:
                return mesh, objective_on(mesh, chain)
            if current is None:
                raise ValueError(
                    f"Optimization {self.name!r} cannot start: {problem} on study "
                    f"{study.name!r}'s initial mesh; check the boundary-condition "
                    "selections against the current design."
                )
            print(
                f"warning: after re-meshing at step {step}, {problem} on study "
                f"{study.name!r}; keeping the previous frozen topology. Pin the "
                "loaded surface with constraints so optimization cannot move it."
            )
            return current

        initial_values = concrete(free)
        method, init, step_fn = self._updater(metadata)
        history: list[dict[str, float]] = []
        trajectory: list[dict[str, Any]] = []
        with _x64_scope():
            # Promote the parameter leaves to float64: the FEM adjoints run in
            # x64, and float32 leaves make the Newton projection's backward
            # pass degenerate exactly on sketch loci (NaN gradients).
            params = {name: jnp.asarray(value) for name, value in concrete(free).items()}
            initial = _serialize(params)
            state = init(params)
            try:
                frozen = refreeze(params, 0, None)
                value_and_grad = jax.value_and_grad(frozen[1])
                for step in range(count):
                    if step > 0 and self.remesh_every > 0 and step % self.remesh_every == 0:
                        frozen = refreeze(params, step, frozen)
                        value_and_grad = jax.value_and_grad(frozen[1])
                    value, grads = value_and_grad(params)
                    self._record_step(step, value, grads, params, history, trajectory, callback)
                    params, state = step_fn(params, grads, state)

                # Final evaluation on a freshly extracted mesh: the reported
                # optimum does not depend on the last frozen topology, and the
                # concrete result carries the final design's solved field.
                # When the final design has moved a BC surface out of its
                # selection, the last frozen topology (its node positions
                # recomputed at the final parameters) answers instead.
                held = concrete(params)
                apply_parameters(target, held)
                final_field = field_at(held)
                problem = None
                try:
                    final_mesh = sim_mesh.build(final_field, rebuild=True)
                except RuntimeError as error:
                    final_mesh = None
                    problem = f"re-extraction failed ({error})"
                if final_mesh is not None:
                    problem = _unresolvable_bc(study, final_mesh)
                if problem is None:
                    result = study.solve(final_field)
                    final_points = None
                else:
                    print(
                        f"warning: after re-meshing at the final design, {problem} on "
                        f"study {study.name!r}; result evaluated on the last frozen "
                        "mesh. Pin the loaded surface with constraints so "
                        "optimization cannot move it."
                    )
                    final_mesh = frozen[0]
                    final_points = recompute(final_field, final_mesh)
                    result = study.solve(mesh=final_mesh, points=final_points)
                final_value = jnp.asarray(self._metric_value(result, final_mesh, final_points))
                if regularizer is not None:
                    final_value = final_value + weight * jnp.asarray(regularizer(held))
                final_value = float(final_value)
            finally:
                apply_parameters(target, initial_values)
        if not math.isfinite(final_value):
            raise ValueError(
                f"Optimization {self.name!r} left the finite range after its last step "
                f"(objective={final_value}); lower the learning rate."
            )
        trajectory.append(
            {"step": count, "objective": final_value, "parameters": _serialize(params)}
        )
        return self._finished(count, method, history, trajectory, params, initial, result=result)
