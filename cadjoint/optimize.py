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
:func:`cadjoint.extract_parameters` collects from ``of``.  The objective is
an ordinary Python function ``params -> scalar`` over that parameter dict —
the same signature the starter scene's ``material_volume`` has — and every
gradient is a real reverse-mode derivative through it
(:func:`jax.value_and_grad`); there is no finite-difference path.

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
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import KW_ONLY, dataclass
from typing import Any, Callable

import numpy as np

__all__ = ["Optimization", "OptimizationRun", "capture_optimizations"]

METHODS = ("adam", "sgd")
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
    """

    name: str
    method: str
    steps: int
    learning_rate: float
    history: list[dict[str, float]]
    trajectory: list[dict[str, Any]]
    parameters: dict[str, float | list[float]]
    initial: dict[str, float | list[float]]

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
            function of the free-parameter dict, minimized by :meth:`run`.
        of: The scene object whose FREE parameters are optimized
            (anything :func:`cadjoint.extract_parameters` accepts).
        steps: Default number of optimizer steps (keyword-only).
        learning_rate: Optimizer step size (keyword-only).
        method: ``"adam"`` (default) or ``"sgd"`` (keyword-only).  Runs
            through optax; plain gradient descent when optax is missing.
    """

    name: str
    objective: Callable[[dict[str, Any]], Any]
    of: Any
    _: KW_ONLY
    steps: int = 30
    learning_rate: float = 0.05
    method: str = "adam"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Optimization needs a non-empty name.")
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
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer.")
        rate = self.learning_rate
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not rate > 0.0:
            raise ValueError("learning_rate must be a positive number.")
        self.learning_rate = float(rate)
        if self.method not in METHODS:
            raise ValueError(f"method must be one of: {', '.join(METHODS)}.")
        _register(self)

    def _free_parameters(self) -> dict[str, Any]:
        from cadjoint.extraction import extract_parameters

        free, _, _ = extract_parameters(self.of)
        return free

    def describe(self) -> dict[str, Any]:
        """JSON-ready payload: everything the viewer needs to display it."""
        return {
            "kind": "optimization",
            "name": self.name,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "method": self.method,
            "parameters": list(self._free_parameters()),
            "objective": getattr(self.objective, "__name__", type(self.objective).__name__),
        }

    def _updater(self):
        """``(method, init, step)`` — optax when available, plain GD otherwise."""
        try:
            import optax
        except ImportError:
            import jax

            rate = self.learning_rate

            def descend(params, grads, state):
                return jax.tree_util.tree_map(lambda p, g: p - rate * g, params, grads), state

            return "gradient-descent", (lambda _params: None), descend

        transform = (
            optax.adam(self.learning_rate)
            if self.method == "adam"
            else optax.sgd(self.learning_rate)
        )

        def apply(params, grads, state):
            updates, state = transform.update(grads, state, params)
            return optax.apply_updates(params, updates), state

        return self.method, transform.init, apply

    def run(self, steps: int | None = None, callback=None) -> OptimizationRun:
        """Minimize the objective over the free parameters of ``of``.

        Pure reverse-mode differentiation (:func:`jax.value_and_grad`)
        through the objective; the scene object itself is not mutated —
        the returned run carries the optimized values.

        Args:
            steps: Number of optimizer steps (default: the declared
                ``steps``).
            callback: Optional ``callback(record)`` invoked with each
                history record as it is produced.

        Returns:
            The finished :class:`OptimizationRun`.

        Raises:
            ValueError: When ``of`` has no free parameters, or the
                objective (or its gradient) leaves the finite range.
        """
        import jax
        import jax.numpy as jnp

        count = self.steps if steps is None else steps
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("steps must be a positive integer.")
        free = self._free_parameters()
        if not free:
            raise ValueError(
                f"Optimization {self.name!r} has nothing to optimize: {type(self.of).__name__} "
                "declares no free parameters (mark Scalars/Vector2s with free=True)."
            )
        params = {name: jnp.asarray(value) for name, value in free.items()}
        objective = self.objective
        value_and_grad = jax.value_and_grad(lambda p: jnp.asarray(objective(p)))
        method, init, step_fn = self._updater()

        initial = _serialize(params)
        history: list[dict[str, float]] = []
        trajectory: list[dict[str, Any]] = []
        state = init(params)
        for step in range(count):
            value, grads = value_and_grad(params)
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
        return OptimizationRun(
            name=self.name,
            method=method,
            steps=count,
            learning_rate=self.learning_rate,
            history=history,
            trajectory=_subsample(trajectory, TRAJECTORY_LIMIT),
            parameters=_serialize(params),
            initial=initial,
        )
