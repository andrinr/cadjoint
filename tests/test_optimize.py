"""Optimizations as first-class code citizens (cadjoint.optimize)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from cadjoint.geometry import Scalar
from cadjoint.optimize import Optimization, capture_optimizations
from cadjoint.sdf.primitives import Sphere

STARTER = Path(__file__).resolve().parent.parent / "scenes" / "starter.py"


def _ball(free: bool = True) -> Sphere:
    return Sphere(Scalar(0.8, free=free, name="radius"))


def _quadratic(params):
    return (params["radius"] - 0.3) ** 2


class TestCapture:
    def test_collects_declarations_in_construction_order(self):
        with capture_optimizations() as optimizations:
            first = Optimization("a", _quadratic, _ball())
            second = Optimization("b", _quadratic, _ball())
        assert optimizations == [first, second]

    def test_construction_outside_a_capture_registers_nothing(self):
        Optimization("stray", _quadratic, _ball())
        with capture_optimizations() as optimizations:
            pass
        assert optimizations == []


class TestValidation:
    def test_rejects_an_empty_name(self):
        with pytest.raises(ValueError, match="name"):
            Optimization("  ", _quadratic, _ball())

    def test_rejects_a_non_callable_objective(self):
        with pytest.raises(ValueError, match="callable"):
            Optimization("o", 3.0, _ball())

    def test_rejects_a_target_without_parameters(self):
        with pytest.raises(ValueError, match="scene object"):
            Optimization("o", _quadratic, {"radius": 1.0})

    @pytest.mark.parametrize("steps", [0, -3, 2.5, True])
    def test_rejects_invalid_steps(self, steps):
        with pytest.raises(ValueError, match="steps"):
            Optimization("o", _quadratic, _ball(), steps=steps)

    @pytest.mark.parametrize("rate", [0.0, -0.1, "fast", True])
    def test_rejects_invalid_learning_rates(self, rate):
        with pytest.raises(ValueError, match="learning_rate"):
            Optimization("o", _quadratic, _ball(), learning_rate=rate)

    def test_rejects_an_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            Optimization("o", _quadratic, _ball(), method="newton")


class TestDescribe:
    def test_reports_the_declaration_for_the_viewer(self):
        optimization = Optimization(
            "min-r", _quadratic, _ball(), steps=12, learning_rate=0.01, method="sgd"
        )
        assert optimization.describe() == {
            "kind": "optimization",
            "name": "min-r",
            "steps": 12,
            "learning_rate": 0.01,
            "method": "sgd",
            "parameters": ["radius"],
            "objective": "_quadratic",
        }

    def test_parameters_list_only_the_free_ones(self):
        # The sphere's material parameters are fixed; only the free radius
        # is subject to (and reported for) optimization.
        described = Optimization("o", _quadratic, _ball()).describe()
        assert described["parameters"] == ["radius"]


class TestRun:
    def test_descends_and_stays_finite_on_a_toy_scene(self):
        run = Optimization("o", _quadratic, _ball(), steps=40, learning_rate=0.1).run()
        assert run.steps == 40
        assert len(run.history) == 40
        assert run.history[-1]["objective"] < run.history[0]["objective"]
        assert all(
            jnp.isfinite(record["objective"]) and jnp.isfinite(record["grad_norm"])
            for record in run.history
        )
        assert run.initial["radius"] == pytest.approx(0.8, abs=1e-6)
        assert abs(run.parameters["radius"] - 0.3) < abs(run.initial["radius"] - 0.3)

    def test_final_parameters_hold_only_free_values(self):
        run = Optimization("o", _quadratic, _ball(), steps=2).run()
        assert set(run.parameters) == set(run.initial) == {"radius"}
        assert isinstance(run.parameters["radius"], float)

    def test_run_steps_override_the_declaration(self):
        run = Optimization("o", _quadratic, _ball(), steps=30).run(steps=3)
        assert run.steps == 3
        assert len(run.history) == 3

    def test_refuses_a_target_without_free_parameters(self):
        optimization = Optimization("o", _quadratic, _ball(free=False))
        with pytest.raises(ValueError, match="free parameters"):
            optimization.run(1)

    def test_raises_when_the_objective_leaves_the_finite_range(self):
        exploding = Optimization("o", lambda p: p["radius"] * jnp.nan, _ball())
        with pytest.raises(ValueError, match="finite"):
            exploding.run(2)

    def test_callback_receives_each_history_record(self):
        seen = []
        run = Optimization("o", _quadratic, _ball(), steps=4).run(callback=seen.append)
        assert seen == run.history

    def test_trajectory_spans_initial_to_final_state(self):
        run = Optimization("o", _quadratic, _ball(), steps=5, learning_rate=0.1).run()
        assert len(run.trajectory) == 6
        assert run.trajectory[0]["step"] == 0
        assert run.trajectory[0]["parameters"] == run.initial
        assert run.trajectory[0]["objective"] == run.history[0]["objective"]
        assert run.trajectory[-1]["step"] == 5
        assert run.trajectory[-1]["parameters"] == run.parameters
        assert run.objective == run.trajectory[-1]["objective"]

    def test_long_trajectories_subsample_but_keep_the_endpoints(self):
        run = Optimization("o", _quadratic, _ball(), steps=150, learning_rate=0.01).run()
        assert len(run.history) == 150
        assert len(run.trajectory) == 100
        assert run.trajectory[0]["step"] == 0
        assert run.trajectory[-1]["step"] == 150
        steps = [entry["step"] for entry in run.trajectory]
        assert steps == sorted(steps)


class TestStarterScene:
    def test_declared_optimization_descends_and_stays_finite(self):
        source = STARTER.read_text(encoding="utf-8")
        namespace = {"__name__": "__starter_optimize_test__"}
        with capture_optimizations() as optimizations:
            exec(compile(source, str(STARTER), "exec"), namespace, namespace)
        assert [optimization.name for optimization in optimizations] == ["min-aluminum"]
        optimization = optimizations[0]

        described = optimization.describe()
        assert described["steps"] == 25
        assert described["learning_rate"] == pytest.approx(0.03)
        assert described["objective"] == "material_volume"
        assert "fin_depth" in described["parameters"]
        assert len(described["parameters"]) == 17

        run = optimization.run(3)
        assert run.history[-1]["objective"] < run.history[0]["objective"]
        assert all(jnp.isfinite(record["objective"]) for record in run.history)
        assert jnp.isfinite(run.trajectory[-1]["objective"])
        assert run.parameters["fin_depth"] < run.initial["fin_depth"]
