"""Optimizations as first-class code citizens (cadjoint.optimize)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from cadjoint.geometry import Scalar, Vector
from cadjoint.optimize import Optimization, capture_optimizations
from cadjoint.sdf.primitives import Box, Sphere

STARTER = Path(__file__).resolve().parent.parent / "scenes" / "starter.py"


def _ball(free: bool = True) -> Sphere:
    return Sphere(Scalar(0.8, free=free, name="radius"))


def _quadratic(params):
    return (params["radius"] - 0.3) ** 2


def _bar_scene() -> Box:
    return Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))


def _bar_study(name: str = "bar"):
    from cadjoint.fem import Dirichlet, Nodes, ThermalStudy

    return ThermalStudy(
        name=name,
        resolution=8,
        conductivity=1.0,
        bcs=[
            Dirichlet(Nodes.side("-x"), 0.0),
            Dirichlet(Nodes.side("+x"), 100.0),
        ],
        bounds=(-1.2, -0.9, -0.9),
        size=(2.4, 1.8, 1.8),
    )


def _elastic_study(name: str = "pull"):
    from cadjoint.fem import ElasticStudy, Fixed, Nodes, Traction

    return ElasticStudy(
        name=name,
        resolution=8,
        youngs=200.0,
        poisson=0.3,
        bcs=[
            Fixed(Nodes.side("-x")),
            Traction(Nodes.side("+x"), (0.0, 0.0, -1.0)),
        ],
        bounds=(-1.2, -0.9, -0.9),
        size=(2.4, 1.8, 1.8),
    )


def _volume(params):
    return jnp.prod(2.0 * params["size"])


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
            "study": None,
            "metric": None,
            "remesh_every": None,
            "regularizer": None,
            "regularizer_weight": 0.0,
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


class TestConstrainedRun:
    def test_updates_project_back_onto_attached_constraints(self):
        # A fixed anchor makes the whole parameter rigid: however hard the
        # objective pulls, every optimizer update is projected back onto
        # the constraint manifold, so the run ends where it started.
        from cadjoint.constraints import FixedConstraint
        from cadjoint.geometry import Vector
        from cadjoint.sdf.primitives import Box

        size = Vector([0.8, 0.5, 0.5], free=True, name="size")
        scene = Box(size)
        FixedConstraint(size, [0.8, 0.5, 0.5])
        run = Optimization(
            "shrink",
            lambda params: jnp.sum(params["size"] ** 2),
            scene,
            steps=5,
            learning_rate=0.1,
        ).run()
        assert run.parameters["size"] == pytest.approx([0.8, 0.5, 0.5], abs=1e-5)


def _exec_starter() -> tuple[dict, list]:
    """Execute the starter scene inside the same capture registries the
    compile worker uses (the study-backed declaration resolves its study by
    name through them)."""
    from cadjoint.fem.simmesh import capture_sim_meshes
    from cadjoint.fem.study import capture_studies

    source = STARTER.read_text(encoding="utf-8")
    namespace = {"__name__": "__starter_optimize_test__"}
    with capture_sim_meshes(), capture_studies(), capture_optimizations() as optimizations:
        exec(compile(source, str(STARTER), "exec"), namespace, namespace)
    return namespace, optimizations


class TestStarterScene:
    def test_the_starter_declares_one_study_backed_optimization(self):
        _, optimizations = _exec_starter()
        assert [optimization.name for optimization in optimizations] == ["cool-sink"]

    def test_declared_study_backed_optimization_descends_on_the_thermal_solve(self):
        pytest.importorskip("jax_fem", reason="study-backed runs need the fem extra")
        import numpy as np

        from cadjoint.constraints import constraint_residuals
        from cadjoint.extraction import extract_parameters

        namespace, optimizations = _exec_starter()
        (optimization,) = optimizations
        scene = namespace["scene"]

        described = optimization.describe(scene)
        assert described["objective"] == "max(sink-conduction)"
        assert described["study"] == "sink-conduction"
        assert described["metric"] == "max"
        assert described["remesh_every"] == 6
        assert described["regularizer"] == "material_volume"
        assert described["regularizer_weight"] == pytest.approx(0.4)
        assert described["steps"] == 12
        assert described["learning_rate"] == pytest.approx(0.004)
        # The study solves on the explicitly declared SimMesh (quality path).
        assert namespace["heat_study"].mesh is namespace["sink_mesh"]
        assert namespace["sink_mesh"].method == "tet10"
        # No declared domain: the scene's free parameters are the design.
        assert "fin_depth" in described["parameters"]
        assert set(namespace["sink_parameters"]) <= set(described["parameters"])
        # Without the scene the parameter list degrades gracefully.
        assert optimization.describe()["parameters"] == []

        # Five steps: the tet10 objective is rougher across the first adam
        # step than the hex one (re-projected DC surface points), so descent
        # is asserted over a short window rather than a single step.
        run = optimization.run(5, scene=scene)
        assert len(run.history) == 5
        assert run.history[-1]["objective"] < run.history[0]["objective"]
        assert all(
            jnp.isfinite(record["objective"]) and jnp.isfinite(record["grad_norm"])
            for record in run.history
        )
        assert jnp.isfinite(run.trajectory[-1]["objective"])
        # The final design was solved concretely on a fresh mesh.
        summary = run.result.describe()
        assert summary["kind"] == "thermal"
        assert summary["nodes"] > 0 and summary["elements"] > 0
        # Every descent step projected back onto the sketch constraints:
        # the final parameters still satisfy the whole declared system.
        _, _, metadata = extract_parameters(scene)
        final = {name: np.asarray(value) for name, value in run.parameters.items()}
        residuals = np.abs(np.asarray(constraint_residuals(final, metadata)))
        assert residuals.size > 0
        assert float(residuals.max()) < 1e-6
        # The run restored the scene's original parameter values.
        assert float(namespace["fin_depth"].value) == pytest.approx(1.2, abs=1e-6)


class TestStudyFormValidation:
    def test_the_two_forms_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Optimization("o", _quadratic, _ball(), study=_bar_study(), metric="mean")

    def test_objective_form_rejects_study_only_arguments(self):
        with pytest.raises(ValueError, match="study form"):
            Optimization("o", _quadratic, _ball(), metric="mean")
        with pytest.raises(ValueError, match="study form"):
            Optimization("o", _quadratic, _ball(), regularizer=_quadratic)
        with pytest.raises(ValueError, match="study form"):
            Optimization("o", _quadratic, _ball(), remesh_every=3)
        with pytest.raises(ValueError, match="study form"):
            Optimization("o", _quadratic, _ball(), regularizer_weight=0.5)

    def test_needs_one_of_the_two_forms(self):
        with pytest.raises(ValueError, match="objective=/of=.*study=/metric="):
            Optimization("o")

    def test_requires_a_known_metric(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            Optimization("o", study=_bar_study(), metric="median")
        with pytest.raises(ValueError, match="metric must be one of"):
            Optimization("o", study=_bar_study())

    def test_compliance_requires_an_elastic_study(self):
        with pytest.raises(ValueError, match="elastic"):
            Optimization("o", study=_bar_study(), metric="compliance")
        elastic = Optimization("o", study=_elastic_study(), metric="compliance")
        assert elastic.metric == "compliance"

    def test_study_names_resolve_against_captured_studies(self):
        from cadjoint.fem.study import capture_studies

        with capture_studies():
            study = _bar_study("named-bar")
            optimization = Optimization("o", study="named-bar", metric="max")
        assert optimization.study is study

        with capture_studies():
            _bar_study("named-bar")
            with pytest.raises(ValueError, match="No declared study named 'nope'"):
                Optimization("o", study="nope", metric="mean")

    def test_study_must_be_a_study_or_name(self):
        with pytest.raises(ValueError, match="ThermalStudy/ElasticStudy"):
            Optimization("o", study=42, metric="mean")

    def test_rejects_a_non_callable_regularizer(self):
        with pytest.raises(ValueError, match="regularizer must be a callable"):
            Optimization("o", study=_bar_study(), metric="mean", regularizer=3.0)

    @pytest.mark.parametrize("weight", [-0.1, float("nan"), "heavy", True])
    def test_rejects_invalid_regularizer_weights(self, weight):
        with pytest.raises(ValueError, match="regularizer_weight"):
            Optimization("o", study=_bar_study(), metric="mean", regularizer_weight=weight)

    @pytest.mark.parametrize("every", [-1, 2.5, True])
    def test_rejects_invalid_remesh_every(self, every):
        with pytest.raises(ValueError, match="remesh_every"):
            Optimization("o", study=_bar_study(), metric="mean", remesh_every=every)

    def test_describe_reports_the_study_declaration(self):
        scene = _bar_scene()
        optimization = Optimization(
            "cool",
            study=_bar_study(),
            metric="mean",
            regularizer=_volume,
            regularizer_weight=0.5,
            steps=4,
            learning_rate=0.05,
        )
        described = optimization.describe(scene)
        assert described == {
            "kind": "optimization",
            "name": "cool",
            "steps": 4,
            "learning_rate": 0.05,
            "method": "adam",
            "parameters": ["size"],
            "objective": "mean(bar)",
            "study": "bar",
            "metric": "mean",
            "remesh_every": 6,
            "regularizer": "_volume",
            "regularizer_weight": 0.5,
        }
        # Without a scene (and no declared domain) the parameters degrade.
        assert optimization.describe()["parameters"] == []


class TestStudyFormRun:
    """Study-backed descent on a small thermal bar (needs jax-fem)."""

    def test_requires_a_scene_when_the_study_meshes_it(self):
        optimization = Optimization("o", study=_bar_study(), metric="mean")
        with pytest.raises(ValueError, match="run\\(scene=...\\)"):
            optimization.run(1)

    def test_descends_finite_and_returns_a_concrete_result(self):
        pytest.importorskip("jax_fem", reason="study-backed runs need the fem extra")
        scene = _bar_scene()
        optimization = Optimization(
            "cool-bar",
            study=_bar_study(),
            metric="mean",
            regularizer=_volume,
            regularizer_weight=5.0,
            remesh_every=0,
            steps=3,
            learning_rate=0.05,
        )
        run = optimization.run(scene=scene)
        assert len(run.history) == 3
        assert run.history[-1]["objective"] < run.history[0]["objective"]
        assert all(
            jnp.isfinite(record["objective"]) and jnp.isfinite(record["grad_norm"])
            for record in run.history
        )
        assert run.trajectory[0]["parameters"] == run.initial
        assert run.trajectory[-1]["parameters"] == run.parameters
        summary = run.result.describe()
        assert summary["kind"] == "thermal"
        assert summary["field"] == "temperature"
        # The regularizer shrinks the bar; the run must not mutate the scene.
        assert all(a < b for a, b in zip(run.parameters["size"], run.initial["size"]))
        assert [float(v) for v in scene.params["size"].value] == pytest.approx(
            [0.8, 0.5, 0.5], abs=1e-6
        )

    def test_remesh_every_refreezes_topology_and_still_descends(self):
        pytest.importorskip("jax_fem", reason="study-backed runs need the fem extra")
        import numpy as np

        scene = _bar_scene()
        study = _bar_study()
        # Node count of the initial topology, from the study's own mesh path.
        initial_nodes = study.solve(lambda p: jnp.asarray(scene(p))).mesh.num_points

        optimization = Optimization(
            "shrink-bar",
            study=study,
            metric="mean",
            regularizer=_volume,
            regularizer_weight=5.0,
            remesh_every=1,
            steps=3,
            learning_rate=0.08,
        )
        run = optimization.run(scene=scene)
        assert run.history[-1]["objective"] < run.history[0]["objective"]
        assert all(np.isfinite(record["objective"]) for record in run.history)
        # The strong shrink changes the extracted topology across refreezes:
        # the final fresh-mesh result has fewer nodes than the initial mesh.
        assert run.result.mesh.num_points != initial_nodes
        assert run.result.mesh.num_points < initial_nodes

    def test_regularizer_composes_with_the_metric(self):
        pytest.importorskip("jax_fem", reason="study-backed runs need the fem extra")
        scene = _bar_scene()
        bare = Optimization(
            "bare", study=_bar_study("bare-bar"), metric="mean", steps=1, learning_rate=0.01
        ).run(scene=scene)
        weighted = Optimization(
            "weighted",
            study=_bar_study("weighted-bar"),
            metric="mean",
            regularizer=_volume,
            regularizer_weight=2.0,
            steps=1,
            learning_rate=0.01,
        ).run(scene=scene)
        # Same initial design, same mesh: the first evaluations differ by
        # exactly weight * regularizer(initial parameters).
        expected = 2.0 * float(_volume({"size": jnp.asarray([0.8, 0.5, 0.5])}))
        assert weighted.history[0]["objective"] - bare.history[0]["objective"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_a_bc_emptied_by_remeshing_falls_back_with_a_warning(self, capsys):
        # The flux is anchored in space at the initial -x wall; a strong
        # shrink pulls the wall out of the selection. Instead of raising at
        # the refreeze (the bug a live run hit), the run keeps the previous
        # frozen topology, warns, and completes with a fallback result.
        pytest.importorskip("jax_fem", reason="study-backed runs need the fem extra")
        from cadjoint.fem import Dirichlet, HeatFlux, Nodes, ThermalStudy

        scene = _bar_scene()
        study = ThermalStudy(
            name="anchored-flux",
            resolution=8,
            conductivity=1.0,
            bcs=[
                Dirichlet(Nodes.side("+x"), 0.0),
                HeatFlux(Nodes.box([-0.95, -1.0, -1.0], [-0.72, 1.0, 1.0]), 4.0),
            ],
            bounds=(-1.2, -0.9, -0.9),
            size=(2.4, 1.8, 1.8),
        )
        optimization = Optimization(
            "shrink-away",
            study=study,
            metric="mean",
            regularizer=_volume,
            regularizer_weight=5.0,
            remesh_every=1,
            steps=2,
            learning_rate=0.15,
        )
        run = optimization.run(scene=scene)
        assert len(run.history) == 2
        assert all(jnp.isfinite(record["objective"]) for record in run.history)
        assert run.result is not None
        assert run.result.describe()["kind"] == "thermal"
        printed = capsys.readouterr().out
        assert "warning: after re-meshing" in printed
        assert "HeatFlux" in printed
        assert "Pin the loaded surface with constraints" in printed

    def test_a_bc_that_never_resolves_is_a_clear_error_at_step_zero(self):
        pytest.importorskip("jax_fem", reason="study-backed runs need the fem extra")
        from cadjoint.fem import Dirichlet, HeatFlux, Nodes, ThermalStudy

        scene = _bar_scene()
        study = ThermalStudy(
            name="misplaced-flux",
            resolution=8,
            conductivity=1.0,
            bcs=[
                Dirichlet(Nodes.side("+x"), 0.0),
                HeatFlux(Nodes.box([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]), 4.0),
            ],
            bounds=(-1.2, -0.9, -0.9),
            size=(2.4, 1.8, 1.8),
        )
        optimization = Optimization("lost", study=study, metric="mean", steps=1)
        with pytest.raises(ValueError, match="cannot start.*HeatFlux"):
            optimization.run(scene=scene)

    def test_compliance_metric_descends_on_an_elastic_study(self):
        pytest.importorskip("jax_fem", reason="study-backed runs need the fem extra")
        scene = _bar_scene()
        optimization = Optimization(
            "stiff-bar",
            study=_elastic_study(),
            metric="compliance",
            remesh_every=0,
            steps=2,
            learning_rate=0.02,
        )
        run = optimization.run(scene=scene)
        assert len(run.history) == 2
        assert all(jnp.isfinite(record["objective"]) for record in run.history)
        assert run.history[-1]["objective"] < run.history[0]["objective"]
        summary = run.result.describe()
        assert summary["kind"] == "elastic"
        assert set(summary["fields"]) == {"displacement", "von_mises"}
