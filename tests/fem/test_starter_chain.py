"""Starter heat sink on the TET10 two-Tesseract chain (the seam validation).

The playground's default scene declares a heat-flux thermal study on a
TET10 SimMesh; these tests run it through the packaged mesher + thermal
tesseracts via :func:`cadjoint.fem.tesseracts.chain.freeze_study_chain`
(the machinery behind ``Optimization(gradient_path="tesseract")``) on the
crease-heavy fin comb:

- stage-2 parity: the thermal tesseract equals ``study.solve`` on the
  chain's frozen mesh at 1e-9;
- the max-temperature design gradient w.r.t. ``fin_depth`` is
  sign-consistent with the direct frozen-topology path on the same mesh;
- a 4-step descent on the chain objective decreases max temperature;
- ``Optimization(..., gradient_path="tesseract")`` runs end to end.

Lattice note (measured): the mesher tesseract meshes the *trilinear
interpolant* of the lattice samples, which self-intersects at the scene's
declared 18x13x11 resolution (both sharp modes) — 24x18x15 is the coarsest
lattice where sharp DC meshes.  The direct path meshes the true SDF at the
declared resolution; this gap is part of the gradient-path trade recorded
in ``research/tet-vs-hex.md``.
"""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tetgen")
pytest.importorskip("jax_fem")
pytest.importorskip("tesseract_core")
pytest.importorskip("tesseract_jax")

import jax
import jax.numpy as jnp

_REPO = Path(__file__).resolve().parents[2]
_SCENE_PATH = _REPO / "scenes" / "starter.py"

#: The coarsest lattice where sharp DC on the interpolant meshes the sink.
_CHAIN_RESOLUTION = (24, 18, 15)


@pytest.fixture(scope="module")
def starter():
    """Execute the starter scene the way the compile worker does."""
    from cadjoint.fem import capture_sim_meshes, capture_studies

    namespace: dict = {}
    with capture_sim_meshes(), capture_studies():
        exec(compile(_SCENE_PATH.read_text(), str(_SCENE_PATH), "exec"), namespace, namespace)
    return namespace


@pytest.fixture(scope="module")
def chain_state(starter):
    """Frozen two-tesseract chain over the starter's own thermal study.

    Reuses the scene's meshing box, conductivity, and boundary conditions
    verbatim; only the lattice is refined to the chain's coarsest meshable
    resolution (see module docstring).
    """
    from cadjoint import extract_parameters, functionalize
    from cadjoint.fem import SimMesh, ThermalStudy
    from cadjoint.fem.tesseracts.chain import freeze_study_chain

    scene = starter["scene"]
    free0, fixed, _ = extract_parameters(scene)
    scene_fn = functionalize(scene)
    declared = starter["sink_mesh"]
    heat_study = starter["heat_study"]
    tet_decl = SimMesh(
        name="sink-mesh-chain",
        resolution=_CHAIN_RESOLUTION,
        bounds=declared.bounds,
        size=declared.size,
        method="tet10",
    )
    study = ThermalStudy(
        name="sink-conduction-chain",
        conductivity=float(heat_study.conductivity),
        bcs=list(heat_study.bcs),
        mesh=tet_decl,
    )

    def field_at(free):
        inner = scene_fn(free, fixed)
        return lambda p: jnp.asarray(inner(p))

    chain = freeze_study_chain(study, tet_decl, field_at(free0))

    def samples_of(fin_depth):
        free = dict(free0)
        free["fin_depth"] = fin_depth
        return field_at(free)(jnp.asarray(chain.lattice))

    def objective_tesseract(fin_depth):
        return chain.metric_value(samples_of(fin_depth), "max")

    def objective_direct(fin_depth):
        from cadjoint.fem.tetmesh import recompute_tet_points

        free = dict(free0)
        free["fin_depth"] = fin_depth
        points = recompute_tet_points(field_at(free), chain.mesh, smooth_passes=2)
        return study.solve(mesh=chain.mesh, points=points).max()

    return free0, study, chain, samples_of, objective_tesseract, objective_direct


class TestStarterChain:
    def test_chain_meshes_the_sink_as_tet10(self, chain_state):
        _, _, chain, _, _, _ = chain_state
        assert chain.mesh.cells.shape[1] == 10
        assert chain.mesh.num_cells > 0
        assert chain.mesh.num_surface > 0

    def test_stage_two_parity_against_the_direct_solve(self, chain_state):
        """Same frozen mesh, same BC resolution: tesseract == direct at 1e-9."""
        free0, study, chain, samples_of, _, _ = chain_state
        fin0 = jnp.asarray(np.asarray(free0["fin_depth"]), dtype=jnp.float64).reshape(())
        _, packaged = chain._solve(jnp.asarray(samples_of(fin0)))
        direct = study.solve(mesh=chain.mesh).temperature
        assert np.abs(np.asarray(packaged) - np.asarray(direct)).max() < 1e-9
        # Physics sanity: the die heats the slug bottom, the fin field is held.
        temperature = np.asarray(direct)
        z = np.asarray(chain.mesh.points)[:, 2]
        assert temperature[z < -0.1].mean() > temperature[z > 0.6].mean()
        assert temperature.max() > 0.0

    def test_fin_depth_gradient_is_sign_consistent_with_the_direct_path(self, chain_state):
        """The seam-validation measurement (recorded in research/tet-vs-hex.md).

        The two paths differentiate different frozen chains (interpolation
        VJP on the interpolant vs Newton re-projection onto the true SDF),
        so the numbers agree in sign and scale, not to solver tolerance —
        the fin comb is the crease-heavy case where the interpolation VJP's
        single-field crease treatment is being validated.
        """
        free0, _, _, _, objective_tesseract, objective_direct = chain_state
        fin0 = jnp.asarray(np.asarray(free0["fin_depth"]), dtype=jnp.float64).reshape(())
        value_t, grad_t = jax.value_and_grad(objective_tesseract)(fin0)
        value_d, grad_d = jax.value_and_grad(objective_direct)(fin0)
        print(
            f"\nstarter chain: tesseract J={float(value_t):.6f} g={float(grad_t):+.6f} | "
            f"direct J={float(value_d):.6f} g={float(grad_d):+.6f}"
        )
        assert np.isfinite(float(grad_t)) and np.isfinite(float(grad_d))
        # Deeper fins conduct the die's heat away: both paths must say so.
        assert float(grad_t) < 0.0
        assert float(grad_d) < 0.0
        # Same order of magnitude (the paths solve slightly different meshes).
        ratio = float(grad_t) / float(grad_d)
        assert 0.2 < ratio < 5.0

    def test_four_step_descent_decreases_max_temperature(self, chain_state):
        free0, _, _, _, objective_tesseract, _ = chain_state
        fin = float(np.asarray(free0["fin_depth"]))
        values = []
        for _ in range(4):
            value, gradient = jax.value_and_grad(objective_tesseract)(jnp.asarray(fin))
            values.append(float(value))
            fin -= 0.05 * float(gradient)
        print(f"\nstarter chain descent J: {[round(v, 6) for v in values]}")
        assert all(np.isfinite(v) for v in values)
        assert values[-1] < values[0]


class TestGradientPathOption:
    def test_default_stays_direct(self, starter):
        from cadjoint.optimize import Optimization

        assert Optimization.__dataclass_fields__["gradient_path"].default == "direct"

    def test_invalid_value_is_rejected(self):
        from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
        from cadjoint.optimize import Optimization

        study = ThermalStudy(
            name="gp-bar",
            resolution=(6, 4, 4),
            conductivity=1.0,
            bcs=[Dirichlet(Nodes.side("-x"), 1.0), Dirichlet(Nodes.side("+x"), 0.0)],
        )
        with pytest.raises(ValueError, match="gradient_path"):
            Optimization("o", study=study, metric="mean", gradient_path="fastest")

    def test_objective_form_rejects_gradient_path(self):
        from cadjoint.geometry.parameters import Vector
        from cadjoint.optimize import Optimization
        from cadjoint.sdf.primitives import Box

        box = Box(Vector([1.0, 0.2, 0.2], free=True, name="size"))
        with pytest.raises(ValueError, match="study form"):
            Optimization(
                "o",
                objective=lambda params: jnp.sum(params["size"] ** 2),
                of=box,
                gradient_path="tesseract",
            )

    def test_optimization_runs_on_the_tesseract_path(self, starter):
        """Two seam steps on the starter study, chain gradients end to end."""
        from cadjoint import extract_parameters
        from cadjoint.fem import SimMesh, ThermalStudy
        from cadjoint.optimize import Optimization

        scene = starter["scene"]
        declared = starter["sink_mesh"]
        heat_study = starter["heat_study"]
        tet_decl = SimMesh(
            name="sink-mesh-chain-opt",
            resolution=_CHAIN_RESOLUTION,
            bounds=declared.bounds,
            size=declared.size,
            method="tet10",
        )
        study = ThermalStudy(
            name="sink-conduction-chain-opt",
            conductivity=float(heat_study.conductivity),
            bcs=list(heat_study.bcs),
            mesh=tet_decl,
        )
        optimization = Optimization(
            "cool-sink-tesseract",
            study=study,
            metric="max",
            gradient_path="tesseract",
            remesh_every=0,
            steps=2,
            learning_rate=0.004,
        )
        run = optimization.run(2, scene=scene)
        assert len(run.history) == 2
        assert all(
            np.isfinite(record["objective"]) and np.isfinite(record["grad_norm"])
            for record in run.history
        )
        assert run.history[0]["grad_norm"] > 0.0
        # The final report is evaluated on the direct path (a fresh true-SDF
        # mesh), independent of the gradient path used during descent.
        assert run.result is not None
        assert run.result.describe()["kind"] == "thermal"
        # The run restored the scene's original parameter values.
        free_after, _, _ = extract_parameters(scene)
        assert float(np.asarray(free_after["fin_depth"])) == pytest.approx(1.2, abs=1e-6)
