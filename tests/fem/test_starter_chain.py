"""Starter heat sink on the packaged Tesseract chains (the seam validation).

The playground's default scene declares a heat-flux thermal study on a
TET10 SimMesh and runs it with ``gradient_path="tesseract-dc"``.  Two
frozen chains can carry that study, differing in where the black box is
cut, and this module measures both **at the scene's own declared
18x13x11 resolution** — no lattice refinement, because none is needed.

``freeze_study_chain_dc`` (``gradient_path="tesseract-dc"``, the narrow
cut: JAX dual contouring on the true SDF, only TetGen behind a tesseract)
is the production path and gets the full battery — declared-resolution
meshing, stage-2 parity, adjoint-vs-finite-differences, agreement with the
direct path, and a descent.

``freeze_study_chain`` (``gradient_path="tesseract"``, the whole meshing
pipeline behind the mesher tesseract, differentiating the *trilinear
interpolant* of the lattice samples) is kept for what it genuinely does on
this scene, which is narrower than it used to be.  Measured here on
2026-09-01, at the declared resolution:

- **tet10 mode freezes and solves, but cannot be re-evaluated at any
  moved design at all.**  Its frozen mesh (6412 points / 3499 cells / 860
  surface) reaches stage-2 parity with the direct solve at exactly 0.0,
  yet a ``fin_depth`` step of **1e-10** — machine noise on a 1.2 mm
  parameter — already makes the mesher tesseract raise
  ``Frozen-topology promise violated``: TetGen's Steiner insertion is not
  continuous in its input, so the point count drifts under perturbations
  far below any design tolerance.  There is no descent to test; the honest
  test asserts the error.
- **hex mode works end to end**: it freezes (960 points / 560 cells),
  parity is exact, its adjoint agrees with central finite differences to
  the mesher gauge, and it descends.  ``Optimization(...,
  gradient_path="tesseract")`` runs on a hex SimMesh and fails on a tet
  one, so the end-to-end seam test uses hex.

Lattice note (measured, and it corrects an older note in this file): the
mesher tesseract meshes the sink fine at the declared 18x13x11 with sharp
DC.  Its meshability is *not* monotone in the lattice — 21x16x13,
24x18x15, 30x22x18, 36x26x22, 42x31x26 and 48x36x30 are all rejected with
"input surface mesh contain self-intersections", while 18x13x11 and
27x20x17 succeed.  Refining is not a fix, which is one more reason the DC
chain is the default.  Numbers recorded in ``research/tet-vs-hex.md``.
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

#: Descent step used by the chain descent tests (large enough that one
#: step clears solver noise by three orders of magnitude: the measured
#: per-step drop is ~1.4e-3 on the DC chain, ~2.5e-4 on the hex chain).
_LEARNING_RATE = 0.05


@pytest.fixture(scope="module")
def starter():
    """Execute the starter scene the way the compile worker does."""
    from cadjoint.fem import capture_sim_meshes, capture_studies

    namespace: dict = {}
    with capture_sim_meshes(), capture_studies():
        exec(compile(_SCENE_PATH.read_text(), str(_SCENE_PATH), "exec"), namespace, namespace)
    return namespace


@pytest.fixture(scope="module")
def scene_field(starter):
    """The starter's free parameters and its traced design-field factory."""
    from cadjoint import extract_parameters, functionalize

    scene = starter["scene"]
    free0, fixed, _ = extract_parameters(scene)
    scene_fn = functionalize(scene)

    def field_at(free):
        inner = scene_fn(free, fixed)
        return lambda p: jnp.asarray(inner(p))

    def field_at_depth(fin_depth):
        free = dict(free0)
        free["fin_depth"] = fin_depth
        return field_at(free)

    fin0 = jnp.asarray(np.asarray(free0["fin_depth"]), dtype=jnp.float64).reshape(())
    return free0, fin0, field_at, field_at_depth


def _declared_study(starter, name, method):
    """A SimMesh/ThermalStudy pair on the scene's own box and resolution.

    Only the mesh method varies; bounds, size, resolution, conductivity and
    boundary conditions come from ``scenes/starter.py`` verbatim.
    """
    from cadjoint.fem import SimMesh, ThermalStudy

    declared = starter["sink_mesh"]
    heat_study = starter["heat_study"]
    sim_mesh = SimMesh(
        name=f"sink-mesh-{name}",
        resolution=declared.resolution,
        bounds=declared.bounds,
        size=declared.size,
        method=method,
    )
    study = ThermalStudy(
        name=f"sink-conduction-{name}",
        conductivity=float(heat_study.conductivity),
        bcs=list(heat_study.bcs),
        mesh=sim_mesh,
    )
    return sim_mesh, study


# ── the DC chain: gradient_path="tesseract-dc", the production path ─────────


@pytest.fixture(scope="module")
def dc_chain(starter, scene_field):
    """The DC chain frozen on the starter at its declared resolution."""
    from cadjoint.fem.tesseracts.chain import freeze_study_chain_dc
    from cadjoint.fem.tetmesh import recompute_tet_points

    _free0, fin0, _field_at, field_at_depth = scene_field
    sim_mesh, study = _declared_study(starter, "dc", "tet10")
    chain = freeze_study_chain_dc(study, sim_mesh, field_at_depth(fin0))

    def objective_dc(fin_depth):
        return chain.metric_value(field_at_depth(fin_depth), "max")

    def objective_direct(fin_depth):
        # The direct path's frozen-topology map on the SAME mesh: Newton
        # re-projection onto the true SDF, interior relaxed by 2 passes
        # (exactly what cadjoint.optimize runs for gradient_path="direct").
        points = recompute_tet_points(field_at_depth(fin_depth), chain.mesh, smooth_passes=2)
        return study.solve(mesh=chain.mesh, points=points).max()

    return sim_mesh, study, chain, objective_dc, objective_direct


@pytest.fixture(scope="module")
def dc_gradients(scene_field, dc_chain):
    """One shared gradient measurement for the DC chain (it costs ~25 s).

    Returns ``(value, adjoint, central_fd, direct_adjoint, eps)``.
    """
    _free0, fin0, _field_at, _field_at_depth = scene_field
    _sim_mesh, _study, _chain, objective_dc, objective_direct = dc_chain
    value, adjoint = jax.value_and_grad(objective_dc)(fin0)
    _direct_value, direct_adjoint = jax.value_and_grad(objective_direct)(fin0)
    eps = 1e-3
    central = (float(objective_dc(fin0 + eps)) - float(objective_dc(fin0 - eps))) / (2.0 * eps)
    print(
        f"\nstarter DC chain: J={float(value):.9f} adjoint={float(adjoint):+.9f} "
        f"FD({eps:.0e})={central:+.9f} | direct adjoint={float(direct_adjoint):+.9f}"
    )
    return float(value), float(adjoint), central, float(direct_adjoint), eps


class TestStarterDCChain:
    def test_meshes_the_sink_as_tet10_at_the_declared_resolution(self, starter, dc_chain):
        """No lattice refinement: the DC chain meshes the scene's own grid."""
        _sim_mesh, _study, chain, _obj, _direct = dc_chain
        assert tuple(_sim_mesh.resolution) == tuple(starter["sink_mesh"].resolution)
        assert chain.mesh.cells.shape[1] == 10
        assert chain.mesh.num_cells > 0
        # The frozen boundary is the DC surface itself: every DC vertex is a
        # surface node of the fill (measured 860 vertices / 1716 triangles).
        assert chain.mesh.num_surface == chain.surface_points.shape[0]
        assert chain.surface_faces.shape[1] == 3
        assert chain.surface_faces.shape[0] > chain.surface_points.shape[0]

    def test_stage_two_parity_against_the_direct_solve(self, scene_field, dc_chain):
        """Same frozen mesh, same BC resolution: tesseract == direct.

        Measured max |dT| = 0.0 exactly (the solver tesseract runs the same
        jax-fem backend on the same nodes); asserted at 1e-9.
        """
        _free0, fin0, _field_at, field_at_depth = scene_field
        _sim_mesh, study, chain, _obj, _direct = dc_chain
        nodes, packaged = chain._solve(chain._extract(field_at_depth(fin0)))
        # The chain is a fixed point at its freeze design: the traced nodes
        # reproduce the frozen mesh (measured max |dx| = 0.0).
        assert np.abs(np.asarray(nodes) - np.asarray(chain.mesh.points)).max() < 1e-12
        direct = study.solve(mesh=chain.mesh).temperature
        assert np.abs(np.asarray(packaged) - np.asarray(direct)).max() < 1e-9
        # Physics sanity: the die heats the slug bottom, the fin field is held.
        temperature = np.asarray(direct)
        z = np.asarray(chain.mesh.points)[:, 2]
        assert temperature[z < -0.1].mean() > temperature[z > 0.6].mean()
        assert temperature.max() > 0.0

    def test_fin_depth_gradient_matches_finite_differences(self, dc_gradients):
        """The DC chain differentiates its own objective exactly.

        Measured at the freeze design: adjoint -0.166356900 vs central FD
        -0.166357126 at eps=1e-3, a relative error of **1.4e-6** (and
        1.2e-7 at eps=3e-4, 1.4e-8 at eps=1e-4 — clean second-order
        convergence, so the discrepancy is FD truncation, not the adjoint).
        Asserted at rtol=1e-4, ~70x the measured error.
        """
        value, adjoint, central, _direct, _eps = dc_gradients
        assert value > 0.0
        assert np.isfinite(adjoint)
        assert np.isclose(adjoint, central, rtol=1e-4), (adjoint, central)

    def test_fin_depth_gradient_is_consistent_with_the_direct_path(self, dc_gradients):
        """Sign and scale agreement with the direct frozen-topology path.

        The two paths differentiate different frozen maps on the same mesh
        (DC's QEF surface, which keeps tangential vertex motion, vs the
        direct path's per-vertex Newton re-projection, which does not), so
        they agree in sign and scale rather than to solver tolerance.
        Measured: DC -0.166356900, direct -0.134186049, ratio **1.240**.
        Bounded at 1.0-1.6 — both paths are deterministic, so this is a
        real regression fence, not a fudge factor.
        """
        _value, adjoint, _central, direct_adjoint, _eps = dc_gradients
        assert np.isfinite(direct_adjoint)
        # Deeper fins conduct the die's heat away: both paths must say so.
        assert adjoint < 0.0
        assert direct_adjoint < 0.0
        ratio = adjoint / direct_adjoint
        assert 1.0 < ratio < 1.6, ratio

    def test_descent_decreases_max_temperature(self, scene_field, dc_chain):
        """Three gradient steps on the chain objective, monotone.

        Measured J: 1.15330727 -> 1.15194002 -> 1.15063003 at lr=0.05
        (drops of 1.4e-3 and 1.3e-3, four orders above solver noise).
        """
        _free0, fin0, _field_at, _field_at_depth = scene_field
        _sim_mesh, _study, _chain, objective_dc, _direct = dc_chain
        fin = float(fin0)
        values = []
        for _ in range(3):
            value, gradient = jax.value_and_grad(objective_dc)(jnp.asarray(fin))
            values.append(float(value))
            fin -= _LEARNING_RATE * float(gradient)
        print(f"\nstarter DC descent J: {[round(v, 8) for v in values]}")
        assert all(np.isfinite(v) for v in values)
        assert values[1] < values[0]
        assert values[2] < values[1]


# ── the interpolant chain: gradient_path="tesseract" ────────────────────────


@pytest.fixture(scope="module")
def hex_chain(starter, scene_field):
    """The interpolant chain frozen on the starter as HEX8."""
    from cadjoint.fem.tesseracts.chain import freeze_study_chain

    _free0, fin0, _field_at, field_at_depth = scene_field
    sim_mesh, study = _declared_study(starter, "hex-chain", "hex")
    chain = freeze_study_chain(study, sim_mesh, field_at_depth(fin0))
    lattice = jnp.asarray(chain.lattice)

    def objective(fin_depth):
        return chain.metric_value(field_at_depth(fin_depth)(lattice), "max")

    return sim_mesh, study, chain, objective


@pytest.fixture(scope="module")
def interpolant_tet_chain(starter, scene_field):
    """The interpolant chain frozen on the starter as TET10."""
    from cadjoint.fem.tesseracts.chain import freeze_study_chain

    _free0, fin0, _field_at, field_at_depth = scene_field
    sim_mesh, study = _declared_study(starter, "tet-chain", "tet10")
    chain = freeze_study_chain(study, sim_mesh, field_at_depth(fin0))
    return sim_mesh, study, chain


class TestStarterInterpolantChainHex:
    """What the whole-pipeline mesher chain genuinely delivers here."""

    def test_meshes_the_sink_as_hex8_at_the_declared_resolution(self, hex_chain):
        _sim_mesh, _study, chain, _obj = hex_chain
        assert chain.mesh.cells.shape[1] == 8
        assert chain.mesh.num_cells > 0
        # Measured at 18x13x11: 960 points / 560 hexes.
        assert chain.mesh.num_points > 0

    def test_stage_two_parity_against_the_direct_solve(self, scene_field, hex_chain):
        """Measured max |dT| = 0.0 exactly; asserted at 1e-9."""
        _free0, fin0, _field_at, field_at_depth = scene_field
        _sim_mesh, study, chain, _obj = hex_chain
        samples = jnp.asarray(field_at_depth(fin0)(jnp.asarray(chain.lattice)))
        _points, packaged = chain._solve(samples)
        direct = study.solve(mesh=chain.mesh).temperature
        assert np.abs(np.asarray(packaged) - np.asarray(direct)).max() < 1e-9
        temperature = np.asarray(direct)
        z = np.asarray(chain.mesh.points)[:, 2]
        assert temperature[z < -0.1].mean() > temperature[z > 0.6].mean()
        assert temperature.max() > 0.0

    def test_fin_depth_gradient_agrees_with_finite_differences(self, scene_field, hex_chain):
        """Adjoint vs central FD, to the mesher gauge.

        Measured: adjoint -0.072205495, central FD -0.070080773 at
        eps=1e-4 — a relative gap of **2.94%** that is *constant* over
        eps = 1e-5, 3e-5, 1e-4, 3e-4, 5e-4 (2.942e-2 - 2.943e-2), so it is
        not FD truncation but the mesher tesseract's gauge: its
        surface-interpolation VJP carries only the normal component of
        boundary-vertex motion.  Asserted at rtol=5e-2.

        eps is capped at 1e-4 deliberately: at 1e-3 and above a lattice
        sample (the closest sits |phi| = 4.5e-3 from the surface) flips
        sign, the voxel count moves 960 -> 956, and the mesher rejects the
        call with ``Frozen-topology promise violated``.
        """
        _free0, fin0, _field_at, _field_at_depth = scene_field
        _sim_mesh, _study, _chain, objective = hex_chain
        value, adjoint = jax.value_and_grad(objective)(fin0)
        eps = 1e-4
        central = (float(objective(fin0 + eps)) - float(objective(fin0 - eps))) / (2.0 * eps)
        print(
            f"\nstarter hex chain: J={float(value):.9f} adjoint={float(adjoint):+.9f} "
            f"FD({eps:.0e})={central:+.9f}"
        )
        assert float(value) > 0.0
        assert float(adjoint) < 0.0
        assert np.isclose(float(adjoint), central, rtol=5e-2), (float(adjoint), central)

    def test_descent_decreases_max_temperature(self, scene_field, hex_chain):
        """Measured J: 0.988622627 -> 0.988371046 -> 0.988124474 at lr=0.05."""
        _free0, fin0, _field_at, _field_at_depth = scene_field
        _sim_mesh, _study, _chain, objective = hex_chain
        fin = float(fin0)
        values = []
        for _ in range(3):
            value, gradient = jax.value_and_grad(objective)(jnp.asarray(fin))
            values.append(float(value))
            fin -= _LEARNING_RATE * float(gradient)
        print(f"\nstarter hex chain descent J: {[round(v, 9) for v in values]}")
        assert all(np.isfinite(v) for v in values)
        assert values[1] < values[0]
        assert values[2] < values[1]


class TestStarterInterpolantChainTet:
    """The tet mode's measured limit, asserted rather than assumed."""

    def test_freezes_and_solves_at_the_declared_resolution(self, interpolant_tet_chain):
        """It does mesh the sink here — stage-2 parity, measured max |dT| = 0.0."""
        _sim_mesh, study, chain = interpolant_tet_chain
        assert chain.mesh.cells.shape[1] == 10
        assert chain.mesh.num_cells > 0
        assert chain.mesh.num_surface > 0
        direct = study.solve(mesh=chain.mesh).temperature
        assert np.isfinite(np.asarray(direct)).all()

    def test_frozen_samples_evaluate_but_a_moved_design_does_not(
        self, scene_field, interpolant_tet_chain
    ):
        """TetGen's Steiner count is not continuous: the chain is a point map.

        At the freeze samples the chain evaluates (measured J = 0.968120173).
        A ``fin_depth`` step of **1e-10** is already enough for the mesher
        tesseract to produce a different number of points than the frozen
        topology promises, and every step from 1e-10 to 1e-2 was measured
        to raise.  1e-3 is asserted here because it is both far above any
        plausible design tolerance and the smallest step the descent tests
        elsewhere in this file actually take.

        This is the documented reason the starter's declared optimization
        uses ``gradient_path="tesseract-dc"`` and not ``"tesseract"``.
        """
        _free0, fin0, _field_at, field_at_depth = scene_field
        _sim_mesh, _study, chain = interpolant_tet_chain
        lattice = jnp.asarray(chain.lattice)
        frozen = float(chain.metric_value(field_at_depth(fin0)(lattice), "max"))
        assert np.isfinite(frozen) and frozen > 0.0
        with pytest.raises(RuntimeError, match="Frozen-topology promise violated"):
            chain.metric_value(field_at_depth(fin0 + 1e-3)(lattice), "max")


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

    def test_optimization_runs_on_the_dc_path(self, starter):
        """Two seam steps on the starter study, DC chain gradients end to end.

        This is the scene's own declared configuration (``scenes/starter.py``
        sets ``gradient_path="tesseract-dc"``), at its own resolution.
        """
        from cadjoint import extract_parameters
        from cadjoint.optimize import Optimization

        scene = starter["scene"]
        _sim_mesh, study = _declared_study(starter, "dc-opt", "tet10")
        optimization = Optimization(
            "cool-sink-tesseract-dc",
            study=study,
            metric="max",
            gradient_path="tesseract-dc",
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
        # Measured: 1.13963743 -> 1.12917570, so the seam really descends.
        assert run.history[1]["objective"] < run.history[0]["objective"]
        # The final report is evaluated on the direct path (a fresh true-SDF
        # mesh), independent of the gradient path used during descent.
        assert run.result is not None
        assert run.result.describe()["kind"] == "thermal"
        # The run restored the scene's original parameter values.
        free_after, _, _ = extract_parameters(scene)
        assert float(np.asarray(free_after["fin_depth"])) == pytest.approx(1.2, abs=1e-6)

    def test_optimization_runs_on_the_interpolant_path_in_hex_mode(self, starter):
        """``gradient_path="tesseract"`` end to end — hex, the mode that runs.

        A tet10 SimMesh on this path raises ``Frozen-topology promise
        violated`` on its first traced call (see
        ``TestStarterInterpolantChainTet``); hex has no TetGen in it and
        completes.  Measured objectives: 0.988697833 -> 0.981505746.
        """
        from cadjoint import extract_parameters
        from cadjoint.optimize import Optimization

        scene = starter["scene"]
        _sim_mesh, study = _declared_study(starter, "hex-opt", "hex")
        optimization = Optimization(
            "cool-sink-tesseract-hex",
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
        assert run.history[1]["objective"] < run.history[0]["objective"]
        assert run.result is not None
        assert run.result.describe()["kind"] == "thermal"
        free_after, _, _ = extract_parameters(scene)
        assert float(np.asarray(free_after["fin_depth"])) == pytest.approx(1.2, abs=1e-6)
