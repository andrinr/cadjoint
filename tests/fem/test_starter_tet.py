"""Starter heat-sink thermal study on the tet path.

The playground's default scene (``scenes/starter.py``) declares the
heat-sink thermal study; these tests prove the tet meshing method carries
it: the sink meshes as TET10 at the scene's own resolution/bounds, the
study solves on it through the standard routing, and the frozen-topology
differentiable path (``recompute_tet_points`` + ``solve(points=...)``)
produces a design gradient that matches central finite differences on the
``fin_depth`` parameter.

Gradient note (measured 2026-09-01): the objective is *kinked at the
freeze design*.  ``fin_depth`` drives the extrusion's end caps, and the
frozen boundary nodes at the fin-tip corners sit exactly on those caps at
``fin_depth = 1.2``, which is a branch boundary of the extrusion's
``max()``.  The adjoint returns the left derivative there (-0.0890,
matching backward differences); central differences average the two
one-sided slopes and return -0.0730.  The FD check therefore probes at
``1.2 + _KINK_OFFSET``, where the two agree to 3.6e-7, and the kink itself
is pinned by its own test rather than hidden behind a loose tolerance.
"""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tetgen")
pytest.importorskip("jax_fem")

import jax
import jax.numpy as jnp

_REPO = Path(__file__).resolve().parents[2]
_SCENE_PATH = _REPO / "scenes" / "starter.py"

#: How far to step off the freeze design before probing the gradient with
#: central differences.  The freeze design is a kink point (see
#: ``test_the_freeze_design_sits_on_a_kink_and_the_adjoint_takes_it``);
#: 0.05 clears it by 500x the largest FD epsilon used here and stays well
#: inside the frozen topology's validity (the boundary point Jacobian is a
#: uniform 0.5 there, measured over all 860 surface nodes).
_KINK_OFFSET = 0.05


@pytest.fixture(scope="module")
def starter():
    """Execute the starter scene the way the compile worker does."""
    from cadjoint.fem import capture_sim_meshes, capture_studies

    namespace: dict = {}
    with capture_sim_meshes(), capture_studies():
        exec(compile(_SCENE_PATH.read_text(), str(_SCENE_PATH), "exec"), namespace, namespace)
    return namespace


@pytest.fixture(scope="module")
def tet_state(starter):
    """TET10 SimMesh + thermal study over the starter's own declaration.

    Reuses the scene's meshing box, resolution, conductivity, and boundary
    conditions verbatim — only the method differs — so this is exactly the
    solve the playground runs when the starter's SimMesh flips to tet10.
    """
    from cadjoint import extract_parameters, functionalize
    from cadjoint.fem import SimMesh, ThermalStudy, recompute_tet_points

    # The body the declared mesh discretizes (its ``domain=``), not the
    # rendered scene: the starter also draws board-level context the physics
    # never sees.
    scene = starter["thermal_body"]
    free0, fixed, _ = extract_parameters(scene)
    scene_fn = functionalize(scene)
    sdf0 = scene_fn(free0, fixed)
    declared = starter["sink_mesh"]
    heat_study = starter["heat_study"]
    tet_decl = SimMesh(
        name="sink-mesh-tet10",
        resolution=declared.resolution,
        bounds=declared.bounds,
        size=declared.size,
        method="tet10",
        domain=sdf0,
    )
    study = ThermalStudy(
        name="sink-conduction-tet10",
        conductivity=float(heat_study.conductivity),
        bcs=list(heat_study.bcs),
        mesh=tet_decl,
    )
    mesh = tet_decl.build()  # topology frozen at the nominal design

    def objective(fin_depth):
        free = dict(free0)
        free["fin_depth"] = fin_depth
        points = recompute_tet_points(scene_fn(free, fixed), mesh)
        return study.solve(points=points).mean()

    return free0, mesh, study, objective


class TestStarterTetThermal:
    def test_sink_meshes_as_tet10_at_the_declared_resolution(self, tet_state):
        _, mesh, _, _ = tet_state
        assert mesh.cells.shape[1] == 10
        assert mesh.num_cells > 0
        assert mesh.num_surface > 0

    def test_heat_study_solves_on_the_tet_mesh(self, tet_state):
        _, mesh, study, _ = tet_state
        result = study.solve()
        assert result.mesh is mesh
        temperature = np.asarray(result.temperature)
        assert np.isfinite(temperature).all()
        # Heat enters at the slug bottom and leaves at the fin field held
        # at 0: the field is hottest at the bottom, and not identically 0.
        z = np.asarray(mesh.points)[:, 2]
        assert temperature[z < -0.1].mean() > temperature[z > 0.6].mean()
        assert float(result.max()) > 0.0
        assert result.describe()["mesh"] == "sink-mesh-tet10"

    def test_fin_depth_gradient_matches_finite_differences(self, tet_state):
        """Adjoint vs central FD, probed where the objective is differentiable.

        The probe is offset by ``+_KINK_OFFSET`` from the freeze design on
        purpose: at the freeze design itself the objective has a genuine
        kink (see the next test), and central differences there straddle
        it.  Away from that one point the agreement is textbook.

        Measured at ``fin_depth = 1.25``: adjoint -0.052300674, central FD
        -0.052300693 at eps=1e-3, a relative error of **3.6e-7**, falling
        to 3.2e-8 at eps=3e-4 and rising to 3.2e-6 at eps=3e-3 — clean
        second-order convergence, so the residual is FD truncation.
        Asserted at rtol=1e-4, ~280x the measured error.
        """
        free0, _, _, objective = tet_state
        depth0 = jnp.asarray(np.asarray(free0["fin_depth"]), dtype=jnp.float64).reshape(())
        probe = depth0 + _KINK_OFFSET
        value, gradient = jax.value_and_grad(objective)(probe)
        assert float(value) > 0.0
        assert np.isfinite(float(gradient))
        eps = 1e-3
        finite_difference = (float(objective(probe + eps)) - float(objective(probe - eps))) / (
            2.0 * eps
        )
        print(
            f"\nstarter tet @ fin_depth={float(probe):.6f}: "
            f"adjoint={float(gradient):+.9f} FD({eps:.0e})={finite_difference:+.9f}"
        )
        assert np.isclose(float(gradient), finite_difference, rtol=1e-4), (
            float(gradient),
            finite_difference,
        )

    def test_the_freeze_design_sits_on_a_kink_and_the_adjoint_takes_it(self, tet_state):
        """Why the probe above is offset: the freeze design is a kink point.

        The fin comb is an extrusion, ``max(profile_2d, |y| - fin_depth/2)``,
        and the frozen boundary nodes at the fin-tip corners sit *exactly*
        on the end cap ``y = +-0.6`` at ``fin_depth = 1.2``.  Shrinking the
        depth leaves them outside, so ``project_points`` drags them with the
        cap (measured ``|dx/d(fin_depth)| = 0.5``); growing it leaves them
        inside, where the cap term is not the active branch and they do not
        move.  The two one-sided derivatives therefore differ, and this is
        a property of the design, not of the solver.

        Measured at eps=1e-4:

        =============  ===========
        adjoint        -0.08897663
        backward FD    -0.08898619
        forward FD     -0.05709389
        central FD     -0.07304004
        =============  ===========

        The adjoint is the **left** derivative and tracks backward FD to
        1.07e-4 relative (3.4e-4 at eps=3e-4, 1.2e-3 at eps=1e-3 — it
        converges as eps shrinks).  Central FD is the mean of the two
        one-sided slopes and is 18% away from either; a central-difference
        check at this design is measuring the kink, not the gradient.
        """
        free0, _, _, objective = tet_state
        depth0 = jnp.asarray(np.asarray(free0["fin_depth"]), dtype=jnp.float64).reshape(())
        value, gradient = jax.value_and_grad(objective)(depth0)
        eps = 1e-4
        ahead = float(objective(depth0 + eps))
        behind = float(objective(depth0 - eps))
        backward = (float(value) - behind) / eps
        forward = (ahead - float(value)) / eps
        central = (ahead - behind) / (2.0 * eps)
        print(
            f"\nstarter tet @ freeze design: adjoint={float(gradient):+.9f} "
            f"bwd={backward:+.9f} fwd={forward:+.9f} central={central:+.9f}"
        )
        # The kink is real: the one-sided slopes differ by 0.0319, i.e. 36%
        # of the gradient itself (measured 0.03189).
        assert forward - backward > 0.03
        # The adjoint is the left derivative (measured rel. error 1.07e-4).
        assert np.isclose(float(gradient), backward, rtol=1e-3), (float(gradient), backward)
        # ...and central FD is not, by 18% (measured 0.1796) — the reason
        # this test exists instead of a widened tolerance on the one above.
        assert backward < central < forward
        assert abs(central - float(gradient)) / abs(float(gradient)) > 0.1
