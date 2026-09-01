"""Starter heat-sink thermal study on the tet path.

The playground's default scene (``scenes/starter.py``) declares the
heat-sink thermal study; these tests prove the tet meshing method carries
it: the sink meshes as TET10 at the scene's own resolution/bounds, the
study solves on it through the standard routing, and the frozen-topology
differentiable path (``recompute_tet_points`` + ``solve(points=...)``)
produces a design gradient that matches central finite differences on the
``fin_depth`` parameter.
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

    scene = starter["scene"]
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
        free0, _, _, objective = tet_state
        depth0 = jnp.asarray(np.asarray(free0["fin_depth"]), dtype=jnp.float64).reshape(())
        value, gradient = jax.value_and_grad(objective)(depth0)
        assert float(value) > 0.0
        assert np.isfinite(float(gradient))
        eps = 1e-3
        finite_difference = (float(objective(depth0 + eps)) - float(objective(depth0 - eps))) / (
            2.0 * eps
        )
        # fin_depth moves the extrusion's flat end caps — a smooth, crease-
        # light parameter, so adjoint and FD agree tightly.
        assert np.isclose(float(gradient), finite_difference, rtol=5e-2), (
            float(gradient),
            finite_difference,
        )
