"""A thermal study on cut cells: no volume mesh, the same result shape everywhere else."""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy
from cadjoint.fem.cutfem import CutMesh
from cadjoint.geometry.parameters import Scalar
from cadjoint.sdf.primitives import Sphere

Q, K, R = 1.0, 2.0, 0.6
EXACT_MEAN = Q * R**2 / (15 * K)  # −kΔT = q in a ball, T = 0 on it: T = q(R² − r²)/(6k)


def _ball_study(resolution: int, name: str):
    ball = Sphere(radius=Scalar(R, free=True, name="R"))
    mesh = SimMesh(
        name=f"{name}-mesh",
        resolution=(resolution,) * 3,
        bounds=(-0.8, -0.8, -0.8),
        size=(1.6, 1.6, 1.6),
        method="cutfem",
        domain=ball,
    )
    study = ThermalStudy(
        name=name,
        conductivity=K,
        source=Q,
        bcs=[Dirichlet(Nodes.sphere([0.0, 0.0, 0.0], 10.0), 0.0)],
        mesh=mesh,
    )
    return ball, mesh, study


def test_a_ball_solves_on_cut_cells_close_to_the_closed_form():
    ball, mesh, study = _ball_study(12, "ball")
    built = mesh.build(ball)
    assert isinstance(built, CutMesh)
    assert built.num_points > 0 and built.cells is None and built.num_cells > 0
    result = study.solve(ball)
    surface = result.nodal_scalar()
    assert surface.shape == (built.num_points,)
    # the surface is held at 0 by Nitsche's method, to a few percent of the peak q R² / (6k)
    assert np.abs(surface).max() < 0.05 * Q * R**2 / (6 * K)
    assert abs(float(result.mean()) / EXACT_MEAN - 1) < 0.15
    assert float(result.max()) > float(result.mean())
    described = result.describe()
    assert described["nodes"] == built.num_points and described["elements"] == built.num_cells
    assert described["mass"] is None and described["refinement"] is None
    report = mesh.inspect(ball)
    assert report["method"] == "cutfem" and report["quality"] == {}


def test_conditions_by_region_and_the_material_refusal():
    ball = Sphere(radius=0.6)
    mesh = SimMesh(
        name="halves-mesh",
        resolution=(10, 10, 10),
        bounds=(-0.8,) * 3,
        size=(1.6,) * 3,
        method="cutfem",
        domain=ball,
    )
    study = ThermalStudy(
        name="halves",
        conductivity=K,
        bcs=[
            HeatFlux(Nodes.halfspace([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]), 1.0),
            Dirichlet(Nodes.halfspace([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), 0.0),
        ],
        mesh=mesh,
    )
    result = study.solve(ball)
    surface = result.nodal_scalar()
    z = np.asarray(result.mesh.points)[:, 2]
    # heated below, held at ambient above: the lower hemisphere is the warm one
    assert surface[z < -0.3].mean() > surface[z > 0.3].mean() + 0.05
    # a selection that catches no facet is refused with the study's name
    empty = ThermalStudy(
        name="empty",
        conductivity=K,
        bcs=[Dirichlet(Nodes.sphere([5.0, 5.0, 5.0], 0.1), 0.0)],
        mesh=mesh,
    )
    with pytest.raises(ValueError, match="matched no boundary nodes|selects no boundary facet"):
        empty.solve(ball)
    materials = ThermalStudy(
        name="mat", bcs=[Dirichlet(Nodes.sphere([0, 0, 0], 10.0), 0.0)], mesh=mesh
    )
    with pytest.raises(ValueError, match="numeric conductivity"):
        materials.solve(ball)


def test_the_study_payload_renders_the_cut_surface():
    from cadjoint.viewer.worker.payloads import _study_payload

    ball, mesh, study = _ball_study(10, "ball-payload")
    result = study.solve(ball)
    payload = _study_payload(study, result, ball)
    assert payload["mesh"]["vertex_count"] == mesh.build(ball).num_points
    assert len(payload["mesh"]["fields"]["temperature"]) == payload["mesh"]["vertex_count"]
    assert payload["mesh"]["indices"] and payload["mesh"]["edges"]
    assert payload["study"]["mesh"] == "ball-payload-mesh"
    assert payload["mesh_info"]["method"] == "cutfem"


def test_the_viewer_simulates_and_inspects_a_cut_cell_study():
    from cadjoint.viewer.worker.fem import _inspect_mesh, _simulate_study

    ball, mesh, study = _ball_study(10, "ball-viewer")
    simulated = _simulate_study(ball, [study], {"name": "ball-viewer"})
    assert simulated["ok"] and simulated["mesh"]["vertex_count"] > 0
    inspected = _inspect_mesh(ball, [mesh], [study], {"name": "ball-viewer-mesh"})
    assert (
        inspected["ok"] and inspected["mesh"]["vertex_count"] == simulated["mesh"]["vertex_count"]
    )


def test_the_optimizer_descends_a_cut_cell_study():
    """Mean temperature grows with the radius, so the descent shrinks it — through the cut-cell gradient."""
    from cadjoint.optimize import Optimization

    ball, mesh, study = _ball_study(8, "ball-opt")
    optimization = Optimization(
        name="cool", study=study, metric="mean", steps=2, learning_rate=0.05
    )
    run = optimization.run(scene=ball)
    assert run.steps == 2 and len(run.history) == 2
    assert run.history[0]["grad_norm"] > 0
    assert float(run.parameters["R"]) < R
    assert run.history[1]["objective"] < run.history[0]["objective"]
    assert run.result is not None and run.result.kind == "thermal"
