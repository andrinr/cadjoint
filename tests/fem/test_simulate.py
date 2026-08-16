"""Tests for jaxcad.fem.simulate (thermal + elastic solves via jax-fem)."""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax_fem")

from jaxcad.fem.hexmesh import GridSpec, sdf_to_hex_mesh
from jaxcad.fem.selection import Nodes
from jaxcad.fem.simulate import elastic_solve, thermal_solve
from jaxcad.geometry.parameters import Vector
from jaxcad.sdf.primitives import Box

# Bar of 2.0 x 0.3 x 0.3 along x.
_HALF = (1.0, 0.15, 0.15)
_LENGTH = 2.0


def _bar():
    return Box(Vector(list(_HALF), free=True, name="size"))


@pytest.fixture(scope="module")
def bar_mesh():
    """Grid aligned with the bar faces: snapping is a no-op, mesh is regular."""
    grid = GridSpec.from_bounds((-1.1, -0.25, -0.25), (2.2, 0.5, 0.5), (22, 5, 5))
    return sdf_to_hex_mesh(_bar(), grid)


def _hot_end(center):
    return center[0] < -0.999


def _cold_end(center):
    return center[0] > 0.999


class TestThermal:
    def test_linear_monotone_gradient(self, bar_mesh):
        result = thermal_solve(
            bar_mesh,
            conductivity=2.0,
            dirichlet=[(_hot_end, 1.0), (_cold_end, 0.0)],
        )
        temperature = np.asarray(result.temperature)
        x = bar_mesh.points[:, 0]
        # Exact linear profile on the regular mesh.
        expected = (1.0 - x) / _LENGTH
        assert np.abs(temperature - expected).max() < 1e-6
        # Strictly monotone plane means along the bar.
        planes = np.unique(np.round(x, 5))
        means = [temperature[np.isclose(x, plane, atol=1e-5)].mean() for plane in planes]
        assert np.all(np.diff(means) < 0)

    def test_misaligned_grid_stays_near_linear(self):
        # No lattice plane matches the bar's y/z faces: snapping reshapes the
        # skin, and end-ring cells distort slightly (end-face corner vertices
        # already sit on the surface, so they cannot reach the box edge).
        grid = GridSpec.from_bounds((-1.1, -0.2, -0.2), (2.2, 0.4, 0.4), (22, 4, 4))
        mesh = sdf_to_hex_mesh(_bar(), grid)
        result = thermal_solve(
            mesh, conductivity=1.0, dirichlet=[(_hot_end, 1.0), (_cold_end, 0.0)]
        )
        temperature = np.asarray(result.temperature)
        expected = (1.0 - mesh.points[:, 0]) / _LENGTH
        assert np.abs(temperature - expected).max() < 0.05

    def test_volumetric_source_sign(self, bar_mesh):
        # -k T'' = q with q > 0 and cold ends must bulge upward mid-bar
        # (validates the sign convention of the source term).
        result = thermal_solve(
            bar_mesh,
            conductivity=1.0,
            source=1.0,
            dirichlet=[(_hot_end, 0.0), (_cold_end, 0.0)],
        )
        temperature = np.asarray(result.temperature)
        mid = np.abs(bar_mesh.points[:, 0]) < 0.1
        # Analytic mid value for the 1D limit: q L^2 / (8 k) = 0.5.
        assert 0.4 < temperature[mid].mean() < 0.6

    def test_vtk_export(self, bar_mesh, tmp_path):
        meshio = pytest.importorskip("meshio")
        result = thermal_solve(
            bar_mesh, conductivity=1.0, dirichlet=[(_hot_end, 1.0), (_cold_end, 0.0)]
        )
        path = tmp_path / "thermal.vtk"
        result.vtk_export(str(path))
        read_back = meshio.read(str(path))
        assert "temperature" in read_back.point_data
        assert read_back.points.shape == bar_mesh.points.shape


class TestElastic:
    @pytest.fixture(scope="class")
    def cantilever(self, bar_mesh):
        return elastic_solve(
            bar_mesh,
            youngs=1000.0,
            poisson=0.3,
            dirichlet=[_hot_end],
            tractions=[(_cold_end, [0.0, 0.0, -1.0])],
        )

    def test_tip_displacement_vs_euler_bernoulli(self, bar_mesh, cantilever):
        displacement = np.asarray(cantilever.displacement)
        tip = np.isclose(bar_mesh.points[:, 0], 1.0)
        tip_uz = displacement[tip, 2].mean()
        force = 1.0 * 0.3 * 0.3
        inertia = 0.3 * 0.3**3 / 12.0
        euler = -force * _LENGTH**3 / (3.0 * 1000.0 * inertia)
        assert tip_uz < 0.0
        # Order-of-magnitude agreement; shear deformation and coarse
        # discretization keep it within ~30% of beam theory (measured 0.94).
        assert 0.7 < tip_uz / euler < 1.3

    def test_fixed_end_stays_put(self, bar_mesh, cantilever):
        displacement = np.asarray(cantilever.displacement)
        fixed = np.isclose(bar_mesh.points[:, 0], -1.0)
        assert np.abs(displacement[fixed]).max() < 1e-10

    def test_von_mises_peaks_at_root(self, bar_mesh, cantilever):
        von_mises = cantilever.von_mises()
        assert von_mises.shape == (bar_mesh.num_cells,)
        centers_x = bar_mesh.points[bar_mesh.cells].mean(axis=1)[:, 0]
        assert centers_x[np.argmax(von_mises)] < -0.5
        # Bending fiber stress at the outermost cell centers (z = +-0.1):
        # M c / I = (0.09 * 2) * 0.1 / 6.75e-4 ~= 26.7 (measured 23.7).
        assert 15.0 < von_mises.max() < 35.0

    def test_vtk_export(self, bar_mesh, cantilever, tmp_path):
        meshio = pytest.importorskip("meshio")
        path = tmp_path / "elastic.vtk"
        cantilever.vtk_export(str(path))
        read_back = meshio.read(str(path))
        assert "displacement" in read_back.point_data
        assert "von_mises" in read_back.cell_data


class TestNodeSelectionPatches:
    def test_thermal_selection_matches_predicate_solution(self, bar_mesh):
        result = thermal_solve(
            bar_mesh,
            conductivity=2.0,
            dirichlet=[(Nodes.side("-x"), 1.0), (Nodes.side("+x"), 0.0)],
        )
        expected = (1.0 - bar_mesh.points[:, 0]) / _LENGTH
        assert np.abs(np.asarray(result.temperature) - expected).max() < 1e-6

    def test_elastic_selection_matches_predicate_solution(self, bar_mesh):
        by_selection = elastic_solve(
            bar_mesh,
            youngs=1000.0,
            poisson=0.3,
            dirichlet=[Nodes.side("-x")],
            tractions=[(Nodes.side("+x"), [0.0, 0.0, -1.0])],
        )
        by_predicate = elastic_solve(
            bar_mesh,
            youngs=1000.0,
            poisson=0.3,
            dirichlet=[_hot_end],
            tractions=[(_cold_end, [0.0, 0.0, -1.0])],
        )
        np.testing.assert_allclose(
            np.asarray(by_selection.displacement),
            np.asarray(by_predicate.displacement),
            atol=1e-10,
        )

    def test_empty_selection_raises(self, bar_mesh):
        with pytest.raises(ValueError, match="matched no boundary nodes"):
            thermal_solve(
                bar_mesh,
                conductivity=1.0,
                dirichlet=[(Nodes.sphere([50.0, 0.0, 0.0], 0.1), 1.0)],
            )

    def test_faceless_selection_raises_for_traction(self, bar_mesh):
        # The corner node alone matches, but no boundary quad has all four
        # of its corners inside the selection.
        corner_only = Nodes.sphere([1.0, 0.15, 0.15], 0.01)
        with pytest.raises(ValueError, match="spans no complete boundary face"):
            elastic_solve(
                bar_mesh,
                youngs=1000.0,
                poisson=0.3,
                dirichlet=[Nodes.side("-x")],
                tractions=[(corner_only, [0.0, 0.0, -1.0])],
            )


class TestHeatFlux:
    def test_end_flux_gives_the_linear_profile(self, bar_mesh):
        # -k T'' = 0 with T(-1) = 0 and k T'(1) = q: T(x) = (q/k)(x + 1).
        result = thermal_solve(
            bar_mesh,
            conductivity=2.0,
            dirichlet=[(Nodes.side("-x"), 0.0)],
            neumann=[(Nodes.side("+x"), 1.0)],
        )
        temperature = np.asarray(result.temperature)
        expected = (bar_mesh.points[:, 0] + 1.0) * (1.0 / 2.0)
        assert np.abs(temperature - expected).max() < 1e-6

    def test_negative_flux_cools_the_body(self, bar_mesh):
        result = thermal_solve(
            bar_mesh,
            conductivity=1.0,
            dirichlet=[(Nodes.side("-x"), 0.0)],
            neumann=[(Nodes.side("+x"), -1.0)],
        )
        temperature = np.asarray(result.temperature)
        tip = np.isclose(bar_mesh.points[:, 0], 1.0)
        assert temperature[tip].max() < 0.0


class TestBackendResolution:
    def test_unknown_backend_raises(self, bar_mesh):
        with pytest.raises(KeyError, match="Unknown FEM backend"):
            thermal_solve(
                bar_mesh,
                conductivity=1.0,
                dirichlet=[(_hot_end, 1.0), (_cold_end, 0.0)],
                backend="no-such-solver",
            )

    def test_empty_patch_raises(self, bar_mesh):
        with pytest.raises(ValueError, match="selected no boundary faces"):
            thermal_solve(
                bar_mesh,
                conductivity=1.0,
                dirichlet=[(lambda center: center[0] > 99.0, 1.0)],
            )
