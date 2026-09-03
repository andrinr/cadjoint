"""The end-cap still meshes and solves after all that modelling.

A part is only "complex" in a useful sense if the complexity survives
discretization. This is the end of the chain: the declared ``SimMesh`` builds a
hex mesh of the housing, both boundary-condition selections find real surface
nodes, and the declared thermal study solves on them.

The specific failure this guards against is silent: a node selection that
matches scattered nodes but spans no complete boundary face resolves fine and
then integrates nothing, so a load quietly does not exist.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("jax_fem", reason="FEM simulation needs the 'fem' extra")


@pytest.fixture(scope="module")
def cap():
    return importlib.import_module("scenes.end_cap")


@pytest.fixture(scope="module")
def mesh(cap):
    return cap.cap_mesh.build()


class TestTheMesh:
    def test_it_meshes_the_housing_only(self, cap, mesh):
        """The bearing, seal and bolt heads are rendered, never simulated."""
        assert cap.cap_mesh.domain is cap.housing
        assert mesh.num_points > 1000

    def test_the_mesh_is_hexes_and_reports_quality(self, cap, mesh):
        info = cap.cap_mesh.inspect()
        assert info["method"] == "hex"
        assert info["elements"] > 500
        assert info["quality"]["scaled_jacobian"]["mean"] > 0.7

    def test_the_box_contains_the_whole_part(self, cap, mesh):
        """Including the dowel mirrored below the mounting face."""
        low = np.asarray(cap.cap_mesh.bounds, dtype=float)
        high = low + np.asarray(cap.cap_mesh.size, dtype=float)
        points = np.asarray(mesh.points, dtype=float)
        assert (points.min(axis=0) > low).all()
        assert (points.max(axis=0) < high).all()
        assert points[:, 2].min() < 0.0, "the mirrored dowel should reach below z = 0"


class TestTheBoundaryConditions:
    def test_every_selection_finds_surface_nodes(self, cap, mesh):
        for bc in cap.cap_study.bcs:
            assert len(bc.nodes.resolve(mesh)) > 0

    def test_the_flux_region_spans_whole_boundary_faces(self, cap, mesh):
        """An area-integrated BC that spans no face integrates nothing."""
        from cadjoint.fem.boundary import faces_from_nodes

        flux = next(bc for bc in cap.cap_study.bcs if type(bc).__name__ == "HeatFlux")
        faces = faces_from_nodes(mesh, flux.nodes.resolve(mesh))
        assert len(faces) > 0

    def test_the_ambient_face_is_the_mounting_face(self, cap, mesh):
        dirichlet = next(bc for bc in cap.cap_study.bcs if type(bc).__name__ == "Dirichlet")
        picked = np.asarray(mesh.points, dtype=float)[dirichlet.nodes.resolve(mesh)]
        assert picked[:, 2].max() < 0.05


class TestTheSolve:
    def test_the_study_solves_and_the_field_is_physical(self, cap):
        result = cap.cap_study.solve()
        temperature = np.asarray(result.temperature, dtype=float)
        assert np.isfinite(temperature).all()
        # Heat enters at the bore and leaves at the clamped mounting face.
        assert float(result.max()) > 0.1
        assert float(result.min() if hasattr(result, "min") else temperature.min()) > -1e-6

    def test_the_hot_spot_is_at_the_bore_not_the_flange(self, cap, mesh):
        result = cap.cap_study.solve()
        points = np.asarray(result.mesh.points, dtype=float)
        temperature = np.asarray(result.temperature, dtype=float)
        hottest = points[int(np.argmax(temperature))]
        assert np.hypot(hottest[0], hottest[1]) < 0.6, "the hot spot should sit near the bore"
