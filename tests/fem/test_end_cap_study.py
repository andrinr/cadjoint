"""The end-cap still meshes and solves after all that modelling.

A part is only "complex" in a useful sense if the complexity survives
discretization. This is the end of the chain: the declared ``SimMesh`` cuts
the lattice's cells with the housing (no volume mesh), both boundary-condition
selections find real surface nodes *and* real boundary facets, and the
declared thermal study solves on them.

The specific failure this guards against is silent: a selection that matches
scattered surface nodes but no boundary facet resolves fine and then
integrates nothing, so a load quietly does not exist.
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

    def test_the_mesh_is_cut_cells_with_nothing_to_grade(self, cap, mesh):
        from cadjoint.fem.cutfem import CutMesh

        assert isinstance(mesh, CutMesh)
        info = cap.cap_mesh.inspect()
        assert info["method"] == "cutfem"
        assert info["elements"] > 500  # active lattice cells
        assert info["quality"] == {}

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

    def test_every_condition_selects_boundary_facets(self, cap, mesh):
        """An area-integrated BC that catches no facet integrates nothing."""
        from cadjoint.fem.cutfem import unresolvable_condition

        assert unresolvable_condition(cap.cap_study.bcs, mesh) is None
        flux = next(bc for bc in cap.cap_study.bcs if type(bc).__name__ == "HeatFlux")
        assert flux.nodes.contains(mesh.structure.centroids).sum() > 20

    def test_the_ambient_face_is_the_mounting_face(self, cap, mesh):
        dirichlet = next(bc for bc in cap.cap_study.bcs if type(bc).__name__ == "Dirichlet")
        picked = np.asarray(mesh.points, dtype=float)[dirichlet.nodes.resolve(mesh)]
        assert picked[:, 2].max() < 0.05


class TestTheSolve:
    @pytest.fixture(scope="class")
    def result(self, cap):
        """One solve for the whole class.

        Both tests below read the same field; solving twice measured 7.1 s of
        setup and bought nothing, because ``solve()`` is deterministic and
        neither test mutates what it returns.
        """
        return cap.cap_study.solve()

    def test_the_study_solves_and_the_field_is_physical(self, result):
        temperature = np.asarray(result.temperature, dtype=float)
        assert np.isfinite(temperature).all()
        # Heat enters at the bore and leaves at the clamped mounting face.
        assert float(result.max()) > 0.1
        # Nitsche holds the ambient face weakly: a small undershoot below 0 is
        # the method, not a sign error.
        assert temperature.min() > -0.02 * temperature.max()

    def test_the_hot_spot_is_at_the_bore_not_the_flange(self, result):
        points = np.asarray(result.mesh.points, dtype=float)
        temperature = np.asarray(result.temperature, dtype=float)
        hottest = points[int(np.argmax(temperature))]
        assert np.hypot(hottest[0], hottest[1]) < 0.6, "the hot spot should sit near the bore"
