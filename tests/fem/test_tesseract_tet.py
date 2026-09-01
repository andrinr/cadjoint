"""Solver-tesseract parity on tet meshes (TET4/TET10) and thermal fluxes.

The packaged ``elastic_jaxfem`` / ``thermal_jaxfem`` tesseracts accept
element-agnostic ``cells`` (K = 4/8/10) plus exact-face targeting and
heat-flux patches; these tests pin them to the direct in-process path:
same displacement/temperature to 1e-9, same adjoint gradient through the
tesseract boundary.
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

from cadjoint.fem.backends import ElasticBCs, ThermalBCs
from cadjoint.fem.selection import Nodes
from cadjoint.fem.tetmesh import (
    recompute_tet_points,
    sdf_to_tet_mesh,
    tet10_complete_nodes,
    tet10_face_midsides,
    tet10_mesh,
    tet_elastic_solve,
    tet_faces_from_nodes,
    tet_thermal_solve,
)
from cadjoint.meshing import GridSpec

_TESSERACTS = Path(__file__).parents[2] / "cadjoint" / "fem" / "tesseracts"
_BAR_GRID = GridSpec.from_bounds((-0.65, -0.32, -0.3), (1.3, 0.64, 0.6), (13, 7, 6))


def _bar_sdf(half_height: float = 0.16):
    def sdf(p):
        p = jnp.asarray(p)
        q = jnp.abs(p) - jnp.stack([jnp.asarray(0.5), jnp.asarray(half_height), jnp.asarray(0.15)])
        return jnp.max(q, axis=-1)

    return sdf


@pytest.fixture(scope="module")
def bar_meshes():
    mesh4 = sdf_to_tet_mesh(_bar_sdf(), _BAR_GRID)
    return {"TET4": mesh4, "TET10": tet10_mesh(mesh4)}


@pytest.fixture(scope="module")
def elastic_tesseract():
    from tesseract_core import Tesseract

    return Tesseract.from_tesseract_api(str(_TESSERACTS / "elastic_jaxfem" / "tesseract_api.py"))


@pytest.fixture(scope="module")
def thermal_tesseract():
    from tesseract_core import Tesseract

    return Tesseract.from_tesseract_api(str(_TESSERACTS / "thermal_jaxfem" / "tesseract_api.py"))


def _elastic_setup(mesh):
    """Clamp -x end, pull the +x end down; exact face targeting."""
    clamp = tet10_complete_nodes(
        mesh, Nodes.halfspace([-0.49, 0.0, 0.0], [-1.0, 0.0, 0.0]).resolve(mesh)
    )
    tip = Nodes.halfspace([0.49, 0.0, 0.0], [1.0, 0.0, 0.0]).resolve(mesh)
    faces = tet_faces_from_nodes(mesh, tip)
    span = np.unique(faces)
    if mesh.edge_parents is not None:
        span = np.concatenate([span, np.unique(tet10_face_midsides(mesh, faces))])
    bcs = ElasticBCs(
        fixed_nodes=[clamp],
        traction_nodes=[span.astype(np.int32)],
        traction_vectors=[np.array([0.0, 0.0, -1.0])],
    )
    inputs = {
        "cells": np.asarray(mesh.cells, dtype=np.int32),
        "fixed_nodes": clamp.astype(np.int32),
        "traction_nodes": span.astype(np.int32),
        "traction_offsets": np.array([0, len(span)], np.int32),
        "traction_vectors": np.array([[0.0, 0.0, -1.0]]),
        "traction_faces": faces.astype(np.int32),
        "traction_face_offsets": np.array([0, len(faces)], np.int32),
        "youngs": np.float64(1000.0),
        "poisson": np.float64(0.3),
    }
    return bcs, faces, inputs


def _thermal_setup(mesh):
    """Dirichlet 0 on the -x end, unit heat inflow on the +x end faces."""
    cold = tet10_complete_nodes(
        mesh, Nodes.halfspace([-0.49, 0.0, 0.0], [-1.0, 0.0, 0.0]).resolve(mesh)
    )
    hot = Nodes.halfspace([0.49, 0.0, 0.0], [1.0, 0.0, 0.0]).resolve(mesh)
    faces = tet_faces_from_nodes(mesh, hot)
    span = np.unique(faces)
    if mesh.edge_parents is not None:
        span = np.concatenate([span, np.unique(tet10_face_midsides(mesh, faces))])
    bcs = ThermalBCs(
        dirichlet_nodes=[cold],
        dirichlet_values=[0.0],
        flux_nodes=[span.astype(np.int32)],
        flux_values=[1.0],
    )
    inputs = {
        "cells": np.asarray(mesh.cells, dtype=np.int32),
        "dirichlet_nodes": cold.astype(np.int32),
        "dirichlet_values": np.zeros(len(cold)),
        "flux_nodes": span.astype(np.int32),
        "flux_offsets": np.array([0, len(span)], np.int32),
        "flux_values": np.array([1.0]),
        "flux_faces": faces.astype(np.int32),
        "flux_face_offsets": np.array([0, len(faces)], np.int32),
        "conductivity": np.float64(2.0),
        "source": np.float64(0.0),
    }
    return bcs, faces, inputs


class TestElasticTesseractOnTets:
    @pytest.mark.parametrize("ele_type", ["TET4", "TET10"])
    def test_apply_matches_direct_solve(self, bar_meshes, elastic_tesseract, ele_type):
        mesh = bar_meshes[ele_type]
        bcs, faces, inputs = _elastic_setup(mesh)
        direct = np.asarray(
            tet_elastic_solve(
                mesh.points,
                mesh.cells,
                bcs,
                youngs=1000.0,
                poisson=0.3,
                ele_type=ele_type,
                traction_faces=[faces],
            )
        )
        result = elastic_tesseract.apply(dict(points=np.asarray(mesh.points), **inputs))
        packaged = np.asarray(result["displacement"])
        assert np.abs(packaged - direct).max() < 1e-9
        assert np.abs(direct).max() > 1e-4  # a real bend, not 0 == 0

    def test_traced_gradient_matches_direct_adjoint(self, bar_meshes, elastic_tesseract):
        from tesseract_jax import apply_tesseract

        mesh = bar_meshes["TET10"]
        bcs, faces, inputs = _elastic_setup(mesh)
        points = jnp.asarray(mesh.points, dtype=jnp.float64)

        def packaged(p):
            outputs = apply_tesseract(elastic_tesseract, dict(points=p, **inputs))
            return jnp.sum(outputs["displacement"] ** 2)

        def direct(p):
            displacement = tet_elastic_solve(
                p,
                mesh.cells,
                bcs,
                youngs=1000.0,
                poisson=0.3,
                ele_type="TET10",
                base_points=np.asarray(mesh.points),
                traction_faces=[faces],
            )
            return jnp.sum(displacement**2)

        grad_packaged = jax.grad(packaged)(points)
        grad_direct = jax.grad(direct)(points)
        assert np.abs(np.asarray(grad_packaged - grad_direct)).max() < 1e-9
        assert float(jnp.linalg.norm(grad_direct)) > 1e-6


class TestThermalTesseractOnTets:
    @pytest.mark.parametrize("ele_type", ["TET4", "TET10"])
    def test_apply_matches_direct_solve(self, bar_meshes, thermal_tesseract, ele_type):
        mesh = bar_meshes[ele_type]
        bcs, faces, inputs = _thermal_setup(mesh)
        direct = np.asarray(
            tet_thermal_solve(
                mesh.points,
                mesh.cells,
                bcs,
                conductivity=2.0,
                ele_type=ele_type,
                flux_faces=[faces],
            )
        )
        result = thermal_tesseract.apply(dict(points=np.asarray(mesh.points), **inputs))
        packaged = np.asarray(result["temperature"])
        assert np.abs(packaged - direct).max() < 1e-9
        # The flux genuinely heats the far end (exact solution T = (q/k)(x + 1/2)).
        x = np.asarray(mesh.points)[:, 0]
        assert np.abs(direct - 0.5 * (x + 0.5)).max() < 1e-6

    def test_traced_gradient_matches_direct_adjoint(self, bar_meshes, thermal_tesseract):
        from tesseract_jax import apply_tesseract

        mesh = bar_meshes["TET10"]
        bcs, faces, inputs = _thermal_setup(mesh)
        points = jnp.asarray(mesh.points, dtype=jnp.float64)

        def packaged(p):
            outputs = apply_tesseract(thermal_tesseract, dict(points=p, **inputs))
            return jnp.sum(outputs["temperature"] ** 2)

        def direct(p):
            temperature = tet_thermal_solve(
                p,
                mesh.cells,
                bcs,
                conductivity=2.0,
                ele_type="TET10",
                base_points=np.asarray(mesh.points),
                flux_faces=[faces],
            )
            return jnp.sum(temperature**2)

        grad_packaged = jax.grad(packaged)(points)
        grad_direct = jax.grad(direct)(points)
        assert np.abs(np.asarray(grad_packaged - grad_direct)).max() < 1e-9
        assert float(jnp.linalg.norm(grad_direct)) > 1e-6

    def test_design_gradient_through_reprojection(self, bar_meshes, thermal_tesseract):
        """d(sum T^2)/d(half_height) through recompute: tesseract == direct.

        The bar's DC vertices sit on box-SDF creases, so central FD
        legitimately disagrees with the one-sided adjoint (measured in
        ``research/tet-vs-hex.md``); the tesseract boundary is isolated by
        comparing the two solver routes on the identical recompute chain.
        """
        from tesseract_jax import apply_tesseract

        mesh = bar_meshes["TET4"]
        bcs, faces, inputs = _thermal_setup(mesh)

        def objective(solver):
            def fun(half_height):
                points = recompute_tet_points(_bar_sdf(half_height), mesh)
                return jnp.sum(solver(points) ** 2)

            return fun

        def packaged(points):
            return apply_tesseract(thermal_tesseract, dict(points=points, **inputs))["temperature"]

        def direct(points):
            return tet_thermal_solve(
                points,
                mesh.cells,
                bcs,
                conductivity=2.0,
                base_points=np.asarray(mesh.points),
                flux_faces=[faces],
            )

        grad_packaged = float(jax.grad(objective(packaged))(jnp.asarray(0.16)))
        grad_direct = float(jax.grad(objective(direct))(jnp.asarray(0.16)))
        assert grad_packaged == pytest.approx(grad_direct, rel=1e-9)
        assert abs(grad_direct) > 1.0  # the design parameter is live
