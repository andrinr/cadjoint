"""Tests for the Tesseract interop path (local, no Docker).

The direct in-process jax-fem backend is the default and primary gradient
path; these tests prove the plugin ABI: the packaged thermal tesseract
executes locally via ``Tesseract.from_tesseract_api`` and composes into
``jax.grad`` through ``tesseract_jax.apply_tesseract``.
"""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax_fem")
pytest.importorskip("tesseract_core")
pytest.importorskip("tesseract_jax")

import jax
import jax.numpy as jnp

from jaxcad.fem.hexmesh import GridSpec, sdf_to_hex_mesh
from jaxcad.fem.simulate import thermal_solve
from jaxcad.geometry.parameters import Vector
from jaxcad.sdf.primitives import Box

jax.config.update("jax_enable_x64", True)


def _hot(center):
    return center[0] < -0.999


def _cold(center):
    return center[0] > 0.999


_BC = [(_hot, 1.0), (_cold, 0.0)]


@pytest.fixture(scope="module")
def bar_mesh():
    bar = Box(Vector([1.0, 0.15, 0.15], free=True, name="size"))
    grid = GridSpec.from_bounds((-1.1, -0.25, -0.25), (2.2, 0.5, 0.5), (22, 5, 5))
    return sdf_to_hex_mesh(bar, grid)


class TestTesseractThermal:
    def test_matches_direct_backend(self, bar_mesh):
        direct = thermal_solve(bar_mesh, conductivity=2.0, dirichlet=_BC)
        tesseract = thermal_solve(bar_mesh, conductivity=2.0, dirichlet=_BC, backend="tesseract")
        difference = np.abs(np.asarray(tesseract.temperature) - np.asarray(direct.temperature))
        assert difference.max() < 1e-9

    def test_gradient_through_tesseract_boundary(self, bar_mesh):
        points = jnp.asarray(bar_mesh.points, dtype=jnp.float64)

        def objective(backend):
            def fun(p):
                result = thermal_solve(
                    bar_mesh, conductivity=2.0, dirichlet=_BC, backend=backend, points=p
                )
                return jnp.sum(result.temperature**2)

            return fun

        grad_direct = jax.grad(objective(None))(points)
        grad_tesseract = jax.grad(objective("tesseract"))(points)
        assert np.abs(np.asarray(grad_tesseract - grad_direct)).max() < 1e-9
        assert float(jnp.linalg.norm(grad_direct)) > 1.0

    def test_elastic_not_packaged(self, bar_mesh):
        from jaxcad.fem.backends import get_backend
        from jaxcad.fem.simulate import elastic_solve

        with pytest.raises(NotImplementedError, match="elastic"):
            elastic_solve(
                bar_mesh,
                youngs=1000.0,
                poisson=0.3,
                dirichlet=[_hot],
                tractions=[(_cold, [0.0, 0.0, -1.0])],
                backend=get_backend("tesseract"),
            )
