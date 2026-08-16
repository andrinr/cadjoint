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
from jaxcad.fem.selection import Nodes
from jaxcad.fem.simulate import thermal_solve
from jaxcad.geometry.parameters import Vector
from jaxcad.sdf.primitives import Box


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

    def test_heat_flux_is_rejected_by_the_schema(self, bar_mesh):
        with pytest.raises(NotImplementedError, match="tesseract schema"):
            thermal_solve(
                bar_mesh,
                conductivity=2.0,
                dirichlet=[(Nodes.side("-x"), 0.0)],
                neumann=[(Nodes.side("+x"), 1.0)],
                backend="tesseract",
            )

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

    def test_gradient_overhead_over_direct_is_bounded(self, bar_mesh):
        """Warm forward+VJP roundtrip through the tesseract boundary stays
        within a generous factor of the direct backend (guards against the
        interop layer regressing into re-building far more than the
        serialization boundary should cost)."""
        import time

        points = jnp.asarray(bar_mesh.points, dtype=jnp.float64)

        def roundtrip(backend):
            def fun(p):
                result = thermal_solve(
                    bar_mesh, conductivity=2.0, dirichlet=_BC, backend=backend, points=p
                )
                return jnp.sum(result.temperature**2)

            return lambda: jax.block_until_ready(jax.grad(fun)(points))

        direct, tesseract = roundtrip(None), roundtrip("tesseract")
        direct()  # warmup (jit compile, petsc init)
        tesseract()
        start = time.perf_counter()
        direct()
        direct_s = time.perf_counter() - start
        start = time.perf_counter()
        tesseract()
        tesseract_s = time.perf_counter() - start
        assert tesseract_s < 20.0 * direct_s + 5.0


class TestTesseractElastic:
    _FIXED = [_hot]
    _TRACTIONS = [(_cold, [0.0, 0.0, -1.0])]

    def _solve(self, mesh, backend=None, points=None):
        from jaxcad.fem.simulate import elastic_solve

        return elastic_solve(
            mesh,
            youngs=1000.0,
            poisson=0.3,
            dirichlet=self._FIXED,
            tractions=self._TRACTIONS,
            backend=backend,
            points=points,
        )

    def test_matches_direct_backend(self, bar_mesh):
        direct = self._solve(bar_mesh)
        tesseract = self._solve(bar_mesh, backend="tesseract")
        difference = np.abs(np.asarray(tesseract.displacement) - np.asarray(direct.displacement))
        assert difference.max() < 1e-9
        # The load genuinely bends the bar (guard against 0 == 0).
        assert np.abs(np.asarray(direct.displacement)).max() > 1e-4

    def test_gradient_through_tesseract_boundary(self, bar_mesh):
        points = jnp.asarray(bar_mesh.points, dtype=jnp.float64)

        def objective(backend):
            def fun(p):
                result = self._solve(bar_mesh, backend=backend, points=p)
                return jnp.sum(result.displacement**2)

            return fun

        grad_direct = jax.grad(objective(None))(points)
        grad_tesseract = jax.grad(objective("tesseract"))(points)
        assert np.abs(np.asarray(grad_tesseract - grad_direct)).max() < 1e-9
        assert float(jnp.linalg.norm(grad_direct)) > 1e-6
