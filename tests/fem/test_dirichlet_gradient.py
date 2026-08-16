"""Differentiable Dirichlet values (lifted formulation), validated vs FD.

jax-fem bakes prescribed Dirichlet values into the DOF elimination at
problem construction — outside ``ad_wrapper``'s parameter path — so the
direct backend lifts the solve: ``T = u0 + g`` with homogeneous Dirichlet
conditions on ``u0`` and the lift ``g`` entering through ``set_params``.
These tests validate ``d(objective)/d(dirichlet value)`` against central
finite differences on the thermal bar.
"""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax_fem")

import jax
import jax.numpy as jnp

from jaxcad.fem.hexmesh import GridSpec, sdf_to_hex_mesh
from jaxcad.fem.selection import Nodes
from jaxcad.fem.simulate import thermal_solve
from jaxcad.geometry.parameters import Vector
from jaxcad.sdf.primitives import Box

_HOT_END = Nodes.side("-x")
_COLD_END = Nodes.side("+x")


@pytest.fixture(scope="module")
def bar_mesh():
    bar = Box(Vector([1.0, 0.15, 0.15], free=True, name="size"))
    grid = GridSpec.from_bounds((-1.1, -0.25, -0.25), (2.2, 0.5, 0.5), (22, 5, 5))
    return sdf_to_hex_mesh(bar, grid)


class TestDirichletValueGradient:
    def test_forward_matches_unlifted_profile(self, bar_mesh):
        """A traced hot-end value reproduces the exact linear profile."""
        result = thermal_solve(
            bar_mesh,
            conductivity=2.0,
            dirichlet=[(_HOT_END, jnp.asarray(1.0)), (_COLD_END, 0.0)],
        )
        temperature = np.asarray(result.temperature)
        expected = (1.0 - bar_mesh.points[:, 0]) / 2.0
        assert np.abs(temperature - expected).max() < 1e-6

    def test_hot_end_value_gradient_matches_fd(self, bar_mesh):
        """d(sum T^2)/d(T_hot) via the adjoint agrees with central FD."""

        def objective(hot_value):
            result = thermal_solve(
                bar_mesh,
                conductivity=2.0,
                dirichlet=[(_HOT_END, hot_value), (_COLD_END, 0.0)],
            )
            return jnp.sum(result.temperature**2)

        base = 1.0
        grad = float(jax.grad(objective)(jnp.asarray(base)))
        eps = 1e-3
        fd = (float(objective(base + eps)) - float(objective(base - eps))) / (2.0 * eps)
        assert grad == pytest.approx(fd, rel=5e-2)
        # The objective genuinely depends on the value (guard against 0 == 0).
        assert abs(grad) > 1.0

    def test_gradient_flows_to_both_patch_values(self, bar_mesh):
        """Both prescribed values carry adjoint gradients, FD-validated."""

        def objective(values):
            hot, cold = values
            result = thermal_solve(
                bar_mesh,
                conductivity=2.0,
                dirichlet=[(_HOT_END, hot), (_COLD_END, cold)],
            )
            # Mean temperature: linear in both values, nonzero sensitivities.
            return jnp.mean(result.temperature)

        base = jnp.asarray([1.0, 0.2])
        grad = np.asarray(jax.grad(objective)(base))
        eps = 1e-3
        for i in range(2):
            step = np.zeros(2)
            step[i] = eps
            fd = (float(objective(base + step)) - float(objective(base - step))) / (2.0 * eps)
            assert grad[i] == pytest.approx(fd, rel=5e-2)
        assert np.all(np.abs(grad) > 0.05)
