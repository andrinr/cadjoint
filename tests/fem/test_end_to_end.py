"""End-to-end design gradient: parameter -> SDF -> hex mesh -> FEM -> objective.

Frozen-topology doctrine: connectivity, the snapped-vertex set, and BC node
sets are extracted once at the nominal design; per candidate parameter only
the node positions are recomputed (differentiably, through the traced SDF's
Newton projection in :func:`jaxcad.fem.recompute_points`) and fed to the
solver via the ``points`` override.  Gradients flow through jax-fem's
adjoint (``ad_wrapper``); the forward solver itself is not jax-traceable
(PETSc assembly), so this is the adjoint path, exercised through the
default direct in-process backend.
"""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax_fem")

import jax
import jax.numpy as jnp

from jaxcad.fem.hexmesh import GridSpec, recompute_points, sdf_to_hex_mesh
from jaxcad.fem.simulate import elastic_solve
from jaxcad.geometry.parameters import Vector
from jaxcad.sdf.primitives import Box

jax.config.update("jax_enable_x64", True)

_NOMINAL_HALF_HEIGHT = 0.15


def _fixed_end(center):
    return center[0] < -0.999


def _loaded_end(center):
    return center[0] > 0.999


class TestEndToEndGradient:
    def test_grad_of_compliance_wrt_half_height(self):
        # Cantilever bar 2.0 x 0.3 x (2h) along x; design parameter h is the
        # half-height.  Thicker beam -> stiffer -> smaller deflections.
        nominal = Box(Vector([1.0, 0.15, _NOMINAL_HALF_HEIGHT], free=True, name="size"))
        grid = GridSpec.from_bounds((-1.1, -0.25, -0.25), (2.2, 0.5, 0.5), (22, 5, 5))
        mesh = sdf_to_hex_mesh(nominal, grid)
        assert mesh.snap_mask.any()

        def objective(half_height):
            def sdf(p):
                return Box.sdf(p, jnp.array([1.0, 0.15, half_height]))

            points = recompute_points(sdf, mesh)
            result = elastic_solve(
                mesh,
                youngs=1000.0,
                poisson=0.3,
                dirichlet=[_fixed_end],
                tractions=[(_loaded_end, [0.0, 0.0, -1.0])],
                points=points,
            )
            return jnp.sum(result.displacement**2)

        # Differentiate slightly off the nominal 0.15: there the y and z
        # half-extents tie, edge vertices sit on a subgradient kink of the
        # box SDF's max(), and AD returns the subgradient while central FD
        # averages the two one-sided slopes (~5% apart).  At 0.16 the
        # objective is smooth and adjoint vs FD agree to ~1e-4.  The frozen
        # topology explicitly supports off-nominal evaluation.
        at = 0.16
        gradient = float(jax.grad(objective)(jnp.asarray(at, dtype=jnp.float64)))
        assert np.isfinite(gradient)
        # Growing the section must reduce total squared displacement.
        assert gradient < 0.0

        eps = 1e-3
        finite_difference = (float(objective(at + eps)) - float(objective(at - eps))) / (2.0 * eps)
        assert np.isclose(gradient, finite_difference, rtol=5e-2)

    def test_forward_objective_responds_to_parameter(self):
        # Same chain evaluated concretely: the frozen-topology remesh must
        # actually change the physics (guards against a silently constant
        # objective making the gradient test vacuous).
        nominal = Box(Vector([1.0, 0.15, _NOMINAL_HALF_HEIGHT], free=True, name="size"))
        grid = GridSpec.from_bounds((-1.1, -0.25, -0.25), (2.2, 0.5, 0.5), (22, 5, 5))
        mesh = sdf_to_hex_mesh(nominal, grid)

        def compliance(half_height):
            def sdf(p):
                return Box.sdf(p, jnp.array([1.0, 0.15, half_height]))

            points = recompute_points(sdf, mesh)
            result = elastic_solve(
                mesh,
                youngs=1000.0,
                poisson=0.3,
                dirichlet=[_fixed_end],
                tractions=[(_loaded_end, [0.0, 0.0, -1.0])],
                points=points,
            )
            return float(jnp.sum(result.displacement**2))

        thin = compliance(0.12)
        thick = compliance(0.18)
        assert thick < thin * 0.7
