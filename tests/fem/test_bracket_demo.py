"""Engineering demo: the L-bracket scene and its FEM shape optimization.

Covers the two halves of the demo: ``scenes/bracket.py`` must stay a valid
playground scene whose SDF meshes watertight, and the optimization chain in
``examples/fem_bracket_optimization.py`` must produce a descent direction
whose adjoint gradient matches finite differences at the nominal design.
"""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax_fem")

import jax
import jax.numpy as jnp

_REPO = Path(__file__).resolve().parents[2]
_SCENE_PATH = _REPO / "scenes" / "bracket.py"
_EXAMPLE_PATH = _REPO / "examples" / "fem_bracket_optimization.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("fem_bracket_optimization", _EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo():
    return _load_example()


@pytest.fixture(scope="module")
def nominal_state(demo):
    """Frozen mesh, objective, and value+gradient at the nominal design."""
    mesh = demo.build_mesh()
    objective = demo.make_objective(mesh)
    theta = jnp.array([demo.NOMINAL_WEB_THICKNESS, demo.NOMINAL_RIB_HEIGHT], dtype=jnp.float64)
    value, gradient = jax.value_and_grad(objective)(theta)
    return demo, objective, theta, value, gradient


class TestBracketScene:
    def test_scene_compiles_in_the_playground(self):
        from cadjoint.viewer.playground import compile_source

        result = compile_source(_SCENE_PATH.read_text())
        assert result.get("ok"), result.get("error")
        assert result.get("scene_wgsl")

    def test_scene_sdf_meshes_watertight(self):
        from cadjoint.meshing import GridSpec, extract_mesh, mesh_report

        namespace: dict = {}
        exec(compile(_SCENE_PATH.read_text(), str(_SCENE_PATH), "exec"), namespace, namespace)
        scene = namespace["scene"]
        grid = GridSpec.from_bounds((-1.45, -1.05, -0.25), (2.9, 2.1, 1.75), (58, 42, 40))
        mesh = extract_mesh(lambda p: scene(p), grid)
        report = mesh_report(lambda p: scene(p), mesh)
        assert report["watertight"]
        # Genus 2: the two bolt holes.
        assert report["euler_characteristic"] == -2


class TestBracketOptimization:
    def test_one_gradient_step_decreases_the_objective(self, nominal_state):
        demo, objective, theta, value, gradient = nominal_state
        assert np.all(np.isfinite(np.asarray(gradient)))
        stepped = theta - jnp.asarray(demo._LEARNING_RATES, dtype=jnp.float64) * gradient
        assert float(objective(stepped)) < float(value)

    def test_adjoint_gradient_matches_finite_differences(self, nominal_state):
        _, objective, theta, _, gradient = nominal_state
        eps = 1e-3
        for index in range(2):
            offset = jnp.zeros_like(theta).at[index].set(eps)
            finite_difference = (
                float(objective(theta + offset)) - float(objective(theta - offset))
            ) / (2.0 * eps)
            assert np.isclose(float(gradient[index]), finite_difference, rtol=5e-2)
