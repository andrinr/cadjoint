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
    pytest.importorskip("optax")  # imported by the example at module level
    return _load_example()


@pytest.fixture(scope="module")
def named_mesh_state(demo):
    """SimMesh-backed study, objective, and value+gradient at the nominal design.

    The named-mesh version of the bracket gradient chain: the SimMesh builds
    (and caches) the frozen-topology hex mesh at the nominal design, the
    study solves on it through ``mesh=``, and per candidate ``theta`` only
    the node positions are recomputed differentiably and passed via
    ``points=`` — proving d(objective)/d(CAD parameter) flows through
    SimMesh.build() -> BC node selection -> solve.
    """
    from cadjoint.fem import ElasticStudy, Fixed, SimMesh, Traction, recompute_points

    theta0 = jnp.asarray(demo.NOMINAL, dtype=jnp.float64)
    nominal_sdf = demo.theta_sdf(np.asarray(theta0))
    sim_mesh = SimMesh(
        name="bracket-grad-mesh",
        resolution=(24, 17, 13),
        domain=nominal_sdf,
        bounds=(-1.3, -0.95, -0.06),
        size=(2.6, 1.9, 1.42),
    )
    study = ElasticStudy(
        name="bracket-pry-named",
        youngs=demo._YOUNGS,
        poisson=demo._POISSON,
        bcs=[
            Fixed(demo.BOLT_CLAMP),
            Traction(demo.WEB_TIP_LOAD, demo._TRACTION),
        ],
        mesh=sim_mesh,
    )
    hex_mesh = sim_mesh.build()  # topology frozen at the nominal design
    assert hex_mesh.snap_mask.any()

    def objective(theta):
        points = recompute_points(demo.theta_sdf(theta), hex_mesh)
        result = study.solve(points=points)  # SimMesh cache serves the frozen mesh
        return result.mean()  # mean displacement magnitude (differentiable helper)

    value, gradient = jax.value_and_grad(objective)(theta0)
    return study, sim_mesh, objective, theta0, value, gradient


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
        grid = GridSpec.from_bounds((-1.45, -1.05, -0.25), (2.9, 2.1, 1.75), (58, 42, 40))

        # Full scene: bracket + mounting slab.  The slab plugs the bolt holes
        # from below (bolts thread into it), so the merged surface is genus 0.
        scene = namespace["scene"]
        report = mesh_report(lambda p: scene(p), extract_mesh(lambda p: scene(p), grid))
        assert report["watertight"]
        assert report["euler_characteristic"] == 2

        # Simulation domain: the bracket alone keeps its two through-holes
        # (genus 2) — this is the field the SimMesh actually hexes.
        bracket = namespace["bracket"]
        report = mesh_report(lambda p: bracket(p), extract_mesh(lambda p: bracket(p), grid))
        assert report["watertight"]
        assert report["euler_characteristic"] == -2

    def test_declared_mesh_and_study_are_wired(self):
        namespace: dict = {}
        exec(compile(_SCENE_PATH.read_text(), str(_SCENE_PATH), "exec"), namespace, namespace)
        sim_mesh = namespace["bracket_mesh"]
        study = namespace["pry_study"]
        assert study.mesh is sim_mesh
        assert sim_mesh.domain is namespace["bracket"]
        assert study.describe()["mesh"] == "bracket-mesh"
        assert study.describe()["domain"] == {"name": "bracket", "type": "Difference"}
        # The mesh is inspectable without any scene SDF (its domain suffices).
        report = sim_mesh.inspect()
        assert report["elements"] > 0
        assert report["quality"]["scaled_jacobian"]["min"] > 0.0


class TestNamedMeshGradient:
    """Bracket adjoint gradient through the named-mesh path.

    Successor of the old example-driven optimization test (the two-step
    Adam smoke now lives next to the example in
    ``examples/test_fem_bracket_optimization.py``): the same physics, but
    routed through SimMesh + ElasticStudy(mesh=...) + SimulationResult.
    """

    def test_gradient_is_finite_and_physical(self, named_mesh_state):
        study, sim_mesh, _, _, value, gradient = named_mesh_state
        assert float(value) > 0.0
        assert np.all(np.isfinite(np.asarray(gradient)))
        # Thickening the plate stiffens the bracket: displacements shrink.
        # (The web component's sign is not invariant for mean |u|: the load
        # arm moves with the outer wall, so only FD agreement pins it down.)
        assert float(gradient[2]) < 0.0
        # The traced solve also landed on the study as its last result.
        assert study.last_result is not None
        assert study.last_result.sim_mesh is sim_mesh

    def test_adjoint_gradient_matches_finite_differences(self, named_mesh_state):
        _, _, objective, theta, _, gradient = named_mesh_state
        eps = 1e-3
        for index in range(theta.shape[0]):
            offset = jnp.zeros_like(theta).at[index].set(eps)
            finite_difference = (
                float(objective(theta + offset)) - float(objective(theta - offset))
            ) / (2.0 * eps)
            assert np.isclose(float(gradient[index]), finite_difference, rtol=5e-2), (
                index,
                float(gradient[index]),
                finite_difference,
            )
