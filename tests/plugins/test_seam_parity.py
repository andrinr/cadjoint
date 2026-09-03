"""The gradient path is unchanged by going through the registry.

The plugin layer is a re-plumbing, not a new numerical path: the frozen
chains now ask for a *kind* instead of importing a ``tesseract_api``
module, and the whole point of the exercise is that the numbers do not
move.  So the tests here run the real seam — ``Optimization.run`` on the
starter heat sink, the scene's own resolution — and compare objectives with
``==``, not ``approx``.

Measured on 2026-09-02 (macOS/arm64, starter hex chain, one step):
``gradient_path="tesseract"`` and its ``"plugins"`` alias both report
J = 0.9886978328015229, and so does a run whose ``thermal_solver`` kind is
filled by a differently-named plugin pointing at the same package — 16
digits, exactly equal, all three.
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

from cadjoint.plugins import PluginSpec, build_registry, get_plugin, set_registry

_REPO = Path(__file__).resolve().parents[2]
_SCENE_PATH = _REPO / "scenes" / "starter.py"
_TESSERACTS = _REPO / "cadjoint" / "fem" / "tesseracts"


@pytest.fixture(scope="module")
def starter():
    """Execute the starter scene the way the compile worker does."""
    from cadjoint.fem import capture_sim_meshes, capture_studies

    namespace: dict = {}
    with capture_sim_meshes(), capture_studies():
        exec(compile(_SCENE_PATH.read_text(), str(_SCENE_PATH), "exec"), namespace, namespace)
    return namespace


def _hex_study(starter):
    """The starter's declared thermal study on a hex ``SimMesh``.

    Hex, because the whole-pipeline mesher's tet mode cannot be re-evaluated
    at a moved design on this scene (TetGen's Steiner count is not
    continuous — ``tests/fem/test_starter_chain.py`` measures it), and this
    test needs a path that actually takes a step.
    """
    from cadjoint.fem import SimMesh, ThermalStudy

    declared = starter["sink_mesh"]
    heat = starter["heat_study"]
    sim_mesh = SimMesh(
        name="plugins-parity-mesh",
        resolution=declared.resolution,
        bounds=declared.bounds,
        domain=declared.domain,
        size=declared.size,
        method="hex",
    )
    return ThermalStudy(
        name="plugins-parity-study",
        conductivity=float(heat.conductivity),
        bcs=list(heat.bcs),
        mesh=sim_mesh,
    )


def _one_step(starter, gradient_path: str) -> float:
    """One optimizer step of the starter on ``gradient_path``; its objective."""
    from cadjoint.optimize import Optimization

    optimization = Optimization(
        f"plugins-parity-{gradient_path}",
        study=_hex_study(starter),
        metric="max",
        gradient_path=gradient_path,
        remesh_every=0,
        steps=1,
        learning_rate=0.004,
    )
    run = optimization.run(1, scene=starter["scene"])
    assert len(run.history) == 1
    assert run.history[0]["grad_norm"] > 0.0
    return float(run.history[0]["objective"])


class TestSeamParity:
    def test_the_plugins_alias_is_the_tesseract_path(self, starter):
        """Same objective to the last bit — the alias renames nothing else."""
        canonical = _one_step(starter, "tesseract")
        alias = _one_step(starter, "plugins")
        print(f"\nstarter hex seam: tesseract J={canonical!r} plugins J={alias!r}")
        assert np.isfinite(canonical)
        assert alias == canonical

    def test_a_renamed_solver_plugin_gives_the_identical_objective(self, starter):
        """The chain really resolves by kind, and resolution costs nothing.

        The ``thermal_solver`` kind is repointed at a plugin registered
        under a *different name* whose spec happens to name the same
        package.  If the chain still imported ``thermal_jaxfem`` directly
        this test could not change anything; because it resolves a kind, it
        does — and the objective is bit-for-bit what it was.
        """
        baseline = _one_step(starter, "tesseract")

        renamed = build_registry(use_entry_points=False)
        renamed.register(
            PluginSpec(
                name="thermal_elsewhere",
                kind="thermal_solver",
                transport="local",
                api_path=_TESSERACTS / "thermal_jaxfem" / "tesseract_api.py",
                version="0.1.0",
            ),
            default=True,
        )
        previous = set_registry(renamed)
        try:
            assert renamed.default_for("thermal_solver") == "thermal_elsewhere"
            through_registry = _one_step(starter, "plugins")
        finally:
            set_registry(previous)
            renamed.close()
        print(f"\nstarter hex seam: renamed-plugin J={through_registry!r}")
        assert through_registry == baseline


class TestChainResolvesKindsNotModules:
    def test_the_chain_asks_for_kinds(self):
        """``chain.py`` resolves slots; the packages fill them."""
        from cadjoint.fem.tesseracts import chain

        assert chain._plugin("mesher").name == "mesher"
        assert chain._plugin("thermal_solver").name == "thermal_jaxfem"
        assert chain._plugin("elastic_solver").name == "elastic_jaxfem"

    def test_a_solver_missing_an_input_the_chain_sends_is_named(self):
        """The capability guard, on the real mismatch it exists for.

        ``elastic_calculix`` declares no ``body_force``, so configuring it
        as the ``elastic_solver`` for a frozen chain has to fail with the
        field named rather than as a validation error inside a traced call.
        """
        from cadjoint.fem.tesseracts.chain import _require_inputs

        calculix = get_plugin("elastic_calculix")
        payload = {"cells": None, "youngs": None, "body_force": None}
        with pytest.raises(ValueError, match=r"body_force"):
            _require_inputs(calculix, payload, "displacement", "elastic_solver")
        # Everything else it does declare passes, including the per-element
        # properties the materials work added.
        _require_inputs(
            calculix,
            {"cells": None, "youngs": None, "cell_youngs": None},
            "displacement",
            "elastic_solver",
        )

    def test_an_output_the_chain_reads_must_exist(self):
        from cadjoint.fem.tesseracts.chain import _require_inputs

        with pytest.raises(ValueError, match=r"does not declare the output 'temperature'"):
            _require_inputs(get_plugin("elastic_jaxfem"), {}, "temperature", "thermal_solver")


class TestBackendsResolveThroughTheRegistry:
    def test_the_tesseract_backend_uses_the_kinds(self):
        from cadjoint.fem.backends import TesseractBackend

        backend = TesseractBackend()
        assert backend._plugin_for("thermal").name == "thermal_jaxfem"
        assert backend._plugin_for("elastic").name == "elastic_jaxfem"

    def test_an_explicit_plugin_name_overrides_the_kind(self):
        from cadjoint.fem.backends import TesseractBackend

        backend = TesseractBackend(elastic="elastic_calculix")
        assert backend._plugin_for("elastic").name == "elastic_calculix"

    def test_a_legacy_api_path_still_works(self):
        """The pre-plugin escape hatch keeps working, as a local spec."""
        from cadjoint.fem.backends import TesseractBackend

        backend = TesseractBackend(
            api_path=_TESSERACTS / "thermal_jaxfem" / "tesseract_api.py",
        )
        plugin = backend._plugin_for("thermal")
        assert plugin.spec.transport == "local"
        assert plugin.schema_hash() == get_plugin("thermal_jaxfem").schema_hash()

    def test_the_calculix_backend_names_a_plugin_not_a_path(self):
        from cadjoint.fem.calculix import CalculixBackend, find_ccx

        if find_ccx() is None:
            pytest.skip("ccx is not installed (set CADJOINT_CCX)")
        backend = CalculixBackend()
        assert backend.plugin_name == "elastic_calculix"
        assert backend._plugin_for("elastic").name == "elastic_calculix"
