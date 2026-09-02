"""The flow solver as a plugin: packaging conformance, and the VJP contract.

Two tiers, the same split ``tests/fem/test_tesseract_packaging.py`` uses:
the config tier validates the package against the installed tesseract-core
schema and always runs; the round-trip tier loads the API in-process through
the registry and is skipped when the ``tesseract`` extra is absent.

``flow_brinkman`` is not in that suite's ``PACKAGES`` table -- it is a
``flow_solver``, not an FEM solver, and its tests belong with the rest of
the flow package -- so the config checks are restated here rather than
inherited.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tesseract_core")

from tesseract_core.sdk.api_parse import (  # noqa: E402
    TesseractConfig,
    get_config,
    validate_tesseract_api,
)

_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = _ROOT / "cadjoint" / "fem" / "tesseracts" / "flow_brinkman"

SHAPE = (6, 12, 6)
SETTINGS = {
    "reynolds": 10.0,
    "characteristic_cells": 6,
    "alpha_max": 200.0,
    "cell_volume": 1.0,
    "max_steps": 6000,
    "tol": 1e-12,
    "adjoint_solver": "gmres",
    "adjoint_tol": 1e-11,
    "adjoint_max_steps": 2000,
    "adjoint_restart": 30,
}


@pytest.fixture(scope="module")
def payload():
    chi = np.zeros(SHAPE)
    chi[2:4, 5:8, 2:4] = 1.0
    return {"chi": chi, "inlet_velocity": np.array([0.0, 0.02, 0.0]), **SETTINGS}


@pytest.fixture(scope="module")
def in_process():
    """The same problem solved directly, as the parity reference."""
    from cadjoint.flow import FlowConfig, SteadyOptions

    return FlowConfig(
        shape=SHAPE,
        inlet_speed=0.02,
        reynolds=10.0,
        characteristic_cells=6,
        alpha_max=200.0,
        steady=SteadyOptions(
            max_steps=6000,
            tol=1e-12,
            check_every=50,
            adjoint_solver="gmres",
            adjoint_tol=1e-11,
            adjoint_max_steps=2000,
            adjoint_restart=30,
        ),
    )


class TestPackaging:
    def test_the_config_parses_under_the_installed_schema(self):
        config = get_config(_PACKAGE)
        assert isinstance(config, TesseractConfig)
        assert config.name == "cadjoint_flow_brinkman"
        assert config.version == "0.1.0"
        assert config.description.strip()

    def test_the_endpoints_validate(self):
        """The SDK's own AST check of apply / abstract_eval / vjp."""
        validate_tesseract_api(_PACKAGE)

    def test_the_requirements_file_matches_the_declared_provider(self):
        config = get_config(_PACKAGE)
        expected = {
            "python-pip": "tesseract_requirements.txt",
            "conda": "tesseract_environment.yaml",
        }[config.build_config.requirements.provider]
        assert (_PACKAGE / expected).is_file()
        stale = {"tesseract_requirements.txt", "tesseract_environment.yaml"} - {expected}
        for name in stale:
            assert not (_PACKAGE / name).exists(), f"stale {name} would be silently ignored"

    def test_it_uses_pip_because_it_needs_no_conda_only_dependency(self):
        """The FEM solvers need conda for petsc4py and gmsh; this one is
        pure JAX on a fixed array and installs from wheels anywhere."""
        assert get_config(_PACKAGE).build_config.requirements.provider == "python-pip"

    def test_the_local_cadjoint_dependency_resolves_to_the_repository_root(self):
        """An off-by-one ``../`` only shows up as a build failure otherwise."""
        lines = (_PACKAGE / "tesseract_requirements.txt").read_text().splitlines()
        local = [line.strip() for line in lines if line.strip().startswith((".", "/", "file://"))]
        assert local, "the package must install cadjoint itself"
        for line in local:
            assert (_PACKAGE / line.split("[")[0]).resolve() == _ROOT


class TestRegistration:
    def test_flow_solver_is_a_kind_the_registry_resolves(self):
        from cadjoint.plugins import KINDS

        assert "flow_solver" in KINDS

    def test_the_builtin_spec_points_at_this_package(self):
        from cadjoint.plugins import builtin_specs

        spec = builtin_specs()["flow_brinkman"]
        assert spec.kind == "flow_solver"
        assert spec.transport == "local"
        assert Path(spec.api_path) == _PACKAGE / "tesseract_api.py"
        assert spec.version == "0.1.0"

    def test_it_is_the_default_for_its_kind(self):
        from cadjoint.plugins import plugin_for_kind

        assert plugin_for_kind("flow_solver").name == "flow_brinkman"


class TestRoundTrip:
    """Loaded through ``Tesseract.from_tesseract_api`` and driven as a plugin."""

    @pytest.fixture(scope="class")
    def plugin(self):
        from cadjoint.plugins import plugin_for_kind

        return plugin_for_kind("flow_solver")

    def test_the_served_schema_declares_the_differentiable_inputs(self, plugin):
        probe = plugin.probe()
        assert probe.status == "ok"
        assert {"apply", "abstract_eval", "vector_jacobian_product"} <= set(probe.endpoints)

    def test_apply_matches_the_in_process_solve(self, plugin, payload, in_process):
        from cadjoint.flow import solve

        served = plugin.apply(payload)
        direct = solve(
            jnp.asarray(payload["chi"]),
            in_process,
            inlet_velocity=jnp.asarray(payload["inlet_velocity"]),
        )
        assert float(np.asarray(served["pressure_drop"])) == pytest.approx(
            float(direct.pressure_drop), rel=1e-12
        )
        assert float(np.asarray(served["heat_transfer"])) == pytest.approx(
            float(direct.heat_transfer), rel=1e-12
        )

    @pytest.mark.parametrize("objective", ["pressure_drop", "heat_transfer"])
    def test_the_vjp_matches_the_in_process_gradient(self, plugin, payload, in_process, objective):
        """The contract that makes this a plugin rather than a script.

        Both sides run the same implicit-function-theorem adjoint, so the
        agreement should be exact rather than merely close.
        """
        from cadjoint.flow import solve

        served = plugin.vjp(payload, ["chi"], [objective], {objective: np.array(1.0)})
        direct = jax.grad(
            lambda c: getattr(
                solve(c, in_process, inlet_velocity=jnp.asarray(payload["inlet_velocity"])),
                objective,
            )
        )(jnp.asarray(payload["chi"]))
        assert np.asarray(served["chi"]) == pytest.approx(np.asarray(direct), rel=1e-10, abs=1e-14)

    def test_the_inlet_velocity_gradient_points_along_the_duct(self, plugin, payload):
        """Pushing harder costs more pressure; pushing sideways costs nothing."""
        served = plugin.vjp(
            payload, ["inlet_velocity"], ["pressure_drop"], {"pressure_drop": np.array(1.0)}
        )
        gradient = np.asarray(served["inlet_velocity"])
        assert gradient[1] > 0.0
        assert abs(gradient[0]) < 1e-9 * gradient[1]
        assert abs(gradient[2]) < 1e-9 * gradient[1]

    def test_a_non_differentiable_input_is_refused_by_name(self, plugin, payload):
        with pytest.raises(Exception, match="reynolds"):
            plugin.vjp(payload, ["reynolds"], ["pressure_drop"], {"pressure_drop": np.array(1.0)})
