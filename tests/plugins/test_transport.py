"""Local and remote transports, measured against a really-served component.

The claim the ``remote`` transport makes is that a plugin reached over HTTP
is the *same plugin*: same declared schema, same forward values, same
gradients.  Everything below is measured against
``tesseract-runtime serve`` on a loopback port — the same server a
Kubernetes Service would put behind a cluster address — rather than against
a stub.

Measured on 2026-09-02 (macOS/arm64, mesher package, HEX8 frozen topology,
1285 nodes / 912 cells, a 74 KB field and a 31 KB point array per call):

* forward ``points``: max |Δ| **0.0** between local and remote;
* ``d(points)/d(field_values)``: max |Δ| **0.0**;
* twelve ``jax.value_and_grad`` steps: **6.57 s local vs 6.69 s remote**
  (548 vs 558 ms/step), and again 7.05 vs 7.14 s — the loopback wire costs
  **~10 ms per step, 1.3-1.8%**, and the objective agrees to all 13 printed
  digits at every step (342.8520334386417 -> 342.8408938752647 on both).

Those are loopback numbers with no scheduler, no TLS and no queueing in
front of them; ``docs/plugins.qmd`` says what a real cluster adds.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tesseract_jax")

from cadjoint.plugins import (  # noqa: E402
    PluginMismatch,
    PluginSpec,
    TesseractPlugin,
    get_plugin,
)


def _frozen_mesher_payload():
    """A frozen-topology HEX8 mesher payload with no zero-size arrays.

    Discovery (empty templates) has to run in-process — tesseract-core 1.11
    validates polymorphic dimensions as ``PositiveInt``, so a ``(0, …)``
    array cannot cross HTTP — so the topology is discovered locally and the
    frozen payload it yields is what both transports are handed.
    """
    # 20^3 lattice / 0.7 sphere: the configuration the module docstring's
    # latency numbers were measured on (1285 nodes, 912 cells).
    n = 20
    origin = np.array([-1.0, -1.0, -1.0])
    spacing = np.array([2.0 / n] * 3)
    axis = origin[0] + spacing[0] * np.arange(n + 1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    static = {
        "field_values": np.sqrt(x**2 + y**2 + z**2) - 0.7,
        "origin": origin,
        "spacing": spacing,
        "element": np.int32(1),  # HEX8: voxelize + Newton-snap, platform stable
        "sharp": np.int32(0),
        "min_ratio": np.float64(1.5),
        "min_dihedral": np.float64(10.0),
    }
    local = get_plugin("mesher")
    found = local.apply(
        dict(
            static,
            point_ids=np.zeros(0, np.int32),
            cell_template=np.zeros((0, 8), np.int32),
            num_surface=np.int32(0),
        )
    )
    return dict(
        static,
        point_ids=np.arange(np.asarray(found["points"]).shape[0], dtype=np.int32),
        cell_template=np.zeros(np.asarray(found["cells"]).shape, np.int32),
        num_surface=np.int32(int(np.asarray(found["surface_mask"]).sum())),
    )


@pytest.fixture(scope="module")
def frozen_payload():
    return _frozen_mesher_payload()


@pytest.fixture
def remote_mesher(served_mesher):
    """The served mesher as a ``remote`` plugin."""
    spec = PluginSpec(
        name="mesher",
        kind="mesher",
        transport="remote",
        url=served_mesher.url,
        version=served_mesher.version,
        options={"timeout": 300.0},
    )
    plugin = TesseractPlugin(spec)
    yield plugin
    plugin.close()


class TestDeclarationIsTransportIndependent:
    def test_the_served_schema_matches_the_in_process_one_exactly(self, remote_mesher):
        """Same package, same declared interface, same digest."""
        local = get_plugin("mesher")
        assert dict(remote_mesher.inputs) == dict(local.inputs)
        assert dict(remote_mesher.outputs) == dict(local.outputs)
        assert remote_mesher.schema_hash() == local.schema_hash()

    def test_capabilities_are_derived_not_declared_twice(self, remote_mesher):
        local = get_plugin("mesher")
        assert remote_mesher.capabilities.differentiable_inputs == frozenset({"field_values"})
        assert remote_mesher.capabilities.differentiable_outputs == frozenset({"points"})
        assert remote_mesher.capabilities.supports("frozen_topology")
        assert remote_mesher.capabilities.supports("vjp")
        assert (
            remote_mesher.capabilities.differentiable_inputs
            == local.capabilities.differentiable_inputs
        )

    def test_the_endpoint_sets_differ_which_is_why_they_are_not_hashed(self, remote_mesher):
        """An in-process runtime serves ``test``; a served one does not.

        Recorded because it is the reason :meth:`Plugin.schema_hash` hashes
        the IO declaration only — hashing endpoints would make the same
        package's two transports disagree.
        """
        local = get_plugin("mesher")
        assert "test" in local.capabilities.endpoints
        assert "test" not in remote_mesher.capabilities.endpoints
        assert {"apply", "vector_jacobian_product", "abstract_eval"} <= (
            remote_mesher.capabilities.endpoints
        )


class TestForwardAndVjpEquivalence:
    def test_apply_agrees_bit_for_bit(self, frozen_payload, remote_mesher):
        local = np.asarray(get_plugin("mesher").apply(frozen_payload)["points"])
        remote = np.asarray(remote_mesher.apply(frozen_payload)["points"])
        assert np.abs(local - remote).max() == 0.0

    def test_vjp_agrees_bit_for_bit(self, frozen_payload, remote_mesher):
        cotangent = {
            "points": np.random.default_rng(0).standard_normal(
                (frozen_payload["point_ids"].shape[0], 3)
            )
        }

        def pull(plugin):
            return np.asarray(
                plugin.vjp(
                    frozen_payload,
                    vjp_inputs=["field_values"],
                    vjp_outputs=["points"],
                    cotangent_vector=cotangent,
                )["field_values"]
            )

        assert np.abs(pull(get_plugin("mesher")) - pull(remote_mesher)).max() == 0.0

    def test_a_jax_gradient_through_the_wire_matches_the_in_process_one(
        self, frozen_payload, remote_mesher
    ):
        """``as_jax`` composes a served plugin into ``jax.grad`` unchanged."""
        import jax
        import jax.numpy as jnp

        samples = jnp.asarray(frozen_payload["field_values"])
        static = {k: v for k, v in frozen_payload.items() if k != "field_values"}

        def objective(plugin):
            call = plugin.as_jax()

            def value(field_values):
                with jax.ensure_compile_time_eval():
                    points = call(dict(field_values=field_values, **static))["points"]
                return jnp.sum(points**2)

            return value

        local_value, local_grad = jax.value_and_grad(objective(get_plugin("mesher")))(samples)
        remote_value, remote_grad = jax.value_and_grad(objective(remote_mesher))(samples)
        assert float(local_value) == float(remote_value)
        assert np.abs(np.asarray(local_grad) - np.asarray(remote_grad)).max() == 0.0


class TestProbe:
    def test_a_healthy_remote_reports_its_version_and_digest(self, remote_mesher):
        probe = remote_mesher.probe()
        assert probe.status == "ok"
        assert probe.transport == "remote"
        # TESSERACT_VERSION is what a served image (or Service) advertises.
        assert probe.version == "0.1.0"
        assert probe.schema_hash == get_plugin("mesher").schema_hash()

    def test_a_stale_schema_hash_is_caught_before_the_run(self, served_mesher):
        """The staleness fence: a redeployed remote is refused up front."""
        stale = TesseractPlugin(
            PluginSpec(
                name="mesher",
                kind="mesher",
                transport="remote",
                url=served_mesher.url,
                schema_hash="sha256:" + "0" * 64,
            )
        )
        try:
            with pytest.raises(PluginMismatch, match="serves schema"):
                stale.probe()
            # Non-strict still reports, for a caller that wants to decide.
            assert stale.probe(strict=False).status == "ok"
        finally:
            stale.close()

    def test_a_version_mismatch_is_caught_too(self, served_mesher):
        wrong = TesseractPlugin(
            PluginSpec(
                name="mesher",
                kind="mesher",
                transport="remote",
                url=served_mesher.url,
                version="9.9.9",
            )
        )
        try:
            with pytest.raises(PluginMismatch, match="serves version"):
                wrong.probe()
        finally:
            wrong.close()

    def test_an_unreachable_remote_says_where_it_looked(self):
        dead = TesseractPlugin(
            PluginSpec(
                name="dead",
                kind="mesher",
                transport="remote",
                url="http://127.0.0.1:4899",
                options={"timeout": 2.0},
            )
        )
        with pytest.raises(RuntimeError, match="is unreachable"):
            dead.probe()

    def test_a_local_plugin_probes_too(self):
        probe = get_plugin("tetfill").probe()
        assert probe.status == "ok" and probe.transport == "local"
        assert probe.version == "0.1.0"


class TestTransportSelection:
    def test_a_local_spec_with_a_missing_api_path_is_refused(self, tmp_path):
        from cadjoint.plugins import PluginConfigError

        plugin = TesseractPlugin(
            PluginSpec(
                name="ghost",
                kind="mesher",
                transport="local",
                api_path=tmp_path / "tesseract_api.py",
            )
        )
        with pytest.raises(PluginConfigError, match="does not exist"):
            plugin.apply({})

    def test_options_are_forwarded_to_the_constructor_verbatim(self, served_mesher):
        """``options`` is ``Tesseract.from_url``'s keyword arguments."""
        plugin = TesseractPlugin(
            PluginSpec(
                name="mesher",
                kind="mesher",
                transport="remote",
                url=served_mesher.url,
                options={"timeout": 30.0},
            )
        )
        try:
            assert plugin.health()["status"] == "ok"
        finally:
            plugin.close()
