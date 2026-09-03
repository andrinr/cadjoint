"""Discovery, spec parsing, and resolution by kind.

Nothing here starts a solve: this is the layer that decides *which*
component answers a kind and *where* it runs, and it has to be checkable
without paying for a mesh.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cadjoint.plugins import (
    BUILTIN_DEFAULTS,
    BUILTIN_PACKAGES,
    ENTRY_POINT_GROUP,
    PluginConfigError,
    PluginRegistry,
    PluginSpec,
    build_registry,
    plugin_for_kind,
    registry,
)
from cadjoint.plugins.registry import builtin_specs, parse_config

_REPO = Path(__file__).resolve().parents[2]


class TestBuiltinDiscovery:
    def test_every_shipped_package_is_registered_as_a_local_plugin(self):
        """Every Tesseract package in this repository, all in-process."""
        specs = builtin_specs()
        for name in BUILTIN_PACKAGES:
            spec = specs[name]
            assert spec.transport == "local"
            assert spec.api_path is not None and spec.api_path.is_file()
            assert spec.kind == BUILTIN_PACKAGES[name][0]

    def test_the_shipped_packages_are_all_there_is(self):
        """Nothing in this checkout provides the private tier's kinds.

        The five in-process kinds are discovered, never bundled: an
        installed ``diff-brep`` registers them through the
        ``cadjoint.plugins`` entry-point group, and with it absent the
        registry simply has no plugin of those kinds
        (``research/two-tier.md`` §5 step 4).
        """
        import cadjoint.tier as tier

        specs = builtin_specs()
        assert set(specs) == set(BUILTIN_PACKAGES)
        assert {spec.kind for spec in specs.values()}.isdisjoint(tier.KINDS)
        assert set(BUILTIN_DEFAULTS).isdisjoint(tier.KINDS)

    def test_versions_come_from_the_packages_not_from_python(self):
        """``tesseract_config.yaml`` is the single source of a version."""
        specs = builtin_specs()
        packaged = [specs[name] for name in BUILTIN_PACKAGES]
        assert {spec.version for spec in packaged} == {"0.1.0"}

    def test_two_solvers_share_the_elastic_kind_and_a_default_picks_one(self):
        """jax-fem and CalculiX both fill ``elastic_solver``; jax-fem wins."""
        assert BUILTIN_PACKAGES["elastic_jaxfem"][0] == "elastic_solver"
        assert BUILTIN_PACKAGES["elastic_calculix"][0] == "elastic_solver"
        built = build_registry(use_entry_points=False)
        assert built.kinds()["elastic_solver"] == ["elastic_calculix", "elastic_jaxfem"]
        assert built.default_for("elastic_solver") == "elastic_jaxfem"
        assert built.default_for("thermal_solver") == BUILTIN_DEFAULTS["thermal_solver"]

    def test_the_process_registry_answers_every_kind_the_pipeline_asks_for(self):
        """The kinds ``chain.py`` and ``backends.py`` resolve all exist."""
        current = registry()
        for kind in ("mesher", "tetfill", "thermal_solver", "elastic_solver"):
            assert current.default_for(kind) in current.names()

    def test_resolution_by_kind_returns_the_same_warm_instance(self):
        """Instances are cached, the way the old per-process loader was."""
        assert plugin_for_kind("mesher") is plugin_for_kind("mesher")

    def test_entry_point_group_is_declared_for_third_parties(self):
        """A third party registers under this group; cadjoint reads it."""
        assert ENTRY_POINT_GROUP == "cadjoint.plugins"
        text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert f'[project.entry-points."{ENTRY_POINT_GROUP}"]' in text


class TestSpecParsing:
    def test_a_remote_table_parses_into_a_remote_spec(self):
        """``options`` is whatever ``Tesseract.from_url`` accepts, verbatim."""
        spec = PluginSpec.from_mapping(
            "thermal_jaxfem",
            {
                "kind": "thermal_solver",
                "transport": "remote",
                "url": "http://thermal.cadjoint.svc.cluster.local:8000",
                "options": {"timeout": 30.0},
                "schema_hash": "sha256:deadbeef",
            },
        )
        assert spec.transport == "remote"
        assert spec.url.endswith(":8000")
        assert spec.options == {"timeout": 30.0}
        assert spec.schema_hash == "sha256:deadbeef"

    def test_environment_variables_expand_in_the_target(self, monkeypatch):
        monkeypatch.setenv("CADJOINT_TEST_HOST", "solver.internal")
        spec = PluginSpec.from_mapping(
            "remote_solver",
            {
                "kind": "elastic_solver",
                "transport": "remote",
                "url": "http://${CADJOINT_TEST_HOST}:8000",
            },
        )
        assert spec.url == "http://solver.internal:8000"

    def test_an_unset_variable_is_an_error_not_an_empty_value(self, monkeypatch):
        monkeypatch.delenv("CADJOINT_TEST_HOST", raising=False)
        with pytest.raises(PluginConfigError, match="CADJOINT_TEST_HOST"):
            PluginSpec.from_mapping(
                "remote_solver",
                {
                    "kind": "elastic_solver",
                    "transport": "remote",
                    "url": "http://${CADJOINT_TEST_HOST}:8000",
                },
            )

    @pytest.mark.parametrize(
        ("table", "message"),
        [
            ({"kind": "mesher", "transport": "carrier-pigeon"}, "transport must be one of"),
            ({"kind": "mesher", "transport": "remote"}, "needs 'url'"),
            ({"kind": "mesher", "transport": "container"}, "needs 'image'"),
            ({"kind": "mesher"}, "needs 'api_path'"),
            ({"transport": "local", "api_path": "x.py"}, "'kind' is required"),
            ({"kind": "mesher", "api_path": "x.py", "colour": "blue"}, "unknown key"),
        ],
    )
    def test_malformed_tables_are_refused_by_name(self, table, message):
        with pytest.raises(PluginConfigError, match=message):
            PluginSpec.from_mapping("broken", table)

    def test_a_spec_round_trips_through_its_table_form(self):
        spec = PluginSpec.from_mapping(
            "remote_solver",
            {"kind": "elastic_solver", "transport": "remote", "url": "http://h:8000"},
        )
        assert PluginSpec.from_mapping("remote_solver", spec.to_mapping()) == spec


class TestConfigFile:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "plugins.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_config_table_replaces_a_builtin_spec_in_place(self, tmp_path):
        """The headline claim: in-process to cluster is configuration only."""
        path = self._write(
            tmp_path,
            """
            [plugins.thermal_jaxfem]
            kind = "thermal_solver"
            transport = "remote"
            url = "http://thermal.cadjoint.svc.cluster.local:8000"
            """,
        )
        built = build_registry(config=path, use_entry_points=False)
        spec = built.spec("thermal_jaxfem")
        assert spec.transport == "remote"
        assert built.default_for("thermal_solver") == "thermal_jaxfem"
        # Everything else keeps running in-process.
        assert built.spec("mesher").transport == "local"

    def test_defaults_repoint_a_kind_at_another_plugin(self, tmp_path):
        path = self._write(
            tmp_path,
            """
            [defaults]
            elastic_solver = "elastic_calculix"
            """,
        )
        built = build_registry(config=path, use_entry_points=False)
        assert built.default_for("elastic_solver") == "elastic_calculix"

    def test_a_new_plugin_becomes_the_default_for_a_kind_nothing_else_fills(self, tmp_path):
        path = self._write(
            tmp_path,
            """
            [plugins.acoustic_remote]
            kind = "acoustic_solver"
            transport = "remote"
            url = "http://acoustics:8000"
            """,
        )
        built = build_registry(config=path, use_entry_points=False)
        assert built.default_for("acoustic_solver") == "acoustic_remote"

    def test_an_unknown_top_level_table_is_refused(self):
        with pytest.raises(PluginConfigError, match="unknown top-level table"):
            parse_config({"plugin": {}}, source="test")

    def test_a_missing_env_override_is_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CADJOINT_PLUGINS", str(tmp_path / "nope.toml"))
        from cadjoint.plugins import config_path

        with pytest.raises(PluginConfigError, match="not a readable"):
            config_path()

    def test_the_env_override_may_name_a_directory(self, monkeypatch, tmp_path):
        self._write(tmp_path, "[plugins]\n")
        monkeypatch.setenv("CADJOINT_PLUGINS", str(tmp_path))
        from cadjoint.plugins import config_path

        assert config_path() == tmp_path / "plugins.toml"


class TestRegistryMutation:
    def test_registering_a_spec_repoints_its_kind(self):
        built = PluginRegistry()
        spec = PluginSpec(
            name="stub",
            kind="thermal_solver",
            transport="remote",
            url="http://stub:8000",
        )
        built.register(spec)
        assert built.names() == ["stub"]
        assert built.default_for("thermal_solver") == "stub"
        assert built.plugin("stub").spec is spec

    def test_an_unknown_name_names_what_is_registered(self):
        built = build_registry(use_entry_points=False)
        with pytest.raises(KeyError, match="Unknown plugin"):
            built.spec("no_such_plugin")

    def test_an_unfilled_kind_says_so(self):
        with pytest.raises(KeyError, match="No plugin registered for kind"):
            PluginRegistry().default_for("thermal_solver")

    def test_an_ambiguous_kind_asks_for_a_default(self):
        built = PluginRegistry()
        for name in ("a", "b"):
            built.register(
                PluginSpec(name=name, kind="elastic_solver", transport="remote", url="http://x"),
                default=False,
            )
        built._defaults.clear()
        with pytest.raises(KeyError, match="no default is set"):
            built.default_for("elastic_solver")
