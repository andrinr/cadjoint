"""Tests for the playground's FEM simulation mode (/api/simulate)."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from jaxcad.viewer import _compile_worker
from jaxcad.viewer.playground import create_server, simulate_source

BOX_SOURCE = """
from jaxcad.geometry import Vector
from jaxcad.sdf.primitives import Box

scene = Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))
"""


# ── Request validation (no worker process involved) ─────────────────────────


class TestSimulateValidation:
    def test_unknown_kind_is_rejected(self):
        result = simulate_source({"source": BOX_SOURCE, "kind": "modal"})
        assert result["ok"] is False
        assert "kind" in result["error"]

    def test_resolution_bounds(self):
        for resolution in (0, 3, 65, 2.5, True, "16"):
            result = simulate_source(
                {"source": BOX_SOURCE, "kind": "probe", "resolution": resolution}
            )
            assert result["ok"] is False
            assert "resolution" in result["error"]

    def test_bc_group_shapes(self):
        for group in ("x+", "++", 3, {"axis": "w", "side": "+"}, {"axis": "x"}):
            result = simulate_source(
                {
                    "source": BOX_SOURCE,
                    "kind": "thermal",
                    "bcs": [{"group": group, "type": "dirichlet", "value": 1.0}],
                }
            )
            assert result["ok"] is False, group

    def test_bc_type_and_value_shapes(self):
        bad = [
            {"group": "+x", "type": "neumann", "value": 1.0},
            {"group": "+x", "type": "dirichlet", "value": "hot"},
            {"group": "+x", "type": "traction", "value": 1.0},
            {"group": "+x", "type": "traction", "value": [1.0, 2.0]},
            "not an object",
        ]
        for bc in bad:
            result = simulate_source({"source": BOX_SOURCE, "kind": "elastic", "bcs": [bc]})
            assert result["ok"] is False, bc

    def test_material_keys_and_values(self):
        for material in ({"density": 1.0}, {"youngs": "steel"}, [1.0]):
            result = simulate_source(
                {"source": BOX_SOURCE, "kind": "elastic", "material": material}
            )
            assert result["ok"] is False, material

    def test_source_must_be_a_string(self):
        result = simulate_source({"kind": "probe"})
        assert result["ok"] is False

    def test_study_kind_requires_a_name(self):
        for name in (None, "", "   ", 7):
            result = simulate_source({"source": BOX_SOURCE, "kind": "study", "name": name})
            assert result["ok"] is False, name
            assert "name" in result["error"]


# ── Worker-level behavior (in-process, no subprocess) ───────────────────────


def _box_scene():
    from jaxcad.geometry import Vector
    from jaxcad.sdf.primitives import Box

    return Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))


class TestProbe:
    def test_probe_returns_catalog_and_surface(self):
        result = _compile_worker._simulate_scene(_box_scene(), {"kind": "probe", "resolution": 12})
        assert result["ok"] is True
        assert result["field"] is None
        mesh = result["mesh"]
        groups = {group["id"] for group in mesh["groups"]}
        assert groups == {"+x", "-x", "+y", "-y", "+z", "-z"}
        for group in mesh["groups"]:
            assert set(group) >= {"id", "axis", "side", "center", "area", "faces", "start", "count"}
        indices = np.asarray(mesh["indices"])
        assert indices.max() < mesh["vertex_count"]
        assert len(mesh["scalars"]) == mesh["vertex_count"]

    def test_unknown_group_names_the_available_ones(self):
        with pytest.raises(ValueError, match=r"\+x"):
            _compile_worker._simulate_scene(
                _box_scene(),
                {
                    "kind": "thermal",
                    "resolution": 8,
                    "bcs": [{"group": "-q", "type": "dirichlet", "value": 0.0}],
                },
            )

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            _compile_worker._simulate_scene(_box_scene(), {"kind": "modal", "resolution": 8})


class TestMissingExtra:
    def test_fem_unavailable_is_a_typed_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_jax_fem(name, *args, **kwargs):
            if name == "jax_fem" or name.startswith("jax_fem."):
                raise ImportError("No module named 'jax_fem'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_jax_fem)
        result = _compile_worker._simulate_scene(
            _box_scene(),
            {
                "kind": "thermal",
                "resolution": 8,
                "bcs": [{"group": "+x", "type": "dirichlet", "value": 0.0}],
            },
        )
        assert result["ok"] is False
        assert result["error_kind"] == "fem_unavailable"
        assert "fem" in result["error"]

    def test_probe_works_without_the_extra(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_jax_fem(name, *args, **kwargs):
            if name == "jax_fem" or name.startswith("jax_fem."):
                raise ImportError("No module named 'jax_fem'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_jax_fem)
        result = _compile_worker._simulate_scene(_box_scene(), {"kind": "probe", "resolution": 8})
        assert result["ok"] is True


# ── HTTP layer ──────────────────────────────────────────────────────────────


@contextmanager
def _running_server():
    server = create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post(base: str, path: str, payload: dict, token: str | None = None) -> Request:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Jaxcad-Token"] = token
    return Request(base + path, data=json.dumps(payload).encode(), headers=headers, method="POST")


class TestHttp:
    def test_simulate_endpoint_requires_the_session_token(self):
        with _running_server() as base:
            with pytest.raises(HTTPError) as error:
                urlopen(_post(base, "/api/simulate", {"source": BOX_SOURCE, "kind": "probe"}))
            assert error.value.code == 403

    def test_fem_unavailable_maps_to_501(self, monkeypatch):
        from jaxcad.viewer import playground

        def unavailable(request, timeout=0):
            return {"ok": False, "error_kind": "fem_unavailable", "error": "install the extra"}

        # Patch the handler table's target: simulate_source is looked up on the
        # module at request time.
        monkeypatch.setattr(playground, "simulate_source", unavailable)
        with _running_server() as base:
            with urlopen(f"{base}/api/session") as response:
                token = json.loads(response.read())["token"]
            with pytest.raises(HTTPError) as error:
                urlopen(
                    _post(
                        base,
                        "/api/simulate",
                        {"source": BOX_SOURCE, "kind": "thermal"},
                        token=token,
                    )
                )
            assert error.value.code == 501
            body = json.loads(error.value.read())
            assert body["error_kind"] == "fem_unavailable"

    def test_validation_error_maps_to_422(self):
        with _running_server() as base:
            with urlopen(f"{base}/api/session") as response:
                token = json.loads(response.read())["token"]
            with pytest.raises(HTTPError) as error:
                urlopen(
                    _post(
                        base,
                        "/api/simulate",
                        {"source": BOX_SOURCE, "kind": "modal"},
                        token=token,
                    )
                )
            assert error.value.code == 422


# ── Real solves (need jax-fem; run in-process to skip the subprocess cost) ──


class TestSolves:
    def test_thermal_solve_produces_a_gradient(self):
        pytest.importorskip("jax_fem", reason="thermal solve needs the fem extra")
        result = _compile_worker._simulate_scene(
            _box_scene(),
            {
                "kind": "thermal",
                "resolution": 10,
                "bcs": [
                    {"group": "-x", "type": "dirichlet", "value": 0.0},
                    {"group": "+x", "type": "dirichlet", "value": 100.0},
                ],
                "material": {"conductivity": 1.0},
            },
        )
        assert result["ok"] is True
        assert result["field"] == "temperature"
        mesh = result["mesh"]
        low, high = mesh["range"]
        assert low == pytest.approx(0.0, abs=1e-3)
        assert high == pytest.approx(100.0, abs=1e-3)
        # Boundary temperatures interpolate between the two prescribed faces.
        assert all(-1e-3 <= value <= 100.0 + 1e-3 for value in mesh["scalars"])

    def test_elastic_solve_reports_von_mises(self):
        pytest.importorskip("jax_fem", reason="elastic solve needs the fem extra")
        result = _compile_worker._simulate_scene(
            _box_scene(),
            {
                "kind": "elastic",
                "resolution": 8,
                "bcs": [
                    {"group": "-x", "type": "dirichlet", "value": 0.0},
                    {"group": "+x", "type": "traction", "value": [0.0, 0.0, -1.0]},
                ],
                "material": {"youngs": 200.0, "poisson": 0.3},
            },
        )
        assert result["ok"] is True
        assert result["field"] == "von_mises"
        low, high = result["mesh"]["range"]
        assert high > low >= 0.0

    def test_thermal_without_dirichlet_is_rejected(self):
        pytest.importorskip("jax_fem", reason="solver path needs the fem extra")
        with pytest.raises(ValueError, match="fixed-temperature"):
            _compile_worker._simulate_scene(
                _box_scene(), {"kind": "thermal", "resolution": 8, "bcs": []}
            )


# ── Declared studies run by name (code-parity path) ─────────────────────────


def _bar_study():
    from jaxcad.fem import Dirichlet, Nodes, ThermalStudy

    return ThermalStudy(
        name="bar",
        resolution=10,
        conductivity=1.0,
        bcs=[
            Dirichlet(Nodes.side("-x"), 0.0),
            Dirichlet(Nodes.side("+x"), 100.0),
        ],
    )


class TestStudySimulate:
    def test_unknown_study_name_lists_the_declared_ones(self):
        with pytest.raises(ValueError, match="'bar'"):
            _compile_worker._simulate_study(
                _box_scene(), [_bar_study()], {"kind": "study", "name": "nope"}
            )

    def test_duplicate_study_names_are_ambiguous(self):
        with pytest.raises(ValueError, match="more than one"):
            _compile_worker._simulate_study(
                _box_scene(), [_bar_study(), _bar_study()], {"kind": "study", "name": "bar"}
            )

    def test_fem_unavailable_is_a_typed_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_jax_fem(name, *args, **kwargs):
            if name == "jax_fem" or name.startswith("jax_fem."):
                raise ImportError("No module named 'jax_fem'")
            return real_import(name, *args, **kwargs)

        study = _bar_study()
        monkeypatch.setattr(builtins, "__import__", no_jax_fem)
        result = _compile_worker._simulate_study(
            _box_scene(), [study], {"kind": "study", "name": "bar"}
        )
        assert result["ok"] is False
        assert result["error_kind"] == "fem_unavailable"

    def test_thermal_study_solves_with_its_declared_settings(self):
        pytest.importorskip("jax_fem", reason="study solve needs the fem extra")
        result = _compile_worker._simulate_study(
            _box_scene(), [_bar_study()], {"kind": "study", "name": "bar"}
        )
        assert result["ok"] is True
        assert result["kind"] == "study"
        assert result["field"] == "temperature"
        low, high = result["mesh"]["range"]
        assert low == pytest.approx(0.0, abs=1e-3)
        assert high == pytest.approx(100.0, abs=1e-3)
        # The declaration itself rides along, with per-BC serializability.
        assert result["study"]["name"] == "bar"
        assert [bc["serializable"] for bc in result["study"]["bcs"]] == [True, True]

    def test_elastic_study_reports_von_mises(self):
        pytest.importorskip("jax_fem", reason="study solve needs the fem extra")
        from jaxcad.fem import ElasticStudy, Fixed, Nodes, Traction

        study = ElasticStudy(
            name="cantilever",
            resolution=8,
            youngs=200.0,
            poisson=0.3,
            bcs=[
                Fixed(Nodes.side("-x")),
                Traction(Nodes.side("+x"), (0.0, 0.0, -1.0)),
            ],
        )
        result = _compile_worker._simulate_study(
            _box_scene(), [study], {"kind": "study", "name": "cantilever"}
        )
        assert result["ok"] is True
        assert result["field"] == "von_mises"
        low, high = result["mesh"]["range"]
        assert high > low >= 0.0

    def test_worker_simulate_source_dispatches_studies(self):
        pytest.importorskip("jax_fem", reason="study solve needs the fem extra")
        source = "\n".join(
            [
                "from jaxcad.fem import Dirichlet, Nodes, ThermalStudy",
                "from jaxcad.geometry import Vector",
                "from jaxcad.sdf.primitives import Box",
                "",
                'scene = Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))',
                "heat = ThermalStudy(name='bar', resolution=8, conductivity=1.0,",
                "                    bcs=[Dirichlet(Nodes.side('-x'), 0.0),",
                "                         Dirichlet(Nodes.side('+x'), 1.0)])",
            ]
        )
        result = _compile_worker._simulate_source(
            {"source": source, "kind": "study", "name": "bar"}
        )
        assert result["ok"] is True
        assert result["field"] == "temperature"
        assert "output" in result
