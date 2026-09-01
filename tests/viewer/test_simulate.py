"""Tests for the playground's FEM endpoints (/api/simulate, /api/mesh_inspect).

The ad-hoc request-body simulation kinds ("probe"/"thermal"/"elastic") are
retired: /api/simulate only runs studies the scene program declares, and
/api/mesh_inspect replaces the probe — it builds a declared SimMesh (or a
study's implicit mesh) and reports it with a quality heatmap.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cadjoint.viewer import _compile_worker
from cadjoint.viewer.playground import create_server, mesh_inspect_source, simulate_source

BOX_SOURCE = """
from cadjoint.geometry import Vector
from cadjoint.sdf.primitives import Box

scene = Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))
"""


# ── Request validation (no worker process involved) ─────────────────────────


class TestSimulateValidation:
    def test_only_study_kind_is_accepted(self):
        for kind in ("modal", "probe", "thermal", "elastic"):
            result = simulate_source({"source": BOX_SOURCE, "kind": kind, "name": "bar"})
            assert result["ok"] is False, kind
            assert "study" in result["error"]

    def test_study_kind_requires_a_name(self):
        for name in (None, "", "   ", 7):
            result = simulate_source({"source": BOX_SOURCE, "kind": "study", "name": name})
            assert result["ok"] is False, name
            assert "name" in result["error"]

    def test_cached_must_be_a_boolean(self):
        for cached in ("yes", 1, [True]):
            result = simulate_source(
                {"source": BOX_SOURCE, "kind": "study", "name": "bar", "cached": cached}
            )
            assert result["ok"] is False, cached
            assert "cached" in result["error"]

    def test_mesh_inspect_name_must_be_a_non_empty_string(self):
        for name in ("", "   ", 7):
            result = mesh_inspect_source({"source": BOX_SOURCE, "name": name})
            assert result["ok"] is False, name
            assert "name" in result["error"]


# ── Worker-level behavior (in-process, no subprocess) ───────────────────────


def _box_scene():
    from cadjoint.geometry import Vector
    from cadjoint.sdf.primitives import Box

    return Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))


def _bar_study():
    from cadjoint.fem import Dirichlet, Nodes, ThermalStudy

    return ThermalStudy(
        name="bar",
        resolution=10,
        conductivity=1.0,
        bcs=[
            Dirichlet(Nodes.side("-x"), 0.0),
            Dirichlet(Nodes.side("+x"), 100.0),
        ],
    )


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
        headers["X-Cadjoint-Token"] = token
    return Request(base + path, data=json.dumps(payload).encode(), headers=headers, method="POST")


class TestHttp:
    @pytest.mark.parametrize("path", ["/api/simulate", "/api/mesh_inspect"])
    def test_fem_endpoints_require_the_session_token(self, path):
        with _running_server() as base:
            with pytest.raises(HTTPError) as error:
                urlopen(_post(base, path, {"source": BOX_SOURCE, "kind": "study", "name": "b"}))
            assert error.value.code == 403

    def test_fem_unavailable_maps_to_501(self, monkeypatch):
        from cadjoint.viewer import playground

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
                        {"source": BOX_SOURCE, "kind": "study", "name": "bar"},
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
                        {"source": BOX_SOURCE, "kind": "thermal"},
                        token=token,
                    )
                )
            assert error.value.code == 422


# ── Declared studies run by name (code-parity path) ─────────────────────────


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
        assert result["cached"] is False
        low, high = result["mesh"]["range"]
        assert low == pytest.approx(0.0, abs=1e-3)
        assert high == pytest.approx(100.0, abs=1e-3)
        # The declaration itself rides along, with per-BC serializability.
        assert result["study"]["name"] == "bar"
        assert [bc["serializable"] for bc in result["study"]["bcs"]] == [True, True]
        # The result summary is the SimulationResult's describe() payload.
        summary = result["result"]
        assert summary["name"] == "bar"
        assert summary["kind"] == "thermal"
        assert summary["field"] == "temperature"
        assert summary["range"] == pytest.approx([low, high], abs=1e-6)
        assert set(summary["fields"]) == {"temperature"}
        # The built mesh's inspection report rides along too.
        info = result["mesh_info"]
        assert info["nodes"] == summary["nodes"]
        assert info["elements"] == summary["elements"]
        assert set(info["quality"]) == {"scaled_jacobian", "aspect_ratio"}

    def test_elastic_study_reports_von_mises(self):
        pytest.importorskip("jax_fem", reason="study solve needs the fem extra")
        from cadjoint.fem import ElasticStudy, Fixed, Nodes, Traction

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
        assert set(result["result"]["fields"]) == {"displacement", "von_mises"}

    def test_cached_serves_the_last_result_without_resolving(self):
        pytest.importorskip("jax_fem", reason="study solve needs the fem extra")
        study = _bar_study()
        scene = _box_scene()
        first = _compile_worker._simulate_study(scene, [study], {"kind": "study", "name": "bar"})
        assert first["cached"] is False
        assert study.last_result is not None

        def explode(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("cached request must not re-solve")

        study.solve = explode
        second = _compile_worker._simulate_study(
            scene, [study], {"kind": "study", "name": "bar", "cached": True}
        )
        assert second["cached"] is True
        assert second["result"] == first["result"]
        assert second["mesh"]["range"] == first["mesh"]["range"]

    def test_cached_falls_back_to_solving_when_nothing_is_stored(self):
        pytest.importorskip("jax_fem", reason="study solve needs the fem extra")
        result = _compile_worker._simulate_study(
            _box_scene(), [_bar_study()], {"kind": "study", "name": "bar", "cached": True}
        )
        assert result["ok"] is True
        assert result["cached"] is False

    def test_worker_simulate_source_dispatches_studies(self):
        pytest.importorskip("jax_fem", reason="study solve needs the fem extra")
        source = "\n".join(
            [
                "from cadjoint.fem import Dirichlet, Nodes, ThermalStudy",
                "from cadjoint.geometry import Vector",
                "from cadjoint.sdf.primitives import Box",
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

    def test_studies_resolve_declared_meshes_by_name(self):
        pytest.importorskip("jax_fem", reason="study solve needs the fem extra")
        source = "\n".join(
            [
                "from cadjoint.fem import Dirichlet, Nodes, SimMesh, ThermalStudy",
                "from cadjoint.geometry import Vector",
                "from cadjoint.sdf.primitives import Box",
                "",
                'scene = Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))',
                "grid = SimMesh(name='grid', resolution=8, bounds=(-1.0, -0.75, -0.75),",
                "               size=(2.0, 1.5, 1.5))",
                "heat = ThermalStudy(name='bar', conductivity=1.0, mesh='grid',",
                "                    bcs=[Dirichlet(Nodes.side('-x'), 0.0),",
                "                         Dirichlet(Nodes.side('+x'), 1.0)])",
            ]
        )
        result = _compile_worker._simulate_source(
            {"source": source, "kind": "study", "name": "bar"}
        )
        assert result["ok"] is True
        assert result["study"]["mesh"] == "grid"
        assert result["result"]["mesh"] == "grid"
        assert result["mesh_info"]["name"] == "grid"


# ── Mesh inspection (probe successor) ───────────────────────────────────────

MESH_SOURCE = "\n".join(
    [
        "from cadjoint.fem import SimMesh",
        "from cadjoint.geometry import Vector",
        "from cadjoint.sdf.primitives import Box",
        "",
        'scene = Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))',
        "grid = SimMesh(name='grid', resolution=8, bounds=(-1.0, -0.75, -0.75),",
        "               size=(2.0, 1.5, 1.5))",
        "",
    ]
)


class TestMeshInspect:
    def test_named_mesh_reports_info_and_quality_surface(self):
        result = _compile_worker._mesh_inspect_source({"source": MESH_SOURCE, "name": "grid"})
        assert result["ok"] is True
        assert result["name"] == "grid"
        assert result["field"] == "scaled_jacobian"
        info = result["info"]
        assert info["nodes"] > 0 and info["elements"] > 0
        assert set(info["quality"]) == {"scaled_jacobian", "aspect_ratio"}
        assert info["grid"]["cells"] == [8, 8, 8]
        mesh = result["mesh"]
        assert set(mesh) >= {"positions", "scalars", "indices", "groups", "range", "vertex_count"}
        assert len(mesh["scalars"]) == mesh["vertex_count"]
        # The heatmap is the per-vertex min scaled jacobian: in (0, 1].
        assert result["quality_scalars"] == mesh["scalars"]
        assert all(0.0 < value <= 1.0 + 1e-9 for value in result["quality_scalars"])
        low, high = mesh["range"]
        assert 0.0 < low <= high <= 1.0 + 1e-9

    def test_single_declared_mesh_needs_no_name(self):
        result = _compile_worker._mesh_inspect_source({"source": MESH_SOURCE})
        assert result["ok"] is True
        assert result["name"] == "grid"

    def test_a_study_name_builds_its_implicit_mesh(self):
        source = "\n".join(
            [
                "from cadjoint.fem import Dirichlet, Nodes, ThermalStudy",
                "from cadjoint.geometry import Vector",
                "from cadjoint.sdf.primitives import Box",
                "",
                'scene = Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))',
                "heat = ThermalStudy(name='bar', resolution=8, conductivity=1.0,",
                "                    bcs=[Dirichlet(Nodes.side('-x'), 0.0)])",
            ]
        )
        result = _compile_worker._mesh_inspect_source({"source": source, "name": "bar"})
        assert result["ok"] is True
        assert result["name"] == "bar::mesh"
        assert result["info"]["elements"] > 0

    def test_unknown_name_lists_meshes_and_studies(self):
        # The worker's main() turns these into {"ok": False} responses.
        with pytest.raises(ValueError, match="'nope'"):
            _compile_worker._inspect_mesh(_box_scene(), [], [], {"name": "nope"})
        with pytest.raises(ValueError, match="'grid'"):
            _compile_worker._mesh_inspect_source({"source": MESH_SOURCE, "name": "nope"})

    def test_ambiguity_without_a_name_is_an_error(self):
        with pytest.raises(ValueError, match="name"):
            _compile_worker._mesh_inspect_source({"source": BOX_SOURCE})

    def test_endpoint_round_trips_over_http(self):
        with _running_server() as base:
            with urlopen(f"{base}/api/session") as response:
                token = json.loads(response.read())["token"]
            with urlopen(
                _post(base, "/api/mesh_inspect", {"source": MESH_SOURCE, "name": "grid"}, token)
            ) as response:
                result = json.loads(response.read())
        assert result["ok"] is True
        assert result["info"]["name"] == "grid"
        assert len(result["quality_scalars"]) == result["mesh"]["vertex_count"]
