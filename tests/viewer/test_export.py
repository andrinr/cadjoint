"""Tests for ``POST /api/export``: one object of the program, as a file.

Three layers, cheapest first.  The request model refuses what it should
before any worker starts, and says why in the dialog's words.  The worker
half writes each format from a real extraction — STL both ways, OBJ merged
and unmerged, STEP through the derived B-rep and through the faceted
fallback — and names the alternatives when the object does not exist.  The
HTTP layer sends the bytes as an attachment under the right content type,
registers the run as an ``export`` job, and keeps the file out of the
result store.
"""

from __future__ import annotations

import json
import struct
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cadjoint.enums import ExportFormat
from cadjoint.viewer._export import (
    EXPORT_CONTENT_TYPES,
    EXPORT_EXTENSIONS,
    export_filename,
    export_scene,
    export_source,
    validate_export_request,
)
from cadjoint.viewer._jobs import JOB_KINDS, RESULT_KINDS
from cadjoint.viewer._limits import (
    EXPORT_DEFAULT_RESOLUTION,
    EXPORT_MAX_RESOLUTION,
    EXPORT_MIN_RESOLUTION,
    MAX_SOURCE_BYTES,
)
from cadjoint.viewer.playground import REGISTRY, create_server

BOX_SOURCE = """
from cadjoint.geometry import Vector
from cadjoint.sdf.primitives import Box

scene = Box(Vector([1.0, 0.6, 0.4]))
"""

# A part bound to its own name, then placed: `part` and `scene` are both
# exportable, `offset` is a Vector and `count` is an int.
PLACED_SOURCE = """
from cadjoint.geometry import Vector
from cadjoint.sdf.primitives import Box
from cadjoint.sdf.transforms import Translate

offset = Vector([0.5, 0.0, 0.0])
count = 2
part = Box(Vector([1.0, 0.6, 0.4]))
scene = Translate(part, offset)
"""

STUDY_SOURCE = """
from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
from cadjoint.geometry import Vector
from cadjoint.sdf.primitives import Box

scene = Box(Vector([0.8, 0.5, 0.5]))
heat = ThermalStudy(
    name="bar",
    resolution=10,
    conductivity=1.0,
    bcs=[Dirichlet(Nodes.side("-x"), 0.0), Dirichlet(Nodes.side("+x"), 100.0)],
)
"""


def _export(tmp_path: Path, source: str, **fields) -> tuple[dict, Path]:
    """Run the worker half in-process and return ``(result, path)``."""
    fmt = fields.get("format", "stl")
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"out.{EXPORT_EXTENSIONS[ExportFormat(fmt)]}"
    request = {"source": source, "mode": "export", "path": str(path), "format": fmt, **fields}
    request.setdefault("resolution", 12)
    return export_scene(request), path


# ── Request validation (no worker process involved) ─────────────────────────


class TestValidation:
    def test_the_enum_and_the_tables_agree(self):
        assert tuple(ExportFormat) == ("obj", "stl", "step", "vtk")
        assert set(EXPORT_EXTENSIONS) == set(ExportFormat) == set(EXPORT_CONTENT_TYPES)
        assert "export" in JOB_KINDS
        # The file is the response; nothing worth replaying is kept.
        assert "export" not in RESULT_KINDS

    def test_the_defaults_are_the_scene_at_the_overlay_resolution(self):
        error, parsed = validate_export_request({"source": BOX_SOURCE, "format": "stl"})
        assert error is None
        assert parsed.name == "scene"
        assert parsed.resolution == EXPORT_DEFAULT_RESOLUTION
        assert parsed.binary is parsed.analytic is parsed.merge_planar is True

    def test_an_unknown_format_names_the_accepted_ones(self):
        error, _ = validate_export_request({"source": BOX_SOURCE, "format": "gltf"})
        assert error["ok"] is False
        assert error["error"] == "Export `format` must be one of: obj, stl, step, vtk."

    @pytest.mark.parametrize(
        "resolution",
        [EXPORT_MIN_RESOLUTION - 1, EXPORT_MAX_RESOLUTION + 1, 0, -8, 12.5, "64", True],
    )
    def test_the_resolution_stays_inside_the_bracket_and_is_an_integer(self, resolution):
        error, _ = validate_export_request(
            {"source": BOX_SOURCE, "format": "stl", "resolution": resolution}
        )
        assert error["ok"] is False
        assert "resolution" in error["error"]

    def test_the_bracket_edges_are_accepted(self):
        for resolution in (EXPORT_MIN_RESOLUTION, EXPORT_MAX_RESOLUTION):
            error, parsed = validate_export_request(
                {"source": BOX_SOURCE, "format": "stl", "resolution": resolution}
            )
            assert error is None and parsed.resolution == resolution

    def test_source_must_be_a_string_and_within_the_limit(self):
        error, _ = validate_export_request({"format": "stl"})
        assert "source" in error["error"]
        error, _ = validate_export_request({"source": 7, "format": "stl"})
        assert "source" in error["error"]
        error, _ = validate_export_request(
            {"source": "#" * (MAX_SOURCE_BYTES + 1), "format": "stl"}
        )
        assert f"{MAX_SOURCE_BYTES:,}" in error["error"]

    def test_unknown_fields_and_empty_names_are_refused(self):
        error, _ = validate_export_request({"source": BOX_SOURCE, "format": "stl", "level": 0.5})
        assert "level" in error["error"]
        error, _ = validate_export_request({"source": BOX_SOURCE, "format": "stl", "name": ""})
        assert "name" in error["error"]

    def test_export_source_refuses_before_starting_a_worker(self):
        result = export_source({"source": BOX_SOURCE, "format": "dxf"})
        assert result["ok"] is False and "format" in result["error"]

    def test_filenames_carry_the_object_under_the_extension(self):
        assert export_filename("scene", ExportFormat.STL) == "scene.stl"
        assert export_filename("sink-conduction", ExportFormat.VTK) == "sink-conduction.vtk"
        assert export_filename("a b/c", ExportFormat.OBJ) == "a_b_c.obj"
        assert export_filename("..", ExportFormat.STEP) == "export.step"


# ── The worker half (in-process, no subprocess) ─────────────────────────────


class TestWorker:
    def test_binary_stl_is_a_closed_triangle_soup(self, tmp_path):
        result, path = _export(tmp_path, BOX_SOURCE, format="stl")
        assert result["ok"] is True, result
        data = path.read_bytes()
        (count,) = struct.unpack("<I", data[80:84])
        assert count == result["report"]["triangles"] > 0
        assert len(data) == 84 + 50 * count
        # An isotropic lattice: the longest axis gets the requested count.
        assert max(result["report"]["resolution"]) == 12
        assert min(result["report"]["resolution"]) >= 1

    def test_ascii_stl_on_request(self, tmp_path):
        result, path = _export(tmp_path, BOX_SOURCE, format="stl", binary=False)
        assert result["ok"] is True, result
        text = path.read_text()
        assert text.startswith("solid ") and text.rstrip().splitlines()[-1].startswith("endsolid")
        assert text.count("facet normal") == result["report"]["triangles"]

    def test_obj_merges_planar_faces_unless_told_not_to(self, tmp_path):
        merged, merged_path = _export(tmp_path / "m", BOX_SOURCE, format="obj")
        plain, plain_path = _export(tmp_path / "p", BOX_SOURCE, format="obj", merge_planar=False)
        assert merged["ok"] is True and plain["ok"] is True
        merged_faces = [line for line in merged_path.read_text().splitlines() if line[:2] == "f "]
        plain_faces = [line for line in plain_path.read_text().splitlines() if line[:2] == "f "]
        # A box collapses towards six n-gons; unmerged it is every triangle.
        assert len(merged_faces) < len(plain_faces) == plain["report"]["triangles"]
        assert any(len(face.split()) > 4 for face in merged_faces)

    def test_step_falls_back_to_the_faceted_writer_and_says_so(self, tmp_path):
        """Asking for analytic STEP without the private tier still writes a file.

        The analytic writer — one ``PLANE`` per face instead of thousands of
        facets — is the ``step_export`` plugin kind, which nothing in this
        repository fills.  The export never fails for want of it: it writes
        the faceted STEP and puts the reason in ``report["tier"]``, so a user
        gets a usable file and one sentence saying what a better one would
        need.  The filled half of this is in
        ``tests/plugins/test_degradation.py``.

        ``tier.absent`` is what makes this true of the *repository* rather
        than of the machine: a developer with diff-brep installed alongside
        would otherwise take the analytic path and never run this.
        """
        from cadjoint import tier
        from cadjoint.enums import PluginKind

        with tier.absent():
            result, path = _export(tmp_path, BOX_SOURCE, format="step")
            assert result["ok"] is True, result
            assert result["report"]["path"] == "mesh"
            assert result["report"]["tier"] == tier.message(PluginKind.STEP_EXPORT.value)
        text = path.read_text()
        assert text.startswith("ISO-10303-21;")
        assert "MANIFOLD_SOLID_BREP(" in text

    def test_step_facets_through_the_mesh_writer_when_asked(self, tmp_path):
        result, path = _export(tmp_path, BOX_SOURCE, format="step", analytic=False)
        assert result["ok"] is True, result
        assert result["report"]["path"] == "mesh"
        assert "faces" not in result["report"]
        text = path.read_text()
        assert text.startswith("ISO-10303-21;")
        assert "MANIFOLD_SOLID_BREP(" in text

    def test_any_sdf_variable_can_be_named(self, tmp_path):
        result, path = _export(tmp_path, PLACED_SOURCE, format="stl", name="part")
        assert result["ok"] is True, result
        assert path.stat().st_size > 84

    def test_a_missing_or_non_sdf_name_lists_the_exportable_ones(self, tmp_path):
        with pytest.raises(ValueError, match=r"no SDF object named 'nope'.*'scene', 'part'"):
            _export(tmp_path, PLACED_SOURCE, format="stl", name="nope")
        with pytest.raises(ValueError, match=r"'count' is a int, not an SDF object"):
            _export(tmp_path, PLACED_SOURCE, format="stl", name="count")

    def test_the_worker_protocol_fields_are_not_request_fields(self, tmp_path):
        # `mode` and `path` ride along on every worker request; the model
        # forbids extras, so the worker has to strip them before validating.
        result, _ = _export(tmp_path, BOX_SOURCE, format="stl")
        assert result["ok"] is True
        assert export_scene({"source": BOX_SOURCE, "format": "stl"})["ok"] is False

    def test_vtk_writes_the_solved_study(self, tmp_path):
        pytest.importorskip("jax_fem")
        pytest.importorskip("meshio")
        result, path = _export(tmp_path, STUDY_SOURCE, format="vtk", name="bar")
        assert result["ok"] is True, result
        assert result["report"] == {"study": "bar", "cached": False}
        text = path.read_text(errors="replace")
        assert text.startswith("# vtk DataFile")
        assert "temperature" in text

    def test_vtk_names_the_declared_studies_on_a_miss(self, tmp_path):
        with pytest.raises(ValueError, match=r"no study named 'nope' \(declared: 'bar'\)"):
            _export(tmp_path, STUDY_SOURCE, format="vtk", name="nope")


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


def _token(base: str) -> str:
    with urlopen(base + "/api/session") as response:
        return json.load(response)["token"]


def _post(base: str, payload: dict, token: str | None = None) -> Request:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Cadjoint-Token"] = token
    return Request(
        base + "/api/export", data=json.dumps(payload).encode(), headers=headers, method="POST"
    )


class TestHttp:
    def test_export_requires_the_session_token(self):
        with _running_server() as base:
            with pytest.raises(HTTPError) as error:
                urlopen(_post(base, {"source": BOX_SOURCE, "format": "stl"}))
            assert error.value.code == 403

    def test_a_bad_request_is_json_and_422(self):
        with _running_server() as base:
            token = _token(base)
            with pytest.raises(HTTPError) as error:
                urlopen(_post(base, {"source": BOX_SOURCE, "format": "dxf"}, token))
            assert error.value.code == 422
            body = json.load(error.value)
            assert body["ok"] is False and "format" in body["error"]
            assert body["job_id"].startswith("job-")

    def test_the_file_comes_back_as_a_tracked_attachment(self):
        with _running_server() as base:
            token = _token(base)
            request = _post(base, {"source": BOX_SOURCE, "format": "stl", "resolution": 8}, token)
            with urlopen(request, timeout=300) as response:
                headers = response.headers
                data = response.read()
            assert headers["Content-Type"] == "model/stl"
            assert headers["Content-Disposition"] == 'attachment; filename="scene.stl"'
            assert headers["Cache-Control"] == "no-store"
            (count,) = struct.unpack("<I", data[80:84])
            assert len(data) == 84 + 50 * count == int(headers["Content-Length"])

            report = json.loads(headers["X-Cadjoint-Export"])
            assert report["format"] == "stl" and report["name"] == "scene"
            assert report["size"] == len(data)
            assert report["report"]["triangles"] == count

            job = REGISTRY.get(headers["X-Cadjoint-Job"])
            assert job is not None
            assert job.kind == "export" and job.status == "done" and job.ok is True
            assert job.fields == {"format": "stl", "resolution": 8}
            # The bytes went to the client, not the store.
            assert job.result_json is None
            assert job.summary()["result_available"] is False
