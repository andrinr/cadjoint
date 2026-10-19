"""The HTTP and worker contract, against the server that serves it.

:mod:`cadjoint.viewer.schema.api` lists every route with its request and
response models, and the frontend's client is typed by the TypeScript
generated from that list. These tests pin the list to the server: every
route it names is answered, every route the server answers is named, every
response a cheap request can provoke matches the route's models, and the
worker boundary refuses a reply its model does not allow.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from cadjoint.enums import ConstraintKind
from cadjoint.studies.selection import Nodes
from cadjoint.viewer import playground
from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer.schema import api, payloads, worker
from cadjoint.viewer.schema.emit import typescript_source
from cadjoint.viewer.source_map.constraints import CONSTRAINT_CLASS_KINDS

ROUTES = {route.path: route for route in api.ROUTES}


@contextmanager
def serving() -> Iterator[tuple[str, str]]:
    """A live playground on a free port, and its session token."""
    server = playground.create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        base = f"http://{host}:{port}"
        with urlopen(f"{base}/api/session") as response:
            token = json.loads(response.read())["token"]
        yield base, token
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture(scope="module")
def live() -> Iterator[tuple[str, str]]:
    with serving() as pair:
        yield pair


def call(
    live: tuple[str, str], method: str, path: str, body: dict | None = None
) -> tuple[int, dict[str, str], bytes]:
    """One request with the session token; ``(status, headers, body)``."""
    base, token = live
    data = None if body is None else json.dumps(body).encode()
    request = Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "X-Cadjoint-Token": token},
    )
    try:
        with urlopen(request) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as error:
        return error.code, dict(error.headers), error.read()


def matches(route: api.Route, document: dict[str, Any]) -> BaseModel:
    """Parse a response against the route's union (plus ``job_id`` when tracked)."""
    if route.tracked:
        api.JobStamped.model_validate({"job_id": document.get("job_id")})
        if document.get("ok") is True and not route.stream:
            # The registry's stamp, modelled once as JobStamped rather than on
            # every success model (failures and stream lines declare it).
            document = {key: value for key, value in document.items() if key != "job_id"}
    errors = []
    for model in route.response:
        try:
            return model.model_validate(document)
        except ValidationError as error:
            errors.append(f"{model.__name__}: {error}")
    raise AssertionError(f"{route.path} answered outside its contract:\n" + "\n".join(errors))


# ── the table covers the server, and the server covers the table ───────────


def test_every_buffered_and_streamed_post_route_is_in_the_table():
    served = set(playground._post_routes()) | set(playground._stream_routes())
    named = {route.path for route in api.ROUTES}
    assert served <= named, f"routes without a contract entry: {sorted(served - named)}"


def test_every_route_in_the_table_is_answered(live):
    for route in api.ROUTES:
        path = route.path.replace("{job_id}", "no-such-job")
        body = None if route.method == "GET" else {}
        status, headers, raw = call(live, route.method, path, body)
        assert status != 404 or b"No job" in raw, f"{route.method} {path} is not routed"
        if not route.stream:
            assert headers["Content-Type"].startswith("application/json"), path


def test_every_response_carries_the_contract_version(live):
    for path in ("/api/session", "/api/jobs", "/api/scenes"):
        _, headers, _ = call(live, "GET", path)
        assert headers[api.API_VERSION_HEADER] == str(api.API_VERSION)


def test_the_session_reports_the_contract_version(live):
    _, _, raw = call(live, "GET", "/api/session")
    session = matches(ROUTES["/api/session"], json.loads(raw))
    assert session.api_version == api.API_VERSION


def test_the_generated_typescript_carries_the_same_version():
    assert f"export type ApiVersion = {api.API_VERSION};" in typescript_source()


# ── what a cheap request gets back matches its route ───────────────────────


CHEAP_REQUESTS: list[tuple[str, str, dict | None]] = [
    ("GET", "/api/session", None),
    ("GET", "/api/capabilities", None),
    ("GET", "/api/scenes", None),
    ("GET", "/api/jobs", None),
    ("POST", "/api/scenes/load", {"name": "../escape.py"}),
    ("POST", "/api/scenes/load", {"name": "no-such-scene.py"}),
    ("POST", "/api/patch", {"source": EXAMPLE_SOURCE, "op": "add_sketch", "origin": [0, 0, 0]}),
    ("POST", "/api/patch", {"source": EXAMPLE_SOURCE, "op": "no_such_op"}),
    ("POST", "/api/lint", {"source": "x = 1\n"}),
    ("POST", "/api/lint", {"source": 3}),
    ("POST", "/api/complete", {"source": "import math\nmath.", "line": 2, "column": 5}),
    ("POST", "/api/signature", {"source": "print(", "line": 1, "column": 6}),
    # Refused before a worker starts, so they cost nothing — and still go
    # through the job registry, so `job_id` is checked too.
    ("POST", "/api/compile", {"source": 3}),
    ("POST", "/api/mesh", {"source": 3}),
    ("POST", "/api/brep", {"source": 3}),
    ("POST", "/api/simulate", {"source": "scene = None"}),
    ("POST", "/api/mesh_inspect", {"source": "scene = None", "name": ""}),
    ("POST", "/api/export", {"source": "scene = None", "format": "nope"}),
    ("POST", "/api/mesh_refresh", {"source": "scene = None"}),
    ("POST", "/api/jobs/cancel_client", {"client_id": "nobody"}),
    ("POST", "/api/jobs/clear", {}),
    ("GET", "/api/jobs/no-such-job", None),
    ("GET", "/api/jobs/no-such-job/result", None),
    ("POST", "/api/jobs/no-such-job/cancel", {}),
]


def _route_for(path: str) -> api.Route:
    if path in ROUTES:
        return ROUTES[path]
    rest = path.removeprefix("/api/jobs/")
    template = "/api/jobs/{job_id}" + ("/" + rest.split("/", 1)[1] if "/" in rest else "")
    return ROUTES[template]


@pytest.mark.parametrize(("method", "path", "body"), CHEAP_REQUESTS)
def test_a_cheap_response_matches_its_route(live, method, path, body):
    _, _, raw = call(live, method, path, body)
    matches(_route_for(path), json.loads(raw))


def test_a_finished_job_reads_back_within_the_contract(live):
    call(live, "POST", "/api/lint", {"source": "x = 1\n"})
    _, _, raw = call(live, "GET", "/api/jobs")
    snapshot = api.JobsSnapshot.model_validate(json.loads(raw))
    assert snapshot.jobs, "the lint request registered a job"
    _, _, raw = call(live, "GET", f"/api/jobs/{snapshot.jobs[0].job_id}")
    matches(ROUTES["/api/jobs/{job_id}"], json.loads(raw))


def test_an_invalid_optimize_request_streams_a_done_line_within_the_contract(live):
    _, _, raw = call(live, "POST", "/api/optimize", {"source": "scene = None"})
    lines = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    assert [line["event"] for line in lines] == ["done"]
    assert isinstance(matches(ROUTES["/api/optimize"], lines[0]), api.OptimizeFailed)


# ── the worker boundary ─────────────────────────────────────────────────────


def test_every_worker_mode_has_a_request_and_a_reply_model():
    assert set(worker.WORKER_REQUESTS) == set(worker.WORKER_REPLIES)
    assert set(worker.WORKER_REQUESTS) == set(worker.WorkerMode.__args__)


def test_a_reply_outside_its_model_becomes_a_contract_failure():
    reply = {"ok": True, "mesh_edges": None}  # `output` is missing
    checked = worker.checked_reply("mesh", reply)
    assert checked["ok"] is False
    assert checked["error_kind"] == "contract"
    assert "output" in checked["error"]
    payloads.WorkerFailure.model_validate(checked)


def test_a_reply_inside_its_model_passes_unchanged():
    reply = {"ok": True, "mesh_edges": None, "output": "", "refresh": {"seed": 1}}
    assert worker.checked_reply("mesh", reply) is reply
    failure = {"ok": False, "error": "boom", "error_kind": "fem_unavailable", "output": ""}
    assert worker.checked_reply("simulate", failure) is failure


def test_a_failure_with_a_stray_field_is_refused_too():
    checked = worker.checked_reply("compile", {"ok": False, "error": "x", "surprise": 1})
    assert checked["error_kind"] == "contract"


def test_the_server_refuses_to_send_a_malformed_worker_request():
    assert worker.checked_request({"mode": "compile", "source": "x"}) is None
    refused = worker.checked_request({"mode": "simulate", "source": "x", "name": "s"})
    assert refused is not None and refused["error_kind"] == "contract"
    assert worker.checked_request({"mode": "dance", "source": "x"})["error_kind"] == "contract"


def test_a_progress_line_outside_its_model_is_dropped():
    good = {
        "event": "progress",
        "step": 1,
        "steps": 3,
        "objective": 1.0,
        "grad_norm": 0.1,
        "elapsed": 0.5,
    }
    assert worker.checked_progress(good) is good
    assert worker.checked_progress({"event": "progress", "step": "one"}) is None


def test_the_worker_client_checks_the_replies_it_reads(monkeypatch):
    """A worker answering off-contract reaches the route as a contract failure."""
    from cadjoint.viewer import _worker_client

    class Finished:
        returncode = 0

        def communicate(self, request, timeout):
            return json.dumps({"ok": True, "kind": "mesh_inspect"}), ""

    monkeypatch.setattr(_worker_client.subprocess, "Popen", lambda *args, **kwargs: Finished())
    monkeypatch.setattr(_worker_client, "attach_process", lambda process: None)
    result = _worker_client.mesh_inspect_source({"source": "scene = None", "name": "m"})
    assert result["error_kind"] == "contract"


# ── the two drift bugs the contract closed ─────────────────────────────────


def test_every_selection_class_describes_into_the_selection_union():
    adapter = TypeAdapter(payloads.StudySelection)
    box = Nodes.box([0, 0, 0], [1, 1, 1])
    selections = [
        box,
        Nodes.sphere([0, 0, 0], 1.0),
        Nodes.halfspace([0, 0, 0], [0, 0, 1]),
        Nodes.cylinder([0, 0, 0], [0, 0, 1], 0.5),
        Nodes.cylinder([0, 0, 0], [0, 0, 1], 0.5, inner=0.25, half_length=1.0),
        Nodes.side("+x"),
        Nodes.predicate(lambda points: points[:, 0] > 0),
        box & Nodes.side("-z"),
        box | Nodes.side("-z"),
        ~box,
    ]
    kinds = set()
    for selection in selections:
        described = selection.describe()
        adapter.validate_python(described)
        kinds.add(described["kind"])
    models = {model.model_fields["kind"].annotation for model in payloads.SELECTION_MODELS}
    declared = {kind for annotation in models for kind in annotation.__args__}
    assert kinds == declared


def test_the_reported_constraint_kinds_are_the_source_maps_and_cover_the_addable_ones():
    reported = set(payloads.ReportedConstraintKind.__args__)
    assert reported == set(CONSTRAINT_CLASS_KINDS.values())
    assert {kind.value for kind in ConstraintKind} <= reported
    # The one kind a program can declare but the viewer cannot add.
    assert reported - {kind.value for kind in ConstraintKind} == {"equal_length"}


def test_the_construction_kinds_are_the_profile_and_every_primitive():
    from cadjoint.construction.solid import DIMENSIONS

    assert set(payloads.ConstructionKind.__args__) == {"profile", *DIMENSIONS}
