"""Local server for the CADJOINT browser playground.

Serves the built frontend (``cadjoint/viewer/static``) and a small JSON API:

- ``GET  /api/session``  session token and the starter program
- ``POST /compile``      run the user's Python in a disposable child process
                         and return WGSL shaders plus the construction tree
- ``POST /patch``        rewrite sketch vertex literals in the user's source
- ``POST /api/mesh``     run the source again and return only the dual-contour
                         mesh edges (requested lazily by the viewer)
- ``POST /api/simulate`` run a study the program declares (``cadjoint.fem``)
                         picked by ``name`` (``kind="study"``); the response
                         carries the solved surface, the result summary, and
                         the built mesh's inspection report
- ``POST /api/mesh_inspect`` build a declared ``SimMesh`` (or a study's
                         implicit mesh) and return its inspection report plus
                         a renderable surface with a scaled-jacobian quality
                         heatmap — look at a mesh before solving on it
- ``POST /api/optimize`` run an ``Optimization`` the program declares
                         (``cadjoint.optimize``), picked by ``name``. The
                         response STREAMS as chunked NDJSON: one
                         ``{"event": "progress", ...}`` line per optimizer
                         step, then ``{"event": "done", ...}`` carrying the
                         full result — the optimized free-parameter values
                         patched back into ``source`` exactly like
                         ``/patch``, plus a ``simulate`` block (the solved
                         optimized design) for study-backed runs
- ``POST /api/export``   run the source again and write one SDF object of it
                         (``name``, default ``scene``) as ``obj``, ``stl`` or
                         ``step`` — or a declared study's solved fields as
                         ``vtk`` — at ``resolution`` cells; the response is
                         the file itself, as an attachment, with the job id
                         in ``X-Cadjoint-Job``
- ``POST /api/lint``     static-analyse the editor's Python with ruff and
                         return CodeMirror-shaped diagnostics (plus the last
                         compile traceback when it named a line of this text)
- ``POST /api/complete`` jedi completions at a caret, resolved against the
                         installed ``cadjoint`` so they know its real types
- ``POST /api/signature`` the signature of the call the caret sits inside
- ``GET  /api/scenes``   list saved scene files in ``./scenes``
- ``POST /api/scenes/load``  read one saved scene file
- ``POST /api/scenes/save``  write one scene file into ``./scenes``
- ``GET  /api/jobs``     every registered job (newest first) plus live totals
                         — running count, CPU, RSS, uptime, host capacity —
                         cheap enough to poll at 1 Hz
- ``GET  /api/jobs/<id>``        one job with its resource samples and, for an
                         optimize run, its per-step progress
- ``GET  /api/jobs/<id>/result`` the full response payload that job produced,
                         byte-identical to the one its request received
- ``POST /api/jobs/<id>/cancel`` kill a running job's worker subprocess
- ``POST /api/jobs/clear``       drop every finished job

Every request that costs real time is registered in
:mod:`cadjoint.viewer._jobs` as it runs: results outlive the panel that asked
for them, a process monitor can see what is burning the machine, and a run
can be cancelled for real.  Registration adds ``job_id`` to every response
(and to every streamed optimize event) and changes nothing else.

Everything is loopback-only and token-gated. ``/compile`` and ``/api/mesh``
execute the editor's Python on this machine — only run code you trust.
``/patch`` is pure text surgery and never executes anything, and neither do
``/api/lint``, ``/api/complete`` and ``/api/signature`` — ruff and jedi read
the program without importing it. Scene files are confined to the ``scenes``
directory under the server's working directory.

This module is the server's front door: the routing table below says which
URL runs which endpoint, and it re-exports the API those endpoints live in.
The parts underneath it:

- :mod:`cadjoint.viewer._http` — response framing, the Host/token checks,
  NDJSON streaming, static files
- :mod:`cadjoint.viewer._worker_client` — the endpoints that run the editor's
  Python in a child process
- :mod:`cadjoint.viewer._export` — ``/api/export``: which writer, which
  content type, and the worker half that extracts and writes
- :mod:`cadjoint.viewer._patch_requests` — ``/patch`` request validation
- :mod:`cadjoint.viewer._intelligence` — the ruff and jedi endpoints
- :mod:`cadjoint.viewer._scenes` — saved scene files
- :mod:`cadjoint.viewer._example_scene` — the starter program
"""

from __future__ import annotations

import argparse
import json
import secrets
import webbrowser
from collections.abc import Callable, Iterable
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer._export import EXPORT_TIMEOUT_SECONDS, export_source
from cadjoint.viewer._http import STATIC_ROOT, PlaygroundBase, resolve_static
from cadjoint.viewer._intelligence import (
    complete_source,
    lint_source,
    record_compile,
    signature_source,
    warm_up,
)
from cadjoint.viewer._jobs import JOB_KINDS, REGISTRY
from cadjoint.viewer._limits import MAX_SOURCE_BYTES
from cadjoint.viewer._patch_requests import patch_source
from cadjoint.viewer._scenes import (
    list_scenes,
    load_scene,
    sanitize_scene_name,
    save_scene,
    scenes_root,
)
from cadjoint.viewer._worker_client import (
    COMPILE_TIMEOUT_SECONDS,
    MESH_INSPECT_TIMEOUT_SECONDS,
    OPTIMIZE_MAX_STEPS,
    OPTIMIZE_TIMEOUT_SECONDS,
    SIMULATE_KINDS,
    SIMULATE_TIMEOUT_SECONDS,
    compile_source,
    mesh_inspect_source,
    mesh_source,
    optimize_source,
    optimize_source_events,
    simulate_source,
    warm_start,
)

__all__ = [
    "COMPILE_TIMEOUT_SECONDS",
    "DEFAULT_PORT",
    "EXAMPLE_SOURCE",
    "EXPORT_TIMEOUT_SECONDS",
    "JOB_KINDS",
    "MAX_SOURCE_BYTES",
    "MESH_INSPECT_TIMEOUT_SECONDS",
    "OPTIMIZE_MAX_STEPS",
    "OPTIMIZE_TIMEOUT_SECONDS",
    "REGISTRY",
    "SIMULATE_KINDS",
    "SIMULATE_TIMEOUT_SECONDS",
    "STATIC_ROOT",
    "compile_source",
    "complete_source",
    "create_server",
    "export_source",
    "lint_source",
    "list_scenes",
    "load_scene",
    "main",
    "make_handler",
    "mesh_inspect_source",
    "mesh_source",
    "optimize_source",
    "optimize_source_events",
    "patch_source",
    "resolve_static",
    "sanitize_scene_name",
    "save_scene",
    "scenes_root",
    "signature_source",
    "simulate_source",
    "warm_start",
]

DEFAULT_PORT = 8765


def _post_routes() -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Buffered POST endpoints, by path.

    Rebuilt per request on purpose: the endpoints are looked up as globals of
    this module at call time, so replacing one here (as the FEM-unavailable
    test does) is honoured by a running server.
    """
    return {
        # `record_compile` is a tap, not a filter: it returns the worker's
        # result unchanged and only remembers a traceback that named a line
        # of this program, so `/api/lint` can show the failure in the gutter.
        "/compile": _tracked(
            "compile",
            lambda payload: record_compile(
                payload.get("source"), compile_source(payload.get("source"))
            ),
        ),
        "/api/mesh": _tracked(
            "mesh",
            lambda payload: record_compile(
                payload.get("source"), mesh_source(payload.get("source"))
            ),
        ),
        "/api/simulate": _tracked("simulate", simulate_source),
        "/api/mesh_inspect": _tracked("mesh_inspect", mesh_inspect_source),
        "/patch": patch_source,
        "/api/lint": _tracked("lint", lint_source),
        "/api/complete": complete_source,
        "/api/signature": signature_source,
        "/api/scenes/load": load_scene,
        "/api/scenes/save": save_scene,
    }


#: Request fields worth remembering on a job: which study, which optimization,
#: how many steps, which export format.  Small scalars only — the source is kept as a hash, not a
#: copy, and nothing else about a request is the monitor's business.
JOB_FIELD_KEYS = ("kind", "name", "steps", "cached", "format", "resolution")


def _job_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """The identifying fields of a request, for its job summary."""
    return {
        key: payload[key]
        for key in JOB_FIELD_KEYS
        if isinstance(payload.get(key), (str, int, float, bool))
    }


def _tracked(
    kind: str, endpoint: Callable[[dict[str, Any]], dict[str, Any]]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a buffered endpoint so its work is registered as a job.

    The contract does not change: the same payload comes back, with a
    ``job_id`` added, and the same payload can be fetched again later from
    ``GET /api/jobs/<id>/result``.  The job is bound to this thread for the
    duration, which is how the worker subprocess it spawns becomes
    cancellable and observable.

    Args:
        kind: One of :data:`cadjoint.viewer._jobs.JOB_KINDS`.
        endpoint: The endpoint function, unchanged.

    Returns:
        A route with the endpoint's signature.
    """

    def route(payload: dict[str, Any]) -> dict[str, Any]:
        with REGISTRY.track(kind, source=payload.get("source"), fields=_job_fields(payload)) as job:
            return REGISTRY.finish(job, endpoint(payload))

    return route


def _tracked_stream(
    kind: str, endpoint: Callable[[dict[str, Any]], Iterable[dict[str, Any]]]
) -> Callable[[dict[str, Any]], Iterable[dict[str, Any]]]:
    """Wrap a streaming endpoint so its work is registered as a job.

    Every relayed event gains ``job_id``; progress events are mirrored onto
    the job as they pass, so a monitor polling ``/api/jobs`` sees the same
    descent the streaming client sees, and the finished run's whole payload
    lands in the result store for replay.

    Args:
        kind: One of :data:`cadjoint.viewer._jobs.JOB_KINDS`.
        endpoint: The event-stream endpoint, unchanged.

    Returns:
        A route with the endpoint's signature.
    """

    def route(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        with REGISTRY.track(kind, source=payload.get("source"), fields=_job_fields(payload)) as job:
            final: dict[str, Any] | None = None
            for event in endpoint(payload):
                if event.get("event") == "progress":
                    job.record_progress(event)
                    yield {**event, "job_id": job.id}
                else:
                    final = {key: value for key, value in event.items() if key != "event"}
            yield {"event": "done", **REGISTRY.finish(job, final)}

    return route


def _export_route(payload: dict[str, Any]) -> dict[str, Any]:
    """``/api/export`` as a tracked job whose result is a file, not JSON.

    The same registration as :func:`_tracked`, with one difference: the
    bytes are lifted out before the job is finished, so what the registry
    records (and would store, if ``export`` were a result kind) is the
    small summary — format, filename, size, report — and never the file.
    The bytes ride back to the handler on the returned dict under ``data``.
    """
    with REGISTRY.track("export", source=payload.get("source"), fields=_job_fields(payload)) as job:
        result = export_source(payload)
        data = result.pop("data", None)
        summary = REGISTRY.finish(job, result)
        if data is not None and summary.get("ok"):
            summary["data"] = data
        return summary


def _stream_routes() -> dict[str, Callable[[dict[str, Any]], Iterable[dict[str, Any]]]]:
    """POST endpoints whose response streams as chunked NDJSON, by path."""
    return {"/api/optimize": _tracked_stream("optimize", optimize_source_events)}


def _response_status(result: dict[str, Any]) -> HTTPStatus:
    """The HTTP status one endpoint result deserves."""
    if result.get("ok"):
        return HTTPStatus.OK
    if result.get("error_kind") == "cancelled":
        # The request did not fail on its own terms: somebody stopped it.
        return HTTPStatus.CONFLICT
    if result.get("error_kind") == "fem_unavailable":
        # A missing optional solver extra is "not implemented here", not a bad
        # request; the UI shows it as an install hint.
        return HTTPStatus.NOT_IMPLEMENTED
    return HTTPStatus.UNPROCESSABLE_ENTITY


def make_handler(token: str):
    """Build a request handler bound to one session token.

    Only routing lives here — which URL runs which endpoint, and what status
    its result carries.  The transport underneath (framing, headers, the
    Host and token checks, NDJSON chunking, static files) is
    :class:`cadjoint.viewer._http.PlaygroundBase`.
    """

    class PlaygroundHandler(PlaygroundBase):
        session_token = token

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._host_allowed():
                return
            path = urlsplit(self.path).path

            if path == "/api/session":
                # Same-origin read only: a cross-origin page cannot see this
                # response, so the token still gates POSTs against CSRF.
                self._send_json(
                    HTTPStatus.OK, {"ok": True, "token": token, "example": EXAMPLE_SOURCE}
                )
                return

            if path == "/api/scenes":
                self._send_json(HTTPStatus.OK, list_scenes())
                return

            if path == "/api/jobs" or path.startswith("/api/jobs/"):
                if not self._token_valid():
                    return
                self._serve_job_read(path)
                return

            self._serve_static(path)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._host_allowed():
                return
            path = urlsplit(self.path).path

            if path == "/api/jobs/clear" or (
                path.startswith("/api/jobs/") and path.endswith("/cancel")
            ):
                if not self._token_valid():
                    return
                self._drain_body()
                self._serve_job_command(path)
                return

            if path == "/api/export":
                # The one endpoint whose answer is a file: the bytes go out
                # as an attachment, and only a failure is JSON.
                if not self._token_valid():
                    return
                payload = self._read_json()
                if payload is None:
                    return
                self._serve_export(_export_route(payload))
                return

            stream = _stream_routes().get(path)
            if stream is not None:
                # Streamed: progress events per optimizer step, then `done`
                # with the full response — see optimize_source_events.
                if not self._token_valid():
                    return
                payload = self._read_json()
                if payload is None:
                    return
                self._stream_ndjson(stream(payload))
                return

            handler = _post_routes().get(path)
            if handler is None:
                self._reject(HTTPStatus.NOT_FOUND, "Not found.")
                return
            if not self._token_valid():
                return
            payload = self._read_json()
            if payload is None:
                return

            result = handler(payload)
            self._send_json(_response_status(result), result)

        def _serve_export(self, result: dict[str, Any]) -> None:
            """Send an export's file as an attachment, or its failure as JSON."""
            data = result.pop("data", None)
            if not result.get("ok") or not isinstance(data, bytes):
                self._send_json(_response_status(result), result)
                return
            self._send(
                HTTPStatus.OK,
                data,
                result["content_type"],
                Cache_Control="no-store",
                Content_Disposition=f'attachment; filename="{result["filename"]}"',
                X_Cadjoint_Job=str(result.get("job_id", "")),
                X_Cadjoint_Export=json.dumps(
                    {key: result.get(key) for key in ("format", "name", "size", "report")}
                ),
            )

        # ── the job registry: what is running, what ran, and its results ──

        def _serve_job_read(self, path: str) -> None:
            """Answer ``GET /api/jobs``, ``/api/jobs/<id>`` and ``.../result``."""
            if path == "/api/jobs":
                self._send_json(HTTPStatus.OK, REGISTRY.snapshot())
                return
            rest = path[len("/api/jobs/") :]
            wants_result = rest.endswith("/result")
            job_id = rest[: -len("/result")] if wants_result else rest
            job = REGISTRY.get(job_id)
            if job is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"No job {job_id!r}.", "job_id": job_id},
                )
                return
            if not wants_result:
                self._send_json(HTTPStatus.OK, {"ok": True, "job": job.detail()})
                return
            if job.status in ("queued", "running"):
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"ok": False, "error": "Job is still running.", "job_id": job_id},
                )
                return
            if job.result_json is None:
                # Either the kind is not kept (lint) or the payload was evicted
                # to stay inside the store's byte budget: the job is history,
                # its result is not.
                self._send_json(
                    HTTPStatus.GONE,
                    {"ok": False, "error": "No stored result for this job.", "job_id": job_id},
                )
                return
            self._send_raw_json(HTTPStatus.OK, job.result_json)

        def _serve_job_command(self, path: str) -> None:
            """Answer ``POST /api/jobs/<id>/cancel`` and ``POST /api/jobs/clear``."""
            if path == "/api/jobs/clear":
                self._send_json(HTTPStatus.OK, REGISTRY.clear())
                return
            job_id = path[len("/api/jobs/") : -len("/cancel")]
            job = REGISTRY.get(job_id)
            if job is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"No job {job_id!r}.", "job_id": job_id},
                )
                return
            if not job.cancel():
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {
                        "ok": False,
                        "error": f"Job is already {job.status}.",
                        "job_id": job_id,
                        "status": job.status,
                    },
                )
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "job_id": job_id, "status": "cancelled"})

    return PlaygroundHandler


def create_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Create a loopback-only playground server."""
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(token))
    server.daemon_threads = True
    # Jedi's first analysis costs a few hundred milliseconds; pay it on a
    # background thread now so the editor's first completion is already warm.
    warm_up()
    # Same idea one order of magnitude up: one background `compile` and one
    # `mesh` of the scene the editor opens with, so the first real request
    # meets a warm XLA cache instead of paying 45-53 s for a cold one.
    # Both are daemon threads and neither delays `serve_forever`.
    warm_start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="local port (default: 8765)")
    parser.add_argument("--open", action="store_true", help="open the playground in your browser")
    args = parser.parse_args()

    server = create_server(args.port)
    address, port = server.server_address
    url = f"http://{address}:{port}/"
    print(f"CADJOINT playground: {url}")
    if not (STATIC_ROOT / "index.html").is_file():
        print("Frontend build missing — run `npm install && npm run build` in frontend/.")
    print("Local source is executed as Python in a timed child process. Only run code you trust.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping playground.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
