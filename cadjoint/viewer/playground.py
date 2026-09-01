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
- ``GET  /api/scenes``   list saved scene files in ``./scenes``
- ``POST /api/scenes/load``  read one saved scene file
- ``POST /api/scenes/save``  write one scene file into ``./scenes``

Everything is loopback-only and token-gated. ``/compile`` and ``/api/mesh``
execute the editor's Python on this machine — only run code you trust.
``/patch`` is pure text surgery and never executes anything. Scene files are
confined to the ``scenes`` directory under the server's working directory.

This module is the server's front door: the routing table below says which
URL runs which endpoint, and it re-exports the API those endpoints live in.
The parts underneath it:

- :mod:`cadjoint.viewer._http` — response framing, the Host/token checks,
  NDJSON streaming, static files
- :mod:`cadjoint.viewer._worker_client` — the endpoints that run the editor's
  Python in a child process
- :mod:`cadjoint.viewer._patch_requests` — ``/patch`` request validation
- :mod:`cadjoint.viewer._scenes` — saved scene files
- :mod:`cadjoint.viewer._example_scene` — the starter program
"""

from __future__ import annotations

import argparse
import secrets
import webbrowser
from collections.abc import Callable, Iterable
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer._http import STATIC_ROOT, PlaygroundBase, resolve_static
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
)

__all__ = [
    "COMPILE_TIMEOUT_SECONDS",
    "DEFAULT_PORT",
    "EXAMPLE_SOURCE",
    "MAX_SOURCE_BYTES",
    "MESH_INSPECT_TIMEOUT_SECONDS",
    "OPTIMIZE_MAX_STEPS",
    "OPTIMIZE_TIMEOUT_SECONDS",
    "SIMULATE_KINDS",
    "SIMULATE_TIMEOUT_SECONDS",
    "STATIC_ROOT",
    "compile_source",
    "create_server",
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
    "simulate_source",
]

DEFAULT_PORT = 8765


def _post_routes() -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Buffered POST endpoints, by path.

    Rebuilt per request on purpose: the endpoints are looked up as globals of
    this module at call time, so replacing one here (as the FEM-unavailable
    test does) is honoured by a running server.
    """
    return {
        "/compile": lambda payload: compile_source(payload.get("source")),
        "/api/mesh": lambda payload: mesh_source(payload.get("source")),
        "/api/simulate": simulate_source,
        "/api/mesh_inspect": mesh_inspect_source,
        "/patch": patch_source,
        "/api/scenes/load": load_scene,
        "/api/scenes/save": save_scene,
    }


def _stream_routes() -> dict[str, Callable[[dict[str, Any]], Iterable[dict[str, Any]]]]:
    """POST endpoints whose response streams as chunked NDJSON, by path."""
    return {"/api/optimize": optimize_source_events}


def _response_status(result: dict[str, Any]) -> HTTPStatus:
    """The HTTP status one endpoint result deserves."""
    if result.get("ok"):
        return HTTPStatus.OK
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

            self._serve_static(path)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._host_allowed():
                return
            path = urlsplit(self.path).path

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

    return PlaygroundHandler


def create_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Create a loopback-only playground server."""
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(token))
    server.daemon_threads = True
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
