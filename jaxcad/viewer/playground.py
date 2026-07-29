"""Local server for the JAXCAD browser playground.

Serves the built frontend (``jaxcad/viewer/static``) and a small JSON API:

- ``GET  /api/session``  session token and the starter program
- ``POST /compile``      run the user's Python in a disposable child process
                         and return WGSL shaders plus the construction tree
- ``POST /patch``        rewrite sketch vertex literals in the user's source

Everything is loopback-only and token-gated. ``/compile`` executes the editor's
Python on this machine — only run code you trust. ``/patch`` is pure text
surgery and never executes anything.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jaxcad.viewer._patch import PatchError, apply_operation

DEFAULT_PORT = 8765
MAX_SOURCE_BYTES = 100_000
COMPILE_TIMEOUT_SECONDS = 20
STATIC_ROOT = Path(__file__).resolve().parent / "static"

CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
}

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

EXAMPLE_SOURCE = """from jaxcad.construction import PolygonProfile, SketchPlane, extrude
from jaxcad.render import Material
from jaxcad.sdf.boolean import Union
from jaxcad.sdf.primitives import Plane, Sphere
from jaxcad.sdf.transforms import Translate

# Sketch a profile on a work plane, then extrude it. Click a vertex handle in
# the viewer to find it here; drag it, or use "Add vertex", and this code is
# rewritten to match.
profile = PolygonProfile(
    [[-1.1, -0.7], [1.1, -0.7], [1.1, 0.3], [0.0, 1.0], [-1.1, 0.3]],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0]),
    name="house",
)
body = extrude(
    profile,
    depth=0.9,
    material=Material(color=[0.85, 0.45, 0.12], roughness=0.35),
)

scene = Union(
    body,
    Translate(
        Sphere(0.45, material=Material(color=[0.1, 0.3, 0.8], roughness=0.22)),
        [1.95, -0.25, 0.0],
    ),
    Plane(-1.25, material=Material(color=[0.12, 0.14, 0.18], roughness=0.8)),
    smoothness=0.0,
)
"""

MISSING_BUILD_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>JAXCAD Playground</title></head>
<body style="font: 15px/1.6 system-ui; max-width: 46rem; margin: 4rem auto; padding: 0 1.5rem">
<h1>Frontend build missing</h1>
<p>The playground UI has not been built into
<code>jaxcad/viewer/static</code>. Build it with:</p>
<pre style="background:#f4f4f2;padding:1rem;border-radius:8px">cd frontend
npm install
npm run build</pre>
<p>Or run the Vite dev server (<code>npm run dev</code>) alongside this server.</p>
</body></html>
"""


def compile_source(source: str, timeout: float = COMPILE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Compile playground source in a disposable child process."""
    if not isinstance(source, str):
        return {"ok": False, "error": "Source must be a string."}
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return {
            "ok": False,
            "error": f"Source is larger than the {MAX_SOURCE_BYTES:,}-byte limit.",
        }

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "jaxcad.viewer._compile_worker"],
            input=json.dumps({"source": source}),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"Compilation exceeded the {timeout:g}-second timeout.",
        }

    if completed.returncode != 0:
        detail = completed.stderr.strip() or "The compiler process exited unexpectedly."
        return {"ok": False, "error": detail[-8_000:]}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return {"ok": False, "error": f"Invalid compiler response:\n{detail[-8_000:]}"}
    if not isinstance(result, dict):
        return {"ok": False, "error": "Invalid compiler response."}
    return result


def patch_source(request: dict[str, Any]) -> dict[str, Any]:
    """Apply one viewer edit to the user's program text.

    Args:
        request: ``{"source", "op", "line", "index"}`` plus ``"xy"`` for
            operations that place a vertex.

    Returns:
        ``{"ok": True, "source": ...}`` or ``{"ok": False, "error": ...}``.
    """
    source = request.get("source")
    operation = request.get("op")
    if not isinstance(source, str):
        return {"ok": False, "error": "The patch request must contain a string `source` field."}
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return {"ok": False, "error": f"Source is larger than the {MAX_SOURCE_BYTES:,}-byte limit."}
    if not isinstance(operation, str):
        return {"ok": False, "error": "The patch request must contain a string `op` field."}

    arguments: dict[str, Any] = {}
    for key in ("line", "index"):
        value = request.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            return {"ok": False, "error": f"The patch request needs an integer `{key}`."}
        arguments[key] = value
    if operation in {"set_vertex", "insert_vertex"}:
        xy = request.get("xy")
        if (
            not isinstance(xy, (list, tuple))
            or len(xy) != 2
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in xy
            )
        ):
            return {"ok": False, "error": "The patch request needs `xy` as two numbers."}
        arguments["xy"] = (float(xy[0]), float(xy[1]))

    try:
        return {"ok": True, "source": apply_operation(source, operation, **arguments)}
    except PatchError as error:
        return {"ok": False, "error": str(error)}


def _is_loopback_host(host_header: str | None) -> bool:
    if not host_header:
        return False
    hostname = urlsplit(f"//{host_header}").hostname
    return hostname in {"127.0.0.1", "localhost", "::1"}


def resolve_static(url_path: str) -> Path | None:
    """Map a URL path to a file inside the static root, or None if unavailable.

    Refuses anything that escapes the static directory, so ``..`` traversal and
    absolute paths cannot reach the rest of the filesystem.
    """
    relative = url_path.lstrip("/") or "index.html"
    candidate = (STATIC_ROOT / relative).resolve()
    try:
        candidate.relative_to(STATIC_ROOT)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def make_handler(token: str):
    """Build a request handler bound to one session token."""

    class PlaygroundHandler(BaseHTTPRequestHandler):
        server_version = "jaxcad-playground"
        protocol_version = "HTTP/1.1"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str, **headers: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(
                status,
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
                Cache_Control="no-store",
            )

        def _reject(self, status: HTTPStatus, message: str) -> None:
            """Refuse a request without reading its body.

            Closes the connection so any unread body cannot be mistaken for the
            next request on a keep-alive connection.
            """
            self.close_connection = True
            self._send_json(status, {"ok": False, "error": message})

        def _host_allowed(self) -> bool:
            if _is_loopback_host(self.headers.get("Host")):
                return True
            self._reject(HTTPStatus.FORBIDDEN, "Invalid Host header.")
            return False

        def _token_valid(self) -> bool:
            supplied = self.headers.get("X-Jaxcad-Token", "")
            if secrets.compare_digest(supplied, token):
                return True
            self._reject(HTTPStatus.FORBIDDEN, "Invalid session token.")
            return False

        def _read_json(self) -> dict[str, Any] | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_SOURCE_BYTES * 2:
                self._reject(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size.")
                return None
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON request."}
                )
                return None
            if not isinstance(payload, dict):
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON request."}
                )
                return None
            return payload

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

            target = resolve_static(path)
            if target is None:
                if path == "/":
                    self._send(
                        HTTPStatus.OK,
                        MISSING_BUILD_PAGE.encode("utf-8"),
                        "text/html; charset=utf-8",
                        Cache_Control="no-store",
                    )
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            body = target.read_bytes()
            content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
            is_entry = target.name == "index.html"
            self._send(
                HTTPStatus.OK,
                body,
                content_type,
                Cache_Control="no-store" if is_entry else "public, max-age=31536000, immutable",
                Content_Security_Policy=CONTENT_SECURITY_POLICY,
            )

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._host_allowed():
                return
            path = urlsplit(self.path).path
            if path not in {"/compile", "/patch"}:
                self._reject(HTTPStatus.NOT_FOUND, "Not found.")
                return
            if not self._token_valid():
                return
            payload = self._read_json()
            if payload is None:
                return

            if path == "/compile":
                result = compile_source(payload.get("source"))
            else:
                result = patch_source(payload)
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.UNPROCESSABLE_ENTITY
            self._send_json(status, result)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

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
    print(f"JAXCAD playground: {url}")
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
