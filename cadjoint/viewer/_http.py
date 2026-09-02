"""HTTP plumbing for the playground server.

Everything about getting bytes on and off the wire, with no knowledge of
which endpoint runs where: response framing and the standard headers, JSON
and chunked-NDJSON writing, the loopback-Host and session-token checks that
gate every request, request-body reading, and static file serving out of
``cadjoint/viewer/static``.

:class:`PlaygroundBase` is the half of the request handler that never
changes when an endpoint is added.  The routing half — which URL calls
which endpoint — lives in :mod:`cadjoint.viewer.playground`, which
subclasses this and binds the session token.
"""

from __future__ import annotations

import json
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cadjoint.viewer._limits import MAX_SOURCE_BYTES

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

MISSING_BUILD_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CADJOINT Playground</title></head>
<body style="font: 15px/1.6 system-ui; max-width: 46rem; margin: 4rem auto; padding: 0 1.5rem">
<h1>Frontend build missing</h1>
<p>The playground UI has not been built into
<code>cadjoint/viewer/static</code>. Build it with:</p>
<pre style="background:#f4f4f2;padding:1rem;border-radius:8px">cd frontend
npm install
npm run build</pre>
<p>Or run the Vite dev server (<code>npm run dev</code>) alongside this server.</p>
</body></html>
"""


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


class PlaygroundBase(BaseHTTPRequestHandler):
    """The playground handler's transport half, without any routes.

    Subclasses bind ``session_token`` to the session's token and implement
    ``do_GET``/``do_POST`` in terms of the helpers here.
    """

    server_version = "cadjoint-playground"
    protocol_version = "HTTP/1.1"
    session_token = ""

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

    def _send_raw_json(self, status: HTTPStatus, body: str) -> None:
        """Send an already-serialized JSON document.

        The job store keeps results as the exact JSON text that was sent the
        first time, so re-fetching one costs neither a parse nor a re-encode
        and the bytes are identical to the original response.
        """
        self._send(
            status,
            body.encode("utf-8"),
            "application/json; charset=utf-8",
            Cache_Control="no-store",
        )

    def _drain_body(self) -> None:
        """Read and discard a request body, keeping keep-alive framing aligned.

        Used by the command endpoints that take no arguments (job cancel and
        clear) but may still be sent an empty JSON object by a client.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if 0 < length <= MAX_SOURCE_BYTES * 2:
            self.rfile.read(length)
        elif length != 0:
            self.close_connection = True

    def _stream_ndjson(self, events) -> None:
        """Relay an event stream as chunked NDJSON, flushed per line.

        HTTP/1.1 chunked framing written by hand: each event becomes one
        ``<hex size>\\r\\n<json line>\\r\\n`` chunk pushed to the socket
        immediately, so the client sees progress while the worker still
        runs.  The stream always ends with the zero-length chunk; a
        client that disconnects mid-run just terminates the generator
        (its ``finally`` stops the worker).
        """
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for event in events:
                chunk = (json.dumps(event) + "\n").encode("utf-8")
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

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
        supplied = self.headers.get("X-Cadjoint-Token", "")
        if secrets.compare_digest(supplied, self.session_token):
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
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON request."})
            return None
        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON request."})
            return None
        return payload

    def _serve_static(self, path: str) -> None:
        """Serve one file from the static root, or the missing-build page."""
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

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")
