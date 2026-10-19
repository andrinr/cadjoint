"""Running an export in a worker and handing the file back (``POST /api/export``).

The server's half of :mod:`cadjoint.viewer.worker.export`: validate the
request against :class:`~cadjoint.viewer.schema.requests.ExportRequest`, run
the worker with a temporary path to write to, and hand the bytes back with
their filename and content type.  The route in
:mod:`cadjoint.viewer.playground` registers the run as an ``export`` job, so
the Processes window sees it and can cancel it, and sends the bytes as an
attachment.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from cadjoint.enums import ExportFormat
from cadjoint.viewer._worker_client import SIMULATE_TIMEOUT_SECONDS, _run_worker
from cadjoint.viewer.worker.export import (
    EXPORT_CONTENT_TYPES,
    export_filename,
    validate_export_request,
)

__all__ = ["EXPORT_TIMEOUT_SECONDS", "export_source"]

# A geometry export is one lattice sweep plus, for STEP, the graph
# extraction the mesh overlay pays for every scene (~17 s warm on the
# starter at the overlay's resolution, and the request may ask for finer).
# A VTK export is a full solve, so it takes the simulate budget instead.
# Either is visible and cancellable as a job.
EXPORT_TIMEOUT_SECONDS = 300


def export_source(request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    """Validate an export request and write the file in a disposable child process.

    The worker writes to a temporary path this process owns; the bytes are
    read back and the directory removed before returning, so nothing is
    left on disk and nothing large crosses the worker's JSON pipe.

    Args:
        request: The raw request body.
        timeout: The worker budget; by default :data:`EXPORT_TIMEOUT_SECONDS`,
            or the simulate budget for ``vtk`` (which solves the study).

    Returns:
        On success ``{"ok": True, "format", "name", "filename",
        "content_type", "size", "data": bytes, "report", "output"}``; on
        failure the worker's ``{"ok": False, "error"}`` (with
        ``error_kind="fem_unavailable"`` when a VTK export has no solver).
    """
    error, parsed = validate_export_request(request)
    if error is not None or parsed is None:
        return error or {"ok": False, "error": "Invalid export request."}
    if timeout is None:
        timeout = (
            SIMULATE_TIMEOUT_SECONDS
            if parsed.format is ExportFormat.VTK
            else EXPORT_TIMEOUT_SECONDS
        )
    filename = export_filename(parsed.name, parsed.format)
    with tempfile.TemporaryDirectory(prefix="cadjoint-export-") as folder:
        path = Path(folder) / filename
        extra = parsed.model_dump(mode="json", exclude={"source"})
        extra["path"] = str(path)
        result = _run_worker(parsed.source, "export", timeout, extra=extra)
        if not result.get("ok"):
            return result
        if not path.is_file():
            return {"ok": False, "error": "The export worker finished without writing a file."}
        data = path.read_bytes()
    return {
        "ok": True,
        "format": parsed.format.value,
        "name": parsed.name,
        "filename": filename,
        "content_type": EXPORT_CONTENT_TYPES[parsed.format],
        "size": len(data),
        "data": data,
        "report": result.get("report") or {},
        "output": result.get("output", ""),
    }
