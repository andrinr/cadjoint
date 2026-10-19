"""The protocol between the server and the worker process it spawns.

The server writes one JSON request with a ``mode`` to the worker's stdin and
reads its JSON reply back (one NDJSON line per request on a served worker;
progress lines before the reply for ``optimize``).  This module models both
halves, one request and one reply per mode, and :func:`checked_reply` is the
boundary check the server runs on everything a worker sends before any of
it reaches a route.

Why check what the worker already built: the worker is a separate process
running arbitrary user code, and until this module only ``compile`` and
``export`` replies were checked at all.  A reply that does not match its
model is a bug on one side of the contract, and it is reported as one — an
``error_kind: "contract"`` failure naming the field — instead of being
forwarded to a browser that was promised something else.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from cadjoint.viewer.schema.api import (
    MeshInspectResult,
    MeshResult,
    OptimizeResult,
    SimulateResult,
)
from cadjoint.viewer.schema.payloads import CompilePayload, Strict, WorkerFailure
from cadjoint.viewer.schema.requests import ExportRequest

WorkerMode = Literal["compile", "mesh", "simulate", "mesh_inspect", "optimize", "export"]


# ── Requests ────────────────────────────────────────────────────────────────


class CompileWorkerRequest(Strict):
    mode: Literal["compile"]
    source: str


class MeshWorkerRequest(Strict):
    mode: Literal["mesh"]
    source: str


class SimulateWorkerRequest(Strict):
    mode: Literal["simulate"]
    source: str
    kind: Literal["study"]
    name: str
    cached: bool


class MeshInspectWorkerRequest(Strict):
    mode: Literal["mesh_inspect"]
    source: str
    name: str | None = None


class OptimizeWorkerRequest(Strict):
    mode: Literal["optimize"]
    source: str
    name: str
    steps: int | None = None


class ExportWorkerRequest(ExportRequest):
    """An :class:`ExportRequest` plus where the server wants the file written."""

    mode: Literal["export"]
    path: str


# ── Replies ─────────────────────────────────────────────────────────────────


class MeshWorkerReply(MeshResult):
    """``mesh`` answers with the route's result plus the overlay's refresh seed.

    ``refresh`` is server state, not payload: the server takes it out of the
    reply and keeps it to re-solve the overlay at a new design
    (``/api/mesh_refresh``).  Its shape is the zero-set protocol's own
    (:mod:`cadjoint.zeroset`) and is checked there, so it stays an open map
    here.
    """

    refresh: dict[str, Any] | None = None


class OptimizeWorkerProgress(Strict):
    """One progress line ``optimize`` prints per finished step."""

    event: Literal["progress"]
    step: int
    steps: int
    objective: float
    grad_norm: float
    elapsed: float


class ExportWorkerReply(Strict):
    """``export`` leaves the bytes on disk and answers with the writer's report."""

    ok: Literal[True]
    report: dict[str, Any]
    output: str


WORKER_REQUESTS: dict[str, type[BaseModel]] = {
    "compile": CompileWorkerRequest,
    "mesh": MeshWorkerRequest,
    "simulate": SimulateWorkerRequest,
    "mesh_inspect": MeshInspectWorkerRequest,
    "optimize": OptimizeWorkerRequest,
    "export": ExportWorkerRequest,
}
"""The request each mode takes."""

WORKER_REPLIES: dict[str, type[BaseModel]] = {
    "compile": CompilePayload,
    "mesh": MeshWorkerReply,
    "simulate": SimulateResult,
    "mesh_inspect": MeshInspectResult,
    "optimize": OptimizeResult,
    "export": ExportWorkerReply,
}
"""The successful reply each mode sends; every mode fails as a :class:`WorkerFailure`."""


def _contract_failure(mode: str, error: ValidationError, what: str = "reply") -> dict[str, Any]:
    first = error.errors()[0]
    where = ".".join(str(part) for part in first.get("loc", ())) or f"({what})"
    return {
        "ok": False,
        "error_kind": "contract",
        "error": (
            f"The worker {what} for {mode!r} does not match the viewer contract "
            f"({error.error_count()} problem(s); first at {where}: {first.get('msg')}). "
            "This is a bug in cadjoint, not in your program."
        ),
    }


def checked_reply(mode: str, reply: dict[str, Any]) -> dict[str, Any]:
    """Pass a worker's reply on if it matches its mode's model, else say why not.

    Args:
        mode: The worker mode the reply answers.
        reply: The decoded reply, with any transport-only flag already removed.

    Returns:
        ``reply`` itself, unchanged, when it validates; otherwise a
        :class:`WorkerFailure` with ``error_kind: "contract"``.
    """
    model = WORKER_REPLIES.get(mode)
    if model is None:
        return reply
    try:
        if reply.get("ok") is True:
            model.model_validate(reply)
        else:
            WorkerFailure.model_validate(reply)
    except ValidationError as error:
        return _contract_failure(mode, error)
    return reply


def checked_progress(event: dict[str, Any]) -> dict[str, Any] | None:
    """An ``optimize`` progress line if it is one, else None (and it is dropped)."""
    try:
        OptimizeWorkerProgress.model_validate(event)
    except ValidationError:
        return None
    return event


def checked_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """None when a request the server built matches its mode's model, else why not.

    The server validates what a client sent before it builds a worker
    request, so a refusal here is the server's own bug, reported as a
    ``contract`` failure instead of a worker started with a request it will
    misread.
    """
    mode = str(request.get("mode"))
    model = WORKER_REQUESTS.get(mode)
    if model is None:
        return {"ok": False, "error_kind": "contract", "error": f"No worker mode {mode!r}."}
    try:
        model.model_validate(request)
    except ValidationError as error:
        return _contract_failure(mode, error, "request")
    return None
