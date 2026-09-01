"""Running the compile worker as a child process, one request at a time.

Every endpoint that has to execute the editor's Python goes through here:
compile, mesh edges, simulate, mesh inspection, and optimize.  Each call
spawns a fresh ``python -m cadjoint.viewer._compile_worker``, writes one
JSON request to its stdin, and reads its JSON response back — a disposable
process per request, bounded by a per-mode timeout, so a runaway program
cannot outlive its request or leak state into the next one.

Optimize is the exception to "one response": the worker streams NDJSON
progress lines while it descends, so it is tailed line by line
(:func:`optimize_source_events`) and also offered buffered
(:func:`optimize_source`).

Request validation here is only what has to happen before a worker is
started — the kind/name/step shapes.  ``/patch`` never reaches this module:
it is pure text surgery (see :mod:`cadjoint.viewer._patch_requests`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from cadjoint.viewer._limits import OVERSIZED_SOURCE_ERROR, exceeds_source_limit

COMPILE_TIMEOUT_SECONDS = 20


def _run_worker(
    source: str, mode: str, timeout: float, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run one compile-worker request in a disposable child process."""
    if not isinstance(source, str):
        return {"ok": False, "error": "Source must be a string."}
    if exceeds_source_limit(source):
        return {"ok": False, "error": OVERSIZED_SOURCE_ERROR}

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "cadjoint.viewer._compile_worker"],
            input=json.dumps({**(extra or {}), "source": source, "mode": mode}),
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


def compile_source(source: str, timeout: float = COMPILE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Compile playground source in a disposable child process."""
    return _run_worker(source, "compile", timeout)


def mesh_source(source: str, timeout: float = COMPILE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Extract only the dual-contour mesh edges, in a disposable child process.

    Mesh extraction dominates a full compile, so the viewer requests it lazily
    through this path only while a mesh overlay is actually turned on.
    """
    return _run_worker(source, "mesh", timeout)


# FEM solves cover meshing plus the assembled solve, which can far exceed the
# ordinary compile budget.
SIMULATE_TIMEOUT_SECONDS = 180
SIMULATE_KINDS = ("study",)
# Mesh inspection only builds the hex mesh (no solve), but big grids still
# outgrow the compile budget.
MESH_INSPECT_TIMEOUT_SECONDS = 60


def simulate_source(
    request: dict[str, Any], timeout: float = SIMULATE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Validate a simulation request and run it in a disposable child process.

    ``kind="study"`` (the only kind) runs a study the scene program itself
    declares (a first-class :mod:`cadjoint.fem.study` object), picked by
    ``name`` — mesh, material, and boundary conditions all come from the
    declaration.  With ``cached=True`` the worker serves the study's
    ``last_result`` without re-solving when one exists; every request runs a
    fresh worker process, so that cache only ever holds a result the scene
    program computed itself (a module-level ``solve()``) — it is per worker
    process, never shared across requests.  The worker reports a missing
    jax-fem extra as ``error_kind="fem_unavailable"``, which the HTTP layer
    maps to 501.
    """
    kind = request.get("kind", "study")
    if kind not in SIMULATE_KINDS:
        return {
            "ok": False,
            "error": (
                "Simulation `kind` must be `study`: declare a ThermalStudy/ElasticStudy "
                "in the program and run it by name."
            ),
        }
    name = request.get("name")
    if not isinstance(name, str) or not name.strip():
        return {
            "ok": False,
            "error": "A study simulation needs `name`: the declared study to run.",
        }
    cached = request.get("cached", False)
    if not isinstance(cached, bool):
        return {"ok": False, "error": "Simulation `cached` must be a boolean."}
    return _run_worker(
        request.get("source"),
        "simulate",
        timeout,
        extra={"kind": kind, "name": name, "cached": cached},
    )


def mesh_inspect_source(
    request: dict[str, Any], timeout: float = MESH_INSPECT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Build one declared SimMesh in a disposable child process and report it.

    ``name`` picks a declared :class:`cadjoint.fem.SimMesh` by name — or a
    declared study, whose mesh (explicit or implicit) is built instead.  With
    no ``name``, a single declared mesh (or, failing that, a single study) is
    used.  The response carries the JSON inspection summary (``info``), a
    renderable boundary surface (``mesh``), and the per-vertex
    scaled-jacobian quality field (``quality_scalars``) so the viewer can
    show a mesh-quality heatmap before anything is solved.
    """
    name = request.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        return {"ok": False, "error": "`name` must be a non-empty string when given."}
    extra = {"name": name} if name is not None else {}
    return _run_worker(request.get("source"), "mesh_inspect", timeout, extra=extra)


# Optimizations run the differentiable objective once per step, which can far
# exceed the compile budget; the step cap keeps one request's work bounded.
# Study-backed optimizations additionally run a full FEM solve (plus its
# adjoint) per step, so the timeout covers a panel-sized study run and the
# worker enforces a tighter measured per-run step cap for them
# (``STUDY_OPTIMIZE_STEP_LIMIT`` in the compile worker).
OPTIMIZE_TIMEOUT_SECONDS = 300
OPTIMIZE_MAX_STEPS = 200


def _validate_optimize_request(request: dict[str, Any]) -> tuple[dict | None, dict[str, Any]]:
    """``(error, extra)`` — request-shape validation shared by both entry points."""
    name = request.get("name")
    if not isinstance(name, str) or not name.strip():
        return {
            "ok": False,
            "error": "An optimization run needs `name`: the declared Optimization to run.",
        }, {}
    extra: dict[str, Any] = {"name": name}
    steps = request.get("steps")
    if steps is not None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            return {"ok": False, "error": "Optimization `steps` must be a positive integer."}, {}
        if steps > OPTIMIZE_MAX_STEPS:
            return {
                "ok": False,
                "error": f"Optimization `steps` is capped at {OPTIMIZE_MAX_STEPS} per request.",
            }, {}
        extra["steps"] = steps
    source = request.get("source")
    if not isinstance(source, str):
        return {"ok": False, "error": "Source must be a string."}, {}
    if exceeds_source_limit(source):
        return {"ok": False, "error": OVERSIZED_SOURCE_ERROR}, {}
    return None, extra


def _stream_optimize_worker(source: str, extra: dict[str, Any], timeout: float):
    """Run the optimize worker and yield its NDJSON events as they arrive.

    The worker prints one flushed ``{"event": "progress", ...}`` line per
    optimizer step and its ordinary result object as the final line; this
    generator tails the pipe, relays the progress events immediately, and
    closes with one ``{"event": "done", ...result}`` event.  A watchdog
    kills the worker at *timeout* and reports it as a clear error event.
    """
    import threading

    process = subprocess.Popen(
        [sys.executable, "-m", "cadjoint.viewer._compile_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_text: list[str] = []
    drain = threading.Thread(target=lambda: stderr_text.append(process.stderr.read()), daemon=True)
    drain.start()
    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        process.kill()

    watchdog = threading.Timer(timeout, _kill)
    watchdog.start()
    final: dict[str, Any] | None = None
    try:
        process.stdin.write(json.dumps({**extra, "source": source, "mode": "optimize"}))
        process.stdin.close()
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # stray non-JSON output must not corrupt the stream
            if not isinstance(event, dict):
                continue
            if event.get("event") == "progress":
                yield event
            else:
                final = event
        process.wait()
    finally:
        watchdog.cancel()
        drain.join(timeout=2)
        if process.poll() is None:  # client went away mid-run: stop the work
            process.kill()
    if timed_out.is_set():
        yield {
            "event": "done",
            "ok": False,
            "error": f"Optimization exceeded the {timeout:g}-second timeout.",
        }
        return
    if final is None or process.returncode != 0:
        detail = (stderr_text[0].strip() if stderr_text and stderr_text[0] else "") or (
            "The optimizer process exited unexpectedly."
        )
        yield {"event": "done", "ok": False, "error": detail[-8_000:]}
        return
    yield {"event": "done", **final}


def optimize_source_events(request: dict[str, Any], timeout: float = OPTIMIZE_TIMEOUT_SECONDS):
    """Yield the NDJSON event stream for one optimize request.

    ``/api/optimize`` responses are chunked NDJSON: one
    ``{"event": "progress", "step", "steps", "objective", "grad_norm",
    "elapsed"}`` line per optimizer step as it completes (``step`` counts
    finished evaluations, 1-based), then exactly one final
    ``{"event": "done", ...}`` line carrying the entire ordinary response
    object — ``ok``/``source``/``history``/``trajectory``/``parameters``
    (plus ``simulate`` for study-backed runs), or ``ok: false`` with
    ``error`` (and ``error_kind`` such as ``fem_unavailable``).  Request
    validation failures arrive the same way, as an immediate ``done``.
    """
    error, extra = _validate_optimize_request(request)
    if error is not None:
        yield {"event": "done", **error}
        return
    yield from _stream_optimize_worker(request["source"], extra, timeout)


def optimize_source(
    request: dict[str, Any], timeout: float = OPTIMIZE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Validate an optimize request and run it in a disposable child process.

    Runs an :class:`cadjoint.optimize.Optimization` the scene program itself
    declares, picked by ``name`` — objective, target object, and optimizer
    settings all come from the declaration; the optional ``steps`` overrides
    the declared step count (both are capped at ``OPTIMIZE_MAX_STEPS``;
    study-backed runs are further capped by the worker's measured
    ``STUDY_OPTIMIZE_STEP_LIMIT``).  The worker writes the optimized
    free-parameter values back into the program text through the patch
    machinery, so the response's ``source`` is the patched program — the
    client adopts it and recompiles, exactly like a ``/patch`` response.
    A study-backed optimization's response additionally carries a
    ``simulate`` block (``field``/``mesh``/``result``/``mesh_info``): the
    optimized design solved on a fresh mesh, in the exact shapes
    ``/api/simulate`` responses use.  A missing jax-fem extra surfaces as
    ``error_kind="fem_unavailable"``.

    This is the buffered form of :func:`optimize_source_events` (the
    ``/api/optimize`` endpoint streams the events instead): progress events
    are consumed and the final ``done`` event is returned as a plain
    response dict.
    """
    final: dict[str, Any] = {}
    for event in optimize_source_events(request, timeout):
        if event.get("event") == "done":
            final = {key: value for key, value in event.items() if key != "event"}
    return final
