"""Running the compile worker as a child process, one request at a time.

Every endpoint that has to execute the editor's Python goes through here:
compile, mesh edges, simulate, mesh inspection, and optimize.  Each call
spawns a fresh ``python -m cadjoint.viewer.worker``, writes one
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

:func:`warm_start` is the one call that is not somebody's request: the
server fires it at startup so the *first* real request meets a warm
compilation cache instead of paying XLA for the whole scene.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
from typing import Any

from cadjoint.viewer._jobs import REGISTRY, attach_process, current_job
from cadjoint.viewer._limits import OVERSIZED_SOURCE_ERROR, exceeds_source_limit
from cadjoint.viewer.worker.protocol import MODULE, RETIRE_FLAG

# The edit round-trip budget. It used to be 20 s, which the gearbox end-cap's
# first compile exceeds against a cold compilation cache; a compile that is
# genuinely long is now visible and cancellable through the job registry, so
# the budget errs on the side of finishing rather than killing real work.
COMPILE_TIMEOUT_SECONDS = 90

# Mesh extraction is a lazy background overlay, not the edit round-trip: dual
# contouring plus feature-edge extraction over the whole scene, and on a cold
# compilation cache the first request pays XLA for every program involved
# (measured on the starter: ~60 s cold, ~17 s warm). It gets its own budget
# so a cold cache cannot make the overlay silently vanish.
MESH_TIMEOUT_SECONDS = 90


#: How each worker mode names itself in a timeout message.
_MODE_NOUNS = {
    "compile": "Compilation",
    "mesh": "Meshing",
    "mesh_inspect": "Mesh inspection",
    "simulate": "Simulation",
    "export": "Export",
    "lint": "Linting",
}


def _run_worker(
    source: str,
    mode: str,
    timeout: float,
    extra: dict[str, Any] | None = None,
    *,
    nice: int = 0,
) -> dict[str, Any]:
    """Run one compile-worker request in a disposable child process.

    Args:
        source: The program to run.
        mode: The worker mode (``compile``, ``mesh``, ...).
        timeout: Seconds before the child is killed.
        extra: Additional request fields.
        nice: Scheduling penalty for the child, 0 for a request the user is
            waiting on.  Speculative work passes a positive value so a real
            request wins the core: a warm-up and a compile otherwise arrive
            as two processes each asking for a full core, and the user waits
            behind work nobody asked for.
    """
    if not isinstance(source, str):
        return {"ok": False, "error": "Source must be a string."}
    if exceeds_source_limit(source):
        return {"ok": False, "error": OVERSIZED_SOURCE_ERROR}

    request = json.dumps({**(extra or {}), "source": source, "mode": mode})
    process = subprocess.Popen(
        [sys.executable, "-m", MODULE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=(lambda: os.nice(nice)) if nice and os.name == "posix" else None,  # noqa: PLW1509
    )
    # Hand the child to the job registry before waiting on it: that is what
    # makes a running request cancellable and its CPU/RSS observable.  Outside
    # a tracked request this is a no-op.
    attach_process(process)
    try:
        stdout, stderr = process.communicate(request, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return {
            "ok": False,
            "error": f"{_MODE_NOUNS.get(mode, 'Compilation')} exceeded the {timeout:g}-second timeout.",
        }

    if process.returncode != 0:
        detail = stderr.strip() or "The compiler process exited unexpectedly."
        return {"ok": False, "error": detail[-8_000:]}
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        detail = stderr.strip() or stdout.strip()
        return {"ok": False, "error": f"Invalid compiler response:\n{detail[-8_000:]}"}
    if not isinstance(result, dict):
        return {"ok": False, "error": "Invalid compiler response."}
    return result


#: Set to ``0`` to make every compile a disposable process again.
WORKER_POOL_ENV = "CADJOINT_WORKER_POOL"

#: The one served worker, and the lock serialising access to its pipes.
_POOL: dict[str, Any] = {"process": None, "served": 0}
_POOL_LOCK = threading.Lock()


def _pool_enabled() -> bool:
    return os.environ.get(WORKER_POOL_ENV, "1") not in {"0", "false", "no", "off"}


def _retire_pooled(kill: bool = True) -> None:
    """Drop the served worker; the next request starts a fresh one."""
    process = _POOL.get("process")
    _POOL["process"] = None
    _POOL["served"] = 0
    if process is None:
        return
    with contextlib.suppress(Exception):
        if kill and process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def _pooled_compile(source: str, timeout: float) -> dict[str, Any] | None:
    """One compile on the served worker, or None to fall back to a fresh one.

    Only ``compile`` is served, and deliberately.  It is the mode a keystroke
    triggers, it has no progress stream to frame around, and it is where the
    fixed costs dominate: a fresh process pays 0.4 s of imports and then
    re-traces what the last one already traced, which on
    ``scenes/motor_shield.py`` is 2.6 s against 0.9 s served.  The heavier
    modes stay disposable, where a runaway solve or a mesher that wedges
    costs one process and nothing else.

    Anything that is not a clean answer retires the worker rather than
    trying to resynchronise its pipes: a half-read response would corrupt
    every request after it, and a fresh process costs 0.4 s.
    """
    if not _pool_enabled():
        return None
    with _POOL_LOCK:
        process = _POOL.get("process")
        if process is not None and process.poll() is not None:
            _retire_pooled(kill=False)
            process = None
        if process is None:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", MODULE, "--serve"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except OSError:
                return None
            _POOL["process"] = process
            _POOL["served"] = 0
        # Still attached to the job, so cancelling a compile still kills a
        # real process — it just retires the pool along with it.
        attach_process(process)
        try:
            process.stdin.write(json.dumps({"source": source, "mode": "compile"}) + "\n")
            process.stdin.flush()
            line = _readline_within(process, timeout)
        except (BrokenPipeError, OSError):
            _retire_pooled()
            return None
        if line is _TIMED_OUT:
            _retire_pooled()
            return {
                "ok": False,
                "error": f"Compilation exceeded the {timeout:g}-second timeout.",
            }
        if not line:
            # EOF: the pipe closed mid-request. Two very different causes.
            _retire_pooled()
            job = current_job()
            if job is not None and job.cancel_requested:
                # Cancelling kills the worker, and the worker is shared, so
                # the closed pipe *is* the cancellation. Falling back here
                # would start a fresh process and redo the work the user
                # just asked us to stop — the one outcome cancelling must
                # not produce.
                return {"ok": False, "error": "Compilation was cancelled."}
            # Otherwise the worker died on its own; a fresh one answers.
            return None
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            _retire_pooled()
            return None
        if not isinstance(result, dict):
            _retire_pooled()
            return None
        _POOL["served"] += 1
        if result.pop(RETIRE_FLAG, False):
            # The worker retired itself — a scene changed a process-global
            # jax config, or it hit its request cap. Its answer is good; the
            # process is not.
            _retire_pooled(kill=False)
        return result


#: Sentinel distinguishing "nothing arrived in time" from "the pipe closed".
#: Conflating them made a dead worker report itself as a timeout, which is a
#: different thing to tell the user and a different thing to do next.
_TIMED_OUT = object()


def _readline_within(process: subprocess.Popen, timeout: float) -> Any:
    """One response line, ``""`` at EOF, or :data:`_TIMED_OUT`."""
    result: list[str] = []

    def read() -> None:
        with contextlib.suppress(Exception):
            result.append(process.stdout.readline())

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(timeout)
    if not result:
        return _TIMED_OUT
    return result[0]


def compile_source(source: str, timeout: float = COMPILE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Compile playground source, on the served worker when there is one."""
    if not isinstance(source, str):
        return {"ok": False, "error": "Source must be a string."}
    if exceeds_source_limit(source):
        return {"ok": False, "error": OVERSIZED_SOURCE_ERROR}
    pooled = _pooled_compile(source, timeout)
    if pooled is not None:
        return pooled
    return _run_worker(source, "compile", timeout)


#: Environment override for :func:`warm_start`: ``0``/``false``/``no``/``off``
#: turns the background warm-up off, anything else turns it on.
WARM_START_ENV = "CADJOINT_WARM_START"

_WARM_STARTED = threading.Event()

#: Programs already warmed in this process, by source hash, so opening the
#: same scene twice costs nothing the second time.
_WARMED: set[str] = set()
_WARM_LOCK = threading.Lock()

#: How far below a request the warm-up runs.  Speculative work should never
#: take a core from work the user is waiting on.
_WARM_NICE = 10


def _warm_start_enabled() -> bool:
    """Whether a server may warm the compilation cache at startup.

    Off under pytest unless asked for explicitly: ``create_server`` is called
    by the live-server tests, and every one of them would otherwise spawn two
    full worker processes it never reads.
    """
    setting = os.environ.get(WARM_START_ENV)
    if setting is not None:
        return setting.strip().lower() not in ("", "0", "false", "no", "off")
    return "PYTEST_CURRENT_TEST" not in os.environ and "pytest" not in sys.modules


def warm_start(source: str | None = None) -> bool:
    """Fill the compilation cache for *source* in the background, once.

    The first ``compile`` of a fresh install pays XLA for every program the
    scene contains.  The cache is on disk and persistent
    (:mod:`cadjoint.cache`), so that cost is paid once by *somebody*;
    issuing the request on a daemon thread at startup means it is not paid
    by the user's first edit, while the browser is still fetching the
    frontend.

    Only ``compile`` is primed.  The mesh used to be primed too, and on every
    scene open, on the theory that the feature-edge overlay should be warm
    by the time it is asked for.  It never was asked for on most opens (the
    overlay is off by default), it grew to several cores and gigabytes on
    the larger scenes, nothing cancelled it on a scene switch or joined it
    when the real request arrived, and it outlived the server.  Measured
    beside a running one, compiles the user was waiting on hit their
    timeout.  The overlay now pays its own cold cache once, on demand.

    Runs at most once per process and never blocks the caller: it returns as
    soon as the thread is started, and returns ``False`` when the warm-up is
    disabled or has already run.  It goes through the ordinary
    disposable-worker path, so a scene that fails to compile costs nothing
    but a logged-nowhere failure.  See :func:`_warm_start_enabled` for the
    ``CADJOINT_WARM_START`` override and the pytest default.

    Args:
        source: The program to warm on. Defaults to the playground's example
            scene, which is what the editor opens with.

    Returns:
        Whether a warm-up thread was started.
    """
    if not _warm_start_enabled():
        return False
    if source is None:
        from cadjoint.viewer._example_scene import EXAMPLE_SOURCE

        source = EXAMPLE_SOURCE
    key = hashlib.sha256(source.encode()).hexdigest()
    with _WARM_LOCK:
        if key in _WARMED:
            return False
        _WARMED.add(key)
    _WARM_STARTED.set()

    def prime() -> None:
        # Under the mesh budget, not the compile one: the point of the
        # warm-up is the cold path, where a compile can outgrow the edit
        # round-trip budget it is held to in a request.
        try:
            # Registered as a `warmup` job so the process monitor can say
            # why a worker is burning CPU right after launch, and niced so
            # a request the user is waiting on wins the core.
            with REGISTRY.track("warmup", source=source, fields={"mode": "compile"}) as job:
                REGISTRY.finish(
                    job,
                    _run_worker(source, "compile", MESH_TIMEOUT_SECONDS, nice=_WARM_NICE),
                )
        except Exception:  # noqa: BLE001 - a cold cache is the only cost of failing
            return

    threading.Thread(target=prime, name="cadjoint-compile-warmup", daemon=True).start()
    return True


def mesh_source(source: str, timeout: float = MESH_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Extract only the dual-contour mesh edges, in a disposable child process.

    Mesh extraction dominates a full compile, so the viewer requests it lazily
    through this path only while a mesh overlay is actually turned on.
    """
    return _run_worker(source, "mesh", timeout)


# FEM solves cover meshing plus the assembled solve, which can far exceed the
# ordinary compile budget.
# A novel design pays XLA for its FEM assembly before it solves — measured at
# 31-77 s for a new node count — and a loaded machine stretches that; the job
# is visible and cancellable, so the budget errs on the side of finishing.
SIMULATE_TIMEOUT_SECONDS = 600
SIMULATE_KINDS = ("study",)
# Mesh inspection only builds the hex mesh (no solve), but big grids still
# outgrow the compile budget.
# Mesh inspection meshes and then measures quality; a tet10 mesh of a large
# part on a cold cache and a busy machine outgrew 60 s. Cancellable, so
# generous.
MESH_INSPECT_TIMEOUT_SECONDS = 300


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
# An optimisation streams its progress and can be cancelled from the process
# window at any step, so its budget is a safety net against a hung worker,
# not a target: two frozen-chain freezes cost about a minute each before the
# first step, and a large part multiplies that.
OPTIMIZE_TIMEOUT_SECONDS = 1800
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
        [sys.executable, "-m", MODULE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    attach_process(process)
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
