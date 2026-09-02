"""The server's registry of running and finished work.

Every request that costs real time — a compile, a mesh extraction, a mesh
inspection, a simulation, an optimization run, a lint, the startup warm-up —
is registered here as a :class:`Job` the moment it starts and stays here
after it ends.  That gives the playground three things a request/response
API cannot:

- **Persistence across the UI.**  A result lives in the registry, not only in
  the response, so a panel that was closed (or a mode that was switched away
  from) can fetch the same payload back by job id instead of re-running a
  9-second solve.
- **A process monitor.**  While a job's worker subprocess runs, its CPU and
  RSS are sampled every :data:`SAMPLE_INTERVAL` seconds, so the UI can show
  what is currently burning the machine and for how long.
- **Cancellation.**  The registry holds the worker's ``Popen``, so a run can
  actually be killed rather than merely abandoned.

The registry never starts work itself: endpoints wrap their existing call in
:meth:`JobRegistry.track`, which binds the job to the calling thread so that
:func:`cadjoint.viewer._worker_client._run_worker` can attach the subprocess
it spawns (:func:`attach_process`) without any endpoint passing it down.

Sampling prefers ``psutil`` (the ``viewer`` extra).  Without it the registry
degrades to :func:`resource.getrusage` totals over all children and says so:
every payload carries ``sampling`` (``"psutil"``, ``"rusage"`` or ``"none"``)
so the UI can label what it is showing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:  # pragma: no cover - exercised by whichever environment is installed
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

#: The kinds of work the registry knows about.
JOB_KINDS = (
    "compile",
    "mesh",
    "mesh_inspect",
    "simulate",
    "optimize",
    "export",
    "lint",
    "warmup",
)

#: Kinds whose full response payload is worth keeping for re-fetching.  Lint
#: is excluded: it is keystroke-cadence work that is cheaper to redo than to
#: remember, and its diagnostics are meaningless once the text has moved on.
RESULT_KINDS = ("compile", "mesh", "mesh_inspect", "simulate", "optimize")

#: How many jobs of any kind are kept.  Older *finished* jobs are dropped
#: first; running jobs are never evicted.
MAX_JOBS = 50
#: How many lint jobs are kept, separately, so that a typing session cannot
#: flush the history of the expensive work the monitor exists to show.
MAX_LINT_JOBS = 10
#: Total budget for stored result payloads, measured as serialized JSON.
MAX_RESULT_BYTES = 64 * 1024 * 1024
#: Samples kept per job (five minutes at :data:`SAMPLE_INTERVAL`).
MAX_SAMPLES = 600
#: Progress events kept per optimize job, for replaying a run's descent.
MAX_PROGRESS_EVENTS = 256
#: Seconds between resource samples of a running worker.
SAMPLE_INTERVAL = 0.5
#: How many samples apart the worker's child processes are re-enumerated.
#: Reading one process's CPU and RSS costs 8 microseconds; asking the OS for
#: its children costs 12.8 ms (measured, macOS arm64), because it walks the
#: whole process table.  Children are long-lived when they exist at all (a
#: ccx solve), so they are looked for every two seconds rather than twice a
#: second, which keeps the sampler under 0.4% of one core.
CHILD_RESCAN_EVERY = 4

_MACOS = sys.platform == "darwin"


def source_hash(source: Any) -> str | None:
    """The sha256 of a program's text, or None when there is no text.

    The frontend matches a stored result to the document it came from with
    this, so a Results panel re-opened after an edit can tell a stale result
    from a current one.

    Args:
        source: The program text, or anything else (which hashes to None).

    Returns:
        A 64-character hex digest, or None when *source* is not a string.
    """
    if not isinstance(source, str):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _rusage_children() -> tuple[float, int]:
    """``(cpu_seconds, max_rss_bytes)`` for every child this process reaped."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if _MACOS else 1024  # ru_maxrss is bytes on macOS, kilobytes on Linux
    return usage.ru_utime + usage.ru_stime, int(usage.ru_maxrss) * scale


class Job:
    """One unit of registered work, alive or finished.

    Attributes are read by the HTTP layer through :meth:`summary` and
    :meth:`detail`; everything that mutates goes through the registry's lock.
    """

    def __init__(
        self,
        job_id: str,
        kind: str,
        *,
        source: Any = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        self.id = job_id
        self.kind = kind
        self.status = "queued"
        self.fields = dict(fields or {})
        self.source_hash = source_hash(source)
        self.source_bytes = len(source.encode("utf-8")) if isinstance(source, str) else 0
        self.submitted_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.pid: int | None = None
        self.ok: bool | None = None
        self.error: str | None = None
        self.progress: dict[str, Any] | None = None
        self.progress_events: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.peak_cpu_percent = 0.0
        self.peak_rss_bytes = 0
        self.cpu_seconds = 0.0
        self.sampling = "none"
        self.result_json: str | None = None
        self.result_bytes = 0
        self.cancel_requested = False
        self._process: subprocess.Popen | None = None
        self._monotonic_start = time.monotonic()
        self._rusage_start = _rusage_children()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Mark the job running and reset its clock."""
        self.status = "running"
        self.started_at = time.time()
        self._monotonic_start = time.monotonic()
        self._rusage_start = _rusage_children()

    def attach(self, process: subprocess.Popen) -> None:
        """Adopt the worker subprocess and begin sampling it."""
        with self._lock:
            self._process = process
            self.pid = process.pid
            cancelled = self.cancel_requested
        if cancelled:
            self._kill(process)
            return
        if psutil is not None:
            self.sampling = "psutil"
            threading.Thread(
                target=self._sample_loop,
                args=(process,),
                name=f"cadjoint-job-{self.id}",
                daemon=True,
            ).start()
        else:
            self.sampling = "rusage"

    def cancel(self) -> bool:
        """Kill the worker, if any, and remember that this was deliberate.

        Returns:
            Whether the job was still cancellable (queued or running).
        """
        if self.status not in ("queued", "running"):
            return False
        with self._lock:
            self.cancel_requested = True
            process = self._process
        if process is not None:
            self._kill(process)
        return True

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        """Kill a worker and anything it spawned (a ccx solve, say)."""
        if psutil is not None:
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    child.kill()
            except Exception:  # noqa: BLE001 - the tree may already be gone
                pass
        with contextlib.suppress(OSError):
            process.kill()

    def complete(self, result: dict[str, Any] | None) -> dict[str, Any]:
        """Record the endpoint's result and finish the job.

        A cancelled job's result is replaced: whatever the killed worker's
        broken pipe produced is noise, and the caller's response has to say
        ``cancelled`` so the UI can tell a cancellation from a crash.

        Args:
            result: The endpoint's response payload, or None.

        Returns:
            The payload to send to the caller, carrying ``job_id``.
        """
        payload: dict[str, Any]
        if self.cancel_requested:
            payload = {"ok": False, "error": "cancelled", "error_kind": "cancelled"}
            self.status = "cancelled"
            self.ok = False
            self.error = "cancelled"
        else:
            payload = dict(result or {})
            self.ok = bool(payload.get("ok"))
            self.status = "done" if self.ok else "failed"
            error = payload.get("error")
            self.error = error[:2000] if isinstance(error, str) else None
        payload["job_id"] = self.id
        self._settle()
        return payload

    def fail(self, message: str) -> None:
        """Finish the job as failed with *message* (an unexpected exception)."""
        if self.finished_at is not None:
            return
        self.status = "cancelled" if self.cancel_requested else "failed"
        self.ok = False
        self.error = message[:2000]
        self._settle()

    def _settle(self) -> None:
        """Stop sampling, freeze the clock, and take the rusage fallback."""
        if self.finished_at is None:
            self.finished_at = time.time()
        self._stop.set()
        if self.sampling == "rusage":
            # No psutil: all this process can say is what its children cost in
            # total, so the numbers are the delta over the job's lifetime and
            # the high-water mark across every child — approximate, and
            # labelled as such by the `sampling` field.  A job that never
            # spawned a worker (`sampling == "none"`) claims nothing at all.
            cpu_seconds, max_rss = _rusage_children()
            self.cpu_seconds = max(0.0, cpu_seconds - self._rusage_start[0])
            self.peak_rss_bytes = max(self.peak_rss_bytes, max_rss)

    def record_progress(self, event: dict[str, Any]) -> None:
        """Mirror one streamed optimize progress event onto the job."""
        self.progress = {
            key: event[key]
            for key in ("step", "steps", "objective", "grad_norm", "elapsed")
            if key in event
        }
        if len(self.progress_events) < MAX_PROGRESS_EVENTS:
            self.progress_events.append(dict(self.progress))

    def store_result(self, payload: dict[str, Any]) -> int:
        """Serialize *payload* into the result store; return its byte size."""
        if self.kind not in RESULT_KINDS:
            return 0
        try:
            text = json.dumps(payload)
        except (TypeError, ValueError):
            return 0
        self.result_json = text
        self.result_bytes = len(text.encode("utf-8"))
        return self.result_bytes

    def drop_result(self) -> int:
        """Forget the stored payload (keeping the summary); return bytes freed."""
        freed = self.result_bytes
        self.result_json = None
        self.result_bytes = 0
        return freed

    # ── sampling ───────────────────────────────────────────────────────────

    def _sample_loop(self, process: subprocess.Popen) -> None:
        """Sample the worker tree's CPU and RSS until it exits."""
        tracked: dict[int, Any] = {}
        try:
            tracked[process.pid] = psutil.Process(process.pid)
        except Exception:  # noqa: BLE001 - the worker may already be gone
            return
        tracked[process.pid].cpu_percent(None)  # prime the interval
        index = 0
        while not self._stop.wait(SAMPLE_INTERVAL):
            if process.poll() is not None:
                break
            cpu = 0.0
            rss = 0
            index += 1
            root = tracked.get(process.pid)
            children: list[Any] = []
            if root is not None and index % CHILD_RESCAN_EVERY == 1:
                try:
                    children = root.children(recursive=True)
                except Exception:  # noqa: BLE001
                    children = []
            for child in children:
                if child.pid not in tracked:
                    tracked[child.pid] = child
                    with contextlib.suppress(Exception):
                        child.cpu_percent(None)
            cpu_seconds = 0.0
            for pid, proc in list(tracked.items()):
                try:
                    cpu += proc.cpu_percent(None)
                    rss += proc.memory_info().rss
                    times = proc.cpu_times()
                    cpu_seconds += times.user + times.system
                except Exception:  # noqa: BLE001 - process exited between calls
                    tracked.pop(pid, None)
            # CPU seconds are read live rather than after the exit: once the
            # worker is reaped its counters are gone, and the rusage total
            # below cannot tell one concurrent child from another.
            self.cpu_seconds = max(self.cpu_seconds, cpu_seconds)
            self._add_sample(cpu, rss)
        if self.cpu_seconds <= 0.0:
            self.cpu_seconds = max(0.0, _rusage_children()[0] - self._rusage_start[0])

    def _add_sample(self, cpu_percent: float, rss_bytes: int) -> None:
        """Append one sample, bounded, and update the peaks."""
        sample = {
            "t": round(time.monotonic() - self._monotonic_start, 3),
            "cpu_percent": round(cpu_percent, 1),
            "rss_bytes": int(rss_bytes),
        }
        samples = self.samples
        if len(samples) >= MAX_SAMPLES:
            del samples[0 : len(samples) - MAX_SAMPLES + 1]
        samples.append(sample)
        self.peak_cpu_percent = max(self.peak_cpu_percent, sample["cpu_percent"])
        self.peak_rss_bytes = max(self.peak_rss_bytes, sample["rss_bytes"])

    # ── reporting ──────────────────────────────────────────────────────────

    @property
    def elapsed_s(self) -> float:
        """Seconds the job has been (or was) running."""
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def summary(self) -> dict[str, Any]:
        """The cheap per-job payload ``GET /api/jobs`` lists (no result, no samples)."""
        latest = self.samples[-1] if self.samples else None
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "fields": self.fields,
            "source_hash": self.source_hash,
            "source_bytes": self.source_bytes,
            "submitted_at": round(self.submitted_at, 3),
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "finished_at": round(self.finished_at, 3) if self.finished_at else None,
            "elapsed_s": round(self.elapsed_s, 3),
            "pid": self.pid,
            "ok": self.ok,
            "error": self.error,
            "progress": self.progress,
            "cpu_percent": latest["cpu_percent"] if latest else 0.0,
            "rss_bytes": latest["rss_bytes"] if latest else 0,
            "peak_cpu_percent": round(self.peak_cpu_percent, 1),
            "peak_rss_bytes": self.peak_rss_bytes,
            "cpu_seconds": round(self.cpu_seconds, 3),
            "sampling": self.sampling,
            "samples": len(self.samples),
            "result_available": self.result_json is not None,
            "result_bytes": self.result_bytes,
        }

    def detail(self) -> dict[str, Any]:
        """The summary plus the resource samples and any optimize progress."""
        return {
            **self.summary(),
            "sample_series": list(self.samples),
            "progress_events": list(self.progress_events),
        }


_ACTIVE = threading.local()


def current_job() -> Job | None:
    """The job bound to this thread, if any."""
    return getattr(_ACTIVE, "job", None)


def attach_process(process: subprocess.Popen) -> None:
    """Attach a freshly spawned worker to this thread's job.

    Called by :mod:`cadjoint.viewer._worker_client` for every worker it
    starts.  A no-op when no job is being tracked, which is what makes the
    worker client usable (and unit-testable) outside the server.

    Args:
        process: The worker subprocess, already started.
    """
    job = current_job()
    if job is not None:
        job.attach(process)


class JobRegistry:
    """The bounded, in-memory history of this server's work."""

    def __init__(
        self,
        max_jobs: int = MAX_JOBS,
        max_lint_jobs: int = MAX_LINT_JOBS,
        max_result_bytes: int = MAX_RESULT_BYTES,
    ) -> None:
        self.max_jobs = max_jobs
        self.max_lint_jobs = max_lint_jobs
        self.max_result_bytes = max_result_bytes
        self.started_at = time.time()
        self.evicted_jobs = 0
        self.evicted_results = 0
        self._jobs: list[Job] = []
        self._by_id: dict[str, Job] = {}
        self._counter = 0
        self._result_bytes = 0
        self._lock = threading.RLock()
        self._self_process = None
        if psutil is not None:
            try:
                self._self_process = psutil.Process(os.getpid())
                self._self_process.cpu_percent(None)
            except Exception:  # noqa: BLE001
                self._self_process = None

    # ── registration ───────────────────────────────────────────────────────

    def submit(self, kind: str, *, source: Any = None, fields: dict[str, Any] | None = None) -> Job:
        """Register a new job in ``queued`` state and evict what no longer fits."""
        with self._lock:
            self._counter += 1
            job = Job(f"job-{self._counter:06d}", kind, source=source, fields=fields)
            self._jobs.append(job)
            self._by_id[job.id] = job
            self._evict_jobs()
            return job

    @contextmanager
    def track(
        self, kind: str, *, source: Any = None, fields: dict[str, Any] | None = None
    ) -> Iterator[Job]:
        """Run a block as a registered job, bound to the calling thread.

        Binding is what lets ``_run_worker`` attach the subprocess it spawns
        without any endpoint threading it through; unbinding happens even if
        the block raises, and an escaping exception finishes the job as
        ``failed``.

        Args:
            kind: One of :data:`JOB_KINDS`.
            source: The program text the request carries, for the source hash.
            fields: The request's identifying fields (study/optimization name).

        Yields:
            The registered :class:`Job`, already running.
        """
        job = self.submit(kind, source=source, fields=fields)
        previous = current_job()
        _ACTIVE.job = job
        job.start()
        try:
            yield job
        except BaseException as error:  # noqa: BLE001 - re-raised below
            job.fail(f"{type(error).__name__}: {error}")
            raise
        finally:
            _ACTIVE.job = previous
            if job.finished_at is None:
                job.complete({"ok": False, "error": "The request ended without a result."})

    def finish(self, job: Job, result: dict[str, Any] | None) -> dict[str, Any]:
        """Complete *job*, store its payload, and return the caller's response."""
        payload = job.complete(result)
        with self._lock:
            self._result_bytes += job.store_result(payload)
            self._evict_results(keep=job)
        return payload

    # ── eviction ───────────────────────────────────────────────────────────

    def _evict_jobs(self) -> None:
        """Drop the oldest finished jobs past the count budgets."""
        while len(self._jobs) > self.max_jobs:
            if not self._drop_oldest():
                break
        lint = [job for job in self._jobs if job.kind == "lint"]
        while len(lint) > self.max_lint_jobs:
            if not self._drop_oldest("lint"):
                break
            lint = [job for job in self._jobs if job.kind == "lint"]

    def _drop_oldest(self, kind: str | None = None) -> bool:
        """Remove the oldest finished job (of *kind*); False if none can go."""
        for index, job in enumerate(self._jobs):
            if job.status in ("queued", "running") or (kind is not None and job.kind != kind):
                continue
            self._jobs.pop(index)
            self._by_id.pop(job.id, None)
            self._result_bytes -= job.result_bytes
            self.evicted_jobs += 1
            return True
        return False

    def _evict_results(self, keep: Job | None = None) -> None:
        """Drop stored payloads, oldest first, until inside the byte budget."""
        while self._result_bytes > self.max_result_bytes:
            for job in self._jobs:
                if job.result_json is None or job is keep:
                    continue
                self._result_bytes -= job.drop_result()
                self.evicted_results += 1
                break
            else:
                return

    def clear(self) -> dict[str, Any]:
        """Drop every finished job; keep whatever is still running."""
        with self._lock:
            kept = [job for job in self._jobs if job.status in ("queued", "running")]
            cleared = len(self._jobs) - len(kept)
            self._jobs = kept
            self._by_id = {job.id: job for job in kept}
            self._result_bytes = sum(job.result_bytes for job in kept)
            return {"ok": True, "cleared": cleared, "remaining": len(kept)}

    # ── reading ────────────────────────────────────────────────────────────

    def get(self, job_id: str) -> Job | None:
        """The job with this id, or None."""
        with self._lock:
            return self._by_id.get(job_id)

    def host(self) -> dict[str, Any]:
        """What the machine has, for the monitor's denominators."""
        if psutil is not None:
            memory = psutil.virtual_memory()
            return {
                "cpu_count": psutil.cpu_count() or os.cpu_count(),
                "mem_total": int(memory.total),
                "mem_available": int(memory.available),
            }
        return {"cpu_count": os.cpu_count(), "mem_total": None, "mem_available": None}

    def snapshot(self) -> dict[str, Any]:
        """The whole ``GET /api/jobs`` payload: summaries, totals, store state.

        Cheap enough to poll at 1 Hz: one pass over at most
        :data:`MAX_JOBS` jobs plus two psutil reads of this process and the
        machine — no per-sample work and no subprocess inspection.

        Returns:
            The list payload described in the module docstring's contract.
        """
        with self._lock:
            jobs = [job.summary() for job in reversed(self._jobs)]
            result_bytes = self._result_bytes
            evicted_jobs = self.evicted_jobs
            evicted_results = self.evicted_results
            count = len(self._jobs)
        running = [job for job in jobs if job["status"] == "running"]
        server = {"cpu_percent": 0.0, "rss_bytes": 0, "pid": os.getpid()}
        if self._self_process is not None:
            try:
                server["cpu_percent"] = round(self._self_process.cpu_percent(None), 1)
                server["rss_bytes"] = int(self._self_process.memory_info().rss)
            except Exception:  # noqa: BLE001
                pass
        return {
            "ok": True,
            "jobs": jobs,
            "totals": {
                "running": len(running),
                "cpu_percent": round(float(sum(job["cpu_percent"] for job in running)), 1),
                "rss_bytes": sum(job["rss_bytes"] for job in running),
                "uptime_s": round(time.time() - self.started_at, 3),
                "server": server,
                "host": self.host(),
                "sampling": "psutil" if psutil is not None else "rusage",
            },
            "store": {
                "jobs": count,
                "max_jobs": self.max_jobs,
                "max_lint_jobs": self.max_lint_jobs,
                "result_bytes": result_bytes,
                "max_result_bytes": self.max_result_bytes,
                "evicted_jobs": evicted_jobs,
                "evicted_results": evicted_results,
            },
        }


#: The server's registry.  One per process, created on import.
REGISTRY = JobRegistry()
