"""The server's job registry: what ran, what is running, and its results.

Three things are pinned down here.  The **bookkeeping** — every timed
request becomes a :class:`~cadjoint.viewer._jobs.Job` with a source hash, a
worker pid and resource samples, and the response it produced stays
fetchable after the panel that asked for it is gone.  The **bounds** — the
registry is a fixed-size window over history, so a long session cannot grow
without limit.  And **cancellation** — which is only worth anything if it
reaches the worker subprocess, so the HTTP tests here cancel a real run and
watch the original request come back saying so.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cadjoint.viewer import _jobs, playground
from cadjoint.viewer._jobs import JobRegistry, attach_process, source_hash

# A program the worker takes a long time to finish, with no numerics in it:
# the point is a request that is still running when the test cancels it.
SLOW_SOURCE = """
import time

from cadjoint.geometry import Vector
from cadjoint.sdf.primitives import Box

time.sleep(30)
scene = Box(Vector([1.0, 1.0, 1.0]))
"""

TINY_SOURCE = "from cadjoint.sdf.primitives import Sphere\n\nscene = Sphere(0.5)\n"

STUDY_SOURCE = "\n".join(
    [
        "from cadjoint.fem import Dirichlet, Nodes, ThermalStudy",
        "from cadjoint.geometry import Vector",
        "from cadjoint.sdf.primitives import Box",
        "",
        'scene = Box(Vector([0.8, 0.5, 0.5], free=True, name="size"))',
        "heat = ThermalStudy(name='bar', resolution=8, conductivity=1.0,",
        "                    bcs=[Dirichlet(Nodes.side('-x'), 0.0),",
        "                         Dirichlet(Nodes.side('+x'), 1.0)])",
        "",
    ]
)


@pytest.fixture
def registry(monkeypatch):
    """A registry of this test's own, wired into the routing table."""
    fresh = JobRegistry()
    monkeypatch.setattr(playground, "REGISTRY", fresh)
    return fresh


def _wait_for(predicate, timeout: float = 20.0, interval: float = 0.05):
    """Poll *predicate* until it returns something truthy, or fail."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout:g}s")


# ── The job model ───────────────────────────────────────────────────────────


class TestSourceHash:
    def test_it_is_the_sha256_of_the_program_text(self):
        import hashlib

        assert source_hash("scene = None") == hashlib.sha256(b"scene = None").hexdigest()

    def test_the_same_text_hashes_the_same_way_twice(self):
        assert source_hash(TINY_SOURCE) == source_hash(TINY_SOURCE)
        assert source_hash(TINY_SOURCE) != source_hash(TINY_SOURCE + "\n")

    @pytest.mark.parametrize("source", [None, 7, {"source": "x"}])
    def test_a_request_without_program_text_has_no_hash(self, source):
        assert source_hash(source) is None


class TestLifecycle:
    def test_a_successful_job_records_its_result_and_its_identity(self, registry):
        with registry.track("simulate", source=TINY_SOURCE, fields={"name": "bar"}) as job:
            payload = registry.finish(job, {"ok": True, "field": "temperature"})

        assert payload == {"ok": True, "field": "temperature", "job_id": job.id}
        summary = job.summary()
        assert summary["kind"] == "simulate"
        assert summary["status"] == "done"
        assert summary["ok"] is True
        assert summary["fields"] == {"name": "bar"}
        assert summary["source_hash"] == source_hash(TINY_SOURCE)
        assert summary["result_available"] is True
        assert json.loads(job.result_json) == payload

    def test_a_failed_result_is_a_failed_job_that_keeps_its_error(self, registry):
        with registry.track("compile", source=TINY_SOURCE) as job:
            registry.finish(job, {"ok": False, "error": "NameError: boom"})

        assert job.status == "failed"
        assert job.ok is False
        assert job.error == "NameError: boom"

    def test_an_exception_inside_the_block_fails_the_job_and_propagates(self, registry):
        with pytest.raises(RuntimeError, match="lost the worker"):
            with registry.track("mesh", source=TINY_SOURCE):
                raise RuntimeError("lost the worker")

        job = registry.snapshot()["jobs"][0]
        assert job["status"] == "failed"
        assert "lost the worker" in job["error"]

    def test_a_block_that_never_reports_still_finishes(self, registry):
        with registry.track("compile", source=TINY_SOURCE) as job:
            pass

        assert job.status == "failed"
        assert job.finished_at is not None

    def test_optimize_progress_is_mirrored_onto_the_job(self, registry):
        with registry.track("optimize", source=TINY_SOURCE, fields={"name": "fit"}) as job:
            for step in range(1, 4):
                job.record_progress(
                    {"event": "progress", "step": step, "steps": 3, "objective": 1.0 / step}
                )
            registry.finish(job, {"ok": True})

        assert job.progress == {"step": 3, "steps": 3, "objective": pytest.approx(1 / 3)}
        assert [event["step"] for event in job.detail()["progress_events"]] == [1, 2, 3]

    def test_progress_events_are_bounded(self, registry, monkeypatch):
        monkeypatch.setattr(_jobs, "MAX_PROGRESS_EVENTS", 4)
        with registry.track("optimize", source=TINY_SOURCE) as job:
            for step in range(20):
                job.record_progress({"step": step})
            registry.finish(job, {"ok": True})

        assert len(job.progress_events) == 4
        assert job.progress == {"step": 19}

    def test_elapsed_counts_up_while_running_and_freezes_when_done(self, registry):
        with registry.track("compile", source=TINY_SOURCE) as job:
            time.sleep(0.05)
            running = job.elapsed_s
            assert job.status == "running"
            registry.finish(job, {"ok": True})
        frozen = job.elapsed_s
        assert running >= 0.05
        time.sleep(0.02)
        assert job.elapsed_s == frozen


# ── Bounds ──────────────────────────────────────────────────────────────────


class TestEviction:
    def _finished(self, registry, kind: str, count: int, result=None) -> None:
        for _ in range(count):
            with registry.track(kind, source=TINY_SOURCE) as job:
                registry.finish(job, result if result is not None else {"ok": True})

    def test_only_the_last_n_jobs_are_kept(self):
        registry = JobRegistry(max_jobs=3)
        self._finished(registry, "compile", 6)

        snapshot = registry.snapshot()
        assert snapshot["store"]["jobs"] == 3
        assert snapshot["store"]["evicted_jobs"] == 3
        # The window is over the newest work, and the oldest ids are gone.
        assert [job["job_id"] for job in snapshot["jobs"]] == [
            "job-000006",
            "job-000005",
            "job-000004",
        ]
        assert registry.get("job-000001") is None

    def test_running_jobs_are_never_evicted(self):
        registry = JobRegistry(max_jobs=2)
        with registry.track("simulate", source=TINY_SOURCE) as running:
            self._finished(registry, "compile", 5)
            assert registry.get(running.id) is running
            assert running.status == "running"
            registry.finish(running, {"ok": True})

    def test_lint_jobs_cannot_flush_the_expensive_history(self):
        registry = JobRegistry(max_jobs=50, max_lint_jobs=2)
        self._finished(registry, "simulate", 1)
        self._finished(registry, "lint", 8)

        kinds = [job["kind"] for job in registry.snapshot()["jobs"]]
        assert kinds.count("lint") == 2
        assert kinds.count("simulate") == 1

    def test_a_lint_result_is_never_stored(self, registry):
        with registry.track("lint", source=TINY_SOURCE) as job:
            registry.finish(job, {"ok": True, "diagnostics": []})

        assert job.result_json is None
        assert job.summary()["result_available"] is False

    def test_results_are_dropped_oldest_first_past_the_byte_budget(self):
        registry = JobRegistry(max_jobs=50, max_result_bytes=4_000)
        big = {"ok": True, "blob": "x" * 3_000}
        self._finished(registry, "compile", 3, result=big)

        snapshot = registry.snapshot()
        available = [job["result_available"] for job in snapshot["jobs"]]
        # Newest first: only the most recent result still fits the budget.
        assert available == [True, False, False]
        assert snapshot["store"]["result_bytes"] <= 4_000
        assert snapshot["store"]["evicted_results"] == 2
        # The jobs themselves survive: history is cheap, payloads are not.
        assert snapshot["store"]["jobs"] == 3

    def test_clearing_drops_finished_work_and_keeps_what_runs(self, registry):
        self._finished(registry, "compile", 3)
        with registry.track("simulate", source=TINY_SOURCE) as running:
            assert registry.clear() == {"ok": True, "cleared": 3, "remaining": 1}
            # The three cleared results went with them; the survivor's does not.
            assert registry.snapshot()["store"]["result_bytes"] == 0
            registry.finish(running, {"ok": True})
        assert [job["job_id"] for job in registry.snapshot()["jobs"]] == [running.id]


class TestSnapshot:
    def test_it_reports_totals_and_what_the_machine_has(self, registry):
        with registry.track("compile", source=TINY_SOURCE) as job:
            registry.finish(job, {"ok": True})

        totals = registry.snapshot()["totals"]
        assert totals["running"] == 0
        assert totals["uptime_s"] >= 0.0
        assert totals["sampling"] in ("psutil", "rusage")
        assert totals["host"]["cpu_count"] >= 1
        assert set(totals["host"]) == {"cpu_count", "mem_total", "mem_available"}

    def test_a_running_job_shows_up_in_the_totals(self, registry):
        with registry.track("simulate", source=TINY_SOURCE) as job:
            totals = registry.snapshot()["totals"]
            assert totals["running"] == 1
            registry.finish(job, {"ok": True})
        assert registry.snapshot()["totals"]["running"] == 0


# ── Watching the worker ─────────────────────────────────────────────────────


def _sleeper(seconds: float) -> subprocess.Popen:
    """A child process that burns CPU for *seconds* and then exits."""
    program = (
        f"import time\nend = time.monotonic() + {seconds}\n"
        "while time.monotonic() < end:\n    pass\n"
    )
    return subprocess.Popen([sys.executable, "-c", program])


class TestSampling:
    def test_a_worker_is_sampled_for_cpu_and_memory_while_it_runs(self, registry):
        pytest.importorskip("psutil", reason="sampling degrades to rusage without psutil")
        with registry.track("compile", source=TINY_SOURCE) as job:
            process = _sleeper(1.5)
            attach_process(process)
            process.wait()
            registry.finish(job, {"ok": True})

        assert job.pid == process.pid
        assert job.sampling == "psutil"
        samples = job.detail()["sample_series"]
        assert len(samples) >= 2, samples
        assert all(sample["rss_bytes"] > 0 for sample in samples)
        assert job.peak_cpu_percent > 0.0
        assert job.peak_rss_bytes > 0
        assert job.cpu_seconds > 0.0
        # Sample times are relative to the job's start and monotonic.
        times = [sample["t"] for sample in samples]
        assert times == sorted(times)

    def test_samples_are_bounded(self, registry, monkeypatch):
        monkeypatch.setattr(_jobs, "MAX_SAMPLES", 3)
        with registry.track("compile", source=TINY_SOURCE) as job:
            for index in range(10):
                job._add_sample(10.0 * index, 1_000 * index)
            registry.finish(job, {"ok": True})

        assert len(job.samples) == 3
        assert job.peak_rss_bytes == 9_000

    def test_without_psutil_it_degrades_to_rusage_totals_and_says_so(self, monkeypatch):
        monkeypatch.setattr(_jobs, "psutil", None)
        registry = JobRegistry()
        with registry.track("compile", source=TINY_SOURCE) as job:
            process = _sleeper(0.4)
            attach_process(process)
            process.wait()
            registry.finish(job, {"ok": True})

        assert job.sampling == "rusage"
        assert job.samples == []
        assert job.cpu_seconds > 0.0
        assert job.peak_rss_bytes > 0
        assert registry.snapshot()["totals"]["sampling"] == "rusage"
        assert registry.snapshot()["totals"]["host"]["mem_total"] is None

    def test_attaching_outside_a_tracked_request_is_a_no_op(self):
        process = _sleeper(0.05)
        attach_process(process)  # the worker client runs outside the server too
        process.wait()

    def test_cancelling_before_the_worker_starts_kills_it_on_arrival(self, registry):
        with registry.track("simulate", source=SLOW_SOURCE) as job:
            assert job.cancel() is True
            process = _sleeper(30)
            attach_process(process)
            assert process.wait(10) != 0
            registry.finish(job, {"ok": False, "error": "The compiler process exited."})

        assert job.status == "cancelled"
        assert job.error == "cancelled"

    def test_a_finished_job_cannot_be_cancelled(self, registry):
        with registry.track("compile", source=TINY_SOURCE) as job:
            registry.finish(job, {"ok": True})
        assert job.cancel() is False
        assert job.status == "done"


class TestClientCancellation:
    """Stopping work by the *client's* name for it, not by the job id.

    A job id is minted here and reaches the browser on the response, which is
    the moment the work is already over — useless as a handle for stopping a
    compile that a newer edit has just replaced.  So a request that can be
    superseded carries a label the client chose before sending it, and this is
    what that label can do.
    """

    def test_it_kills_the_running_job_that_carries_the_label(self, registry):
        with registry.track("compile", source=SLOW_SOURCE, client_id="c7") as job:
            process = _sleeper(30)
            attach_process(process)
            assert registry.cancel_client("c7") == [job.id]
            assert process.wait(10) != 0
            registry.finish(job, {"ok": False, "error": "The compiler process exited."})

        assert job.status == "cancelled"
        assert job.error == "cancelled"

    def test_it_leaves_work_under_another_label_alone(self, registry):
        with registry.track("compile", source=TINY_SOURCE, client_id="mine") as job:
            assert registry.cancel_client("someone-else") == []
            registry.finish(job, {"ok": True})
        assert job.status == "done"

    def test_a_request_cancelled_before_it_arrives_is_born_cancelled(self, registry):
        # The supersession races the request it supersedes: the browser can
        # ask to stop a compile whose POST is still crossing the loopback.
        assert registry.cancel_client("early") == []
        with registry.track("compile", source=SLOW_SOURCE, client_id="early") as job:
            assert job.cancel_requested is True
            process = _sleeper(30)
            attach_process(process)  # killed on arrival, exactly as if late
            assert process.wait(10) != 0
            registry.finish(job, {"ok": False, "error": "The compiler process exited."})

        assert job.status == "cancelled"

    def test_the_pending_set_is_bounded(self, registry):
        for index in range(_jobs.MAX_PENDING_CANCELS * 3):
            registry.cancel_client(f"c{index}")
        # Long-lived state fed by a client that supersedes on every drag has
        # to forget: only the most recent labels are still remembered.
        assert len(registry._cancelled_clients) == _jobs.MAX_PENDING_CANCELS
        with registry.track("compile", source=TINY_SOURCE, client_id="c0") as job:
            registry.finish(job, {"ok": True})
        assert job.status == "done"

    def test_a_job_with_no_label_is_never_matched(self, registry):
        with registry.track("compile", source=TINY_SOURCE) as job:
            assert registry.cancel_client("anything") == []
            registry.finish(job, {"ok": True})
        assert job.client_id is None
        assert job.status == "done"


# ── Over live HTTP ──────────────────────────────────────────────────────────


@contextmanager
def _running_server():
    server = playground.create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class Client:
    """The three lines of urllib every test below would otherwise repeat."""

    def __init__(self, base: str) -> None:
        self.base = base
        with urlopen(f"{base}/api/session") as response:
            self.token = json.loads(response.read())["token"]

    def _request(self, path: str, payload=None, method: str | None = None) -> Request:
        headers = {"Content-Type": "application/json", "X-Cadjoint-Token": self.token}
        data = json.dumps(payload).encode() if payload is not None else None
        return Request(
            self.base + path,
            data=data,
            headers=headers,
            method=method or ("POST" if data is not None else "GET"),
        )

    def call(self, path: str, payload=None, method: str | None = None):
        """``(status, parsed body)``, treating an HTTP error as a response."""
        try:
            with urlopen(self._request(path, payload, method)) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def text(self, path: str) -> str:
        """The raw response body, for byte-identity checks."""
        with urlopen(self._request(path)) as response:
            return response.read().decode("utf-8")

    def stream(self, path: str, payload) -> list[dict]:
        with urlopen(self._request(path, payload)) as response:
            return [json.loads(line) for line in response if line.strip()]


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/jobs", None),
        ("/api/jobs/job-000001", None),
        ("/api/jobs/job-000001/result", None),
        ("/api/jobs/job-000001/cancel", {}),
        ("/api/jobs/cancel_client", {"client_id": "c1"}),
        ("/api/jobs/clear", {}),
    ],
)
def test_the_job_endpoints_require_the_session_token(path, payload):
    with _running_server() as base:
        request = Request(
            base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403


class TestHttpRegistry:
    def test_a_compile_is_listed_and_its_result_comes_back_byte_identical(self, registry):
        with _running_server() as base:
            client = Client(base)
            status, body = client.call("/compile", {"source": TINY_SOURCE})
            assert status == 200
            assert body["ok"] is True
            job_id = body["job_id"]
            original = json.dumps(body)

            status, listing = client.call("/api/jobs")
            assert status == 200
            summary = listing["jobs"][0]
            assert summary["job_id"] == job_id
            assert summary["kind"] == "compile"
            assert summary["status"] == "done"
            assert summary["source_hash"] == source_hash(TINY_SOURCE)
            assert summary["pid"] > 0
            assert summary["elapsed_s"] > 0.0
            assert "result" not in summary, "the listing must stay cheap to poll"

            status, detail = client.call(f"/api/jobs/{job_id}")
            assert status == 200
            assert detail["job"]["job_id"] == job_id
            assert "sample_series" in detail["job"]

            # The panel that asked for this is gone; the payload is not.
            assert client.text(f"/api/jobs/{job_id}/result") == original

    def test_a_simulate_result_survives_the_response_being_discarded(self, registry):
        pytest.importorskip("jax_fem", reason="a study solve needs the fem extra")
        with _running_server() as base:
            client = Client(base)
            status, body = client.call(
                "/api/simulate", {"source": STUDY_SOURCE, "kind": "study", "name": "bar"}
            )
            assert status == 200, body
            job_id = body["job_id"]
            expected = json.dumps(body)
            del body  # the Results panel closes, the mode switches, the tab moves

            refetched = client.text(f"/api/jobs/{job_id}/result")
            assert refetched == expected
            result = json.loads(refetched)
            assert result["ok"] is True
            assert result["field"] == "temperature"
            assert result["job_id"] == job_id

            summary = client.call("/api/jobs")[1]["jobs"][0]
            assert summary["kind"] == "simulate"
            assert summary["fields"] == {"kind": "study", "name": "bar"}
            assert summary["result_bytes"] == len(refetched.encode("utf-8"))

    def test_an_unknown_job_is_a_404_and_a_running_one_has_no_result_yet(self, registry):
        with _running_server() as base:
            client = Client(base)
            assert client.call("/api/jobs/job-424242")[0] == 404
            assert client.call("/api/jobs/job-424242/result")[0] == 404

            done = threading.Thread(
                target=lambda: client.call("/compile", {"source": SLOW_SOURCE}), daemon=True
            )
            done.start()
            job_id = _wait_for(
                lambda: next(
                    (
                        job["job_id"]
                        for job in client.call("/api/jobs")[1]["jobs"]
                        if job["status"] == "running"
                    ),
                    None,
                )
            )
            status, body = client.call(f"/api/jobs/{job_id}/result")
            assert status == 409
            assert "running" in body["error"]
            client.call(f"/api/jobs/{job_id}/cancel", {})
            done.join(20)

    def test_a_lint_job_is_recorded_but_its_diagnostics_are_not_kept(self, registry):
        pytest.importorskip("ruff", reason="/api/lint needs the editor extra")
        with _running_server() as base:
            client = Client(base)
            status, body = client.call("/api/lint", {"source": TINY_SOURCE})
            assert status == 200
            job_id = body["job_id"]
            assert client.call("/api/jobs")[1]["jobs"][0]["kind"] == "lint"
            status, gone = client.call(f"/api/jobs/{job_id}/result")
            assert status == 410
            assert gone["job_id"] == job_id

    def test_clearing_empties_the_list(self, registry):
        with _running_server() as base:
            client = Client(base)
            client.call("/compile", {"source": TINY_SOURCE})
            assert client.call("/api/jobs")[1]["store"]["jobs"] == 1

            status, cleared = client.call("/api/jobs/clear", {})
            assert status == 200
            assert cleared == {"ok": True, "cleared": 1, "remaining": 0}
            assert client.call("/api/jobs")[1]["jobs"] == []


class TestHttpCancellation:
    def test_cancelling_a_running_compile_kills_the_worker_and_answers_the_caller(self, registry):
        answer: dict = {}
        with _running_server() as base:
            client = Client(base)
            caller = threading.Thread(
                target=lambda: answer.update(
                    zip(
                        ("status", "body"),
                        client.call("/compile", {"source": SLOW_SOURCE}),
                        strict=True,
                    )
                ),
                daemon=True,
            )
            caller.start()
            running = _wait_for(
                lambda: next(
                    (
                        job
                        for job in client.call("/api/jobs")[1]["jobs"]
                        if job["status"] == "running" and job["pid"]
                    ),
                    None,
                )
            )
            pid = running["pid"]
            assert client.call("/api/jobs")[1]["totals"]["running"] == 1

            status, cancelled = client.call(f"/api/jobs/{running['job_id']}/cancel", {})
            assert status == 200
            assert cancelled == {"ok": True, "job_id": running["job_id"], "status": "cancelled"}

            caller.join(20)
            # The request that started the work hears about it, in its own
            # response, rather than hanging for the full 30-second sleep.
            assert answer["status"] == 409
            assert answer["body"] == {
                "ok": False,
                "error": "cancelled",
                "error_kind": "cancelled",
                "job_id": running["job_id"],
            }

            summary = client.call(f"/api/jobs/{running['job_id']}")[1]["job"]
            assert summary["status"] == "cancelled"
            assert summary["elapsed_s"] < 25.0, "the sleep was never waited out"
            # And the child is really gone, not merely abandoned.
            _wait_for(lambda: not _pid_alive(pid), timeout=10.0)

            # A second cancel has nothing left to stop.
            assert client.call(f"/api/jobs/{running['job_id']}/cancel", {})[0] == 409

    def test_superseding_a_compile_by_client_label_kills_its_worker(self, registry):
        """What a second drag does to the first drag's compile.

        The browser labels the request before it sends it, so it can stop the
        work without ever having seen a job id — which is the only handle it
        could have while the request is still in flight.
        """
        answer: dict = {}
        with _running_server() as base:
            client = Client(base)
            caller = threading.Thread(
                target=lambda: answer.update(
                    zip(
                        ("status", "body"),
                        client.call("/compile", {"source": SLOW_SOURCE, "client_id": "drag-1"}),
                        strict=True,
                    )
                ),
                daemon=True,
            )
            caller.start()
            running = _wait_for(
                lambda: next(
                    (
                        job
                        for job in client.call("/api/jobs")[1]["jobs"]
                        if job["status"] == "running" and job["pid"]
                    ),
                    None,
                )
            )
            pid = running["pid"]

            status, stopped = client.call("/api/jobs/cancel_client", {"client_id": "drag-1"})
            assert status == 200
            assert stopped == {
                "ok": True,
                "client_id": "drag-1",
                "cancelled": [running["job_id"]],
            }

            caller.join(20)
            assert answer["status"] == 409
            assert answer["body"]["error_kind"] == "cancelled"

            summary = client.call(f"/api/jobs/{running['job_id']}")[1]["job"]
            assert summary["status"] == "cancelled"
            assert summary["elapsed_s"] < 25.0, "the sleep was never waited out"
            # The point of cancelling rather than merely ignoring the answer:
            # a superseded compile is a whole core for twenty-five seconds.
            _wait_for(lambda: not _pid_alive(pid), timeout=10.0)

    def test_a_cancel_without_a_label_is_refused(self, registry):
        with _running_server() as base:
            client = Client(base)
            assert client.call("/api/jobs/cancel_client", {})[0] == 400
            assert client.call("/api/jobs/cancel_client", {"client_id": "x" * 200})[0] == 400

    def test_cancelling_a_streamed_optimize_ends_the_stream(self, registry):
        events: list = []
        with _running_server() as base:
            client = Client(base)
            source = SLOW_SOURCE + (
                "\nfrom cadjoint.optimize import Optimization\n"
                "shrink = Optimization(name='fit', objective=lambda p: 0.0, of=scene, steps=4)\n"
            )
            caller = threading.Thread(
                target=lambda: events.extend(
                    client.stream("/api/optimize", {"source": source, "name": "fit"})
                ),
                daemon=True,
            )
            caller.start()
            running = _wait_for(
                lambda: next(
                    (
                        job
                        for job in client.call("/api/jobs")[1]["jobs"]
                        if job["status"] == "running" and job["pid"]
                    ),
                    None,
                )
            )
            assert running["kind"] == "optimize"
            assert client.call(f"/api/jobs/{running['job_id']}/cancel", {})[0] == 200
            caller.join(20)

        assert events, "the stream produced nothing"
        assert events[-1] == {
            "event": "done",
            "ok": False,
            "error": "cancelled",
            "error_kind": "cancelled",
            "job_id": running["job_id"],
        }


def _pid_alive(pid: int) -> bool:
    """Whether *pid* is still a live process (not merely an unreaped zombie)."""
    psutil = pytest.importorskip("psutil")
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
