"""The server's startup compilation-cache warm-up.

``warm_start`` issues one ``compile`` and one ``mesh`` of the scene the
editor opens with, on a daemon thread, so the first real request meets a
warm on-disk XLA cache instead of paying 45-53 s for a cold one.  What
these tests pin down is everything around that: that it never blocks the
caller, that it runs at most once *per program*, that it asks for exactly
those two modes on exactly that source and no other scene, that it yields
the core to a request the user is waiting on, and that a test run does not
silently spawn two worker processes per server it creates.
"""

from __future__ import annotations

import threading

import pytest

from cadjoint.viewer import _worker_client
from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer._worker_client import WARM_START_ENV, _warm_start_enabled, warm_start


@pytest.fixture
def unwarmed(monkeypatch):
    """A process that has not warmed yet, and forgets it afterwards."""
    monkeypatch.setattr(_worker_client, "_WARM_STARTED", threading.Event())
    monkeypatch.setattr(_worker_client, "_WARMED", set())


@pytest.fixture
def recorded(monkeypatch):
    """Record ``_run_worker`` calls instead of spawning worker processes."""
    calls: list[tuple[str, str, float]] = []

    def fake(source, mode, timeout, extra=None, *, nice=0):
        calls.append((source, mode, timeout))
        return {"ok": True}

    monkeypatch.setattr(_worker_client, "_run_worker", fake)
    return calls


class TestGating:
    def test_off_under_pytest_by_default(self, monkeypatch):
        monkeypatch.delenv(WARM_START_ENV, raising=False)
        assert not _warm_start_enabled()

    @pytest.mark.parametrize("setting", ["1", "true", "yes", "on", "please"])
    def test_the_environment_can_force_it_on(self, monkeypatch, setting):
        monkeypatch.setenv(WARM_START_ENV, setting)
        assert _warm_start_enabled()

    @pytest.mark.parametrize("setting", ["0", "false", "no", "off", "", "  "])
    def test_the_environment_can_force_it_off(self, monkeypatch, setting):
        monkeypatch.setenv(WARM_START_ENV, setting)
        assert not _warm_start_enabled()

    def test_a_server_created_in_a_test_run_does_not_warm(self, unwarmed, recorded, monkeypatch):
        """``create_server`` is called by the live-server tests.

        Without the pytest default every one of them would spawn a full
        ``compile`` and a full ``mesh`` worker it never reads.
        """
        from cadjoint.viewer.playground import create_server

        monkeypatch.delenv(WARM_START_ENV, raising=False)
        server = create_server(0)
        try:
            assert recorded == []
        finally:
            server.server_close()


class TestWarmStart:
    def test_it_declines_when_disabled(self, unwarmed, recorded, monkeypatch):
        monkeypatch.setenv(WARM_START_ENV, "0")
        assert warm_start() is False
        assert recorded == []

    def test_it_asks_for_one_compile_and_one_mesh_of_the_example_scene(
        self, unwarmed, recorded, monkeypatch
    ):
        monkeypatch.setenv(WARM_START_ENV, "1")
        started = threading.Event()
        real_thread = threading.Thread

        def watched(*args, **kwargs):
            thread = real_thread(*args, **kwargs)
            started.thread = thread  # type: ignore[attr-defined]
            started.set()
            return thread

        monkeypatch.setattr(_worker_client.threading, "Thread", watched)
        assert warm_start() is True
        assert started.wait(1.0), "no warm-up thread was started"
        started.thread.join(timeout=60)  # type: ignore[attr-defined]

        # Only the scene the editor opens with, as one compile and one mesh.
        # Warming the whole scenes directory here is what made a user's first
        # compile queue behind work nobody asked for; the others are warmed
        # when they are opened.
        assert [mode for _source, mode, _timeout in recorded] == ["compile", "mesh"]
        assert [source for source, _mode, _timeout in recorded] == [EXAMPLE_SOURCE] * 2
        # Both run under the mesh budget: the warm-up exists for the cold
        # path, where even a compile can outgrow the edit round-trip budget.
        assert {timeout for _s, _m, timeout in recorded} == {_worker_client.MESH_TIMEOUT_SECONDS}

    def test_it_warms_a_given_source_and_only_once(self, unwarmed, recorded, monkeypatch):
        monkeypatch.setenv(WARM_START_ENV, "1")
        threads: list[threading.Thread] = []
        real_thread = threading.Thread
        monkeypatch.setattr(
            _worker_client.threading,
            "Thread",
            lambda *a, **k: threads.append(real_thread(*a, **k)) or threads[-1],
        )
        assert warm_start("scene = None") is True
        assert warm_start("scene = None") is False, "warm_start must run once per process"
        for thread in threads:
            thread.join(timeout=60)
        assert [source for source, _mode, _timeout in recorded] == ["scene = None"] * 2

    def test_it_does_not_block_the_caller(self, unwarmed, monkeypatch):
        """A slow worker must not hold up ``create_server``."""
        monkeypatch.setenv(WARM_START_ENV, "1")
        release = threading.Event()

        def blocking(source, mode, timeout, extra=None):
            release.wait(30)
            return {"ok": True}

        monkeypatch.setattr(_worker_client, "_run_worker", blocking)
        try:
            assert warm_start() is True
        finally:
            release.set()


class TestItYieldsToTheUser:
    """The warm-up is speculative, so it must never take a core from a request."""

    def test_the_warm_up_child_runs_niced(self, unwarmed, monkeypatch):
        monkeypatch.setenv(WARM_START_ENV, "1")
        seen: list[int] = []

        def record(source, mode, timeout, extra=None, *, nice=0):
            seen.append(nice)
            return {"ok": True}

        monkeypatch.setattr(_worker_client, "_run_worker", record)
        threads: list[threading.Thread] = []
        real_thread = threading.Thread
        monkeypatch.setattr(
            _worker_client.threading,
            "Thread",
            lambda *a, **k: threads.append(real_thread(*a, **k)) or threads[-1],
        )
        assert warm_start("scene = None") is True
        for thread in threads:
            thread.join(timeout=60)
        assert seen == [_worker_client._WARM_NICE] * 2
        assert _worker_client._WARM_NICE > 0, "a warm-up must rank below a request"

    def test_a_request_is_not_niced(self, monkeypatch):
        """The same launcher runs a user's compile at ordinary priority."""
        import inspect

        signature = inspect.signature(_worker_client._run_worker)
        assert signature.parameters["nice"].default == 0


class TestEachSceneWarmsWhenItIsOpened:
    """Scenes other than the editor's own are warmed on open, not at launch."""

    def test_opening_a_scene_warms_it_once(self, unwarmed, recorded, monkeypatch, tmp_path):
        monkeypatch.setenv(WARM_START_ENV, "1")
        (tmp_path / "widget.py").write_text("scene = None  # widget\n")
        monkeypatch.setenv("CADJOINT_SCENES_DIR", str(tmp_path))
        threads: list[threading.Thread] = []
        real_thread = threading.Thread
        monkeypatch.setattr(
            _worker_client.threading,
            "Thread",
            lambda *a, **k: threads.append(real_thread(*a, **k)) or threads[-1],
        )
        from cadjoint.viewer._scenes import load_scene

        assert load_scene({"name": "widget.py"})["ok"]
        assert load_scene({"name": "widget.py"})["ok"], "a second open must not warm again"
        for thread in threads:
            thread.join(timeout=60)
        assert [source for source, _mode, _timeout in recorded] == ["scene = None  # widget\n"] * 2

    def test_warming_is_per_program_not_per_process(self, unwarmed, recorded, monkeypatch):
        monkeypatch.setenv(WARM_START_ENV, "1")
        threads: list[threading.Thread] = []
        real_thread = threading.Thread
        monkeypatch.setattr(
            _worker_client.threading,
            "Thread",
            lambda *a, **k: threads.append(real_thread(*a, **k)) or threads[-1],
        )
        assert _worker_client.warm_scene("scene = 1") is True
        assert _worker_client.warm_scene("scene = 1") is False
        assert _worker_client.warm_scene("scene = 2") is True
        for thread in threads:
            thread.join(timeout=60)
        assert {source for source, _mode, _timeout in recorded} == {"scene = 1", "scene = 2"}
