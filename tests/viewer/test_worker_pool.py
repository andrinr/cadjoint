"""The served compile worker, and the three ways it must give itself up.

Keeping a worker between requests is what makes an edit cheap: a fresh
process pays its imports and then re-traces what the last one already
traced.  It also gives up the property that made a disposable worker safe —
that nothing a scene does can reach the next scene — and a scene is
arbitrary Python.  These tests are that property, put back one case at a
time.
"""

from __future__ import annotations

import pytest

from cadjoint.viewer import _worker_client as client

pytest.importorskip("jax", reason="a compile worker imports jax")

_SCENE = "from cadjoint.sdf.primitives import Sphere\nscene = Sphere(1.0)\n"


@pytest.fixture(autouse=True)
def _fresh_pool(monkeypatch):
    monkeypatch.delenv(client.WORKER_POOL_ENV, raising=False)
    client._retire_pooled()
    yield
    client._retire_pooled()


def test_a_second_compile_reuses_the_process():
    first = client.compile_source(_SCENE)
    process = client._POOL["process"]
    second = client.compile_source(_SCENE)
    assert first["ok"] and second["ok"]
    assert client._POOL["process"] is process, "the worker was replaced between requests"
    assert client._POOL["served"] == 2


def test_a_scene_that_changes_a_process_global_retires_the_worker():
    """The hazard a disposable worker made impossible.

    `jax_enable_x64` is process-global; `scenes/duct_sink.py` documents that
    flipping it at module scope makes *everything* after it float64. Served,
    that would reach the next scene, so the worker answers and then goes.
    """
    tainting = "import jax\njax.config.update('jax_enable_x64', True)\n" + _SCENE
    result = client.compile_source(tainting)
    assert result["ok"], "the tainting scene itself is valid and must still compile"
    assert client._POOL["process"] is None, "a tainted process must not be reused"
    # And the next compile is served by a fresh, untainted one.
    assert client.compile_source(_SCENE)["ok"]


def test_the_retire_notice_never_reaches_the_caller():
    """It is worker-to-client bookkeeping, not part of the compile payload."""
    from cadjoint.viewer.worker import RETIRE_FLAG

    assert RETIRE_FLAG not in client.compile_source(_SCENE)


def test_a_raising_scene_leaves_the_worker_usable():
    failed = client.compile_source("raise RuntimeError('boom')\n")
    assert failed["ok"] is False
    assert "boom" in failed["error"]
    assert client.compile_source(_SCENE)["ok"], "one bad scene must not wedge the worker"


def test_the_pool_can_be_turned_off(monkeypatch):
    monkeypatch.setenv(client.WORKER_POOL_ENV, "0")
    assert client.compile_source(_SCENE)["ok"]
    assert client._POOL["process"] is None, "no worker is kept when the pool is off"


def test_a_dead_worker_is_replaced_rather_than_reported_as_a_timeout():
    """EOF and "nothing arrived" are different, and were once conflated.

    A worker killed between requests closes the pipe immediately. Reading
    that as a timeout would tell the user their scene was too slow and make
    them wait the full budget to hear it.
    """
    client.compile_source(_SCENE)
    process = client._POOL["process"]
    assert process is not None
    process.kill()
    process.wait(timeout=10)

    result = client.compile_source(_SCENE)
    assert result["ok"], "the request must be answered, by a new process"
    assert "timeout" not in (result.get("error") or "").lower()


def test_cancelling_a_served_compile_does_not_redo_it():
    """The regression a shared worker makes possible.

    Cancelling kills the worker. With a disposable process that ended the
    request; with a shared one the closed pipe looks exactly like a worker
    that died on its own, and the obvious recovery — start a fresh process
    and retry — would redo the work the user just asked us to stop.
    """
    import threading
    import time

    from cadjoint.viewer._jobs import REGISTRY

    slow = "import time\nfrom cadjoint.sdf.primitives import Sphere\ntime.sleep(6)\nscene = Sphere(1.0)\n"
    client.compile_source(_SCENE)  # warm the pool so the kill lands mid-request

    with REGISTRY.track("compile", source=slow) as job:
        threading.Thread(target=lambda: (time.sleep(1.5), job.cancel()), daemon=True).start()
        started = time.perf_counter()
        result = client.compile_source(slow)
        elapsed = time.perf_counter() - started

    assert result["ok"] is False, "a cancelled compile must not come back successful"
    assert "cancelled" in result["error"].lower()
    assert elapsed < 5.0, "it returned when cancelled, rather than running the scene again"
