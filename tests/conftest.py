"""Session-wide test setup: give the suite the compilation cache the app has.

Every cadjoint process that matters enables JAX's persistent compilation
cache (:func:`cadjoint.cache.enable_compilation_cache`) — the viewer's
workers do it because each request is a fresh process.  The test session
never did, so every run paid XLA for the same few thousand programs: a
mesh extraction is 12.7 s cold against 5.9 s warm, a simulation 14.3
against 3.1, an optimization step 62 against 19 (the measurements in
:mod:`cadjoint.cache`).  Those are the suite's dominant cost.

The cache is keyed by the lowered HLO plus the backend and JAX version, so
it cannot serve a stale executable for changed code; ``CADJOINT_NO_COMPILATION_CACHE=1``
turns it off for a run that wants to measure cold compilation.
"""

from __future__ import annotations


def pytest_configure(config):
    """Turn the cache on before anything is collected, and register ``slow``.

    Earliest possible: a module that compiles something while it is
    imported should find the cache already in place.

    A handful of tests are minutes each because they extract a real mesh or
    run a real descent; they are the suite's cost and they are also what
    makes it worth running.  They stay in the default run — a gate that
    skips its expensive half is not a gate — and carry the marker so a
    working loop can leave them out::

        pytest -m "not slow"          # the fast half
        pytest -m slow                # only the expensive ones
    """
    from cadjoint.cache import enable_compilation_cache

    enable_compilation_cache()
    config.addinivalue_line(
        "markers", "slow: minutes-long (a real mesh extraction, solve or descent)"
    )
