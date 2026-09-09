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

It also fails the run if a module leaves ``jax_enable_x64`` on.  That flag
is process-global: a module that sets it at import, or in a fixture that
does not restore it, silently makes every later module's arrays float64,
and the WGSL emitter — which has no 64-bit numeric type — starts refusing
scenes it should accept.  The symptom lands nowhere near the cause, passes
when the offending file is run alone, and stayed hidden here for as long as
CI could not run the suite at all.
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


def pytest_sessionstart(session):
    """Remember the precision the session began at."""
    import jax

    session.config._cadjoint_x64 = jax.config.jax_enable_x64


def pytest_sessionfinish(session, exitstatus):
    """Fail the run if a module left ``jax_enable_x64`` flipped.

    Checked once, at the end, rather than per test: a module is entitled to
    hold the flag on for its own duration through a scoped fixture, and only
    failing to put it back is the defect.  Modules that need double
    precision should follow ``tests/fem/conftest.py`` — save, set, yield,
    restore.
    """
    import jax

    before = getattr(session.config, "_cadjoint_x64", None)
    if before is None or jax.config.jax_enable_x64 == before:
        return
    session.exitstatus = 1
    print(
        "\nERROR: the session left jax_enable_x64 as "
        f"{jax.config.jax_enable_x64} (it began {before}). It is "
        "process-global, so this makes every later module's arrays float64 "
        "and the WGSL emitter refuse scenes it should accept. Set it in a "
        "fixture that restores it, as tests/fem/conftest.py does."
    )
