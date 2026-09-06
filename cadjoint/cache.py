"""Persistent XLA compilation cache for cadjoint's short-lived processes.

The viewer runs every request in a fresh worker subprocess, so nothing
survives between edits by default: each ``/compile`` retraces the scene's
SDF and each ``/api/mesh`` retraces the dual-contouring pipeline, paying
XLA compilation every time for programs that rarely change.

JAX can persist compiled executables to disk, keyed by the lowered HLO
plus the backend and JAX version, so a later process reuses them instead
of recompiling.  Measured on the starter scene, in-process per mode
(``benchmarks/jax_compile_profile.py``, 2026-09-05):

================  ==========  ==========  ==================
mode              cold        warm        programs
================  ==========  ==========  ==================
``compile``       2.1 s       1.1 s       103
``mesh``          12.7 s      5.9 s       455
``mesh_inspect``  6.9 s       1.9 s       436
``simulate``      14.3 s      3.1 s       790
``optimize`` (2)  62.1 s      19.2 s      2068
================  ==========  ==========  ==================

Nearly every one of those programs is a single eager primitive; the cache
turns the op-by-op cold cliff into sub-second cache reads.  What it cannot
hold is a program with a host callback — the optimizer's jitted frozen
objective calls its Tesseracts through one, so its ~5 s compile is paid by
every process (``research/performance.md`` §15).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["cache_directory", "enable_compilation_cache"]

#: Cap the on-disk cache. Entries are evicted least-recently-used above it.
_MAX_BYTES = 2 * 1024**3

#: Cache every compilation. JAX's own default (1 s) is tuned for a few large
#: model programs; a scene compiles as hundreds of sub-second programs whose
#: aggregate is the cost, so a per-program threshold would skip all of them.
_MIN_COMPILE_SECONDS = 0.0


def cache_directory() -> Path:
    """Where compiled executables are stored.

    ``CADJOINT_CACHE_DIR`` overrides it; otherwise it follows the XDG
    cache location, falling back to ``~/.cache``.
    """
    override = os.environ.get("CADJOINT_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base).expanduser() / "cadjoint" / "jax"


def enable_compilation_cache(directory: str | os.PathLike[str] | None = None) -> Path | None:
    """Point JAX's persistent compilation cache at a stable directory.

    Safe to call more than once and safe to call late — JAX reads the
    setting when it compiles, not when it imports.  Set
    ``CADJOINT_NO_COMPILATION_CACHE=1`` to opt out.

    Args:
        directory: Cache location; defaults to :func:`cache_directory`.

    Returns:
        The directory in use, or ``None`` when caching is disabled or
        unavailable in this JAX build.
    """
    if os.environ.get("CADJOINT_NO_COMPILATION_CACHE"):
        return None
    try:
        import jax
    except ImportError:  # pragma: no cover - jax is a hard dependency
        return None

    path = Path(directory) if directory is not None else cache_directory()
    try:
        path.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(path))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", _MIN_COMPILE_SECONDS)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_compilation_cache_max_size", _MAX_BYTES)
    except Exception:
        # A cache is an optimisation: a read-only home directory or a JAX
        # build without these knobs must not stop the worker from running.
        return None
    return path
