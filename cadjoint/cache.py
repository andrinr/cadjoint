"""Persistent XLA compilation cache for cadjoint's short-lived processes.

The viewer runs every request in a fresh worker subprocess, so nothing
survives between edits by default: each ``/compile`` retraces the scene's
SDF and each ``/api/mesh`` retraces the dual-contouring pipeline, paying
XLA compilation every time for programs that rarely change.

JAX can persist compiled executables to disk, keyed by the lowered HLO
plus the backend and JAX version, so a later process reuses them instead
of recompiling.  A worker mode compiles hundreds to thousands of programs
(``benchmarks/jax_compile_profile.py`` reports them per mode), nearly every
one a single eager primitive; the cache turns that op-by-op cold cost into
sub-second cache reads.  What it cannot
hold is a program with a host callback — the optimizer's jitted frozen
objective calls its Tesseracts through one, so its ~5 s compile is paid by
every process (``research/performance.md`` §15).
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

__all__ = [
    "cache_directory",
    "enable_compilation_cache",
    "min_compile_seconds",
    "prune_compilation_cache",
]

#: Cap the on-disk cache. :func:`prune_compilation_cache` removes the oldest
#: entries above it.
#:
#: The cap is enforced here, not by JAX's ``jax_compilation_cache_max_size``.
#: With that option set, JAX evicts on *every* write: under a file lock it
#: globs the whole directory, stats every entry and reads a per-entry
#: access-time file, so each new program costs O(entries) — seconds per
#: write at tens of thousands of entries.  Without the option JAX writes an
#: entry and nothing else; a torn entry read by a concurrent
#: process is a warning and a recompile, never a wrong result.
_MAX_BYTES = 2 * 1024**3

#: How often :func:`prune_compilation_cache` walks the directory.  One walk
#: of 75,000 entries is a few seconds; once a day it is noise, while the
#: check itself is one ``stat`` of :data:`_PRUNE_STAMP`.
_PRUNE_INTERVAL_SECONDS = 24 * 3600

#: Marker whose mtime records the last prune.
_PRUNE_STAMP = ".cadjoint-pruned"

#: Cache every compilation. JAX's own default (1 s) is tuned for a few large
#: model programs; a scene compiles as hundreds of sub-second programs whose
#: aggregate is the cost, so a per-program threshold would skip all of them.
#:
#: Entries are written under one lockfile in the cache directory, and under
#: several concurrent sessions JAX sometimes reports abandoning a write
#: ("the file lock could not be acquired").  That does not make concurrent
#: sessions serialise or the cache stop converging: concurrent sessions
#: sharing this cache run faster than the same sessions with private caches,
#: because the shared cache hits.
#:
#: ``CADJOINT_CACHE_MIN_COMPILE_SECONDS`` raises the floor as an escape
#: hatch for a machine where the lock does contend;
#: :mod:`benchmarks.jax_compile_profile` is how to find out whether it
#: does.
_MIN_COMPILE_SECONDS = 0.0

#: Environment override for the floor above.
_MIN_COMPILE_ENV = "CADJOINT_CACHE_MIN_COMPILE_SECONDS"


def min_compile_seconds() -> float:
    """The per-program compile-time floor for caching, honouring the override.

    Returns:
        The floor in seconds; :data:`_MIN_COMPILE_SECONDS` unless
        ``CADJOINT_CACHE_MIN_COMPILE_SECONDS`` names a non-negative number,
        which an unparseable or negative value is not.
    """
    raw = os.environ.get(_MIN_COMPILE_ENV)
    if raw is None:
        return _MIN_COMPILE_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _MIN_COMPILE_SECONDS
    return value if value >= 0.0 else _MIN_COMPILE_SECONDS


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


def prune_compilation_cache(
    directory: str | os.PathLike[str] | None = None,
    max_bytes: int = _MAX_BYTES,
    *,
    force: bool = False,
) -> int:
    """Delete the oldest cache entries until the directory fits *max_bytes*.

    Runs at most once per :data:`_PRUNE_INTERVAL_SECONDS` unless *force*:
    the check is one ``stat`` of a stamp file, cheap enough for every worker
    process to make.  Age is the later of an entry's access and modification
    time, so on a filesystem that records reads it is least-recently-used,
    and least-recently-written otherwise.  It also removes the ``-atime``
    files JAX's own eviction left beside every entry; nothing reads them
    once that eviction is off.

    Args:
        directory: The cache; defaults to :func:`cache_directory`.
        max_bytes: Size budget for the ``-cache`` entries.
        force: Prune even if the last prune was recent.

    Returns:
        The number of entries removed (stale ``-atime`` files not counted).
    """
    path = Path(directory) if directory is not None else cache_directory()
    stamp = path / _PRUNE_STAMP
    now = time.time()
    try:
        if not force and now - stamp.stat().st_mtime < _PRUNE_INTERVAL_SECONDS:
            return 0
    except OSError:
        pass  # never pruned, or no directory yet

    entries: list[tuple[float, int, str]] = []
    total = 0
    try:
        with os.scandir(path) as listing:
            for item in listing:
                if item.name.endswith("-atime"):
                    _unlink(item.path)
                elif item.name.endswith("-cache") and item.is_file():
                    info = item.stat()
                    entries.append((max(info.st_atime, info.st_mtime), info.st_size, item.path))
                    total += info.st_size
    except OSError:
        return 0

    removed = 0
    for _age, size, entry in sorted(entries):
        if total <= max_bytes:
            break
        if _unlink(entry):
            total -= size
            removed += 1
    with contextlib.suppress(OSError):
        stamp.touch()
    return removed


def _unlink(path: str) -> bool:
    """Remove *path*; another process may have removed it first."""
    try:
        os.unlink(path)
    except OSError:
        return False
    return True


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
        jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_seconds())
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        # No jax_compilation_cache_max_size: see _MAX_BYTES for why the cap
        # is enforced by the prune below instead of on every write.
        prune_compilation_cache(path)
    except Exception:
        # A cache is an optimisation: a read-only home directory or a JAX
        # build without these knobs must not stop the worker from running.
        return None
    return path
