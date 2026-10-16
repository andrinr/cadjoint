"""The persistent compilation cache: configuration and opt-out."""

from __future__ import annotations

import jax
import pytest

from cadjoint import cache


@pytest.fixture
def restore_jax_cache_config():
    keys = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
        "jax_compilation_cache_max_size",
    )
    saved = {k: getattr(jax.config, k) for k in keys}
    yield
    for k, v in saved.items():
        jax.config.update(k, v)


def test_cache_directory_prefers_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CADJOINT_CACHE_DIR", str(tmp_path / "custom"))
    assert cache.cache_directory() == tmp_path / "custom"


def test_cache_directory_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("CADJOINT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache.cache_directory() == tmp_path / "cadjoint" / "jax"


def test_enable_points_jax_at_the_directory(monkeypatch, tmp_path, restore_jax_cache_config):
    monkeypatch.delenv("CADJOINT_NO_COMPILATION_CACHE", raising=False)
    monkeypatch.delenv("CADJOINT_CACHE_MIN_COMPILE_SECONDS", raising=False)
    target = tmp_path / "jaxcache"
    assert cache.enable_compilation_cache(target) == target
    assert target.is_dir()
    assert jax.config.jax_compilation_cache_dir == str(target)
    # Every program must be eligible: a scene is hundreds of sub-second
    # compilations, so a per-program time floor would cache none of them.
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0


def test_min_compile_seconds_defaults_to_caching_everything(monkeypatch):
    monkeypatch.delenv("CADJOINT_CACHE_MIN_COMPILE_SECONDS", raising=False)
    assert cache.min_compile_seconds() == 0.0


@pytest.mark.parametrize("raw,expected", [("0.05", 0.05), ("1", 1.0), ("0", 0.0)])
def test_min_compile_seconds_reads_the_override(monkeypatch, raw, expected):
    monkeypatch.setenv("CADJOINT_CACHE_MIN_COMPILE_SECONDS", raw)
    assert cache.min_compile_seconds() == expected


@pytest.mark.parametrize("raw", ["", "nonsense", "-3"])
def test_a_bad_override_falls_back_rather_than_raising(monkeypatch, raw):
    """A cache is an optimisation; a typo in an env var must not stop a run."""
    monkeypatch.setenv("CADJOINT_CACHE_MIN_COMPILE_SECONDS", raw)
    assert cache.min_compile_seconds() == 0.0


def test_the_override_reaches_jax(monkeypatch, tmp_path, restore_jax_cache_config):
    monkeypatch.delenv("CADJOINT_NO_COMPILATION_CACHE", raising=False)
    monkeypatch.setenv("CADJOINT_CACHE_MIN_COMPILE_SECONDS", "0.25")
    cache.enable_compilation_cache(tmp_path / "jaxcache")
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.25


def test_opt_out_leaves_jax_alone(monkeypatch, tmp_path, restore_jax_cache_config):
    monkeypatch.setenv("CADJOINT_NO_COMPILATION_CACHE", "1")
    before = jax.config.jax_compilation_cache_dir
    assert cache.enable_compilation_cache(tmp_path / "ignored") is None
    assert jax.config.jax_compilation_cache_dir == before
    assert not (tmp_path / "ignored").exists()


def test_compiled_program_round_trips_through_disk(tmp_path, restore_jax_cache_config):
    """A second process must find the executable the first one compiled."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        from cadjoint.cache import enable_compilation_cache
        enable_compilation_cache({str(tmp_path)!r})
        import jax, jax.numpy as jnp
        f = jax.jit(lambda x: jnp.sin(x) * 3.0 + jnp.cos(x) ** 2)
        f(jnp.arange(64.0)).block_until_ready()
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
    entries_after_first = sum(1 for p in tmp_path.rglob("*") if p.is_file())
    assert entries_after_first >= 1
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
    entries_after_second = sum(1 for p in tmp_path.rglob("*") if p.is_file())
    # A hit does not write a new entry.
    assert entries_after_second == entries_after_first


def test_enable_leaves_jax_eviction_off(monkeypatch, tmp_path, restore_jax_cache_config):
    """Issue #53: JAX's max_size scans the whole directory on every write.

    At 75,000 entries that scan measured 2-8 s per new program, so the cap
    is enforced by :func:`cadjoint.cache.prune_compilation_cache` instead.
    """
    monkeypatch.delenv("CADJOINT_NO_COMPILATION_CACHE", raising=False)
    jax.config.update("jax_compilation_cache_max_size", -1)
    cache.enable_compilation_cache(tmp_path / "jaxcache")
    assert jax.config.jax_compilation_cache_max_size == -1


def _entry(directory, name, size, age_seconds):
    import os
    import time

    path = directory / f"{name}-cache"
    path.write_bytes(b"x" * size)
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def test_prune_removes_the_oldest_entries_down_to_the_budget(tmp_path):
    oldest = _entry(tmp_path, "a", 400, age_seconds=300)
    middle = _entry(tmp_path, "b", 400, age_seconds=200)
    newest = _entry(tmp_path, "c", 400, age_seconds=100)
    assert cache.prune_compilation_cache(tmp_path, max_bytes=900, force=True) == 1
    assert not oldest.exists()
    assert middle.exists() and newest.exists()


def test_prune_clears_the_atime_files_jax_eviction_left(tmp_path):
    kept = _entry(tmp_path, "a", 10, age_seconds=10)
    (tmp_path / "a-atime").write_bytes(b"\0" * 8)
    assert cache.prune_compilation_cache(tmp_path, max_bytes=10**6, force=True) == 0
    assert kept.exists()
    assert not (tmp_path / "a-atime").exists()


def test_prune_runs_at_most_once_per_interval_unless_forced(tmp_path):
    _entry(tmp_path, "a", 400, age_seconds=300)
    _entry(tmp_path, "b", 400, age_seconds=200)
    assert cache.prune_compilation_cache(tmp_path, max_bytes=10**6) == 0  # first run stamps
    assert cache.prune_compilation_cache(tmp_path, max_bytes=500) == 0  # within the interval
    assert cache.prune_compilation_cache(tmp_path, max_bytes=500, force=True) == 1


def test_prune_of_a_missing_directory_is_a_no_op(tmp_path):
    assert cache.prune_compilation_cache(tmp_path / "absent", force=True) == 0
