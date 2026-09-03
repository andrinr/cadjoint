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
    target = tmp_path / "jaxcache"
    assert cache.enable_compilation_cache(target) == target
    assert target.is_dir()
    assert jax.config.jax_compilation_cache_dir == str(target)
    # Every program must be eligible: a scene is hundreds of sub-second
    # compilations, so a per-program time floor would cache none of them.
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0


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
