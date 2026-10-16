"""The package metadata against what the code and its dependencies need."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _floor(specifier: str) -> tuple[int, int]:
    """The ``(major, minor)`` of a ``>=X.Y`` lower bound."""
    match = re.search(r">=\s*(\d+)\.(\d+)", specifier)
    assert match, f"no >= lower bound in {specifier!r}"
    return int(match.group(1)), int(match.group(2))


def _declared() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["requires-python"]


def test_requires_python_covers_the_jax_floor():
    """pip must refuse a Python that installed cadjoint could not import (#51).

    The floor was ``>=3.9`` while the code uses 3.10 syntax at import time and
    current JAX needs 3.11: pip installed the package on 3.9 and ``import
    cadjoint`` then raised.
    """
    for dependency in ("jax", "jaxlib"):
        needed = importlib.metadata.metadata(dependency)["Requires-Python"]
        assert _floor(_declared()) >= _floor(needed), (dependency, needed)


def test_lock_matches_requires_python():
    """uv.lock is resolved for the declared range, not a stale wider one."""
    lock = (ROOT / "uv.lock").read_text()
    match = re.search(r'^requires-python = "([^"]+)"', lock, flags=re.MULTILINE)
    assert match, "uv.lock has no requires-python"
    assert _floor(match.group(1)) == _floor(_declared())
