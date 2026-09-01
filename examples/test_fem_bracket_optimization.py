"""Smoke test for the flagship optimization example.

Lives next to the example so a bare ``pytest`` run collects it without any
configuration.  Two Adam steps at low resolution must descend and the
adjoint gradient must agree with finite differences — the cheapest signal
that the whole parameters -> SDF -> mesh -> FEM -> objective chain still
holds together.
"""

from __future__ import annotations

import pytest


def test_optimization_smoke():
    jax = pytest.importorskip("jax")
    pytest.importorskip("jax_fem")
    pytest.importorskip("optax")
    # The example enables x64 at import time (correct for a standalone
    # script); the rest of the suite must not inherit that.
    previous = jax.config.jax_enable_x64
    try:
        from fem_bracket_optimization import run_smoke

        run_smoke()
    finally:
        jax.config.update("jax_enable_x64", previous)
