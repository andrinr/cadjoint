"""Scope jax's x64 mode to the FEM suite.

jax-fem requires float64, but flipping ``jax_enable_x64`` at module import
time poisons every test that runs later in the same process (float32
suites start seeing float64 arrays).  This fixture turns x64 on for the
FEM tests and restores the previous setting afterwards.
"""

from __future__ import annotations

import jax
import pytest


@pytest.fixture(autouse=True, scope="package")
def _fem_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)
