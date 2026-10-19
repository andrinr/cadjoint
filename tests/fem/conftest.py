"""Scope jax's x64 mode to the FEM suite.

jax-fem requires float64, but flipping ``jax_enable_x64`` at module import
time poisons every test that runs later in the same process (float32
suites start seeing float64 arrays).  This fixture turns x64 on for the
FEM tests and restores the previous setting afterwards.
"""

from __future__ import annotations

import pytest

from tests.helpers import x64


@pytest.fixture(autouse=True, scope="package")
def _fem_x64():
    with x64():
        yield
