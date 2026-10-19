"""Scope jax's x64 mode to the flow suite.

The lattice Boltzmann march runs thousands of steps and its convergence is
measured against tolerances near 1e-9, which float32 cannot represent as a
*relative* change -- in single precision the residual floors out around
1e-7 and the fixed point is never reached.  The same reasoning as
``tests/fem/conftest.py``, and the same care: flipping ``jax_enable_x64``
at import time would poison every float32 suite that runs later in the
process, so it is a package-scoped fixture that restores what it found.
"""

from __future__ import annotations

import pytest

from tests.helpers import x64


@pytest.fixture(autouse=True, scope="package")
def _flow_x64():
    with x64():
        yield
