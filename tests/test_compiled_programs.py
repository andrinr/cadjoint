"""`cadjoint._compiled.call_jit`: a fresh jit that does not recompile a program it has seen.

The contract has three parts, and each is a way the helper could go wrong:
an identical program must be compiled once; a program that differs only in a
baked constant — the case a fresh-per-call jit exists to get right — must
not be served the other's executable; and inside a transformation it must
behave exactly like ``jax.jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint import _compiled


@pytest.fixture
def compiles(monkeypatch):
    """Count XLA compilations (and persistent-cache reads) from here on."""
    from jax._src import compiler

    seen = []
    original = compiler.compile_or_get_cached

    def counted(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(compiler, "compile_or_get_cached", counted)
    _compiled._EXECUTABLES.clear()
    return seen


def test_an_identical_program_is_compiled_once(compiles):
    x = jnp.arange(7.0)

    def scaled(c):
        return lambda p: jnp.sin(p) * c

    first = _compiled.call_jit(scaled(2.0), x)
    before = len(compiles)
    again = _compiled.call_jit(scaled(2.0), x)  # a new function, the same program
    assert len(compiles) == before, "the second call must reuse the executable"
    np.testing.assert_array_equal(np.asarray(first), np.asarray(again))


def test_a_different_constant_is_a_different_program(compiles):
    x = jnp.arange(4.0)
    table = np.random.default_rng(0).random(50_000).astype(np.float32)
    nudged = table.copy()
    nudged[-1] += 1.0

    a = _compiled.call_jit(lambda p: p[0] + table, x)
    b = _compiled.call_jit(lambda p: p[0] + nudged, x)
    assert float(a[-1]) != float(b[-1])
    np.testing.assert_array_equal(np.asarray(b), np.asarray(x[0] + nudged))


def test_inside_a_transformation_it_is_plain_jit():
    x = jnp.arange(5.0)
    gradient = jax.grad(lambda t: _compiled.call_jit(lambda p: jnp.sum(p * t), x))(1.0)
    assert float(gradient) == float(jnp.sum(x))
    batched = jax.vmap(lambda t: _compiled.call_jit(lambda p: p * t, x))(jnp.arange(3.0))
    np.testing.assert_array_equal(np.asarray(batched), np.outer(np.arange(3.0), np.arange(5.0)))
