"""How an SDF tree turns itself into JAX operations.

Two consumers trace the same tree and want opposite things from it.

*XLA* wants **structure**: an ``(N, 2)`` vertex array reduced once, a pattern
whose child is traced once under :func:`jax.vmap`, a shared subtree emitted as
one ``func.func``.  Program size then scales with the *shapes* in the tree
rather than with its unrolled leaf count, which is what makes a 168-vertex
profile cost the same to compile as a 12-vertex one.

*WGSL* wants **scalars**: the shader backend maps StableHLO tensors onto
``f32``/``vec2``–``vec4``/``mat2``–``mat4`` and nothing else, so any array with
more than four rows — every stacked vertex loop, every batched pattern
instance — is untranslatable.  Under :func:`scalar_lowering` the same tree
re-emits itself the way it always did: one op chain per vertex, one child copy
per pattern instance.

The flag is a :class:`~contextvars.ContextVar`, so it is per-thread and
per-async-task and never leaks out of the ``with`` block that set it.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar

__all__ = ["scalar_lowering", "vectorized_lowering", "is_scalar_lowering"]

_SCALAR: ContextVar[bool] = ContextVar("cadjoint_sdf_scalar_lowering", default=False)


def is_scalar_lowering() -> bool:
    """True while the tree must emit one scalar op chain per element.

    Returns:
        Whether a :func:`scalar_lowering` block is active.
    """
    return _SCALAR.get()


@contextlib.contextmanager
def scalar_lowering():
    """Emit unrolled, scalar-only operations inside this block.

    Required by the WGSL backend, whose type mapping stops at ``vec4``.

    Yields:
        None.
    """
    token = _SCALAR.set(True)
    try:
        yield
    finally:
        _SCALAR.reset(token)


@contextlib.contextmanager
def vectorized_lowering():
    """Emit array-shaped operations inside this block (the default).

    Yields:
        None.
    """
    token = _SCALAR.set(False)
    try:
        yield
    finally:
        _SCALAR.reset(token)
