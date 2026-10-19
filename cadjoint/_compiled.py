"""Run a freshly built ``jax.jit`` without recompiling a program already compiled.

Several kernels jit a *new* function object on every call, on purpose: they
close over opaque fields that may capture an outer trace's tracers, or a
scene whose parameters are mutated between calls, so a jit cached on the
function would hold a dead tracer or bake in a stale constant
(:func:`cadjoint.zeroset.project.project`, the constraint solver's scanned
Newton run).  JAX keys its in-process executable cache on the function
object, so every such call compiles from scratch — including the calls
whose program is byte for byte one compiled a moment ago.

In a cold test run about half of ``project``'s programs and four fifths of
``_newton_scan``'s are identical to one the same process already compiled.
The persistent compilation cache would serve those as disk reads, but a
process without it — a library user, a test run with the cache off — pays
XLA every time.

:func:`call_jit` keeps the fresh-function contract and shares the
*executable*: it lowers the call, and looks the result up by the module's
bytecode with debug information stripped, which is the identity JAX's own
persistent cache uses (``jax._src.cache_key``).  Closed-over constants are
inlined in the module (``jax_use_simplified_jaxpr_constants`` is off by
default; with it on, the helper steps aside), so two calls share a program
only when they would have compiled the same one, and the result cannot
change by a bit.

Inside any trace — a ``grad``, ``vmap`` or outer ``jit`` — it is plain
``jax.jit(fun)(*args)``: the fields may then close over that trace's
tracers, which a separately lowered program cannot see, and the outer
transformation compiles the whole thing anyway.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

__all__ = ["call_jit"]

#: Compiled executables by module digest, newest last.  A working set: the
#: repeats this exists for are a few calls apart.
_EXECUTABLES: OrderedDict[str, Any] = OrderedDict()
_EXECUTABLES_LIMIT = 32


def call_jit(fun: Callable[..., Any], *args: Any) -> Any:
    """``jax.jit(fun)(*args)``, reusing an identical program compiled earlier.

    Args:
        fun: A function of array arguments; may close over anything.
        *args: Its positional arguments (array-likes).

    Returns:
        What ``jax.jit(fun)(*args)`` returns.
    """
    import jax
    from jax._src import core

    jitted = jax.jit(fun)
    if not core.trace_state_clean() or jax.config.jax_use_simplified_jaxpr_constants:
        return jitted(*args)
    lowered = jitted.lower(*args)
    key = _digest(lowered)
    compiled = _EXECUTABLES.get(key)
    if compiled is None:
        compiled = lowered.compile()
        _EXECUTABLES[key] = compiled
        while len(_EXECUTABLES) > _EXECUTABLES_LIMIT:
            _EXECUTABLES.popitem(last=False)
    else:
        _EXECUTABLES.move_to_end(key)
    return compiled(*args)


def _digest(lowered: Any) -> str:
    """SHA-256 of the lowered module's bytecode, debug locations stripped.

    Locations name the Python source a program was traced from; two calls
    from different lines that lower to the same arithmetic are the same
    program.
    """
    import io

    from jax._src.lib.mlir import passmanager

    module = lowered.compiler_ir("stablehlo")
    with module.context:
        stripped = module.operation.clone()
        passmanager.PassManager.parse("builtin.module(strip-debuginfo)").run(stripped)
        out = io.BytesIO()
        stripped.write_bytecode(file=out)
    return hashlib.sha256(out.getvalue()).hexdigest()
