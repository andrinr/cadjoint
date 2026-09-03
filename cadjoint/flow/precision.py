"""Double precision, scoped to the solve rather than to the process.

The flow solve genuinely needs float64.  Its convergence is measured as a
*relative* change near 1e-12, which float32 cannot represent -- in single
precision the momentum residual floors out around 1e-7, the energy solve's
GMRES stalls, and a gradient comes back ``NaN``.  None of that is in
question.

What is in question is who turns it on.  ``jax_enable_x64`` is a **process
global**: flipping it at a scene's module scope means every array the
process makes afterwards is float64, including the ones the WGSL backend
has to emit for a shader -- and WebGPU has no ``f64``, so the viewer's
compile worker fails with a ``KeyError`` on a scene that merely *declares* a
flow study.  A scene that cannot be opened is a broken scene, whatever its
physics does.

So the solve enables double precision and puts it back, exactly as
:func:`cadjoint.fem.backends._x64_scope` does for jax-fem.  The scene file
then declares geometry and a study like every other scene, and stays
float32 for the shader, the mesh and the overlays.

**A scope covers a forward solve, and cannot cover a gradient.**  This is
the same boundary :func:`cadjoint.fem.backends._x64_scope` documents, and it
is worth stating in full because it decides who has to do what.
``jax.grad`` runs the transposed computation *after* the traced forward call
has returned, so by then this scope has exited -- and the top-level backward
pass then has to materialise the float64 intermediates the forward built
while the process is back in single precision, which it refuses to do
("dtype=float32 and shard dtype=float64").  Wrapping pieces of the backward
does not help: the failure is in JAX's own cotangent bookkeeping, not in any
one operator.

So the contract is:

* **Forward solves scope themselves.**  ``FlowStudy.solve`` enters here, so
  the viewer -- which compiles, renders and previews but never
  differentiates -- is unaffected, and a scene may declare a flow study
  without any precision ceremony.
* **Callers who differentiate enable x64 for their whole process**, exactly
  as the FEM path requires: ``tests/flow/conftest.py`` does it package-wide,
  and ``scenes/duct_sink.py``'s ``main`` wraps its own body.

:func:`cadjoint.flow.steady._steady_bwd` enters the scope in its own right
anyway, so the adjoint's re-trace of one lattice step is consistent whichever
way it was reached.  Entering twice is harmless: this restores whatever it
found rather than assuming it found ``False``.

The ``flow_brinkman`` Tesseract is deliberately not changed to use this: it
runs in its own process whose only job is that solve, so a permanent flip
there costs nothing and there is no shader in it to break.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["double_precision"]


@contextmanager
def double_precision() -> Iterator[None]:
    """Enable jax's x64 mode for the duration, restoring the caller's setting.

    Nests safely: the previous value is read on the way in and written back
    on the way out, so an inner scope inside an outer one (or inside a test
    suite that already enabled x64 package-wide) leaves the setting exactly
    as it found it.

    Yields:
        Nothing; the effect is the process-wide flag while the block runs.
    """
    import jax

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)
