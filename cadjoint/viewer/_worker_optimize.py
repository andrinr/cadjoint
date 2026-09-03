"""The worker's ``mode="optimize"``: run a declared Optimization.

Descends one :class:`cadjoint.optimize.Optimization` the program declares,
streams a progress line per step to the worker's real stdout (the pipe the
playground server tails), and writes the optimized parameter values back
into the program text through the patch machinery, so the response's
``source`` is a patched program exactly like a ``/patch`` response.  A
study-backed run additionally packages the optimized design's solve through
:mod:`cadjoint.viewer._worker_payloads`.

The step caps that bound one HTTP-sized run live here.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from typing import Any

from cadjoint.viewer._worker_payloads import _study_payload
from cadjoint.viewer._worker_scene import (
    _FEM_UNAVAILABLE_MESSAGE,
    _execute_scene,
    _named_optimization,
)

# The server validates requested step counts, but the declaration itself may
# ask for more than one HTTP-bounded run should pay for; the worker caps both.
OPTIMIZE_STEP_LIMIT = 200
# Study-backed runs solve a FEM problem (plus its adjoint) per step.  Measured
# per-step wall clock at the declared scene resolutions: starter thermal
# ~3.7 s steady (~17 s on JIT/topology-refreeze steps), bracket elastic
# compliance ~5-7 s steady (~20 s on refreeze), plus ~8 s for the final
# fresh-mesh evaluation — so 30 steps stays under the server's 300-second
# /api/optimize budget with headroom even for the heavier elastic runs.
STUDY_OPTIMIZE_STEP_LIMIT = 30


def _run_optimization(
    source: str,
    namespace: dict[str, Any],
    request: dict[str, Any],
    progress_out: Any = None,
) -> dict[str, Any]:
    """Run one declared optimization by name and patch its result into source.

    The optimizer is a patch layer: the optimized free-parameter values are
    written back into the program text through the same exact-repr patch
    machinery the viewer's other edits use, and the client adopts the
    returned ``source`` and recompiles — code parity, like ``/patch``.

    A study-backed optimization additionally carries a ``simulate`` block —
    the optimized design solved on a freshly extracted mesh and packaged
    through the exact :func:`_study_payload` shapes ``/api/simulate``
    responses use (``field``/``mesh``/``result``/``mesh_info``) — so the
    frontend displays the optimized part with its field through the
    existing simulation pipeline.  Study-backed steps are capped at
    ``STUDY_OPTIMIZE_STEP_LIMIT`` (a FEM solve plus adjoint per step);
    objective-form runs keep the ``OPTIMIZE_STEP_LIMIT`` cap.

    ``progress_out`` (the worker's real stdout pipe) receives one flushed
    NDJSON line per optimizer step as it completes —
    ``{"event", "step", "steps", "objective", "grad_norm", "elapsed"}``
    with ``step`` counting completed evaluations (1-based) — which the
    playground server relays to the client as chunked NDJSON.
    """
    import time

    from cadjoint.viewer._patch import set_parameter_values

    optimization = _named_optimization(namespace["__optimizations__"], request.get("name"))
    steps = request.get("steps")
    steps = optimization.steps if steps is None else int(steps)
    study_backed = optimization.study is not None
    run_steps = min(steps, STUDY_OPTIMIZE_STEP_LIMIT if study_backed else OPTIMIZE_STEP_LIMIT)
    started = time.monotonic()

    def emit_progress(record: dict[str, float]) -> None:
        if progress_out is None:
            return
        print(
            json.dumps(
                {
                    "event": "progress",
                    "step": int(record["step"]) + 1,
                    "steps": run_steps,
                    "objective": record["objective"],
                    "grad_norm": record["grad_norm"],
                    "elapsed": round(time.monotonic() - started, 3),
                }
            ),
            file=progress_out,
            flush=True,
        )

    if study_backed:
        try:
            import jax_fem  # noqa: F401
        except ImportError:
            return {
                "ok": False,
                "error_kind": "fem_unavailable",
                "error": _FEM_UNAVAILABLE_MESSAGE,
            }
        run = optimization.run(steps=run_steps, callback=emit_progress, scene=namespace["scene"])
    else:
        run = optimization.run(steps=run_steps, callback=emit_progress)
    patched = set_parameter_values(source, run.parameters)
    payload = {
        "ok": True,
        "kind": "optimize",
        "name": optimization.name,
        "method": run.method,
        "steps": run.steps,
        "source": patched,
        "history": run.history,
        "trajectory": run.trajectory,
        "parameters": run.parameters,
        "initial": run.initial,
    }
    if study_backed:
        import jax.numpy as jnp

        from cadjoint.extraction import apply_parameters

        # The run restored the target's original parameter values; put the
        # optimized ones back so the packaged payload (and any mesh rebuild
        # inspection triggers) describes the final design consistently —
        # matching the patched source the client is about to adopt.
        scene = namespace["scene"]
        apply_parameters(optimization._study_target(scene), run.parameters)
        sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
        packaged = _study_payload(optimization.study, run.result, sdf)
        payload["simulate"] = {
            key: packaged[key] for key in ("field", "mesh", "result", "mesh_info")
        }
    return payload


def _optimize_source(request: dict[str, Any]) -> dict[str, Any]:
    """Run the optimize mode: exec scene -> declared optimization -> patch.

    Per-step progress lines stream to the worker's REAL stdout (the pipe
    the playground server tails) while the user program's own prints stay
    captured into ``output`` — the stdout redirect below swaps
    ``sys.stdout``, so the reference grabbed first keeps writing to the
    pipe.
    """
    progress_out = sys.stdout
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(request["source"])
        result = _run_optimization(request["source"], namespace, request, progress_out)
    result["output"] = captured.getvalue()[-8_000:]
    return result
