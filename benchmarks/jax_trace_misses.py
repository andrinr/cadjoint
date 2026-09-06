"""Why a worker retraces: the tracing-cache misses of one request, by site.

The persistent compilation cache removes XLA time (see
``jax_compile_profile.py``), and what it cannot touch is *tracing* — Python
work every process redoes.  A program that retraces is usually not retracing
for a mysterious reason: JAX will say which call site missed and why, and the
answer is nearly always one of two things.

* **A new function object.** A ``jax.jit`` defined inside the function that
  calls it is rebuilt per call, so its trace cache starts empty every time.
  Hoisting fixes it, unless the closure captures something whose identity
  changes, because JAX keys a closure's cache on identity.
* **A new input shape.** The same program called with ``f32[1024,3]`` and
  then ``f32[256,3]`` is two traces.  Padding the argument to bucketed
  capacities collapses them, and it is what lets the *persistent* cache pay
  across processes — the only mechanism that helps a disposable worker.

This script attributes the misses so the two are told apart rather than
guessed at, which is the whole reason it exists: on 2026-09-06 the private
tier's feature-edge extraction looked like a closure problem and was mostly
a shape one — 166 shape misses against 18 new-function ones, at four sites,
about 22 s of retracing on ``scenes/starter.py``.  Guessing would have sent
the fix to the wrong half.

    python benchmarks/jax_trace_misses.py --scene scenes/starter.py --mode mesh

``--mode graph`` narrows it to the private tier's B-rep extraction, which is
where a mesh request spends its time when that tier is installed.
"""

from __future__ import annotations

import argparse
import collections
import io
import logging
import re
import time
from contextlib import redirect_stdout
from pathlib import Path

#: `TRACING CACHE MISS at <file>:<line> (<name>) costing <ms> ms`
_MISS = re.compile(
    r"TRACING CACHE MISS at (?P<file>[^\s:]+):(?P<line>\d+)[^(]*\((?P<name>[^)]*)\)"
    r"\s*costing\s*(?P<ms>[\d.]+) ms"
)
#: The reason JAX gives, one of a small vocabulary.
_REASONS = (
    ("shape", "key with different input types"),
    ("new function", "never seen function"),
    ("other key", "all previously seen cache keys are different"),
)


def _classify(block: str) -> str:
    for label, needle in _REASONS:
        if needle in block:
            return label
    return "unattributed"


def _run(scene: Path, mode: str) -> str:
    """Run one request with miss explanations on, returning the log."""
    import jax

    jax.config.update("jax_explain_cache_misses", True)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("jax._src.pjit").addHandler(handler)
    logging.getLogger("jax._src.pjit").setLevel(logging.WARNING)

    from cadjoint.cache import enable_compilation_cache

    enable_compilation_cache()
    source = scene.read_text()

    started = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        if mode == "graph":
            from diff_brep.edges import _extract_graph

            from cadjoint.viewer._edge_overlay import _overlay_grid
            from cadjoint.viewer._worker_scene import _execute_scene

            _extract_graph(
                _execute_scene(source)["scene"], _overlay_grid(), blend_tolerance=None, steps=4
            )
        else:
            from cadjoint.viewer import _compile_worker as worker

            {"mesh": worker._mesh_source, "compile": worker._compile_source}[mode](source)
    print(f"{mode} on {scene.name}: {time.perf_counter() - started:.1f} s")
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", type=Path, default=Path("scenes/starter.py"))
    parser.add_argument("--mode", default="mesh", choices=["mesh", "compile", "graph"])
    parser.add_argument("--top", type=int, default=10)
    arguments = parser.parse_args()

    log = _run(arguments.scene, arguments.mode)
    blocks = log.split("TRACING CACHE MISS")
    sites: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    reasons: collections.Counter[str] = collections.Counter()
    for block in blocks[1:]:
        match = _MISS.search("TRACING CACHE MISS" + block)
        if match is None:
            continue
        where = f"{Path(match['file']).name}:{match['line']}"
        reason = _classify(block)
        reasons[reason] += 1
        sites[(where, match["name"].strip())].append(float(match["ms"]))

    total = sum(sum(times) for times in sites.values())
    print(f"\n{sum(reasons.values())} tracing-cache misses, {total / 1000:.1f} s of tracing\n")
    print("  by reason:")
    for reason, count in reasons.most_common():
        print(f"    {count:5d}  {reason}")
    print("\n  by site:")
    rows = sorted(sites.items(), key=lambda item: -sum(item[1]))
    for (where, name), times in rows[: arguments.top]:
        print(f"    {len(times):5d} x {sum(times) / 1000:6.2f} s  {where:28s} {name}")
    if not sites:
        print("    none — every jit hit its trace cache")


if __name__ == "__main__":
    main()
