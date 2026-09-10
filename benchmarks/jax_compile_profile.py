"""Where a playground request's JAX time goes: trace, lower, compile, run.

Every worker mode (``compile``, ``mesh``, ``mesh_inspect``, ``simulate``,
``optimize``) is driven in-process on one scene while three JAX internals
are wrapped with timers:

* ``pjit._create_pjit_jaxpr`` — Python tracing to a jaxpr (misses only; the
  tracing cache returns hits without entering the function);
* ``pxla._cached_lowering_to_hlo`` — jaxpr to StableHLO;
* ``compiler.compile_or_get_cached`` — StableHLO to an executable, either
  by XLA or by a persistent-cache read (the two are told apart by JAX's
  own ``/jax/compilation_cache/cache_hits`` events).

Eager (op-by-op) JAX takes the same three steps per primitive, so the
program count here is the number of executables the request dispatched,
and the remainder ``wall - trace - lower - compile`` is Python, eager
dispatch and everything outside JAX (TetGen, PETSc, JSON).

Run one mode in a fresh process, twice, against one cache directory:

    CADJOINT_CACHE_DIR=/tmp/cc python benchmarks/jax_compile_profile.py \\
        --scene scenes/starter.py --mode mesh --json cold.json
    CADJOINT_CACHE_DIR=/tmp/cc python benchmarks/jax_compile_profile.py \\
        --scene scenes/starter.py --mode mesh --json warm.json

The first run is the cold cliff (XLA compiles every program); the second
is what a user sees after the server's warm-up.  ``--repeat 2`` runs the
mode twice in one process, which is what a persistent worker would see.
``--public`` unregisters the private tier for the run, which is how the
public paths — the lattice edge overlay above all — are profiled on a
machine that has ``diff-brep`` installed.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Hooks. Installed before cadjoint imports so every program is counted.
# --------------------------------------------------------------------------
import jax  # noqa: E402
import jax.monitoring  # noqa: E402
from jax._src import compiler as _compiler  # noqa: E402
from jax._src import pjit as _pjit  # noqa: E402
from jax._src.interpreters import pxla as _pxla  # noqa: E402

PROGRAMS: list[dict[str, Any]] = []
_COUNTS = collections.Counter()
_CURRENT: dict[str, Any] = {}


def _on_event(name: str, **_: Any) -> None:
    if name.startswith("/jax/compilation_cache/"):
        _COUNTS[name.rsplit("/", 1)[1]] += 1
        if name.endswith("cache_hits"):
            _CURRENT["hit"] = True


jax.monitoring.register_event_listener(_on_event)

_orig_trace = _pjit._create_pjit_jaxpr
_orig_lower = _pxla._cached_lowering_to_hlo
_orig_compile = _compiler.compile_or_get_cached

TRACE = {"seconds": 0.0, "calls": 0}
LOWER = {"seconds": 0.0, "calls": 0}
TRACES: list[dict[str, Any]] = []


def _trace(fun, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        return _orig_trace(fun, *args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        TRACE["seconds"] += dt
        TRACE["calls"] += 1
        name = getattr(fun, "__name__", None) or getattr(getattr(fun, "f", None), "__name__", "?")
        TRACES.append({"name": str(name), "seconds": dt})


def _lower(*args, **kwargs):
    t0 = time.perf_counter()
    try:
        return _orig_lower(*args, **kwargs)
    finally:
        LOWER["seconds"] += time.perf_counter() - t0
        LOWER["calls"] += 1


def _compile(backend, computation, *args, **kwargs):
    sym = computation.operation.attributes["sym_name"]
    name = str(sym).strip('"')
    _CURRENT.clear()
    t0 = time.perf_counter()
    try:
        return _orig_compile(backend, computation, *args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        text_bytes = None
        with contextlib.suppress(Exception):  # size is informational
            text_bytes = len(computation.operation.get_asm(binary=True))
        PROGRAMS.append(
            {
                "name": name,
                "seconds": dt,
                "cache_hit": bool(_CURRENT.get("hit")),
                "hlo_bytes": text_bytes,
            }
        )


_pjit._create_pjit_jaxpr = _trace
_pxla._cached_lowering_to_hlo = _lower
_compiler.compile_or_get_cached = _compile

# --------------------------------------------------------------------------
# Driving the worker modes.
# --------------------------------------------------------------------------


def _run_mode(mode: str, source: str, names: dict[str, str], steps: int) -> dict[str, Any]:
    from cadjoint.viewer.worker import main as worker

    if mode == "compile":
        return worker._compile_source(source)
    if mode == "mesh":
        return worker._mesh_source(source)
    if mode == "mesh_inspect":
        return worker._mesh_inspect_source({"source": source, "name": names["mesh"]})
    if mode == "simulate":
        return worker._simulate_source({"source": source, "name": names["study"]})
    if mode == "optimize":
        from cadjoint.viewer.worker.optimize import _optimize_source

        return _optimize_source({"source": source, "name": names["optimization"], "steps": steps})
    raise ValueError(f"unknown mode {mode!r}")


def _declared_names(source: str) -> dict[str, str]:
    """First declared SimMesh / study / optimization name, by a cheap exec."""
    from cadjoint.viewer.worker.scene import _execute_scene

    namespace = _execute_scene(source)
    out = {}
    meshes = namespace.get("__sim_meshes__") or []
    studies = namespace.get("__studies__") or []
    optimizations = namespace.get("__optimizations__") or []
    if meshes:
        out["mesh"] = meshes[0].name
    if studies:
        out["study"] = studies[0].name
    if optimizations:
        out["optimization"] = optimizations[0].name
    return out


def _family(name: str) -> str:
    """Group program names by the function they came from.

    Eager primitives are named after the primitive (``jit(sin)``,
    ``jit(_where)``); jitted functions after the Python function.  Strip
    the ``jit(`` wrapper and any trailing counter.
    """
    n = name
    if n.startswith("jit(") and n.endswith(")"):
        n = n[4:-1]
    return n.rstrip("0123456789_")


def _report(label: str, wall: float, before: int) -> dict[str, Any]:
    programs = PROGRAMS[before:]
    compile_s = sum(p["seconds"] for p in programs if not p["cache_hit"])
    hit_s = sum(p["seconds"] for p in programs if p["cache_hit"])
    families = collections.defaultdict(lambda: {"count": 0, "seconds": 0.0, "hits": 0})
    for p in programs:
        f = families[_family(p["name"])]
        f["count"] += 1
        f["seconds"] += p["seconds"]
        f["hits"] += int(p["cache_hit"])
    top = sorted(families.items(), key=lambda kv: -kv[1]["seconds"])[:15]
    biggest = sorted(programs, key=lambda p: -p["seconds"])[:10]
    traces = collections.defaultdict(lambda: {"count": 0, "seconds": 0.0})
    for t in TRACES:
        traces[t["name"]]["count"] += 1
        traces[t["name"]]["seconds"] += t["seconds"]
    top_traces = sorted(traces.items(), key=lambda kv: -kv[1]["seconds"])[:10]
    return {
        "label": label,
        "wall_seconds": wall,
        "trace_seconds": TRACE["seconds"],
        "trace_calls": TRACE["calls"],
        "lower_seconds": LOWER["seconds"],
        "lower_calls": LOWER["calls"],
        "compile_seconds": compile_s,
        "cache_read_seconds": hit_s,
        "programs": len(programs),
        "programs_compiled": sum(1 for p in programs if not p["cache_hit"]),
        "programs_from_cache": sum(1 for p in programs if p["cache_hit"]),
        "distinct_programs": len({p["name"] for p in programs}),
        "hlo_bytes": sum(p["hlo_bytes"] or 0 for p in programs),
        "other_seconds": wall - TRACE["seconds"] - LOWER["seconds"] - compile_s - hit_s,
        "families": [{"family": k, **v} for k, v in top],
        "biggest": biggest,
        "traces": [{"function": k, **v} for k, v in top_traces],
    }


def _print(r: dict[str, Any]) -> None:
    def row(k: str, v: Any) -> None:
        print(f"  {k:<22} {v}")

    print(f"== {r['label']}")
    row("wall", f"{r['wall_seconds']:.2f} s")
    row("trace (jaxpr)", f"{r['trace_seconds']:.2f} s  ({r['trace_calls']} misses)")
    row("lower (StableHLO)", f"{r['lower_seconds']:.2f} s  ({r['lower_calls']} calls)")
    row("compile (XLA)", f"{r['compile_seconds']:.2f} s  ({r['programs_compiled']} programs)")
    row("cache reads", f"{r['cache_read_seconds']:.2f} s  ({r['programs_from_cache']} programs)")
    row("other (python/eager)", f"{r['other_seconds']:.2f} s")
    row("programs dispatched", f"{r['programs']} ({r['distinct_programs']} distinct)")
    row("HLO bytes", f"{r['hlo_bytes'] / 1e6:.2f} MB")
    print("  top families by compile+read time:")
    for f in r["families"]:
        print(
            f"    {f['family']:<40} {f['count']:>5} x  {f['seconds']:.2f} s  ({f['hits']} cached)"
        )
    print("  biggest single programs:")
    for p in r["biggest"]:
        hit = "cache" if p["cache_hit"] else "xla"
        print(f"    {p['name']:<40} {p['seconds']:.2f} s  {hit}")
    print("  top traced functions:")
    for t in r["traces"]:
        print(f"    {t['function']:<40} {t['count']:>5} x  {t['seconds']:.2f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", default="scenes/starter.py")
    ap.add_argument(
        "--mode",
        default="compile",
        choices=["compile", "mesh", "mesh_inspect", "simulate", "optimize"],
    )
    ap.add_argument("--steps", type=int, default=2, help="optimize steps")
    ap.add_argument("--repeat", type=int, default=1, help="run the mode N times in-process")
    ap.add_argument(
        "--public",
        action="store_true",
        help="unregister the private tier's plugin kinds for the run, so the public "
        "paths (the lattice edge overlay, the faceted exporter) are what is profiled "
        "even where diff-brep is installed",
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from cadjoint.cache import enable_compilation_cache

    cache_dir = enable_compilation_cache()
    source = Path(args.scene).read_text()

    t0 = time.perf_counter()
    names = _declared_names(source)
    exec_s = time.perf_counter() - t0

    from cadjoint import tier

    results = []
    with tier.absent() if args.public else contextlib.nullcontext():
        for i in range(args.repeat):
            before = len(PROGRAMS)
            TRACES.clear()
            TRACE["seconds"] = TRACE["calls"] = 0
            LOWER["seconds"] = LOWER["calls"] = 0
            t0 = time.perf_counter()
            out = _run_mode(args.mode, source, names, args.steps)
            wall = time.perf_counter() - t0
            if isinstance(out, dict) and out.get("ok") is False:
                print(out.get("error"), file=sys.stderr)
                sys.exit(1)
            r = _report(f"{args.mode} run {i + 1}/{args.repeat}", wall, before)
            r["scene"] = args.scene
            r["cache_dir"] = str(cache_dir)
            r["scene_exec_seconds"] = exec_s
            r["tier"] = "public" if args.public else "installed"
            r["jax_cache_events"] = dict(_COUNTS)
            r["jax_version"] = jax.__version__
            results.append(r)
            _print(r)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    os.environ.setdefault("CADJOINT_WARM_START", "0")
    main()
