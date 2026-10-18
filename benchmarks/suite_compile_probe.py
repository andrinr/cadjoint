"""Where the test suite's XLA seconds go: every compile, its program and its call site.

A pytest plugin and its report in one file.  As a plugin it wraps
``jax._src.compiler.compile_or_get_cached`` and writes one JSON line per
compilation — program name, seconds, HLO bytes, a digest of the module and
the innermost repository frames that asked for it — to one file per
process.  As a script it reads those files back and says what the seconds
were spent on::

    rm -rf /tmp/probe
    PYTHONPATH=.:benchmarks CADJOINT_NO_COMPILATION_CACHE=1 COMPILE_PROBE_DIR=/tmp/probe \\
        pytest tests -m "not slow" -n 4 -p suite_compile_probe -p no:cacheprovider
    python benchmarks/suite_compile_probe.py /tmp/probe

Three totals matter and they answer different questions
(``research/performance.md`` §19):

* **all** — what a cache-off run pays;
* **in-worker repeats** — compiles of a module the same process had already
  compiled (a fresh ``jax.jit`` per call, a cache keyed on object identity);
  a persistent cache would serve these from disk, nothing else does;
* **globally unique** — what a cold run with an empty shared persistent
  cache pays, the floor that only changing the programs can lower.

``PROBE_DEPTH`` (default 4) sets how many repository frames a site keeps.
Programs under 3 KB of HLO are counted as one-op eager programs.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import sys
import time
import traceback

_SMALL_BYTES = 3000
_STATE: dict = {"test": "<collect>", "out": None}


def _out():
    if _STATE["out"] is None:
        directory = os.environ["COMPILE_PROBE_DIR"]
        os.makedirs(directory, exist_ok=True)
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        path = os.path.join(directory, f"{worker}-{os.getpid()}.jsonl")
        _STATE["out"] = open(path, "a", buffering=1)  # noqa: SIM115 - lives for the process
    return _STATE["out"]


def _install() -> None:
    from jax._src import compiler

    original = compiler.compile_or_get_cached
    depth = int(os.environ.get("PROBE_DEPTH", "4"))

    def probed(backend, computation, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original(backend, computation, *args, **kwargs)
        finally:
            seconds = time.perf_counter() - start
            name = str(computation.operation.attributes["sym_name"]).strip('"')
            try:
                blob = computation.operation.get_asm(binary=True)
                size, digest = len(blob), hashlib.sha1(blob).hexdigest()[:16]
            except Exception:  # noqa: BLE001 - size is informational
                size, digest = -1, ""
            frames = [
                frame
                for frame in traceback.extract_stack()[:-1]
                if "/site-packages/" not in frame.filename and __file__ not in frame.filename
            ]
            site = " < ".join(
                f"{'/'.join(frame.filename.split('/')[-2:])}:{frame.lineno}"
                for frame in frames[-1 : -depth - 1 : -1]
            )
            record = {"t": _STATE["test"], "n": name, "s": seconds, "b": size, "h": digest}
            record["site"] = site
            _out().write(json.dumps(record) + "\n")

    compiler.compile_or_get_cached = probed


if "COMPILE_PROBE_DIR" in os.environ:
    _install()


def pytest_runtest_setup(item):
    _STATE["test"] = item.nodeid


def report(directory: str, top: int = 25) -> None:
    """Print the three totals and the sites that own them."""
    total = repeats = unique = small = 0.0
    seen_anywhere: set[str] = set()
    by_site = collections.defaultdict(lambda: [0, 0.0, 0, 0.0])
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        seen_here: set[str] = set()
        with open(path) as lines:
            records = [json.loads(line) for line in lines]
        for r in records:
            site = r["site"].split(" < ")[0]
            entry = by_site[site]
            entry[0] += 1
            entry[1] += r["s"]
            total += r["s"]
            small += r["s"] if 0 <= r["b"] < _SMALL_BYTES else 0.0
            if r["h"] in seen_here:
                repeats += r["s"]
                entry[2] += 1
                entry[3] += r["s"]
            seen_here.add(r["h"])
            if r["h"] not in seen_anywhere:
                unique += r["s"]
                seen_anywhere.add(r["h"])
    print(f"XLA seconds, all compiles      {total:8.1f}")
    print(f"  of which one-op (<3 KB HLO)  {small:8.1f}")
    print(f"  in-worker identical repeats  {repeats:8.1f}")
    print(f"  globally unique programs     {unique:8.1f}  ({len(seen_anywhere)} modules)")
    print(f"\n{'seconds':>8} {'compiles':>8} {'repeats':>8} {'rep. s':>7}  site")
    for site, (n, s, rn, rs) in sorted(by_site.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"{s:8.1f} {n:8d} {rn:8d} {rs:7.1f}  {site}")


if __name__ == "__main__":
    report(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25)
