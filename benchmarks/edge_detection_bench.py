"""Benchmark harness for stage 1 of the meshing pipeline (edge detection).

Measures wall time, throughput, and numerical fidelity of
``sample_grid`` -> ``find_crossing_edges`` -> ``edge_hermite_data`` on three
reference shapes, and (when scikit-image is installed) times
``skimage.measure.marching_cubes`` on the same sampled volumes for context.

Run from the repository root::

    python benchmarks/edge_detection_bench.py --resolutions 16 32 64
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cadjoint.meshing import GridSpec, edge_hermite_data, find_crossing_edges, sample_grid
from cadjoint.sdf.primitives import Box

try:
    from skimage import measure as _skimage_measure
except ImportError:  # pragma: no cover - optional context reference
    _skimage_measure = None

_BOX_HALF_EXTENTS = (0.4, 0.5, 0.6)
_UNION_RADIUS = 0.6
_UNION_OFFSET = 0.35


def _sphere_sdf(p: Array) -> Array:
    return jnp.linalg.norm(p) - 1.0


def _box_sdf(p: Array) -> Array:
    return Box.sdf(p, jnp.asarray(_BOX_HALF_EXTENTS))


def _union_sdf(p: Array) -> Array:
    left = jnp.linalg.norm(p - jnp.asarray([-_UNION_OFFSET, 0.0, 0.0])) - _UNION_RADIUS
    right = jnp.linalg.norm(p - jnp.asarray([_UNION_OFFSET, 0.0, 0.0])) - _UNION_RADIUS
    return jnp.minimum(left, right)


# Shape name -> (sdf, lower corner, extent).  Bounds pad the surface so it
# never touches the sampled volume's boundary.
_SHAPES: dict[str, tuple[Callable[[Array], Array], tuple[float, ...], tuple[float, ...]]] = {
    "sphere": (_sphere_sdf, (-1.3,) * 3, (2.6,) * 3),
    "box": (_box_sdf, (-0.8,) * 3, (1.6,) * 3),
    "union": (_union_sdf, (-1.2,) * 3, (2.4,) * 3),
}


def _best_of(fn: Callable[[], object], repeats: int) -> tuple[float, object]:
    """Run ``fn`` ``repeats`` times; return (best wall time in seconds, last result)."""
    best = float("inf")
    result: object = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - start)
    return best, result


def _sphere_gradient_error(grid: GridSpec, edges) -> float:
    """Max relative error of autodiff dt/dr against the analytic value r / (p . d)."""

    def t_of_radius(radius: Array) -> Array:
        return edge_hermite_data(lambda p: jnp.linalg.norm(p) - radius, grid, edges).t

    radius = jnp.asarray(1.0)
    # Forward-mode Jacobian: identical values to jax.jacobian (reverse mode)
    # but a single pass for a scalar input.
    dt_dr = np.asarray(jax.jacfwd(t_of_radius)(radius), dtype=np.float64)

    hermite = edge_hermite_data(lambda p: jnp.linalg.norm(p) - radius, grid, edges)
    points = np.asarray(hermite.points, dtype=np.float64)
    directions = np.eye(3)[np.asarray(edges.axis, dtype=np.int32)] * np.asarray(grid.spacing)
    analytic = float(radius) / np.sum(points * directions, axis=-1)
    return float(np.max(np.abs(dt_dr - analytic) / np.abs(analytic)))


def _run_case(name: str, resolution: int, repeats: int) -> dict[str, object]:
    sdf, bounds, size = _SHAPES[name]
    grid = GridSpec.from_bounds(bounds, size, resolution)

    start = time.perf_counter()
    values = sample_grid(sdf, grid)
    sample_cold = time.perf_counter() - start
    sample_time, values = _best_of(lambda: sample_grid(sdf, grid), repeats)

    find_time, edges = _best_of(lambda: find_crossing_edges(values), repeats)

    start = time.perf_counter()
    jax.block_until_ready(edge_hermite_data(sdf, grid, edges))
    hermite_cold = time.perf_counter() - start
    hermite_time, hermite = _best_of(lambda: edge_hermite_data(sdf, grid, edges), repeats)

    result: dict[str, object] = {
        "shape": name,
        "resolution": resolution,
        "edges": edges.count,
        "sample_cold_s": sample_cold,
        "sample_s": sample_time,
        "find_s": find_time,
        "hermite_cold_s": hermite_cold,
        "hermite_s": hermite_time,
        "edges_per_s": edges.count / hermite_time,
        "max_abs_residual": float(np.max(np.abs(np.asarray(hermite.values)))),
    }
    if name == "sphere":
        radii = np.linalg.norm(np.asarray(hermite.points, dtype=np.float64), axis=-1)
        result["max_sphere_error"] = float(np.max(np.abs(radii - 1.0)))
        result["max_grad_rel_err"] = _sphere_gradient_error(grid, edges)
    if _skimage_measure is not None:
        mc_time, _ = _best_of(
            lambda: _skimage_measure.marching_cubes(values, level=0.0, spacing=grid.spacing),
            repeats,
        )
        result["marching_cubes_s"] = mc_time
    return result


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:.2f}"


def _print_tables(results: list[dict[str, object]]) -> None:
    print("## Stage timings (best of repeats, warm; milliseconds)\n")
    print(
        "| shape | res | edges | sample (ms) | find (ms) | hermite (ms) "
        "| hermite compile (ms) | hermite edges/s |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row['shape']} | {row['resolution']} | {row['edges']} "
            f"| {_fmt_ms(row['sample_s'])} | {_fmt_ms(row['find_s'])} "
            f"| {_fmt_ms(row['hermite_s'])} | {_fmt_ms(row['hermite_cold_s'])} "
            f"| {row['edges_per_s']:.3g} |"
        )
    print(
        "\nCompile column is the first (cold) `edge_hermite_data` call, which includes "
        "jit tracing and compilation; warm columns time later calls."
    )

    print("\n## Fidelity\n")
    print("| shape | res | max abs residual | max sphere error | grad dt/dr max rel err |")
    print("|---|---:|---:|---:|---:|")
    for row in results:
        sphere_err = f"{row['max_sphere_error']:.3g}" if "max_sphere_error" in row else "-"
        grad_err = f"{row['max_grad_rel_err']:.3g}" if "max_grad_rel_err" in row else "-"
        print(
            f"| {row['shape']} | {row['resolution']} | {row['max_abs_residual']:.3g} "
            f"| {sphere_err} | {grad_err} |"
        )

    if any("marching_cubes_s" in row for row in results):
        print("\n## Context: skimage.measure.marching_cubes on the same volumes\n")
        print("| shape | res | marching_cubes (ms) | hermite (ms) |")
        print("|---|---:|---:|---:|")
        for row in results:
            if "marching_cubes_s" in row:
                print(
                    f"| {row['shape']} | {row['resolution']} "
                    f"| {_fmt_ms(row['marching_cubes_s'])} | {_fmt_ms(row['hermite_s'])} |"
                )
    else:
        print("\nscikit-image not installed; skipping the marching-cubes context reference.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Grid cells per axis to benchmark (default: 16 32 64).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed repetitions per stage; the best time is reported (default: 3).",
    )
    parser.add_argument("--json", type=str, default=None, help="Optional path to dump results.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    if any(resolution < 1 for resolution in args.resolutions):
        parser.error("resolutions must be at least 1.")

    results = [
        _run_case(name, resolution, args.repeats)
        for name in _SHAPES
        for resolution in args.resolutions
    ]
    _print_tables(results)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nWrote JSON results to {args.json}")


if __name__ == "__main__":
    main()
