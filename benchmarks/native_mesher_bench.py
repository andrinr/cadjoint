"""Benchmark the native (Rust) dual-contouring core against the Python pipeline.

Two modes, run from the repository root:

Stage-by-stage profile of the pure-Python/JAX reference pipeline (used to
decide what to move to native code)::

    .venv/bin/python benchmarks/native_mesher_bench.py profile --resolutions 32 64 128

Native-vs-Python comparison (once ``native/`` is built)::

    .venv/bin/python benchmarks/native_mesher_bench.py compare --resolutions 32 64 128

Timings separate first-call cost (jit trace + compile for the JAX stages)
from steady-state execute time (min over ``--repeats``).  All numbers are
wall-clock milliseconds on the current machine.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint.meshing.dual_contouring import (
    dual_faces,
    extract_mesh,
    qef_vertices,
    sharp_qef_vertices,
)
from cadjoint.meshing.edge_detection import (
    GridSpec,
    edge_hermite_data,
    find_crossing_edges,
    sample_grid,
)
from cadjoint.meshing.features import manifold_cell_incidence
from cadjoint.sdf.boolean.smooth import smooth_min
from cadjoint.sdf.primitives.box import Box
from cadjoint.sdf.primitives.cylinder import Cylinder
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon

BOX_SIZE = (0.4, 0.5, 0.6)


def sphere_sdf(p):
    return jnp.sqrt(jnp.sum(p * p)) - 1.0


def box_sdf(p):
    return Box.sdf(p, jnp.asarray(BOX_SIZE, dtype=jnp.float32))


def bracket_sdf(p):
    """The L-bracket from ``examples/fem_bracket_optimization.py`` (nominal)."""
    p = jnp.asarray(p)
    half_plate = 0.1
    plate = Box.sdf(
        p - jnp.array([0.0, 0.0, 1.0]) * half_plate,
        jnp.stack([jnp.asarray(1.2), jnp.asarray(0.8), jnp.asarray(half_plate)]),
    )
    q_web = jnp.stack([p[..., 0], p[..., 2], p[..., 1] + 0.7], axis=-1)
    web = ExtrudedPolygon.sdf(
        q_web,
        depth=0.16,
        v0=jnp.array([-1.1, 0.0]),
        v1=jnp.array([1.1, 0.0]),
        v2=jnp.array([0.85, 1.2]),
        v3=jnp.array([-0.85, 1.2]),
    )
    q_rib = jnp.stack([p[..., 1], p[..., 2], p[..., 0]], axis=-1)
    rib = ExtrudedPolygon.sdf(
        q_rib,
        depth=0.12,
        v0=jnp.array([0.55, 0.02]),
        v1=jnp.array([-0.62, 0.02]),
        v2=jnp.array([-0.62, 0.88]),
    )
    body = smooth_min(smooth_min(plate, web, 0.05), rib, 0.05)
    for bolt_x in (-0.7, 0.7):
        hole = Cylinder.sdf(p - jnp.array([bolt_x, 0.35, half_plate]), 0.16, 0.2)
        body = jnp.maximum(body, -hole)
    return body


SCENES = {
    "sphere": (sphere_sdf, (-1.3, -1.3, -1.3), (2.6, 2.6, 2.6)),
    "box": (box_sdf, (-0.85, -0.95, -1.05), (1.7, 1.9, 2.1)),
    "bracket": (bracket_sdf, (-1.3, -0.95, -0.06), (2.6, 1.9, 1.42)),
}


def timed(function, repeats):
    """(first_call_ms, steady_ms, result): first call, then min of repeats."""
    start = time.perf_counter()
    result = function()
    first = time.perf_counter() - start
    steady = first
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        steady = min(steady, time.perf_counter() - start)
    return first * 1e3, steady * 1e3, result


def profile_stage_breakdown(name, sdf, grid: GridSpec, repeats: int) -> dict:
    """Time every pipeline stage separately on one scene/resolution."""
    row = {"scene": name, "resolution": grid.cells[0]}

    def block(x):
        jax.block_until_ready(x)
        return x

    row["sample_first_ms"], row["sample_ms"], values = timed(
        lambda: block(sample_grid(sdf, grid)), repeats
    )
    row["edges_ms"] = timed(lambda: find_crossing_edges(values), repeats)[1]
    edges = find_crossing_edges(values)
    inside = values < 0.0
    row["incidence_ms"] = timed(lambda: manifold_cell_incidence(edges, grid, inside), repeats)[1]
    incidence = manifold_cell_incidence(edges, grid, inside)
    row["edge_count"] = edges.count
    row["cell_count"] = incidence.count

    row["hermite_first_ms"], row["hermite_ms"], hermite = timed(
        lambda: block(edge_hermite_data(sdf, grid, edges)), repeats
    )
    row["qef_smooth_first_ms"], row["qef_smooth_ms"], (vertices, _normals) = timed(
        lambda: block(qef_vertices(hermite, incidence, grid)), repeats
    )
    row["qef_sharp_ms"] = timed(lambda: sharp_qef_vertices(hermite, incidence, grid), repeats)[1]
    positions = np.asarray(vertices, dtype=np.float64)
    row["faces_ms"] = timed(lambda: dual_faces(edges, incidence, grid, positions), repeats)[1]

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row["extract_total_ms"] = timed(lambda: extract_mesh(sdf, grid), repeats)[1]
    return row


def gradient_breakdown(name, sdf, grid: GridSpec, repeats: int) -> dict:
    """Trace/compile vs execute time of one frozen-topology gradient step."""
    values = sample_grid(sdf, grid)
    edges = find_crossing_edges(values)
    incidence = manifold_cell_incidence(edges, grid, values < 0.0)

    def loss(scale):
        def scaled(p):
            return sdf(p * scale)

        hermite = edge_hermite_data(scaled, grid, edges)
        vertices, _ = qef_vertices(hermite, incidence, grid)
        return jnp.sum(vertices**2)

    gradient = jax.jit(jax.grad(loss))
    first, steady, _ = timed(lambda: jax.block_until_ready(gradient(jnp.float32(1.0))), repeats)
    return {
        "scene": name,
        "resolution": grid.cells[0],
        "grad_first_ms": first,
        "grad_ms": steady,
    }


def make_grid(name: str, resolution: int) -> GridSpec:
    _sdf, bounds, size = SCENES[name]
    return GridSpec.from_bounds(bounds, size, resolution)


def run_profile(resolutions, repeats, scenes):
    rows, grad_rows = [], []
    for name in scenes:
        sdf = SCENES[name][0]
        for resolution in resolutions:
            grid = make_grid(name, resolution)
            rows.append(profile_stage_breakdown(name, sdf, grid, repeats))
            grad_rows.append(gradient_breakdown(name, sdf, grid, repeats))
    print("## Python pipeline stage breakdown (ms; first = includes trace/compile)\n")
    keys = [
        "scene",
        "resolution",
        "edge_count",
        "cell_count",
        "sample_first_ms",
        "sample_ms",
        "edges_ms",
        "incidence_ms",
        "hermite_first_ms",
        "hermite_ms",
        "qef_smooth_first_ms",
        "qef_smooth_ms",
        "qef_sharp_ms",
        "faces_ms",
        "extract_total_ms",
    ]
    print("| " + " | ".join(k.replace("_ms", "") for k in keys) + " |")
    print("|" + "---|" * len(keys))
    for row in rows:
        cells = [f"{row[k]:.1f}" if isinstance(row[k], float) else str(row[k]) for k in keys]
        print("| " + " | ".join(cells) + " |")
    print("\n## Frozen-topology gradient (ms)\n")
    print("| scene | res | grad first (trace+compile) | grad steady |")
    print("|---|---:|---:|---:|")
    for row in grad_rows:
        print(
            f"| {row['scene']} | {row['resolution']} | {row['grad_first_ms']:.1f} "
            f"| {row['grad_ms']:.1f} |"
        )
    return {"stages": rows, "gradients": grad_rows}


def run_compare(resolutions, repeats, scenes):
    """Native vs Python execute-time comparison (requires the built cdylib)."""
    from cadjoint.meshing import native

    rows = []
    for name in scenes:
        sdf = SCENES[name][0]
        for resolution in resolutions:
            grid = make_grid(name, resolution)
            values = sample_grid(sdf, grid)
            inside = values < 0.0
            edges = find_crossing_edges(values)
            incidence = manifold_cell_incidence(edges, grid, inside)
            hermite = jax.block_until_ready(edge_hermite_data(sdf, grid, edges))
            vertices = sharp_qef_vertices(hermite, incidence, grid)

            row = {"scene": name, "resolution": resolution, "edge_count": edges.count}
            row["py_edges_ms"] = timed(lambda v=values: find_crossing_edges(v), repeats)[1]
            row["nat_edges_ms"] = timed(
                lambda v=values: native.find_crossing_edges_native(v), repeats
            )[1]
            row["py_incidence_ms"] = timed(
                lambda e=edges, g=grid, i=inside: manifold_cell_incidence(e, g, i), repeats
            )[1]
            row["nat_incidence_ms"] = timed(
                lambda e=edges, g=grid, i=inside: native.manifold_cell_incidence_native(e, g, i),
                repeats,
            )[1]
            row["py_qef_sharp_ms"] = timed(
                lambda h=hermite, i=incidence, g=grid: sharp_qef_vertices(h, i, g), repeats
            )[1]
            row["nat_qef_sharp_ms"] = timed(
                lambda h=hermite, i=incidence, g=grid: native.sharp_qef_vertices_native(h, i, g),
                repeats,
            )[1]
            positions = np.asarray(vertices, dtype=np.float64)
            row["py_faces_ms"] = timed(
                lambda e=edges, i=incidence, g=grid, p=positions: dual_faces(e, i, g, p), repeats
            )[1]
            row["nat_faces_ms"] = timed(
                lambda e=edges, i=incidence, g=grid, p=positions: native.dual_faces_native(
                    e, i, g, p
                ),
                repeats,
            )[1]

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row["py_extract_ms"] = timed(lambda s=sdf, g=grid: extract_mesh(s, g), repeats)[1]
                row["nat_extract_ms"] = timed(
                    lambda s=sdf, g=grid: native.extract_mesh_native(s, g), repeats
                )[1]
            rows.append(row)

    print("## Native vs Python (steady-state execute, ms)\n")
    keys = [
        "scene",
        "resolution",
        "edge_count",
        "py_edges_ms",
        "nat_edges_ms",
        "py_incidence_ms",
        "nat_incidence_ms",
        "py_qef_sharp_ms",
        "nat_qef_sharp_ms",
        "py_faces_ms",
        "nat_faces_ms",
        "py_extract_ms",
        "nat_extract_ms",
    ]
    print("| " + " | ".join(k.replace("_ms", "") for k in keys) + " |")
    print("|" + "---|" * len(keys))
    for row in rows:
        cells = [f"{row[k]:.2f}" if isinstance(row[k], float) else str(row[k]) for k in keys]
        print("| " + " | ".join(cells) + " |")
    return {"compare": rows}


def run_grad(resolutions, repeats, scenes):
    """Frozen-topology gradient step: reference QEF vs tesseract-backed native.

    Runs under jax x64 (the native QEF tesseract schema is float64). Times
    one eager `jax.grad` of `sum(vertices**2)` w.r.t. a design scale through
    `edge_hermite_data` + the QEF; first call includes trace overhead.
    """
    jax.config.update("jax_enable_x64", True)
    from cadjoint.meshing import native

    rows = []
    for name in scenes:
        sdf = SCENES[name][0]
        for resolution in resolutions:
            grid = make_grid(name, resolution)
            values = sample_grid(sdf, grid)
            edges = find_crossing_edges(values)
            incidence = manifold_cell_incidence(edges, grid, values < 0.0)

            def loss_with(qef, sdf=sdf, grid=grid, edges=edges, incidence=incidence):
                def loss(scale):
                    def scaled(p):
                        return sdf(p * scale)

                    hermite = edge_hermite_data(scaled, grid, edges)
                    vertices, _ = qef(hermite, incidence, grid)
                    return jnp.sum(vertices**2)

                return jax.grad(loss)

            reference = loss_with(qef_vertices)
            native_grad = loss_with(native.qef_vertices_native)
            row = {"scene": name, "resolution": resolution, "edge_count": edges.count}
            row["py_grad_first_ms"], row["py_grad_ms"], _ = timed(
                lambda g=reference: float(g(jnp.float64(1.0))), repeats
            )
            row["nat_grad_first_ms"], row["nat_grad_ms"], _ = timed(
                lambda g=native_grad: float(g(jnp.float64(1.0))), repeats
            )
            rows.append(row)
    print("## Frozen-topology gradient: reference vs tesseract-backed native (ms)\n")
    print("| scene | res | edges | py first | py steady | native first | native steady |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['scene']} | {row['resolution']} | {row['edge_count']} "
            f"| {row['py_grad_first_ms']:.1f} | {row['py_grad_ms']:.1f} "
            f"| {row['nat_grad_first_ms']:.1f} | {row['nat_grad_ms']:.1f} |"
        )
    return {"grad": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["profile", "compare", "grad"])
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenes", type=str, nargs="+", default=list(SCENES))
    parser.add_argument("--json", type=str, default=None)
    arguments = parser.parse_args()
    if arguments.mode == "profile":
        payload = run_profile(arguments.resolutions, arguments.repeats, arguments.scenes)
    elif arguments.mode == "grad":
        payload = run_grad(arguments.resolutions, arguments.repeats, arguments.scenes)
    else:
        payload = run_compare(arguments.resolutions, arguments.repeats, arguments.scenes)
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
