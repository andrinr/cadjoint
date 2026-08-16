"""Benchmark dual-contour mesh generation: quality, gradients, and timing.

Run from the repository root::

    .venv/bin/python benchmarks/dual_contouring_bench.py --resolutions 16 32

Reports, per shape and resolution:

- extraction wall time and triangle count;
- watertightness (every undirected edge shared by exactly two triangles);
- sharp-corner placement error on the box, against scikit-image marching
  cubes on the same volume when available (the number dual contouring
  exists for);
- signed-volume error against the analytic solid volume;
- sampled surface deviation (max |field| over area-uniform mesh samples) and
  sampled self-intersection count from :func:`jaxcad.meshing.mesh_report`;
- 5th-percentile triangle minimum angle (degrees);
- triangle count and wall time of error-bounded simplification
  (:func:`jaxcad.meshing.simplify.simplify_mesh` at a quarter cell size);
- wall time of one reverse-mode gradient of a mesh loss w.r.t. a design
  parameter.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxcad.meshing.diagnostics import mesh_report
from jaxcad.meshing.dual_contouring import extract_mesh, qef_vertices
from jaxcad.meshing.edge_detection import (
    GridSpec,
    edge_hermite_data,
    find_crossing_edges,
    sample_grid,
)
from jaxcad.meshing.features import cell_edge_incidence
from jaxcad.meshing.simplify import simplify_mesh
from jaxcad.sdf.primitives import Box

BOX_SIZE = np.array([0.4, 0.5, 0.6])
BOX_CORNERS = (
    np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * BOX_SIZE
)


def sphere_sdf(p):
    return jnp.sqrt(jnp.sum(p * p)) - 1.0


def box_sdf(p):
    return Box.sdf(p, jnp.asarray(BOX_SIZE, dtype=jnp.float32))


SHAPES = {
    "sphere": (sphere_sdf, (-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 4.0 / 3.0 * np.pi),
    "box": (box_sdf, (-0.85, -0.95, -1.05), (1.7, 1.9, 2.1), float(np.prod(2 * BOX_SIZE))),
}


def best_time(function, repeats):
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
    return min(durations), result


def watertight(faces: np.ndarray) -> bool:
    if faces.shape[0] == 0:
        return False
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def corner_error(vertices: np.ndarray) -> float:
    return max(float(np.min(np.linalg.norm(vertices - corner, axis=1))) for corner in BOX_CORNERS)


def minimum_angle_p5(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    angles = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u = q - p
        v = r - p
        cosine = np.einsum("ij,ij->i", u, v) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-30
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return float(np.percentile(np.minimum(np.minimum(angles[0], angles[1]), angles[2]), 5))


def marching_cubes_reference(sdf, grid: GridSpec):
    try:
        from skimage import measure
    except ImportError:
        return None
    values = sample_grid(sdf, grid)
    start = time.perf_counter()
    verts, _faces, _, _ = measure.marching_cubes(values, level=0.0, spacing=tuple(grid.spacing))
    duration = time.perf_counter() - start
    verts = verts + np.asarray(grid.origin)
    return duration, verts


def gradient_time(sdf_of, grid: GridSpec, repeats: int) -> float:
    edges = find_crossing_edges(sample_grid(sdf_of(1.0), grid))
    incidence = cell_edge_incidence(edges, grid)

    def loss(scale):
        hermite = edge_hermite_data(sdf_of(scale), grid, edges)
        vertices, _ = qef_vertices(hermite, incidence, grid)
        return jnp.sum(vertices**2)

    gradient = jax.grad(loss)
    float(gradient(jnp.float32(1.0)))  # warm up
    duration, _ = best_time(lambda: float(gradient(jnp.float32(1.0))), repeats)
    return duration


def run(resolutions, repeats):
    rows = []
    for name, (sdf, bounds, size, exact_volume) in SHAPES.items():
        for resolution in resolutions:
            grid = GridSpec.from_bounds(bounds, size, resolution)
            extract_mesh(sdf, grid)  # warm up jit caches
            duration, mesh = best_time(lambda g=grid, s=sdf: extract_mesh(s, g), repeats)
            # All benchmark fields are exact SDFs (Lipschitz 1), so octree
            # pruning is exact here; it returns the identical mesh.
            sparse_duration, _ = best_time(
                lambda g=grid, s=sdf: extract_mesh(s, g, lipschitz=1.0), repeats
            )
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            report = mesh_report(sdf, mesh)  # outside the timed region
            # Error-bounded simplification at a quarter cell; timed once
            # (deterministic host-side pass loop, no jit cache to warm).
            simplify_error = 0.25 * float(np.max(grid.spacing))
            simplify_start = time.perf_counter()
            simplified = simplify_mesh(mesh, sdf, error=simplify_error)
            simplify_duration = time.perf_counter() - simplify_start
            row = {
                "shape": name,
                "resolution": resolution,
                "triangles": int(mesh.faces.shape[0]),
                "extract_ms": duration * 1e3,
                "sparse_ms": sparse_duration * 1e3,
                "watertight": watertight(mesh.faces),
                "volume_rel_err": abs(signed_volume(vertices, mesh.faces) - exact_volume)
                / exact_volume,
                "deviation_max": report["surface_deviation"]["max_abs"],
                "self_intersections": report["self_intersections"]["count"],
                "min_angle_p5_deg": minimum_angle_p5(vertices, mesh.faces),
                "simplified_tris": int(simplified.faces.shape[0]),
                "simplify_ms": simplify_duration * 1e3,
            }
            if name == "box":
                row["corner_err"] = corner_error(vertices)
                reference = marching_cubes_reference(sdf, grid)
                if reference is not None:
                    mc_time, mc_vertices = reference
                    row["mc_ms"] = mc_time * 1e3
                    row["mc_corner_err"] = corner_error(mc_vertices)
            if name == "sphere":
                scaled = lambda s: (lambda p: jnp.sqrt(jnp.sum(p * p)) - s)  # noqa: E731
                row["grad_ms"] = gradient_time(scaled, grid, repeats) * 1e3
            rows.append(row)
    return rows


def print_markdown(rows):
    print("## Dual contouring quality and timing\n")
    print(
        "| shape | res | tris | extract (ms) | sparse (ms) | watertight | volume rel err "
        "| dev max | self-isect | min angle p5 (deg) | simplified tris (ms) | corner err "
        "| MC corner err | MC (ms) | grad (ms) |"
    )
    print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['shape']} | {row['resolution']} | {row['triangles']} "
            f"| {row['extract_ms']:.1f} | {row['sparse_ms']:.1f} | {row['watertight']} "
            f"| {row['volume_rel_err']:.2e} | {row['deviation_max']:.2e} "
            f"| {row['self_intersections']} | {row['min_angle_p5_deg']:.1f} "
            f"| {row['simplified_tris']} ({row['simplify_ms']:.1f}) "
            f"| {row.get('corner_err', float('nan')):.2e} "
            f"| {row.get('mc_corner_err', float('nan')):.2e} "
            f"| {row.get('mc_ms', float('nan')):.1f} "
            f"| {row.get('grad_ms', float('nan')):.1f} |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=str, default=None)
    arguments = parser.parse_args()
    rows = run(arguments.resolutions, arguments.repeats)
    print_markdown(rows)
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
