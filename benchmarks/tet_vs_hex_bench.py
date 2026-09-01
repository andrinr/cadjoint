"""Tet-vs-hex at matched *accuracy*: the benchmark the method decision needs.

The research tables in ``research/tet-vs-hex.md`` compare families at
matched sampling lattices, where TET10 pays 8-60x hex wall-clock — but the
lattices produce very different DOF and accuracy.  The honest production
question is: **what does each family cost at the mesh size where it first
reaches the converged answer** (here: within ~3% of the reference bracket
compliance)?  This script measures exactly that, plus the
``d(compliance)/d(rib_height)`` adjoint at those matched-accuracy sizes —
the hex-aliasing exhibit (the staircased hex boundary under-reports the
inclined rib's sensitivity).

Run from the repository root (takes on the order of 15 minutes; the TET10
reference solve dominates)::

    .venv/bin/python benchmarks/tet_vs_hex_bench.py

Setup (matching the research note): the parameterized bracket of
``examples/fem_bracket_optimization.py`` at the nominal design, bolt-ball
clamps, prying traction on the outer web wall above z = 1, selected by the
note's **mesh-independent center+normal face rule** (outward normal within
~25 degrees of -y, center below y = -0.7 and above z = 1) — the node-set
selection's spanned area jumps with lattice alignment, which would poison
the cross-resolution comparison.  Compliance is the load work
``W = integral(t . u dA)`` over the loaded patch; the reported
``Wn = W / area^2`` normalizes away the residual load-patch area
differences between meshes (the applied force scales with patch area).
Both families mesh the same box — deeper in z than the demo grid, because
dual contouring needs the zero surface fully inside the lattice (the
filleted union dips just below z = 0).
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Imports follow the x64 flip by design (the backward pass needs float64).
# ruff: noqa: E402
from cadjoint.fem.hexmesh import (
    GridSpec,
    faces_from_nodes,
    recompute_points,
    sdf_to_hex_mesh,
    select_faces,
)
from cadjoint.fem.simulate import elastic_solve
from cadjoint.fem.tetmesh import (
    load_work_quads,
    load_work_tri6,
    recompute_tet_points,
    sdf_to_tet_mesh,
    tet10_face_midsides,
    tet10_mesh,
)

_REPO = Path(__file__).resolve().parents[1]

# Shared meshing box: demo bounds with the z floor deepened so the DC
# surface closes (the smooth-min fillet reaches z ~ -0.063).
BOUNDS = (-1.3, -0.95, -0.16)
SIZE = (2.6, 1.9, 1.52)

# Ladders: roughly cubic cells; y/z counts scale with the box aspect.
HEX_LADDER = [(24, 18, 14), (30, 22, 18), (36, 26, 21), (42, 31, 25), (48, 35, 28)]
HEX_REFERENCE = (66, 48, 39)
TET10_LADDER = [
    (22, 16, 13),
    (24, 18, 14),
    (26, 19, 16),
    (28, 21, 17),
    (30, 22, 18),
    (32, 23, 19),
    (34, 25, 20),
]

ACCURACY_BAND = 0.03  # "matched accuracy": within 3% of the reference Wn


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "fem_bracket_optimization", _REPO / "examples" / "fem_bracket_optimization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grid(resolution) -> GridSpec:
    return GridSpec.from_bounds(BOUNDS, SIZE, resolution)


def _wall_rule(center, normal):
    """Mesh-independent load rule: the outer (-y) web wall above z = 1.

    Matching the research note's methodology — a face belongs to the load
    patch when its outward normal points along -y (within 45 degrees:
    snapped hex faces at the wall's top edge tilt, but top faces are +z
    and web-taper faces +-x dominant, so the cone is unambiguous) and its
    center lies on the outer wall above the tip cut.  Face-level selection
    keeps the loaded area nearly constant across resolutions and mesh
    families (the exact wall patch is ~0.348), where node-set spanning
    jumps with lattice alignment.
    """
    return normal[1] < -0.7 and center[1] < -0.7 and center[2] > 1.0


def _tri_area(points, faces) -> float:
    tri = np.asarray(points)[np.asarray(faces)[:, :3]]
    return float(
        0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1).sum()
    )


def _quad_area(points, faces) -> float:
    quad = np.asarray(points)[np.asarray(faces)]
    return float(
        0.5
        * np.linalg.norm(np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1]), axis=-1).sum()
    )


class HexCase:
    """One hex resolution: mesh, load patch, compliance, adjoint."""

    method = "hex"

    def __init__(self, demo, resolution):
        self.demo = demo
        self.resolution = resolution
        started = time.perf_counter()
        self.mesh = sdf_to_hex_mesh(demo.theta_sdf(np.asarray(demo.NOMINAL)), _grid(resolution))
        self.t_mesh = time.perf_counter() - started
        # The backend loads the faces spanned by the patch's node set; keep
        # the work integral and area consistent with exactly those faces.
        rule_faces = select_faces(self.mesh, _wall_rule).nodes
        node_set = np.unique(rule_faces).astype(np.int32)
        self.faces = faces_from_nodes(self.mesh, node_set).nodes
        if self.faces.shape[0] != rule_faces.shape[0]:
            print(
                f"  note: hex {resolution} spans {self.faces.shape[0]} faces "
                f"for {rule_faces.shape[0]} rule faces"
            )
        self.area = _quad_area(self.mesh.points, self.faces)
        self.dof = 3 * self.mesh.num_points
        self.cells = self.mesh.num_cells

    def _work(self, points):
        result = elastic_solve(
            self.mesh,
            youngs=self.demo._YOUNGS,
            poisson=self.demo._POISSON,
            dirichlet=[self.demo.BOLT_CLAMP],
            tractions=[(_wall_rule, list(self.demo._TRACTION))],
            points=points,
        )
        solve_points = self.mesh.points if points is None else points
        return load_work_quads(
            solve_points, result.displacement, self.faces, np.asarray(self.demo._TRACTION)
        )

    def solve(self) -> float:
        started = time.perf_counter()
        work = float(self._work(None))
        self.t_solve = time.perf_counter() - started
        return work

    def gradient(self):
        def objective(theta):
            points = recompute_points(self.demo.theta_sdf(theta), self.mesh)
            return self._work(points)

        started = time.perf_counter()
        _, gradient = jax.value_and_grad(objective)(jnp.asarray(self.demo.NOMINAL))
        self.t_grad = time.perf_counter() - started
        return np.asarray(gradient)


class Tet10Case:
    """One TET10 resolution: mesh (sharp -> smooth fallback), compliance, adjoint."""

    method = "tet10"

    def __init__(self, demo, resolution):
        self.demo = demo
        self.resolution = resolution
        sdf = demo.theta_sdf(np.asarray(demo.NOMINAL))
        started = time.perf_counter()
        try:
            tet4 = sdf_to_tet_mesh(sdf, _grid(resolution), sharp=True)
            self.sharp = True
        except RuntimeError:
            tet4 = sdf_to_tet_mesh(sdf, _grid(resolution), sharp=False)
            self.sharp = False
        self.mesh = tet10_mesh(tet4)
        self.t_mesh = time.perf_counter() - started
        # The tet solve targets exactly the rule's boundary triangles.
        corner_faces = select_faces(self.mesh, _wall_rule).nodes
        self.faces6 = np.concatenate(
            [corner_faces, tet10_face_midsides(self.mesh, corner_faces)], axis=1
        )
        self.area = _tri_area(self.mesh.points, corner_faces)
        self.dof = 3 * self.mesh.num_points
        self.cells = self.mesh.num_cells

    def _work(self, points):
        result = elastic_solve(
            self.mesh,
            youngs=self.demo._YOUNGS,
            poisson=self.demo._POISSON,
            dirichlet=[self.demo.BOLT_CLAMP],
            tractions=[(_wall_rule, list(self.demo._TRACTION))],
            points=points,
        )
        solve_points = self.mesh.points if points is None else points
        return load_work_tri6(
            solve_points, result.displacement, self.faces6, np.asarray(self.demo._TRACTION)
        )

    def solve(self) -> float:
        started = time.perf_counter()
        work = float(self._work(None))
        self.t_solve = time.perf_counter() - started
        return work

    def gradient(self):
        def objective(theta):
            points = recompute_tet_points(self.demo.theta_sdf(theta), self.mesh)
            return self._work(points)

        started = time.perf_counter()
        _, gradient = jax.value_and_grad(objective)(jnp.asarray(self.demo.NOMINAL))
        self.t_grad = time.perf_counter() - started
        return np.asarray(gradient)


def _run_ladder(demo, factory, ladder, label):
    rows = []
    for resolution in ladder:
        try:
            case = factory(demo, resolution)
        except (RuntimeError, ValueError) as error:
            print(f"{label} {resolution}: mesh failed ({str(error)[:70]})")
            continue
        work = case.solve()
        normalized = work / case.area**2
        rows.append(
            {
                "method": case.method,
                "resolution": list(resolution),
                "cells": case.cells,
                "dof": case.dof,
                "area": round(case.area, 6),
                "work": round(work, 6),
                "wn": round(normalized, 6),
                "t_mesh": round(case.t_mesh, 2),
                "t_solve": round(case.t_solve, 2),
                "sharp": getattr(case, "sharp", None),
                "_case": case,
            }
        )
        print(
            f"{label} {resolution}: cells={case.cells} DOF={case.dof} "
            f"Wn={normalized:.4f} mesh {case.t_mesh:.1f}s solve {case.t_solve:.1f}s"
        )
    return rows


def main() -> None:
    demo = _load_demo()

    print("== hex ladder ==")
    hex_rows = _run_ladder(demo, HexCase, [*HEX_LADDER, HEX_REFERENCE], "hex")
    print("== tet10 ladder ==")
    tet_rows = _run_ladder(demo, Tet10Case, TET10_LADDER, "tet10")

    reference = hex_rows[-1]["wn"]
    tet_fine = tet_rows[-1]["wn"] if tet_rows else float("nan")
    print(
        f"\nreference Wn: hex@{hex_rows[-1]['resolution']} = {reference:.4f} "
        f"(finest tet10 = {tet_fine:.4f}, family spread "
        f"{abs(tet_fine - reference) / reference:.1%})"
    )

    matched = {}
    for label, rows in (("hex", hex_rows[:-1]), ("tet10", tet_rows)):
        inside = [row for row in rows if abs(row["wn"] - reference) / reference <= ACCURACY_BAND]
        if not inside:
            print(f"{label}: no ladder entry within {ACCURACY_BAND:.0%} of the reference")
            continue
        pick = min(inside, key=lambda row: row["dof"])
        matched[label] = pick
        print(
            f"{label} matched-accuracy pick: res {pick['resolution']} DOF {pick['dof']} "
            f"Wn {pick['wn']:.4f} ({(pick['wn'] - reference) / reference:+.1%}) "
            f"mesh+solve {pick['t_mesh'] + pick['t_solve']:.1f}s"
        )

    print("\n== rib_height adjoint at the matched-accuracy sizes ==")
    for label, pick in matched.items():
        case = pick["_case"]
        gradient = case.gradient()
        pick["grad"] = [round(float(g), 6) for g in gradient]
        pick["t_grad"] = round(case.t_grad, 2)
        print(
            f"{label} res {pick['resolution']}: grad(web, rib, plate) = "
            f"({gradient[0]:+.4f}, {gradient[1]:+.4f}, {gradient[2]:+.4f}) "
            f"[{case.t_grad:.1f}s]"
        )

    payload = {
        "bounds": BOUNDS,
        "size": SIZE,
        "reference_wn": reference,
        "hex": [{k: v for k, v in row.items() if k != "_case"} for row in hex_rows],
        "tet10": [{k: v for k, v in row.items() if k != "_case"} for row in tet_rows],
        "matched": {
            label: {k: v for k, v in row.items() if k != "_case"} for label, row in matched.items()
        },
    }
    print("\nJSON:", json.dumps(payload))


if __name__ == "__main__":
    main()
