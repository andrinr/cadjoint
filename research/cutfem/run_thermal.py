"""Head to head on a scene's own heat study: cut cells against the tet mesh.

The starter's ``ThermalStudy`` (conductivity, die flux on the slug bottom,
ambient on the fin field) solved two ways on the same design and the same
objective — the mean temperature over the body — with the derivative in
every free parameter of the scene from each route:

* the mesh route: the study's TET10 mesh at its declared resolution and at
  scaled-up ones, node positions recomputed differentiably at frozen
  topology (``recompute_tet_points``), jax-fem's adjoint;
* the cut-cell route: the scene lowered to its node table, the conditions
  read onto the boundary facets by the study's own selections, ``jax.grad``
  through the assembly.

Each route is checked against central differences of its own objective —
the tet route at three steps, since a direct solve on sliver tets has a
noise floor that finite differences divide by 2δ; the cut cells both at
fixed structure (the function ``jax.grad`` differentiates) and
re-classified (what an optimiser sees, region membership and all) — and
the routes against each other by the angle between their gradients.

    PYTHONPATH=. python research/cutfem/run_thermal.py [scenes/starter.py] \\
        [--levels 16,32,48] [--n-sub 1] [--tet-scales 1,1.5,2]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from research.cutfem import cutfem as C


def _unflatten(free: dict, theta):
    """The flat θ back into the scene's free-parameter dict, in ``lower``'s order."""
    out, i = {}, 0
    for name, value in free.items():
        shape = np.asarray(value).shape
        size = int(np.prod(shape)) if shape else 1
        out[name] = theta[i : i + size].reshape(shape) if shape else theta[i]
        i += size
    return out


def _tet_mean_temperature(points, cells, temperature):
    """Volume-weighted mean of the corner-averaged temperature over the tets."""
    corners = cells[:, :4]
    p = points[corners]
    volume = jnp.abs(jnp.linalg.det(p[:, 1:] - p[:, :1])) / 6.0
    return jnp.sum(volume * temperature[corners].mean(axis=1)) / jnp.sum(volume)


def _central(fn, theta, indices, delta):
    out = {}
    for i in indices:
        e = np.zeros(len(theta))
        e[i] = delta
        out[i] = (float(fn(theta + e)) - float(fn(theta - e))) / (2 * delta)
    return out


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", default="scenes/starter.py")
    parser.add_argument("--levels", default="16,32,48", help="cut-cell grids, cells across")
    parser.add_argument("--n-sub", type=int, default=1)
    parser.add_argument(
        "--tet-scales", default="1,1.5,2", help="multiples of the study's resolution"
    )
    parser.add_argument(
        "--fd", type=int, default=3, help="parameters checked by finite differences"
    )
    parser.add_argument("--fd-step", type=float, default=1e-5)
    args = parser.parse_args()

    from cadjoint import extract_parameters, functionalize
    from cadjoint.fem import SimMesh
    from cadjoint.fem.motion import recompute_tet_points
    from cadjoint.fem.study import Dirichlet, HeatFlux
    from cadjoint.viewer.worker.scene import _execute_scene
    from cadjoint.zeroset import lower

    namespace = _execute_scene(Path(args.scene).read_text())
    body, study, declared = (
        namespace["thermal_body"],
        namespace["heat_study"],
        namespace["sink_mesh"],
    )
    free, fixed, _ = extract_parameters(body)
    model = lower(body)
    theta = jnp.asarray(model.theta)
    names = model.names
    fn = functionalize(body)
    delta = args.fd_step
    print(f"{args.scene}: study {study.name!r}, k = {study.conductivity}, {len(theta)} parameters")

    # ── the mesh route, at the declared resolution and finer ────────────────
    tet_rows = []
    fd_tet: dict = {}
    picks = None
    for scale in (float(x) for x in args.tet_scales.split(",")):
        resolution = tuple(int(round(r * scale)) for r in declared.resolution)
        sim_mesh = SimMesh(
            name=f"{declared.name}@{scale:g}",
            resolution=resolution,
            bounds=declared.bounds,
            size=declared.size,
            method=declared.method,
            domain=body,
        )
        t0 = time.perf_counter()
        mesh = sim_mesh.build(body)
        t_mesh = time.perf_counter() - t0

        def tet_objective(th, mesh=mesh):
            sdf = fn(_unflatten(free, th), fixed)
            points = recompute_tet_points(sdf, mesh)
            result = study.solve(body, mesh=mesh, points=points)
            return _tet_mean_temperature(
                points, jnp.asarray(mesh.cells), result.solution.temperature
            )

        t0 = time.perf_counter()
        j, g = jax.value_and_grad(tet_objective)(theta)
        j, g = float(j), np.asarray(g)
        t_solve = time.perf_counter() - t0
        if picks is None:
            picks = np.argsort(-np.abs(g))[: args.fd]
            fd_tet = {
                d: _central(tet_objective, theta, picks, d) for d in (delta * 10, delta, delta / 10)
            }
        tet_rows.append(
            (scale, resolution, mesh.points.shape[0], mesh.cells.shape[0], j, g, t_mesh, t_solve)
        )
        print(
            f"tet10 x{scale:g}: {resolution} -> {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} tets; "
            f"mesh {t_mesh:.1f} s, J+grad {t_solve:.1f} s; J = {j:.6f}"
        )
    g_tet = tet_rows[-1][5]

    # ── the cut-cell route ─────────────────────────────────────────────────
    problem = C.Thermal(
        conductivity=float(study.conductivity),
        source=float(study.source),
        dirichlet=tuple(
            (bc.nodes.contains, float(bc.value)) for bc in study.bcs if isinstance(bc, Dirichlet)
        ),
        neumann=tuple(
            (bc.nodes.contains, float(bc.flux)) for bc in study.bcs if isinstance(bc, HeatFlux)
        ),
    )
    lo = np.asarray(declared.bounds, float)
    hi = lo + np.asarray(declared.size, float)
    cut_rows = []
    fd_fixed: dict = {}
    fd_free: dict = {}
    for level, n_across in enumerate(int(x) for x in args.levels.split(",")):
        h = float((hi - lo).max()) / n_across
        grid = C.grid_for(lo, hi, h)
        t0 = time.perf_counter()
        s = C.classify(model, theta, grid, n_sub=args.n_sub, problem=problem)
        t_classify = time.perf_counter() - t0
        objective = jax.jit(lambda t, s=s: C.mean_temperature(t, s))
        value_and_grad = jax.jit(jax.value_and_grad(lambda t, s=s: C.mean_temperature(t, s)))
        t0 = time.perf_counter()
        j, g = value_and_grad(theta)
        j, g = float(j), np.asarray(g)
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        value_and_grad(theta)
        t_warm = time.perf_counter() - t0
        if level == 0:
            fd_fixed = _central(objective, theta, picks, delta)

            def free_objective(t, grid=grid):
                return C.mean_temperature(
                    t, C.classify(model, t, grid, n_sub=args.n_sub, problem=problem)
                )

            fd_free = _central(free_objective, theta, picks, delta)
        cut_rows.append((n_across, h, s.summary(), j, g, t_first, t_warm, t_classify))
        print(
            f"cut cells /{n_across}: h = {h:.3f}, {s.summary()['dofs']} DOFs; J = {j:.6f}; warm {t_warm:.1f} s"
        )

    # ── report ─────────────────────────────────────────────────────────────
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
    print()
    print("| route | resolution | DOFs | J = mean T | ‖dJ/dθ‖ | cos ∠(·, finest tet) | J+grad s |")
    print("|---|---|---|---|---|---|---|")
    for scale, resolution, nodes, tets, j, g, t_mesh, t_solve in tet_rows:
        print(
            f"| tet10 ×{scale:g} | {resolution} ({tets} tets) | {nodes} | {j:.6f} | {np.linalg.norm(g):.5f} "
            f"| {cos(g, g_tet):.4f} | {t_solve:.1f} eager (mesh {t_mesh:.1f}) |"
        )
    for _n_across, h, summary, j, g, t_first, t_warm, t_classify in cut_rows:
        print(
            f"| cut cells | h = {h:.3f} ({summary['cut']} cut of {summary['cells']} cells) | {summary['dofs']} "
            f"| {j:.6f} | {np.linalg.norm(g):.5f} | {cos(g, g_tet):.4f} "
            f"| {t_warm:.1f} warm ({t_first:.0f} first, classify {t_classify:.1f}) |"
        )
    print()
    g0_tet, g0_cut = tet_rows[0][5], cut_rows[0][4]
    steps = sorted(fd_tet)
    print(
        f"| parameter | tet ×{tet_rows[0][0]:g} AD | tet FD δ="
        + " / ".join(f"{d:g}" for d in steps)
        + " | tet finest AD "
        f"| cut /{cut_rows[0][0]} AD | cut FD fixed structure | cut FD re-classified | cut finest AD |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for i in np.argsort(-np.abs(g_tet))[:8]:
        cells = [
            f"{g0_tet[i]:+.5f}",
            " / ".join(f"{fd_tet[d][i]:+.5f}" for d in steps) if i in fd_tet[steps[0]] else "–",
            f"{g_tet[i]:+.5f}",
            f"{g0_cut[i]:+.5f}",
            f"{fd_fixed[i]:+.5f}" if i in fd_fixed else "–",
            f"{fd_free[i]:+.5f}" if i in fd_free else "–",
            f"{cut_rows[-1][4][i]:+.5f}",
        ]
        print(f"| {names[i]} | " + " | ".join(cells) + " |")
    print(
        f"\nAD vs own FD, max rel over {args.fd} parameters: "
        + "; ".join(
            f"tet ×{tet_rows[0][0]:g} δ={d:g} {max(_rel(fd_tet[d][i], g0_tet[i]) for i in fd_tet[d]):.1e}"
            for d in steps
        )
        + f"; cut /{cut_rows[0][0]} fixed structure {max(_rel(fd_fixed[i], g0_cut[i]) for i in fd_fixed):.1e}, "
        f"re-classified {max(_rel(fd_free[i], g0_cut[i]) for i in fd_free):.1e}"
    )


if __name__ == "__main__":
    main()
