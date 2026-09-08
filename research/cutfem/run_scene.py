"""The solver on a cadjoint scene: lowered through ``cadjoint.zeroset``, solved in 3D.

Torsion-like Poisson (−Δu = 1, u = 0 on the surface, J = ∫u) on a shipped
scene, at a few grid sizes: J, ‖dJ/dθ‖ over every free parameter of the
scene, the time for J and its gradient with the sparse solve, and a
central-difference check on a few parameters at the coarsest level.

    PYTHONPATH=. python research/cutfem/run_scene.py [scenes/starter.py] [--levels 12,16,24]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from research.cutfem import cutfem as C


def bounds_of(f, theta, extent: float = 3.0, n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """The bounding box of {f < 0} from a coarse scan of the viewer's volume."""
    axis = np.linspace(-extent, extent, n)
    pts = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    inside = C._scan(f, theta, pts) < 0
    if not inside.any():
        raise SystemExit("the scene has no interior in the viewer's volume")
    return pts[inside].min(axis=0), pts[inside].max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", default="scenes/starter.py")
    parser.add_argument("--levels", default="12,16,24", help="cells across the longest side")
    parser.add_argument("--n-sub", type=int, default=2)
    parser.add_argument(
        "--fd", type=int, default=3, help="parameters to check by FD, coarsest level"
    )
    args = parser.parse_args()

    from cadjoint.viewer.worker.scene import _execute_scene
    from cadjoint.zeroset import lower

    t0 = time.perf_counter()
    root = _execute_scene(Path(args.scene).read_text())["scene"]
    model = lower(root)
    theta = jnp.asarray(model.theta)
    f = C.as_field(model)
    lo, hi = bounds_of(f, theta)
    print(
        f"{args.scene}: {len(model.nodes)} nodes, {len(theta)} parameters "
        f"({', '.join(model.names[:6])}{', …' if len(model.names) > 6 else ''}); "
        f"bounds {np.round(lo, 2).tolist()} – {np.round(hi, 2).tolist()}; "
        f"lowered and scanned in {time.perf_counter() - t0:.1f} s"
    )
    span = float((hi - lo).max())
    print()
    print(
        "| cells across | h | DOFs | cut cells | quadrature pts | J | ‖dJ/dθ‖ | max |dJ/dθ_k| (name) | J+grad s | FD check |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for level, n_across in enumerate(int(x) for x in args.levels.split(",")):
        h = span / n_across
        pad = 2 * h
        grid = C.grid_for(lo - pad, hi + pad, h)
        t1 = time.perf_counter()
        s = C.classify(model, theta, grid, n_sub=args.n_sub, mode="sharp")
        t_classify = time.perf_counter() - t1
        value_and_grad = jax.jit(jax.value_and_grad(lambda t, s=s: C.objective(t, s)))
        t1 = time.perf_counter()
        J, g = value_and_grad(theta)
        J, g = float(J), np.asarray(g)
        t_solve = time.perf_counter() - t1
        t1 = time.perf_counter()
        value_and_grad(theta)
        t_warm = time.perf_counter() - t1
        (pts, w, _), (bpts, _, _, _) = C.quadrature(theta, s)
        k = int(np.argmax(np.abs(g)))
        fd_note = "-"
        if level == 0 and args.fd:
            picks = np.argsort(-np.abs(g))[: args.fd]

            def free(t, grid=grid):
                return C.objective(t, C.classify(model, t, grid, n_sub=args.n_sub, mode="sharp"))

            errs = []
            for i in picks:
                e = np.zeros(len(theta))
                e[i] = 1e-5
                fd = (float(free(theta + e)) - float(free(theta - e))) / 2e-5
                errs.append(abs(fd - g[i]) / max(abs(g[i]), 1e-12))
            fd_note = "max rel " + f"{max(errs):.1e}" + f" over {[model.names[i] for i in picks]}"
        summary = s.summary()
        print(
            f"| {n_across} | {h:.3f} | {summary['dofs']} | {summary['cut']} | {len(w)} + {len(bpts)} "
            f"| {J:.6f} | {np.linalg.norm(g):.4f} | {abs(g[k]):.4f} ({model.names[k]}) "
            f"| {t_solve:.1f} (warm {t_warm:.1f}, classify {t_classify:.1f}) | {fd_note} |"
        )


if __name__ == "__main__":
    main()
