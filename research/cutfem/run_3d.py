"""The 3D smoke test: a sphere on a coarse Q1 hexahedral grid.

−Δu = 1 in the ball of radius R has u = (R² − r²)/6, J = ∫u = 4πR⁵/45 and
dJ/dR = 4πR⁴/9. Prints, for both quadrature modes, J and dJ/dR against
those values, the translation derivative (analytically zero), and, up to
1.6/16, ``jax.grad`` against central differences of the objective with the
structure re-classified at each evaluation. The finest level has 10 k
dense DOFs and takes about 15 s.

    PYTHONPATH=. python research/cutfem/run_3d.py
"""

from __future__ import annotations

import time

import jax
import numpy as np

from research.cutfem import cutfem as C
from research.cutfem import model as M
from research.cutfem.report import order, sci, table


def main():
    model, theta, _ = M.sphere()
    radius = theta[0]
    j_exact, dj_exact = 4 * np.pi * radius**5 / 45, 4 * np.pi * radius**4 / 9
    print(f"sphere R = {radius}: J = {j_exact:.6f}, dJ/dR = {dj_exact:.6f}\n")
    levels = (8, 12, 16, 24, 32)
    ratios = [b / a for a, b in zip(levels[:-1], levels[1:])]
    for mode, alpha, n_sub in (("sharp", 0.0, 2), ("smooth", 0.5, 2)):
        rows, j_errs, g_errs = [], [], []
        for n in levels:
            t0 = time.time()
            h = 1.6 / n
            grid = C.grid_for([-0.8] * 3, [0.8] * 3, h)
            s = C.classify(model, theta, grid, n_sub=n_sub, mode=mode, alpha=alpha)
            obj = jax.jit(lambda t, s=s: C.objective(t, s))
            j = float(obj(theta))
            g = np.asarray(jax.grad(obj)(theta))
            if n <= 16:

                def free(t, grid=grid, n_sub=n_sub, mode=mode, alpha=alpha):
                    return C.objective(t, C.classify(model, t, grid, n_sub, mode, alpha))

                fd = C.central_difference(free, theta, 1e-5)
                agreement = f"{np.abs(g - fd).max() / np.abs(fd).max():.1e}"
            else:
                agreement = "-"
            j_errs.append((j - j_exact) / j_exact)
            g_errs.append((g[0] - dj_exact) / dj_exact)
            rows.append(
                [
                    f"1.6/{n}",
                    s.n_dofs,
                    sci(j_errs[-1]),
                    None,
                    sci(g_errs[-1]),
                    None,
                    f"{np.abs(g[1:]).max():.1e}",
                    agreement,
                    f"{time.time() - t0:.0f}",
                ]
            )
        jo, go = order(j_errs, ratios), order(g_errs, ratios)
        for r, a, b in zip(rows, jo, go):
            r[3], r[5] = a, b
        print(
            f"### {mode} (n_sub = {n_sub}" + (f", α = {alpha}" if mode == "smooth" else "") + ")\n"
        )
        print(
            table(
                [
                    "h",
                    "DOFs",
                    "J rel err",
                    "order",
                    "dJ/dR rel err",
                    "order",
                    "|dJ/dc|",
                    "AD vs FD",
                    "s",
                ],
                rows,
            )
        )
        print()


if __name__ == "__main__":
    main()
