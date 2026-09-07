"""The 2D experiments: every table in README.md, as markdown.

    PYTHONPATH=. python research/cutfem/run_2d.py [section ...]

Sections, all by default: ``convergence`` (the disc against its analytic
solution, AD against finite differences, the Hadamard formula), ``blend``
(the disc–box smooth union, AD against finite differences and the
derivative's own convergence), ``alpha`` (the smoothed indicator's
half-width), ``penalty`` (ghost and Nitsche parameters, condition numbers),
``phase`` (the disc shifted through the grid), ``delta`` (the finite
difference step and the sub-corner sign flips it crosses).
"""

from __future__ import annotations

import sys
import time

import jax
import numpy as np

from research.cutfem import cutfem as C
from research.cutfem import model as M
from research.cutfem.report import order, sci, table

DISC_BOX = ([-0.8, -0.8], [0.8, 0.8])
BLEND_BOX = ([-0.95, -0.55], [0.85, 0.55])
MODES = (("sharp", 0.0, 4), ("smooth", 0.5, 8))


def _label(mode, alpha, n_sub):
    return f"{mode} (n_sub = {n_sub}" + (f", α = {alpha}" if mode == "smooth" else "") + ")"


def _structure(model, theta, box, h, mode, alpha, n_sub):
    return C.classify(model, theta, C.grid_for(*box, h), n_sub=n_sub, mode=mode, alpha=alpha)


def _derivatives(model, theta, s, fd_delta=1e-5):
    """AD, FD at fixed structure, FD with the structure re-classified per evaluation."""
    obj = jax.jit(lambda t: C.objective(t, s))
    j = float(obj(theta))
    g = np.asarray(jax.grad(obj)(theta))
    fixed = C.central_difference(obj, theta, fd_delta)

    def free(t):
        return C.objective(t, C.classify(model, t, s.grid, s.n_sub, s.mode, s.eps / s.h))

    return j, g, fixed, C.central_difference(free, theta, fd_delta)


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def convergence():
    model, theta, _ = M.disc()
    radius = theta[0]
    j_exact, dj_exact = np.pi * radius**4 / 8, np.pi * radius**3 / 2
    print(f"## Disc: R = {radius}, J = πR⁴/8 = {j_exact:.6f}, dJ/dR = πR³/2 = {dj_exact:.6f}\n")
    for mode, alpha, n_sub in MODES:
        rows, j_errs, g_errs = [], [], []
        for h in (1 / 8, 1 / 16, 1 / 32, 1 / 64):
            t0 = time.time()
            s = _structure(model, theta, DISC_BOX, h, mode, alpha, n_sub)
            j, g, fixed, free = _derivatives(model, theta, s)
            had = np.asarray(C.hadamard(theta, s))
            j_errs.append((j - j_exact) / j_exact)
            g_errs.append((g[0] - dj_exact) / dj_exact)
            rows.append(
                [
                    f"1/{round(1 / h)}",
                    s.n_dofs,
                    sci(j_errs[-1]),
                    None,
                    sci(g_errs[-1]),
                    None,
                    f"{np.abs(g[1:]).max():.1e}",
                    sci((had[0] - dj_exact) / dj_exact),
                    f"{_rel(g, free):.1e}",
                    f"{_rel(g, fixed):.1e}",
                    f"{time.time() - t0:.0f}",
                ]
            )
        for r, a, b in zip(rows, order(j_errs), order(g_errs)):
            r[3], r[5] = a, b
        print(f"### {_label(mode, alpha, n_sub)}\n")
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
                    "Hadamard rel err",
                    "AD vs FD (re-classified)",
                    "AD vs FD (fixed)",
                    "s",
                ],
                rows,
            )
        )
        print()


def blend():
    model, theta, names = M.blended_union()
    print(f"## Disc ∪ box with smooth minimum: θ = {dict(zip(names, theta.tolist()))}\n")
    for mode, alpha, n_sub in MODES:
        rows, prev = [], None
        for h in (1 / 16, 1 / 32, 1 / 64):
            t0 = time.time()
            s = _structure(model, theta, BLEND_BOX, h, mode, alpha, n_sub)
            j, g, fixed, free = _derivatives(model, theta, s)
            rows.append(
                [
                    f"1/{round(1 / h)}",
                    s.n_dofs,
                    f"{j:.6f}",
                    " ".join(f"{x:.6f}" for x in g),
                    f"{_rel(g, free):.1e}",
                    f"{_rel(g, fixed):.1e}",
                    "-" if prev is None else f"{_rel(g, prev):.1e}",
                    f"{time.time() - t0:.0f}",
                ]
            )
            prev = g
        print(f"### {_label(mode, alpha, n_sub)}\n")
        print(
            table(
                [
                    "h",
                    "DOFs",
                    "J",
                    "dJ/dθ by AD (R, k, hy)",
                    "AD vs FD (re-classified)",
                    "AD vs FD (fixed)",
                    "change from coarser h",
                    "s",
                ],
                rows,
            )
        )
        print()


def alpha():
    model, theta, _ = M.disc()
    radius = theta[0]
    j_exact, dj_exact = np.pi * radius**4 / 8, np.pi * radius**3 / 2
    print("## The smoothed indicator's half-width ε = α h (disc, n_sub = 8)\n")
    rows = []
    for mode, a in (
        ("sharp", 0.0),
        ("smooth", 0.125),
        ("smooth", 0.25),
        ("smooth", 0.5),
        ("smooth", 1.0),
    ):
        for h in (1 / 16, 1 / 32, 1 / 64):
            s = _structure(model, theta, DISC_BOX, h, mode, a, 8)
            obj = jax.jit(lambda t, s=s: C.objective(t, s))
            j = float(obj(theta))
            g = np.asarray(jax.grad(obj)(theta))
            area = float(C.volume(theta, s))
            rows.append(
                [
                    mode if mode == "sharp" else f"smooth α = {a}",
                    f"1/{round(1 / h)}",
                    sci((area - np.pi * radius**2) / (np.pi * radius**2)),
                    sci((j - j_exact) / j_exact),
                    sci((g[0] - dj_exact) / dj_exact),
                    f"{np.abs(g[1:]).max():.1e}",
                ]
            )
    print(table(["mode", "h", "|Ω_h| rel err", "J rel err", "dJ/dR rel err", "|dJ/dc|"], rows))
    print()


def penalty():
    model, theta, _ = M.disc()
    radius = theta[0]
    j_exact, dj_exact = np.pi * radius**4 / 8, np.pi * radius**3 / 2
    h = 1 / 32
    print(f"## Ghost penalty γ_g (disc, h = 1/{round(1 / h)}, γ_N = 20)\n")
    rows = []
    for mode, a, n_sub in MODES:
        s = _structure(model, theta, DISC_BOX, h, mode, a, n_sub)
        for ghost in (0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0):
            K, _ = C.assemble(theta, s, ghost=ghost)
            cond, lo, _ = C.condition_number(K)
            obj = jax.jit(lambda t, s=s, ghost=ghost: C.objective(t, s, ghost=ghost))
            j = float(obj(theta))
            g = np.asarray(jax.grad(obj)(theta))
            rows.append(
                [
                    mode,
                    f"{ghost:g}",
                    "indefinite" if lo <= 0 else f"{cond:.1e}",
                    sci(lo, 1),
                    sci((j - j_exact) / j_exact),
                    sci((g[0] - dj_exact) / dj_exact),
                    f"{np.abs(g[1:]).max():.1e}",
                ]
            )
    print(table(["mode", "γ_g", "cond(K)", "λ_min", "J rel err", "dJ/dR rel err", "|dJ/dc|"], rows))
    print(f"\n## Nitsche penalty γ_N (disc, h = 1/{round(1 / h)}, γ_g = 0.1)\n")
    rows = []
    for mode, a, n_sub in MODES:
        s = _structure(model, theta, DISC_BOX, h, mode, a, n_sub)
        for nitsche in (5.0, 10.0, 20.0, 50.0, 100.0):
            K, _ = C.assemble(theta, s, nitsche=nitsche)
            cond, lo, _ = C.condition_number(K)
            obj = jax.jit(lambda t, s=s, nitsche=nitsche: C.objective(t, s, nitsche=nitsche))
            j = float(obj(theta))
            g = np.asarray(jax.grad(obj)(theta))
            rows.append(
                [
                    mode,
                    f"{nitsche:g}",
                    "indefinite" if lo <= 0 else f"{cond:.1e}",
                    sci(lo, 1),
                    sci((j - j_exact) / j_exact),
                    sci((g[0] - dj_exact) / dj_exact),
                    f"{np.abs(g[1:]).max():.1e}",
                ]
            )
    print(table(["mode", "γ_N", "cond(K)", "λ_min", "J rel err", "dJ/dR rel err", "|dJ/dc|"], rows))
    print()


def phase():
    radius = 0.6
    j_exact, dj_exact = np.pi * radius**4 / 8, np.pi * radius**3 / 2
    rng = np.random.default_rng(1)
    print("## The disc shifted through the grid: 8 random centres in [0, h)²\n")
    rows = []
    for mode, a, n_sub in MODES:
        for h in (1 / 16, 1 / 32, 1 / 64):
            grid = C.grid_for(*DISC_BOX, h)
            j_err, g_err, c_err, h_err = [], [], [], []
            for _ in range(8):
                model, theta, _ = M.disc(radius, tuple(rng.uniform(0, h, 2)))
                s = C.classify(model, theta, grid, n_sub=n_sub, mode=mode, alpha=a)
                obj = jax.jit(lambda t, s=s: C.objective(t, s))
                j = float(obj(theta))
                g = np.asarray(jax.grad(obj)(theta))
                had = np.asarray(C.hadamard(theta, s))
                j_err.append((j - j_exact) / j_exact)
                g_err.append((g[0] - dj_exact) / dj_exact)
                c_err.append(np.abs(g[1:]).max())
                h_err.append((had[0] - dj_exact) / dj_exact)
            rows.append(
                [
                    mode,
                    f"1/{round(1 / h)}",
                    f"[{sci(min(j_err))}, {sci(max(j_err))}]",
                    f"[{sci(min(g_err))}, {sci(max(g_err))}]",
                    f"{max(c_err):.1e}",
                    f"[{sci(min(h_err))}, {sci(max(h_err))}]",
                ]
            )
    print(
        table(
            ["mode", "h", "J rel err", "dJ/dR rel err by AD", "max |dJ/dc|", "Hadamard rel err"],
            rows,
        )
    )
    print()


def delta():
    model, theta, _ = M.blended_union()
    h = 1 / 32
    print(f"## The finite-difference step (disc ∪ box, h = 1/{round(1 / h)})\n")
    rows = []
    for mode, a, n_sub in MODES:
        s = _structure(model, theta, BLEND_BOX, h, mode, a, n_sub)
        obj = jax.jit(lambda t, s=s: C.objective(t, s))
        g = np.asarray(jax.grad(obj)(theta))
        corners = s.x_sub[s.cut].reshape(-1, 2)
        f_sub = np.asarray(M.field(model)(theta, corners))
        jac = np.asarray(M.parameter_jacobian(model)(theta, corners))

        def free(t, s=s):
            return C.objective(t, C.classify(model, t, s.grid, s.n_sub, s.mode, s.eps / s.h))

        for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            fixed = C.central_difference(obj, theta, d)
            reclassified = C.central_difference(free, theta, d)
            flips = [int((np.abs(f_sub) < d * np.abs(jac[:, k])).sum()) for k in range(3)]
            rows.append(
                [
                    mode,
                    f"{d:.0e}",
                    " ".join(str(n) for n in flips),
                    f"{_rel(reclassified, g):.1e}",
                    f"{_rel(fixed, g):.1e}",
                ]
            )
    print(
        table(
            [
                "mode",
                "δ",
                "sub-corner flips within δ (R, k, hy)",
                "FD re-classified vs AD",
                "FD fixed vs AD",
            ],
            rows,
        )
    )
    print()


SECTIONS = {
    "convergence": convergence,
    "blend": blend,
    "alpha": alpha,
    "penalty": penalty,
    "phase": phase,
    "delta": delta,
}


def main():
    names = sys.argv[1:] or list(SECTIONS)
    for name in names:
        t0 = time.time()
        SECTIONS[name]()
        print(f"({name}: {time.time() - t0:.0f} s)\n")


if __name__ == "__main__":
    main()
