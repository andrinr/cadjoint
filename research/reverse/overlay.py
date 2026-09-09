"""How close is a cadjoint scene to the solid it was reverse-engineered from?

A parametric model of a casting is an *idealisation*, so "does it match?" is
the wrong question and "where does it not, and did I mean that?" is the right
one.  This module answers the second.  It samples both solids on the same
lattice and reports:

* **agreement** — intersection over union, plus the two one-sided errors kept
  apart, because they mean opposite things.  Iron the model has and the STEP
  does not is *added* material (a fillet not in the casting, a cored passage
  left solid); iron the STEP has and the model does not is *missing* material
  (a boss, a rib, a flange nobody modelled).  One number hides which.
* **where** — the disagreement broken down by height, so a model that is right
  everywhere except the crankcase says so instead of averaging it away.
* **how far** — the model's own signed distance sampled at points on the
  STEP's surface.  Zero would be an exact match; the distribution's spread is
  the deviation in millimetres, in the units of the part.
* **a picture** — plan and section rasters colour-coded four ways (both,
  neither, model only, STEP only), because a human reading one of those images
  finds a whole missing feature in a second and a table of percentages does
  not.

The scene declares the frame.  A CAD part sits wherever its author left it and
in whatever axes they used; the scene is authored on a sane origin.  Rather
than pass a transform on the command line every time, a scene that was built
from a STEP carries a ``REFERENCE`` dict saying how the two relate — see
``scenes/cylinder_block.py``.  A scene without one can still be compared by
passing ``--origin`` and ``--axes``.

Usage::

    python -m research.reverse.overlay --scene scenes.cylinder_block \\
        [--step PATH] [--mesh-cache cache.npz] [--step-mm 3.0] \\
        [--png overlay.png] [--json overlay.json]

The STEP is never copied into the repository, and neither this module nor the
scene needs it to import.  Without the file the scene still builds and its
tests still run; only the comparison is unavailable, which is the right way
round for a reference that lives on one machine.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any

import numpy as np


def scene_field(module: Any, attribute: str = "block"):
    """The scene's SDF as a plain ``(n, 3) -> (n,)`` callable, in scene units."""
    import jax
    import jax.numpy as jnp

    solid = getattr(module, attribute)
    evaluate = jax.jit(jax.vmap(solid))

    def field(points: np.ndarray) -> np.ndarray:
        return np.asarray(evaluate(jnp.asarray(points, dtype=jnp.float32)))

    return field


def reference_frame(module: Any, origin=None, axes=None, unit_mm=None) -> dict[str, Any]:
    """The world-mm to scene-unit map, from the scene or from the arguments."""
    declared = getattr(module, "REFERENCE", {}) or {}
    frame = {
        "step": declared.get("step"),
        "origin_mm": np.asarray(origin if origin is not None else declared["origin_mm"], float),
        "axes": np.asarray(axes if axes is not None else declared["axes"], float),
        "unit_mm": float(unit_mm if unit_mm is not None else declared["unit_mm"]),
    }
    return frame


def to_scene(points_mm: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    """World millimetres to scene units."""
    return ((points_mm - frame["origin_mm"]) @ frame["axes"].T) / frame["unit_mm"]


def compare(
    vertices: np.ndarray,
    triangles: np.ndarray,
    field,
    frame: dict[str, Any],
    step_mm: float = 3.0,
    margin_mm: float = 6.0,
) -> dict[str, Any]:
    """Sample both solids on one lattice and score the agreement.

    Args:
        vertices: The STEP's tessellation, in world millimetres.
        triangles: Its triangles.
        field: The scene's SDF, taking scene-unit points.
        frame: From :func:`reference_frame`.
        step_mm: Lattice pitch.  Three millimetres over a 350 mm casting is
            about two million samples and a few seconds; one millimetre is
            fifty times that and says the same thing.
        margin_mm: How far outside the STEP's box to sample, so that material
            the model has and the casting does not is counted rather than
            cropped away.

    Returns:
        A dict of scores, the per-height breakdown, and the two occupancy
        lattices themselves for anything that wants to draw them.
    """
    from research.reverse.mesh_query import RayGrid

    low = vertices.min(axis=0) - margin_mm
    high = vertices.max(axis=0) + margin_mm
    axes_mm = [np.arange(low[i], high[i] + step_mm, step_mm) for i in range(3)]

    grid = RayGrid(vertices, triangles, axis=2, cell=max(4.0, step_mm))
    truth, odd = grid.inside_lattice(axes_mm[0], axes_mm[1], axes_mm[2])

    xx, yy, zz = np.meshgrid(*axes_mm, indexing="ij")
    points_mm = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    values = np.concatenate(
        [field(to_scene(chunk, frame)) for chunk in np.array_split(points_mm, 64)]
    )
    model = (values < 0.0).reshape(truth.shape)

    both = int((truth & model).sum())
    only_truth = int((truth & ~model).sum())
    only_model = int((model & ~truth).sum())
    union = both + only_truth + only_model
    cell_cm3 = step_mm**3 / 1000.0

    by_height = []
    for k, z in enumerate(axes_mm[2]):
        t, m = truth[:, :, k], model[:, :, k]
        inter = int((t & m).sum())
        uni = int((t | m).sum())
        if uni == 0:
            continue
        by_height.append(
            {
                "z_mm": float(z),
                "iou": inter / uni,
                "step_only_cm2": float((t & ~m).sum()) * step_mm**2 / 100.0,
                "model_only_cm2": float((m & ~t).sum()) * step_mm**2 / 100.0,
            }
        )

    # What could a *perfect* model score? This part's mean wall is 2V/A ~ 6.6 mm
    # and its surface runs to 11,000 cm^2, so a lattice IoU is dominated by
    # boundary cells: displacing the casting against itself by a millimetre --
    # which is the error a very good idealisation makes -- already costs a
    # fifth of the score. Reporting the ceiling beside the score is the
    # difference between "0.36 is bad" and "0.36 is what a 4 mm surface error
    # looks like on a part made of 6 mm walls".
    ceiling = {}
    for offset in (1.0, 2.0):
        shifted, _ = grid.inside_lattice(axes_mm[0] + offset, axes_mm[1] + offset, axes_mm[2])
        ceiling[f"{offset:g}mm"] = float((truth & shifted).sum() / max((truth | shifted).sum(), 1))

    return {
        "step_mm": step_mm,
        "lattice": [len(a) for a in axes_mm],
        "ceiling_iou": ceiling,
        "odd_rays": odd,
        "iou": both / union if union else 0.0,
        "recall": both / (both + only_truth) if (both + only_truth) else 0.0,
        "precision": both / (both + only_model) if (both + only_model) else 0.0,
        "step_volume_cm3": (both + only_truth) * cell_cm3,
        "model_volume_cm3": (both + only_model) * cell_cm3,
        "missing_cm3": only_truth * cell_cm3,
        "added_cm3": only_model * cell_cm3,
        "by_height": by_height,
        "_axes": axes_mm,
        "_truth": truth,
        "_model": model,
    }


def surface_deviation(
    vertices: np.ndarray,
    triangles: np.ndarray,
    field,
    frame: dict[str, Any],
    samples: int = 40000,
    seed: int = 0,
) -> dict[str, Any]:
    """The model's own distance, sampled at points on the STEP's surface.

    Args:
        vertices: STEP tessellation vertices, world millimetres.
        triangles: Its triangles.
        field: The scene's SDF.
        frame: From :func:`reference_frame`.
        samples: How many surface points to draw, area-weighted.
        seed: RNG seed, so a re-run of a comparison reports the same number.

    Returns:
        Percentiles of the signed deviation in millimetres.  A perfect match
        would be zero everywhere; a positive value is the model's surface
        lying outside that point, a negative one inside it.
    """
    rng = np.random.default_rng(seed)
    tri = vertices[triangles]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(tri), size=samples, p=area / area.sum())
    u = rng.random((samples, 1))
    v = rng.random((samples, 1))
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    chosen = tri[pick]
    points = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0]) + v * (chosen[:, 2] - chosen[:, 0])

    values = np.concatenate([field(to_scene(chunk, frame)) for chunk in np.array_split(points, 16)])
    deviation = values * frame["unit_mm"]
    return {
        "samples": samples,
        "median_abs_mm": float(np.median(np.abs(deviation))),
        "mean_abs_mm": float(np.mean(np.abs(deviation))),
        "p05_mm": float(np.percentile(deviation, 5)),
        "p50_mm": float(np.percentile(deviation, 50)),
        "p95_mm": float(np.percentile(deviation, 95)),
        "within_2mm": float(np.mean(np.abs(deviation) < 2.0)),
        "within_5mm": float(np.mean(np.abs(deviation) < 5.0)),
        "within_10mm": float(np.mean(np.abs(deviation) < 10.0)),
    }


def draw(result: dict[str, Any], path: str, title: str = "") -> None:
    """Four-colour rasters of the agreement, on plans and on sections."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    axes_mm = result["_axes"]
    truth, model = result["_truth"], result["_model"]
    # 0 neither, 1 model only, 2 STEP only, 3 both.
    code = truth.astype(np.uint8) * 2 + model.astype(np.uint8)
    cmap = ListedColormap(["#ffffff", "#d62728", "#1f77b4", "#bdbdbd"])

    zs = axes_mm[2]
    plan_at = [zs[int(f * (len(zs) - 1))] for f in (0.06, 0.25, 0.45, 0.65, 0.82, 0.97)]
    ys = axes_mm[1]
    section_at = [ys[int(f * (len(ys) - 1))] for f in (0.25, 0.5, 0.75)]

    fig, grid = plt.subplots(3, 3, figsize=(19, 15))
    for ax, z in zip(grid.ravel()[:6], plan_at):
        k = int(np.argmin(np.abs(zs - z)))
        ax.imshow(
            code[:, :, k].T,
            origin="lower",
            extent=[axes_mm[0][0], axes_mm[0][-1], axes_mm[1][0], axes_mm[1][-1]],
            cmap=cmap,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        ax.set_title(f"plan z = {z:.0f} mm", fontsize=9)
    for ax, y in zip(grid.ravel()[6:], section_at):
        j = int(np.argmin(np.abs(ys - y)))
        ax.imshow(
            code[:, j, :].T,
            origin="lower",
            extent=[axes_mm[0][0], axes_mm[0][-1], axes_mm[2][0], axes_mm[2][-1]],
            cmap=cmap,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        ax.set_title(f"section y = {y:.0f} mm", fontsize=9)
    for ax in grid.ravel():
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{title}   grey both · blue STEP only (missing) · red model only (added)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def format_report(result: dict[str, Any], deviation: dict[str, Any] | None) -> str:
    lines = []
    add = lines.append
    add(f"lattice          {result['lattice']} at {result['step_mm']} mm")
    add(f"odd rays         {result['odd_rays']} (columns the parity test could not pair)")
    add("")
    add(f"IoU              {result['iou']:.4f}")
    for label, value in (result.get("ceiling_iou") or {}).items():
        add(f"  ceiling        {value:.4f}   the casting against itself displaced {label}")
    add(f"recall           {result['recall']:.4f}   (of the casting, how much the model has)")
    add(f"precision        {result['precision']:.4f}   (of the model, how much is really there)")
    add("")
    add(f"STEP volume      {result['step_volume_cm3']:9.1f} cm^3")
    add(f"model volume     {result['model_volume_cm3']:9.1f} cm^3")
    add(f"missing          {result['missing_cm3']:9.1f} cm^3  (casting has it, model does not)")
    add(f"added            {result['added_cm3']:9.1f} cm^3  (model has it, casting does not)")
    if deviation:
        add("")
        add("signed deviation at points on the casting's surface, mm")
        add(f"  median |d|     {deviation['median_abs_mm']:8.2f}")
        add(f"  mean |d|       {deviation['mean_abs_mm']:8.2f}")
        add(
            f"  5 / 50 / 95    {deviation['p05_mm']:7.2f} {deviation['p50_mm']:7.2f} "
            f"{deviation['p95_mm']:7.2f}"
        )
        add(
            f"  within 2/5/10  {deviation['within_2mm']:.3f} {deviation['within_5mm']:.3f} "
            f"{deviation['within_10mm']:.3f}"
        )
    add("")
    add("where the disagreement is, by height")
    add(f"  {'z mm':>8} {'IoU':>7} {'missing cm2':>12} {'added cm2':>11}")
    rows = result["by_height"]
    for entry in rows[:: max(1, len(rows) // 24)]:
        add(
            f"  {entry['z_mm']:8.0f} {entry['iou']:7.3f} {entry['step_only_cm2']:12.1f} "
            f"{entry['model_only_cm2']:11.1f}"
        )
    worst = sorted(rows, key=lambda e: e["iou"])[:5]
    add("")
    add("worst five slices: " + ", ".join(f"z={e['z_mm']:.0f} ({e['iou']:.2f})" for e in worst))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="scenes.cylinder_block")
    parser.add_argument("--attribute", default="block", help="the SDF on the scene to compare")
    parser.add_argument("--step", help="the STEP file; defaults to the scene's REFERENCE")
    parser.add_argument("--mesh-cache", help="npz to cache the tessellation in")
    parser.add_argument("--step-mm", type=float, default=3.0)
    parser.add_argument("--deflection", type=float, default=0.35)
    parser.add_argument("--png")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--no-deviation", action="store_true")
    args = parser.parse_args(argv)

    module = importlib.import_module(args.scene)
    frame = reference_frame(module)
    path = args.step or frame["step"]
    if not path or not os.path.exists(path):
        print(f"the reference STEP is not on this machine: {path!r}", file=sys.stderr)
        return 2

    from research.reverse.mesh_query import load_mesh
    from research.reverse.step_probe import read_step

    if args.mesh_cache and os.path.exists(args.mesh_cache):
        vertices, triangles = load_mesh(args.mesh_cache)
    else:
        part = read_step(path, deflection=args.deflection, mesh_cache=args.mesh_cache)
        vertices, triangles = part.vertices, part.triangles

    field = scene_field(module, args.attribute)
    result = compare(vertices, triangles, field, frame, step_mm=args.step_mm)
    deviation = None if args.no_deviation else surface_deviation(vertices, triangles, field, frame)
    print(format_report(result, deviation))

    if args.png:
        draw(result, args.png, title=f"{args.scene}.{args.attribute} vs {os.path.basename(path)}")
        print(f"\nwrote {args.png}")
    if args.json_path:
        payload = {k: v for k, v in result.items() if not k.startswith("_")}
        payload["deviation"] = deviation
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
