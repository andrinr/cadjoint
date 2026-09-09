"""Section-by-section comparison: what is wrong, not just how much.

The global IoU in :mod:`research.reverse.overlay` says a model disagrees with
its reference.  It cannot say *how*.  A section can: put the casting's
silhouette and the model's on the same axes at the heights where the part
changes character, and the answer reads off the picture — "the outline is
eight millimetres proud along the whole −x flank between z 280 and 380" — in a
way no scalar will ever produce.

So this module draws sections and scores each one on its own.  The planes to
use are the ones where the part stops being one thing and starts being
another: for a cylinder block, the deck, the jacket band, the jacket floor,
the bore bay, the bulkheads and the pan rail, plus cross-sections through a
bore and through a web.  Those are not general — they are this part's
skeleton — so they are passed in rather than guessed.

Each section reports its own area IoU together with the two one-sided errors
in square centimetres, which is what turns "0.36" into a work list ordered by
how much iron each item is worth.

Usage::

    python -m research.reverse.sections --scene scenes.cylinder_block \\
        --mesh-cache cache.npz --png sections.png [--json sections.json] \\
        [--planes "z=412,z=380,y=91"]

Planes are given in the STEP's own world millimetres, because that is the
frame the reference images are in and the point is to compare like with like.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any

import numpy as np

#: The heights and stations where this casting changes character.  Deck, the
#: jacket band top and middle, the jacket floor, the bore bay, the bulkheads
#: and the pan rail; then across a bore and across a web.
DEFAULT_PLANES = (
    "z=412",
    "z=380",
    "z=300",
    "z=285",
    "z=250",
    "z=200",
    "y=5",
    "y=48",
    "y=91",
    "y=134",
    "y=263",
)

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_plane(text: str) -> tuple[int, float]:
    """``"z=412"`` to ``(2, 412.0)``."""
    axis, _, value = text.partition("=")
    axis = axis.strip().lower()
    if axis not in AXIS_INDEX:
        raise ValueError(f"a plane is 'x=', 'y=' or 'z=' followed by a number, got {text!r}")
    return AXIS_INDEX[axis], float(value)


def section(
    vertices: np.ndarray,
    triangles: np.ndarray,
    field,
    frame: dict[str, Any],
    axis: int,
    offset: float,
    step_mm: float = 1.0,
    margin_mm: float = 8.0,
) -> dict[str, Any]:
    """Both silhouettes on one plane, and how much of each the other has.

    Args:
        vertices: The STEP's tessellation, world millimetres.
        triangles: Its triangles.
        field: The scene's SDF, taking scene-unit points.
        frame: From :func:`research.reverse.overlay.reference_frame`.
        axis: 0, 1 or 2 — which coordinate the plane is normal to.
        offset: Where along that coordinate the plane sits.
        step_mm: Raster pitch.  One millimetre is affordable on a plane and
            resolves a 5 mm barrel wall five ways, which the 3 mm lattice the
            global score uses does not.
        margin_mm: How far past the casting's box to raster, so material the
            model has and the casting does not is seen rather than cropped.

    Returns:
        The two boolean rasters, the axes they are on, and the scores.
    """
    from research.reverse.mesh_query import RayGrid
    from research.reverse.overlay import to_scene

    u, v = (a for a in range(3) if a != axis)
    low = vertices.min(axis=0) - margin_mm
    high = vertices.max(axis=0) + margin_mm
    us = np.arange(low[u], high[u] + step_mm, step_mm)
    vs = np.arange(low[v], high[v] + step_mm, step_mm)

    grid = RayGrid(vertices, triangles, axis=axis, cell=4.0)
    truth, odd = grid.inside_lattice(us, vs, np.array([offset]))
    truth = truth[:, :, 0]

    uu, vv = np.meshgrid(us, vs, indexing="ij")
    points = np.empty((uu.size, 3))
    points[:, u] = uu.ravel()
    points[:, v] = vv.ravel()
    points[:, axis] = offset
    values = np.concatenate([field(to_scene(c, frame)) for c in np.array_split(points, 16)])
    model = (values < 0.0).reshape(truth.shape)

    both = int((truth & model).sum())
    union = int((truth | model).sum())
    cell = step_mm**2 / 100.0
    return {
        "axis": "xyz"[axis],
        "offset": offset,
        "iou": both / union if union else 1.0,
        "step_cm2": float((truth).sum()) * cell,
        "model_cm2": float((model).sum()) * cell,
        "missing_cm2": float((truth & ~model).sum()) * cell,
        "added_cm2": float((model & ~truth).sum()) * cell,
        "odd_rays": odd,
        "_us": us,
        "_vs": vs,
        "_truth": truth,
        "_model": model,
        "_uv": ("xyz"[u], "xyz"[v]),
    }


def draw(results: list[dict[str, Any]], path: str, title: str = "") -> None:
    """One panel per section: the casting filled, the model outlined over it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    columns = min(4, len(results))
    rows = (len(results) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.8 * rows), squeeze=False)
    cmap = ListedColormap(["#ffffff", "#d62728", "#1f77b4", "#9e9e9e"])
    for ax, result in zip(axes.ravel(), results):
        code = result["_truth"].astype(np.uint8) * 2 + result["_model"].astype(np.uint8)
        ax.imshow(
            code.T,
            origin="lower",
            extent=[result["_us"][0], result["_us"][-1], result["_vs"][0], result["_vs"][-1]],
            cmap=cmap,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        ax.set_title(
            f"{result['axis']} = {result['offset']:.0f}   IoU {result['iou']:.3f}\n"
            f"missing {result['missing_cm2']:.0f} cm², added {result['added_cm2']:.0f} cm²",
            fontsize=9,
        )
        ax.set_xlabel(result["_uv"][0])
        ax.set_ylabel(result["_uv"][1])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, color="k", lw=0.3)
    for ax in axes.ravel()[len(results) :]:
        ax.axis("off")
    fig.suptitle(f"{title}   grey both · blue casting only · red model only", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def format_table(results: list[dict[str, Any]]) -> str:
    lines = [
        f"{'plane':>10} {'IoU':>7} {'casting':>9} {'model':>9} {'missing':>9} {'added':>9}",
        f"{'':>10} {'':>7} {'cm2':>9} {'cm2':>9} {'cm2':>9} {'cm2':>9}",
    ]
    for r in results:
        lines.append(
            f"{r['axis'] + ' = ' + format(r['offset'], '.0f'):>10} {r['iou']:7.3f} "
            f"{r['step_cm2']:9.1f} {r['model_cm2']:9.1f} {r['missing_cm2']:9.1f} "
            f"{r['added_cm2']:9.1f}"
        )
    mean = float(np.mean([r["iou"] for r in results]))
    lines.append(f"{'mean':>10} {mean:7.3f}")
    worst = sorted(results, key=lambda r: r["iou"])[:3]
    lines.append(
        "worst: " + ", ".join(f"{r['axis']}={r['offset']:.0f} ({r['iou']:.3f})" for r in worst)
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="scenes.cylinder_block")
    parser.add_argument("--attribute", default="block")
    parser.add_argument("--step")
    parser.add_argument("--mesh-cache")
    parser.add_argument("--planes", default=",".join(DEFAULT_PLANES))
    parser.add_argument("--step-mm", type=float, default=1.0)
    parser.add_argument("--png")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)

    module = importlib.import_module(args.scene)
    from research.reverse.mesh_query import load_mesh
    from research.reverse.overlay import reference_frame, scene_field
    from research.reverse.step_probe import read_step

    frame = reference_frame(module)
    path = args.step or frame["step"]
    if not path or not os.path.exists(path):
        print(f"the reference STEP is not on this machine: {path!r}", file=sys.stderr)
        return 2

    if args.mesh_cache and os.path.exists(args.mesh_cache):
        vertices, triangles = load_mesh(args.mesh_cache)
    else:
        part = read_step(path, mesh_cache=args.mesh_cache)
        vertices, triangles = part.vertices, part.triangles

    field = scene_field(module, args.attribute)
    results = [
        section(vertices, triangles, field, frame, *parse_plane(p), step_mm=args.step_mm)
        for p in args.planes.split(",")
        if p.strip()
    ]
    print(format_table(results))

    if args.png:
        draw(results, args.png, title=f"{args.scene}.{args.attribute} vs the casting")
        print(f"\nwrote {args.png}")
    if args.json_path:
        payload = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
