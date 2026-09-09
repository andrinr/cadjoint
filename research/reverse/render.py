"""Side-by-side render of a casting and the scene built from it, on the CPU.

The viewer answers "does my scene look right?" with a GPU, and sometimes the
GPU is not available — it is shared with a test suite, or the work is running
somewhere headless.  This module answers the same question with numpy and a
sphere tracer, which is slower and entirely sufficient for the one thing that
matters here: putting the reference and the model side by side, lit
identically, from the same camera.

Both solids are rendered the same way on purpose.  The casting is a triangle
soup and the scene is a distance field, and it would be easy to flatter one by
shading it better than the other; instead both produce a depth buffer and both
are shaded from that buffer's own gradient.  Whatever is left is geometry.

The camera is **orthographic**, which is what makes the mesh side cheap: rotate
the triangles into camera coordinates and the depth along every pixel is an
axis-aligned ray query, which :mod:`research.reverse.mesh_query` already does.
The scene side is sphere-traced along the same rays.

Usage::

    python -m research.reverse.render --scene scenes.cylinder_block \\
        --mesh-cache cache.npz --png compare.png \\
        [--azimuth 35 --elevation 22] [--pixels 420]

Azimuth is measured about the casting's up axis from the model's +x, and
elevation above the horizon, so ``--azimuth 35 --elevation 22`` is the
three-quarter view a catalogue photograph of a block is usually taken from.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys

import numpy as np


def camera_frame(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Rows: right, up, forward — an orthonormal frame looking at the origin."""
    a = math.radians(azimuth_deg)
    e = math.radians(elevation_deg)
    forward = np.array([-math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), -math.sin(e)])
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, forward])


def mesh_depth(vertices, triangles, frame, pixels, bounds, margin=1.06):
    """Nearest-surface depth per pixel, by rotating the soup into camera space."""
    from research.reverse.mesh_query import RayGrid

    local = vertices @ frame.T
    centre = 0.5 * (bounds[0] + bounds[1]) @ frame.T
    half = margin * 0.5 * np.abs(bounds[1] - bounds[0]) @ np.abs(frame.T)
    span = float(max(half[0], half[1]))
    us = np.linspace(centre[0] - span, centre[0] + span, pixels)
    vs = np.linspace(centre[1] - span, centre[1] + span, pixels)

    grid = RayGrid(local, triangles, axis=2, cell=4.0)
    depth = np.full((pixels, pixels), np.nan)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            hits = grid.crossings(float(u), float(v))
            if len(hits):
                depth[i, j] = hits.min()
    return depth, (us, vs)


def sdf_depth(field, frame, pixels, bounds, unit_mm, origin_mm, axes, margin=1.06, steps=96):
    """Nearest-surface depth per pixel, by sphere tracing the scene's field."""
    import jax
    import jax.numpy as jnp

    centre = 0.5 * (bounds[0] + bounds[1]) @ frame.T
    half = margin * 0.5 * np.abs(bounds[1] - bounds[0]) @ np.abs(frame.T)
    span = float(max(half[0], half[1]))
    us = np.linspace(centre[0] - span, centre[0] + span, pixels)
    vs = np.linspace(centre[1] - span, centre[1] + span, pixels)
    uu, vv = np.meshgrid(us, vs, indexing="ij")

    near = centre[2] - 1.4 * span
    far = centre[2] + 1.4 * span
    inverse = frame.T  # camera coordinates back to world
    starts = np.stack([uu.ravel(), vv.ravel(), np.full(uu.size, near)], axis=-1) @ inverse.T
    direction = inverse @ np.array([0.0, 0.0, 1.0])

    scene_start = jnp.asarray(((starts - origin_mm) @ axes.T) / unit_mm, dtype=jnp.float32)
    scene_dir = jnp.asarray((direction @ axes.T) / np.linalg.norm(direction), dtype=jnp.float32)
    limit = float((far - near) / unit_mm)

    def march(origin):
        def body(_, state):
            t, done = state
            d = field(origin + t * scene_dir)
            step = jnp.clip(d, 0.0, 0.05)
            hit = d < 2.0e-4
            return (jnp.where(done | hit, t, t + step), done | hit | (t > limit))

        t, done = jax.lax.fori_loop(0, steps, body, (jnp.float32(0.0), jnp.bool_(False)))
        return jnp.where(done & (t <= limit), t, jnp.nan)

    traced = jax.jit(jax.vmap(march))
    out = np.concatenate([np.asarray(traced(chunk)) for chunk in jnp.array_split(scene_start, 24)])
    return near + out.reshape(uu.shape) * unit_mm, (us, vs)


def shade(depth):
    """Lambert shading from the depth buffer's own gradient."""
    filled = np.where(np.isnan(depth), np.nanmax(depth) + 30.0, depth)
    gy, gx = np.gradient(filled)
    normal = np.stack([-gx, -gy, np.ones_like(filled)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([0.42, 0.55, 0.72])
    light /= np.linalg.norm(light)
    lambert = np.clip(normal @ light, 0.0, 1.0)
    image = 0.20 + 0.80 * lambert**0.9
    return np.where(np.isnan(depth), np.nan, image)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="scenes.cylinder_block")
    parser.add_argument("--attribute", default="block")
    parser.add_argument("--step")
    parser.add_argument("--mesh-cache")
    parser.add_argument("--azimuth", type=float, default=35.0)
    parser.add_argument("--elevation", type=float, default=22.0)
    parser.add_argument("--pixels", type=int, default=420)
    parser.add_argument("--png", default="compare.png")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    module = importlib.import_module(args.scene)
    from research.reverse.mesh_query import load_mesh
    from research.reverse.overlay import reference_frame
    from research.reverse.step_probe import read_step

    frame_spec = reference_frame(module)
    path = args.step or frame_spec["step"]
    if not path or not os.path.exists(path):
        print(f"the reference STEP is not on this machine: {path!r}", file=sys.stderr)
        return 2
    if args.mesh_cache and os.path.exists(args.mesh_cache):
        vertices, triangles = load_mesh(args.mesh_cache)
    else:
        part = read_step(path, mesh_cache=args.mesh_cache)
        vertices, triangles = part.vertices, part.triangles

    # Render in the SCENE's frame, so "azimuth 0" means the same thing to both.
    origin = frame_spec["origin_mm"]
    axes = frame_spec["axes"]
    unit = frame_spec["unit_mm"]
    world = (vertices - origin) @ axes.T
    bounds = np.stack([world.min(axis=0), world.max(axis=0)])
    frame = camera_frame(args.azimuth, args.elevation)

    left, _ = mesh_depth(world, triangles, frame, args.pixels, bounds)
    # The mesh was already put into the scene's frame above, so the tracer is
    # given an identity map and only has to divide millimetres by the unit.
    right, _ = sdf_depth(
        getattr(module, args.attribute),
        frame,
        args.pixels,
        bounds,
        unit,
        np.zeros(3),
        np.eye(3),
    )

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for ax, depth, title in (
        (axs[0], left, "the casting (STEP, 4069 faces)"),
        (axs[1], right, f"{args.scene}.{args.attribute}"),
    ):
        ax.imshow(
            np.ma.masked_invalid(shade(depth)).T,
            origin="lower",
            cmap="bone",
            vmin=0.0,
            vmax=1.05,
            interpolation="bilinear",
        )
        ax.set_title(title, fontsize=11)
        ax.set_facecolor("#eef1f4")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"orthographic, azimuth {args.azimuth:.0f} elevation {args.elevation:.0f}", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(args.png, dpi=110)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
