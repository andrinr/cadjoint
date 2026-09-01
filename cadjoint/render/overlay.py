"""Construction-tree overlay: draw sketches on top of rendered images.

The construction tree (sketch planes, profiles) is never part of the SDF
render. Instead it is projected through the same camera and drawn as a
wireframe overlay — edges, vertex handles, and plane outlines — on the
matplotlib axes showing the rendered image.

Example:
    ```python
    image = render_scene(scene, settings)
    fig, ax = plt.subplots()
    ax.imshow(image)
    draw_plane(ax, profile.plane, scene.camera, settings.resolution)
    draw_profile(ax, profile, scene.camera, settings.resolution)
    ```
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from cadjoint.render._plotting import require_matplotlib
from cadjoint.render.scene import Camera


def project_points(
    points: Array,
    camera: Camera,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space points through a camera to pixel coordinates.

    Inverts the perspective ray construction used by the forward renderer, so
    overlay geometry lands exactly on the rendered pixels.

    Args:
        points: World-space points, shape (N, 3).
        camera: The camera the image was rendered with.
        resolution: (height, width) of the rendered image in pixels.

    Returns:
        pixels: (N, 2) float array of (col, row) pixel coordinates.
        valid: (N,) bool mask — False for points at or behind the camera.
    """
    h, w = resolution
    pos = jnp.asarray(camera.position, dtype=jnp.float32)
    target = jnp.asarray(camera.target, dtype=jnp.float32)

    def _normalize(v):
        return v / jnp.sqrt(jnp.sum(v**2) + 1e-12)

    forward = _normalize(target - pos)
    world_up = jnp.array([0.0, 1.0, 0.0])
    right = _normalize(jnp.cross(forward, world_up))
    down = _normalize(jnp.cross(right, forward))

    v = jnp.asarray(points, dtype=jnp.float32) - pos
    z = v @ forward
    valid = z > 1e-6
    z_safe = jnp.where(valid, z, 1.0)
    x = (v @ right) / z_safe
    y = (v @ down) / z_safe

    fx = camera.fov
    fy = fx / w * h
    col = (x + fx) * (w - 1) / (2.0 * fx)
    row = (fy - y) * (h - 1) / (2.0 * fy)
    pixels = jnp.stack([col, row], axis=-1)
    return np.asarray(pixels), np.asarray(valid)


def draw_polyline(
    ax,
    points_world: Array,
    camera: Camera,
    resolution: tuple[int, int],
    **plot_kwargs,
) -> None:
    """Project and draw a world-space polyline on an image axes."""
    pixels, valid = project_points(jnp.asarray(points_world), camera, resolution)
    if not valid.all():
        return  # skip segments crossing the camera plane
    ax.plot(pixels[:, 0], pixels[:, 1], **plot_kwargs)


def draw_profile(
    ax,
    profile,
    camera: Camera,
    resolution: tuple[int, int],
    color: str = "#d9ff57",
    vertex_color: str = "#ff8167",
    linewidth: float = 1.6,
    vertex_size: float = 28.0,
) -> None:
    """Draw a PolygonProfile: closed edge loop plus vertex handles.

    Args:
        ax: Matplotlib axes already showing the rendered image.
        profile: :class:`~cadjoint.construction.sketch.PolygonProfile`.
        camera: Camera the image was rendered with.
        resolution: (height, width) of the rendered image.
        color: Edge color.
        vertex_color: Vertex handle color.
        linewidth: Edge line width.
        vertex_size: Vertex marker area (matplotlib scatter ``s``).
    """
    require_matplotlib()
    draw_polyline(ax, profile.edges_world(), camera, resolution, color=color, lw=linewidth)
    pixels, valid = project_points(profile.world_vertices(), camera, resolution)
    ax.scatter(
        pixels[valid, 0],
        pixels[valid, 1],
        s=vertex_size,
        c=vertex_color,
        zorder=5,
        edgecolors="black",
        linewidths=0.5,
    )


def draw_plane(
    ax,
    plane,
    camera: Camera,
    resolution: tuple[int, int],
    extent: float = 1.5,
    color: str = "#95958f",
    linewidth: float = 0.9,
) -> None:
    """Draw a SketchPlane as a dashed square outline of size ``2·extent``.

    Args:
        ax: Matplotlib axes already showing the rendered image.
        plane: :class:`~cadjoint.construction.sketch.SketchPlane`.
        camera: Camera the image was rendered with.
        resolution: (height, width) of the rendered image.
        extent: Half-size of the drawn plane patch in plane units.
        color: Outline color.
        linewidth: Outline line width.
    """
    require_matplotlib()
    e = extent
    corners = jnp.array([[-e, -e], [e, -e], [e, e], [-e, e], [-e, -e]])
    draw_polyline(
        ax, plane.to_world(corners), camera, resolution, color=color, lw=linewidth, ls="--"
    )
