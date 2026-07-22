"""Marching cubes mesh extraction and rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def render_marching_cubes(
    sdf: Callable[[Array], Array],
    bounds: tuple[float, float, float] = (-3, -3, -3),
    size: tuple[float, float, float] = (6, 6, 6),
    resolution: int = 50,
    ax: Axes | None = None,
    color: str = "cyan",
    alpha: float = 0.7,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> Axes:
    """Render an SDF by extracting a mesh with marching cubes.

    Requires ``scikit-image``.

    Args:
        sdf: Signed distance function, callable ``(point: Array[3]) → Array[]``.
        bounds: Lower corner (x, y, z) of the evaluation volume.
        size: Extent in each dimension (dx, dy, dz).
        resolution: Grid points per dimension.
        ax: Existing 3D axes; creates new figure if None.
        color: Mesh face colour.
        alpha: Mesh transparency.
        title: Axes title.
        figsize: Figure size in inches (width, height).

    Returns:
        The matplotlib 3D axes.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    try:
        from skimage import measure
    except ImportError as err:
        raise ImportError(
            'render_marching_cubes requires the render extra. Install with: pip install "jaxcad[render]"'
        ) from err

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    x = jnp.linspace(bounds[0], bounds[0] + size[0], resolution)
    y = jnp.linspace(bounds[1], bounds[1] + size[1], resolution)
    z = jnp.linspace(bounds[2], bounds[2] + size[2], resolution)

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    volume = np.array(jax.vmap(sdf)(points).reshape(resolution, resolution, resolution))

    try:
        verts, faces, _, _ = measure.marching_cubes(
            volume,
            level=0.0,
            spacing=(size[0] / resolution, size[1] / resolution, size[2] / resolution),
        )
        verts[:, 0] += bounds[0]
        verts[:, 1] += bounds[1]
        verts[:, 2] += bounds[2]

        mesh = Poly3DCollection(verts[faces], alpha=alpha, edgecolor="k", linewidths=0.1)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)

        ax.set_xlim(bounds[0], bounds[0] + size[0])
        ax.set_ylim(bounds[1], bounds[1] + size[1])
        ax.set_zlim(bounds[2], bounds[2] + size[2])

    except (ValueError, RuntimeError) as e:
        print(f"Warning: marching cubes failed — {e}")
        print("The SDF may not have a zero-level surface within the given bounds.")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title or "3D Mesh (Marching Cubes)", fontsize=12)
    return ax
