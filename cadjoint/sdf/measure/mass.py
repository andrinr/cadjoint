"""Differentiable mass estimation for SDFs with materials."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from cadjoint.render.material import unspecified_materials
from cadjoint.sdf.base import SDF


def material_mass(
    sdf: SDF,
    points: Array | None = None,
    cell_volume: float | None = None,
    *,
    bounds: tuple[float, float, float] = (-3, -3, -3),
    size: tuple[float, float, float] = (6, 6, 6),
    resolution: int = 50,
    epsilon: float = 0.01,
) -> Array:
    """Estimate the mass of an SDF from its material field, as a JAX scalar.

    The mass counterpart of :func:`~cadjoint.sdf.measure.volume.volume`, and
    deliberately the same integral: the smooth interior indicator
    ``sigma(-d/eps)`` is weighted by the density the scene's material field
    reports at each sample point.  It therefore reduces to ``rho * volume`` for
    a single-material body and picks up the real per-region densities for a
    multi-material one.  Because the material field blends smoothly across CSG
    interfaces, the result is differentiable with respect to the shape
    parameters that move those interfaces *and* with respect to any density
    marked ``free``.

    Use it to regularize an optimization by mass rather than by volume.  For a
    single material the two differ only by a constant; as soon as a design can
    trade copper for aluminium, mass is the quantity that actually matters.
    A scene that already regularizes on its own sample lattice swaps in
    directly::

        # volume regularizer
        cell_volume * jnp.sum(jax.nn.sigmoid(-sdf(cells) / 0.03))
        # the same thing, weighted by what each cell is made of
        material_mass(scene, cells, cell_volume, epsilon=0.03)

    A mesh-based counterpart — exact on the elements a study solved on, rather
    than on a sampling lattice — is :func:`cadjoint.fem.properties.total_mass`.

    Args:
        sdf:         The SDF to measure; its materials must specify a density.
        points:      Optional ``(M, 3)`` sample points.  When omitted, samples
                     are taken on the regular lattice given by ``bounds`` /
                     ``size`` / ``resolution``, exactly like ``volume``.
        cell_volume: Volume each sample stands for.  Required with ``points``;
                     derived from the lattice otherwise.
        bounds:      Lower corner (x, y, z) of the sampling box (lattice mode).
        size:        Extent (dx, dy, dz) of the sampling box (lattice mode).
        resolution:  Samples per axis, ``resolution**3`` total (lattice mode).
        epsilon:     Smoothing width.  Smaller -> sharper / more accurate;
                     larger -> smoother gradients.

    Returns:
        Differentiable scalar mass estimate in kg, for geometry in metres and
        densities in kg/m^3.

    Raises:
        ValueError: If any material in the tree does not specify a density
            (reported by name, rather than silently returning NaN), or if
            ``points`` is given without ``cell_volume``.
    """
    missing = unspecified_materials(sdf, "density")
    if missing:
        raise ValueError(
            f"material_mass needs a density on every material; these state none: "
            f"{', '.join(sorted(set(missing)))}. See cadjoint.materials for a "
            "catalogue of real values."
        )
    if points is None:
        x = jnp.linspace(bounds[0], bounds[0] + size[0], resolution)
        y = jnp.linspace(bounds[1], bounds[1] + size[1], resolution)
        z = jnp.linspace(bounds[2], bounds[2] + size[2], resolution)
        grid_x, grid_y, grid_z = jnp.meshgrid(x, y, z, indexing="ij")
        points = jnp.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
        cell_volume = (size[0] / resolution) * (size[1] / resolution) * (size[2] / resolution)
    elif cell_volume is None:
        raise ValueError(
            "material_mass(sdf, points) also needs cell_volume — the volume each "
            "sample stands for — since it cannot be inferred from arbitrary points."
        )

    distances = jax.vmap(sdf)(jnp.asarray(points))
    indicators = jax.nn.sigmoid(-jnp.reshape(distances, (-1,)) / epsilon)
    densities = jax.vmap(lambda point: jnp.asarray(sdf.material_at(point)["density"]))(
        jnp.asarray(points)
    )
    return jnp.sum(indicators * jnp.reshape(densities, (-1,))) * cell_volume
