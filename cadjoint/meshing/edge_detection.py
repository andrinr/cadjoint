"""Grid-edge crossing detection and differentiable Hermite data.

This is the input stage of the meshing pipeline.  It answers two questions for
an implicit field sampled over a regular grid:

1. **Which lattice edges does the surface cross?**  This is discrete: the set
   of crossing edges is piecewise constant in the design parameters and is
   computed host-side with concrete values (:func:`find_crossing_edges`).
2. **Where exactly does the surface cross each edge, and what is the field
   gradient there?**  This is continuous: :func:`edge_hermite_data` locates the
   root on every crossing edge and evaluates the spatial gradient at it.  Both
   outputs carry exact JAX derivatives with respect to any design parameters
   the field closes over.

Differentiability comes from implicit differentiation rather than from
unrolling the root search.  Bisection runs on ``stop_gradient`` values, and a
final Newton correction ``t = t0 - f(x(t0)) / (df/dt)(x(t0))`` re-attaches the
gradient: at a converged root the derivative of that expression is exactly the
implicit-function-theorem value ``dt/dθ = -(∂f/∂θ) / (∂f/∂t)``.

The split mirrors the rest of the planned pipeline: discrete topology is
frozen per extraction, continuous motion is differentiable between topology
events.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def _safe_normalize(vectors: Array, fallback, *, epsilon: float) -> Array:
    """Normalize the rows of ``vectors``; degenerate rows take ``fallback``.

    Rows with squared length at most ``epsilon**2`` are replaced by the
    broadcast ``fallback`` instead of being divided.  The double-``where``
    keeps the degenerate branch constant, so in reverse mode a zero row
    cannot NaN the gradients of healthy rows.
    """
    squared = jnp.sum(vectors**2, axis=-1, keepdims=True)
    valid = squared > epsilon**2
    safe = jnp.where(valid, squared, 1.0)
    return jnp.where(valid, vectors / jnp.sqrt(safe), fallback)


class GridSpec(NamedTuple):
    """Regular sampling grid described by static Python values.

    Attributes:
        origin: World position of lattice vertex ``(0, 0, 0)``.
        spacing: Cell edge length per axis.
        cells: Number of cells per axis; the lattice has ``cells + 1`` vertices
            per axis.
    """

    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    cells: tuple[int, int, int]

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float],
        size: tuple[float, float, float],
        resolution: int | tuple[int, int, int],
    ) -> GridSpec:
        """Build a grid from a lower corner, an extent, and a cell count.

        Args:
            bounds: Lower corner of the sampled volume.
            size: Positive extent of the sampled volume per axis.
            resolution: Cells per axis, as one integer or a triplet.
        """
        cells = (resolution,) * 3 if isinstance(resolution, int) else tuple(resolution)
        if len(cells) != 3 or any(int(count) != count or count < 1 for count in cells):
            raise ValueError("resolution must contain three integers greater than or equal to 1.")
        origin = np.asarray(bounds, dtype=np.float64)
        extent = np.asarray(size, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("bounds must contain three finite numbers.")
        if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent <= 0):
            raise ValueError("size must contain three positive finite numbers.")
        cells = tuple(int(count) for count in cells)
        spacing = tuple(float(value) for value in extent / np.asarray(cells))
        return cls(origin=tuple(float(value) for value in origin), spacing=spacing, cells=cells)

    @property
    def lattice_shape(self) -> tuple[int, int, int]:
        """Number of lattice vertices per axis."""
        return (self.cells[0] + 1, self.cells[1] + 1, self.cells[2] + 1)

    def lattice_points(self) -> np.ndarray:
        """All lattice vertex positions, shaped ``(nx + 1, ny + 1, nz + 1, 3)``."""
        axes = [
            self.origin[axis] + self.spacing[axis] * np.arange(self.cells[axis] + 1)
            for axis in range(3)
        ]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


class CrossingEdges(NamedTuple):
    """Lattice edges whose endpoint field values straddle the isovalue.

    Attributes:
        axis: Edge direction per edge, ``0``/``1``/``2`` for x/y/z, shaped
            ``(edge_count,)``.
        index: Lattice coordinates of each edge's start vertex, shaped
            ``(edge_count, 3)``.
        start_inside: Whether the start vertex was inside (``value < level``)
            at detection time, shaped ``(edge_count,)``.  Root refinement
            bisects against this frozen orientation, which keeps it robust
            when a lattice vertex lies exactly on the surface and its
            re-evaluated sign is float noise.
    """

    axis: np.ndarray
    index: np.ndarray
    start_inside: np.ndarray

    @property
    def count(self) -> int:
        return int(self.axis.shape[0])


class HermiteData(NamedTuple):
    """Surface crossings on grid edges with field gradients.

    All arrays are JAX values and differentiate with respect to any design
    parameters the field closes over.

    Attributes:
        points: World position of each crossing, shaped ``(edge_count, 3)``.
        gradients: Raw spatial field gradient at each crossing, shaped
            ``(edge_count, 3)``.  Not normalized; see :meth:`unit_normals`.
        t: Crossing parameter along each edge, ``0`` at the start vertex and
            ``1`` at the end vertex, shaped ``(edge_count,)``.
        values: Field residual at each returned point, shaped
            ``(edge_count,)``.  Near zero for a converged root.
    """

    points: Array
    gradients: Array
    t: Array
    values: Array

    def unit_normals(self, *, epsilon: float = 1e-12) -> Array:
        """Safely normalized gradients; zero gradients stay zero.

        Safe in reverse mode too: the zero branch is constant, so a
        degenerate row cannot NaN the gradients of healthy rows.
        """
        return _safe_normalize(self.gradients, 0.0, epsilon=epsilon)


def sample_grid(sdf: Callable[[Array], Array], grid: GridSpec) -> np.ndarray:
    """Evaluate the field at every lattice vertex.

    Args:
        sdf: JAX-traceable scalar field ``(point[3]) -> scalar``.
        grid: Sampling grid.

    Returns:
        Concrete values shaped like :attr:`GridSpec.lattice_shape`.
    """
    points = grid.lattice_points().reshape((-1, 3))
    values = np.asarray(jax.jit(jax.vmap(sdf))(jnp.asarray(points)), dtype=np.float64)
    if values.shape != (points.shape[0],):
        raise ValueError(
            "The implicit field must return one scalar per point; "
            f"received shape {values.shape} for {points.shape[0]} points."
        )
    if not np.isfinite(values).all():
        raise ValueError("The implicit field returned a non-finite value.")
    return values.reshape(grid.lattice_shape)


def find_crossing_edges(values: np.ndarray, *, level: float = 0.0) -> CrossingEdges:
    """Find every lattice edge whose endpoints lie on opposite sides of ``level``.

    A vertex with ``value - level`` exactly zero counts as outside, matching
    the sign convention used during root refinement.

    Args:
        values: Lattice values shaped ``(nx + 1, ny + 1, nz + 1)``.
        level: Isovalue to extract.

    Returns:
        The crossing edge set, ordered by axis and then by lattice index.
    """
    lattice = np.asarray(values, dtype=np.float64)
    if lattice.ndim != 3 or any(count < 2 for count in lattice.shape):
        raise ValueError(
            f"values must be a lattice with at least two vertices per axis; received shape {lattice.shape}."
        )
    if not np.isfinite(lattice).all():
        raise ValueError("values must be finite; the lattice contains NaN or infinity.")
    if not np.isfinite(level):
        raise ValueError("level must be finite.")
    inside = (lattice - level) < 0
    axis_indices = []
    axis_labels = []
    axis_orientation = []
    for axis in range(3):
        leading = tuple(slice(None, -1) if a == axis else slice(None) for a in range(3))
        trailing = tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
        crossing = np.argwhere(inside[leading] != inside[trailing])
        axis_indices.append(crossing)
        axis_labels.append(np.full((crossing.shape[0],), axis, dtype=np.int8))
        axis_orientation.append(inside[leading][tuple(crossing.T)])
    return CrossingEdges(
        axis=np.concatenate(axis_labels),
        index=np.concatenate(axis_indices).astype(np.int32).reshape((-1, 3)),
        start_inside=np.concatenate(axis_orientation),
    )


def edge_hermite_data(
    sdf: Callable[[Array], Array],
    grid: GridSpec,
    edges: CrossingEdges,
    *,
    level: float = 0.0,
    bisection_iterations: int = 16,
    newton_steps: int = 1,
) -> HermiteData:
    """Locate the surface crossing on every edge and evaluate its gradient.

    The function is JAX-traceable end to end.  Calling it under ``jax.grad``
    or ``jax.jit`` with a field that closes over traced design parameters
    yields exact implicit derivatives of the crossing positions, parameters,
    and gradients.  The edge set itself stays frozen; re-detect edges after
    parameter changes large enough to alter the crossing topology.

    If a frozen edge no longer brackets a root under new parameter values,
    bisection collapses toward an endpoint and the Newton correction steps
    toward the nearest root along the edge line, possibly outside ``[0, 1]``.
    A proposed Newton step is kept only when it does not worsen the field
    residual, so tangency-degenerate edges — where the slope is float noise
    and the linearization is meaningless — fall back to the bisection point
    instead of jumping.  This keeps positions continuous across small
    parameter perturbations, which is what an optimization loop needs between
    re-extractions.

    Args:
        sdf: JAX-traceable scalar field ``(point[3]) -> scalar``.
        grid: The grid the edges were detected on.
        edges: Crossing edges from :func:`find_crossing_edges`.
        level: Isovalue to extract.
        bisection_iterations: Sign-based bracket halvings; run on
            ``stop_gradient`` values, so they add accuracy but no gradient
            path.
        newton_steps: Differentiable Newton corrections applied after
            bisection.  Must be at least 1 for the crossing parameter ``t``
            (and hence ``points`` and the positional part of ``gradients``)
            to carry parameter gradients; with 0 they are constant, though
            ``values`` and ``gradients`` still differentiate as field
            evaluations at fixed points.

    Returns:
        Hermite data for every edge, in the order of ``edges``.
    """
    if bisection_iterations < 0:
        raise ValueError("bisection_iterations must be non-negative.")
    if newton_steps < 0:
        raise ValueError("newton_steps must be non-negative.")
    if not np.isfinite(level):
        raise ValueError("level must be finite.")

    # Derive both edge endpoints in float64 with the same expressions as
    # lattice_points(), then round once.  Independent float32 arithmetic
    # would evaluate the field at slightly different coordinates than
    # detection did on grids away from the origin, silently breaking the
    # frozen brackets.
    origin64 = np.asarray(grid.origin, dtype=np.float64)
    spacing64 = np.asarray(grid.spacing, dtype=np.float64)
    index64 = np.asarray(edges.index, dtype=np.float64)
    one_hot = np.eye(3, dtype=np.float64)[np.asarray(edges.axis, dtype=np.int32)]
    starts = jnp.asarray(origin64 + index64 * spacing64)
    ends = jnp.asarray(origin64 + (index64 + one_hot) * spacing64)
    offsets = ends - starts

    def field(points: Array) -> Array:
        return jax.vmap(sdf)(points) - level

    def frozen_field(points: Array) -> Array:
        return jax.lax.stop_gradient(field(points))

    def at(t: Array) -> Array:
        return starts + t[:, None] * offsets

    edge_count = edges.count
    lo = jnp.zeros((edge_count,), dtype=starts.dtype)
    hi = jnp.ones((edge_count,), dtype=starts.dtype)
    f_lo = frozen_field(starts)
    f_hi = frozen_field(ends)
    # Bisect against the frozen detection-time orientation, not the live sign
    # of f_lo: when a lattice vertex sits exactly on the surface, its
    # re-evaluated sign is float noise and would break the bracket invariant.
    start_inside = jnp.asarray(edges.start_inside, dtype=bool)

    def halve(_iteration, state):
        lo, hi, f_lo, f_hi = state
        mid = 0.5 * (lo + hi)
        f_mid = frozen_field(at(mid))
        start_side = jnp.signbit(f_mid) == start_inside
        lo = jnp.where(start_side, mid, lo)
        f_lo = jnp.where(start_side, f_mid, f_lo)
        hi = jnp.where(start_side, hi, mid)
        f_hi = jnp.where(start_side, f_hi, f_mid)
        return lo, hi, f_lo, f_hi

    lo, hi, f_lo, f_hi = jax.lax.fori_loop(0, bisection_iterations, halve, (lo, hi, f_lo, f_hi))

    # One secant step inside the final bracket removes most bisection bias.
    denominator = f_lo - f_hi
    bracketed = jnp.abs(denominator) > 1e-30
    fraction = jnp.where(bracketed, f_lo / jnp.where(bracketed, denominator, 1.0), 0.5)
    t = lo + jnp.clip(fraction, 0.0, 1.0) * (hi - lo)

    gradient = jax.vmap(jax.grad(sdf))
    for _step in range(newton_steps):
        t0 = jax.lax.stop_gradient(t)
        points = at(t0)
        residuals = field(points)
        slopes = jnp.sum(gradient(points) * offsets, axis=-1)
        usable = jnp.abs(jax.lax.stop_gradient(slopes)) > 1e-12
        safe_slopes = jnp.where(usable, slopes, 1.0)
        # A degenerate slope (edge nearly tangent to the surface) must not
        # fling the root off the edge; one edge length is the trust region.
        correction = jnp.clip(jnp.where(usable, residuals / safe_slopes, 0.0), -1.0, 1.0)
        proposed = t0 - correction
        # On a tangency-degenerate edge the slope is float noise and the
        # linearization is meaningless; keep a step only if it does not
        # worsen the frozen residual, else stay at the bisection point.
        improved = jnp.abs(frozen_field(at(jax.lax.stop_gradient(proposed)))) <= jnp.abs(
            jax.lax.stop_gradient(residuals)
        )
        t = jnp.where(improved, proposed, t0)

    points = at(t)
    gradients = gradient(points)
    # A root can land bit-exactly on a surface plane whose SDF has a dead
    # subgradient there (epsilon-smoothed norms, |x| at 0, polygon
    # boundaries): the secant is exact on piecewise-linear walls, so this is
    # common, not exotic.  Re-evaluate degenerate gradients a fraction of an
    # edge toward the inside endpoint — the same smooth branch, off the
    # measure-zero point.
    inward = jnp.where(jnp.asarray(edges.start_inside, dtype=bool), -1e-3, 1e-3)
    fallback = gradient(at(jax.lax.stop_gradient(t) + inward))
    degenerate = jnp.sum(jax.lax.stop_gradient(gradients) ** 2, axis=-1) < 1e-12
    gradients = jnp.where(degenerate[:, None], fallback, gradients)
    return HermiteData(points=points, gradients=gradients, t=t, values=field(points))


def detect_edges(
    sdf: Callable[[Array], Array],
    grid: GridSpec,
    *,
    level: float = 0.0,
    bisection_iterations: int = 16,
    newton_steps: int = 1,
) -> tuple[CrossingEdges, HermiteData]:
    """Sample, detect crossing edges, and refine Hermite data in one call.

    Detection uses concrete values, so this convenience entry point cannot run
    under a JAX transformation.  Inside ``jax.grad``, call
    :func:`edge_hermite_data` with a previously detected edge set instead.
    """
    values = sample_grid(sdf, grid)
    edges = find_crossing_edges(values, level=level)
    hermite = edge_hermite_data(
        sdf,
        grid,
        edges,
        level=level,
        bisection_iterations=bisection_iterations,
        newton_steps=newton_steps,
    )
    return edges, hermite
