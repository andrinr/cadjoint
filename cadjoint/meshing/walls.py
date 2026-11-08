"""How thick a part's walls are, measured on its field, in lattice cells.

Dual contouring resolves a wall only when the lattice has room for it.
About four cells across give a wall flat faces and its true thickness
(:data:`WALL_CELLS`); under about two (:data:`MIN_WALL_CELLS`) its two faces
share cells, so the mesh is scalloped, closes through it, or falls apart into
pieces, and its thickness is the grid's rather than the part's.  A caller
picks the cell from the thinnest wall it means to model (``cell = wall /
4``); :func:`wall_thickness` checks the result after the fact, so a caller can
refuse or warn under the rule.

The measurement is a ray from every mesh vertex into the material, along
the opposite of its normal, sphere-traced through the field until the field
is positive again and then bisected onto the far face.  A ray counts as a
wall's only when it leaves through a face turned back at it — the field's
gradient at the exit within 30 degrees of opposite to the ray's start — so a
ray from beside a convex edge, which leaves through a neighbouring face,
measures how near the edge it started and not a wall.  On a flat or curved
wall the two faces are opposite, and at a crease the vertex normal is the
bisector, which only lengthens the chord, so the least of the counted rays
is the thinnest wall.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from cadjoint.meshing.dual_contouring import Mesh

__all__ = ["MIN_WALL_CELLS", "WALL_CELLS", "WallThickness", "wall_thickness"]

#: Cells across a wall below which its thickness is the grid's: closed,
#: scalloped, or fallen apart.  A caller refuses a part under this.
MIN_WALL_CELLS = 2.0
#: Cells across a wall that the mesh resolves with flat faces and its true
#: thickness: the cell to model a part at is its thinnest wall over this.
WALL_CELLS = 4.0

# Sphere-tracing budget per ray, then bisections of the last bracket.
_STEPS = 48
_HALVINGS = 8
# The exit's gradient within this of opposite to the start's normal.
_OPPOSITE = math.cos(math.radians(30.0))


class WallThickness(NamedTuple):
    """Wall thickness at every vertex, and the thinnest of them.

    Attributes:
        per_vertex: Thickness along each vertex's inward ray, in model units,
            ``(V,)``; ``nan`` where the ray measured no wall (it started
            outside, left through a neighbouring face, or found no exit).
        cell: The lattice spacing the counts are in.
        thinnest: The least measured thickness, or ``None`` when no vertex
            measured one.
        at: The vertex where it is, ``(3,)``, or ``None``.
    """

    per_vertex: np.ndarray
    cell: float
    thinnest: float | None
    at: np.ndarray | None

    @property
    def cells(self) -> float | None:
        """The thinnest wall in lattice cells, or ``None``."""
        return None if self.thinnest is None else self.thinnest / self.cell

    @property
    def resolved(self) -> bool:
        """Whether every measured wall is at least :data:`MIN_WALL_CELLS` across."""
        return self.thinnest is None or self.thinnest >= MIN_WALL_CELLS * self.cell


def wall_thickness(sdf: Callable[[Any], Any], mesh: Mesh, cell: float) -> WallThickness:
    """Measure the wall under every vertex of a mesh, on the field itself.

    Each ray starts an eighth of a cell inside the vertex (a start already
    outside leaves the vertex unmeasured, never a wall of zero), steps by
    the field's magnitude — a lower bound on the distance to any surface, so
    no crossing is stepped over — and at least an eighth of a cell, never
    past the mesh's diagonal; the last inside and outside depths are then
    bisected.  Every ray runs at once: the field is evaluated on all of
    them in one shape per call, so it compiles once, about sixty calls in
    all.

    Args:
        sdf: The part's field on ``(n, 3)`` points.
        mesh: The mesh whose vertices the rays start from, with the field's
            own outward normals (on a crease, the bisector) — a
            :class:`~cadjoint.meshing.Surface`'s ``mesh``.
        cell: The lattice spacing, the unit of :attr:`WallThickness.cells`
            (:attr:`Surface.cell <cadjoint.meshing.Surface.cell>`).

    Returns:
        The :class:`WallThickness`.

    Raises:
        ValueError: For a cell that is not positive, or vertices and
            normals of different shapes.

    Example:
        ``walls = wall_thickness(scene, surface.mesh, surface.cell)``, then
        refuse when ``not walls.resolved``.
    """
    import jax
    import jax.numpy as jnp

    if not (math.isfinite(cell) and cell > 0):
        raise ValueError(f"cell must be a positive finite number, got {cell!r}.")
    points = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    inward = -np.asarray(mesh.normals, dtype=np.float64).reshape(-1, 3)
    if inward.shape != points.shape:
        raise ValueError(f"normals {inward.shape} do not match vertices {points.shape}.")
    count = points.shape[0]
    if not count:
        return WallThickness(np.empty(0), float(cell), None, None)
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-30)
    program = jax.jit(lambda p: jnp.asarray(sdf(p)).reshape(-1))

    def at(depth: np.ndarray) -> np.ndarray:
        where = points + inward * depth[:, None]
        return np.asarray(program(jnp.asarray(where, dtype=jnp.float32)), dtype=np.float64)

    least = cell / 8.0
    reach = max(float(np.linalg.norm(np.ptp(points, axis=0))), least)
    depth = np.full(count, least)
    value = at(depth)
    low = depth.copy()  # deepest depth known inside
    high = np.full(count, np.nan)  # shallowest depth known outside
    beyond = np.zeros(count)  # the field there
    tracing = value < 0.0
    for _ in range(_STEPS):
        if not tracing.any():
            break
        ahead = np.where(
            tracing, np.minimum(depth + np.maximum(np.abs(value), least), reach), depth
        )
        found = at(ahead)
        left = tracing & (found >= 0.0)
        low = np.where(tracing & ~left, ahead, low)
        high = np.where(left, ahead, high)
        beyond = np.where(left, found, beyond)
        depth, value = np.where(tracing, ahead, depth), np.where(tracing, found, value)
        tracing &= ~left & (ahead < reach)
    crossed = ~np.isnan(high)
    if not crossed.any():
        return WallThickness(np.full(count, np.nan), float(cell), None, None)
    high = np.where(crossed, high, low)
    for _ in range(_HALVINGS):
        middle = 0.5 * (low + high)
        found = at(middle)
        within = found < 0.0
        low, high = np.where(within, middle, low), np.where(within, high, middle)
        beyond = np.where(within, beyond, found)

    # The exit's outward gradient, by forward differences off the outside point.
    step = cell / 16.0
    gradient = np.stack(
        [
            np.asarray(
                program(jnp.asarray(points + inward * high[:, None] + step * axis, jnp.float32)),
                dtype=np.float64,
            )
            - beyond
            for axis in np.eye(3)
        ],
        axis=1,
    )
    length = np.linalg.norm(gradient, axis=1)
    facing = np.einsum("ij,ij->i", gradient, inward) >= _OPPOSITE * np.where(
        length > 0.0, length, np.inf
    )
    per_vertex = np.where(crossed & facing, 0.5 * (low + high), np.nan)
    if np.isnan(per_vertex).all():
        return WallThickness(per_vertex, float(cell), None, None)
    thinnest = int(np.nanargmin(per_vertex))
    return WallThickness(per_vertex, float(cell), float(per_vertex[thinnest]), points[thinnest])
