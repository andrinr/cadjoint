"""A part as a polygon mesh: the extraction knobs and the one call that uses them.

:func:`extract_mesh` takes a :class:`~cadjoint.meshing.GridSpec` and the
pipeline's own parameters; most callers have a part and want a mesh of it.
:func:`surface_grid` lays the lattice — over the part's fitted box or one the
caller gives, by a cell count on the longest axis or by a cell size —
:class:`SurfaceOptions` says how fine, whether creases are placed exactly,
how far the result may be simplified and whether feature edges come with it,
and :func:`extract_surface` runs the pipeline and returns a :class:`Surface`:

- ``mesh`` — welded, one vertex per point, each with the
  field's own normal, closed wherever the part is: what a mesh consumer
  indexes (STL, tet meshing, simplification, a B-rep of facets).
- ``corner_normals`` and :meth:`Surface.render_mesh` — the shading normals,
  split at creases: a vertex on a crease or a corner takes one normal per
  smooth side, from the Hermite normals of the cell the sharp QEF placed it
  in.  ``render_mesh()`` is the mesh a renderer takes (glTF and friends),
  with one vertex per side.
- ``feature_edges`` — the part's creases and CSG seams as line segments
  (:func:`~cadjoint.meshing.lattice_feature_edges`), when asked for.
- ``clipped`` — whether the part reaches the lattice's faces, so the mesh is
  cut open there.

The thinnest wall is measured on the result, against the field, by
:func:`~cadjoint.meshing.wall_thickness`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from cadjoint.meshing.bounds import fit_bounds
from cadjoint.meshing.dual_contouring import Extraction, Mesh, extraction
from cadjoint.meshing.edge_detection import GridSpec

__all__ = [
    "CREASE_ANGLE",
    "SURFACE_DEFAULT_RESOLUTION",
    "SURFACE_MAX_RESOLUTION",
    "SURFACE_MIN_RESOLUTION",
    "RenderMesh",
    "Surface",
    "SurfaceOptions",
    "extract_surface",
    "surface_grid",
]

#: Lattice cells along the part's longest axis: the floor below which the
#: lattice says nothing about a part, the default, and the ceiling past which
#: a dense sample stops fitting in memory (256 is 17M samples on a cube).
SURFACE_MIN_RESOLUTION = 8
SURFACE_DEFAULT_RESOLUTION = 64
SURFACE_MAX_RESOLUTION = 256

#: Degrees between two Hermite normals of one cell above which they lie on
#: different sides of a crease.  The feature classifier's default fires at a
#: normal fan of about 29 degrees, so a cell it calls a crease splits here.
CREASE_ANGLE = 30.0

#: A box as its lower and upper corner, ``((x0, y0, z0), (x1, y1, z1))``.
Bounds = Sequence[Sequence[float]]


@dataclass(frozen=True)
class SurfaceOptions:
    """How a part becomes a polygon mesh.

    Attributes:
        resolution: Lattice cells along the part's longest axis, for a grid
            :func:`extract_surface` fits itself; the other two axes take
            proportionally fewer, so cells are cubes.  Sets the smallest
            feature the mesh can represent (about two cells) and the
            triangle count (about the square of it).  A grid passed to
            :func:`extract_surface` replaces it.
        sharp: Place vertices exactly on creases and corners (the
            rank-revealing QEF); ``False`` keeps the smoother Tikhonov
            placement, which rounds every edge by a fraction of a cell.
        simplify: Merge triangles while the surface stays within this
            fraction of a cell of the field (:func:`~cadjoint.meshing.
            simplify_mesh`, creases kept); ``0`` keeps every lattice
            triangle.  Flat faces collapse first, down to what the pinned
            crease vertices fan out to: ``0.05`` takes a 24-cell box from
            3164 triangles to 400.
        feature_edges: Also find the part's creases and CSG seams
            (:func:`~cadjoint.meshing.lattice_feature_edges`), as chords
            to draw over the mesh; the mesh itself is the same either way.
            The seam projection compiles once per scene.
    """

    resolution: int = SURFACE_DEFAULT_RESOLUTION
    sharp: bool = True
    simplify: float = 0.0
    feature_edges: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.resolution, bool)
            or not isinstance(self.resolution, int)
            or not SURFACE_MIN_RESOLUTION <= self.resolution <= SURFACE_MAX_RESOLUTION
        ):
            raise ValueError(
                f"resolution must be an integer in [{SURFACE_MIN_RESOLUTION}, "
                f"{SURFACE_MAX_RESOLUTION}], got {self.resolution!r}."
            )
        if not isinstance(self.sharp, bool):
            raise TypeError(f"sharp must be a bool, got {type(self.sharp).__name__}.")
        if not (np.isfinite(self.simplify) and 0.0 <= self.simplify <= 1.0):
            raise ValueError(f"simplify must be in [0, 1] cells, got {self.simplify!r}.")
        if not isinstance(self.feature_edges, bool):
            raise TypeError(
                f"feature_edges must be a bool, got {type(self.feature_edges).__name__}."
            )


class RenderMesh(NamedTuple):
    """A surface for shading: one vertex per smooth side of every point.

    Attributes:
        vertices: Positions ``(N, 3)``.
        normals: Unit normals ``(N, 3)``: the field's own on a smooth
            surface, a side's on a crease.
        faces: Triangles ``(F, 3)`` into ``vertices``, the surface mesh's
            triangles in its order, counterclockwise seen from outside.
        rows: The surface mesh's vertex each one is a copy of, ``(N,)``.
    """

    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    rows: np.ndarray


@dataclass(frozen=True, eq=False)
class Surface:
    """A part's surface as :func:`extract_surface` makes it.

    Attributes:
        mesh: The welded mesh: one vertex per point, ``normals``
            the field's own (on a crease, the mean of both sides).
        grid: The lattice it was extracted on.
        clipped: Whether the part reaches the lattice's faces: the mesh is
            open there (or empty, when the whole lattice is inside).
        corner_normals: The shading normal at each triangle corner,
            ``(F, 3, 3)``: the vertex's own normal on a smooth surface, the
            normal of the triangle's side on a crease or corner.
        feature_edges: The creases and CSG seams as ``(E, 2, 3)`` chords,
            each a segment between two feature vertices on the curve (seam
            ends Newton-projected onto the exact seam, so a chord may sit a
            fraction of a cell off the facets); empty unless
            ``SurfaceOptions(feature_edges=True)``.
    """

    mesh: Mesh
    grid: GridSpec
    clipped: bool
    corner_normals: np.ndarray
    feature_edges: np.ndarray

    @property
    def cell(self) -> float:
        """The lattice's finest spacing, the unit walls and errors are counted in."""
        return float(min(self.grid.spacing))

    def render_mesh(self) -> RenderMesh:
        """The mesh with a vertex for each smooth side, for shading.

        Corners that share a vertex and a normal share a vertex here, so a
        smooth surface keeps its welded vertices and only crease and corner
        vertices are copied, once per side.
        """
        faces = np.asarray(self.mesh.faces, dtype=np.int64).reshape(-1, 3)
        positions = np.asarray(self.mesh.vertices, dtype=np.float64)
        if not faces.shape[0]:
            empty = np.empty((0, 3))
            return RenderMesh(empty, empty, faces, np.empty(0, dtype=np.int64))
        corners = np.concatenate(
            [faces.reshape(-1, 1).astype(np.float64), self.corner_normals.reshape(-1, 3)], axis=1
        )
        keys, first, inverse = np.unique(corners, axis=0, return_index=True, return_inverse=True)
        rows = faces.reshape(-1)[first]
        return RenderMesh(
            vertices=positions[rows],
            normals=keys[:, 1:],
            faces=inverse.reshape(-1, 3).astype(np.int64),
            rows=rows,
        )


def _box(bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    """``bounds`` checked: its lower corner and its extent."""
    try:
        low, high = (np.asarray(corner, dtype=np.float64) for corner in bounds)
    except (TypeError, ValueError):
        raise ValueError(f"bounds must be ((x0, y0, z0), (x1, y1, z1)), got {bounds!r}.") from None
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all():
        raise ValueError(f"bounds must be two corners of three finite numbers, got {bounds!r}.")
    if not (low < high).all():
        raise ValueError(f"bounds must have x0 < x1, y0 < y1 and z0 < z1, got {bounds!r}.")
    return low, high - low


def surface_grid(
    sdf: Callable[[Any], Any],
    resolution: int | None = None,
    *,
    cell: float | None = None,
    bounds: Bounds | None = None,
) -> GridSpec:
    """A near-isotropic lattice over the part, by cell count or by cell size.

    The box is ``bounds`` when given — one the caller knows holds the whole
    part; :attr:`Surface.clipped` says when it does not — and otherwise
    :func:`~cadjoint.meshing.fit_bounds`, which finds the part at whatever
    scale it is modelled, so a part drawn in millimetres is not clipped to a
    default volume.

    Args:
        sdf: The part's field on ``(..., 3)`` points.
        resolution: Cells along the box's longest axis; the other axes take
            proportionally fewer.
        cell: Instead of ``resolution``, the largest cell edge in model
            units: each axis takes ``ceil(extent / cell)`` cells, so the
            box is kept exactly and every spacing is at most ``cell``.
        bounds: The box, ``((x0, y0, z0), (x1, y1, z1))``.

    Raises:
        ValueError: Unless exactly one of ``resolution`` and ``cell`` is
            given, or for a cell that is not a positive number, bounds that
            are not a box, or a part :func:`fit_bounds` cannot find.
    """
    if (resolution is None) == (cell is None):
        raise ValueError("Give exactly one of resolution and cell.")
    if bounds is None:
        origin, size = fit_bounds(sdf)
        low, extent = np.asarray(origin, dtype=np.float64), np.asarray(size, dtype=np.float64)
    else:
        low, extent = _box(bounds)
    if cell is None:
        cells = tuple(
            int(max(1, round(resolution * axis / float(extent.max())))) for axis in extent
        )
    else:
        if isinstance(cell, bool) or not (isinstance(cell, (int, float)) and math.isfinite(cell)):
            raise ValueError(f"cell must be a positive finite number, got {cell!r}.")
        if not cell > 0:
            raise ValueError(f"cell must be a positive finite number, got {cell!r}.")
        cells = tuple(max(1, math.ceil(float(axis) / cell - 1e-9)) for axis in extent)
    return GridSpec.from_bounds(tuple(low), tuple(extent), cells)


def extract_surface(
    sdf: Callable[[Any], Any],
    options: SurfaceOptions = SurfaceOptions(),
    *,
    grid: GridSpec | None = None,
) -> Surface:
    """Mesh the zero set of ``sdf`` with the given options.

    Dual contouring on the lattice, then the feature edges when asked for,
    a weld of coincident vertices (as :func:`~cadjoint.meshing.remove_slivers`:
    the triangles a sharp concave corner collapses are dropped, and the rows
    no triangle uses any more with them), the simplification when asked
    for, and the crease-split shading normals.

    Args:
        sdf: The part's field on ``(..., 3)`` points: an SDF node, whose
            Boolean structure names the CSG seams, or a plain callable.
        options: The extraction knobs.
        grid: The lattice to extract on (:func:`surface_grid`, by cell or
            over the caller's bounds); fitted at ``options.resolution``
            otherwise.

    Returns:
        The :class:`Surface`.  Nothing in it carries derivatives: the
            extraction is the forward mesh (see :mod:`cadjoint.meshing` for
            the differentiable stages).
    """
    import jax.numpy as jnp

    from cadjoint.meshing.export import _coincident_representatives
    from cadjoint.meshing.simplify import _collapsed_faces, _compacted

    field = lambda p: jnp.asarray(sdf(p))  # noqa: E731
    grid = surface_grid(field, options.resolution) if grid is None else grid
    with warnings.catch_warnings():
        # The lattice's faces are checked on the samples below (clipped);
        # the extractor's warning would only repeat that, and only where
        # the surface happens to cross them.
        warnings.filterwarnings("ignore", message="The isosurface crosses the extraction boundary")
        stages = extraction(field, grid, sharp=options.sharp)
    clipped = _reaches_faces(stages.values)
    mesh = stages.mesh
    if not mesh.faces.shape[0]:
        return Surface(mesh, grid, clipped, np.empty((0, 3, 3)), np.empty((0, 2, 3)))

    chords = np.empty((0, 2, 3))
    if options.feature_edges:
        from cadjoint.meshing.feature_edges import lattice_feature_edges

        # The chords keep their own, seam-projected endpoints: moving the
        # mesh's vertices there instead folds it wherever two seams run
        # within a cell or two of each other (measured: the space-frame
        # node's tube rims at 0.825 mm, 9985 crossing pairs; the clevis,
        # 30 open edges).
        projected, links = lattice_feature_edges(
            sdf, grid, stages.hermite, stages.incidence, mesh.vertices, mesh.quads
        )
        chords = projected[links]

    # Weld: every row onto the lowest row at its position, then compact.
    representative = _coincident_representatives(np.asarray(mesh.vertices))
    faces = representative[np.asarray(mesh.faces, dtype=np.int64)]
    faces = faces[
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    ]
    quads = representative[np.asarray(mesh.quads, dtype=np.int64)]
    ordered = np.sort(quads, axis=1)
    quads = quads[(ordered[:, 1:] != ordered[:, :-1]).all(axis=1)]
    mesh, remap = _compacted(mesh._replace(faces=faces), faces)
    mesh = mesh._replace(quads=remap[quads].astype(np.int32))
    vertex_of_row = remap[representative]

    if options.simplify > 0.0 and mesh.faces.shape[0]:
        collapsed = _collapsed_faces(
            mesh, field, options.simplify * min(grid.spacing), True, None, 30.0, 100
        )
        mesh, remap = _compacted(mesh, collapsed)
        vertex_of_row = np.where(vertex_of_row >= 0, remap[np.maximum(vertex_of_row, 0)], -1)
    corner_normals = _corner_normals(mesh, stages, vertex_of_row)
    return Surface(mesh, grid, clipped, corner_normals, chords)


def _reaches_faces(values: np.ndarray | None) -> bool:
    """Whether any sample on the lattice's six faces lies inside the part."""
    if values is None:
        return False
    values = np.asarray(values)
    faces = (values[0], values[-1], values[:, 0], values[:, -1], values[:, :, 0], values[:, :, -1])
    return any(bool((face < 0.0).any()) for face in faces)


def _corner_normals(mesh: Mesh, stages: Extraction, vertex_of_row: np.ndarray) -> np.ndarray:
    """The shading normal at every triangle corner, split where a crease runs.

    A vertex whose cell the feature classifier calls a face keeps the
    field's normal.  At a crease or corner cell the incident Hermite normals
    — the field's exact normals where the surface crosses the cell's edges —
    fall into sides: two lie on one side when they are within
    :data:`CREASE_ANGLE`, and a side reaches as far as such pairs join it.
    Each side's normal is the mean of its samples, and each triangle corner
    takes the side its triangle faces most.  A vertex welded from several
    rows pools their samples.
    """
    from cadjoint.meshing.features import FACE, classify_feature_cells

    vertex_normals = np.asarray(mesh.normals, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    corners = vertex_normals[faces]
    hermite, incidence = stages.hermite, stages.incidence
    classes = classify_feature_cells(hermite, incidence).classes
    featured = np.flatnonzero((classes != FACE) & (vertex_of_row >= 0))
    if not featured.size:
        return corners

    # Every (vertex, Hermite normal) sample of a feature row, grouped by vertex.
    unit = np.asarray(hermite.unit_normals(), dtype=np.float64)
    slots = incidence.edge_ids[featured]
    owner = np.broadcast_to(vertex_of_row[featured][:, None], slots.shape)
    samples = unit[np.maximum(slots, 0)]
    valid = (slots >= 0) & (np.linalg.norm(samples, axis=-1) > 0.5)
    owner, samples = owner[valid], samples[valid]
    order = np.argsort(owner, kind="stable")
    owner, samples = owner[order], samples[order]
    vertices, start, count = np.unique(owner, return_index=True, return_counts=True)
    width = int(count.max())
    slot = np.arange(owner.size) - np.repeat(start, count)
    group = np.repeat(np.arange(vertices.size), count)
    padded = np.zeros((vertices.size, width, 3))
    padded[group, slot] = samples
    present = np.zeros((vertices.size, width), dtype=bool)
    present[group, slot] = True

    # Single-linkage sides: propagate the lowest slot label along close pairs.
    close = np.einsum("vai,vbi->vab", padded, padded) >= math.cos(math.radians(CREASE_ANGLE))
    close &= present[:, :, None] & present[:, None, :]
    label = np.broadcast_to(np.arange(width), (vertices.size, width)).copy()
    for _ in range(width):
        joined = np.where(close, label[:, None, :], width).min(axis=2)
        if np.array_equal(joined, label):
            break
        label = joined
    side_sum = np.zeros((vertices.size, width, 3))
    np.add.at(side_sum, (group, label[group, slot]), samples)
    length = np.linalg.norm(side_sum, axis=-1)
    side_normal = side_sum / np.maximum(length, 1e-30)[..., None]
    has_side = length > 0.0
    split = has_side.sum(axis=1) > 1
    if not split.any():
        return corners

    # Each corner at a split vertex takes the side its triangle faces most.
    lookup = np.full(mesh.vertices.shape[0], -1, dtype=np.int64)
    lookup[vertices[split]] = np.arange(int(split.sum()))
    sides, has_side = side_normal[split], has_side[split]
    positions = np.asarray(mesh.vertices, dtype=np.float64)
    a, b, c = (positions[faces[:, k]] for k in range(3))
    facing = np.cross(b - a, c - a)
    at = lookup[faces]
    triangle, corner = np.nonzero(at >= 0)
    candidates = sides[at[triangle, corner]]
    score = np.einsum("csi,ci->cs", candidates, facing[triangle])
    flat = np.linalg.norm(facing[triangle], axis=1) <= 1e-12 * max(
        1.0, float(np.abs(positions).max())
    )
    score = np.where(
        flat[:, None], np.einsum("csi,ci->cs", candidates, corners[triangle, corner]), score
    )
    score = np.where(has_side[at[triangle, corner]], score, -np.inf)
    corners[triangle, corner] = candidates[np.arange(triangle.size), score.argmax(axis=1)]
    return corners
