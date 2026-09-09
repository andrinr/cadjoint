"""Gmsh tet meshes of a scene, with every node tagged by the patches that own it.

The public tier's second tet route.  The DC route
(:func:`cadjoint.fem.tetmesh.sdf_to_tet_mesh`) hands TetGen a quad soup
whose element count is set by the *lattice*; this one hands Gmsh the same
dual-contour surface as an STL and lets Gmsh's HXT size the elements by the
*part*, put its second-order midside nodes on a reparametrised surface, and
report which surface, curve or point each node lies on::

    scene → dual contouring → STL → Gmsh (classify, HXT, order 2) → tet10 → OwnedNodes

Gmsh's ``classifySurfaces`` splits the triangle soup into smooth regions
along the lattice's feature cells and ``createGeometry`` parametrises each
one, so what Gmsh meshes is one surface per face of the part rather than
one per facet (Gmsh tutorial ``t13``).  An exact STEP — the private tier's
analytic writer, or any CAD file — takes the same road with
``geometry_format="step"``; the mesher and its contract are the same, only
the input is finer.

**Ownership is a residual test and nothing else.**  For every Gmsh surface
entity ``|f_p|`` is evaluated on its nodes for every patch ``p`` of the
scene's public decomposition
(:func:`~cadjoint.meshing.patch_fields.scene_patch_fields`); a patch whose
*worst* node clears the bar owns the surface, and a curve or point node
keeps, of the patches its bounding surfaces own, those that vanish at the
node itself.  Arity is the count kept, capped at the entity's codimension.
A node nothing owns is a blend node: the scene's own zero set holds it.  No
graph is consulted — this is the :class:`~cadjoint.plugins.OwnedNodes`
record the private ``node_map`` consumes, and the reason the route can be
public at all.

**The snap.**  Gmsh's own surface nodes lie on the STL's *facets*, and on a
curved face the chord sags: ``h²/8r`` is 3.5e-3 on a 0.25 bore at a 0.083
cell, above a bar of 2.7e-3, so the whole bore would read as a blend.  So
before tagging, every boundary node is Newton-projected onto the scene's
zero set with the public arity-1 :func:`~cadjoint.fem.motion.project_points`
— the DC path's own tool — clamped at the bar, and the projected position
is kept only where it confirms *more* patches than the facet position did
(or the same number at a lower residual).  Measured on the plate at
``target_size=0.16``: 264 of the 1 700 boundary nodes move and the bore
stops being a blend — **265 blend nodes and 2 blend surfaces become 0 and
0**, arity ``{0: 1370, 1: 1267, 2: 156, 3: 8}`` becoming
``{0: 1105, 1: 1494, 2: 194, 3: 8}`` — while the worst radius ratio is
unchanged at 0.293 (an analytic STEP of the same part gives 0.308 with the
same zero blends).  The snap buys *ownership*, not position: clamped at the
bar it leaves the bore's nodes and midsides within 2.4e-3 of ``r = 0.25``,
under the 2.7e-3 bar but nowhere near on it.  Putting them on it is the
``node_map``'s job, and 2.4e-3 < bar is exactly its precondition.  On the
starter's blended thermal body the rule moves 963 nodes without changing a
single tag (the fillets are blends either way, 713 of them) and the worst
radius ratio is 0.171 against 0.175 unsnapped and 0.215 from the analytic
STEP.  Ratios on a blended part vary by a percent or two between HXT runs;
the counts do not.

**Topology is frozen, positions are the private tier's.**  What Gmsh
decides — how many nodes, which cells, which entity owns which node — is
discovered once and held.  What moves under a design change is the
positions, and that map (per-arity Newton onto the owning patches with the
implicit-function adjoint, midsides re-solved on their surfaces, interior
nodes following by Laplacian relaxation) is the ``node_map`` plugin kind.
Without it a Gmsh mesh is *frozen geometry*: it solves, inspects and
exports, and :mod:`cadjoint.tier` says so when a derivative is asked for.

**Licence.**  Gmsh is GPL-2.0-or-later.  Nothing here imports it at module
scope and it is not a dependency of the library; it is an optional extra
(``pip install 'cadjoint[gmsh]'``) or, through the ``tet_mesher`` plugin,
the ``cadjoint_tet_gmsh`` image where the licence boundary is a process
boundary.

**Units.**  A STEP declaring ``SI_UNIT(.METRE.)`` is converted to
millimetres by OCCT's reader by default — a silent 1000x that turns a sane
target element size into a 10-billion-element request.
``Geometry.OCCTargetUnit = "M"`` is what stops it, and the imported bounds
are checked against the caller's rather than trusted.
"""

from __future__ import annotations

import tempfile
import time
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cadjoint.enums import PluginKind
from cadjoint.fem.tetmesh import TetMesh
from cadjoint.plugins.contracts import OwnedNodes

__all__ = [
    "CLASSIFY_ANGLE",
    "GEOMETRY_FORMATS",
    "HXT",
    "TET_MESHER_KIND",
    "GmshMesh",
    "GmshTetMesh",
    "assign_ownership",
    "dc_surface_stl",
    "design_values",
    "gmsh_available",
    "gmsh_tet_mesh",
    "gmsh_topology",
    "gmsh_version",
    "owned_nodes",
    "patch_table",
    "sdf_gmsh_tet_mesh",
    "snap_toward_patches",
    "surface_stl",
    "tet_mesh_from_gmsh",
]

#: The plugin slot this mesher fills — the one
#: :data:`cadjoint.plugins.registry.BUILTIN_PACKAGES` files ``tet_gmsh``
#: under.  Kinds are resolved by string, so this is the enum's value.
TET_MESHER_KIND = PluginKind.TET_MESHER.value

#: Gmsh's 3D algorithm number for HXT — the parallel Delaunay refinement
#: that makes this route worth taking at all.
HXT = 10

#: The geometry a call may carry: an exact STEP solid, or the dual-contour
#: surface as an ASCII STL.
GEOMETRY_FORMATS = ("step", "stl")

#: Dihedral angle, in degrees, above which ``classifySurfaces`` starts a new
#: surface.  Measured identical at 30, 40 and 60 on the plate: the DC
#: surface's creases are 90-degree folds and its faces are flat or smooth,
#: so nothing sits near the threshold.
CLASSIFY_ANGLE = 40.0

#: Gmsh's tet element types, keyed by node count.
_TET_TYPES = {4: 4, 10: 11}

#: Gmsh's own tet10 midside order is ``(0,1) (1,2) (0,2) (0,3) (2,3) (1,3)``
#: while meshio's ``tetra10`` (and :data:`cadjoint.fem.elements.TET10_EDGES`)
#: is ``(0,1) (1,2) (2,0) (0,3) (1,3) (2,3)`` — the last two are swapped.
#: Verified against the midpoint of each named corner pair, not assumed.
_GMSH_TO_MESHIO_TET10 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], dtype=np.int64)

#: Field callables of the scene's patch decomposition, world frame, flattened.
FieldTable = Sequence[Callable[[Any], Any]]


# ── the Gmsh wheel ───────────────────────────────────────────────────────────


def gmsh_available() -> bool:
    """Whether the optional ``gmsh`` wheel is importable."""
    try:
        import gmsh  # noqa: F401
    except ImportError:
        return False
    return True


def gmsh_version() -> str:
    """The installed Gmsh version string.

    Returns:
        The version, e.g. ``"4.15.2"``.

    Raises:
        ImportError: If the ``gmsh`` extra is not installed.
    """
    import gmsh

    return str(gmsh.__version__)


@contextmanager
def _gmsh_session(verbose: bool = False):
    """Initialise Gmsh, yield the module, and always finalise.

    Gmsh's API is a process-global singleton with no re-entrancy, so the
    session is opened and closed around one mesh rather than left running.

    Raises:
        ImportError: If the ``gmsh`` extra is not installed.
    """
    try:
        import gmsh
    except ImportError as error:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "The Gmsh tet mesher needs the optional 'gmsh' extra: "
            "pip install 'cadjoint[gmsh]'.  Gmsh is GPL-2.0-or-later, which "
            "is why it is not a dependency of the Apache-2.0 library.  Or run it "
            "through the 'tet_mesher' plugin (the cadjoint_tet_gmsh image)."
        ) from error
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        yield gmsh
    finally:
        gmsh.finalize()


# ── the geometry: the DC surface as STL ──────────────────────────────────────


def surface_stl(mesh: Any) -> str:
    """A dual-contour :class:`~cadjoint.meshing.Mesh` as ASCII STL text.

    ASCII rather than binary because the geometry crosses the ``tet_mesher``
    plugin boundary as a string, and because binary STL stores float32
    while the ASCII writer keeps eight significant digits.
    """
    from cadjoint.meshing.export import save_stl

    with tempfile.TemporaryDirectory(prefix="cadjoint-stl-") as scratch:
        path = Path(scratch) / "surface.stl"
        save_stl(mesh, path, binary=False)
        return path.read_text(encoding="ascii")


def dc_surface_stl(sdf: Callable[[Any], Any], grid: Any, **options: Any) -> tuple[str, Any]:
    """Dual-contour ``sdf`` on ``grid`` and return its surface as STL text.

    Args:
        sdf: The field to extract; an SDF node or a plain callable.
        grid: The :class:`~cadjoint.meshing.GridSpec`.
        **options: Forwarded to :func:`~cadjoint.meshing.extract_mesh`.

    Returns:
        ``(stl_text, mesh)`` — the STL and the mesh it was written from.
    """
    import jax.numpy as jnp

    from cadjoint.meshing.dual_contouring import extract_mesh

    field_fn = lambda p: jnp.asarray(sdf(p))  # noqa: E731
    with warnings.catch_warnings():
        # A meshing box is fitted to the part with a margin; a crossing at
        # its boundary means a scene that fills it, and clipping is the point.
        warnings.filterwarnings("ignore", message="The isosurface crosses the extraction boundary")
        mesh = extract_mesh(field_fn, grid, **options)
    return surface_stl(mesh), mesh


# ── the Gmsh black box ───────────────────────────────────────────────────────


def _tet_cells(gmsh_module: Any, tag_row: np.ndarray, order: int) -> np.ndarray:
    """The volume's tets, remapped to node rows and reordered to meshio.

    Raises:
        RuntimeError: If Gmsh produced no tets of the requested order.
    """
    wanted = _TET_TYPES[4 if order == 1 else 10]
    types, _tags, connectivity = gmsh_module.model.mesh.getElements(3)
    for element_type, nodes in zip(np.asarray(types), connectivity):
        if int(element_type) != wanted:
            continue
        per_cell = 4 if order == 1 else 10
        mapped = tag_row[np.asarray(nodes, dtype=np.int64)].reshape(-1, per_cell)
        return mapped if order == 1 else mapped[:, _GMSH_TO_MESHIO_TET10]
    raise RuntimeError(
        f"Gmsh produced no order-{order} tetrahedra; the surface may not have closed "
        "into a volume (a STEP shell that did not sew, or an open STL)."
    )


def _reorder(
    total: int, cells: np.ndarray, entity_dim: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray | None]:
    """Permute nodes into the TetMesh layout and rebuild ``edge_parents``.

    Returns:
        ``(permutation, cells, num_surface, num_corner_points, edge_parents)``
            where ``permutation`` maps a new row to the old row it came from.
    """
    from cadjoint.fem.elements import TET10_EDGES

    is_corner = np.zeros(total, dtype=bool)
    is_corner[np.unique(cells[:, :4])] = True
    boundary = is_corner & (entity_dim < 3)
    interior = is_corner & ~boundary
    permutation = np.concatenate(
        [np.flatnonzero(boundary), np.flatnonzero(interior), np.flatnonzero(~is_corner)]
    )
    inverse = np.empty(total, dtype=np.int64)
    inverse[permutation] = np.arange(total)
    cells = inverse[cells]
    num_surface = int(boundary.sum())
    num_corner = int(is_corner.sum())
    if order == 1:
        return permutation, cells.astype(np.int32), num_surface, num_corner, None

    # Match tet10_from_tet4's convention exactly: ``edge_parents`` is the
    # lexicographically sorted unique corner-pair table and midside node
    # ``num_corner + k`` belongs to row ``k``.  Gmsh numbers its midsides its
    # own way, so the midside block is permuted onto that convention rather
    # than the convention bent onto Gmsh.
    pairs = np.sort(cells[:, TET10_EDGES], axis=2).reshape(-1, 2)
    midsides = cells[:, 4:].reshape(-1)
    edge_parents, first = np.unique(pairs, axis=0, return_index=True)
    midside_order = midsides[first] - num_corner
    permutation = np.concatenate(
        [permutation[:num_corner], permutation[num_corner:][midside_order]]
    )
    shuffle = np.empty(total, dtype=np.int64)
    shuffle[:num_corner] = np.arange(num_corner)
    shuffle[num_corner + midside_order] = num_corner + np.arange(midside_order.size)
    return (
        permutation,
        shuffle[cells].astype(np.int32),
        num_surface,
        num_corner,
        edge_parents.astype(np.int64),
    )


def _check_step_import(step_text: str, entities: dict) -> None:
    """Refuse a STEP that OCCT could only read as a fragment.

    Gmsh imports the *highest-dimensional* shape and nothing else.  When a
    shell does not sew — a dual-contoured surface that folds on itself at
    the declared resolution is the standing case — OCCT hands back a
    compound of one small valid solid and one large open shell, and Gmsh
    keeps the solid.  Meshing then succeeds, quickly, on a part that is not
    the part.

    Counting ``ADVANCED_FACE`` in the file and comparing it with what
    arrived is what turns that into an error.  It is a text-level check on
    purpose: it needs no kernel and it is exact, because every STEP writer
    cadjoint knows emits one ``ADVANCED_FACE`` per face.

    Raises:
        RuntimeError: If there is no solid, or if faces went missing.
    """
    declared = step_text.count("ADVANCED_FACE")
    arrived = len(entities[2])
    if not entities[3]:
        raise RuntimeError(
            f"The STEP file's {declared} faces carry no solid volume; the shell did not sew."
        )
    if arrived < declared:
        raise RuntimeError(
            f"The STEP file declares {declared} faces but OCCT could only build a solid "
            f"from {arrived} of them, so Gmsh would mesh a fragment of the part rather "
            "than the part. The shell is not watertight at this resolution; re-extract "
            "on a finer grid."
        )


def _check_stl_import(entities: dict) -> None:
    """Refuse an STL whose classification did not close into one volume.

    Raises:
        RuntimeError: If no surface was classified or no volume was built.
    """
    if not entities[2]:
        raise RuntimeError("Gmsh classified no surfaces from the STL; the surface is empty.")
    if not entities[3]:
        raise RuntimeError(
            "Gmsh could not close the classified STL surfaces into a volume; the "
            "dual-contour surface is open at this resolution (re-extract on a finer grid "
            "or with a larger meshing box)."
        )


def _import_step(gmsh: Any, step_text: str, scratch: Path) -> dict:
    path = scratch / "geometry.step"
    path.write_text(step_text, encoding="ascii")
    # OCCT's STEP reader targets millimetres and cadjoint writes metres.
    # Without this the model is 1000x too big and a sane target size asks
    # for something like 1e10 elements — the first thing that bit here.
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.model.add("cadjoint_step")
    gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    entities = {dim: gmsh.model.getEntities(dim) for dim in (0, 1, 2, 3)}
    _check_step_import(step_text, entities)
    return entities


def _import_stl(gmsh: Any, stl_text: str, scratch: Path, classify_angle: float) -> dict:
    path = scratch / "geometry.stl"
    path.write_text(stl_text, encoding="ascii")
    gmsh.model.add("cadjoint_stl")
    gmsh.merge(str(path))
    # Split the triangle soup into smooth regions at ``classify_angle``,
    # keep the boundary curves between them, and parametrise each region so
    # it can be remeshed as a surface rather than constrained to its facets
    # (Gmsh tutorial t13).  The curve angle of 180 degrees keeps a boundary
    # curve whole however it bends, so a bore rim is one curve.
    gmsh.model.mesh.classifySurfaces(np.radians(classify_angle), True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    surfaces = gmsh.model.getEntities(2)
    if surfaces:
        loop = gmsh.model.geo.addSurfaceLoop([tag for _dim, tag in surfaces])
        gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()
    entities = {dim: gmsh.model.getEntities(dim) for dim in (0, 1, 2, 3)}
    _check_stl_import(entities)
    return entities


def gmsh_topology(
    geometry: str,
    *,
    geometry_format: str = "step",
    target_size: float,
    order: int = 2,
    algorithm: int = HXT,
    optimize: bool = True,
    classify_angle: float = CLASSIFY_ANGLE,
    verbose: bool = False,
) -> dict[str, Any]:
    """Mesh a solid with Gmsh — the whole of the opaque part, and no more.

    Nothing here knows about patches.  What comes back is what Gmsh alone
    can say: node positions, cells, and *which entity owns each node*,
    expressed as the surface tags whose closure the node lies on
    (:func:`assign_ownership` is what turns those into patches).  Keeping
    the cut here is what lets the mesher live behind a Tesseract — a
    container can be handed a file and cannot be handed a scene, whose patch
    fields are live Python callables.

    Args:
        geometry: The file's contents (ASCII): a STEP solid, or the
            dual-contour surface as STL (:func:`surface_stl`).
        geometry_format: ``"step"`` or ``"stl"``.
        target_size: Uniform element size in the file's own units.
        order: 1 for TET4, 2 for TET10.
        algorithm: Gmsh's ``Mesh.Algorithm3D``; :data:`HXT` by default.
        optimize: Run Gmsh's tet optimizer after generation.
        classify_angle: STL only — see :data:`CLASSIFY_ANGLE`.
        verbose: Let Gmsh write to the terminal.

    Returns:
        ``points`` ``(n, 3)``, ``cells`` ``(t, 4|10)`` in meshio order,
            ``entity_dim`` ``(n,)``, ``bounding_surfaces`` ``(n, k)`` of
            surface tags (``-1`` padded), ``num_surface``,
            ``num_corner_points``, ``edge_parents`` (``None`` at order 1),
            ``bounds`` ``(2, 3)``, ``cad_entities``, and the timings.

    Raises:
        ImportError: If the ``gmsh`` extra is not installed.
        RuntimeError: If the geometry carries no solid, or Gmsh makes no tets.
        ValueError: If ``order`` is not 1 or 2, or the format is unknown.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 (TET4) or 2 (TET10); got {order}.")
    if geometry_format not in GEOMETRY_FORMATS:
        raise ValueError(
            f"geometry_format must be one of {', '.join(GEOMETRY_FORMATS)}; got {geometry_format!r}."
        )
    with (
        _gmsh_session(verbose) as gmsh,
        tempfile.TemporaryDirectory(prefix="cadjoint-gmsh-") as scratch,
    ):
        started = time.perf_counter()
        if geometry_format == "step":
            entities = _import_step(gmsh, geometry, Path(scratch))
        else:
            entities = _import_stl(gmsh, geometry, Path(scratch), classify_angle)
        import_seconds = time.perf_counter() - started

        for name, value in (
            ("Mesh.MeshSizeMin", target_size),
            ("Mesh.MeshSizeMax", target_size),
            ("Mesh.MeshSizeFromCurvature", 0),
            ("Mesh.MeshSizeFromPoints", 0),
            ("Mesh.MeshSizeExtendFromBoundary", 0),
            ("Mesh.Algorithm3D", algorithm),
            ("Mesh.ElementOrder", order),
            ("Mesh.Optimize", 1 if optimize else 0),
            # 0 keeps the midside nodes ON the (reparametrised) surface,
            # which is the whole reason for going through a kernel: a
            # curved midside is what a straight-sided promotion cannot give.
            ("Mesh.SecondOrderLinear", 0),
        ):
            gmsh.option.setNumber(name, float(value))

        started = time.perf_counter()
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        mesh_seconds = time.perf_counter() - started

        result = _harvest(gmsh, entities, order)

    # The nodes' own extent, not ``getBoundingBox``: OCC pads a Bnd_Box by a
    # gap that on the starter's thermal body reads 1.8% wide, which is enough
    # to trip a unit check but is not a unit error.  A node lies *on* the
    # solid, so this measures the solid.
    points = result["points"]
    result["bounds"] = np.stack([points.min(axis=0), points.max(axis=0)])
    result["cad_entities"] = {dim: len(entities[dim]) for dim in (0, 1, 2, 3)}
    result["import_seconds"] = import_seconds
    result["mesh_seconds"] = mesh_seconds
    result["geometry_format"] = geometry_format
    return result


def _harvest(gmsh: Any, entities: dict, order: int) -> dict[str, Any]:
    """Pull nodes, cells and entity incidence out of a generated model."""
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    positions = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    total = tags.size
    tag_row = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    tag_row[tags] = np.arange(total)

    # A curve is bounded by the surfaces it separates and a point by the
    # curves that meet at it, so walking getAdjacencies upwards collects
    # exactly the surfaces whose fields a node has to satisfy.
    surfaces_above: dict[tuple[int, int], list[int]] = {}
    for dim in (2, 1, 0):
        for _dim, tag in entities[dim]:
            if dim == 2:
                surfaces_above[(2, int(tag))] = [int(tag)]
                continue
            upward, _downward = gmsh.model.getAdjacencies(dim, tag)
            collected: list[int] = []
            for parent in np.asarray(upward, dtype=np.int64):
                collected.extend(surfaces_above.get((dim + 1, int(parent)), []))
            surfaces_above[(dim, int(tag))] = sorted(set(collected))

    entity_dim = np.full(total, 3, dtype=np.int8)
    incidence: list[tuple[np.ndarray, list[int]]] = []
    width = 1
    for dim in (0, 1, 2):
        for _dim, tag in entities[dim]:
            node_tags, _coords, _param = gmsh.model.mesh.getNodes(dim, int(tag))
            rows = tag_row[np.asarray(node_tags, dtype=np.int64)]
            if rows.size == 0:
                continue
            entity_dim[rows] = dim
            above = surfaces_above[(dim, int(tag))]
            width = max(width, len(above))
            incidence.append((rows, above))
    bounding = np.full((total, width), -1, dtype=np.int32)
    for rows, above in incidence:
        bounding[np.ix_(rows, np.arange(len(above)))] = np.asarray(above, dtype=np.int32)

    cells = _tet_cells(gmsh, tag_row, order)
    permutation, cells, num_surface, num_corner, edge_parents = _reorder(
        total, cells, entity_dim, order
    )
    return {
        "points": positions[permutation],
        "cells": cells,
        "entity_dim": entity_dim[permutation].astype(np.int32),
        "bounding_surfaces": bounding[permutation],
        "num_surface": num_surface,
        "num_corner_points": num_corner,
        "edge_parents": edge_parents,
    }


# ── ownership by residual ────────────────────────────────────────────────────


def patch_table(scene: Any) -> list[Callable[[Any], Any]]:
    """The scene's public patch fields, world frame, flattened in table order.

    The order is :func:`~cadjoint.meshing.patch_fields.scene_patch_fields`'s
    (leaves depth-first, each leaf's patches in declaration order), which
    is the numbering :class:`~cadjoint.plugins.OwnedNodes` carries.  A
    plain callable has no decomposition and yields an empty table, so every
    boundary node of its mesh is a blend node — the scene owns them all.
    """
    from cadjoint.meshing.patch_fields import scene_patch_fields

    if scene is None or not hasattr(scene, "children"):
        return []
    decomposition = scene_patch_fields(scene)
    return [patch for fields in decomposition.fields for patch in fields]


def design_values(scene: Any) -> dict[str, np.ndarray]:
    """The scene's free-parameter values, name to array (empty for a callable)."""
    if scene is None or not hasattr(scene, "children"):
        return {}
    from cadjoint.extraction import extract_parameters

    free, _fixed, _metadata = extract_parameters(scene)
    return {name: np.asarray(value, dtype=np.float64) for name, value in free.items()}


def _residuals(fields: FieldTable, points: np.ndarray) -> np.ndarray:
    """``|f_p(x)|`` for every patch ``p`` at every point, shaped ``(n, P)``.

    Evaluated once for the whole node set, never per entity: a part with two
    hundred entities would otherwise pay two hundred JAX dispatches, and
    the cost of an eager JAX program is per *call*, not per point
    (``research/performance.md`` §6.2).
    """
    import jax
    import jax.numpy as jnp

    if not len(fields):
        return np.zeros((points.shape[0], 0), dtype=np.float64)
    probes = jnp.asarray(points, dtype=jnp.float32)
    values = [jax.vmap(patch)(probes) for patch in fields]
    return np.abs(np.asarray(jnp.stack(values, axis=-1), dtype=np.float64))


def snap_toward_patches(
    scene: Any,
    fields: FieldTable,
    points: np.ndarray,
    rows: np.ndarray,
    *,
    bar: float,
    steps: int = 8,
) -> tuple[np.ndarray, int]:
    """Project boundary nodes onto the scene's zero set where that helps ownership.

    The public arity-1 :func:`~cadjoint.fem.motion.project_points`, clamped
    at ``bar``; a node keeps the projected position only where it then
    clears the bar on more patches than before, or on as many at a lower
    residual.  See the module docstring for why and for the numbers.

    Args:
        scene: The field to project onto (a callable).
        fields: The patch table the ownership test will use.
        points: All node positions, ``(n, 3)``.
        rows: The boundary rows to consider.
        bar: The residual bar, which is also the displacement clamp.
        steps: Newton iterations.

    Returns:
        ``(points, moved)`` — a copy with the accepted rows replaced, and
            how many rows were.
    """
    import jax.numpy as jnp

    from cadjoint.fem.motion import project_points

    if rows.size == 0 or scene is None:
        return np.array(points, dtype=np.float64), 0
    before = np.asarray(points[rows], dtype=np.float64)
    field_fn = lambda p: jnp.asarray(scene(p))  # noqa: E731
    after = np.asarray(
        project_points(field_fn, jnp.asarray(before), float(bar), steps=steps), dtype=np.float64
    )
    if not len(fields):
        # Nothing to confirm against: the scene owns every boundary node,
        # so the projected position is simply the better one.
        out = np.array(points, dtype=np.float64)
        out[rows] = after
        return out, int(rows.size)
    residual_before, residual_after = _residuals(fields, before), _residuals(fields, after)
    confirmed_before = (residual_before <= bar).sum(axis=1)
    confirmed_after = (residual_after <= bar).sum(axis=1)
    take = (confirmed_after > confirmed_before) | (
        (confirmed_after == confirmed_before)
        & (residual_after.min(axis=1) < residual_before.min(axis=1))
    )
    out = np.array(points, dtype=np.float64)
    out[rows] = np.where(take[:, None], after, before)
    return out, int(take.sum())


def _owner_rows(
    candidates: list[int], residuals: np.ndarray, arity_cap: int, bar: float
) -> tuple[np.ndarray, int]:
    """Pick up to ``arity_cap`` of ``candidates`` that vanish at this node.

    A curve node inherits its patches from the surfaces that bound it, and
    inheriting is not the same as belonging: where a face is faceted, two
    adjacent facet surfaces can both point at the same neighbouring plane
    while the curve between them runs along a blend, a chord away from it.
    Taking that plane on the strength of the inheritance alone is what
    drags the node onto the plane's unbounded extension, so the same bar
    the surfaces had to clear is applied here, at the node itself.
    """
    scores = residuals.reshape(-1)
    unique = sorted({patch for patch in candidates if float(scores[patch]) <= bar})
    if not unique:
        return np.full(3, -1, dtype=np.int32), 0
    order = np.argsort(scores[np.asarray(unique, dtype=np.int64)], kind="stable")
    chosen = [unique[position] for position in order[:arity_cap]]
    row = np.full(3, -1, dtype=np.int32)
    row[: len(chosen)] = np.asarray(sorted(chosen), dtype=np.int32)
    return row, len(chosen)


def assign_ownership(fields: FieldTable, topology: dict[str, Any], *, bar: float) -> dict[str, Any]:
    """Tag every node with the patches that own it, by residual alone.

    Per Gmsh surface entity, ``|f_p|`` is evaluated on its nodes for every
    patch; the patches whose **worst** node clears ``bar`` are the
    candidates, and the one with the smallest typical residual owns the
    surface.  The maximum, not the median, is what separates a mesh from a
    spoiled one: an entity straddling a blend's edge has most of its nodes
    hugging the neighbouring plane and a few peeling away, and judged by
    the median it would pass as that plane — the projection would then drag
    the peeling nodes onto the plane's unbounded extension.  Curve and point
    nodes inherit their bounding surfaces' patches and keep those that
    vanish at the node itself, up to the entity's codimension.

    Args:
        fields: The patch table (:func:`patch_table`).
        topology: A :func:`gmsh_topology` result (positions possibly snapped).
        bar: ``|f_patch|`` above which a surface is not that patch's.

    Returns:
        ``owner_patches`` ``(n, 3)``, ``owner_arity`` ``(n,)``,
            ``blend_mask`` ``(n,)`` and the counts under ``stats``.
    """
    positions = np.asarray(topology["points"], dtype=np.float64)
    entity_dim = np.asarray(topology["entity_dim"], dtype=np.int64)
    bounding = np.asarray(topology["bounding_surfaces"], dtype=np.int64)
    total = positions.shape[0]
    residuals = _residuals(fields, positions)

    surface_patch: dict[int, int] = {}
    on_surface = entity_dim == 2
    for tag in np.unique(bounding[on_surface][:, 0]) if on_surface.any() else []:
        rows = np.flatnonzero(on_surface & (bounding[:, 0] == tag))
        if residuals.shape[1] == 0:
            surface_patch[int(tag)] = -1
            continue
        worst = residuals[rows].max(axis=0)
        typical = np.median(residuals[rows], axis=0)
        passing = np.flatnonzero(worst <= bar)
        surface_patch[int(tag)] = int(passing[np.argmin(typical[passing])]) if passing.size else -1

    owner_patches = np.full((total, 3), -1, dtype=np.int32)
    owner_arity = np.zeros(total, dtype=np.int8)
    blend_mask = np.zeros(total, dtype=bool)
    for row in np.flatnonzero(entity_dim < 3):
        tags = [int(tag) for tag in bounding[row] if tag >= 0]
        candidates = [surface_patch[tag] for tag in tags if surface_patch.get(tag, -1) >= 0]
        # A surface node solves one field, a curve node two, a point three:
        # the arity is the entity's codimension, ``3 - dim`` — capped there,
        # not fixed at it, since only the patches that actually vanish at
        # the node are kept (a circle-seam vertex is a dim-0 point where
        # only two patches meet, and solving three would over-determine it).
        chosen, arity = _owner_rows(candidates, residuals[row], 3 - int(entity_dim[row]), bar)
        if not arity:
            blend_mask[row] = True
            continue
        owner_patches[row] = chosen
        owner_arity[row] = arity
    return {
        "owner_patches": owner_patches,
        "owner_arity": owner_arity,
        "blend_mask": blend_mask,
        "stats": {
            "nodes": total,
            "patches": int(residuals.shape[1]),
            "nodes_by_dim": {dim: int((entity_dim == dim).sum()) for dim in (0, 1, 2, 3)},
            "blend_nodes": int(blend_mask.sum()),
            "blend_surfaces": int(sum(1 for patch in surface_patch.values() if patch < 0)),
            "arity_counts": {arity: int((owner_arity == arity).sum()) for arity in (0, 1, 2, 3)},
        },
    }


def owned_nodes(
    fields: FieldTable,
    topology: dict[str, Any],
    *,
    bar: float,
    design: dict[str, np.ndarray] | None = None,
) -> tuple[OwnedNodes, dict[str, Any]]:
    """The :class:`~cadjoint.plugins.OwnedNodes` record of a topology.

    Returns:
        ``(owned, stats)`` — the record, and :func:`assign_ownership`'s counts.
    """
    ownership = assign_ownership(fields, topology, bar=bar)
    points = np.asarray(topology["points"], dtype=np.float64)
    total = points.shape[0]
    parents = topology.get("edge_parents")
    parents = np.zeros((0, 2), dtype=np.int32) if parents is None else np.asarray(parents)
    midside = np.zeros(total, dtype=bool)
    midside[int(topology["num_corner_points"]) :] = True
    owned = OwnedNodes(
        seeds=points,
        patches=ownership["owner_patches"],
        arity=ownership["owner_arity"],
        entity_dim=np.asarray(topology["entity_dim"]),
        blend=ownership["blend_mask"],
        midside=midside,
        edge_parents=parents,
        cells=np.asarray(topology["cells"]),
        bar=float(bar),
        design=design or {},
    )
    return owned, ownership["stats"]


# ── the mesh ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GmshMesh:
    """A Gmsh tet mesh with every node owned — a static record.

    Node rows are ordered the way :class:`~cadjoint.fem.tetmesh.TetMesh`
    expects them: boundary corners, then interior corners, then (for order
    2) the shared midside block, so ``points[:num_surface]`` is the boundary
    and ``edge_parents`` describes the trailing rows.

    Attributes:
        points: Node positions ``(n, 3)`` as meshed (and snapped).
        cells: Connectivity ``(t, 4)`` or ``(t, 10)`` in meshio order,
            positive volume.
        owned: The ownership record the ``node_map`` kind consumes.
        entity_dim: Per node, the dimension of the entity Gmsh gave it to
            (0 vertex, 1 curve, 2 surface, 3 volume).
        num_surface: Number of leading boundary *corner* nodes.
        num_corner_points: Number of corner nodes (excludes midsides).
        edge_parents: ``(e, 2)`` corner pairs for the midside block, or
            ``None`` for order 1.
        max_step: The displacement clamp a re-projection should use.
        stats: Timings and counts — see :func:`gmsh_tet_mesh`.
    """

    points: np.ndarray
    cells: np.ndarray
    owned: OwnedNodes
    entity_dim: np.ndarray
    num_surface: int
    num_corner_points: int
    edge_parents: np.ndarray | None
    max_step: float
    stats: dict = field(default_factory=dict)

    @property
    def order(self) -> int:
        """Element order: 1 for TET4, 2 for TET10."""
        return 1 if self.edge_parents is None else 2

    @property
    def owner_patches(self) -> np.ndarray:
        """Per node, the owning global patch indices, ``(n, 3)``, ``-1`` padded."""
        return self.owned.patches

    @property
    def owner_arity(self) -> np.ndarray:
        """Per node, how many patches own it (0 for volume and blend nodes)."""
        return self.owned.arity

    @property
    def blend_mask(self) -> np.ndarray:
        """Boundary nodes no patch owns."""
        return self.owned.blend


def _plugin_topology(
    name: str, geometry: str, geometry_format: str, options: dict
) -> dict[str, Any]:
    """Run :func:`gmsh_topology` through a registered ``tet_mesher`` plugin.

    The plugin's ``apply`` is the discovery call: templates are empty, so it
    runs Gmsh and reports the topology it found.

    Args:
        name: Registered plugin name, or ``"default"`` for whatever fills
            the ``tet_mesher`` kind.
        geometry: The file's contents.
        geometry_format: ``"step"`` or ``"stl"``.
        options: ``target_size``, ``order``, ``algorithm``, ``optimize``.

    Returns:
        The same mapping :func:`gmsh_topology` returns.
    """
    from cadjoint.plugins import get_plugin, plugin_for_kind

    plugin = plugin_for_kind(TET_MESHER_KIND) if name == "default" else get_plugin(name)
    # Every template is empty: that is what puts the call in discovery mode,
    # where Gmsh runs and the topology it finds comes back.  The frozen mode
    # (templates filled, ``node_positions`` given) is the traced call, and
    # this function is never on that path.
    result = plugin.apply(
        {
            "geometry": geometry,
            "geometry_format": geometry_format,
            "target_size": np.float64(options["target_size"]),
            "order": np.int32(options["order"]),
            "algorithm": np.int32(options["algorithm"]),
            "node_positions": np.zeros((0, 3), np.float64),
            "node_ids": np.zeros(0, np.int32),
            "cell_template": np.zeros((0, 0), np.int32),
            "entity_dim_template": np.zeros(0, np.int32),
            "bounding_template": np.zeros((0, 0), np.int32),
            "edge_parent_template": np.zeros((0, 2), np.int32),
        }
    )
    parents = np.asarray(result["edge_parents"], dtype=np.int64)
    return {
        "points": np.asarray(result["nodes"], dtype=np.float64),
        "cells": np.asarray(result["cells"], dtype=np.int32),
        "entity_dim": np.asarray(result["entity_dim"], dtype=np.int32),
        "bounding_surfaces": np.asarray(result["bounding_surfaces"], dtype=np.int64),
        "num_surface": int(np.asarray(result["num_surface"])),
        "num_corner_points": int(np.asarray(result["num_corner_points"])),
        "edge_parents": parents if parents.size else None,
        "bounds": np.asarray(result["bounds"], dtype=np.float64),
        "cad_entities": {},
        "import_seconds": float("nan"),
        "mesh_seconds": float("nan"),
        "geometry_format": geometry_format,
    }


def _topology(
    geometry: str, geometry_format: str, options: dict, plugin: str | None, verbose: bool
) -> dict[str, Any]:
    """In-process Gmsh when it is importable, else the ``tet_mesher`` plugin."""
    if plugin is None:
        if gmsh_available():
            return gmsh_topology(
                geometry, geometry_format=geometry_format, verbose=verbose, **options
            )
        plugin = "default"
    return _plugin_topology(plugin, geometry, geometry_format, options)


def gmsh_tet_mesh(
    geometry: str,
    scene: Any = None,
    *,
    grid: Any,
    geometry_format: str = "stl",
    target_size: float | None = None,
    order: int = 2,
    algorithm: int = HXT,
    optimize: bool = True,
    blend_tolerance: float | None = None,
    plugin: str | None = None,
    fields: FieldTable | None = None,
    snap: bool = True,
    expected_bounds: np.ndarray | None = None,
    verbose: bool = False,
) -> GmshMesh:
    """Mesh a geometry with Gmsh and give every node its owners.

    Args:
        geometry: The geometry text — the DC surface as STL
            (:func:`dc_surface_stl`) or an exact STEP solid.
        scene: The scene the geometry came from: an SDF node (its public
            patch decomposition is the ownership table and its free
            parameters are recorded) or a plain callable (no table; every
            boundary node is scene-owned).  ``None`` skips the snap and,
            unless ``fields`` is given, tags nothing.
        grid: The sampling grid the geometry was extracted on; sets the
            default target size (its smallest spacing), the blend bar and
            the re-projection clamp, so a declared resolution means roughly
            what it means on the DC path.
        geometry_format: ``"stl"`` (default) or ``"step"``.
        target_size: Uniform element size in model units.
        order: 1 for TET4, 2 for TET10.
        algorithm: Gmsh's ``Mesh.Algorithm3D``; :data:`HXT` by default.
        optimize: Run Gmsh's tet optimizer after generation.
        blend_tolerance: The residual bar; defaults to ``1e-3`` times the
            grid diagonal, the derived B-rep's own export-grade bar.
        plugin: Run the mesher through this registered ``tet_mesher``
            plugin (``"default"`` for the kind's default) instead of an
            in-process Gmsh.  ``None`` imports Gmsh when it is installed and
            falls back to the default plugin otherwise.
        fields: A patch table to tag against instead of the scene's own
            (the analytic-STEP route passes its graph's).
        snap: Snap boundary nodes toward their patches first (see the
            module docstring); off for an exact STEP, whose nodes are on
            the surface already.
        expected_bounds: ``(2, 3)`` min/max the meshed solid must span to
            5 %, the backstop against a silent unit conversion.
        verbose: Let Gmsh write to the terminal.

    Returns:
        The :class:`GmshMesh`; ``stats`` carries the timings, the counts
            per entity dimension and per arity, the blend counts, and how
            many boundary nodes the snap moved.

    Raises:
        ImportError: Without Gmsh and without the ``tesseract`` extra.
        RuntimeError: If Gmsh meshes a solid whose extent does not match
            ``expected_bounds`` — the signature of a silent unit conversion.
    """
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    if target_size is None:
        target_size = float(spacing.min())
    if blend_tolerance is None:
        extent = spacing * np.asarray(grid.cells, dtype=np.float64)
        blend_tolerance = 1e-3 * float(np.linalg.norm(extent))
    bar = float(blend_tolerance)
    if fields is None:
        fields = patch_table(scene)

    options = {
        "target_size": float(target_size),
        "order": order,
        "algorithm": algorithm,
        "optimize": optimize,
    }
    topology = _topology(geometry, geometry_format, options, plugin, verbose)

    if expected_bounds is not None:
        imported = topology["bounds"][1] - topology["bounds"][0]
        span = np.asarray(expected_bounds[1], dtype=np.float64) - np.asarray(
            expected_bounds[0], dtype=np.float64
        )
        # Deliberately blunt: 5% is far wider than any chord error and far
        # narrower than the 1000x a millimetre default costs.
        if not np.allclose(imported, span, rtol=0.05, atol=1e-9):
            raise RuntimeError(
                f"Gmsh meshed a solid spanning {np.round(imported, 6).tolist()} but the "
                f"geometry spans {np.round(span, 6).tolist()}. Either a STEP unit conversion "
                "did not take (Geometry.OCCTargetUnit) or the surface closed into something "
                "other than the part."
            )

    started = time.perf_counter()
    entity_dim = np.asarray(topology["entity_dim"], dtype=np.int64)
    moved = 0
    if snap and scene is not None:
        topology = dict(topology)
        topology["points"], moved = snap_toward_patches(
            scene, fields, topology["points"], np.flatnonzero(entity_dim < 3), bar=bar
        )
    owned, ownership_stats = owned_nodes(fields, topology, bar=bar, design=design_values(scene))
    ownership_seconds = time.perf_counter() - started

    return GmshMesh(
        points=np.asarray(topology["points"], dtype=np.float64),
        cells=np.asarray(topology["cells"], dtype=np.int32),
        owned=owned,
        entity_dim=entity_dim.astype(np.int8),
        num_surface=int(topology["num_surface"]),
        num_corner_points=int(topology["num_corner_points"]),
        edge_parents=topology["edge_parents"],
        max_step=0.5 * float(np.linalg.norm(spacing)),
        stats={
            "geometry_format": geometry_format,
            "target_size": float(target_size),
            "bar": bar,
            "import_seconds": topology["import_seconds"],
            "mesh_seconds": topology["mesh_seconds"],
            "ownership_seconds": ownership_seconds,
            "snapped_nodes": moved,
            "cells": int(np.asarray(topology["cells"]).shape[0]),
            "cad_entities": topology["cad_entities"],
            **ownership_stats,
        },
    )


def sdf_gmsh_tet_mesh(
    scene: Any,
    grid: Any,
    *,
    order: int = 2,
    target_size: float | None = None,
    plugin: str | None = None,
    **options: Any,
) -> GmshMesh:
    """The whole public route: dual-contour ``scene`` on ``grid``, mesh it with Gmsh.

    Args:
        scene: An SDF node or a plain callable field.
        grid: The :class:`~cadjoint.meshing.GridSpec`.
        order: 1 for TET4, 2 for TET10.
        target_size: See :func:`gmsh_tet_mesh`.
        plugin: See :func:`gmsh_tet_mesh`.
        **options: Forwarded to :func:`gmsh_tet_mesh`.

    Returns:
        The :class:`GmshMesh`; ``stats["dc_seconds"]`` and
            ``stats["dc_triangles"]`` record the surface extraction.
    """
    started = time.perf_counter()
    stl, surface = dc_surface_stl(scene, grid)
    dc_seconds = time.perf_counter() - started
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    mesh = gmsh_tet_mesh(
        stl,
        scene,
        grid=grid,
        geometry_format="stl",
        target_size=target_size,
        order=order,
        plugin=plugin,
        expected_bounds=bounds,
        **options,
    )
    mesh.stats["dc_seconds"] = dc_seconds
    mesh.stats["dc_triangles"] = int(np.asarray(surface.faces).shape[0])
    return mesh


# ── handover ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GmshTetMesh(TetMesh):
    """A :class:`~cadjoint.fem.tetmesh.TetMesh` that came from Gmsh.

    Everything downstream of a mesh — the solvers, BC resolution, VTK,
    inspection — sees a ``TetMesh``.  What is added is the ownership record
    the ``node_map`` kind needs to move the nodes, and the name of the
    mesher, so :class:`~cadjoint.fem.simmesh.SimMesh` and
    :mod:`cadjoint.optimize` can tell that this mesh does not follow the
    design through :func:`~cadjoint.fem.motion.recompute_tet_points`.

    Attributes:
        owned: The :class:`~cadjoint.plugins.OwnedNodes` record.
        mesher: ``"gmsh"``.
        stats: The :class:`GmshMesh`'s counts and timings.
    """

    owned: OwnedNodes | None = None
    mesher: str = "gmsh"
    stats: dict[str, Any] = field(default_factory=dict)


def tet_mesh_from_gmsh(
    mesh: GmshMesh, *, grid: Any = None, points: np.ndarray | None = None
) -> GmshTetMesh:
    """Wrap a :class:`GmshMesh` as the FEM layer's :class:`GmshTetMesh` — static.

    The node layout is already the one :class:`~cadjoint.fem.tetmesh.TetMesh`
    documents — boundary corners, interior corners, midsides — so this only
    attaches the boundary triangles and the frozen base positions.

    ``edge_parents`` here describes *which corners a midside lies between*,
    not a straight-sided construction: the midsides are on the surface.
    Consumers that read it as a corner-pair table (BC completion, TET10
    element assembly) are unaffected.  The positions are the seeds; what
    moves them under a design change is the ``node_map`` kind.

    Args:
        mesh: The Gmsh mesh.
        grid: The sampling grid, recorded on the mesh.
        points: Positions to use instead of :attr:`GmshMesh.points`.

    Returns:
        A :class:`GmshTetMesh`.
    """
    from cadjoint.fem.tetmesh import tet_boundary_faces

    positions = mesh.points if points is None else np.asarray(points, dtype=np.float64)
    corners = np.asarray(mesh.cells[:, :4], dtype=np.int64)
    return GmshTetMesh(
        points=positions,
        cells=np.asarray(mesh.cells, dtype=np.int32),
        num_surface=mesh.num_surface,
        boundary_tris=tet_boundary_faces(corners),
        base_points=mesh.points.copy(),
        max_step=mesh.max_step,
        grid=grid,
        edge_parents=mesh.edge_parents,
        owned=mesh.owned,
        stats=dict(mesh.stats),
    )
