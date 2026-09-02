"""Tet10 straight from the exact B-rep, meshed by Gmsh, owned by the graph.

The DC route to TET10 (:func:`cadjoint.fem.tetmesh.sdf_to_tet_mesh` then
:func:`~cadjoint.fem.tetmesh.tet10_mesh`) spends almost none of its time in
TetGen.  ``research/performance.md`` measures TetGen at 11 ms; the cost is
dual contouring and the JAX projection of a surface whose node count is set
by *the lattice*, not by the part.  Worse, the surface it hands TetGen is a
quad soup whose triangles fold on themselves where the lattice grazes a
feature — which is why the end-cap does not mesh at its declared resolution
at all.

:mod:`cadjoint.brep.step` already writes that part as **exact geometry**: a
plane is a ``PLANE`` trimmed by its own four-vertex loop, a bore is one
``CYLINDRICAL_SURFACE``.  A real CAD mesher can take that file and produce a
tet mesh sized by the *model*.  So this module is the short route:

    graph → exact STEP → Gmsh (HXT, order 2) → tet10 → owned nodes

with **the ownership put back on**.  Gmsh returns each node's owning CAD
entity, and each CAD entity is matched back to a :class:`~cadjoint.brep.graph.BRepFace`
(by a nearest-quad vote, confirmed by the patch field's own residual — OCC
tags are not stable across a rewrite, and Gmsh reports a cylinder's type as
``Unknown``, so neither tag nor type is trustworthy on its own).  From the
face comes the patch, and from the patch arity comes the projection:

- a **surface** node solves ``f_a = 0`` — one field,
- a **curve** node solves ``f_a = f_b = 0`` — the two faces it separates,
- a **point** node solves ``f_a = f_b = f_c = 0``,
- a **blend** node lies on no patch at all and solves ``scene(x) = 0``,
- a **volume** node is mesh gauge and follows the interior Laplacian.

Every one of those is :func:`cadjoint.brep.project.project` at a different
arity, so the whole node set differentiates in the design parameters through
the same implicit-function adjoint the graph already uses — *including the
midside nodes*, which is the point.  Gmsh's ``setOrder(2)`` puts a midside
node on the CAD surface; re-solving it against its own patch is what keeps it
on the cylinder when the bore radius changes, instead of drifting to the
straight-sided midpoint the DC path is stuck with.

**Topology is frozen, positions are not.**  What Gmsh decides — how many
nodes, which cells, which entity owns which node — is discovered once and
held.  What moves under a design change is only the positions, recomputed
here.  That is the same contract
:func:`cadjoint.brep.plc.recompute_plc_points` honours, and it is what lets
the mesher live behind a Tesseract whose ``vector_jacobian_product`` is a
pass-through (:mod:`cadjoint.fem.tesseracts.tet_gmsh`).

**Licence.**  Gmsh is GPL-2.0-or-later.  Nothing here imports it at module
scope and it is not a dependency of the library; it is an optional extra
(``pip install 'cadjoint[gmsh]'``) reached through the plugin registry, so
the Apache-2.0 library never links it.

**Units.**  :mod:`cadjoint.brep.step` declares ``SI_UNIT(.METRE.)`` and
OCCT's STEP reader converts to millimetres by default — a silent 1000x that
turns a sane target element size into a 10-billion-element request.
``Geometry.OCCTargetUnit = "M"`` is what stops it, and the bounding box is
checked against the graph's own afterwards rather than trusted.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cadjoint.brep.graph import BRep
from cadjoint.brep.project import project_batched, project_fields
from cadjoint.brep.step import save_brep_step
from cadjoint.enums import PluginKind

__all__ = [
    "HXT",
    "TET_MESHER_KIND",
    "GmshMesh",
    "assign_ownership",
    "gmsh_available",
    "gmsh_tet_mesh",
    "gmsh_topology",
    "gmsh_version",
    "parameterised_points",
    "recompute_gmsh_points",
    "tet_mesh_from_gmsh",
]

#: The plugin slot this mesher fills — the same one
#: :data:`cadjoint.plugins.registry.BUILTIN_PACKAGES` files ``tet_gmsh``
#: under.  Kinds are resolved by string, so this is the enum's value.
TET_MESHER_KIND = PluginKind.TET_MESHER.value

#: Gmsh's 3D algorithm number for HXT — the parallel Delaunay refinement
#: that makes this route worth taking at all.
HXT = 10

#: Gmsh's tet element types, keyed by node count.
_TET_TYPES = {4: 4, 10: 11}

#: Gmsh's own tet10 midside order is ``(0,1) (1,2) (0,2) (0,3) (2,3) (1,3)``
#: while meshio's ``tetra10`` (and :data:`cadjoint.fem.elements.TET10_EDGES`)
#: is ``(0,1) (1,2) (2,0) (0,3) (1,3) (2,3)`` — the last two are swapped.
#: Verified against the midpoint of each named corner pair, not assumed.
_GMSH_TO_MESHIO_TET10 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], dtype=np.int64)


@dataclass(frozen=True)
class GmshMesh:
    """A Gmsh tet mesh of the exact B-rep, with every node owned.

    Node rows are ordered the way :class:`~cadjoint.fem.tetmesh.TetMesh`
    expects them: boundary corners, then interior corners, then (for order
    2) the shared midside block, so ``points[:num_surface]`` is the boundary
    and ``edge_parents`` describes the trailing rows.

    Attributes:
        points: Node positions ``(n, 3)`` as Gmsh placed them.
        cells: Connectivity ``(t, 4)`` or ``(t, 10)`` in meshio order,
            positive volume.
        owner_patches: Per node, the global patch indices it must be solved
            against, ``(n, 3)`` and ``-1``-padded.
        owner_arity: Per node, how many of those there are — 0 for a volume
            node and for a blend node (which has no patch; see
            :attr:`blend_mask`).
        blend_mask: Nodes that lie on a blend face and are solved against
            the *scene's* zero set instead of a patch's.
        owner_face: Per node, the :class:`~cadjoint.brep.graph.BRepFace` it
            was matched to, ``-1`` for a volume node.
        entity_dim: Per node, the dimension of the CAD entity Gmsh gave it
            to (0 vertex, 1 curve, 2 surface, 3 volume).
        num_surface: Number of leading boundary *corner* nodes.
        num_corner_points: Number of corner nodes (excludes midsides).
        edge_parents: ``(e, 2)`` corner pairs for the midside block, or
            ``None`` for order 1.
        max_step: Projection displacement clamp.
        stats: Timings and counts — see :func:`gmsh_tet_mesh`.
    """

    points: np.ndarray
    cells: np.ndarray
    owner_patches: np.ndarray
    owner_arity: np.ndarray
    blend_mask: np.ndarray
    owner_face: np.ndarray
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

    def blend_nodes_by_face(self) -> dict[int, int]:
        """How many nodes each blend face owns, keyed by face index.

        Key ``-1`` counts the blend nodes no face was matched to at all —
        nodes on a CAD curve whose bounding surfaces are facets so small
        that Gmsh gave them no interior node of their own, leaving nothing
        to vote with.  They are blend nodes for the right reason (nothing
        owns them, so the scene does) and are reported separately rather
        than folded into a face they only nearly belong to.

        Returns:
            ``face index -> node count``, with ``-1`` for the unmatched.
        """
        counts: dict[int, int] = {}
        for face_id in self.owner_face[self.blend_mask]:
            counts[int(face_id)] = counts.get(int(face_id), 0) + 1
        return counts


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
            "is why it is not a dependency of the Apache-2.0 library."
        ) from error
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        yield gmsh
    finally:
        gmsh.finalize()


# ── entity ownership ─────────────────────────────────────────────────────────


def _face_of_position(brep: BRep, points: np.ndarray) -> np.ndarray:
    """Nearest-quad vote: the B-rep face closest to each position.

    OCC face tags are assigned by the STEP reader in its own order and Gmsh
    reports a cylindrical surface's type as ``Unknown``, so neither the tag
    nor the type identifies a face.  The tessellation does: every quad knows
    its face (:attr:`BRep.quad_face`), and a Gmsh node lies on the same face
    as the quad whose centroid is nearest it.
    """
    from scipy.spatial import cKDTree

    quads = np.asarray(brep.mesh.quads, dtype=np.int64)
    centroids = brep.points[quads].mean(axis=1)
    _distance, nearest = cKDTree(centroids).query(points)
    return brep.quad_face[np.asarray(nearest, dtype=np.int64)].astype(np.int64)


def _patch_residuals(brep: BRep, points: np.ndarray) -> np.ndarray:
    """``|f_p(x)|`` for every patch ``p`` at every point, shaped ``(n, P)``.

    Evaluated once for the whole node set, never per entity: a part with two
    hundred CAD entities would otherwise pay two hundred JAX dispatches, and
    the cost of an eager JAX program is per *call*, not per point
    (``research/performance.md`` §6.2).
    """
    import jax
    import jax.numpy as jnp

    probes = jnp.asarray(points, dtype=jnp.float32)
    values = [jax.vmap(patch.field)(probes) for patch in brep.patches]
    return np.abs(np.asarray(jnp.stack(values, axis=-1), dtype=np.float64))


def _surface_owner(
    brep: BRep, positions: np.ndarray, residuals: np.ndarray, tolerance: float
) -> tuple[int, int]:
    """Decide ``(face, patch)`` for one Gmsh surface entity; ``patch = -1`` = blend.

    The vote picks the face, the residual confirms it.  A face whose patch
    does not actually vanish on the entity's nodes is not that face, so the
    patch with the smallest residual is tried instead, and when none of them
    vanishes the entity is a blend — the same exact test
    :func:`~cadjoint.brep.graph.extract_brep` uses, needing no angle
    threshold.

    **The confirmation is the worst node, not the typical one**, and that is
    the difference between a mesh and a spoiled one.  The graph facets every
    non-analytic face, so an entity that straddles a blend's edge has most of
    its nodes hugging the neighbouring plane and a few peeling away.  Judged
    by the median it passes as that plane, and the projection then drags the
    peeling nodes flat onto the plane's *unbounded* extension — measured on
    the starter's thermal body as a 2.5e-2 pull and a drop in the worst
    radius ratio from 0.215 to 0.198.  Judged by the maximum, the entity is
    a blend, the scene holds it, and the mesh keeps its quality.  The median
    still *chooses* between candidates, where it is the robust statistic.
    """
    votes = _face_of_position(brep, positions)
    modal = int(np.bincount(votes).argmax())
    candidate = brep.faces[modal].patch
    typical = np.median(residuals, axis=0)
    worst = residuals.max(axis=0)
    if candidate >= 0 and float(worst[candidate]) <= tolerance:
        return modal, int(candidate)
    best = int(np.argmin(typical))
    if float(worst[best]) <= tolerance:
        owning = [face.index for face in brep.faces if face.patch == best]
        return (owning[0] if owning else modal), best
    return modal, -1


def _owner_rows(
    candidates: list[int], residuals: np.ndarray, arity_cap: int, tolerance: float
) -> tuple[np.ndarray, int]:
    """Pick up to ``arity_cap`` of ``candidates`` that vanish at this node.

    A curve node inherits its patches from the surfaces that bound it, and
    inheriting is not the same as belonging: where the graph facets a face,
    two adjacent facet surfaces can both point at the same neighbouring
    plane while the curve between them runs along the blend, a chord away
    from it.  Taking that plane on the strength of the inheritance alone is
    what drags the node onto the plane's unbounded extension, so the same
    bar the surfaces had to clear is applied here, at the node itself.  A
    node that clears it for nothing is a blend node and the caller says so.

    Args:
        candidates: Patch indices inherited from the bounding surfaces.
        residuals: ``|f_p(x)|`` at this node, shaped ``(1, P)``.
        arity_cap: The entity's codimension, ``3 - dim``.
        tolerance: The blend bar, as :func:`assign_ownership` sets it.

    Returns:
        The ``-1``-padded owner row ``(3,)`` and how many of it is filled.
    """
    scores = residuals.reshape(-1)
    unique = sorted({patch for patch in candidates if float(scores[patch]) <= tolerance})
    if not unique:
        return np.full(3, -1, dtype=np.int32), 0
    order = np.argsort(scores[np.asarray(unique, dtype=np.int64)])
    chosen = [unique[position] for position in order[:arity_cap]]
    row = np.full(3, -1, dtype=np.int32)
    row[: len(chosen)] = np.asarray(sorted(chosen), dtype=np.int32)
    return row, len(chosen)


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
        f"Gmsh produced no order-{order} tetrahedra; the STEP solid may not have "
        "sewn into a closed volume (check the graph's faceted faces)."
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


def gmsh_topology(
    step_text: str,
    *,
    target_size: float,
    order: int = 2,
    algorithm: int = HXT,
    optimize: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Mesh a STEP solid with Gmsh — the whole of the opaque part, and no more.

    Nothing here knows about the ownership graph.  What comes back is what
    Gmsh alone can say: node positions, cells, and *which CAD entity owns
    each node*, expressed as the surface tags whose closure the node lies on
    (:func:`assign_ownership` is what turns those into patches).  Keeping the
    cut here is what lets the mesher live behind a Tesseract — a container
    can be handed a STEP file and cannot be handed a :class:`BRep`, whose
    patch fields are live Python callables.

    Args:
        step_text: The STEP file's contents (ASCII), as
            :func:`~cadjoint.brep.step.save_brep_step` writes it.
        target_size: Uniform element size in the file's own units.
        order: 1 for TET4, 2 for TET10.
        algorithm: Gmsh's ``Mesh.Algorithm3D``; :data:`HXT` by default.
        optimize: Run Gmsh's tet optimizer after generation.
        verbose: Let Gmsh write to the terminal.

    Returns:
        ``points`` ``(n, 3)``, ``cells`` ``(t, 4|10)`` in meshio order,
            ``entity_dim`` ``(n,)``, ``bounding_surfaces`` ``(n, k)`` of OCC
            surface tags (``-1`` padded), ``num_surface``,
            ``num_corner_points``, ``edge_parents`` (``None`` at order 1),
            ``bounds`` ``(2, 3)``, ``cad_entities``, and the timings.

    Raises:
        ImportError: If the ``gmsh`` extra is not installed.
        RuntimeError: If the file carries no solid, or Gmsh makes no tets.
        ValueError: If ``order`` is not 1 or 2.
    """
    import tempfile

    if order not in (1, 2):
        raise ValueError(f"order must be 1 (TET4) or 2 (TET10); got {order}.")
    with _gmsh_session(verbose) as gmsh, tempfile.TemporaryDirectory("cadjoint-gmsh") as scratch:
        path = Path(scratch) / "brep.step"
        path.write_text(step_text, encoding="ascii")
        started = time.perf_counter()
        # OCCT's STEP reader targets millimetres and the graph writes metres.
        # Without this the model is 1000x too big and a sane target size asks
        # for something like 1e10 elements — the first thing that bit here.
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.add("cadjoint_brep")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        import_seconds = time.perf_counter() - started

        entities = {dim: gmsh.model.getEntities(dim) for dim in (0, 1, 2, 3)}
        _check_import(step_text, entities)
        for name, value in (
            ("Mesh.MeshSizeMin", target_size),
            ("Mesh.MeshSizeMax", target_size),
            ("Mesh.MeshSizeFromCurvature", 0),
            ("Mesh.MeshSizeFromPoints", 0),
            ("Mesh.MeshSizeExtendFromBoundary", 0),
            ("Mesh.Algorithm3D", algorithm),
            ("Mesh.ElementOrder", order),
            ("Mesh.Optimize", 1 if optimize else 0),
            # 0 keeps the midside nodes ON the CAD surface, which is the
            # whole reason for going through a kernel: a curved midside is
            # what a straight-sided promotion cannot give.
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
    return result


def _check_import(step_text: str, entities: dict) -> None:
    """Refuse a STEP that OCCT could only read as a fragment.

    Gmsh imports the *highest-dimensional* shape and nothing else.  When the
    graph's shell does not sew — a dual-contoured surface that folds on
    itself at the declared resolution is the standing case — OCCT hands back
    a compound of one small valid solid and one large open shell, and Gmsh
    keeps the solid.  Meshing then succeeds, quickly, on a part that is not
    the part: the end-cap housing at ``(26, 26, 13)`` comes through as twelve
    faces of a 0.1 m chip out of a 2 m casting.

    Counting ``ADVANCED_FACE`` in the file the writer produced and comparing
    it with what arrived is what turns that into an error.  It is a
    text-level check on purpose: it needs no kernel and it is exact, because
    :func:`~cadjoint.brep.step.save_brep_step` writes one ``ADVANCED_FACE``
    per face it reports.

    Args:
        step_text: The STEP file's contents.
        entities: ``dim -> [(dim, tag)]`` as Gmsh reports after import.

    Raises:
        RuntimeError: If there is no solid, or if faces went missing.
    """
    declared = step_text.count("ADVANCED_FACE")
    arrived = len(entities[2])
    if not entities[3]:
        raise RuntimeError(
            f"The STEP file's {declared} faces carry no solid volume; the shell did not "
            "sew (inspect save_brep_step's 'dropped' count)."
        )
    if arrived < declared:
        raise RuntimeError(
            f"The STEP file declares {declared} faces but OCCT could only build a solid "
            f"from {arrived} of them, so Gmsh would mesh a fragment of the part rather "
            "than the part. The graph's shell is not watertight at this resolution — "
            "the same defect that makes TetGen refuse the dual-contour surface. "
            "Re-extract the B-rep on a finer grid."
        )


def _harvest(gmsh: Any, entities: dict, order: int) -> dict[str, Any]:
    """Pull nodes, cells and CAD-entity incidence out of a generated model."""
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


# ── putting the graph's ownership back on ────────────────────────────────────


def assign_ownership(
    brep: BRep, topology: dict[str, Any], *, blend_tolerance: float | None = None
) -> dict[str, np.ndarray]:
    """Match Gmsh's CAD entities back to the graph, node by node.

    Args:
        brep: The extracted graph.
        topology: A :func:`gmsh_topology` result.
        blend_tolerance: ``|f_patch|`` above which a surface is declared a
            blend; defaults to ``1e-3`` times the grid diagonal, matching
            :func:`~cadjoint.brep.graph.extract_brep`.

    Returns:
        ``owner_patches`` ``(n, 3)``, ``owner_arity`` ``(n,)``,
            ``blend_mask`` ``(n,)``, ``owner_face`` ``(n,)`` and the counts
            under ``stats``.
    """
    if blend_tolerance is None:
        extent = np.asarray(brep.grid.spacing) * np.asarray(brep.grid.cells)
        blend_tolerance = 1e-3 * float(np.linalg.norm(extent))
    positions = np.asarray(topology["points"], dtype=np.float64)
    entity_dim = np.asarray(topology["entity_dim"], dtype=np.int64)
    bounding = np.asarray(topology["bounding_surfaces"], dtype=np.int64)
    total = positions.shape[0]
    residuals = _patch_residuals(brep, positions)

    # Surfaces first: every curve and point below them inherits their
    # patches.  A surface with no interior node of its own gets no entry —
    # there is nothing to vote or confirm with — so anything bounded only by
    # such surfaces inherits nothing and falls to the scene, which is the
    # conservative answer and the right one for a facet that small.
    surface_patch: dict[int, int] = {}
    surface_face: dict[int, int] = {}
    on_surface = entity_dim == 2
    for tag in np.unique(bounding[on_surface][:, 0]):
        rows = np.flatnonzero(on_surface & (bounding[:, 0] == tag))
        face, patch = _surface_owner(brep, positions[rows], residuals[rows], blend_tolerance)
        surface_patch[int(tag)] = patch
        surface_face[int(tag)] = face

    owner_patches = np.full((total, 3), -1, dtype=np.int32)
    owner_arity = np.zeros(total, dtype=np.int8)
    blend_mask = np.zeros(total, dtype=bool)
    owner_face = np.full(total, -1, dtype=np.int64)
    for row in np.flatnonzero(entity_dim < 3):
        tags = [int(tag) for tag in bounding[row] if tag >= 0]
        owner_face[row] = next((surface_face[tag] for tag in tags if tag in surface_face), -1)
        candidates = [surface_patch[tag] for tag in tags if surface_patch.get(tag, -1) >= 0]
        # A surface node solves one field, a curve node two, a point three:
        # the arity is the entity's codimension, ``3 - dim`` — capped there,
        # not fixed at it, since only the patches that actually vanish at the
        # node are kept (the plate's two circle-seam vertices are dim-0
        # points where only two patches meet, and solving three would
        # over-determine them).
        chosen, arity = _owner_rows(
            candidates, residuals[row], 3 - int(entity_dim[row]), blend_tolerance
        )
        if not arity:
            # No patch owns this node: it is blend geometry, and it is the
            # scene's own zero set that holds it.
            blend_mask[row] = True
            continue
        owner_patches[row] = chosen
        owner_arity[row] = arity
    return {
        "owner_patches": owner_patches,
        "owner_arity": owner_arity,
        "blend_mask": blend_mask,
        "owner_face": owner_face,
        "stats": {
            "nodes": total,
            "nodes_by_dim": {dim: int((entity_dim == dim).sum()) for dim in (0, 1, 2, 3)},
            "blend_nodes": int(blend_mask.sum()),
            "blend_surfaces": int(sum(1 for patch in surface_patch.values() if patch < 0)),
            "arity_counts": {arity: int((owner_arity == arity).sum()) for arity in (0, 1, 2, 3)},
        },
    }


def gmsh_tet_mesh(
    brep: BRep,
    *,
    target_size: float | None = None,
    order: int = 2,
    algorithm: int = HXT,
    optimize: bool = True,
    step_path: str | Path | None = None,
    blend_tolerance: float | None = None,
    plugin: str | None = None,
    verbose: bool = False,
) -> GmshMesh:
    """Mesh the graph's exact STEP with Gmsh and give every node its owner.

    Args:
        brep: The extracted graph.
        target_size: Uniform element size in model units; defaults to the
            graph's own smallest grid spacing, so a declared resolution means
            roughly what it means on the DC path.
        order: 1 for TET4, 2 for TET10 (the reason this module exists).
        algorithm: Gmsh's ``Mesh.Algorithm3D``; :data:`HXT` by default.
        optimize: Run Gmsh's tet optimizer after generation.
        step_path: Where to keep the intermediate STEP; discarded when
            omitted.
        blend_tolerance: See :func:`assign_ownership`.
        plugin: Run the mesher through this registered plugin (the
            ``tet_mesher`` kind) instead of importing Gmsh in this process.
        verbose: Let Gmsh write to the terminal.

    Returns:
        The :class:`GmshMesh`.  ``stats`` carries ``step_seconds``,
            ``import_seconds``, ``mesh_seconds``, ``ownership_seconds``, the
            STEP face plan, the CAD entity counts, and the node counts per
            entity dimension and per arity.

    Raises:
        ImportError: If the ``gmsh`` extra is not installed.
        RuntimeError: If Gmsh imports a solid whose size does not match the
            graph's — the signature of a silent unit conversion.
    """
    spacing = np.asarray(brep.grid.spacing, dtype=np.float64)
    if target_size is None:
        target_size = float(spacing.min())

    started = time.perf_counter()
    step_report = save_brep_step(brep, step_path) if step_path else None
    if step_path is None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="cadjoint-gmsh-") as scratch:
            path = Path(scratch) / "brep.step"
            step_report = save_brep_step(brep, path)
            step_text = path.read_text(encoding="ascii")
    else:
        step_text = Path(step_path).read_text(encoding="ascii")
    step_seconds = time.perf_counter() - started

    options = {
        "target_size": float(target_size),
        "order": order,
        "algorithm": algorithm,
        "optimize": optimize,
    }
    topology = (
        _plugin_topology(plugin, step_text, options)
        if plugin
        else gmsh_topology(step_text, verbose=verbose, **options)
    )

    imported = topology["bounds"][1] - topology["bounds"][0]
    span = np.asarray(brep.points.max(axis=0) - brep.points.min(axis=0))
    # The backstop behind _check_import, and deliberately blunt: 5% is far
    # wider than any chord error and far narrower than the 1000x a millimetre
    # default costs, so what reaches here is a shape that is not the part.
    if not np.allclose(imported, span, rtol=0.05, atol=1e-9):
        raise RuntimeError(
            f"Gmsh meshed a solid spanning {np.round(imported, 6).tolist()} but the graph "
            f"spans {np.round(span, 6).tolist()}. Either the STEP unit conversion did not "
            "take (Geometry.OCCTargetUnit) or the file's shell sewed into something other "
            "than the part."
        )

    started = time.perf_counter()
    ownership = assign_ownership(brep, topology, blend_tolerance=blend_tolerance)
    ownership_seconds = time.perf_counter() - started

    return GmshMesh(
        points=topology["points"],
        cells=topology["cells"],
        owner_patches=ownership["owner_patches"],
        owner_arity=ownership["owner_arity"],
        blend_mask=ownership["blend_mask"],
        owner_face=ownership["owner_face"],
        entity_dim=np.asarray(topology["entity_dim"], dtype=np.int8),
        num_surface=int(topology["num_surface"]),
        num_corner_points=int(topology["num_corner_points"]),
        edge_parents=topology["edge_parents"],
        max_step=0.5 * float(np.linalg.norm(spacing)),
        stats={
            "target_size": float(target_size),
            "step_seconds": step_seconds,
            "import_seconds": topology["import_seconds"],
            "mesh_seconds": topology["mesh_seconds"],
            "ownership_seconds": ownership_seconds,
            "cells": int(np.asarray(topology["cells"]).shape[0]),
            "step_faces": (step_report or {}).get("faces", {}),
            "cad_entities": topology["cad_entities"],
            **ownership["stats"],
        },
    )


def _plugin_topology(name: str, step_text: str, options: dict) -> dict[str, Any]:
    """Run :func:`gmsh_topology` through a registered ``tet_mesher`` plugin.

    The plugin's ``apply`` is the discovery call: templates are empty, so it
    runs Gmsh and reports the topology it found.

    Args:
        name: Registered plugin name, or ``"default"`` for whatever fills
            the ``tet_mesher`` kind.
        step_text: The STEP file's contents.
        options: ``target_size``, ``order``, ``algorithm``, ``optimize``.

    Returns:
        The same mapping :func:`gmsh_topology` returns.
    """
    from cadjoint.plugins import get_plugin, plugin_for_kind

    plugin = plugin_for_kind(TET_MESHER_KIND) if name == "default" else get_plugin(name)
    # Every template is empty: that is what puts the call in discovery mode,
    # where Gmsh runs and the topology it finds comes back.  The frozen mode
    # (templates filled, ``node_positions`` given) is the traced call, and
    # this function is never on that path — the positions it would hand back
    # are the ones :func:`recompute_gmsh_points` solves here.
    result = plugin.apply(
        {
            "step": step_text,
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
    }


# ── differentiable positions over the frozen topology ────────────────────────


def recompute_gmsh_points(
    brep: BRep,
    mesh: GmshMesh,
    *,
    scene: Any = None,
    smooth_passes: int = 0,
    max_step: float | None = None,
) -> np.ndarray:
    """Re-solve every node against its own owners; the concrete forward.

    The whole node set is replaced by the solution of its own system: one
    field on a surface, two on a curve, three at a point, the scene's own
    field on a blend.  Midside nodes are solved the same way as corners,
    which is what keeps a midside on the cylinder rather than at the
    straight-sided midpoint.

    Args:
        brep: The graph whose patch table supplies the fields.
        mesh: The frozen Gmsh topology.
        scene: Root SDF node, needed only when the mesh has blend nodes.
        smooth_passes: Laplacian passes carrying boundary motion into the
            volume nodes; 0 leaves them where Gmsh put them.
        max_step: Projection clamp; defaults to the mesh's own.

    Returns:
        Node positions shaped like :attr:`GmshMesh.points`.

    Raises:
        ValueError: If the mesh has blend nodes but no ``scene`` was given.
    """
    if max_step is None:
        max_step = mesh.max_step
    field_table = [patch.field for patch in brep.patches]
    solved = mesh.points.copy()
    by_arity: dict[int, list[int]] = {}
    for row in range(solved.shape[0]):
        arity = int(mesh.owner_arity[row])
        if arity:
            by_arity.setdefault(arity, []).append(row)
    for _arity, rows in sorted(by_arity.items()):
        index = np.asarray(rows, dtype=np.int64)
        members = mesh.owner_patches[index, : int(mesh.owner_arity[index[0]])].astype(np.int32)
        solved[index] = project_batched(field_table, members, solved[index], max_step=max_step)
    if mesh.blend_mask.any():
        if scene is None:
            raise ValueError(
                f"{int(mesh.blend_mask.sum())} nodes lie on blend faces, which no patch owns; "
                "pass scene= so they can be solved against the scene's own zero set."
            )
        index = np.flatnonzero(mesh.blend_mask)
        solved[index] = project_fields([scene], solved[index], max_step=max_step)
    if smooth_passes > 0:
        solved = _smoothed(mesh, solved, smooth_passes)
    return solved


def _smoothed(mesh: GmshMesh, solved: np.ndarray, passes: int) -> np.ndarray:
    """Carry boundary motion into the volume nodes by Laplacian passes."""
    moving = mesh.entity_dim < 3
    neighbours = _node_adjacency(mesh)
    positions = solved.copy()
    for _ in range(passes):
        averaged = np.zeros_like(positions)
        counts = np.zeros(positions.shape[0])
        np.add.at(averaged, neighbours[:, 0], positions[neighbours[:, 1]])
        np.add.at(counts, neighbours[:, 0], 1.0)
        interior = ~moving & (counts > 0)
        positions[interior] = averaged[interior] / counts[interior, None]
    return positions


def _node_adjacency(mesh: GmshMesh) -> np.ndarray:
    """Directed node-pair list from the cell connectivity, both ways."""
    cells = np.asarray(mesh.cells, dtype=np.int64)
    width = cells.shape[1]
    left = np.repeat(cells, width, axis=1).reshape(-1)
    right = np.tile(cells, (1, width)).reshape(-1)
    keep = left != right
    return np.stack([left[keep], right[keep]], axis=1)


def parameterised_points(
    scene: Any,
    mesh: GmshMesh,
    params: dict,
    *,
    max_step: float | None = None,
    steps: int = 8,
):
    """The same solve, traced in the scene's free parameters.

    :func:`recompute_gmsh_points` closes over concrete patch fields, which is
    fast and carries no derivative.  This rebuilds the fields under whatever
    parameter values are handed in — :func:`cadjoint.brep.drag.patch_field_fn`
    is the mechanism — so ``jax.grad`` of anything downstream reaches the
    design parameters through the projection's implicit-function adjoint.

    One :func:`~cadjoint.brep.project.project` call is made per *distinct
    owner set* rather than per arity: the traced field builder is a Python
    closure over a fixed patch list, so it cannot gather per-point the way
    :func:`~cadjoint.brep.project.project_batched` does.  A hard part has a
    few dozen distinct sets, which is a few dozen calls — fine for a
    gradient, and the reason the concrete forward keeps its own fast path.

    Args:
        scene: Root SDF node.
        mesh: The frozen Gmsh topology.
        params: Free-parameter mapping (partial mappings merge over the
            scene's current values).
        max_step: Projection clamp; defaults to the mesh's own.
        steps: Newton iterations.

    Returns:
        A traced ``(n, 3)`` array of node positions.  Blend and volume nodes
            are returned at their frozen positions, so they carry no
            parameter derivative.
    """
    import jax.numpy as jnp

    from cadjoint.brep.drag import patch_field_fn
    from cadjoint.brep.project import project

    if max_step is None:
        max_step = mesh.max_step
    # Left at the default dtype rather than forced to float32 the way the
    # forward-only project_fields does: a finite-difference check of a
    # volume against a radius needs the x64 the FEM suite already enables.
    positions = jnp.asarray(mesh.points)
    groups: dict[tuple[int, ...], list[int]] = {}
    for row in range(mesh.points.shape[0]):
        arity = int(mesh.owner_arity[row])
        if arity:
            key = tuple(int(v) for v in mesh.owner_patches[row, :arity])
            groups.setdefault(key, []).append(row)
    for owners, rows in groups.items():
        index = jnp.asarray(rows, dtype=jnp.int32)
        solved = project(
            patch_field_fn(scene, owners),
            params,
            positions[index],
            max_step=max_step,
            steps=steps,
        )
        positions = positions.at[index].set(solved)
    return positions


# ── handover ─────────────────────────────────────────────────────────────────


def tet_mesh_from_gmsh(brep: BRep, mesh: GmshMesh, *, points: np.ndarray | None = None):
    """Wrap a :class:`GmshMesh` as the FEM layer's :class:`TetMesh`.

    The node layout is already the one :class:`~cadjoint.fem.tetmesh.TetMesh`
    documents — boundary corners, interior corners, midsides — so this only
    attaches the boundary triangles and the frozen base positions.

    Note that ``edge_parents`` here describes *which corners a midside lies
    between*, not a straight-sided construction: the midsides are on the CAD
    surface and are re-solved by :func:`recompute_gmsh_points`, not rebuilt
    as corner midpoints.  Consumers that only read it as a corner-pair table
    (BC completion, TET10 element assembly) are unaffected.

    Args:
        brep: The graph the mesh came from (its grid rides along).
        mesh: The frozen Gmsh mesh.
        points: Positions to use instead of :attr:`GmshMesh.points`.

    Returns:
        A :class:`~cadjoint.fem.tetmesh.TetMesh`.
    """
    from cadjoint.fem.tetmesh import TetMesh, tet_boundary_faces

    positions = mesh.points if points is None else np.asarray(points, dtype=np.float64)
    corners = np.asarray(mesh.cells[:, :4], dtype=np.int64)
    return TetMesh(
        points=positions,
        cells=np.asarray(mesh.cells, dtype=np.int32),
        num_surface=mesh.num_surface,
        boundary_tris=tet_boundary_faces(corners),
        base_points=mesh.points.copy(),
        max_step=mesh.max_step,
        grid=brep.grid,
        edge_parents=mesh.edge_parents,
    )
