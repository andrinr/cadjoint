"""Post-extraction simplification of dual-contour meshes.

Seventh stage of the meshing pipeline.  The uniform extraction places one
vertex per active cell, so planar and gently curved regions carry far more
triangles than their geometry needs.  :func:`simplify_mesh` merges those
regions after the fact by collapsing mesh edges — vertex clustering in the
spirit of Ju et al.'s QEF-merged octree leaves, run on the extracted mesh
instead of inside the extractor — under three hard guarantees:

- **Bounded error.**  A collapse must keep the merged region's QEF error
  (RMS distance to the region's accumulated, area-weighted tangent planes)
  under ``error``, and every triangle the collapse modifies is re-sampled
  against the implicit field: a centroid or edge midpoint farther than
  ``error`` from the zero set rejects the collapse.
- **Feature preservation.**  Mesh edges whose dihedral angle exceeds
  ``feature_angle`` — creases, corners, CSG seams — pin their endpoints;
  callers can pin additional vertex rows through ``feature_mask`` (for
  example the non-``FACE`` rows of
  :func:`~cadjoint.meshing.features.classify_feature_cells` or the seam rows
  of :func:`~cadjoint.meshing.features.detect_branch_changes`, both of which
  are aligned with the mesh's vertex rows).  Pinned vertices are never moved
  or removed, so features survive bitwise.
- **Topology safety.**  Only manifold interior edges collapse, the link
  condition rejects any collapse that would pinch the surface, and a
  collapse that would flip or degenerate a surviving triangle is refused.
  A watertight input stays watertight with the same Euler characteristic.

Collapses are half-edge collapses (one endpoint slides into the other), so
every surviving vertex keeps its extracted position exactly — sharp QEF
corner and seam placements are preserved to the bit.  Everything here is
concrete host-side numpy; nothing sits on a differentiation path, and the
result is deterministic for a given input.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cadjoint.meshing.dual_contouring import Mesh

# A modified triangle whose new normal has squared length below this is
# treated as degenerate and rejects its collapse.
_MIN_NORMAL_SQ = 1e-24

# Face normals shorter than this mark the face (and its vertices) degenerate.
_DEGENERATE_NORMAL = 1e-15


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unnormalized face normals (twice the area vector) of a triangle list."""
    edge_ab = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge_ac = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    return np.cross(edge_ab, edge_ac)


def _edge_occurrences(
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected edges of a triangle list with occurrence bookkeeping.

    Args:
        faces: Triangle indices shaped ``(m, 3)``.

    Returns:
        ``(edges, counts, order, starts)``: the unique undirected edges and
            their usage counts, plus ``order`` sorting the ``3 m`` directed
            occurrences by edge and ``starts`` indexing each edge's first
            occurrence in that order.  Occurrence ``j`` lives in face ``j % m``
            opposite corner ``(j // m + 2) % 3``.
    """
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    edges, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return edges, counts, order, starts


def _protected_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    preserve_features: bool,
    feature_mask: np.ndarray | None,
    cos_threshold: float,
) -> np.ndarray:
    """Vertices that must never move or vanish.

    Endpoints of boundary and non-manifold edges (and of degenerate faces)
    are always pinned for safety.  With ``preserve_features``, endpoints of
    mesh edges whose dihedral angle exceeds the threshold — creases,
    corners, CSG seams — and any rows in ``feature_mask`` are pinned too.
    """
    normals = _face_normals(vertices, faces)
    lengths = np.linalg.norm(normals, axis=1)
    unit = normals / np.maximum(lengths, 1e-300)[:, None]

    edges, counts, order, starts = _edge_occurrences(faces)
    face_count = faces.shape[0]
    protected = np.zeros(vertices.shape[0], dtype=bool)
    protected[edges[counts != 2].reshape(-1)] = True
    protected[faces[lengths < _DEGENERATE_NORMAL].reshape(-1)] = True
    if preserve_features:
        manifold = counts == 2
        first = order[starts[manifold]]
        second = order[starts[manifold] + 1]
        cosine = np.einsum("ki,ki->k", unit[first % face_count], unit[second % face_count])
        protected[edges[manifold][cosine < cos_threshold].reshape(-1)] = True
        if feature_mask is not None:
            protected |= feature_mask
    return protected


def _vertex_quadrics(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted plane quadrics accumulated per vertex.

    Returns ``(quadratic, linear, constant, weight)`` such that the QEF of
    vertex ``v`` at point ``p`` is ``p·Q·p + 2 l·p + c`` with total plane
    weight ``w``; dividing by ``w`` gives the mean squared distance to the
    vertex's incident face planes.
    """
    normals = _face_normals(vertices, faces)
    lengths = np.linalg.norm(normals, axis=1)
    areas = 0.5 * lengths
    unit = normals / np.maximum(lengths, 1e-300)[:, None]
    offsets = -np.einsum("ki,ki->k", unit, vertices[faces[:, 0]])

    outer = areas[:, None, None] * unit[:, :, None] * unit[:, None, :]
    linear = (areas * offsets)[:, None] * unit
    constant = areas * offsets**2

    count = vertices.shape[0]
    quadratic_sum = np.zeros((count, 3, 3))
    linear_sum = np.zeros((count, 3))
    constant_sum = np.zeros(count)
    weight_sum = np.zeros(count)
    for corner in range(3):
        np.add.at(quadratic_sum, faces[:, corner], outer)
        np.add.at(linear_sum, faces[:, corner], linear)
        np.add.at(constant_sum, faces[:, corner], constant)
        np.add.at(weight_sum, faces[:, corner], areas)
    return quadratic_sum, linear_sum, constant_sum, weight_sum


def _collapse_pass(
    vertices: np.ndarray,
    faces: np.ndarray,
    protected: np.ndarray,
    quadrics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    evaluate: Callable[[Array], Array],
    error: float,
) -> tuple[np.ndarray, int]:
    """Apply one deterministic batch of independent half-edge collapses.

    Candidates over manifold interior edges are ordered by QEF error and
    accepted greedily under the link condition and the normal-flip check;
    each acceptance locks the removed vertex's one-ring so every candidate's
    checks stay valid against the pass-start topology.  Tentative collapses
    are then validated against the implicit field in one batched evaluation
    (centroids and edge midpoints of every modified triangle must stay
    within ``error`` of the zero set) and the survivors are applied.

    Returns:
        ``(faces, applied)``: the rewritten triangle list and how many
            collapses were applied.  The quadric arrays are updated in place
            (the kept vertex absorbs the removed vertex's planes).
    """
    quadratic, linear, constant, weight = quadrics
    count = vertices.shape[0]
    face_count = faces.shape[0]

    edges, counts, order, starts = _edge_occurrences(faces)
    manifold = np.flatnonzero(counts == 2)
    if manifold.size == 0:
        return faces, 0
    first = order[starts[manifold]]
    second = order[starts[manifold] + 1]
    edge_faces = np.stack([first % face_count, second % face_count], axis=1)
    opposite = np.stack(
        [
            faces[first % face_count, (first // face_count + 2) % 3],
            faces[second % face_count, (second // face_count + 2) % 3],
        ],
        axis=1,
    )

    neighbors: list[set[int]] = [set() for _ in range(count)]
    for u, v in edges:
        neighbors[u].add(int(v))
        neighbors[v].add(int(u))
    vertex_faces: list[list[int]] = [[] for _ in range(count)]
    for index, (i, j, k) in enumerate(faces):
        vertex_faces[i].append(index)
        vertex_faces[j].append(index)
        vertex_faces[k].append(index)

    # Both directions of every manifold edge, filtered by protection and the
    # QEF bound (vectorized), then ordered by (error, removed, kept).
    endpoint_a = edges[manifold]
    candidate_a = np.concatenate([endpoint_a[:, 0], endpoint_a[:, 1]])
    candidate_b = np.concatenate([endpoint_a[:, 1], endpoint_a[:, 0]])
    candidate_faces = np.concatenate([edge_faces, edge_faces])
    candidate_opposite = np.concatenate([opposite, opposite])

    target = vertices[candidate_b]
    merged_quadratic = quadratic[candidate_a] + quadratic[candidate_b]
    merged_linear = linear[candidate_a] + linear[candidate_b]
    merged_constant = constant[candidate_a] + constant[candidate_b]
    merged_weight = weight[candidate_a] + weight[candidate_b]
    qef = (
        np.einsum("ki,kij,kj->k", target, merged_quadratic, target)
        + 2.0 * np.einsum("ki,ki->k", merged_linear, target)
        + merged_constant
    )
    rms = np.sqrt(np.maximum(qef, 0.0) / np.maximum(merged_weight, 1e-300))
    eligible = np.flatnonzero(~protected[candidate_a] & (rms <= error))
    ordered = eligible[np.lexsort((candidate_b[eligible], candidate_a[eligible], rms[eligible]))]

    locked = np.zeros(count, dtype=bool)
    tentative: list[tuple[int, int]] = []
    slices: list[tuple[int, int]] = []
    sample_blocks: list[np.ndarray] = []
    offset = 0
    for candidate in ordered:
        removed = int(candidate_a[candidate])
        kept = int(candidate_b[candidate])
        if locked[removed] or locked[kept]:
            continue
        wing_a = int(candidate_opposite[candidate, 0])
        wing_b = int(candidate_opposite[candidate, 1])
        if wing_a == wing_b:
            continue
        # Link condition: the shared neighbors of the endpoints must be
        # exactly the two wing vertices, or the collapse pinches the surface.
        if (neighbors[removed] & neighbors[kept]) != {wing_a, wing_b}:
            continue
        dropped = {int(candidate_faces[candidate, 0]), int(candidate_faces[candidate, 1])}
        modified = [index for index in vertex_faces[removed] if index not in dropped]
        if not modified:
            continue
        old_triangles = faces[modified]
        new_triangles = np.where(old_triangles == removed, kept, old_triangles)
        old_normals = _face_normals(vertices, old_triangles)
        new_normals = _face_normals(vertices, new_triangles)
        if np.any(np.einsum("ki,ki->k", new_normals, new_normals) < _MIN_NORMAL_SQ):
            continue
        if np.any(np.einsum("ki,ki->k", old_normals, new_normals) <= 0.0):
            continue
        corners = vertices[new_triangles]
        centroids = corners.mean(axis=1)
        midpoints = 0.5 * (corners + np.roll(corners, -1, axis=1))
        samples = np.concatenate([centroids[:, None, :], midpoints], axis=1).reshape((-1, 3))
        tentative.append((removed, kept))
        slices.append((offset, offset + samples.shape[0]))
        sample_blocks.append(samples)
        offset += samples.shape[0]
        locked[removed] = True
        locked[kept] = True
        for vertex in neighbors[removed]:
            locked[vertex] = True

    if not tentative:
        return faces, 0

    deviations = np.abs(
        np.asarray(evaluate(jnp.asarray(np.concatenate(sample_blocks))), dtype=np.float64)
    )
    replacement = np.arange(count, dtype=np.int64)
    applied = 0
    for (removed, kept), (start, stop) in zip(tentative, slices):
        if float(deviations[start:stop].max()) > error:
            continue
        replacement[removed] = kept
        quadratic[kept] += quadratic[removed]
        linear[kept] += linear[removed]
        constant[kept] += constant[removed]
        weight[kept] += weight[removed]
        applied += 1
    if applied == 0:
        return faces, 0

    rewritten = replacement[faces]
    keep = (
        (rewritten[:, 0] != rewritten[:, 1])
        & (rewritten[:, 1] != rewritten[:, 2])
        & (rewritten[:, 2] != rewritten[:, 0])
    )
    return rewritten[keep], applied


def simplify_mesh(
    mesh: Mesh,
    sdf: Callable[[Array], Array],
    *,
    error: float,
    preserve_features: bool = True,
    feature_mask: np.ndarray | None = None,
    feature_angle: float = 30.0,
    max_passes: int = 100,
) -> Mesh:
    """Simplify a dual-contour mesh by error-bounded, feature-safe collapses.

    Repeatedly applies batches of independent half-edge collapses (see the
    module docstring for the guarantees) until no collapse passes the error
    bound and the safety checks.  Because collapses are half-edge, every
    surviving vertex keeps its extracted position bitwise; with
    ``preserve_features`` (the default) all vertices on sharp mesh edges —
    box corners, crease chains, CSG seams — survive unchanged.

    Args:
        mesh: The extracted mesh to simplify.
        sdf: The implicit field the mesh was extracted from, mapping one
            point ``(3,)`` to a scalar; used to reject collapses whose
            modified triangles stray from the zero set.
        error: Positive deviation bound in world units.  Bounds both the
            merged region's RMS QEF error and the sampled field deviation
            (centroids and edge midpoints) of every modified triangle.
        preserve_features: Pin the endpoints of sharp mesh edges (dihedral
            angle above ``feature_angle``) and any ``feature_mask`` rows.
            ``False`` disables both signals; boundary and non-manifold
            protection always stays on.
        feature_mask: Optional boolean mask over the mesh's vertex rows
            pinning extra vertices — for example non-``FACE`` rows from
            :func:`~cadjoint.meshing.features.classify_feature_cells` or seam
            rows from
            :func:`~cadjoint.meshing.features.detect_branch_changes`, both
            aligned with the vertex rows of the same extraction.
        feature_angle: Dihedral angle in degrees, strictly between 0 and
            180, above which a mesh edge counts as sharp.
        max_passes: Safety cap on collapse passes; the loop normally stops
            earlier, when a pass applies no collapse.

    Returns:
        A new :class:`~cadjoint.meshing.dual_contouring.Mesh` whose vertices
        are a subset of the input's (positions, normals, and cells carried
        over unchanged) with reindexed faces.  ``quads`` is empty: collapsed
        connectivity no longer corresponds to extraction quads.

    Raises:
        ValueError: If ``error`` is not positive, ``feature_angle`` is not
            strictly between 0 and 180, ``max_passes`` is not positive, or
            ``feature_mask`` has the wrong shape.
    """
    if not error > 0:
        raise ValueError("error must be positive.")
    if not 0 < feature_angle < 180:
        raise ValueError("feature_angle must be strictly between 0 and 180 degrees.")
    if max_passes < 1:
        raise ValueError("max_passes must be positive.")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 3))
    count = vertices.shape[0]
    mask = None
    if feature_mask is not None:
        mask = np.asarray(feature_mask, dtype=bool)
        if mask.shape != (count,):
            raise ValueError(f"feature_mask must have shape ({count},); received {mask.shape}.")
    if faces.shape[0] == 0:
        return mesh

    protected = _protected_vertices(
        vertices,
        faces,
        preserve_features=preserve_features,
        feature_mask=mask,
        cos_threshold=float(np.cos(np.radians(feature_angle))),
    )
    quadrics = _vertex_quadrics(vertices, faces)
    evaluate = jax.vmap(sdf)

    for _ in range(max_passes):
        faces, applied = _collapse_pass(vertices, faces, protected, quadrics, evaluate, error)
        if applied == 0:
            break

    used = np.unique(faces)
    remap = np.full(count, -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    original_vertices = np.asarray(mesh.vertices)
    return Mesh(
        vertices=jnp.asarray(original_vertices[used]),
        faces=remap[faces].astype(np.int32),
        quads=np.empty((0, 4), dtype=np.int32),
        normals=jnp.asarray(np.asarray(mesh.normals)[used]),
        cells=np.asarray(mesh.cells)[used].copy(),
    )
