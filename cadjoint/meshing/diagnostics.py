"""Mesh diagnostics: deviation from the field, self-intersections, and quality.

Sixth stage of the meshing pipeline.  Everything here inspects a concrete
extracted :class:`~cadjoint.meshing.dual_contouring.Mesh` after the fact —
host-side numpy throughout, with JAX used only to evaluate the implicit
field at sample points.  Nothing in this module sits on a differentiation
path.

Two of the checks are *sampled* rather than exhaustive, and say so in their
docstrings: :func:`surface_deviation` measures the one-sided mesh-to-surface
distance at random points on the mesh, and :func:`self_intersections` tests
a random subset of non-adjacent triangle pairs.  Both are deterministic for
a given seed.  :func:`mesh_report` bundles them with triangle-quality
statistics, watertightness, and the Euler characteristic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jax import Array

    from cadjoint.meshing.dual_contouring import Mesh

# Barycentric / parametric tolerance for the segment-triangle crossing test:
# crossings closer than this to a triangle edge or a segment endpoint do not
# count, so exactly-touching configurations are not reported.
_CROSSING_EPSILON = 1e-9

# Triangles with area below this are counted as degenerate and excluded from
# the angle / aspect-ratio statistics.
_DEGENERATE_AREA = 1e-12


def _mesh_arrays(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 3))
    return vertices, faces


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges_ab = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edges_ac = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(edges_ab, edges_ac), axis=1)


def _undirected_edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edges of a triangle list and their usage counts."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)


def surface_deviation(
    sdf: Callable[[Array], Array],
    mesh: Mesh,
    *,
    samples: int = 2048,
    seed: int = 0,
) -> dict[str, Any]:
    """Sampled one-sided deviation of the mesh from the implicit surface.

    Draws ``samples`` points uniformly by triangle area (triangles chosen
    with probability proportional to their area, then uniform barycentric
    coordinates within each) using the deterministic
    ``np.random.default_rng(seed)``, and evaluates ``|sdf|`` at every point.
    For a signed distance field this is the exact mesh-to-surface distance
    at the samples; for a general implicit field it is the field residual.

    This is *one-sided*: it bounds how far the mesh strays from the zero
    set, but says nothing about surface regions the mesh misses entirely.
    The opposite direction (surface-to-mesh) needs a closest-point query
    against the mesh and is out of scope here.

    Args:
        sdf: Implicit field mapping one point ``(3,)`` to a scalar; sampled
            points are evaluated through ``jax.vmap``.
        mesh: The extracted mesh to measure.
        samples: Number of surface samples; must be positive.
        seed: Seed for the deterministic sample draw.

    Returns:
        Dict with ``max_abs`` and ``mean_abs`` (floats, the absolute field
        values over the samples) and ``samples`` (the count actually drawn).

    Raises:
        ValueError: If ``samples`` is not positive, the mesh has no faces,
            or the total triangle area is zero.
    """
    if samples <= 0:
        raise ValueError("samples must be positive.")
    vertices, faces = _mesh_arrays(mesh)
    if faces.shape[0] == 0:
        raise ValueError("Cannot measure deviation of a mesh with no faces.")
    areas = _triangle_areas(vertices, faces)
    total_area = float(areas.sum())
    if not total_area > 0:
        raise ValueError("Cannot sample a mesh whose total triangle area is zero.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(faces.shape[0], size=samples, p=areas / total_area)
    r1 = rng.random(samples)
    r2 = rng.random(samples)
    sqrt_r1 = np.sqrt(r1)
    weight_a = 1.0 - sqrt_r1
    weight_b = sqrt_r1 * (1.0 - r2)
    weight_c = sqrt_r1 * r2
    corners = vertices[faces[chosen]]  # (samples, 3, 3)
    points = (
        weight_a[:, None] * corners[:, 0]
        + weight_b[:, None] * corners[:, 1]
        + weight_c[:, None] * corners[:, 2]
    )

    values = np.abs(np.asarray(jax.vmap(sdf)(jnp.asarray(points)), dtype=np.float64))
    return {
        "max_abs": float(values.max()),
        "mean_abs": float(values.mean()),
        "samples": int(samples),
    }


def _segments_cross_triangles(
    start: np.ndarray, end: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Whether each open segment crosses the interior of its triangle.

    Segment-plane clipping: each segment is clipped against its triangle's
    plane, and the clip point is tested against the triangle interior in
    barycentric coordinates.  Segments parallel to the plane (including the
    coplanar case) and degenerate triangles never report a crossing.

    Args:
        start: Segment start points ``(k, 3)``.
        end: Segment end points ``(k, 3)``.
        triangles: Triangle corner positions ``(k, 3, 3)``.

    Returns:
        Boolean array ``(k,)``.
    """
    edge_ab = triangles[:, 1] - triangles[:, 0]
    edge_ac = triangles[:, 2] - triangles[:, 0]
    normal = np.cross(edge_ab, edge_ac)
    direction = end - start
    denominator = np.einsum("ki,ki->k", normal, direction)
    scale = np.linalg.norm(normal, axis=1) * np.linalg.norm(direction, axis=1)
    oblique = np.abs(denominator) > _CROSSING_EPSILON * np.maximum(scale, 1e-300)
    safe_denominator = np.where(oblique, denominator, 1.0)
    t = np.einsum("ki,ki->k", normal, triangles[:, 0] - start) / safe_denominator
    crossing = oblique & (t > _CROSSING_EPSILON) & (t < 1.0 - _CROSSING_EPSILON)

    clip = start + t[:, None] * direction - triangles[:, 0]
    d00 = np.einsum("ki,ki->k", edge_ab, edge_ab)
    d01 = np.einsum("ki,ki->k", edge_ab, edge_ac)
    d11 = np.einsum("ki,ki->k", edge_ac, edge_ac)
    d20 = np.einsum("ki,ki->k", clip, edge_ab)
    d21 = np.einsum("ki,ki->k", clip, edge_ac)
    gram = d00 * d11 - d01 * d01
    nondegenerate = gram > _CROSSING_EPSILON * np.maximum(d00 * d11, 1e-300)
    safe_gram = np.where(nondegenerate, gram, 1.0)
    u = (d11 * d20 - d01 * d21) / safe_gram
    v = (d00 * d21 - d01 * d20) / safe_gram
    inside = (
        nondegenerate
        & (u > _CROSSING_EPSILON)
        & (v > _CROSSING_EPSILON)
        & (u + v < 1.0 - _CROSSING_EPSILON)
    )
    return crossing & inside


def self_intersections(mesh: Mesh, *, pairs: int = 4096, seed: int = 0) -> dict[str, Any]:
    """Sampled self-intersection test over non-adjacent triangle pairs.

    Draws ``pairs`` random triangle pairs with the deterministic
    ``np.random.default_rng(seed)``, discards pairs that share a vertex
    index (adjacent triangles legitimately touch), and tests the rest by
    segment-plane clipping: two non-coplanar triangles intersect exactly
    when an edge of one crosses the interior of the other, so all six edge
    segments are clipped against the opposing triangle's plane and checked
    in barycentric coordinates.  Coplanar overlaps and exact touching are
    not reported.

    This is a *sampled* check, not a proof: it tests ``O(pairs)`` of the
    ``O(t^2)`` triangle pairs, so a clean report means no intersection was
    found among the tested pairs, not that none exists.  Raise ``pairs``
    for more coverage.

    Args:
        mesh: The mesh to test.
        pairs: Number of random pairs to draw; must be positive.
        seed: Seed for the deterministic pair draw.

    Returns:
        Dict with ``count`` (number of distinct intersecting pairs found),
        ``pairs`` (their sorted triangle-index pairs, shaped ``(count, 2)``),
        and ``tested`` (how many non-adjacent pairs were actually tested).

    Raises:
        ValueError: If ``pairs`` is not positive.
    """
    if pairs <= 0:
        raise ValueError("pairs must be positive.")
    vertices, faces = _mesh_arrays(mesh)
    if faces.shape[0] < 2:
        return {"count": 0, "pairs": np.empty((0, 2), dtype=np.int64), "tested": 0}

    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, faces.shape[0], size=(pairs, 2))
    drawn = np.unique(np.sort(drawn, axis=1), axis=0)
    first_faces = faces[drawn[:, 0]]
    second_faces = faces[drawn[:, 1]]
    adjacent = (first_faces[:, :, None] == second_faces[:, None, :]).any(axis=(1, 2))
    tested = drawn[~adjacent]
    if tested.shape[0] == 0:
        return {"count": 0, "pairs": np.empty((0, 2), dtype=np.int64), "tested": 0}

    first = vertices[faces[tested[:, 0]]]  # (m, 3, 3)
    second = vertices[faces[tested[:, 1]]]
    intersecting = np.zeros(tested.shape[0], dtype=bool)
    for edge in range(3):
        start_a, end_a = first[:, edge], first[:, (edge + 1) % 3]
        start_b, end_b = second[:, edge], second[:, (edge + 1) % 3]
        intersecting |= _segments_cross_triangles(start_a, end_a, second)
        intersecting |= _segments_cross_triangles(start_b, end_b, first)

    offending = tested[intersecting]
    return {
        "count": int(offending.shape[0]),
        "pairs": offending,
        "tested": int(tested.shape[0]),
    }


def triangle_quality(mesh: Mesh) -> dict[str, Any]:
    """Triangle shape statistics: minimum angles, aspect ratios, degeneracy.

    The aspect ratio is the longest edge over the smallest altitude
    (``l_max^2 / (2 area)``), which is ``2 / sqrt(3) ≈ 1.155`` for an
    equilateral triangle and grows without bound for slivers.  Triangles
    with area below ``1e-12`` are counted as degenerate and excluded from
    the percentile statistics (which are ``nan`` if nothing remains).

    Args:
        mesh: The mesh to measure.

    Returns:
        Dict with ``min_angle_p5`` / ``min_angle_p50`` (degrees),
        ``aspect_p50`` / ``aspect_p95``, ``degenerate_count``, and
        ``triangle_count``.
    """
    vertices, faces = _mesh_arrays(mesh)
    if faces.shape[0] == 0:
        return {
            "min_angle_p5": float("nan"),
            "min_angle_p50": float("nan"),
            "aspect_p50": float("nan"),
            "aspect_p95": float("nan"),
            "degenerate_count": 0,
            "triangle_count": 0,
        }
    areas = _triangle_areas(vertices, faces)
    degenerate = areas < _DEGENERATE_AREA
    corners = vertices[faces]
    edge_lengths = np.stack(
        [np.linalg.norm(corners[:, (i + 1) % 3] - corners[:, i], axis=1) for i in range(3)],
        axis=1,
    )

    valid = ~degenerate
    if not np.any(valid):
        min_angle_p5 = min_angle_p50 = aspect_p50 = aspect_p95 = float("nan")
    else:
        angles = []
        for i in range(3):
            u = corners[valid, (i + 1) % 3] - corners[valid, i]
            v = corners[valid, (i + 2) % 3] - corners[valid, i]
            cosine = np.einsum("ki,ki->k", u, v) / (
                np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
            )
            angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        min_angles = np.minimum(np.minimum(angles[0], angles[1]), angles[2])
        aspect = edge_lengths[valid].max(axis=1) ** 2 / (2.0 * areas[valid])
        min_angle_p5 = float(np.percentile(min_angles, 5))
        min_angle_p50 = float(np.percentile(min_angles, 50))
        aspect_p50 = float(np.percentile(aspect, 50))
        aspect_p95 = float(np.percentile(aspect, 95))
    return {
        "min_angle_p5": min_angle_p5,
        "min_angle_p50": min_angle_p50,
        "aspect_p50": aspect_p50,
        "aspect_p95": aspect_p95,
        "degenerate_count": int(np.count_nonzero(degenerate)),
        "triangle_count": int(faces.shape[0]),
    }


def mesh_report(
    sdf: Callable[[Array], Array],
    mesh: Mesh,
    *,
    samples: int = 2048,
    pairs: int = 4096,
    seed: int = 0,
) -> dict[str, Any]:
    """Bundle every diagnostic into one report.

    Runs :func:`surface_deviation`, :func:`self_intersections`, and
    :func:`triangle_quality`, and adds watertightness (every undirected
    edge shared by exactly two triangles) plus the Euler characteristic
    ``V - E + F`` over the vertices actually referenced by the faces
    (2 for a closed sphere-like surface).

    Args:
        sdf: Implicit field the mesh was extracted from.
        mesh: The extracted mesh.
        samples: Forwarded to :func:`surface_deviation`.
        pairs: Forwarded to :func:`self_intersections`.
        seed: Forwarded to both sampled diagnostics.

    Returns:
        Dict with keys ``surface_deviation``, ``self_intersections``,
        ``triangle_quality``, ``watertight``, and ``euler_characteristic``.
    """
    _, faces = _mesh_arrays(mesh)
    if faces.shape[0] == 0:
        watertight = False
        euler = 0
    else:
        _, counts = _undirected_edge_counts(faces)
        watertight = bool(np.all(counts == 2))
        vertex_count = int(np.unique(faces).size)
        euler = vertex_count - int(counts.size) + int(faces.shape[0])
    return {
        "surface_deviation": surface_deviation(sdf, mesh, samples=samples, seed=seed),
        "self_intersections": self_intersections(mesh, pairs=pairs, seed=seed),
        "triangle_quality": triangle_quality(mesh),
        "watertight": watertight,
        "euler_characteristic": euler,
    }
