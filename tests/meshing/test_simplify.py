"""Tests for post-extraction, topology-safe mesh simplification."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.meshing.diagnostics import surface_deviation
from cadjoint.meshing.dual_contouring import Mesh, extract_mesh
from cadjoint.meshing.edge_detection import GridSpec, find_crossing_edges, sample_grid
from cadjoint.meshing.features import (
    active_branches,
    detect_branch_changes,
    manifold_cell_incidence,
)
from cadjoint.meshing.simplify import simplify_mesh
from cadjoint.sdf.primitives import Box

BOX_SIZE = np.array([0.4, 0.5, 0.6])
BOX_CORNERS = (
    np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * BOX_SIZE
)


def sphere_sdf(p):
    return jnp.sqrt(jnp.sum(p * p)) - 1.0


def box_sdf(p):
    return Box.sdf(p, jnp.asarray(BOX_SIZE, dtype=jnp.float32))


def sphere_a(p):
    return jnp.sqrt(jnp.sum((p - jnp.array([-0.5, 0.0, 0.0])) ** 2)) - 1.0


def sphere_b(p):
    return jnp.sqrt(jnp.sum((p - jnp.array([0.5, 0.0, 0.0])) ** 2)) - 1.0


def union_sdf(p):
    return jnp.minimum(sphere_a(p), sphere_b(p))


def sphere_mesh(resolution: int = 32):
    grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), resolution)
    return extract_mesh(sphere_sdf, grid), grid


def box_mesh(resolution: int = 20):
    grid = GridSpec.from_bounds((-0.85, -0.95, -1.05), (1.7, 1.9, 2.1), resolution)
    return extract_mesh(box_sdf, grid), grid


def union_mesh(resolution: int = 32):
    grid = GridSpec.from_bounds((-1.8, -1.3, -1.3), (3.6, 2.6, 2.6), resolution)
    return extract_mesh(union_sdf, grid), grid


def edge_counts(faces: np.ndarray) -> np.ndarray:
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    return counts


def is_watertight(faces: np.ndarray) -> bool:
    return faces.shape[0] > 0 and bool(np.all(edge_counts(faces) == 2))


def euler_characteristic(faces: np.ndarray) -> int:
    return int(np.unique(faces).size) - int(edge_counts(faces).size) + int(faces.shape[0])


def preserved_exactly(original: Mesh, simplified: Mesh, rows) -> bool:
    """Whether every original vertex row in ``rows`` survives bitwise."""
    source = np.asarray(original.vertices)
    result = np.asarray(simplified.vertices)
    return all(bool(np.any(np.all(result == source[row], axis=1))) for row in rows)


class TestSphere:
    def test_reduces_triangles_within_deviation_bound(self):
        mesh, _ = sphere_mesh()
        error = 0.02
        simplified = simplify_mesh(mesh, sphere_sdf, error=error)

        before = mesh.faces.shape[0]
        after = simplified.faces.shape[0]
        assert after < 0.6 * before  # well beyond the required 40% drop
        assert is_watertight(simplified.faces)
        assert euler_characteristic(simplified.faces) == 2
        assert surface_deviation(sphere_sdf, simplified)["max_abs"] <= 2 * error

    def test_deterministic(self):
        mesh, _ = sphere_mesh(24)
        first = simplify_mesh(mesh, sphere_sdf, error=0.02)
        second = simplify_mesh(mesh, sphere_sdf, error=0.02)
        assert np.array_equal(first.faces, second.faces)
        assert np.array_equal(np.asarray(first.vertices), np.asarray(second.vertices))
        assert np.array_equal(first.cells, second.cells)

    def test_feature_mask_pins_rows_exactly(self):
        mesh, _ = sphere_mesh(24)
        pinned = np.zeros(np.asarray(mesh.vertices).shape[0], dtype=bool)
        pinned[::7] = True
        simplified = simplify_mesh(mesh, sphere_sdf, error=0.05, feature_mask=pinned)
        assert preserved_exactly(mesh, simplified, np.flatnonzero(pinned))

    def test_tighter_bound_collapses_less(self):
        mesh, _ = sphere_mesh(24)
        loose = simplify_mesh(mesh, sphere_sdf, error=0.05)
        tight = simplify_mesh(mesh, sphere_sdf, error=0.005)
        assert loose.faces.shape[0] < tight.faces.shape[0] <= mesh.faces.shape[0]
        assert surface_deviation(sphere_sdf, tight)["max_abs"] <= 2 * 0.005


class TestBox:
    def test_planar_faces_collapse_and_corners_survive(self):
        mesh, _ = box_mesh()
        simplified = simplify_mesh(mesh, box_sdf, error=1e-3)

        assert simplified.faces.shape[0] < 0.35 * mesh.faces.shape[0]
        assert is_watertight(simplified.faces)
        assert euler_characteristic(simplified.faces) == 2

        # All eight corners preserved exactly: the original vertex nearest
        # each analytic corner survives bitwise (half-edge collapses never
        # move a surviving vertex).
        source = np.asarray(mesh.vertices, dtype=np.float64)
        rows = [int(np.argmin(np.linalg.norm(source - c, axis=1))) for c in BOX_CORNERS]
        assert preserved_exactly(mesh, simplified, rows)
        result = np.asarray(simplified.vertices, dtype=np.float64)
        for corner in BOX_CORNERS:
            assert float(np.min(np.linalg.norm(result - corner, axis=1))) < 1e-6

    def test_all_crease_vertices_survive(self):
        mesh, _ = box_mesh(16)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        # Endpoints of mesh edges whose dihedral angle exceeds 30 degrees.
        normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        owner = np.tile(np.arange(faces.shape[0]), 3)
        edges, inverse = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        pair = order.reshape(-1)  # watertight: exactly two occurrences per edge
        first, second = pair[0::2], pair[1::2]
        cosine = np.einsum("ki,ki->k", unit[owner[first]], unit[owner[second]])
        crease_rows = np.unique(edges[cosine < np.cos(np.radians(30.0))])
        assert crease_rows.size > 0

        simplified = simplify_mesh(mesh, box_sdf, error=1e-3)
        assert preserved_exactly(mesh, simplified, crease_rows)


class TestCsgUnion:
    def test_seam_preserved_by_dihedral_signal(self):
        mesh, _ = union_mesh()
        simplified = simplify_mesh(mesh, union_sdf, error=0.02)

        assert simplified.faces.shape[0] < mesh.faces.shape[0]
        assert is_watertight(simplified.faces)
        assert euler_characteristic(simplified.faces) == 2

        # Vertices on both spheres at once lie on the seam circle; sharp
        # extraction places them there and they must survive bitwise.
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        on_a = np.abs(np.linalg.norm(vertices - [-0.5, 0, 0], axis=1) - 1.0) < 5e-3
        on_b = np.abs(np.linalg.norm(vertices - [0.5, 0, 0], axis=1) - 1.0) < 5e-3
        seam_rows = np.flatnonzero(on_a & on_b)
        assert seam_rows.size > 20
        assert preserved_exactly(mesh, simplified, seam_rows)

    def test_seam_preserved_by_branch_change_mask(self):
        mesh, grid = union_mesh(24)
        values = sample_grid(union_sdf, grid)
        edges = find_crossing_edges(values)
        incidence = manifold_cell_incidence(edges, grid, np.asarray(values < 0.0))
        assert incidence.count == np.asarray(mesh.vertices).shape[0]

        from cadjoint.meshing.edge_detection import edge_hermite_data

        hermite = edge_hermite_data(union_sdf, grid, edges)
        branches = active_branches([sphere_a, sphere_b], hermite.points)
        seam_mask = detect_branch_changes(branches, incidence)
        assert np.any(seam_mask)

        simplified = simplify_mesh(mesh, union_sdf, error=0.02, feature_mask=seam_mask)
        assert preserved_exactly(mesh, simplified, np.flatnonzero(seam_mask))
        assert is_watertight(simplified.faces)


class TestValidationAndEdgeCases:
    def test_rejects_bad_arguments(self):
        mesh, _ = sphere_mesh(16)
        with pytest.raises(ValueError, match="error"):
            simplify_mesh(mesh, sphere_sdf, error=0.0)
        with pytest.raises(ValueError, match="feature_angle"):
            simplify_mesh(mesh, sphere_sdf, error=0.01, feature_angle=180.0)
        with pytest.raises(ValueError, match="max_passes"):
            simplify_mesh(mesh, sphere_sdf, error=0.01, max_passes=0)
        with pytest.raises(ValueError, match="feature_mask"):
            simplify_mesh(mesh, sphere_sdf, error=0.01, feature_mask=np.zeros(3, dtype=bool))

    def test_empty_mesh_passes_through(self):
        grid = GridSpec.from_bounds((2.0, 2.0, 2.0), (1.0, 1.0, 1.0), 4)
        mesh = extract_mesh(sphere_sdf, grid)
        assert mesh.faces.shape[0] == 0
        simplified = simplify_mesh(mesh, sphere_sdf, error=0.01)
        assert simplified.faces.shape[0] == 0

    def test_vertices_are_subset_of_input(self):
        mesh, _ = sphere_mesh(24)
        simplified = simplify_mesh(mesh, sphere_sdf, error=0.02)
        source = {tuple(row) for row in np.asarray(mesh.vertices).tolist()}
        for row in np.asarray(simplified.vertices).tolist():
            assert tuple(row) in source
