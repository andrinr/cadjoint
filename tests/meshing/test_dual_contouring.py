"""Tests for cadjoint.meshing.dual_contouring (QEF vertices + dual connectivity)."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.meshing.dual_contouring import extract_mesh, qef_vertices
from cadjoint.meshing.edge_detection import (
    GridSpec,
    edge_hermite_data,
    find_crossing_edges,
    sample_grid,
)
from cadjoint.meshing.features import cell_edge_incidence
from cadjoint.sdf.primitives import Box


def sphere_sdf(radius):
    def sdf(p):
        return jnp.sqrt(jnp.sum(p * p)) - radius

    return sdf


def box_sdf(size):
    return lambda p: Box.sdf(p, size)


SPHERE_GRID = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)
BOX_GRID = GridSpec.from_bounds((-0.75, -0.75, -0.75), (1.5, 1.5, 1.5), 15)
BOX_SIZE = jnp.array([0.5, 0.5, 0.5])


def undirected_edge_counts(faces: np.ndarray) -> np.ndarray:
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    return counts


def euler_characteristic(vertex_count: int, faces: np.ndarray) -> int:
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edge_count = np.unique(np.sort(directed, axis=1), axis=0).shape[0]
    return vertex_count - edge_count + faces.shape[0]


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


class TestSphereMesh:
    def extract(self):
        return extract_mesh(sphere_sdf(1.0), SPHERE_GRID)

    def test_watertight_manifold(self):
        mesh = self.extract()
        counts = undirected_edge_counts(mesh.faces)
        # Closed manifold: every undirected edge is shared by exactly two triangles.
        np.testing.assert_array_equal(np.unique(counts), [2])
        assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == 2

    def test_outward_orientation_and_volume(self):
        mesh = self.extract()
        volume = signed_volume(np.asarray(mesh.vertices, dtype=np.float64), mesh.faces)
        assert volume > 0  # counterclockwise from outside
        np.testing.assert_allclose(volume, 4.0 / 3.0 * np.pi, rtol=2e-2)

    def test_vertices_near_surface(self):
        mesh = self.extract()
        residuals = np.abs(np.asarray(jax.vmap(sphere_sdf(1.0))(mesh.vertices)))
        # Tangent planes of a curved cell intersect slightly off-surface;
        # the error is bounded by the cell-size chord error.
        assert residuals.max() < 5e-3

    def test_vertices_stay_in_their_cells(self):
        mesh = self.extract()
        origin = np.asarray(SPHERE_GRID.origin)
        spacing = np.asarray(SPHERE_GRID.spacing)
        cell_min = origin + mesh.cells * spacing
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        assert np.all(vertices >= cell_min - 1e-6)
        assert np.all(vertices <= cell_min + spacing + 1e-6)

    def test_extraction_is_deterministic(self):
        first = self.extract()
        second = self.extract()
        np.testing.assert_array_equal(first.faces, second.faces)
        np.testing.assert_array_equal(first.quads, second.quads)


class TestBoxMesh:
    def extract(self):
        return extract_mesh(box_sdf(BOX_SIZE), BOX_GRID)

    def test_corners_are_reproduced(self):
        mesh = self.extract()
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        corners = np.array(
            [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]
        )
        for corner in corners:
            distance = np.min(np.linalg.norm(vertices - corner, axis=1))
            # Marching cubes at this spacing is ~0.05 off; the regularized QEF
            # keeps the bias at the regularization scale.
            assert distance < 1e-3

    def test_watertight_with_exact_volume(self):
        mesh = self.extract()
        np.testing.assert_array_equal(np.unique(undirected_edge_counts(mesh.faces)), [2])
        volume = signed_volume(np.asarray(mesh.vertices, dtype=np.float64), mesh.faces)
        np.testing.assert_allclose(volume, 1.0, rtol=5e-3)


class TestUnionMesh:
    def test_hard_csg_union_is_watertight(self):
        offset = jnp.array([0.8, 0.0, 0.0])

        def union(p):
            return jnp.minimum(
                jnp.sqrt(jnp.sum(p * p)) - 1.0,
                jnp.sqrt(jnp.sum((p - offset) ** 2)) - 1.0,
            )

        grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (3.4, 2.6, 2.6), (34, 26, 26))
        mesh = extract_mesh(union, grid)
        np.testing.assert_array_equal(np.unique(undirected_edge_counts(mesh.faces)), [2])
        assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == 2
        assert signed_volume(np.asarray(mesh.vertices, dtype=np.float64), mesh.faces) > 0


class TestDifferentiability:
    def test_sphere_radius_gradient_vs_finite_differences(self):
        edges = find_crossing_edges(sample_grid(sphere_sdf(1.0), SPHERE_GRID))
        incidence = cell_edge_incidence(edges, SPHERE_GRID)

        def loss(radius):
            hermite = edge_hermite_data(sphere_sdf(radius), SPHERE_GRID, edges)
            vertices, _ = qef_vertices(hermite, incidence, SPHERE_GRID)
            return jnp.sum(vertices**2)

        gradient = float(jax.grad(loss)(jnp.float32(1.0)))
        eps = 1e-3
        finite = (float(loss(jnp.float32(1.0 + eps))) - float(loss(jnp.float32(1.0 - eps)))) / (
            2 * eps
        )
        np.testing.assert_allclose(gradient, finite, rtol=1e-2)

    def test_box_corner_jacobian_is_identity(self):
        # The defining advantage over a normal-only implicit backward pass:
        # the corner vertex tracks all three half-extents with the full
        # identity Jacobian, including tangential sliding along the faces.
        # A normal-projected gradient would give the rank-1 matrix n n^T
        # with every entry near 1/3.
        edges = find_crossing_edges(sample_grid(box_sdf(BOX_SIZE), BOX_GRID))
        incidence = cell_edge_incidence(edges, BOX_GRID)
        hermite = edge_hermite_data(box_sdf(BOX_SIZE), BOX_GRID, edges)
        vertices, _ = qef_vertices(hermite, incidence, BOX_GRID)
        corner_row = int(
            np.argmin(np.linalg.norm(np.asarray(vertices) - np.array([0.5, 0.5, 0.5]), axis=1))
        )

        def corner_vertex(size):
            data = edge_hermite_data(box_sdf(size), BOX_GRID, edges)
            placed, _ = qef_vertices(data, incidence, BOX_GRID)
            return placed[corner_row]

        jacobian = np.asarray(jax.jacobian(corner_vertex)(BOX_SIZE))
        np.testing.assert_allclose(jacobian, np.eye(3), atol=5e-2)

    def test_qef_vertices_match_under_jit(self):
        edges = find_crossing_edges(sample_grid(sphere_sdf(1.0), SPHERE_GRID))
        incidence = cell_edge_incidence(edges, SPHERE_GRID)

        def vertices_of(radius):
            hermite = edge_hermite_data(sphere_sdf(radius), SPHERE_GRID, edges)
            return qef_vertices(hermite, incidence, SPHERE_GRID)[0]

        eager = vertices_of(jnp.float32(1.0))
        jitted = jax.jit(vertices_of)(jnp.float32(1.0))
        np.testing.assert_allclose(np.asarray(eager), np.asarray(jitted), atol=1e-6)


class TestEdgeCases:
    def test_open_mesh_warns_at_boundary(self):
        tight = GridSpec.from_bounds((-0.9, -0.9, -0.9), (1.8, 1.8, 1.8), 18)
        with pytest.warns(UserWarning, match="open"):
            mesh = extract_mesh(sphere_sdf(1.0), tight)
        assert mesh.faces.shape[0] > 0

    def test_empty_field_returns_empty_mesh(self):
        far = lambda p: jnp.sqrt(jnp.sum((p - 100.0) ** 2)) - 1.0  # noqa: E731
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mesh = extract_mesh(far, SPHERE_GRID)
        assert mesh.vertices.shape == (0, 3)
        assert mesh.faces.shape == (0, 3)
        assert mesh.quads.shape == (0, 4)

    def test_regularization_must_be_positive(self):
        edges = find_crossing_edges(sample_grid(sphere_sdf(1.0), SPHERE_GRID))
        incidence = cell_edge_incidence(edges, SPHERE_GRID)
        hermite = edge_hermite_data(sphere_sdf(1.0), SPHERE_GRID, edges)
        with pytest.raises(ValueError, match="regularization"):
            qef_vertices(hermite, incidence, SPHERE_GRID, regularization=0.0)
