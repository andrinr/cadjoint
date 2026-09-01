"""Tests for cadjoint.meshing.diagnostics (deviation, intersections, quality)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.meshing.diagnostics import (
    mesh_report,
    self_intersections,
    surface_deviation,
    triangle_quality,
)
from cadjoint.meshing.dual_contouring import Mesh, extract_mesh
from cadjoint.meshing.edge_detection import GridSpec

SPHERE_GRID = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)


def sphere_sdf(p):
    return jnp.sqrt(jnp.sum(p * p)) - 1.0


@pytest.fixture(scope="module")
def sphere_mesh() -> Mesh:
    return extract_mesh(sphere_sdf, SPHERE_GRID)


def hand_mesh(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    """Build a minimal Mesh around explicit vertices and triangle faces."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    return Mesh(
        vertices=jnp.asarray(vertices),
        faces=faces,
        quads=np.empty((0, 4), dtype=np.int32),
        normals=jnp.zeros_like(jnp.asarray(vertices)),
        cells=np.zeros((vertices.shape[0], 3), dtype=np.int32),
    )


@pytest.fixture()
def crossing_mesh() -> Mesh:
    # A large triangle in the z = 0 plane, pierced through its interior by a
    # vertical triangle; the two share no vertex indices.
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.2, 0.0, -1.0],
            [0.2, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return hand_mesh(vertices, [[0, 1, 2], [3, 4, 5]])


class TestSurfaceDeviation:
    def test_sphere_deviation_small(self, sphere_mesh):
        report = surface_deviation(sphere_sdf, sphere_mesh)
        assert report["samples"] == 2048
        assert 0.0 < report["mean_abs"] <= report["max_abs"]
        assert report["max_abs"] < 5e-3

    def test_deterministic_and_seed_sensitive(self, sphere_mesh):
        first = surface_deviation(sphere_sdf, sphere_mesh, samples=256, seed=3)
        second = surface_deviation(sphere_sdf, sphere_mesh, samples=256, seed=3)
        other = surface_deviation(sphere_sdf, sphere_mesh, samples=256, seed=4)
        assert first == second
        assert first != other

    def test_rejects_bad_inputs(self, sphere_mesh):
        with pytest.raises(ValueError, match="samples"):
            surface_deviation(sphere_sdf, sphere_mesh, samples=0)
        empty = hand_mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))
        with pytest.raises(ValueError, match="no faces"):
            surface_deviation(sphere_sdf, empty)


class TestSelfIntersections:
    def test_sphere_mesh_clean(self, sphere_mesh):
        report = self_intersections(sphere_mesh)
        assert report["count"] == 0
        assert report["pairs"].shape == (0, 2)
        assert report["tested"] > 0

    def test_crossing_pair_detected(self, crossing_mesh):
        report = self_intersections(crossing_mesh, pairs=64)
        assert report["count"] == 1
        np.testing.assert_array_equal(report["pairs"], [[0, 1]])

    def test_touching_triangles_not_reported(self):
        # Two triangles meeting exactly along the segment y = 0, z = 0
        # without sharing vertex indices: touching, not crossing.
        vertices = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mesh = hand_mesh(vertices, [[0, 1, 2], [3, 4, 5]])
        assert self_intersections(mesh, pairs=64)["count"] == 0

    def test_single_triangle_trivially_clean(self):
        mesh = hand_mesh(np.eye(3), [[0, 1, 2]])
        report = self_intersections(mesh, pairs=8)
        assert report["count"] == 0
        assert report["tested"] == 0
        assert report["pairs"].shape == (0, 2)

    def test_rejects_bad_pairs(self, sphere_mesh):
        with pytest.raises(ValueError, match="pairs"):
            self_intersections(sphere_mesh, pairs=0)


class TestTriangleQuality:
    def test_sphere_percentiles_sane(self, sphere_mesh):
        quality = triangle_quality(sphere_mesh)
        assert quality["triangle_count"] == sphere_mesh.faces.shape[0]
        assert quality["degenerate_count"] == 0
        # The minimum angle of any triangle is at most 60 degrees.
        assert 0.0 < quality["min_angle_p5"] <= quality["min_angle_p50"] <= 60.0
        # Aspect ratio is bounded below by the equilateral value 2/sqrt(3).
        equilateral = 2.0 / np.sqrt(3.0)
        assert equilateral <= quality["aspect_p50"] <= quality["aspect_p95"]

    def test_degenerate_triangle_counted(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 3.0, 0.0],
                [4.0, 4.0, 0.0],  # collinear: zero area
            ]
        )
        quality = triangle_quality(hand_mesh(vertices, [[0, 1, 2], [3, 4, 5]]))
        assert quality["degenerate_count"] == 1
        assert quality["triangle_count"] == 2
        # Statistics come from the surviving right triangle alone.
        assert quality["min_angle_p5"] == pytest.approx(45.0, abs=1e-9)
        assert quality["aspect_p50"] == pytest.approx(np.sqrt(2.0) ** 2 / (2.0 * 0.5))

    def test_empty_mesh(self):
        quality = triangle_quality(hand_mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)))
        assert quality["triangle_count"] == 0
        assert quality["degenerate_count"] == 0
        assert np.isnan(quality["min_angle_p5"])


class TestMeshReport:
    def test_sphere_report(self, sphere_mesh):
        report = mesh_report(sphere_sdf, sphere_mesh, samples=512, pairs=1024, seed=1)
        assert report["watertight"] is True
        assert report["euler_characteristic"] == 2
        assert report["surface_deviation"]["samples"] == 512
        assert report["surface_deviation"]["max_abs"] < 5e-3
        assert report["self_intersections"]["count"] == 0
        assert report["triangle_quality"]["degenerate_count"] == 0

    def test_open_mesh_not_watertight(self, crossing_mesh):
        report = mesh_report(lambda p: jnp.sum(p * p), crossing_mesh, samples=16, pairs=16)
        assert report["watertight"] is False
        assert report["self_intersections"]["count"] == 1
