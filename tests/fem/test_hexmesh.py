"""Tests for cadjoint.fem.hexmesh (SDF voxelization + boundary snapping)."""

from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint.fem.hexmesh import (
    GridSpec,
    corner_tet_volumes,
    recompute_points,
    sdf_to_hex_mesh,
    select_faces,
)
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.primitives import Box, Sphere


def _box(half=0.5):
    return Box(Vector([half, half, half], free=True, name="size"))


def _aligned_grid():
    """Cell size 0.125; box faces at +-0.5 land exactly on lattice planes."""
    return GridSpec.from_bounds((-0.75, -0.75, -0.75), (1.5, 1.5, 1.5), 12)


def _offset_grid():
    """No lattice plane coincides with the box surface, so snapping must work."""
    return GridSpec.from_bounds((-0.79, -0.77, -0.81), (1.6, 1.6, 1.6), 13)


def _edge_face_counts(faces: np.ndarray) -> Counter:
    counts: Counter = Counter()
    for quad in faces:
        for a, b in zip(quad, np.roll(quad, -1)):
            counts[(min(a, b), max(a, b))] += 1
    return counts


class TestBoxMesh:
    def test_expected_cell_count(self):
        mesh = sdf_to_hex_mesh(_box(), _aligned_grid())
        # Box volume 1.0, cell volume 0.125^3, centers strictly inside: 8^3.
        assert mesh.num_cells == 512
        assert mesh.num_points == 9**3

    def test_boundary_faces_on_surface_after_snapping(self):
        box = _box()
        mesh = sdf_to_hex_mesh(box, _offset_grid())
        faces = mesh.all_boundary_faces()
        values = np.abs(np.asarray(box(jnp.asarray(faces.centers))))
        spacing = max(mesh.grid.spacing)
        # Flat-face quads land exactly on the surface.  Quads chamfering an
        # edge of the box have all four vertices on the surface but a center
        # up to ~h/(2*sqrt(2)) ~= 0.35h from the sharp edge — inherent to
        # hexahedral chamfering, not a snapping failure.
        assert values.max() < 0.4 * spacing
        assert (values < 1e-6).mean() > 0.75

    def test_no_inverted_elements(self):
        mesh = sdf_to_hex_mesh(_box(), _offset_grid())
        volumes = corner_tet_volumes(mesh.points, mesh.cells)
        assert volumes.min() > 0.0

    def test_boundary_groups_cover_six_sides(self):
        mesh = sdf_to_hex_mesh(_box(), _aligned_grid())
        assert set(mesh.boundary_faces) == {"+x", "-x", "+y", "-y", "+z", "-z"}
        # 8x8 lattice faces per side of the aligned box.
        for group in mesh.boundary_faces.values():
            assert group.nodes.shape == (64, 4)

    def test_select_faces_by_center_and_normal(self):
        mesh = sdf_to_hex_mesh(_box(), _aligned_grid())
        right = select_faces(mesh, lambda center: center[0] > 0.49)
        assert right.nodes.shape[0] == 64
        assert np.allclose(right.centers[:, 0], 0.5)
        outward = select_faces(mesh, lambda center, normal: normal[0] > 0.9)
        assert outward.nodes.shape[0] == 64
        # Outward normals of the +x side point along +x.
        assert np.allclose(outward.normals, [1.0, 0.0, 0.0], atol=1e-12)


class TestSphereMesh:
    def test_snapped_vertices_on_surface(self):
        sphere = Sphere(Scalar(value=0.6, free=True, name="radius"))
        mesh = sdf_to_hex_mesh(sphere, _aligned_grid())
        assert mesh.snap_mask.any()
        snapped = mesh.points[mesh.snap_mask]
        values = np.abs(np.asarray(sphere(jnp.asarray(snapped))))
        assert values.max() < 1e-3
        # The guard may hold back a few vertices, but most must snap.
        boundary_vertices = np.unique(mesh.all_boundary_faces().nodes)
        assert mesh.snap_mask.sum() > 0.8 * boundary_vertices.size

    def test_no_inverted_elements(self):
        sphere = Sphere(Scalar(value=0.6, free=True, name="radius"))
        mesh = sdf_to_hex_mesh(sphere, _aligned_grid())
        assert corner_tet_volumes(mesh.points, mesh.cells).min() > 0.0

    def test_watertight_boundary(self):
        sphere = Sphere(Scalar(value=0.6, free=True, name="radius"))
        mesh = sdf_to_hex_mesh(sphere, _aligned_grid())
        counts = _edge_face_counts(mesh.all_boundary_faces().nodes)
        assert set(counts.values()) == {2}

    def test_expected_cell_count(self):
        sphere = Sphere(Scalar(value=0.6, free=True, name="radius"))
        mesh = sdf_to_hex_mesh(sphere, _aligned_grid())
        expected = 4.0 / 3.0 * np.pi * 0.6**3 / 0.125**3
        assert abs(mesh.num_cells - expected) < 0.15 * expected


class TestFrozenTopologyRecompute:
    def test_recompute_matches_extraction(self):
        box = _box()
        mesh = sdf_to_hex_mesh(box, _offset_grid())
        rebuilt = np.asarray(recompute_points(box, mesh))
        moved = np.linalg.norm(rebuilt - mesh.points, axis=-1)
        assert moved.max() < 1e-6

    def test_gradient_reaches_design_parameter(self):
        mesh = sdf_to_hex_mesh(_box(), _offset_grid())

        def surface_spread(half):
            def sdf(p):
                return Box.sdf(p, jnp.array([half, half, half]))

            points = recompute_points(sdf, mesh)
            return jnp.sum(points**2)

        gradient = jax.grad(surface_spread)(0.5)
        assert np.isfinite(gradient)
        # Growing the box moves snapped vertices outward: strictly positive.
        assert gradient > 0.0
        eps = 1e-4
        fd = (surface_spread(0.5 + eps) - surface_spread(0.5 - eps)) / (2 * eps)
        assert np.isclose(gradient, fd, rtol=5e-3)
