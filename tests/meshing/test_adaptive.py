"""Tests for cadjoint.meshing.adaptive (octree-pruned surface-cell search)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.meshing.adaptive import sparse_crossing_edges, surface_cells
from cadjoint.meshing.dual_contouring import extract_mesh
from cadjoint.meshing.edge_detection import GridSpec, find_crossing_edges, sample_grid
from cadjoint.sdf.primitives import Box


def sphere_sdf(p):
    return jnp.sqrt(jnp.sum(p * p)) - 1.0


def box_sdf(p):
    return Box.sdf(p, jnp.array([0.4, 0.5, 0.6]))


def union_sdf(p):
    return jnp.minimum(
        sphere_sdf(p), jnp.sqrt(jnp.sum((p - jnp.array([0.8, 0.0, 0.0])) ** 2)) - 1.0
    )


def plane_sdf(p):
    return p[2] - 0.1


CASES = [
    ("sphere", sphere_sdf, GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)),
    ("box", box_sdf, GridSpec.from_bounds((-0.85, -0.95, -1.05), (1.7, 1.9, 2.1), 17)),
    ("union", union_sdf, GridSpec.from_bounds((-1.3, -1.3, -1.3), (3.4, 2.6, 2.6), (34, 26, 26))),
    ("plane", plane_sdf, GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), 8)),
]


class TestEquivalenceWithDense:
    @pytest.mark.parametrize("name,sdf,grid", CASES, ids=[case[0] for case in CASES])
    def test_identical_crossing_edges(self, name, sdf, grid):
        del name
        dense = find_crossing_edges(sample_grid(sdf, grid))
        sparse = sparse_crossing_edges(sdf, grid)
        np.testing.assert_array_equal(dense.axis, sparse.axis)
        np.testing.assert_array_equal(dense.index, sparse.index)
        np.testing.assert_array_equal(dense.start_inside, sparse.start_inside)

    def test_identical_mesh_through_extract(self):
        grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)
        dense = extract_mesh(sphere_sdf, grid)
        sparse = extract_mesh(sphere_sdf, grid, lipschitz=1.0)
        np.testing.assert_array_equal(dense.faces, sparse.faces)
        np.testing.assert_array_equal(dense.quads, sparse.quads)
        np.testing.assert_allclose(
            np.asarray(dense.vertices), np.asarray(sparse.vertices), atol=1e-7
        )

    def test_overestimated_bound_stays_exact(self):
        # A too-large Lipschitz bound only weakens pruning, never the result.
        grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)
        dense = find_crossing_edges(sample_grid(sphere_sdf, grid))
        sparse = sparse_crossing_edges(sphere_sdf, grid, lipschitz=4.0)
        np.testing.assert_array_equal(dense.index, sparse.index)

    def test_sub_lipschitz_field_stays_exact(self):
        half = lambda p: 0.5 * sphere_sdf(p)  # noqa: E731 - Lipschitz 0.5 <= claimed 1.0
        grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)
        dense = find_crossing_edges(sample_grid(half, grid))
        sparse = sparse_crossing_edges(half, grid, lipschitz=1.0)
        np.testing.assert_array_equal(dense.index, sparse.index)


class TestPruning:
    def test_evaluation_count_scales_with_surface(self):
        grid = GridSpec.from_bounds((-2.0, -2.0, -2.0), (4.0, 4.0, 4.0), 64)
        stats: dict = {}
        sparse_crossing_edges(sphere_sdf, grid, stats=stats)
        dense_evaluations = np.prod([count + 1 for count in grid.cells])
        assert stats["evaluations"] < 0.15 * dense_evaluations
        assert stats["candidate_cells"] > 0

    def test_underclaimed_bound_can_only_lose_edges(self):
        # Contract documentation: claiming Lipschitz 1 for a 2-Lipschitz field
        # may prune surface cells; the correct bound restores exactness.
        steep = lambda p: 2.0 * sphere_sdf(p)  # noqa: E731
        grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)
        dense = find_crossing_edges(sample_grid(steep, grid))
        underclaimed = sparse_crossing_edges(steep, grid, lipschitz=1.0)
        correct = sparse_crossing_edges(steep, grid, lipschitz=2.0)
        assert underclaimed.count <= dense.count
        np.testing.assert_array_equal(dense.index, correct.index)


class TestEdgeCases:
    def test_empty_field(self):
        far = lambda p: jnp.sqrt(jnp.sum((p - 100.0) ** 2)) - 1.0  # noqa: E731
        grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)
        assert surface_cells(far, grid).shape == (0, 3)
        edges = sparse_crossing_edges(far, grid)
        assert edges.count == 0
        assert edges.index.shape == (0, 3)

    def test_lipschitz_must_be_positive(self):
        grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)
        with pytest.raises(ValueError, match="lipschitz"):
            sparse_crossing_edges(sphere_sdf, grid, lipschitz=0.0)
