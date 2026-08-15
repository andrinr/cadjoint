"""Tests for sharp-feature detection in :mod:`jaxcad.meshing.features`."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcad.meshing.edge_detection import (
    GridSpec,
    detect_edges,
    edge_hermite_data,
    find_crossing_edges,
)
from jaxcad.meshing.features import (
    CORNER,
    CREASE,
    FACE,
    active_branches,
    cell_edge_incidence,
    classify_feature_cells,
    detect_branch_changes,
)
from jaxcad.sdf.primitives import Box, Sphere
from jaxcad.sdf.primitives.cylinder import Cylinder


def _cell_centers(incidence, grid: GridSpec) -> np.ndarray:
    """World-space centers of the active cells."""
    origin = np.asarray(grid.origin)
    spacing = np.asarray(grid.spacing)
    return origin + (incidence.cells.astype(np.float64) + 0.5) * spacing


@pytest.fixture(scope="module")
def box_pipeline():
    """Cube with half-extents 0.5 on a grid whose corners land mid-cell."""

    def sdf(p):
        return Box.sdf(p, jnp.array([0.5, 0.5, 0.5]))

    grid = GridSpec.from_bounds((-0.75,) * 3, (1.5,) * 3, 15)
    edges, hermite = detect_edges(sdf, grid)
    incidence = cell_edge_incidence(edges, grid)
    features = classify_feature_cells(hermite, incidence)
    return grid, edges, hermite, incidence, features


@pytest.fixture(scope="module")
def sphere_pipeline():
    """Unit sphere on the verified baseline grid."""

    def sdf(p):
        return Sphere.sdf(p, 1.0)

    grid = GridSpec.from_bounds((-1.3,) * 3, (2.6,) * 3, 26)
    edges, hermite = detect_edges(sdf, grid)
    incidence = cell_edge_incidence(edges, grid)
    features = classify_feature_cells(hermite, incidence)
    return grid, edges, hermite, incidence, features


class TestBoxClassification:
    def test_exactly_the_eight_corner_cells_are_corners(self, box_pipeline):
        _, _, _, incidence, features = box_pipeline
        corner_cells = {tuple(cell) for cell in incidence.cells[features.classes == CORNER]}
        expected = set(itertools.product((2, 12), repeat=3))
        assert corner_cells == expected
        assert int(np.sum(features.classes == CORNER)) == 8

    def test_edge_midspan_cell_is_crease(self, box_pipeline):
        _, _, _, incidence, features = box_pipeline
        lookup = {tuple(cell): row for row, cell in enumerate(incidence.cells)}
        assert features.classes[lookup[(12, 12, 7)]] == CREASE

    def test_face_center_cell_is_face(self, box_pipeline):
        _, _, _, incidence, features = box_pipeline
        lookup = {tuple(cell): row for row, cell in enumerate(incidence.cells)}
        assert features.classes[lookup[(12, 7, 7)]] == FACE


class TestSphereClassification:
    def test_every_active_cell_is_face(self, sphere_pipeline):
        _, _, _, _, features = sphere_pipeline
        np.testing.assert_array_equal(features.classes, FACE)

    def test_crease_measure_stays_small(self, sphere_pipeline):
        _, _, _, _, features = sphere_pipeline
        assert float(jnp.max(features.crease_measure)) < 0.15


class TestCylinderClassification:
    def test_rim_creases_without_corners(self):
        def sdf(p):
            return Cylinder.sdf(p, 0.5, 0.5)

        grid = GridSpec.from_bounds((-0.75,) * 3, (1.5,) * 3, 15)
        edges, hermite = detect_edges(sdf, grid)
        incidence = cell_edge_incidence(edges, grid)
        features = classify_feature_cells(hermite, incidence)

        assert int(np.sum(features.classes == CREASE)) > 0
        assert int(np.sum(features.classes == CORNER)) == 0

        # The sharp rims sit at z = +/-0.5, i.e. mid-cell in lattice layers 2
        # and 12, at radial distance 0.5 from the axis.
        crease_centers = _cell_centers(incidence, grid)[features.classes == CREASE]
        crease_layers = incidence.cells[features.classes == CREASE][:, 2]
        assert set(np.unique(crease_layers)) <= {2, 12}
        radial = np.hypot(crease_centers[:, 0], crease_centers[:, 1])
        np.testing.assert_array_less(radial, 0.66)
        np.testing.assert_array_less(0.39, radial)


class TestDifferentiability:
    def test_crease_measure_gradient_wrt_box_size_is_finite(self):
        # Regression test: flat-face cells of a box have covariance
        # eigenvalues that are exactly zero, and an unguarded
        # sqrt(clip(eigenvalues)) NaN'd the reverse-mode gradient of every
        # cell through the shared normals array.  classify_feature_cells now
        # double-where-guards the sqrt so degenerate directions carry a zero
        # subgradient.
        grid = GridSpec.from_bounds((-0.75,) * 3, (1.5,) * 3, 15)
        size = jnp.array([0.34, 0.44, 0.54])
        edges, _ = detect_edges(lambda p: Box.sdf(p, size), grid)
        incidence = cell_edge_incidence(edges, grid)

        def summed_crease(size_vector):
            hermite = edge_hermite_data(lambda p: Box.sdf(p, size_vector), grid, edges)
            features = classify_feature_cells(hermite, incidence)
            return jnp.sum(features.crease_measure)

        gradient = jax.grad(summed_crease)(size)
        assert gradient.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(gradient)))

    def test_crease_measure_gradient_wrt_sphere_radius_is_finite(self):
        # Companion to the box case: with strictly positive singular values
        # (curved surface, no degenerate cells) the same pipeline
        # differentiates cleanly, isolating the sqrt-at-zero failure above.
        grid = GridSpec.from_bounds((-1.3,) * 3, (2.6,) * 3, 26)
        radius = jnp.array(1.0)
        edges, _ = detect_edges(lambda p: Sphere.sdf(p, radius), grid)
        incidence = cell_edge_incidence(edges, grid)

        def summed_crease(r):
            hermite = edge_hermite_data(lambda p: Sphere.sdf(p, r), grid, edges)
            features = classify_feature_cells(hermite, incidence)
            return jnp.sum(features.crease_measure)

        gradient = jax.grad(summed_crease)(radius)
        assert bool(jnp.isfinite(gradient))


class TestIncidenceStructure:
    def test_each_edge_touches_at_most_four_cells(self, box_pipeline):
        _, edges, _, incidence, _ = box_pipeline
        valid_ids = incidence.edge_ids[incidence.edge_ids >= 0]
        appearances = np.bincount(valid_ids, minlength=edges.count)
        np.testing.assert_array_less(appearances, 5)

    def test_counts_match_padding(self, box_pipeline):
        _, _, _, incidence, _ = box_pipeline
        columns = np.arange(12)[None, :]
        valid = incidence.edge_ids >= 0
        np.testing.assert_array_equal(np.sum(valid, axis=1), incidence.counts)
        # Valid entries are left-packed: exactly the first ``counts`` columns.
        np.testing.assert_array_equal(valid, columns < incidence.counts[:, None])

    def test_pairs_are_geometrically_adjacent(self, box_pipeline):
        _, edges, _, incidence, _ = box_pipeline
        rows, cols = np.nonzero(incidence.edge_ids >= 0)
        cells = incidence.cells[rows]
        edge_ids = incidence.edge_ids[rows, cols]
        axes = edges.axis[edge_ids].astype(np.int64)
        starts = edges.index[edge_ids]

        along = np.take_along_axis(cells, axes[:, None], axis=1)[:, 0]
        along_edge = np.take_along_axis(starts, axes[:, None], axis=1)[:, 0]
        np.testing.assert_array_equal(along, along_edge)

        transverse = np.arange(3)[None, :] != axes[:, None]
        delta = (starts - cells)[transverse].reshape((-1, 2))
        assert np.all((delta == 0) | (delta == 1))

    def test_cells_lie_in_grid_bounds(self, box_pipeline):
        grid, _, _, incidence, _ = box_pipeline
        assert np.all(incidence.cells >= 0)
        assert np.all(incidence.cells < np.asarray(grid.cells))


class TestBranchChanges:
    @staticmethod
    def _fields():
        def left(p):
            return Sphere.sdf(p, 1.0)

        def right(p):
            return Sphere.sdf(p - jnp.array([0.8, 0.0, 0.0]), 1.0)

        return left, right

    @staticmethod
    def _seam_grid() -> GridSpec:
        return GridSpec.from_bounds((-1.3, -1.3, -1.3), (3.4, 2.6, 2.6), (34, 26, 26))

    def _seam_cell_x_centers(self, combined, mode: str) -> np.ndarray:
        left, right = self._fields()
        grid = self._seam_grid()
        edges, hermite = detect_edges(combined, grid)
        incidence = cell_edge_incidence(edges, grid)
        branches = active_branches([left, right], hermite.points, mode=mode)
        seam = detect_branch_changes(branches, incidence)
        return _cell_centers(incidence, grid)[seam][:, 0]

    def test_union_seam_lies_on_the_analytic_plane(self):
        left, right = self._fields()
        centers_x = self._seam_cell_x_centers(lambda p: jnp.minimum(left(p), right(p)), "min")
        assert centers_x.size > 0
        spacing = self._seam_grid().spacing[0]
        np.testing.assert_array_less(np.abs(centers_x - 0.4), spacing + 1e-9)

    def test_intersection_seam_lies_on_the_analytic_plane(self):
        left, right = self._fields()
        centers_x = self._seam_cell_x_centers(lambda p: jnp.maximum(left(p), right(p)), "max")
        assert centers_x.size > 0
        spacing = self._seam_grid().spacing[0]
        np.testing.assert_array_less(np.abs(centers_x - 0.4), spacing + 1e-9)

    def test_single_branch_surface_has_no_seam(self):
        # A sphere of radius 0.2 nested inside the unit sphere: the union
        # field equals the outer sphere everywhere (r - 1 <= r - 0.2), so the
        # outer child wins on every surface sample and no seam cells appear.
        # Two genuinely different fields, unlike the degenerate
        # identical-children control where argmin ties are broken to 0.
        def outer(p):
            return Sphere.sdf(p, 1.0)

        def inner(p):
            return Sphere.sdf(p, 0.2)

        grid = GridSpec.from_bounds((-1.3,) * 3, (2.6,) * 3, 26)
        edges, hermite = detect_edges(lambda p: jnp.minimum(outer(p), inner(p)), grid)
        incidence = cell_edge_incidence(edges, grid)
        branches = active_branches([outer, inner], hermite.points, mode="min")
        np.testing.assert_array_equal(branches, 0)
        seam = detect_branch_changes(branches, incidence)
        assert int(np.sum(seam)) == 0


class TestEmptyEdgeSet:
    def test_incidence_and_classification_handle_no_crossings(self):
        def sdf(p):
            return Sphere.sdf(p, 1.0)

        # Grid entirely outside the sphere: all lattice values positive.
        grid = GridSpec.from_bounds((2.0, 2.0, 2.0), (1.0, 1.0, 1.0), 2)
        values = np.full(grid.lattice_shape, 5.0)
        edges = find_crossing_edges(values)
        assert edges.count == 0

        incidence = cell_edge_incidence(edges, grid)
        assert incidence.count == 0
        assert incidence.edge_ids.shape == (0, 12)
        assert incidence.counts.shape == (0,)

        hermite = edge_hermite_data(sdf, grid, edges)
        features = classify_feature_cells(hermite, incidence)
        assert features.singular_values.shape == (0, 3)
        assert features.crease_measure.shape == (0,)
        assert features.corner_measure.shape == (0,)
        assert features.classes.shape == (0,)

        seam = detect_branch_changes(np.empty((0,), dtype=np.int32), incidence)
        assert seam.shape == (0,)


class TestValidation:
    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.5, 1.5])
    def test_crease_threshold_outside_unit_interval_raises(self, sphere_pipeline, threshold):
        _, _, hermite, incidence, _ = sphere_pipeline
        with pytest.raises(ValueError, match="crease_threshold"):
            classify_feature_cells(hermite, incidence, crease_threshold=threshold)

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.5, 1.5])
    def test_corner_threshold_outside_unit_interval_raises(self, sphere_pipeline, threshold):
        _, _, hermite, incidence, _ = sphere_pipeline
        with pytest.raises(ValueError, match="corner_threshold"):
            classify_feature_cells(hermite, incidence, corner_threshold=threshold)

    def test_detect_branch_changes_rejects_non_1d_branches(self, sphere_pipeline):
        _, _, _, incidence, _ = sphere_pipeline
        with pytest.raises(ValueError, match="one-dimensional"):
            detect_branch_changes(np.zeros((4, 1), dtype=np.int32), incidence)

    def test_active_branches_rejects_fewer_than_two_children(self):
        def sdf(p):
            return Sphere.sdf(p, 1.0)

        with pytest.raises(ValueError, match="at least two"):
            active_branches([sdf], jnp.zeros((2, 3)))

    def test_active_branches_rejects_unknown_mode(self):
        def sdf(p):
            return Sphere.sdf(p, 1.0)

        def other(p):
            return Sphere.sdf(p, 2.0)

        with pytest.raises(ValueError, match="mode"):
            active_branches([sdf, other], jnp.zeros((2, 3)), mode="sum")


class TestFeatureCellLinks:
    def test_box_feature_chains_connect_corners(self):
        from jaxcad.meshing.features import feature_cell_links

        grid = GridSpec.from_bounds((-0.75,) * 3, (1.5,) * 3, 15)
        sdf = lambda p: Box.sdf(p, jnp.array([0.5, 0.5, 0.5]))  # noqa: E731
        edges, hermite = detect_edges(sdf, grid)
        incidence = cell_edge_incidence(edges, grid)
        features = classify_feature_cells(hermite, incidence)
        mask = features.classes != FACE
        links = feature_cell_links(mask, incidence, grid)

        assert links.shape[1] == 2
        assert links.shape[0] > 0
        # Links only connect feature cells, and each pair is a lattice
        # neighbor (Chebyshev distance 1).
        assert bool(np.all(mask[links.reshape(-1)]))
        gaps = np.abs(incidence.cells[links[:, 0]] - incidence.cells[links[:, 1]])
        assert int(gaps.max()) == 1
        # Every corner cell participates in the chain graph.
        corner_rows = np.flatnonzero(features.classes == CORNER)
        linked = set(links.reshape(-1).tolist())
        assert all(int(row) in linked for row in corner_rows)

    def test_empty_mask_yields_no_links(self):
        from jaxcad.meshing.features import feature_cell_links

        grid = GridSpec.from_bounds((-1.3,) * 3, (2.6,) * 3, 13)
        edges, _hermite = detect_edges(lambda p: jnp.linalg.norm(p) - 1.0, grid)
        incidence = cell_edge_incidence(edges, grid)
        links = feature_cell_links(np.zeros(incidence.count, dtype=bool), incidence, grid)
        assert links.shape == (0, 2)

    def test_junction_mask_drops_corner_shortcuts(self):
        from jaxcad.meshing.features import feature_cell_links

        grid = GridSpec.from_bounds((-0.75,) * 3, (1.5,) * 3, 15)
        sdf = lambda p: Box.sdf(p, jnp.array([0.5, 0.5, 0.5]))  # noqa: E731
        edges, hermite = detect_edges(sdf, grid)
        incidence = cell_edge_incidence(edges, grid)
        features = classify_feature_cells(hermite, incidence)
        mask = features.classes != FACE
        junctions = features.classes == CORNER
        links = feature_cell_links(mask, incidence, grid, junction_mask=junctions)

        adjacency: dict[int, set[int]] = {}
        for a, b in links:
            adjacency.setdefault(int(a), set()).add(int(b))
            adjacency.setdefault(int(b), set()).add(int(a))
        # No remaining link may shortcut around a corner cell...
        for a, b in links:
            if junctions[a] or junctions[b]:
                continue
            assert not any(junctions[c] for c in adjacency[int(a)] & adjacency[int(b)])
        # ...while every corner stays connected to its chains.
        for row in np.flatnonzero(junctions):
            assert int(row) in adjacency
