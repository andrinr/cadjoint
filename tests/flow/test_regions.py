"""Node selections resolved volumetrically on a lattice.

The selection language is the mesh one, read through ``describe()``.  These
tests pin the two things that could silently drift: that every kind means
on cells what it means on nodes, and that the two kinds a lattice cannot
honour are refused with the alternative named rather than approximated.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.fem import Nodes
from cadjoint.flow import FlowGrid, region_mask


@pytest.fixture
def centers():
    """Cell centres of a small lattice spanning the unit cube about the origin."""
    grid = FlowGrid(shape=(8, 8, 8), origin=(-1.0, -1.0, -1.0), size=(2.0, 2.0, 2.0))
    return np.asarray(grid.centers(), dtype=np.float64)


class TestKinds:
    """Each selection kind picks the cells its geometry contains."""

    def test_box_selects_the_cells_inside_it(self, centers):
        mask = region_mask(Nodes.box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]), centers)
        inside = np.all(np.abs(centers) <= 0.5, axis=-1)

        assert np.array_equal(mask, inside)
        assert mask.shape == centers.shape[:-1]

    def test_sphere_selects_by_radius(self, centers):
        mask = region_mask(Nodes.sphere([0.0, 0.0, 0.0], 0.6), centers)

        assert np.array_equal(mask, np.sum(centers**2, axis=-1) <= 0.36)

    def test_halfspace_selects_the_normal_side(self, centers):
        mask = region_mask(Nodes.halfspace([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), centers)

        assert np.array_equal(mask, centers[..., 2] >= 0.0)

    def test_cylinder_honours_bore_and_length(self, centers):
        selection = Nodes.cylinder(
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.8, inner=0.3, half_length=0.5
        )
        mask = region_mask(selection, centers)
        radial = np.hypot(centers[..., 0], centers[..., 1])
        expected = (radial <= 0.8) & (radial >= 0.3) & (np.abs(centers[..., 2]) <= 0.5)

        assert np.array_equal(mask, expected)

    def test_none_selects_every_cell(self, centers):
        assert region_mask(None, centers).all()


class TestComposition:
    """``&``, ``|`` and ``~`` compose on cells as they do on nodes."""

    def test_and_or_not(self, centers):
        left = Nodes.halfspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        right = Nodes.halfspace([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        both = region_mask(left & right, centers)
        either = region_mask(left | right, centers)
        neither = region_mask(~(left | right), centers)

        assert np.array_equal(both, region_mask(left, centers) & region_mask(right, centers))
        assert np.array_equal(either, region_mask(left, centers) | region_mask(right, centers))
        assert np.array_equal(neither, ~either)

    def test_complement_is_volumetric_not_surface(self, centers):
        """``~box`` is every other *cell*, interior ones included.

        On a mesh the complement lives inside the boundary surface; on a
        lattice there is no surface to be inside of, and a heated region's
        complement has to include the cells the heat conducts into.
        """
        box = Nodes.box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        outside = region_mask(~box, centers)

        assert outside.sum() + region_mask(box, centers).sum() == centers[..., 0].size


class TestRefusals:
    """The two kinds a lattice cannot honour name the alternative."""

    def test_side_is_refused_naming_the_alternative(self, centers):
        with pytest.raises(ValueError, match="halfspace"):
            region_mask(Nodes.side("+x"), centers)

    def test_predicate_is_refused_as_unserializable(self, centers):
        with pytest.raises(ValueError, match="not serializable"):
            region_mask(Nodes.predicate(lambda points: points[:, 0] > 0), centers)

    def test_a_refusal_survives_composition(self, centers):
        """A refused kind nested inside a combination still refuses."""
        selection = Nodes.box([-1, -1, -1], [1, 1, 1]) & Nodes.side("+x")

        with pytest.raises(ValueError, match="halfspace"):
            region_mask(selection, centers)
