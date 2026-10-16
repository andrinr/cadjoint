"""Fitting the extraction lattice to a part at its own scale (issue #49)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.geometry import Vector
from cadjoint.meshing import GridSpec
from cadjoint.meshing.bounds import SCAN_HALF_WIDTHS, boundary_inside, fit_bounds
from cadjoint.sdf.primitives import Box, Sphere
from cadjoint.sdf.transforms import Translate


def _field(sdf):
    return lambda p: jnp.asarray(sdf(jnp.asarray(p)))


def _assert_holds(bounds, half_extent, centre=(0.0, 0.0, 0.0), slack=0.1):
    """The box contains the part and is within *slack* of its size per side."""
    origin, size = (np.asarray(v) for v in bounds)
    low, high = np.asarray(centre) - half_extent, np.asarray(centre) + half_extent
    assert np.all(origin <= low) and np.all(origin + size >= high)
    margin = slack * 2.0 * np.asarray(half_extent).max()
    assert np.all(low - origin <= margin) and np.all(origin + size - high <= margin)


@pytest.mark.parametrize(
    "half_extent",
    [
        (50.0, 20.0, 3.0),  # the issue's plate, in millimetres: clipped at ±3 before
        (10.0, 10.0, 10.0),  # larger than the default box in every axis: empty before
        (0.2, 0.15, 0.1),  # smaller than the old scan spacing in z
        (800.0, 300.0, 40.0),
    ],
)
def test_a_box_is_found_at_its_own_scale(half_extent):
    half = np.asarray(half_extent)
    bounds = fit_bounds(_field(Box(Vector(list(half_extent)))))
    _assert_holds(bounds, half)


def test_a_plate_thinner_than_the_scan_spacing_is_found():
    """0.05 thick: no 1/24-spaced sample need land inside, the near-surface rule finds it."""
    half = np.array([2.0, 1.5, 0.025])
    bounds = fit_bounds(_field(Box(Vector(list(half)))))
    _assert_holds(bounds, half, slack=0.15)


def test_an_off_centre_part_is_found_where_it_is():
    sdf = Translate(Sphere(radius=5.0), Vector([120.0, -40.0, 7.0]))
    # Found in the 256 cube: the refine pass's pad is 1.5 of its cells plus the
    # near band, about 15 % of this part per side.
    bounds = fit_bounds(_field(sdf))
    _assert_holds(bounds, np.full(3, 5.0), centre=(120.0, -40.0, 7.0), slack=0.2)


def test_padding_widens_every_side():
    sdf = _field(Sphere(radius=1.5))
    (origin, size), (padded_origin, padded_size) = fit_bounds(sdf), fit_bounds(sdf, padding=0.5)
    np.testing.assert_allclose(np.subtract(origin, padded_origin), 0.5, atol=1e-6)
    np.testing.assert_allclose(np.subtract(padded_size, size), 1.0, atol=1e-6)


def test_an_unbounded_field_is_refused_rather_than_clipped():
    half_space = lambda p: jnp.asarray(p)[..., 2]  # noqa: E731
    with pytest.raises(ValueError, match="unbounded"):
        fit_bounds(half_space)


def test_a_part_wider_than_the_last_scan_is_refused():
    huge = SCAN_HALF_WIDTHS[-1] * 2.0
    with pytest.raises(ValueError, match="No closed part"):
        fit_bounds(_field(Box(Vector([huge, 1.0, 1.0]))))


def test_boundary_inside_sees_a_grid_wholly_inside_the_part():
    """No surface crosses this grid, so the extractor's warning cannot fire."""
    sdf = _field(Box(Vector([10.0, 10.0, 10.0])))
    inside = GridSpec.from_bounds((-3.0, -3.0, -3.0), (6.0, 6.0, 6.0), 8)
    around = GridSpec.from_bounds((-11.0, -11.0, -11.0), (22.0, 22.0, 22.0), 8)
    assert boundary_inside(sdf, inside)
    assert not boundary_inside(sdf, around)
