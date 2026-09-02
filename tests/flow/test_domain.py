"""The geometry-to-penalisation coupling: the grid, and the profiles on it.

The property under test throughout is *compact support*, because that is
what the whole penalisation rests on: ``alpha_max`` multiplies ``chi``
everywhere, so a profile that leaves a tail in the open channels makes the
fluid porous exactly as fast as it makes the solid solid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.flow import PROFILES, FlowGrid, sample_solid_fraction, solid_fraction

COMPACT = ("smootherstep", "smoothstep")


@pytest.fixture
def grid():
    return FlowGrid(shape=(4, 8, 4), origin=(-1.0, -2.0, -0.5), size=(2.0, 4.0, 1.0))


class TestFlowGrid:
    def test_spacing_and_volume_follow_from_size(self, grid):
        assert grid.spacing == pytest.approx((0.5, 0.5, 0.25))
        assert grid.cell_volume == pytest.approx(0.5 * 0.5 * 0.25)
        assert grid.cells == 4 * 8 * 4

    def test_centers_are_cell_centres_not_corners(self, grid):
        """Half-cell offsets: a sampled SDF must not sit on the boundary."""
        centers = grid.centers()
        assert centers.shape == (4, 8, 4, 3)
        assert np.asarray(centers[0, 0, 0]) == pytest.approx([-0.75, -1.75, -0.375])
        assert np.asarray(centers[-1, -1, -1]) == pytest.approx([0.75, 1.75, 0.375])

    def test_flat_centers_matches_centers(self, grid):
        assert np.allclose(grid.flat_centers(), grid.centers().reshape(-1, 3))

    def test_suggested_epsilon_is_half_a_cell(self, grid):
        """Two cells of interface: narrow enough that a thin fin fills."""
        assert grid.suggested_epsilon() == pytest.approx(0.5 * (0.5 * 0.5 * 0.25) ** (1 / 3))


class TestSolidFraction:
    @pytest.mark.parametrize("profile", PROFILES)
    def test_the_surface_is_the_half_contour(self, profile):
        assert float(solid_fraction(jnp.array(0.0), 1.0, profile)) == pytest.approx(0.5)

    @pytest.mark.parametrize("profile", PROFILES)
    def test_it_is_monotone_decreasing_in_distance(self, profile):
        d = jnp.linspace(-3.0, 3.0, 101)
        chi = np.asarray(solid_fraction(d, 0.5, profile))
        assert np.all(np.diff(chi) <= 1e-12)
        assert np.all((chi >= 0.0) & (chi <= 1.0))

    @pytest.mark.parametrize("profile", COMPACT)
    def test_a_compact_profile_is_exactly_zero_and_one_outside_the_band(self, profile):
        """The property the penalisation depends on: no tail to amplify."""
        eps = 0.5
        assert float(solid_fraction(jnp.array(-eps), eps, profile)) == 1.0
        assert float(solid_fraction(jnp.array(eps), eps, profile)) == 0.0
        far = jnp.array([-10.0, -2.0, 2.0, 10.0])
        assert np.asarray(solid_fraction(far, eps, profile)) == pytest.approx([1, 1, 0, 0])

    def test_the_sigmoid_keeps_a_tail_which_is_why_it_is_not_the_default(self):
        """Documented contrast, not an accident: sigmoid never reaches zero."""
        tail = float(solid_fraction(jnp.array(4.0), 0.5, "sigmoid"))
        assert 0.0 < tail
        assert tail == pytest.approx(3.35e-4, rel=0.1)

    @pytest.mark.parametrize("profile", COMPACT)
    def test_a_compact_profile_has_no_gradient_outside_the_band(self, profile):
        """Moving a distant surface cannot change a cell's occupancy."""
        slope = jax.grad(lambda d: solid_fraction(d, 0.5, profile))
        assert float(slope(jnp.array(5.0))) == 0.0
        assert float(slope(jnp.array(-5.0))) == 0.0

    @pytest.mark.parametrize("profile", COMPACT)
    def test_the_derivative_is_continuous_across_the_band_edge(self, profile):
        """C1 at the join -- what keeps the clamp from kinking the gradient."""
        slope = jax.grad(lambda d: solid_fraction(d, 0.5, profile))
        inside = float(slope(jnp.array(-0.5 + 1e-7)))
        outside = float(slope(jnp.array(-0.5 - 1e-7)))
        assert inside == pytest.approx(outside, abs=1e-5)

    def test_only_smootherstep_is_c2_at_the_band_edge(self):
        """Why it is the default: the cubic's curvature jumps, the quintic's does not.

        A jump in the second derivative is what drops a central difference
        from second-order to first, which is exactly what
        ``cadjoint.flow.domain`` measures.
        """
        curvature = jax.grad(jax.grad(lambda d: solid_fraction(d, 0.5, "smootherstep")))
        assert float(curvature(jnp.array(-0.5 + 1e-6))) == pytest.approx(0.0, abs=1e-3)
        cubic = jax.grad(jax.grad(lambda d: solid_fraction(d, 0.5, "smoothstep")))
        assert abs(float(cubic(jnp.array(-0.5 + 1e-6)))) > 1.0

    def test_an_unknown_profile_is_refused(self):
        with pytest.raises(ValueError, match="profile must be one of"):
            solid_fraction(jnp.array(0.0), 1.0, "heaviside")


class TestSampling:
    @staticmethod
    def _sphere(radius):
        return lambda points: jnp.linalg.norm(points, axis=-1) - radius

    def test_sampling_a_sphere_fills_the_middle_and_clears_the_corners(self, grid):
        # The nearest cell centre sits 0.367 from the origin, so a 0.4
        # sphere buries it by 0.033 -- epsilon has to be under that for the
        # cell to saturate at all.
        chi = sample_solid_fraction(self._sphere(0.4), grid, epsilon=0.02)
        assert chi.shape == grid.shape
        assert float(chi.max()) == pytest.approx(1.0)
        assert float(chi[0, 0, 0]) == 0.0

    def test_the_sampled_field_is_differentiable_in_the_geometry(self, grid):
        """The whole point: ``d chi / d(design)`` exists with no meshing.

        Growing the sphere can only add solid, so the total must rise.
        """

        def total(radius):
            return jnp.sum(sample_solid_fraction(self._sphere(radius), grid, epsilon=0.3))

        assert float(jax.grad(total)(0.4)) > 0.0

    def test_epsilon_defaults_to_the_grids_suggestion(self, grid):
        auto = sample_solid_fraction(self._sphere(0.4), grid)
        explicit = sample_solid_fraction(self._sphere(0.4), grid, grid.suggested_epsilon())
        assert np.allclose(auto, explicit)
