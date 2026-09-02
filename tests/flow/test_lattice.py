"""The D3Q19 tables, and the one formula that turns a viscosity into a rate."""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.flow import CS2, OPP, C, Q, W, omega_from_viscosity, viscosity_from_omega
from cadjoint.flow.lattice import D


class TestLattice:
    """The moment identities every derivation in the package assumes.

    ``lattice.py`` asserts these at import, which protects the running
    solver but says nothing in a test report.  Restating them here means a
    typo in the velocity table is named rather than merely raised.
    """

    def test_the_velocity_set_is_the_expected_shape(self):
        assert C.shape == (Q, D) == (19, 3)
        assert set(np.abs(C).sum(axis=1)) == {0, 1, 2}

    def test_the_weights_are_a_probability_distribution(self):
        assert W.sum() == pytest.approx(1.0)
        assert np.all(W > 0)

    def test_the_first_moment_vanishes(self):
        """No net momentum at rest, or the fluid would drift unforced."""
        assert np.allclose(W @ C, 0.0)

    def test_the_second_moment_is_isotropic(self):
        """``sum_q w_q c_qa c_qb = cs^2 delta_ab`` -- the Navier-Stokes limit."""
        assert np.allclose(np.einsum("q,qa,qb->ab", W, C, C), CS2 * np.eye(3))

    def test_opposite_indices_negate_the_velocity(self):
        """Bounce-back is a permutation, so ``OPP`` must be an involution."""
        assert np.all(C[OPP] == -C)
        assert np.all(OPP[OPP] == np.arange(Q))


class TestRelaxation:
    def test_viscosity_and_omega_are_inverse(self):
        for omega in (0.5, 1.0, 1.4, 1.9):
            assert omega_from_viscosity(viscosity_from_omega(omega)) == pytest.approx(omega)

    def test_omega_one_is_the_lattice_viscosity(self):
        """``nu = cs^2 (1/omega - 1/2)`` puts ``omega = 1`` at ``cs^2/2``."""
        assert viscosity_from_omega(1.0) == pytest.approx(CS2 / 2.0)

    def test_a_viscosity_outside_the_stable_range_is_refused(self):
        """A negative viscosity would put omega above 2 and diverge."""
        with pytest.raises(ValueError, match="outside the stable range"):
            omega_from_viscosity(-0.1)
