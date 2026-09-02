"""The D3Q19 lattice: velocities, weights, and the bounce-back permutation.

Constants only, as NumPy arrays built once at import.  Everything here is
static under JAX tracing -- the velocity table indexes a ``roll`` and the
opposite-index table indexes a ``gather``, so neither ever becomes a traced
value and neither costs anything at runtime.

D3Q19 is the smallest three-dimensional set that is still isotropic enough
for the Navier-Stokes limit (D3Q15 is cheaper but its fourth-order moment
tensor is anisotropic; D3Q27 is more robust at high Reynolds number and 40%
more expensive).  At the Reynolds numbers a heat sink sees in forced
convection, D3Q19 is the standard choice.
"""

from __future__ import annotations

import numpy as np

#: Number of discrete velocities.
Q = 19

#: Spatial dimension.
D = 3


def _build_velocities() -> np.ndarray:
    """The 19 discrete velocities, rest first then the two speed shells.

    Returns:
        ``(19, 3)`` integer array of lattice velocities.
    """
    rest = [(0, 0, 0)]
    face = (
        [(s, 0, 0) for s in (1, -1)] + [(0, s, 0) for s in (1, -1)] + [(0, 0, s) for s in (1, -1)]
    )
    edge = []
    for a in (1, -1):
        for b in (1, -1):
            edge.extend([(a, b, 0), (a, 0, b), (0, a, b)])
    return np.array(rest + face + edge, dtype=np.int32)


#: ``(19, 3)`` lattice velocities.
C = _build_velocities()

#: ``(19,)`` lattice weights: 1/3 at rest, 1/18 on the face shell, 1/36 on
#: the edge shell.  They sum to one and reproduce the Maxwellian moments up
#: to fourth order.
W = np.where(
    np.abs(C).sum(axis=1) == 0,
    1.0 / 3.0,
    np.where(np.abs(C).sum(axis=1) == 1, 1.0 / 18.0, 1.0 / 36.0),
).astype(np.float64)


def _build_opposite() -> np.ndarray:
    """Index of ``-c_i`` for each velocity, the bounce-back permutation.

    Returns:
        ``(19,)`` integer array with ``C[OPP[i]] == -C[i]``.
    """
    lookup = {tuple(c): i for i, c in enumerate(C)}
    return np.array([lookup[tuple(-c)] for c in C], dtype=np.int32)


#: ``(19,)`` bounce-back permutation.
OPP = _build_opposite()

#: Lattice speed of sound squared, ``1/3`` for every standard DnQm set.
CS2 = 1.0 / 3.0

# Sanity invariants, checked once at import: these are the properties every
# derivation below assumes, and a typo in the tables above would otherwise
# surface as a subtly wrong viscosity rather than an error.
assert C.shape == (Q, D)
assert np.isclose(W.sum(), 1.0)
assert np.all(C[OPP] == -C)
assert np.allclose(W @ C, 0.0)
assert np.allclose(np.einsum("q,qa,qb->ab", W, C, C), CS2 * np.eye(D))


def omega_from_viscosity(viscosity: float) -> float:
    """BGK relaxation rate for a kinematic viscosity in lattice units.

    The Chapman-Enskog expansion gives ``nu = cs^2 (1/omega - 1/2)``, so a
    viscosity fixes the single BGK relaxation rate.

    Args:
        viscosity: Kinematic viscosity in lattice units (cells^2 per step).

    Returns:
        The relaxation rate ``omega``, in ``(0, 2)``.

    Raises:
        ValueError: If the viscosity puts ``omega`` outside its stable range.
    """
    omega = 1.0 / (viscosity / CS2 + 0.5)
    if not 0.0 < omega < 2.0:
        raise ValueError(
            f"viscosity {viscosity} gives omega={omega:.4f}, outside the stable range (0, 2)."
        )
    return omega


def viscosity_from_omega(omega: float) -> float:
    """Kinematic viscosity in lattice units for a BGK relaxation rate.

    Args:
        omega: The relaxation rate, in ``(0, 2)``.

    Returns:
        The kinematic viscosity in lattice units.
    """
    return CS2 * (1.0 / omega - 0.5)
