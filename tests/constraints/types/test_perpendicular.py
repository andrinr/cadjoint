"""Tests for PerpendicularConstraint."""

import jax.numpy as jnp
import pytest

from cadjoint.constraints import PerpendicularConstraint
from cadjoint.geometry.parameters import Vector


@pytest.mark.parametrize(
    "v1_xyz,v2_xyz,expected_residual",
    [
        ([1, 0, 0], [0, 1, 0], 0.0),  # Perpendicular
        ([1, 0, 0], [0, 0, 1], 0.0),  # Perpendicular in different plane
        ([1, 0, 0], [1, 1, 0], 1.0),  # 45 degrees, dot product = 1
        ([1, 0, 0], [2, 0, 0], 2.0),  # Parallel, dot product = 2
    ],
)
def test_perpendicular_constraint_residual(v1_xyz, v2_xyz, expected_residual):
    """Test that perpendicular constraint residual is correct."""
    v1 = Vector(v1_xyz, free=True, name="v1")
    v2 = Vector(v2_xyz, free=True, name="v2")

    constraint = PerpendicularConstraint(v1, v2)
    param_values = {"v1": v1.xyz, "v2": v2.xyz}

    residual = constraint.compute_residual(param_values)

    assert jnp.abs(residual - expected_residual) < 1e-6


def test_perpendicular_constraint_dof_reduction():
    """Test that perpendicular constraint reduces DOF by 1."""
    v1 = Vector([1, 0, 0], free=True, name="v1")
    v2 = Vector([0, 1, 0], free=True, name="v2")

    constraint = PerpendicularConstraint(v1, v2)

    assert constraint.dof_reduction() == 1


def test_perpendicular_constraint_get_parameters():
    """Test that get_parameters returns both vectors."""
    v1 = Vector([1, 0, 0], free=True, name="v1")
    v2 = Vector([0, 1, 0], free=True, name="v2")

    constraint = PerpendicularConstraint(v1, v2)
    params = constraint.get_parameters()

    assert len(params) == 2
    assert params[0] is v1
    assert params[1] is v2
