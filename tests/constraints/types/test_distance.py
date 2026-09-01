"""Tests for DistanceConstraint."""

import jax.numpy as jnp
import pytest

from cadjoint.constraints import DistanceConstraint
from cadjoint.geometry.parameters import Vector


@pytest.mark.parametrize(
    "p1_xyz,p2_xyz,target_dist,expected_residual",
    [
        ([0, 0, 0], [1, 0, 0], 1.0, 0.0),  # Satisfied
        ([0, 0, 0], [1, 0, 0], 0.5, 0.5),  # Violated
        ([0, 0, 0], [3, 4, 0], 5.0, 0.0),  # 3-4-5 triangle
        ([1, 2, 3], [1, 2, 6], 3.0, 0.0),  # Offset points
    ],
)
def test_distance_constraint_residual(p1_xyz, p2_xyz, target_dist, expected_residual):
    """Test that residual is computed correctly."""
    p1 = Vector(p1_xyz, free=True, name="p1")
    p2 = Vector(p2_xyz, free=True, name="p2")

    constraint = DistanceConstraint(p1, p2, target_dist)

    param_values = {"p1": p1.xyz, "p2": p2.xyz}
    residual = constraint.compute_residual(param_values)

    assert jnp.abs(residual - expected_residual) < 1e-6


def test_distance_constraint_dof_reduction():
    """Test that distance constraint reduces DOF by 1."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraint = DistanceConstraint(p1, p2, 1.0)

    assert constraint.dof_reduction() == 1


def test_distance_constraint_get_parameters():
    """Test that get_parameters returns both points."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraint = DistanceConstraint(p1, p2, 1.0)
    params = constraint.get_parameters()

    assert len(params) == 2
    assert params[0] is p1
    assert params[1] is p2
