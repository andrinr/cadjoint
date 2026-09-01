"""Tests for AngleConstraint."""

import jax.numpy as jnp
import pytest

from cadjoint.constraints import AngleConstraint
from cadjoint.geometry.parameters import Vector


@pytest.mark.parametrize(
    "v1_xyz,v2_xyz,target_angle,expected_residual",
    [
        ([1, 0, 0], [0, 1, 0], jnp.pi / 2, 0.0),  # Perpendicular
        ([1, 0, 0], [2, 0, 0], jnp.pi / 2, 1.0),  # Parallel vectors, target 90°: cos(0°)-cos(90°)=1
        ([1, 0, 0], [0, 0, 1], jnp.pi / 2, 0.0),  # Perpendicular in different plane
        ([1, 0, 0], [-1, 0, 0], jnp.pi, 0.0),  # Opposite direction
    ],
)
def test_angle_constraint_residual(v1_xyz, v2_xyz, target_angle, expected_residual):
    """Test that angle constraint residual is computed correctly."""
    v1 = Vector(v1_xyz, free=True, name="v1")
    v2 = Vector(v2_xyz, free=True, name="v2")

    constraint = AngleConstraint(v1, v2, target_angle)
    param_values = {"v1": v1.xyz, "v2": v2.xyz}

    residual = constraint.compute_residual(param_values)

    assert jnp.abs(residual - expected_residual) < 1e-6


def test_angle_constraint_dof_reduction():
    """Test that angle constraint reduces DOF by 1."""
    v1 = Vector([1, 0, 0], free=True, name="v1")
    v2 = Vector([0, 1, 0], free=True, name="v2")

    constraint = AngleConstraint(v1, v2, jnp.pi / 2)

    assert constraint.dof_reduction() == 1


def test_angle_constraint_get_parameters():
    """Test that get_parameters returns both vectors."""
    v1 = Vector([1, 0, 0], free=True, name="v1")
    v2 = Vector([0, 1, 0], free=True, name="v2")

    constraint = AngleConstraint(v1, v2, jnp.pi / 2)
    params = constraint.get_parameters()

    assert len(params) == 2
    assert params[0] is v1
    assert params[1] is v2
