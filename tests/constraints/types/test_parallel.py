"""Tests for ParallelConstraint."""

import jax.numpy as jnp
import pytest

from cadjoint.constraints import ParallelConstraint
from cadjoint.geometry.parameters import Vector


@pytest.mark.parametrize(
    "v1_xyz,v2_xyz,is_satisfied",
    [
        ([1, 0, 0], [2, 0, 0], True),  # Parallel
        ([1, 0, 0], [-3, 0, 0], True),  # Anti-parallel
        ([0, 1, 0], [0, 5, 0], True),  # Different magnitude
        ([1, 0, 0], [0, 1, 0], False),  # Perpendicular
        ([1, 1, 0], [1, 0, 0], False),  # Not parallel
    ],
)
def test_parallel_constraint_residual(v1_xyz, v2_xyz, is_satisfied):
    """Test that parallel constraint residual is correct."""
    v1 = Vector(v1_xyz, free=True, name="v1")
    v2 = Vector(v2_xyz, free=True, name="v2")

    constraint = ParallelConstraint(v1, v2)
    param_values = {"v1": v1.xyz, "v2": v2.xyz}

    residual = constraint.compute_residual(param_values)

    # Cross product should be zero for parallel vectors
    if is_satisfied:
        assert jnp.linalg.norm(residual) < 1e-6
    else:
        assert jnp.linalg.norm(residual) > 1e-3


def test_parallel_constraint_dof_reduction():
    """Test that parallel constraint reduces DOF by 2."""
    v1 = Vector([1, 0, 0], free=True, name="v1")
    v2 = Vector([2, 0, 0], free=True, name="v2")

    constraint = ParallelConstraint(v1, v2)

    assert constraint.dof_reduction() == 2


def test_parallel_constraint_get_parameters():
    """Test that get_parameters returns both vectors."""
    v1 = Vector([1, 0, 0], free=True, name="v1")
    v2 = Vector([2, 0, 0], free=True, name="v2")

    constraint = ParallelConstraint(v1, v2)
    params = constraint.get_parameters()

    assert len(params) == 2
    assert params[0] is v1
    assert params[1] is v2
