"""Parallel constraint between two vectors."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from cadjoint.constraints.types.base import Constraint
from cadjoint.geometry.parameters import Parameter, Vector


@dataclass
class ParallelConstraint(Constraint):
    """Constraint that forces two vectors to be parallel.

    This is equivalent to AngleConstraint with angle=0 or angle=π.

    Reduces DOF by 2 (two scalar equations from cross product = 0).

    Args:
        vector1: First direction vector (Vector parameter)
        vector2: Second direction vector (Vector parameter)

    Example:
        ```python
        v1 = Vector([1, 0, 0], free=True, name='v1')
        v2 = Vector([2, 0, 0], free=True, name='v2')
        constraint = ParallelConstraint(v1, v2)
        ```
    """

    vector1: Vector
    vector2: Vector

    def compute_residual(self, param_values: dict[str, Array]) -> Array:
        """Compute parallel constraint residual: v1 × v2.

        Two vectors are parallel iff their cross product is zero.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Cross product (3D array, should be ~[0, 0, 0] when satisfied)
        """
        v1_name = self.vector1.name
        v2_name = self.vector2.name

        if v1_name not in param_values or v2_name not in param_values:
            raise ValueError(f"Parameter values must include '{v1_name}' and '{v2_name}'")

        v1_val = param_values[v1_name]
        v2_val = param_values[v2_name]

        # Cross product (for 3D vectors only)
        return jnp.cross(v1_val, v2_val)

    def dof_reduction(self) -> int:
        """Parallel constraint adds 2 independent equations (cross product in 3D)."""
        return 2

    def get_parameters(self) -> list[Parameter]:
        """Return both vectors involved in the parallel constraint."""
        return [self.vector1, self.vector2]

    def __repr__(self) -> str:
        return f"ParallelConstraint({self.vector1.name}, {self.vector2.name})"
