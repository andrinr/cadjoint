"""Angle constraint between two vectors."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from cadjoint.constraints.types.base import Constraint
from cadjoint.geometry.parameters import Parameter, Scalar, Vector


@dataclass
class AngleConstraint(Constraint):
    """Constraint that fixes the angle between two vectors.

    This constraint enforces: arccos(v1 · v2 / (||v1|| ||v2||)) = angle

    Reduces DOF by 1 (one scalar equation).

    Args:
        vector1: First direction vector (Vector parameter)
        vector2: Second direction vector (Vector parameter)
        angle: Target angle in radians (Scalar parameter or float)

    Example:
        ```python
        v1 = Vector([1, 0, 0], free=True, name='v1')
        v2 = Vector([0, 1, 0], free=True, name='v2')
        constraint = AngleConstraint(v1, v2, jnp.pi / 2)
        ```
    """

    vector1: Vector
    vector2: Vector
    angle: Scalar | float

    def __post_init__(self):
        from cadjoint.geometry.parameters import as_parameter

        if not isinstance(self.angle, Scalar):
            self.angle = as_parameter(self.angle)
        self._cos_target = jnp.cos(self.angle.value)
        super().__post_init__()

    def compute_residual(self, param_values: dict[str, Array]) -> Array:
        """Compute angle constraint residual: v1·v2/(||v1||||v2||) - cos(θ).

        Using cos(θ) instead of arccos avoids singularities at 0/π where
        the arccos gradient vanishes, which can stall the LM solver.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Scalar residual (0 when constraint is satisfied)
        """
        v1_name = self.vector1.name
        v2_name = self.vector2.name

        if v1_name not in param_values or v2_name not in param_values:
            raise ValueError(f"Parameter values must include '{v1_name}' and '{v2_name}'")

        v1_val = param_values[v1_name]
        v2_val = param_values[v2_name]

        # Normalize vectors
        v1_norm = v1_val / jnp.linalg.norm(v1_val)
        v2_norm = v2_val / jnp.linalg.norm(v2_val)

        # Residual: cos(current_angle) - cos(target_angle)
        return jnp.dot(v1_norm, v2_norm) - self._cos_target

    def dof_reduction(self) -> int:
        """Angle constraint adds 1 scalar equation, reducing DOF by 1."""
        return 1

    def get_parameters(self) -> list[Parameter]:
        """Return both vectors involved in the angle constraint."""
        return [self.vector1, self.vector2]

    def __repr__(self) -> str:
        return f"AngleConstraint({self.vector1.name}, {self.vector2.name}, θ={self.angle.value})"
