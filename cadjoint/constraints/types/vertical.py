"""Vertical constraint between two sketch points."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from cadjoint.constraints.types._sketch import coerce_parameter_fields, require_values
from cadjoint.constraints.types.base import Constraint
from cadjoint.geometry.parameters import Parameter, Vector2


@dataclass
class VerticalConstraint(Constraint):
    """Constraint that forces two sketch points onto a vertical line.

    This constraint enforces: x1 = x2 (in sketch-plane coordinates).

    Reduces DOF by 1 (one scalar equation).

    Args:
        point1: First sketch point (Vector2 parameter)
        point2: Second sketch point (Vector2 parameter)

    Example:
        ```python
        p1 = Vector2([0, 0], free=True, name='p1')
        p2 = Vector2([1, 2], free=True, name='p2')
        constraint = VerticalConstraint(p1, p2)
        ```
    """

    point1: Vector2
    point2: Vector2

    def __post_init__(self):
        coerce_parameter_fields(self, ("point1", "point2"))
        super().__post_init__()

    def compute_residual(self, param_values: dict[str, Array]) -> Array:
        """Compute vertical constraint residual: x1 - x2.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Scalar residual (0 when constraint is satisfied)
        """
        p1_val, p2_val = require_values(param_values, [self.point1.name, self.point2.name])
        return p1_val[0] - p2_val[0]

    def dof_reduction(self) -> int:
        """Vertical constraint adds 1 scalar equation."""
        return 1

    def get_parameters(self) -> list[Parameter]:
        """Return both points involved in the vertical constraint."""
        return [self.point1, self.point2]

    def __repr__(self) -> str:
        return f"VerticalConstraint({self.point1.name}, {self.point2.name})"
