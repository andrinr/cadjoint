"""Coincident constraint between two sketch points."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from jaxcad.constraints.types._sketch import coerce_parameter_fields, require_values
from jaxcad.constraints.types.base import Constraint
from jaxcad.geometry.parameters import Parameter, Vector2


@dataclass
class CoincidentConstraint(Constraint):
    """Constraint that forces two sketch points to coincide.

    This constraint enforces: p1 = p2 (both components, in sketch-plane
    coordinates).

    Reduces DOF by 2 (one equation per component).

    Args:
        point1: First sketch point (Vector2 parameter)
        point2: Second sketch point (Vector2 parameter)

    Example:
        ```python
        p1 = Vector2([0, 0], free=True, name='p1')
        p2 = Vector2([1, 1], free=True, name='p2')
        constraint = CoincidentConstraint(p1, p2)
        ```
    """

    point1: Vector2
    point2: Vector2

    def __post_init__(self):
        coerce_parameter_fields(self, ("point1", "point2"))
        super().__post_init__()

    def compute_residual(self, param_values: dict[str, Array]) -> Array:
        """Compute coincident constraint residual: p1 - p2.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            2-component residual ([0, 0] when constraint is satisfied)
        """
        p1_val, p2_val = require_values(param_values, [self.point1.name, self.point2.name])
        return p1_val - p2_val

    def dof_reduction(self) -> int:
        """Coincident constraint adds 2 equations (one per component)."""
        return 2

    def get_parameters(self) -> list[Parameter]:
        """Return both points involved in the coincident constraint."""
        return [self.point1, self.point2]

    def __repr__(self) -> str:
        return f"CoincidentConstraint({self.point1.name}, {self.point2.name})"
