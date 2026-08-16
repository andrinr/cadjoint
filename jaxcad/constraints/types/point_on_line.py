"""Point-on-line constraint for sketch points."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from jaxcad.constraints.types._sketch import coerce_parameter_fields, cross_2d, require_values
from jaxcad.constraints.types.base import Constraint
from jaxcad.geometry.parameters import Parameter, Vector2


@dataclass
class PointOnLineConstraint(Constraint):
    """Constraint that forces a sketch point onto the infinite line through two points.

    This constraint enforces collinearity of ``point``, ``line1``, and
    ``line2`` via the 2D cross product:
    cross(point - line1, line2 - line1) = 0

    Reduces DOF by 1 (one scalar equation).

    Args:
        point: The point to place on the line (Vector2 parameter)
        line1: First point defining the line (Vector2 parameter)
        line2: Second point defining the line (Vector2 parameter)

    Example:
        ```python
        p = Vector2([1, 1], free=True, name='p')
        l1 = Vector2([0, 0], free=True, name='l1')
        l2 = Vector2([2, 0], free=True, name='l2')
        constraint = PointOnLineConstraint(p, l1, l2)
        ```
    """

    point: Vector2
    line1: Vector2
    line2: Vector2

    def __post_init__(self):
        coerce_parameter_fields(self, ("point", "line1", "line2"))
        super().__post_init__()

    def compute_residual(self, param_values: dict[str, Array]) -> Array:
        """Compute point-on-line residual: cross(point - line1, line2 - line1).

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Scalar residual (0 when the point lies on the line)
        """
        p_val, l1_val, l2_val = require_values(
            param_values, [self.point.name, self.line1.name, self.line2.name]
        )
        return cross_2d(p_val - l1_val, l2_val - l1_val)

    def dof_reduction(self) -> int:
        """Point-on-line constraint adds 1 scalar equation."""
        return 1

    def get_parameters(self) -> list[Parameter]:
        """Return the point and both line-defining points."""
        return [self.point, self.line1, self.line2]

    def __repr__(self) -> str:
        return f"PointOnLineConstraint({self.point.name}, {self.line1.name}, {self.line2.name})"
