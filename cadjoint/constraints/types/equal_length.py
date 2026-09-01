"""Equal-length constraint between two sketch edges."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from cadjoint.constraints.types._sketch import coerce_parameter_fields, require_values
from cadjoint.constraints.types.base import Constraint
from cadjoint.geometry.parameters import Parameter, Vector2


@dataclass
class EqualLengthConstraint(Constraint):
    """Constraint that forces two sketch edges to have equal length.

    Edges are given by their endpoint parameters. This constraint enforces:
    ||a1 - a2|| = ||b1 - b2||

    Reduces DOF by 1 (one scalar equation).

    Args:
        a1: First endpoint of edge A (Vector2 parameter)
        a2: Second endpoint of edge A (Vector2 parameter)
        b1: First endpoint of edge B (Vector2 parameter)
        b2: Second endpoint of edge B (Vector2 parameter)

    Example:
        ```python
        a1 = Vector2([0, 0], free=True, name='a1')
        a2 = Vector2([1, 0], free=True, name='a2')
        b1 = Vector2([0, 1], free=True, name='b1')
        b2 = Vector2([2, 1], free=True, name='b2')
        constraint = EqualLengthConstraint(a1, a2, b1, b2)
        ```
    """

    a1: Vector2
    a2: Vector2
    b1: Vector2
    b2: Vector2

    def __post_init__(self):
        coerce_parameter_fields(self, ("a1", "a2", "b1", "b2"))
        super().__post_init__()

    def compute_residual(self, param_values: dict[str, Array]) -> Array:
        """Compute equal-length residual: ||a1 - a2|| - ||b1 - b2||.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Scalar residual (0 when constraint is satisfied)
        """
        a1_val, a2_val, b1_val, b2_val = require_values(
            param_values, [self.a1.name, self.a2.name, self.b1.name, self.b2.name]
        )
        return jnp.linalg.norm(a1_val - a2_val) - jnp.linalg.norm(b1_val - b2_val)

    def dof_reduction(self) -> int:
        """Equal-length constraint adds 1 scalar equation."""
        return 1

    def get_parameters(self) -> list[Parameter]:
        """Return all four endpoints involved in the constraint."""
        return [self.a1, self.a2, self.b1, self.b2]

    def __repr__(self) -> str:
        return (
            f"EqualLengthConstraint({self.a1.name}, {self.a2.name}, {self.b1.name}, {self.b2.name})"
        )
