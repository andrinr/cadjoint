"""Parallel constraint between two sketch edges."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from jaxcad.constraints.types._sketch import coerce_parameter_fields, cross_2d, require_values
from jaxcad.constraints.types.base import Constraint
from jaxcad.geometry.parameters import Parameter, Vector2


@dataclass
class ParallelEdgesConstraint(Constraint):
    """Constraint that forces two sketch edges to be parallel.

    Edges are given by their endpoint parameters. Unlike
    :class:`~jaxcad.constraints.types.parallel.ParallelConstraint`, which
    operates on 3D direction vectors, this constrains the differences of 2D
    sketch points via the 2D cross product:
    cross(a2 - a1, b2 - b1) = 0

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
        b2 = Vector2([1, 2], free=True, name='b2')
        constraint = ParallelEdgesConstraint(a1, a2, b1, b2)
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
        """Compute parallel-edges residual: cross(a2 - a1, b2 - b1).

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Scalar residual (0 when the edges are parallel)
        """
        a1_val, a2_val, b1_val, b2_val = require_values(
            param_values, [self.a1.name, self.a2.name, self.b1.name, self.b2.name]
        )
        return cross_2d(a2_val - a1_val, b2_val - b1_val)

    def dof_reduction(self) -> int:
        """Parallel-edges constraint adds 1 scalar equation (2D cross product)."""
        return 1

    def get_parameters(self) -> list[Parameter]:
        """Return all four endpoints involved in the constraint."""
        return [self.a1, self.a2, self.b1, self.b2]

    def __repr__(self) -> str:
        return (
            f"ParallelEdgesConstraint({self.a1.name}, {self.a2.name}, "
            f"{self.b1.name}, {self.b2.name})"
        )
