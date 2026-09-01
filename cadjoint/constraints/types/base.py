"""Base class for geometric constraints.

A constraint defines a relationship between parameters that must be satisfied.
Each constraint reduces the degrees of freedom (DOF) by the number of
independent constraint equations it introduces.
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass, field

from jax import Array

from cadjoint.fluent import Fluent
from cadjoint.geometry.parameters import Parameter


@dataclass
class Constraint(Fluent):
    """Base class for geometric constraints.

    Constraints are represented as residual equations c(x) = 0.
    The Jacobian ∂c/∂x is computed via AD over the flat residual function
    in dof.py, not per-constraint.

    Subclasses must implement:
    - compute_residual(): Returns constraint violation c(x)
    - dof_reduction(): Number of DOF this constraint removes
    - get_parameters(): List of parameters involved in this constraint

    Attributes:
        weight: Scalar applied to this constraint's residuals in penalty-based
            optimization. Use to normalize across types or scales — e.g.
            ``weight=1/distance`` makes a DistanceConstraint residual
            dimensionless. Defaults to 1.0.
    """

    weight: float = field(default=1.0, kw_only=True)

    def __post_init__(self):
        """Register this constraint on all parameters it references."""
        self.params = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), Parameter)
        }
        for param in self.get_parameters():
            param.add_constraint(self)

    @abstractmethod
    def compute_residual(self, param_values: dict[str, Array]) -> Array:
        """Compute constraint residual c(x).

        Args:
            param_values: Dictionary mapping parameter names to their current values

        Returns:
            Constraint residual (scalar or array). Zero means constraint is satisfied.
        """
        pass

    @abstractmethod
    def dof_reduction(self) -> int:
        """Number of degrees of freedom this constraint removes.

        Returns:
            Number of independent constraint equations (typically 1)
        """
        pass

    @abstractmethod
    def get_parameters(self) -> list[Parameter]:
        """Get list of parameters involved in this constraint.

        Returns:
            List of Parameter objects that this constraint depends on
        """
        pass
