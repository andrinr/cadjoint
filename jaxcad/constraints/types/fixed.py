"""Fixed (anchor) constraint pinning a parameter to a target value."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints.types.base import Constraint
from jaxcad.geometry.parameters import Parameter


@dataclass
class FixedConstraint(Constraint):
    """Constraint that anchors a free parameter to a fixed target value.

    This constraint enforces: param = target

    Unlike marking a parameter ``free=False``, an anchored parameter stays in
    the optimization vector, so it can participate in DOF counting and be
    released later without restructuring the scene.

    Reduces DOF by ``param.value.size`` (one equation per component).

    Args:
        param: Parameter to anchor (Scalar, Vector, or Vector2)
        target: Target value (array-like matching the parameter's shape)

    Example:
        ```python
        v0 = Vector2([0, 0], free=True, name='v0')
        anchor = FixedConstraint(v0, [0.0, 0.0])
        ```
    """

    param: Parameter
    target: float | Array

    def __post_init__(self):
        self.target = jnp.asarray(self.target, dtype=jnp.float32)
        if self.target.shape != jnp.shape(self.param.value):
            raise ValueError(
                f"FixedConstraint target shape {self.target.shape} does not match "
                f"parameter shape {jnp.shape(self.param.value)}"
            )
        super().__post_init__()

    def compute_residual(self, param_values: dict[str, Array]) -> Array:
        """Compute anchor residual: param - target (one equation per component)."""
        name = self.param.name
        if name not in param_values:
            raise ValueError(f"Parameter values must include '{name}'")
        return jnp.atleast_1d(param_values[name] - self.target)

    def dof_reduction(self) -> int:
        """One equation per parameter component."""
        return int(jnp.asarray(self.param.value).size)

    def get_parameters(self) -> list[Parameter]:
        """The single anchored parameter."""
        return [self.param]
