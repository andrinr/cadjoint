"""Base class for fluent API pattern shared by SDFs and Constraints.

This module provides shared utility methods for parameter management.

Architecture:
- Thin wrappers around pure functions
- Store parameters in self.params dict
- Auto-cast raw values to Parameter objects
- Enable fluent API and composition patterns

Subclasses (SDF, Constraint) handle their own initialization logic.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadjoint.geometry.parameters import Parameter


class Fluent(ABC):
    """Abstract base class for fluent API objects (SDFs, Constraints, etc.).

    Provides shared utility methods for parameter management:
    - _cast_params(): Convert raw values to Parameter objects
    - _extract_param_values(): Extract values for compilation

    Subclasses define self.params dict and handle their own __init_subclass__.
    """

    params: dict[str, Parameter]

    def _cast_params(self) -> None:
        """Convert all values in self.params to Parameter objects.

        This is automatically called after __init__ via __init_subclass__.
        Converts raw values (floats, ints, arrays) to Parameter objects,
        while leaving existing Parameter objects unchanged.
        """
        from cadjoint.geometry.parameters import as_parameter

        # Convert each value in the params dict
        for key, value in self.params.items():
            # as_parameter already handles case where value is already a Parameter
            self.params[key] = as_parameter(value)

    def children(self) -> list[Fluent]:
        """Return the direct child nodes of this object in the tree.

        Override in subclasses that have children (e.g. Transform, BooleanOp).
        The default implementation returns an empty list (leaf node).
        """
        return []

    def _extract_param_values(self) -> dict:
        """Extract raw numeric values from Parameter objects for compilation.

        Returns:
            Dictionary mapping parameter names to their raw values (arrays/floats).
            Vector parameters are converted to 3D arrays (xyz), Scalars to floats.
        """
        from cadjoint.geometry.parameters import Vector

        params = {}
        for param_name, param in self.params.items():
            if isinstance(param, Vector):
                params[param_name] = param.xyz
            else:
                params[param_name] = param.value
        return params
