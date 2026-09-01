"""Geometric entities and parameters for parametric CAD.

This module provides:
1. Parameter types (Vector, Scalar) for optimization
2. Geometric primitives (Line, Circle) for construction
3. Re-exports for convenience

The geometry layer is independent of SDFs and constraints.
"""

# Parameter types
from cadjoint.geometry.parameters import (
    NamedParams,
    Parameter,
    PathParams,
    Scalar,
    Vector,
    Vector2,
    as_parameter,
    deduplicate_params,
)

# Geometric primitives
from cadjoint.geometry.primitives import (
    Circle,
    Line,
)

__all__ = [
    # Parameters
    "Parameter",
    "Scalar",
    "Vector",
    "Vector2",
    "as_parameter",
    "PathParams",
    "NamedParams",
    "deduplicate_params",
    # Primitives
    "Line",
    "Circle",
]
