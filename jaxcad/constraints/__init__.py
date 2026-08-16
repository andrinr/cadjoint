"""Constraint system for geometric relationships.

This module provides a constraint system that allows expressing geometric
relationships and automatically reduces degrees of freedom (DOF) during optimization.

Architecture:
- Constraints define relationships between parameters (distance, angle, etc.)
- residual.py builds flat residual functions and handles parameter vector pack/unpack
- null_space.py computes reduced DOF space via null-space projection
- solve.py drives optimization to satisfy constraints

Constraint families:
- Value constraints on 3D parameters: Fixed, Distance, Angle, Parallel,
  Perpendicular (the latter two act on direction vectors).
- 2D sketch constraints on Vector2 points: Horizontal, Vertical, Coincident,
  EqualLength, PointOnLine, and the edge pairs ParallelEdges /
  PerpendicularEdges (edges given by their endpoint parameters).

Example:
    from jaxcad.geometry import Vector, Scalar
    from jaxcad.constraints import DistanceConstraint, null_space

    # Two free points (6 DOF total)
    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([1, 0, 0], free=True, name='p2')

    # Distance constraint (reduces DOF by 1)
    constraint = DistanceConstraint(p1, p2, Scalar(0.2))

    # Extract reduced DOF (5 DOF instead of 6)
    reduced_params, null_space = null_space([constraint], [p1, p2])
"""

from __future__ import annotations

from jaxcad.constraints.null_space import (
    NullSpaceMap,
    all_parameters,
    null_space,
    total_dof_reduction,
)
from jaxcad.constraints.residual import (
    build_residual_fn,
    compute_param_vector,
    unpack_param_vector,
)
from jaxcad.constraints.solve import (
    constraint_residuals,
    make_bounds_projection,
    make_manifold_projection,
    project_bounds,
    project_to_manifold,
    satisfy_constraints,
    solve_constraints,
)
from jaxcad.constraints.types.angle import AngleConstraint

# Import all constraint types
from jaxcad.constraints.types.base import Constraint
from jaxcad.constraints.types.coincident import CoincidentConstraint
from jaxcad.constraints.types.distance import DistanceConstraint
from jaxcad.constraints.types.equal_length import EqualLengthConstraint
from jaxcad.constraints.types.fixed import FixedConstraint
from jaxcad.constraints.types.horizontal import HorizontalConstraint
from jaxcad.constraints.types.parallel import ParallelConstraint
from jaxcad.constraints.types.parallel_edges import ParallelEdgesConstraint
from jaxcad.constraints.types.perpendicular import PerpendicularConstraint
from jaxcad.constraints.types.perpendicular_edges import PerpendicularEdgesConstraint
from jaxcad.constraints.types.point_on_line import PointOnLineConstraint
from jaxcad.constraints.types.vertical import VerticalConstraint

# Re-export parameter types for convenience
from jaxcad.geometry.parameters import Parameter, Scalar, Vector, Vector2

# Convenience aliases (for backward compatibility with planned API)
Distance = DistanceConstraint
Angle = AngleConstraint
Parallel = ParallelConstraint
Perpendicular = PerpendicularConstraint
Fixed = FixedConstraint
Horizontal = HorizontalConstraint
Vertical = VerticalConstraint
Coincident = CoincidentConstraint
EqualLength = EqualLengthConstraint
PointOnLine = PointOnLineConstraint
ParallelEdges = ParallelEdgesConstraint
PerpendicularEdges = PerpendicularEdgesConstraint

# Type alias for Point (just a Vector)
Point = Vector

__all__ = [
    # Base class
    "Constraint",
    # Constraint types
    "DistanceConstraint",
    "AngleConstraint",
    "ParallelConstraint",
    "PerpendicularConstraint",
    "FixedConstraint",
    "HorizontalConstraint",
    "VerticalConstraint",
    "CoincidentConstraint",
    "EqualLengthConstraint",
    "PointOnLineConstraint",
    "ParallelEdgesConstraint",
    "PerpendicularEdgesConstraint",
    # DOF free functions
    "NullSpaceMap",
    "all_parameters",
    "total_dof_reduction",
    "build_residual_fn",
    "compute_param_vector",
    "unpack_param_vector",
    "null_space",
    # Solver
    "solve_constraints",
    "project_bounds",
    "project_to_manifold",
    "satisfy_constraints",
    "constraint_residuals",
    "make_bounds_projection",
    "make_manifold_projection",
    # Aliases
    "Distance",
    "Angle",
    "Parallel",
    "Perpendicular",
    "Fixed",
    "Horizontal",
    "Vertical",
    "Coincident",
    "EqualLength",
    "PointOnLine",
    "ParallelEdges",
    "PerpendicularEdges",
    # Re-exports
    "Parameter",
    "Scalar",
    "Vector",
    "Vector2",
    "Point",
]
