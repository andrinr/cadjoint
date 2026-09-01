"""Line geometric entity.

A Line is a parametric geometric entity defined by two points in 3D space.
It can be used to define spatial relationships, construct SDFs, or as a path
for operations like Repeat.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from cadjoint.geometry.parameters import Vector


@dataclass
class Line:
    """Parametric line defined by start and end points.

    A Line is a first-class geometric entity that can be used to:
    - Define spatial relationships and constraints
    - Construct oriented primitives (e.g., Box.from_line(line, thickness))
    - Serve as a path for spatial operations (e.g., Repeat along line)
    - Render as wireframe in 3D viewers

    Args:
        start: Start point (Vector or 3D array)
        end: End point (Vector or 3D array)

    Example:
        # Fixed line
        line = Line(start=[0, 0, 0], end=[1, 0, 0])

        # Line with free parameters for optimization
        p1 = Vector([0, 0, 0], free=True, name='start')
        p2 = Vector([1, 0, 0], free=True, name='end')
        line = Line(start=p1, end=p2)

        # Query line properties
        mid = line.midpoint()
        length = line.length()
        direction = line.direction()

        # Sample along line
        point_at_half = line.sample(0.5)  # t in [0, 1]
    """

    start: Vector
    end: Vector

    def __post_init__(self):
        """Convert raw arrays to Vector parameters if needed."""
        from cadjoint.geometry.parameters import as_parameter

        self.start = as_parameter(self.start)
        self.end = as_parameter(self.end)

    def sample(self, t: float | Array) -> Array:
        """Sample point along the line at parameter t.

        Args:
            t: Parameter value(s) in [0, 1] where:
               - t=0 returns start point
               - t=1 returns end point
               - 0 < t < 1 interpolates between start and end

        Returns:
            3D point(s) along the line
        """
        t = jnp.asarray(t)
        return (1 - t) * self.start.xyz + t * self.end.xyz

    def direction(self, normalized: bool = True) -> Array:
        """Get the direction vector from start to end.

        Args:
            normalized: If True, return unit vector. If False, return raw direction.

        Returns:
            Direction vector [x, y, z]
        """
        direction = self.end.xyz - self.start.xyz
        if normalized:
            return direction / jnp.linalg.norm(direction)
        return direction

    def length(self) -> float:
        """Get the length of the line segment.

        Returns:
            Euclidean distance between start and end points
        """
        return jnp.linalg.norm(self.end.xyz - self.start.xyz)

    def midpoint(self) -> Array:
        """Get the midpoint of the line.

        Returns:
            3D point at the center of the line
        """
        return (self.start.xyz + self.end.xyz) / 2

    def tangent(self, _t: float | Array) -> Array:
        """Get the tangent vector at parameter t.

        For a line, the tangent is constant everywhere.

        Args:
            t: Parameter value (unused for lines, kept for API consistency)

        Returns:
            Normalized tangent vector [x, y, z]
        """
        return self.direction(normalized=True)

    def closest_point(self, p: Array) -> Array:
        """Find the closest point on the line to a given point.

        Args:
            p: Query point [x, y, z]

        Returns:
            Closest point on the line segment
        """
        p = jnp.asarray(p)
        start = self.start.xyz
        end = self.end.xyz

        # Vector from start to end
        line_vec = end - start
        line_len_sq = jnp.dot(line_vec, line_vec)

        # Handle degenerate case (start == end)
        if line_len_sq < 1e-10:
            return start

        # Parameter t for closest point
        t = jnp.dot(p - start, line_vec) / line_len_sq
        t = jnp.clip(t, 0.0, 1.0)

        return self.sample(t)

    def distance_to_point(self, p: Array) -> float:
        """Compute distance from a point to the line segment.

        Args:
            p: Query point [x, y, z]

        Returns:
            Minimum distance to the line segment
        """
        closest = self.closest_point(p)
        return jnp.linalg.norm(p - closest)

    def __repr__(self) -> str:
        return f"Line(start={self.start.xyz}, end={self.end.xyz})"
