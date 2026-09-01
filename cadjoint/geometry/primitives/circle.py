"""Circle geometric entity.

A Circle is a parametric geometric entity defined by a center point, radius,
and normal vector defining the plane orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from cadjoint.geometry.parameters import Scalar, Vector, as_parameter


@dataclass
class Circle:
    """Parametric circle in 3D space.

    A Circle is a first-class geometric entity that can be used to:
    - Define planar constraints and relationships
    - Construct SDF primitives (e.g., Cylinder.from_circle(circle, height))
    - Serve as a profile for operations (e.g., revolve, sweep)
    - Render as wireframe/curve in 3D viewers

    Args:
        center: Center point (Vector or 3D array)
        radius: Circle radius (Scalar or float)
        normal: Normal vector defining the plane orientation (Vector or 3D array, default: [0, 0, 1])

    The circle lies in a plane perpendicular to the normal vector.

    Example:
        # Circle in XY plane
        circle = Circle(center=[0, 0, 0], radius=1.0)

        # Tilted circle with free parameters
        center = Vector([0, 0, 0], free=True, name='center')
        radius = Scalar(1.5, free=True, name='radius')
        normal = Vector([1, 1, 1], free=False)  # direction vector
        circle = Circle(center=center, radius=radius, normal=normal)

        # Sample points on the circle
        point = circle.sample(0.0)  # Point at angle=0
        point = circle.sample(jnp.pi)  # Point at angle=π
    """

    center: Vector
    radius: Scalar
    normal: Vector

    def __post_init__(self) -> None:
        """Convert raw values to parameter types and compute local frame."""
        self.center = as_parameter(self.center)
        self.radius = as_parameter(self.radius)
        self.normal = as_parameter(self.normal).normalize()

        self._compute_local_frame()

    def _compute_local_frame(self) -> None:
        """Compute orthonormal local frame (U, V, N) from normal vector.

        U: first tangent direction
        V: second tangent direction (perpendicular to U)
        N: normal direction
        """
        # Find a vector not parallel to normal
        up = jnp.array([0.0, 0.0, 1.0])
        if jnp.abs(jnp.dot(self.normal.xyz, up)) > 0.99:
            up = jnp.array([1.0, 0.0, 0.0])

        # Compute U (first tangent) axis
        u = jnp.cross(up, self.normal.xyz)
        u = u / jnp.linalg.norm(u)

        # Store as Vector parameter (direction, w=0)
        self.u_axis = as_parameter(u)  # Direction vector

        # Compute V (second tangent) axis
        v = jnp.cross(self.normal.xyz, u)
        self.v_axis = as_parameter(v).normalize()

    def sample(self, theta: float | Array) -> Array:
        """Sample point on the circle at angle theta.

        Args:
            theta: Angle in radians, where:
                   - theta=0 corresponds to center + radius * u_axis
                   - theta increases counter-clockwise when viewed along normal

        Returns:
            3D point on the circle
        """
        theta = jnp.asarray(theta)

        u_vec = self.u_axis.xyz
        v_vec = self.v_axis.xyz

        # Parametric circle: center + radius * (cos(θ) * u + sin(θ) * v)
        return self.center.xyz + self.radius.value * (
            jnp.cos(theta) * u_vec + jnp.sin(theta) * v_vec
        )

    def sample_uniform(self, n_points: int) -> Array:
        """Sample n points uniformly distributed around the circle.

        Args:
            n_points: Number of points to sample

        Returns:
            Array of shape (n_points, 3) with points on circle
        """
        theta = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
        return jax.vmap(self.sample)(theta)

    def tangent(self, theta: float | Array) -> Array:
        """Get the tangent vector at angle theta.

        Args:
            theta: Angle in radians

        Returns:
            Normalized tangent vector (perpendicular to radius)
        """
        theta = jnp.asarray(theta)

        u_vec = self.u_axis.xyz
        v_vec = self.v_axis.xyz

        # Tangent: d/dθ [cos(θ) * u + sin(θ) * v] = -sin(θ) * u + cos(θ) * v
        tangent = -jnp.sin(theta) * u_vec + jnp.cos(theta) * v_vec
        return tangent / jnp.linalg.norm(tangent)

    def area(self) -> float:
        """Get the area of the circle.

        Returns:
            Area: π * r²
        """
        return jnp.pi * self.radius.value**2

    def circumference(self) -> float:
        """Get the circumference of the circle.

        Returns:
            Circumference: 2 * π * r
        """
        return 2 * jnp.pi * self.radius.value

    def closest_point(self, p: Array) -> Array:
        """Find the closest point on the circle to a given point.

        Args:
            p: Query point [x, y, z]

        Returns:
            Closest point on the circle
        """
        p = jnp.asarray(p)

        # Project point onto circle plane
        to_point = p - self.center.xyz
        distance_along_normal = jnp.dot(to_point, self.normal.xyz)
        projected = p - distance_along_normal * self.normal.xyz

        # Vector from center to projected point
        radial = projected - self.center.xyz
        radial_length = jnp.linalg.norm(radial)

        # If at center, pick arbitrary point on circle
        if radial_length < 1e-8:
            return self.sample(0.0)

        # Project onto circle
        radial_normalized = radial / radial_length
        return self.center.xyz + self.radius.value * radial_normalized

    def distance_to_point(self, p: Array) -> float:
        """Compute distance from a point to the circle.

        Args:
            p: Query point [x, y, z]

        Returns:
            Minimum distance to the circle
        """
        closest = self.closest_point(p)
        return jnp.linalg.norm(p - closest)

    def __repr__(self) -> str:
        return f"Circle(center={self.center.xyz}, radius={self.radius.value}, normal={self.normal.xyz})"
