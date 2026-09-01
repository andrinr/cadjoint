"""Base class for transform SDFs."""

from __future__ import annotations

from cadjoint.sdf import SDF


class Transform(SDF):
    """Base class for transform SDFs (unary operations in the SDF tree).

    Transforms modify a child SDF - translate, rotate, scale, twist, etc.
    They have one child.

    Subclasses must implement:
    - @staticmethod def sdf(child_sdf, p: Array, **params) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable

    Subclasses should store:
    - self.sdf: The child SDF being transformed
    - self.params: Dictionary of Parameter objects
    """

    def children(self) -> list:
        return [self.sdf]
