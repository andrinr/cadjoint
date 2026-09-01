"""Base class for boolean operation SDFs."""

from __future__ import annotations

from cadjoint.sdf import SDF


class BooleanOp(SDF):
    """Base class for boolean operation SDFs.

    Boolean operations combine one or more SDFs - union, intersection, difference, etc.

    Subclasses must implement:
    - @staticmethod def sdf(child_sdfs: tuple, p: Array, **params) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable

    Subclasses should store:
    - self.sdfs: Tuple of child SDFs
    - self.params: Dictionary of Parameter objects
    """

    def children(self) -> list:
        return list(self.sdfs)
