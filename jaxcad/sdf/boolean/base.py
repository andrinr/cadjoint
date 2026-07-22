"""Base class for boolean operation SDFs."""

from __future__ import annotations

from jaxcad.sdf import SDF


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

    @property
    def is_exact(self) -> bool:
        """Whether every child is an exact signed-distance function."""
        return all(child.is_exact for child in self.sdfs)

    @staticmethod
    def _validate_children(
        sdfs, operation: str, *, min_count: int = 1, exact_count: int | None = None
    ) -> tuple:
        """Validate and normalize the children of a boolean operation."""
        sdfs = tuple(sdfs)
        expected = exact_count if exact_count is not None else min_count
        valid_count = len(sdfs) == expected if exact_count is not None else len(sdfs) >= min_count
        if not valid_count:
            qualifier = "exactly" if exact_count is not None else "at least"
            raise ValueError(f"{operation} requires {qualifier} {expected} SDF operands.")
        if not all(isinstance(sdf, SDF) for sdf in sdfs):
            raise TypeError(f"{operation} operands must all be SDF instances.")
        return sdfs

    def children(self) -> list:
        return list(self.sdfs)
