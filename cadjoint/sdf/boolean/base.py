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

    # A boolean leaves its base operand's surface where it was — cutting a
    # hole in a plate does not move the plate's top face — so the base's
    # analytic face references still land on the result and are forwarded by
    # SDF.__getattr__. Two caveats live in that method's docstring: the
    # boundary polygon is the *uncut* outline, and a smoothness > 0 blend
    # rounds the surface near the seam while leaving it exact away from it.
    inherits_faces = True

    def children(self) -> list:
        return list(self.sdfs)
