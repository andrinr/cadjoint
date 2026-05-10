"""Base class for primitive SDFs."""

from __future__ import annotations

from jaxcad.sdf import SDF


class Primitive(SDF):
    """Base class for primitive SDFs (leaf nodes in the SDF tree).

    Primitives are the basic building blocks - spheres, boxes, cylinders, etc.
    They have no SDF children, but may have a Fluent material child so that
    ``extract_parameters`` can discover material parameters automatically.

    Subclasses must implement:
    - @staticmethod def sdf(p: Array, **params) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable
    """

    def children(self) -> list:
        mat = getattr(self, "material", None)
        return [mat] if mat is not None else []
