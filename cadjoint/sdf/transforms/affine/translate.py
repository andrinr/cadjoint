"""Translate transformation for SDFs."""

from __future__ import annotations

from jax import Array

from cadjoint.geometry.parameters import Vector
from cadjoint.sdf import SDF
from cadjoint.sdf.transforms.base import Transform


class Translate(Transform):
    """Translate an SDF by a vector offset.

    Note: For SDFs, we translate by applying the *inverse* transform to the
    query point. This moves the geometry in the opposite direction.

    Args:
        sdf: The SDF to translate
        offset: Translation vector (Array or Vector constraint)
    """

    def __init__(self, sdf: SDF, offset: Array | Vector):
        self.sdf = sdf
        self.params = {"offset": offset}

    @staticmethod
    def _transform_point(p: Array, offset: Array) -> Array:
        return p - offset

    @staticmethod
    def sdf(child_sdf, p: Array, offset: Array) -> Array:
        """Pure function for translation.

        Args:
            child_sdf: SDF function to translate
            p: Query point(s)
            offset: Translation vector [x, y, z]

        Returns:
            Translated SDF value
        """
        return child_sdf(Translate._transform_point(p, offset))

    def __call__(self, p: Array) -> Array:
        """Evaluate translated SDF."""
        return Translate.sdf(self.sdf, p, self.params["offset"].xyz)

    def material_at(self, p: Array) -> dict:
        return self.sdf.material_at(Translate._transform_point(p, self.params["offset"].xyz))

    def patch_fields(self):
        """Forward the child's patch fields through the inverse translation.

        Query points map into the child frame exactly as :meth:`sdf` does
        (``p - offset``), so patch ids and feature edges stay exact.
        """
        child_fields = self.sdf.patch_fields()
        if child_fields is None:
            return None
        offset = self.params["offset"].xyz
        return [
            (lambda p, f=field: f(Translate._transform_point(p, offset))) for field in child_fields
        ]

    def to_functional(self):
        """Return pure function for compilation."""
        return Translate.sdf
