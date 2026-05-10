"""Base SDF (Signed Distance Function) class.

Architecture:
- Pure functions are the source of truth for SDF evaluation
- SDF classes are thin wrappers providing fluent API (method chaining, operators)
- Compilation unwraps classes to pure functions for JAX tracing
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable

from jax import Array

from jaxcad.fluent import Fluent

if TYPE_CHECKING:
    from jaxcad.geometry.parameters import Parameter


class SDF(Fluent):
    """Abstract base class for Signed Distance Functions.

    An SDF represents geometry implicitly as a function f(p) -> distance,
    where:
    - f(p) < 0: point p is inside the shape
    - f(p) = 0: point p is on the surface
    - f(p) > 0: point p is outside the shape

    Attributes:
        params: Dictionary of Parameter objects for this SDF operation

    SDF classes are thin wrappers around pure functions:
    - Provide fluent API: .translate().rotate() chaining
    - Enable operators: sphere | box, sphere & box
    - Store parameters for later compilation
    - Actual computation happens in pure functions

    Each SDF instance should store its parameters in self.params dictionary:
        self.params = {
            'radius': Scalar(value=1.0, free=True, name='radius'),
            'offset': Vector(value=[0, 0, 0], free=False),
        }

    Subclasses must implement:
    - @staticmethod def sdf(...): Pure function for computation (CONVENTION)
    - __call__(p): Evaluate SDF (delegates to sdf())
    - to_functional(): Return the static sdf method

    Pattern for primitives:
        @staticmethod
        def sdf(p: Array, param1: float, param2: float) -> Array:
            # Pure computation here

        def __call__(self, p: Array) -> Array:
            return ClassName.sdf(p, self.params['param1'].value, self.params['param2'].value)

        def to_functional(self):
            return ClassName.sdf

    Pattern for transforms:
        @staticmethod
        def sdf(child_sdf: Callable, p: Array, param: float) -> Array:
            # Transform computation here

        def __call__(self, p: Array) -> Array:
            return ClassName.sdf(self.child_sdf, p, self.params['param'].value)

        def to_functional(self):
            return ClassName.sdf
    """

    # Set to False on approximate SDFs (e.g. non-isometric deformations like
    # Twist) so the renderer uses ao=1 instead of the unreliable gradient norm.
    is_exact: bool = True

    params: dict[str, Parameter]

    def __init_subclass__(cls, **kwargs):
        """Automatically wrap __init__ to call _cast_params after initialization."""
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Auto-cast params after initialization
            if hasattr(self, "params"):
                self._cast_params()

        cls.__init__ = new_init

    @abstractmethod
    def __call__(self, p: Array) -> Array:
        """Evaluate the signed distance at point(s) p.

        This is for direct use only. During compilation, this is replaced
        by the pure function from to_functional().

        Args:
            p: Point(s) to evaluate, shape (..., 3) for 3D or (..., 2) for 2D

        Returns:
            Signed distance value(s), shape (...)
        """
        pass

    @abstractmethod
    def to_functional(self) -> Callable:
        """Return pure function for JAX tracing and compilation.

        Returns:
            Pure function with signature: (p: Array, **params) -> Array
            where params are the primitive/transform parameters.

        Example:
            ```python
            sphere = Sphere(radius=1.0)
            func = sphere.to_functional()
            ```
        """
        pass

    def __or__(self, other: SDF) -> SDF:
        """Union operator: self | other"""
        from jaxcad.sdf.boolean import Union

        return Union((self, other))

    def __add__(self, other: SDF) -> SDF:
        """Union operator: self + other"""
        from jaxcad.sdf.boolean import Union

        return Union((self, other))

    def __and__(self, other: SDF) -> SDF:
        """Intersection operator: self & other"""
        from jaxcad.sdf.boolean import Intersection

        return Intersection((self, other))

    def __sub__(self, other: SDF) -> SDF:
        """Difference operator: self - other"""
        from jaxcad.sdf.boolean import Difference

        return Difference((self, other))

    def __xor__(self, other: SDF) -> SDF:
        """Xor operator: self ^ other"""
        from jaxcad.sdf.boolean import Xor

        return Xor((self, other))

    def material_at(self, _p: Array) -> dict:
        """Return material properties at point p.

        Default implementation returns a white, matte, opaque material.
        Subclasses override this to return per-primitive or blended materials.

        Args:
            p: Query point, shape (3,).

        Returns:
            Dict with keys 'color', 'roughness', 'metallic', 'opacity'.
        """
        from jaxcad.render.material import Material

        return Material().as_dict()
