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

from cadjoint.fluent import Fluent

if TYPE_CHECKING:
    from cadjoint.geometry.parameters import Parameter

# The face-reference names a construction feature binds onto the SDF it
# generated. Kept here as data so the SDF layer can forward them without
# importing anything from cadjoint.construction — the dependency runs the
# other way, and must keep running the other way.
_FACE_REFERENCES = frozenset({"faces", "face", "cap", "side", "axis"})


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

    # Set to True on nodes that leave their FIRST child's surface where it was,
    # so that child's analytic face references still land on this node's own
    # surface. See __getattr__ below; the flag is opt-in because most nodes
    # move the surface and a forwarded face would then quietly be wrong.
    inherits_faces: bool = False

    params: dict[str, Parameter]

    def __getattr__(self, name: str):
        """Forward a base child's face references through face-preserving nodes.

        :func:`cadjoint.construction.faces.attach_faces` binds ``faces``,
        ``face``, ``cap``, ``side`` and ``axis`` onto the SDF a generator
        returns, so ``extrude(profile, depth).cap("+")`` is a plain instance
        attribute and never reaches this method. What *did* reach it before
        was every downstream node: ``Difference(body, hole).cap("+")`` raised
        ``AttributeError``, so the moment a body had a hole in it, it stopped
        being something you could sketch on — which is most of feature-based
        CAD.

        Booleans and patterns do not move the base body's surface: cutting a
        hole in a plate leaves the plate's top face in the same plane, and
        copy 0 of a pattern *is* the original. Those nodes set
        :attr:`inherits_faces` and forward, so a face reference survives the
        rest of the feature tree. Nodes that displace the surface — every
        affine transform, ``Shell``, ``Offset`` — deliberately do not: their
        child's face plane is no longer on their own surface, and returning
        it would be a silent lie rather than a loud ``AttributeError``.

        The forwarded face keeps the *plane* exact — that is what a downstream
        sketch, mirror or pattern actually consumes. Its boundary polygon is
        the uncut outline, so a face a tool has since carved into reports the
        area it had before the cut; :meth:`Face.contains` is correspondingly
        optimistic there. This mirrors how a B-rep modeller keeps a face's
        identity across the features applied to it.

        Args:
            name: The attribute Python failed to find by normal lookup.

        Returns:
            The base child's attribute, for the five face-reference names on a
                node that declares :attr:`inherits_faces`.

        Raises:
            AttributeError: For every other name, and when no base child
                carries the reference — with the node's own type in the
                message, so the failure still reads as this node's.
        """
        # type(self) — not self — so a node still inside __init__ (no children
        # yet) cannot recurse back into this lookup.
        if name in _FACE_REFERENCES and type(self).inherits_faces:
            children = self.children()
            if children:
                try:
                    return getattr(children[0], name)
                except AttributeError:
                    pass
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

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
        from cadjoint.sdf.boolean import Union

        return Union((self, other))

    def __add__(self, other: SDF) -> SDF:
        """Union operator: self + other"""
        from cadjoint.sdf.boolean import Union

        return Union((self, other))

    def __and__(self, other: SDF) -> SDF:
        """Intersection operator: self & other"""
        from cadjoint.sdf.boolean import Intersection

        return Intersection((self, other))

    def __sub__(self, other: SDF) -> SDF:
        """Difference operator: self - other"""
        from cadjoint.sdf.boolean import Difference

        return Difference((self, other))

    def __xor__(self, other: SDF) -> SDF:
        """Xor operator: self ^ other"""
        from cadjoint.sdf.boolean import Xor

        return Xor((self, other))

    def patch_fields(self) -> list[Callable[[Array], Array]] | None:
        """Return this node's smooth patch fields in its own frame, if known.

        A patch field decomposition expresses the node's surface as pieces of
        the zero sets of smooth scalar fields ``f_i`` — the node's SDF is
        internally a ``min``/``max`` composition over them (Box: six face
        half-space distances; Cylinder: side plus two caps).  A surface point
        ``p`` belongs to patch ``argmin_i |f_i(p)|``, and the node's exact
        feature edges are exactly where that ownership switches.

        Transforms forward the protocol by mapping the query point into the
        child frame exactly as their ``sdf`` does.  Nodes without an exact
        decomposition return ``None`` (the default), which consumers treat
        gracefully as "one opaque patch".

        Returns:
            List of callables ``f_i(p)`` accepting points shaped ``(..., 3)``
            in this node's frame and returning values shaped ``(...)``, or
            ``None`` when no exact decomposition is available.
        """
        return None

    def material_at(self, _p: Array) -> dict:
        """Return material properties at point p.

        Default implementation returns a white, matte, opaque material.
        Subclasses override this to return per-primitive or blended materials.

        Args:
            p: Query point, shape (3,).

        Returns:
            Dict with keys 'color', 'roughness', 'metallic', 'opacity'.
        """
        from cadjoint.render.material import Material

        return Material().as_dict()
