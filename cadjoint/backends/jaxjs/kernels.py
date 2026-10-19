"""The jax-js kernels, one set per node type, traced from cadjoint's own ``sdf``.

Every node's ``sdf`` has one of three shapes, and each becomes kernels the
same way:

* **primitive** ``sdf(p, *params)``: one kernel, ``(p, *params) -> d``.
* **operation** ``sdf(child, p, *params)`` or ``sdf((a, b, ...), p, *params)``:
  traced with *recording* children.  A child that stores the point it is
  asked about yields the **query** kernel ``(p, *params) -> q``, the point the
  child is evaluated at (omitted when that is ``p`` itself).  A child that
  returns an input value yields the **combine** kernel ``(d..., *params) ->
  d``, what the node makes of its children's distances (omitted when it
  returns the child's value unchanged).  An n-ary node is traced with two
  children and folded left, as every cadjoint Boolean loops.
* **pattern** (``LinearPattern``, ``PolarPattern``): the query kernel takes
  one instance's parameter, so one kernel serves any count and skip mask;
  the combine is the two-instance ``min`` folded over the kept instances.

So the runtime (``web-spike/src/runtime.ts``) knows only *how to wire* a
query, a child and a combine.  The formulas themselves exist once, in the
Python this module traces.

Profiles (extruded, revolved, lofted polygons) take their vertices as one
``(N, 2)`` array, and ``N`` is fixed by the trace.  They are emitted at
:data:`POLYGON_BUCKETS` sizes; the runtime pads a profile to the next bucket
by repeating its last vertex.  A repeated vertex is an edge of length zero,
which the polygon distance already treats exactly (``_MIN_EDGE_SQUARED``): it
contributes the distance to a vertex the real edges reach anyway, and it
never flips the inside parity.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from cadjoint.sdf._lowering import scalar_lowering
from cadjoint.sdf.boolean.difference import Difference
from cadjoint.sdf.boolean.intersection import Intersection
from cadjoint.sdf.boolean.union import Union
from cadjoint.sdf.boolean.xor import Xor
from cadjoint.sdf.primitives.box import Box
from cadjoint.sdf.primitives.capsule import Capsule
from cadjoint.sdf.primitives.cylinder import Cylinder
from cadjoint.sdf.primitives.loft import LoftedPolygon
from cadjoint.sdf.primitives.plane import Plane
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon, RevolvedPolygon
from cadjoint.sdf.primitives.round_box import RoundBox
from cadjoint.sdf.primitives.sphere import Sphere
from cadjoint.sdf.primitives.torus import Torus
from cadjoint.sdf.transforms.affine.rotate import Rotate
from cadjoint.sdf.transforms.affine.scale import Scale
from cadjoint.sdf.transforms.affine.translate import Translate
from cadjoint.sdf.transforms.deformations.twist import Twist
from cadjoint.sdf.transforms.fields import Mirror, Offset, Shell
from cadjoint.sdf.transforms.patterns import LinearPattern, PolarPattern, _rotate_about_traced

from ._stablehlo import function_to_ts

__all__ = ["NODE_TYPES", "POLYGON_BUCKETS", "KernelSet", "NodeType", "kernel_sets"]

F32 = jnp.float32
POINT = jnp.zeros(3, F32)
SCALAR = jnp.zeros((), F32)

#: Profile sizes a polygon kernel is emitted for.  A profile of N vertices
#: runs the smallest bucket >= N; past the last one the runtime refuses it.
POLYGON_BUCKETS = (4, 8, 16, 32, 64)


@dataclass(frozen=True)
class NodeType:
    """One cadjoint node class, as the scene graph and the runtime see it.

    Attributes:
        name: The scene-graph ``type``: the class name in snake case.
        cls: The cadjoint class.
        kind: ``"primitive"``, ``"operation"`` or ``"pattern"``.
        params: Kernel parameter names in kernel order, as the scene graph
            stores them.  Profiles store one ``vertices`` (``(N, 2)``) entry
            (two for a loft) in place of ``v0 .. v{N-1}``.
        static: Parameters fixed at trace time, kept in the scene graph but
            not passed to kernels (a pattern's ``count`` and ``skip_mask``).
        nary: Whether the node takes a tuple of children.
    """

    name: str
    cls: type
    kind: str
    params: tuple[str, ...]
    static: tuple[str, ...] = ()
    nary: bool = False


@dataclass
class KernelSet:
    """The kernels one node type compiles to, by role, as TypeScript sources.

    ``fns`` maps a role (``"sdf"``, ``"query"``, ``"combine"``, or
    ``"sdf_<bucket>"`` for profiles) to the TS function name, and ``sources``
    holds those functions.
    """

    node: NodeType
    fns: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def add(self, role: str, fn, example_args) -> None:
        name = f"{self.node.name}__{role}"
        self.fns[role] = name
        self.sources.append(function_to_ts(name, fn, example_args))


def _signature_params(fn, skip: int) -> tuple[str, ...]:
    names = list(inspect.signature(fn).parameters)[skip:]
    return tuple(n for n in names if not n.startswith("*"))


def _primitive(cls, name: str) -> NodeType:
    return NodeType(name, cls, "primitive", _signature_params(cls.sdf, 1))


def _operation(cls, name: str, nary: bool = False) -> NodeType:
    return NodeType(name, cls, "operation", _signature_params(cls.sdf, 2), nary=nary)


#: Every node type the jax-js backend covers, by scene-graph name.
NODE_TYPES: dict[str, NodeType] = {
    t.name: t
    for t in [
        _primitive(Sphere, "sphere"),
        _primitive(Box, "box"),
        _primitive(RoundBox, "round_box"),
        _primitive(Cylinder, "cylinder"),
        _primitive(Capsule, "capsule"),
        _primitive(Torus, "torus"),
        _primitive(Plane, "plane"),
        NodeType(
            "extruded_polygon",
            ExtrudedPolygon,
            "primitive",
            ("depth", "draft", "twist", "vertices"),
        ),
        NodeType("revolved_polygon", RevolvedPolygon, "primitive", ("offset", "vertices")),
        NodeType(
            "lofted_polygon", LoftedPolygon, "primitive", ("height", "vertices", "vertices_b")
        ),
        _operation(Translate, "translate"),
        _operation(Rotate, "rotate"),
        _operation(Scale, "scale"),
        _operation(Twist, "twist"),
        _operation(Mirror, "mirror"),
        _operation(Offset, "offset"),
        _operation(Shell, "shell"),
        _operation(Union, "union", nary=True),
        _operation(Difference, "difference", nary=True),
        _operation(Intersection, "intersection", nary=True),
        _operation(Xor, "xor", nary=True),
        NodeType(
            "linear_pattern",
            LinearPattern,
            "pattern",
            ("direction", "spacing"),
            static=("count", "skip_mask"),
        ),
        NodeType(
            "polar_pattern",
            PolarPattern,
            "pattern",
            ("origin", "direction"),
            static=("count", "skip_mask"),
        ),
    ]
}

#: Example arguments for each parameter name, fixing shapes for the trace.
_EXAMPLES = {
    "size": jnp.ones(3, F32),
    "offset": jnp.zeros(3, F32),
    "axis": jnp.asarray([0.0, 0.0, 1.0], F32),
    "scale": jnp.ones(3, F32),
    "origin": jnp.zeros(3, F32),
    "normal": jnp.asarray([1.0, 0.0, 0.0], F32),
    "direction": jnp.asarray([1.0, 0.0, 0.0], F32),
}


def _example(node: NodeType, name: str):
    if node.cls is RevolvedPolygon and name == "offset":
        return SCALAR
    return _EXAMPLES.get(name, SCALAR)


def _is_identity(recorded, p) -> bool:
    return all(q is p for q in recorded)


def _operation_kernels(node: NodeType) -> KernelSet:
    """Query and combine kernels for an operation, by recording its children."""
    ks = KernelSet(node)
    examples = [_example(node, n) for n in node.params]
    arity = 2 if node.nary else 1

    def call(children, p, params):
        return node.cls.sdf(tuple(children) if node.nary else children[0], p, *params)

    # Query: what point each child is asked about.  Run eagerly once to see
    # whether that is p itself (the same object), so identity queries emit
    # nothing.
    # A child may be asked more than once (Scale evaluates both sides of a
    # where); every ask must then be the same computation, and the first
    # stands for all.
    def queries(p, *params):
        recorded = [[] for _ in range(arity)]
        children = [(lambda q, r=r: (r.append(q), jnp.zeros((), F32))[1]) for r in recorded]
        call(children, p, params)
        return [r[0] for r in recorded]

    recorded = queries(POINT, *examples)
    if not _is_identity(recorded, POINT):
        if node.nary:
            raise NotImplementedError(f"{node.cls.__name__}: n-ary node with a non-identity query")
        ks.add("query", lambda p, *params: queries(p, *params)[0], [POINT, *examples])

    # Combine: what the node makes of its children's distances.  p is a
    # constant here; the check below proves the combine never reads it.
    def combine(*args):
        ds, params = args[:arity], args[arity:]
        children = [(lambda _q, d=d: d) for d in ds]
        return call(children, POINT, params)

    marker = [jnp.full((), 0.123, F32) + i for i in range(arity)]
    out = combine(*marker, *examples)
    if not (arity == 1 and out is marker[0]):
        _check_combine_ignores_p(node, call, arity, examples)
        ks.add("combine", combine, [*([SCALAR] * arity), *examples])
    return ks


def _check_combine_ignores_p(node: NodeType, call, arity: int, examples) -> None:
    def with_p(p, *ds_params):
        ds, params = ds_params[:arity], ds_params[arity:]
        return call([(lambda _q, d=d: d) for d in ds], p, params)

    from jax._src.interpreters.partial_eval import dce_jaxpr

    jaxpr = jax.make_jaxpr(with_p)(POINT, *([SCALAR] * arity), *examples).jaxpr
    # Dead code first: Scale computes p / scale for its child and, with the
    # child replaced by a value, never uses it.
    _, used_inputs = dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars))
    if used_inputs[0]:
        raise NotImplementedError(f"{node.cls.__name__}: combine reads the query point")


def _pattern_kernels(node: NodeType) -> KernelSet:
    """A pattern's per-instance query and its two-instance combine.

    ``LinearPattern``'s instance ``i`` is its two-copy form traced with the
    spacing scaled by ``i`` (the runtime passes ``spacing * i``, rounded in
    f32 exactly as cadjoint's ``spacing * kept`` is).  ``PolarPattern``'s is
    cadjoint's ``_rotate_about_traced`` at the instance angle the runtime
    computes as cadjoint does, ``-2π·i / count``.
    """
    ks = KernelSet(node)
    if node.cls is LinearPattern:

        def query(p, direction, scaled_spacing):
            recorded = []

            def child(q):
                recorded.append(q)
                return jnp.zeros((), F32)

            with scalar_lowering():
                LinearPattern.sdf(child, p, direction, scaled_spacing, 2, 0)
            return recorded[1]

        ks.add("query", query, [POINT, _EXAMPLES["direction"], SCALAR])

        def combine(a, b, direction, spacing):
            values = iter([a, b])
            with scalar_lowering():
                return LinearPattern.sdf(lambda _q: next(values), POINT, direction, spacing, 2, 0)

        ks.add(
            "combine",
            lambda a, b: combine(a, b, _EXAMPLES["direction"], SCALAR + 1.0),
            [SCALAR, SCALAR],
        )
    else:
        ks.add("query", _rotate_about_traced, [POINT, POINT, _EXAMPLES["direction"], SCALAR])

        def combine(a, b):
            values = iter([a, b])
            with scalar_lowering():
                return PolarPattern.sdf(
                    lambda _q: next(values), POINT, 2, POINT, _EXAMPLES["direction"], 0
                )

        ks.add("combine", combine, [SCALAR, SCALAR])
    return ks


def _profile(verts):
    return {f"v{i}": verts[i] for i in range(verts.shape[0])}


def _polygon_kernels(node: NodeType) -> KernelSet:
    ks = KernelSet(node)
    for n in POLYGON_BUCKETS:
        v = jnp.zeros((n, 2), F32)
        if node.cls is ExtrudedPolygon:
            # draft and twist traced, so the general form is emitted: with
            # both zero it equals the plain form exactly (cos 0 = 1, sin 0 = 0).
            def fn(p, depth, draft, twist, verts):
                return ExtrudedPolygon.sdf(p, depth, draft=draft, twist=twist, **_profile(verts))

            ks.add(f"sdf_{n}", fn, [POINT, SCALAR, SCALAR, SCALAR, v])
        elif node.cls is RevolvedPolygon:
            ks.add(
                f"sdf_{n}",
                lambda p, offset, verts: RevolvedPolygon.sdf(p, offset, **_profile(verts)),
                [POINT, SCALAR, v],
            )
        else:

            def loft(p, height, a, b):
                profiles = {**_profile(a), **{f"w{i}": b[i] for i in range(b.shape[0])}}
                return LoftedPolygon.sdf(p, height, **profiles)

            ks.add(f"sdf_{n}", loft, [POINT, SCALAR, v, v])
    return ks


def kernel_sets(names=None) -> list[KernelSet]:
    """Trace and translate every node type's kernels.

    Args:
        names: Scene-graph type names to emit; all of :data:`NODE_TYPES` when None.

    Returns:
        One :class:`KernelSet` per node type, in :data:`NODE_TYPES` order.
    """
    out = []
    for name, node in NODE_TYPES.items():
        if names is not None and name not in names:
            continue
        if node.cls in (ExtrudedPolygon, RevolvedPolygon, LoftedPolygon):
            out.append(_polygon_kernels(node))
        elif node.kind == "primitive":
            ks = KernelSet(node)
            ks.add("sdf", node.cls.sdf, [POINT, *[_example(node, n) for n in node.params]])
            out.append(ks)
        elif node.kind == "pattern":
            out.append(_pattern_kernels(node))
        else:
            out.append(_operation_kernels(node))
    return out
