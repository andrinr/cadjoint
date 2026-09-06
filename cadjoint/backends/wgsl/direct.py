"""A second WGSL path that never builds a jaxpr.

The shipped backend (:mod:`cadjoint.backends.wgsl.codegen`) reaches WGSL by
tracing the scene with JAX, exporting it to StableHLO, and walking that.  XLA
never produces an executable on that path — it is used as a tracer, and a
warm compile of ``scenes/motor_shield.py`` spends 37 % of its time on the
round trip (``research/performance.md`` §15).

This module walks the SDF object graph instead and writes WGSL from it
directly.  It exists to be *compared*, not to replace anything: the traced
path stays the source of truth for gradients, meshing and the constraint
system, and it stays the one whose shader is provably the same function the
CPU evaluates.

**The cost this buys speed with.** Every kernel below is a second
implementation of a distance function whose first implementation is the
primitive's ``sdf`` staticmethod.  Two implementations of the same maths
diverge unless something forces them not to, and this project's whole claim
is that the thing you see is the thing you differentiate.
:mod:`tests.backends.test_wgsl_direct` is that force: it executes both
shaders on the same points and requires agreement.  Do not add a kernel here
without adding it there.

Coverage is deliberately partial — the primitives, transforms and booleans
whose maths is a few lines.  A node this module does not know raises
:class:`UnsupportedNode`, which is how the comparison harness reports what a
scene would still need.  Profiles are the notable absence, and they are also
where the traced path's output balloons: a polygon's vertices become code
because WGSL cannot type an ``(N, 2)`` array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["DirectProgram", "UnsupportedNode", "compile_sdf_direct", "supported_nodes"]


class UnsupportedNode(Exception):
    """A node kind this backend has no kernel for."""


@dataclass(frozen=True)
class DirectProgram:
    """WGSL plus the profile vertices it reads.

    Attributes:
        wgsl: The module source.
        vertices: Every profile's vertices, concatenated, as ``(V, 2)``
            float32.  Empty when the scene has no profile.  A polygon kernel
            reads its own slice by the offset and count baked into its call,
            which is what keeps a profile's *code* the same size whether it
            has four vertices or four hundred.
    """

    wgsl: str
    vertices: np.ndarray

    def __str__(self) -> str:  # so callers can treat it as source
        return self.wgsl

    def __len__(self) -> int:
        return len(self.wgsl)


@dataclass(frozen=True)
class _Kernel:
    """One primitive's WGSL body.

    Attributes:
        params: The primitive's parameter names, in the order ``body`` reads
            them.  Each becomes a function argument.
        arity: How many floats each parameter occupies (1 or 3).
        body: WGSL statements ending in a ``return``, with ``p`` as the query
            point and the parameter names bound as arguments.
    """

    params: tuple[str, ...]
    arity: dict[str, int]
    body: str


# ── primitives ───────────────────────────────────────────────────────────────
#
# Each mirrors the ``sdf`` staticmethod of the class it is keyed by, line for
# line where WGSL allows it.  Where it does not, the deviation is commented:
# an unexplained deviation is a bug that the differential test will find but
# a reader should not have to.

_PRIMITIVES: dict[str, _Kernel] = {
    "Sphere": _Kernel(
        params=("radius",),
        arity={"radius": 1},
        body="    return length(p) - radius;",
    ),
    "Box": _Kernel(
        params=("size",),
        arity={"size": 3},
        # The Python branches on inside/outside rather than smoothing the
        # outside norm, so that a point on a face differentiates through the
        # inside branch. WGSL has no gradient to protect, but the *value* must
        # match, so the branch is kept rather than replaced by the usual
        # `length(max(q, 0)) + min(max_q, 0)` one-liner.
        body=(
            "    let q = abs(p) - size;\n"
            "    let max_q = max(q.x, max(q.y, q.z));\n"
            "    let clamped = max(q, vec3<f32>(0.0));\n"
            "    let squared = dot(clamped, clamped);\n"
            "    if (max_q <= 0.0) { return max_q; }\n"
            "    return sqrt(squared);"
        ),
    ),
    "Torus": _Kernel(
        params=("major_radius", "minor_radius"),
        arity={"major_radius": 1, "minor_radius": 1},
        body=(
            "    let q_xy = length(p.xy) - major_radius;\n"
            "    return length(vec2<f32>(q_xy, p.z)) - minor_radius;"
        ),
    ),
    "Cylinder": _Kernel(
        params=("radius", "height"),
        arity={"radius": 1, "height": 1},
        # The radial epsilon stays, as in the Python: its degenerate point is
        # the axis, which is interior and never on the surface. The
        # inside/outside branch is Box's, for the same reason.
        body=(
            "    let d_xy = sqrt(p.x * p.x + p.y * p.y + 1e-20) - radius;\n"
            "    let d_z = abs(p.z) - height;\n"
            "    let max_d = max(d_xy, d_z);\n"
            "    if (max_d <= 0.0) { return max_d; }\n"
            "    let a = max(d_xy, 0.0);\n"
            "    let b = max(d_z, 0.0);\n"
            "    return sqrt(a * a + b * b);"
        ),
    ),
    "Capsule": _Kernel(
        params=("radius", "height"),
        arity={"radius": 1, "height": 1},
        body=(
            "    let z = clamp(p.z, -height, height);\n"
            "    return length(vec3<f32>(p.x, p.y, p.z - z)) - radius;"
        ),
    ),
}

#: The profile buffer every polygon kernel reads, and the loop that reads it.
#:
#: This is the whole point of the exercise. The traced backend holds
#: `scalar_lowering` for its entire trace, which keeps each vertex a separate
#: 2-vector and *unrolls* the edge loop — so a sixteen-point profile emits
#: sixteen copies of the segment maths, per instance. WGSL cannot type an
#: `(N, 2)` function parameter, which is why that was the only option there;
#: it can perfectly well index a storage array, which is this.
#:
#: The arithmetic is `_polygon_distance_scalar` line for line, including the
#: pairwise-equality spelling of the even-odd crossing test.
_PROFILE_BUFFER = """@group(1) @binding(0) var<storage, read> profile_vertices: array<vec2<f32>>;"""

_PROFILE_DISTANCE = """fn profile_distance(p: vec2<f32>, offset: u32, count: u32) -> f32 {
    var d = dot(p - profile_vertices[offset], p - profile_vertices[offset]);
    var s = 1.0;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let j = (i + count - 1u) % count;
        let vi = profile_vertices[offset + i];
        let vj = profile_vertices[offset + j];
        let e = vj - vi;
        let w = p - vi;
        let t = clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        let b = w - e * t;
        d = min(d, dot(b, b));
        let c1 = p.y >= vi.y;
        let c2 = p.y < vj.y;
        let c3 = (e.x * w.y) > (e.y * w.x);
        if ((c1 == c2) && (c2 == c3)) { s = -s; }
    }
    return s * sqrt(d);
}"""

_EXTRUDED = """fn prim_extrudedpolygon(p: vec3<f32>, offset: u32, count: u32, depth: f32) -> f32 {
    let d2 = profile_distance(p.xy, offset, count);
    let dz = abs(p.z) - depth * 0.5;
    let max_d = max(d2, dz);
    if (max_d <= 0.0) { return max_d; }
    let a = max(d2, 0.0);
    let b = max(dz, 0.0);
    return sqrt(a * a + b * b);
}"""

_REVOLVED = """fn prim_revolvedpolygon(p: vec3<f32>, offset: u32, count: u32, radial: f32) -> f32 {
    let r = sqrt(p.x * p.x + p.z * p.z + 1e-20) - radial;
    return profile_distance(vec2<f32>(r, p.y), offset, count);
}"""

#: Rodrigues, with the angle a runtime value so a polar pattern can loop.
#: The Python folds its sine and cosine at trace time because every angle is
#: a static `i*2*pi/count`; here the loop index is the variable, so they are
#: computed per instance instead.
_ROTATE = """fn rotate_about(p: vec3<f32>, origin: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let v = p - origin;
    let c = cos(angle);
    let s = sin(angle);
    let parallel = axis * dot(v, axis);
    return origin + v * c + cross(axis, v) * s + parallel * (1.0 - c);
}"""

_SHARED: dict[str, str] = {}

# ── transforms ───────────────────────────────────────────────────────────────
#
# A transform is a map on the query point, so it needs no function of its own:
# the emitter binds a new point and calls the child with it.

_TRANSFORMS: dict[str, tuple[tuple[str, ...], dict[str, int], str]] = {
    "Translate": (("offset",), {"offset": 3}, "{p} - {offset}"),
    # Scale divides the point and multiplies the distance back, so it is the
    # one transform that also rewrites the *result*; see `_emit_node`.
    "Scale": (("factor",), {"factor": 1}, "{p} / {factor}"),
}

#: Booleans, as `(combiner, identity)` over child distances.
_BOOLEANS = frozenset({"Union", "Intersection", "Difference"})


_SHARED.update(
    {
        "profile_distance": _PROFILE_DISTANCE,
        "prim_extrudedpolygon": _EXTRUDED,
        "prim_revolvedpolygon": _REVOLVED,
        "rotate_about": _ROTATE,
        **{
            f"prim_{name.lower()}": (
                "fn prim_{sym}(p: vec3<f32>, {args}) -> f32 {{\n{body}\n}}".format(
                    sym=name.lower(),
                    args=", ".join(
                        f"{parameter}: {'f32' if kernel.arity[parameter] == 1 else 'vec3<f32>'}"
                        for parameter in kernel.params
                    ),
                    body=kernel.body,
                )
            )
            for name, kernel in _PRIMITIVES.items()
        },
    }
)


def supported_nodes() -> dict[str, tuple[str, ...]]:
    """What this backend can emit, for a harness to report against."""
    return {
        "primitives": tuple(sorted([*_PRIMITIVES, "ExtrudedPolygon", "RevolvedPolygon"])),
        "transforms": tuple(sorted([*_TRANSFORMS, "Rotate", "LinearPattern", "PolarPattern"])),
        "booleans": tuple(sorted(_BOOLEANS)),
    }


def _value(node: Any, name: str) -> np.ndarray:
    """A parameter's current value as a flat float array."""
    parameter = node.params[name]
    value = getattr(parameter, "value", parameter)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _literal(values: np.ndarray, arity: int) -> str:
    """A WGSL literal for a parameter value."""
    if arity == 1:
        return f"{float(values[0]):.9g}"
    parts = ", ".join(f"{float(component):.9g}" for component in values[:arity])
    return f"vec{arity}<f32>({parts})"


class _Emitter:
    """Emits one WGSL function per node, memoized on node identity.

    A function per node rather than one inlined expression, for two reasons
    the traced path cannot have. A subtree used twice is emitted once and
    called twice, where a trace under `scalar_lowering` emits it twice. And a
    pattern becomes a *loop* over its child's function — the traced form
    unrolls every instance, because a shader cannot carry the batch axis a
    `vmap` would give it.
    """

    def __init__(self) -> None:
        self.functions: list[str] = []
        self._kernels: set[str] = set()
        self._nodes: dict[int, str] = {}
        self.vertices: list[tuple[float, float]] = []

    def profile_slice(self, node: Any) -> tuple[int, int]:
        """Append a profile's vertices to the pool, returning `(offset, count)`."""
        names = sorted(
            (name for name in node.params if name.startswith("v") and name[1:].isdigit()),
            key=lambda name: int(name[1:]),
        )
        offset = len(self.vertices)
        for name in names:
            x, y = _value(node, name)[:2]
            self.vertices.append((float(x), float(y)))
        return offset, len(names)

    def _need(self, *symbols: str) -> None:
        """Emit a shared kernel body once per module."""
        for symbol in symbols:
            if symbol in self._kernels:
                continue
            self._kernels.add(symbol)
            self.functions.append(_SHARED[symbol])

    def function_for(self, node: Any) -> str:
        """The WGSL function evaluating *node*, emitted once per node."""
        key = id(node)
        if key in self._nodes:
            return self._nodes[key]
        symbol = f"node{len(self._nodes)}"
        # Reserve the name before recursing so a cycle would fail loudly
        # rather than looping; the graph is a DAG, but nothing enforces it.
        self._nodes[key] = symbol
        body = self._body(node)
        self.functions.append(f"fn {symbol}(p: vec3<f32>) -> f32 {{\n{body}\n}}")
        return symbol

    def _body(self, node: Any) -> str:
        name = type(node).__name__

        if name in _PRIMITIVES:
            kernel = _PRIMITIVES[name]
            self._need(f"prim_{name.lower()}")
            arguments = ", ".join(
                _literal(_value(node, parameter), kernel.arity[parameter])
                for parameter in kernel.params
            )
            return f"    return prim_{name.lower()}(p, {arguments});"

        if name in ("ExtrudedPolygon", "RevolvedPolygon"):
            return self._profile_body(node, name)

        if name == "Rotate":
            # The rotation matrix is built from concrete parameters, so it is
            # folded here exactly as the Python folds it at trace time, and
            # the shader multiplies by a constant matrix. `_transform_point`
            # applies the transpose, which for a rotation is the inverse:
            # the point moves into the child's frame, not the geometry.
            from cadjoint.sdf.transforms.affine.rotate import Rotate

            matrix = np.asarray(
                Rotate._rotation_matrix(_value(node, "axis"), float(_value(node, "angle")[0])),
                dtype=np.float64,
            ).T
            child = self.function_for(node.sdf)
            columns = ", ".join(
                f"vec3<f32>({matrix[0][c]:.9g}, {matrix[1][c]:.9g}, {matrix[2][c]:.9g})"
                for c in range(3)
            )
            # WGSL spells a matrix by columns, so `mat3x3(c0, c1, c2) * p`
            # computes the same product as the row-major `R.T @ p` above.
            return f"    return {child}(mat3x3<f32>({columns}) * p);"

        if name in _TRANSFORMS:
            names, arity, template = _TRANSFORMS[name]
            bindings = {
                parameter: _literal(_value(node, parameter), arity[parameter])
                for parameter in names
            }
            child = self.function_for(node.sdf)
            moved = template.format(p="p", **bindings)
            if name == "Scale":
                return f"    return {child}({moved}) * {bindings['factor']};"
            return f"    return {child}({moved});"

        if name in ("LinearPattern", "PolarPattern"):
            return self._pattern_body(node, name)

        if name in _BOOLEANS:
            return self._boolean_body(node, name)

        raise UnsupportedNode(f"{name} has no direct WGSL kernel. Supported: {supported_nodes()}")

    def _profile_body(self, node: Any, name: str) -> str:
        if name == "ExtrudedPolygon":
            # Draft and twist bend the walls; guessing a kernel for them is
            # exactly the silent divergence this backend must not introduce.
            for unsupported in ("draft", "twist"):
                if unsupported in node.params:
                    raise UnsupportedNode(
                        f"ExtrudedPolygon with {unsupported!r} has no direct kernel yet."
                    )
            self._need("profile_distance", "prim_extrudedpolygon")
            offset, count = self.profile_slice(node)
            depth = _literal(_value(node, "depth"), 1)
            return f"    return prim_extrudedpolygon(p, {offset}u, {count}u, {depth});"
        self._need("profile_distance", "prim_revolvedpolygon")
        offset, count = self.profile_slice(node)
        radial_offset = _literal(_value(node, "offset"), 1)
        return f"    return prim_revolvedpolygon(p, {offset}u, {count}u, {radial_offset});"

    def _pattern_body(self, node: Any, name: str) -> str:
        child = self.function_for(node.sdf)
        count = int(_value(node, "count")[0])
        mask = int(_value(node, "skip_mask")[0]) if "skip_mask" in node.params else 0
        kept = [i for i in range(count) if not mask >> i & 1]
        if len(kept) <= 1:
            return f"    return {child}(p);"
        # Instance 0 is the child at `p` itself, exactly as the Python has it:
        # for a polar pattern `origin + (p - origin)` equals `p` only up to
        # rounding, and copy 0 is the one face references are declared against.
        lines = [f"    var d = {child}(p);"]
        indices = ", ".join(f"{i}" for i in kept[1:])
        lines.append(f"    let instances = array<i32, {len(kept) - 1}>({indices});")
        lines.append(f"    for (var k = 0; k < {len(kept) - 1}; k = k + 1) {{")
        lines.append("        let i = f32(instances[k]);")
        if name == "LinearPattern":
            direction = _value(node, "direction")
            axis = direction / (np.linalg.norm(direction) or 1.0)
            spacing = float(_value(node, "spacing")[0])
            lines.append(f"        let q = p - {_literal(axis, 3)} * ({spacing:.9g} * i);")
        else:
            self._need("rotate_about")
            origin = _literal(_value(node, "origin"), 3)
            axis = _value(node, "direction")
            axis = axis / (np.linalg.norm(axis) or 1.0)
            lines.append(f"        let theta = -6.283185307179586 * i / {count:.9g};")
            lines.append(f"        let q = rotate_about(p, {origin}, {_literal(axis, 3)}, theta);")
        lines.append(f"        d = min(d, {child}(q));")
        lines.append("    }")
        lines.append("    return d;")
        return "\n".join(lines)

    def _boolean_body(self, node: Any, name: str) -> str:
        smoothness = float(_value(node, "smoothness")[0]) if "smoothness" in node.params else 0.0
        k = max(smoothness * 4.0, 1e-10)
        children = [self.function_for(child) for child in node.children()]
        lines = [f"    var d = {children[0]}(p);"]
        for child in children[1:]:
            operand = f"(-{child}(p))" if name == "Difference" else f"{child}(p)"
            combine = "min" if name == "Union" else "max"
            sign = "-" if name == "Union" else "+"
            lines.append(f"    let e = {operand};")
            lines.append(f"    let h = max({k:.9g} - abs(d - e), 0.0);")
            lines.append(f"    d = {combine}(d, e) {sign} h * h * 0.25 / {k:.9g};")
        lines.append("    return d;")
        # Each operand needs its own `let`, so re-bind per iteration.
        return "\n".join(_unique_lets(lines))


def _unique_lets(lines: list[str]) -> list[str]:
    """Give every `let e`/`let h` in a body a distinct name."""
    out, index = [], 0
    for line in lines:
        if line.strip().startswith("let e ="):
            index += 1
        out.append(
            line.replace(" e ", f" e{index} ")
            .replace(" h ", f" h{index} ")
            .replace("(d - e)", f"(d - e{index})")
            .replace("d, e)", f"d, e{index})")
        )
    return out


def compile_sdf_direct(geometry: Any, *, entry_point: str = "sdf") -> DirectProgram:
    """Emit WGSL for *geometry* by walking it, with no JAX in the path.

    Args:
        geometry: Root SDF node.
        entry_point: Name for the generated `point -> f32` function.

    Returns:
        A :class:`DirectProgram`: the module, and the profile vertices its
        polygon kernels read.

    Raises:
        UnsupportedNode: For a node kind this backend has no kernel for.
    """
    emitter = _Emitter()
    root = emitter.function_for(geometry)
    entry = f"fn {entry_point}(p: vec3<f32>) -> f32 {{\n    return {root}(p);\n}}"
    sections = list(emitter.functions)
    if emitter.vertices:
        sections.insert(0, _PROFILE_BUFFER)
    return DirectProgram(
        wgsl="\n\n".join([*sections, entry]),
        vertices=np.asarray(emitter.vertices, dtype=np.float32).reshape(-1, 2),
    )
