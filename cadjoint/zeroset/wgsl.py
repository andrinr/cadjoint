"""Emit a node table as WGSL: a fold with one case per form.

The output declares ``fn sdf(p: vec3<f32>) -> f32``, the model's field, and
one function per shape node it reaches, ``fn s<i>(p: vec3<f32>) -> f32``.
Expressions are emitted as ``let`` bindings inside the shape function that
uses them, in table order, which is already topological.  Nothing here
knows what a box or a fillet is; the vocabulary was the frontend's.

The design enters either as literals baked into the module (``theta=None``
is not allowed; pass the values) or, with ``theta_binding``, as a storage
buffer ``array<f32>`` read by index, so a parameter change is a buffer
write and not a recompile.

This is the extraction backend's fold, not a fourth viewer shader form.
The viewer's module also carries materials (``material_base``,
``material_optics``), which are the viewer's concern and not the zero
set's: the table stays material-free so the protocol stays about surfaces,
and the viewer keeps :mod:`cadjoint.backends.wgsl.direct`, already its
fast default.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cadjoint.zeroset.table import Model

__all__ = ["emit", "emit_dual"]

_UNARY = {
    "NEG": "-({a})",
    "ABS": "abs({a})",
    "SIGN": "sign({a})",
    "SQRT": "sqrt(max({a}, 0.0))",
    "EXP": "exp({a})",
    "LOG": "log({a})",
    "SIN": "sin({a})",
    "COS": "cos({a})",
    "TAN": "tan({a})",
}
_BINARY = {
    "ADD": "({a} + {b})",
    "SUB": "({a} - {b})",
    "MUL": "({a} * {b})",
    "DIV": "({a} / {b})",
    "MIN": "min({a}, {b})",
    "MAX": "max({a}, {b})",
    "POW": "pow({a}, {b})",
    "ATAN2": "atan2({a}, {b})",
}
_AXIS = {"X": "p.x", "Y": "p.y", "Z": "p.z"}


def _f32(v: float) -> str:
    s = repr(float(v))
    if "e" in s or "E" in s:
        s = f"{float(v):.9e}"
    return s if "." in s or "e" in s else s + ".0"


def _kind(node: dict[str, Any]) -> str:
    return next(iter(node))


class _Emitter:
    def __init__(self, model: Model, theta_binding: tuple[int, int] | None) -> None:
        self.nodes, self.theta, self.binding = model.nodes, model.theta, theta_binding
        self.functions: list[str] = []
        self.emitted: set[int] = set()

    # -- expressions inside one shape function ------------------------------

    def _reachable(self, roots: Iterable[int]) -> list[int]:
        """Expression nodes under *roots*, in table (topological) order."""
        seen: set[int] = set()
        stack = list(roots)
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            node = self.nodes[i]
            kind = _kind(node)
            if kind == "unary":
                stack.append(node["unary"]["a"])
            elif kind == "binary":
                stack.extend((node["binary"]["a"], node["binary"]["b"]))
        return sorted(seen)

    def _expr_lets(self, roots: Iterable[int], children: list[str] | None) -> list[str]:
        lines = []
        for i in self._reachable(roots):
            node = self.nodes[i]
            kind = _kind(node)
            if kind == "lit":
                rhs = _f32(node["lit"])
            elif kind == "coord":
                rhs = _AXIS[node["coord"]]
            elif kind == "param":
                k = node["param"]
                rhs = f"theta[{k}u]" if self.binding else _f32(self.theta[k])
            elif kind == "childValue":
                if children is None:
                    raise ValueError("`childValue` outside a blend rule")
                rhs = children[node["childValue"]]
            elif kind == "unary":
                rhs = _UNARY[node["unary"]["op"]].format(a=f"e{node['unary']['a']}")
            elif kind == "binary":
                b = node["binary"]
                rhs = _BINARY[b["op"]].format(a=f"e{b['a']}", b=f"e{b['b']}")
            else:
                raise ValueError(f"node {i} ({kind}) is not an expression")
            lines.append(f"    let e{i}: f32 = {rhs};")
        return lines

    # -- shapes, one function each ----------------------------------------------

    def shape(self, i: int) -> str:
        name = f"s{i}"
        if i in self.emitted:
            return name
        self.emitted.add(i)
        node = self.nodes[i]
        kind = _kind(node)
        body: list[str]
        if kind == "patch":
            e = node["patch"]
            body = [*self._expr_lets([e], None), f"    return e{e};"]
        elif kind == "warp":
            w = node["warp"]
            child = self.shape(w["child"])
            roots = [w["imageX"], w["imageY"], w["imageZ"], w["scale"]]
            body = [
                *self._expr_lets(roots, None),
                f"    let q = vec3<f32>(e{w['imageX']}, e{w['imageY']}, e{w['imageZ']});",
                f"    return e{w['scale']} * {child}(q);",
            ]
        elif kind == "combine":
            c = node["combine"]
            kids = [self.shape(ch) for ch in c["children"]]
            calls = [f"    let c{k}: f32 = {fn}(p);" for k, fn in enumerate(kids)]
            if "blend" in c:
                rule = c["blend"]["rule"]
                body = [
                    *calls,
                    *self._expr_lets([rule], [f"c{k}" for k in range(len(kids))]),
                    f"    return e{rule};",
                ]
            elif c["owning"] == "COMPLEMENT":
                body = [*calls, "    return -c0;"]
            else:
                op = "min" if c["owning"] == "MIN" else "max"
                acc = "c0"
                for k in range(1, len(kids)):
                    acc = f"{op}({acc}, c{k})"
                body = [*calls, f"    return {acc};"]
        else:
            raise ValueError(f"node {i} ({kind}) is not a shape")
        self.functions.append(f"fn {name}(p: vec3<f32>) -> f32 {{\n" + "\n".join(body) + "\n}")
        return name


def emit(model: Model, *, theta_binding: tuple[int, int] | None = None) -> str:
    """WGSL for the model's field.

    Args:
        model: The table.
        theta_binding: ``(group, binding)`` of a read-only storage buffer
            ``array<f32>`` holding θ; omitted, θ is baked in as literals.

    Returns:
        A module declaring ``fn sdf(p: vec3<f32>) -> f32``.
    """
    em = _Emitter(model, theta_binding)
    root = em.shape(model.root)
    head = ["// zeroset.v1 model: one function per shape node, in table order"]
    if theta_binding is not None:
        group, binding = theta_binding
        head.append(f"@group({group}) @binding({binding}) var<storage, read> theta: array<f32>;")
    return "\n".join(
        [*head, "", *em.functions, "", f"fn sdf(p: vec3<f32>) -> f32 {{ return {root}(p); }}", ""]
    )


# ------------------------------------------------------------ dual numbers
#
# The same fold with a different carrier: every expression is a vec4<f32>,
# (value, ∂/∂x, ∂/∂y, ∂/∂z), so a surface returns its field and gradient in
# one call.  That is what a projection solve needs, and it is the only
# reason a backend on the GPU needs any derivative at all: the derivative
# in the design stays the host's.

_DUAL_UNARY = {
    "NEG": "-{a}",
    "ABS": "sign({a}.x) * {a}",
    "SIGN": "vec4<f32>(sign({a}.x), 0.0, 0.0, 0.0)",
    "SQRT": "zs_sqrt({a})",
    "EXP": "vec4<f32>(exp({a}.x), exp({a}.x) * {a}.yzw)",
    "LOG": "vec4<f32>(log({a}.x), {a}.yzw / {a}.x)",
    "SIN": "vec4<f32>(sin({a}.x), cos({a}.x) * {a}.yzw)",
    "COS": "vec4<f32>(cos({a}.x), -sin({a}.x) * {a}.yzw)",
    "TAN": "zs_tan({a})",
}
_DUAL_BINARY = {
    "ADD": "({a} + {b})",
    "SUB": "({a} - {b})",
    "MUL": "vec4<f32>({a}.x * {b}.x, {a}.x * {b}.yzw + {b}.x * {a}.yzw)",
    "DIV": "zs_div({a}, {b})",
    "MIN": "select({b}, {a}, {a}.x <= {b}.x)",
    "MAX": "select({b}, {a}, {a}.x >= {b}.x)",
    "POW": "zs_pow({a}, {b})",
    "ATAN2": "zs_atan2({a}, {b})",
}
_DUAL_AXIS = {
    "X": "vec4<f32>(p.x, 1.0, 0.0, 0.0)",
    "Y": "vec4<f32>(p.y, 0.0, 1.0, 0.0)",
    "Z": "vec4<f32>(p.z, 0.0, 0.0, 1.0)",
}

_DUAL_HELPERS = """
fn zs_sqrt(a: vec4<f32>) -> vec4<f32> {
    let s = sqrt(max(a.x, 0.0));
    return vec4<f32>(s, select(a.yzw / (2.0 * s), vec3<f32>(0.0), s <= 0.0));
}
fn zs_tan(a: vec4<f32>) -> vec4<f32> {
    let t = tan(a.x);
    return vec4<f32>(t, (1.0 + t * t) * a.yzw);
}
fn zs_div(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let inv = 1.0 / b.x;
    let f = a.x * inv;
    return vec4<f32>(f, (a.yzw - f * b.yzw) * inv);
}
fn zs_pow(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let f = pow(a.x, b.x);
    return vec4<f32>(f, b.x * pow(a.x, b.x - 1.0) * a.yzw + f * log(max(a.x, 1e-30)) * b.yzw);
}
fn zs_atan2(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let r2 = a.x * a.x + b.x * b.x;
    return vec4<f32>(atan2(a.x, b.x), (b.x * a.yzw - a.x * b.yzw) / r2);
}
"""


class _DualEmitter(_Emitter):
    """Shape and surface functions returning ``vec4<f32>``: value and gradient."""

    def _expr_lets(self, roots: Iterable[int], children: list[str] | None) -> list[str]:
        lines = []
        for i in self._reachable(roots):
            node = self.nodes[i]
            kind = _kind(node)
            if kind == "lit":
                rhs = f"vec4<f32>({_f32(node['lit'])}, 0.0, 0.0, 0.0)"
            elif kind == "coord":
                rhs = _DUAL_AXIS[node["coord"]]
            elif kind == "param":
                k = node["param"]
                value = f"theta[{k}u]" if self.binding else _f32(self.theta[k])
                rhs = f"vec4<f32>({value}, 0.0, 0.0, 0.0)"
            elif kind == "childValue":
                if children is None:
                    raise ValueError("`childValue` outside a blend rule")
                rhs = children[node["childValue"]]
            elif kind == "unary":
                rhs = _DUAL_UNARY[node["unary"]["op"]].format(a=f"e{node['unary']['a']}")
            elif kind == "binary":
                b = node["binary"]
                rhs = _DUAL_BINARY[b["op"]].format(a=f"e{b['a']}", b=f"e{b['b']}")
            else:
                raise ValueError(f"node {i} ({kind}) is not an expression")
            lines.append(f"    let e{i}: vec4<f32> = {rhs};")
        return lines

    @staticmethod
    def _warp_lines(w: dict[str, Any], child: str, lets: list[str]) -> list[str]:
        """Map the point, call the child at it, and chain: ∂f/∂p = (∂q/∂p)ᵀ ∂f/∂q."""
        ix, iy, iz, sc = w["imageX"], w["imageY"], w["imageZ"], w["scale"]
        return [
            *lets,
            f"    let q = vec3<f32>(e{ix}.x, e{iy}.x, e{iz}.x);",
            f"    let inner = {child}(q);",
            f"    let grad = inner.y * e{ix}.yzw + inner.z * e{iy}.yzw + inner.w * e{iz}.yzw;",
            f"    return vec4<f32>(e{sc}.x * inner.x, e{sc}.x * grad + inner.x * e{sc}.yzw);",
        ]

    def shape(self, i: int) -> str:
        name = f"d{i}"
        if i in self.emitted:
            return name
        self.emitted.add(i)
        node = self.nodes[i]
        kind = _kind(node)
        body: list[str]
        if kind == "patch":
            e = node["patch"]
            body = [*self._expr_lets([e], None), f"    return e{e};"]
        elif kind == "warp":
            w = node["warp"]
            child = self.shape(w["child"])
            body = self._warp_lines(
                w, child, self._expr_lets([w["imageX"], w["imageY"], w["imageZ"], w["scale"]], None)
            )
        elif kind == "combine":
            c = node["combine"]
            kids = [self.shape(ch) for ch in c["children"]]
            calls = [f"    let c{k}: vec4<f32> = {fn}(p);" for k, fn in enumerate(kids)]
            if "blend" in c:
                rule = c["blend"]["rule"]
                body = [
                    *calls,
                    *self._expr_lets([rule], [f"c{k}" for k in range(len(kids))]),
                    f"    return e{rule};",
                ]
            elif c["owning"] == "COMPLEMENT":
                body = [*calls, "    return -c0;"]
            else:
                pick = "<=" if c["owning"] == "MIN" else ">="
                acc = "c0"
                for k in range(1, len(kids)):
                    acc = f"select(c{k}, {acc}, {acc}.x {pick} c{k}.x)"
                body = [*calls, f"    return {acc};"]
        else:
            raise ValueError(f"node {i} ({kind}) is not a shape")
        self.functions.append(
            f"fn {name}(p: vec3<f32>) -> vec4<f32> {{\n" + "\n".join(body) + "\n}"
        )
        return name

    def surface(self, k: int, kind: str, warps: list[int], what: Any) -> str:
        """A census entry as ``fn surf<k>(p) -> vec4``: its own field, through its warps."""
        # the leaf, in the frame of the innermost warp
        if kind == "patch":
            leaf_body = [*self._expr_lets([what], None), f"    return e{what};"]
        elif kind == "band":
            leaf_body = [f"    return {self.shape(what)}(p);"]
        else:  # rim or fold: the expression over the blend's children
            i, e = what
            c = self.nodes[i]["combine"]
            kids = [self.shape(ch) for ch in c["children"]]
            calls = [f"    let c{j}: vec4<f32> = {fn}(p);" for j, fn in enumerate(kids)]
            leaf_body = [
                *calls,
                *self._expr_lets([e], [f"c{j}" for j in range(len(kids))]),
                f"    return e{e};",
            ]
        name = f"surf{k}_{len(warps)}"
        self.functions.append(
            f"fn {name}(p: vec3<f32>) -> vec4<f32> {{\n" + "\n".join(leaf_body) + "\n}"
        )
        # each warp above it, innermost first, wraps the function below
        for depth, wi in reversed(list(enumerate(warps))):
            w = self.nodes[wi]["warp"]
            inner, name = name, f"surf{k}_{depth}"
            lines = self._warp_lines(
                w, inner, self._expr_lets([w["imageX"], w["imageY"], w["imageZ"], w["scale"]], None)
            )
            self.functions.append(
                f"fn {name}(p: vec3<f32>) -> vec4<f32> {{\n" + "\n".join(lines) + "\n}"
            )
        return name


def emit_dual(model: Model, *, theta_binding: tuple[int, int] | None = None) -> str:
    """WGSL where every field returns ``vec4<f32>``: value and gradient.

    Declares ``fn field(p) -> vec4<f32>`` for the model and
    ``fn surface(id: u32, p) -> vec4<f32>`` dispatching over the census in
    ``PatchId`` order, which is what a projection solve iterates on.
    """
    from cadjoint.zeroset.evaluate import census

    em = _DualEmitter(model, theta_binding)
    root = em.shape(model.root)
    names = [
        em.surface(k, kind, warps, what) for k, (kind, warps, what) in enumerate(census(model))
    ]
    cases = "\n".join(f"        case {k}u: {{ return {n}(p); }}" for k, n in enumerate(names))
    dispatch = (
        "fn surface(id: u32, p: vec3<f32>) -> vec4<f32> {\n    switch id {\n"
        + cases
        + "\n        default: { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }\n    }\n}"
    )
    head = ["// zeroset.v1 model, dual: (value, ∂x, ∂y, ∂z) per node"]
    if theta_binding is not None:
        group, binding = theta_binding
        head.append(f"@group({group}) @binding({binding}) var<storage, read> theta: array<f32>;")
    return "\n".join(
        [
            *head,
            _DUAL_HELPERS,
            *em.functions,
            "",
            dispatch,
            "",
            f"fn field(p: vec3<f32>) -> vec4<f32> {{ return {root}(p); }}",
            "",
        ]
    )
