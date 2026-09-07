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
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cadjoint.zeroset.table import Model

__all__ = ["emit"]

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
