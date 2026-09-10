"""Evaluate a node table with JAX: the host's oracle.

Everything here is a fold over the table, so it is what any walker in any
language does; being JAX, it is also differentiable in the design and in
the point, which is all the implicit-function derivative needs.

``field(model)`` returns a jittable ``(theta, points) -> values``.
``surfaces(model)`` returns the census: every surface's own field in the
model's frame, in the order ``PatchId`` indexes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from cadjoint.zeroset.table import Model, node_kind

__all__ = ["census", "field", "surfaces"]

Field = Callable[[jax.Array, jax.Array], jax.Array]  #: (theta [P], points [N, 3]) -> values [N]

_UNARY = {
    "NEG": jnp.negative,
    "ABS": jnp.abs,
    "SIGN": jnp.sign,
    # Guarded like the GPU fold's: the derivative at 0 is finite (0) rather
    # than the infinity a bare sqrt would hand the design derivative.
    "SQRT": lambda a: jnp.sqrt(jnp.maximum(a, 1e-300)),
    "EXP": jnp.exp,
    "LOG": jnp.log,
    "SIN": jnp.sin,
    "COS": jnp.cos,
    "TAN": jnp.tan,
}
_BINARY = {
    "ADD": jnp.add,
    "SUB": jnp.subtract,
    "MUL": jnp.multiply,
    "DIV": jnp.divide,
    "MIN": jnp.minimum,
    "MAX": jnp.maximum,
    "POW": jnp.power,
    "ATAN2": jnp.arctan2,
}
_AXIS = {"X": 0, "Y": 1, "Z": 2}


class _Walker:
    """One evaluation of the table at a batch of points.

    Shape values are memoised per (node, points); expressions per (node,
    child values); both for the life of one walk.
    """

    def __init__(self, nodes: list[dict[str, Any]], theta: jax.Array) -> None:
        self.nodes, self.theta = nodes, theta
        self._shapes: dict[tuple[int, int], jax.Array] = {}
        self._alive: list[jax.Array] = []  # keyed arrays must outlive the memo, or ids recur

    def expr(
        self, i: int, pts: jax.Array, children: list[jax.Array] | None, memo: dict
    ) -> jax.Array:
        if i in memo:
            return memo[i]
        node = self.nodes[i]
        kind = node_kind(node)
        if kind == "lit":
            v = jnp.full(pts.shape[0], node["lit"])
        elif kind == "coord":
            v = pts[:, _AXIS[node["coord"]]]
        elif kind == "param":
            v = jnp.full(pts.shape[0], self.theta[node["param"]])
        elif kind == "childValue":
            if children is None:
                raise ValueError("`childValue` outside a blend rule")
            v = children[node["childValue"]]
        elif kind == "unary":
            v = _UNARY[node["unary"]["op"]](self.expr(node["unary"]["a"], pts, children, memo))
        elif kind == "binary":
            b = node["binary"]
            v = _BINARY[b["op"]](
                self.expr(b["a"], pts, children, memo), self.expr(b["b"], pts, children, memo)
            )
        else:
            raise ValueError(f"node {i} ({kind}) is not an expression")
        memo[i] = v
        return v

    def image(self, warp: dict[str, Any], pts: jax.Array) -> tuple[jax.Array, jax.Array]:
        memo: dict = {}
        q = jnp.stack(
            [self.expr(warp[k], pts, None, memo) for k in ("imageX", "imageY", "imageZ")], axis=1
        )
        return q, self.expr(warp["scale"], pts, None, memo)

    def shape(self, i: int, pts: jax.Array) -> jax.Array:
        self._alive.append(pts)
        key = (i, id(pts))
        if key in self._shapes:
            return self._shapes[key]
        node = self.nodes[i]
        kind = node_kind(node)
        if kind == "patch":
            v = self.expr(node["patch"], pts, None, {})
        elif kind == "warp":
            q, scale = self.image(node["warp"], pts)
            v = scale * self.shape(node["warp"]["child"], q)
        elif kind == "combine":
            c = node["combine"]
            vals = [self.shape(ch, pts) for ch in c["children"]]
            if "blend" in c:
                v = self.expr(c["blend"]["rule"], pts, vals, {})
            elif c["owning"] == "COMPLEMENT":
                v = -vals[0]
            else:
                op = jnp.minimum if c["owning"] == "MIN" else jnp.maximum
                v = vals[0]
                for b in vals[1:]:
                    v = op(v, b)
        else:
            raise ValueError(f"node {i} ({kind}) is not a shape")
        self._shapes[key] = v
        return v


def field(model: Model) -> Field:
    """The model's field as a function of the design and a batch of points."""
    nodes, root = model.nodes, model.root

    def f(theta: jax.Array, pts: jax.Array) -> jax.Array:
        return _Walker(nodes, theta).shape(root, jnp.asarray(pts))

    return f


def census(model: Model) -> list[tuple[str, list[int], int | tuple[int, int]]]:
    """The surfaces in ``PatchId`` order: ``(kind, path, what)``.

    ``kind`` is ``patch``, ``band``, ``rim`` or ``fold``; ``path`` the warp
    nodes above it, outermost first; ``what`` the patch's expression, the
    band's combine node, or ``(combine node, expression)`` for a rim or fold.
    """
    nodes = model.nodes
    out: list[tuple[str, list[int], Any]] = []

    def go(i: int, warps: list[int]) -> None:
        node = nodes[i]
        kind = node_kind(node)
        if kind == "patch":
            out.append(("patch", warps, node["patch"]))
        elif kind == "warp":
            go(node["warp"]["child"], [*warps, i])
        else:
            c = node["combine"]
            if "blend" in c:
                out.append(("band", warps, i))
                if "transition" in c["blend"]:
                    out.append(("rim", warps, (i, c["blend"]["transition"])))
                for aux in c["blend"].get("auxiliary", []):
                    out.append(("fold", warps, (i, aux)))
            for ch in c["children"]:
                go(ch, warps)

    go(model.root, [])
    return out


def surfaces(model: Model) -> list[tuple[str, Field]]:
    """Every surface's own field in the model's frame, in ``PatchId`` order.

    A patch's field is its expression at the point carried through the
    warps above it, scaled as they scale; a band's is the blend node's
    value; a rim's or fold's is its expression over the blend's children.
    """
    nodes = model.nodes

    def through(warps: list[int], leaf: Callable[[_Walker, jax.Array], jax.Array]) -> Field:
        def f(theta: jax.Array, pts: jax.Array) -> jax.Array:
            w = _Walker(nodes, theta)
            q, scale = jnp.asarray(pts), jnp.ones(len(pts))
            for i in warps:
                q2, s = w.image(nodes[i]["warp"], q)
                scale, q = scale * s, q2
            return scale * leaf(w, q)

        return f

    out: list[tuple[str, Field]] = []
    for kind, warps, what in census(model):
        if kind == "patch":
            out.append((kind, through(warps, lambda w, q, e=what: w.expr(e, q, None, {}))))
        elif kind == "band":
            out.append((kind, through(warps, lambda w, q, i=what: w.shape(i, q))))
        else:
            i, e = what

            def over_children(w: _Walker, q: jax.Array, i: int = i, e: int = e) -> jax.Array:
                vals = [w.shape(ch, q) for ch in nodes[i]["combine"]["children"]]
                return w.expr(e, q, vals, {})

            out.append((kind, through(warps, over_children)))
    return out
