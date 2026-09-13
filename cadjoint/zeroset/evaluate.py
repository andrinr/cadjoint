"""Evaluate a node table with JAX: the host's oracle.

Everything here is a fold over the table, so it is what any walker in any
language does; being JAX, it is also differentiable in the design and in
the point, which is all the implicit-function derivative needs.

``field(model)`` returns a jittable ``(theta, points) -> values``.
``surfaces(model)`` returns the census: every surface's own field in the
model's frame, in the order ``PatchId`` indexes.
``gathered(model, wanted)`` returns several of those surfaces at once, from
*one* walk, which is what keeps a program that needs many of them small.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from cadjoint.zeroset.table import Model, node_kind

__all__ = ["census", "field", "gathered", "surfaces"]

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


def _batched(v: Any, n: int) -> jax.Array:
    """``v`` as an ``(n,)`` array — a no-op unless it came out per-walk constant."""
    if jnp.ndim(v) == 0:
        return jnp.broadcast_to(jnp.asarray(v), (n,))
    return v


class _Walker:
    """One evaluation of the table at a batch of points.

    Shape values are memoised per (node, points); expressions per (node,
    child values); both for the life of one walk.

    **A subexpression that does not read a coordinate stays a scalar**, and
    is only broadcast to the batch where a batched value is actually needed
    — at a warp's image, at a shape, at a census surface.  A sixth of a
    real part's expression nodes are point-independent (the arithmetic that
    turns parameters into a transform), and materialising each of them as a
    full column used to cost a ``broadcast_in_dim`` *and*, because a Python
    float is weakly typed and the points are not, a no-op
    ``convert_element_type``: on ``scenes/motor_shield.py`` those two
    primitives alone were 6 100 of one walk's 20 217 equations.  Same
    arithmetic, same last bit — a broadcast and a scalar multiply differ
    only in how much of it XLA has to schedule, and XLA's compile cost is
    superlinear in that (``research/performance.md`` §18).
    """

    def __init__(self, nodes: list[dict[str, Any]], theta: jax.Array) -> None:
        self.nodes, self.theta = nodes, theta
        self._shapes: dict[tuple[int, int], jax.Array] = {}
        self._params: dict[int, jax.Array] = {}  # one read of theta per parameter, not per use
        self._alive: list[jax.Array] = []  # keyed arrays must outlive the memo, or ids recur

    def expr(
        self, i: int, pts: jax.Array, children: list[jax.Array] | None, memo: dict
    ) -> jax.Array:
        if i in memo:
            return memo[i]
        node = self.nodes[i]
        kind = node_kind(node)
        if kind == "lit":
            v = node["lit"]  # a weakly typed scalar: it promotes as the column it met did
        elif kind == "coord":
            v = pts[:, _AXIS[node["coord"]]]
        elif kind == "param":
            p = node["param"]
            if p not in self._params:
                self._params[p] = self.theta[p]
            v = self._params[p]
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
        n = pts.shape[0]
        q = jnp.stack(
            [
                _batched(self.expr(warp[k], pts, None, memo), n)
                for k in ("imageX", "imageY", "imageZ")
            ],
            axis=1,
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
            v = _batched(self.expr(node["patch"], pts, None, {}), pts.shape[0])
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


def _leaf(
    nodes: list[dict[str, Any]], kind: str, what: Any
) -> Callable[[_Walker, jax.Array], jax.Array]:
    """A census surface's own value at the point the warps above it carry it to.

    The one definition of what a surface *is*, shared by :func:`surfaces`
    and :func:`gathered` so the two cannot answer differently.
    """
    if kind == "patch":
        return lambda w, q: w.expr(what, q, None, {})
    if kind == "band":
        return lambda w, q: w.shape(what, q)
    i, e = what

    def over_children(w: _Walker, q: jax.Array) -> jax.Array:
        vals = [w.shape(ch, q) for ch in nodes[i]["combine"]["children"]]
        return w.expr(e, q, vals, {})

    return over_children


def surfaces(model: Model) -> list[tuple[str, Field]]:
    """Every surface's own field in the model's frame, in ``PatchId`` order.

    A patch's field is its expression at the point carried through the
    warps above it, scaled as they scale; a band's is the blend node's
    value; a rim's or fold's is its expression over the blend's children.

    One callable per surface, so a caller that wants many of them walks the
    table many times.  :func:`gathered` is the same values from one walk.
    """
    nodes = model.nodes

    def through(warps: list[int], leaf: Callable[[_Walker, jax.Array], jax.Array]) -> Field:
        def f(theta: jax.Array, pts: jax.Array) -> jax.Array:
            w = _Walker(nodes, theta)
            q, scale = jnp.asarray(pts), 1.0
            for i in warps:
                q2, s = w.image(nodes[i]["warp"], q)
                scale, q = scale * s, q2
            return _batched(scale * leaf(w, q), len(pts))

        return f

    return [(kind, through(warps, _leaf(nodes, kind, what))) for kind, warps, what in census(model)]


def _span(nodes: list[dict[str, Any]], i: int, memo: dict[int, int]) -> int:
    """How many census surfaces the subtree at ``i`` contributes.

    A count, not a list: it depends on the node alone and not on the warps
    above it, so it memoises, and it is what lets :func:`gathered` skip a
    subtree holding nothing it was asked for.
    """
    if i in memo:
        return memo[i]
    node = nodes[i]
    kind = node_kind(node)
    if kind == "patch":
        n = 1
    elif kind == "warp":
        n = _span(nodes, node["warp"]["child"], memo)
    else:
        c = node["combine"]
        n = 0
        if "blend" in c:
            n += 1 + ("transition" in c["blend"]) + len(c["blend"].get("auxiliary", []))
        n += sum(_span(nodes, ch, memo) for ch in c["children"])
    memo[i] = n
    return n


def gathered(model: Model, wanted: Sequence[int]) -> Field:
    """``(theta, points) -> (N, len(wanted))``: those census surfaces, from one walk.

    The same numbers :func:`surfaces` gives one surface at a time — to the
    last bit, because the scale is accumulated down the warp chain in the
    same order and the leaves are the same :func:`_leaf` — but the table is
    descended *once* for all of them, sharing every node two surfaces have
    in common.  A band and the rim beside it overlap almost entirely, and
    every patch under a warp shares that warp's image, so on a real part
    this is the difference between a program the size of the table and one
    the size of the surface list times the table.

    Only the subtrees that hold something in ``wanted`` are descended into,
    so asking for a few cheap patches of a large tree stays cheap.

    Args:
        model: A :class:`~cadjoint.zeroset.table.Model`.
        wanted: Census (``PatchId``) indices, in the column order to return.
    """
    nodes, root = model.nodes, model.root
    column = {int(s): k for k, s in enumerate(wanted)}
    if len(column) != len(wanted):
        raise ValueError("a surface may be asked for once")
    ordered = sorted(column)  # ascending PatchIds, to skip by
    spans: dict[int, int] = {}

    def f(theta: jax.Array, pts: jax.Array) -> jax.Array:
        w = _Walker(nodes, theta)
        pts = jnp.asarray(pts)
        out: list[Any] = [None] * len(column)
        alive = [pts]  # the memo is keyed on id(): these must outlive the walk
        seen = [0]  # census surfaces passed so far, i.e. the next PatchId
        ahead = [0]  # how many of `ordered` are behind `seen`

        def surface(kind: str, what: Any, q: jax.Array, scale: jax.Array) -> None:
            k = column.get(seen[0])
            seen[0] += 1
            if k is not None:
                ahead[0] += 1
                out[k] = _batched(scale * _leaf(nodes, kind, what)(w, q), len(pts))

        def go(i: int, q: jax.Array, scale: jax.Array) -> None:
            span = _span(nodes, i, spans)
            if ahead[0] >= len(ordered) or ordered[ahead[0]] >= seen[0] + span:
                seen[0] += span  # nothing wanted down there
                return
            node = nodes[i]
            kind = node_kind(node)
            if kind == "patch":
                surface("patch", node["patch"], q, scale)
            elif kind == "warp":
                image, factor = w.image(node["warp"], q)
                alive.append(image)
                go(node["warp"]["child"], image, scale * factor)
            else:
                c = node["combine"]
                if "blend" in c:
                    surface("band", i, q, scale)
                    if "transition" in c["blend"]:
                        surface("rim", (i, c["blend"]["transition"]), q, scale)
                    for aux in c["blend"].get("auxiliary", []):
                        surface("fold", (i, aux), q, scale)
                for ch in c["children"]:
                    go(ch, q, scale)

        go(root, pts, 1.0)
        return jnp.stack(out, axis=-1)

    return f
