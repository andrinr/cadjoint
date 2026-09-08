"""Implicit models in the zero-set grammar, evaluated in JAX.

A model is the flat node table of ``research/zero-set.proto`` in its proto
JSON form: expressions (``lit``, ``coord``, ``param``, ``childValue``,
``unary``, ``binary``) and shapes (``patch``, ``warp``, ``combine``), with
children before parents and a parameter vector θ. This module walks that
table with ``jax.numpy`` so that ``jax.grad`` runs through the field in
both the point and the parameters, and lowers three small models by hand
in the same conventions as ``research/zeroset_test/lower.py``.

The field f(p, θ) is negative inside the model and zero on its boundary.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Model = dict[str, Any]

_TINY = 1e-300  # keeps sqrt's gradient finite exactly at zero, where it is not needed


def _safe_sqrt(v):
    return jnp.sqrt(jnp.maximum(v, _TINY))


_UNARY: dict[str, Callable] = {
    "NEG": lambda a: -a,
    "ABS": jnp.abs,
    "SIGN": jnp.sign,
    "SQRT": _safe_sqrt,
    "EXP": jnp.exp,
    "LOG": jnp.log,
    "SIN": jnp.sin,
    "COS": jnp.cos,
    "TAN": jnp.tan,
}

_BINARY: dict[str, Callable] = {
    "ADD": lambda a, b: a + b,
    "SUB": lambda a, b: a - b,
    "MUL": lambda a, b: a * b,
    "DIV": lambda a, b: a / b,
    "MIN": jnp.minimum,
    "MAX": jnp.maximum,
    "POW": jnp.power,
    "ATAN2": jnp.arctan2,
}


def evaluate(model: Model, theta, p):
    """The model's field at one point.

    Args:
        model: A node table in proto JSON form, with ``node`` and ``root``.
        theta: The parameter vector, indexed by ``param`` nodes.
        p: One point, a 3-vector; a 2D model reads only its first two entries.

    Returns:
        The scalar field value, negative inside.
    """
    nodes = model["node"]

    def expr(i, point, children):
        n = nodes[i]
        if "lit" in n:
            return jnp.asarray(n["lit"], dtype=jnp.result_type(float))
        if "coord" in n:
            return point["XYZ".index(n["coord"])]
        if "param" in n:
            return theta[n["param"]]
        if "childValue" in n:
            if children is None:
                raise ValueError("`childValue` outside a blend rule")
            return children[n["childValue"]]
        if "unary" in n:
            u = n["unary"]
            return _UNARY[u["op"]](expr(u["a"], point, children))
        if "binary" in n:
            b = n["binary"]
            return _BINARY[b["op"]](expr(b["a"], point, children), expr(b["b"], point, children))
        raise ValueError(f"node {i} is not an expression")

    def shape(i, point):
        n = nodes[i]
        if "patch" in n:
            return expr(n["patch"], point, None)
        if "warp" in n:
            w = n["warp"]
            image = jnp.stack([expr(w[k], point, None) for k in ("imageX", "imageY", "imageZ")])
            return expr(w["scale"], point, None) * shape(w["child"], image)
        if "combine" in n:
            c = n["combine"]
            values = [shape(k, point) for k in c["children"]]
            if "blend" in c:
                return expr(c["blend"]["rule"], point, values)
            rule = c["owning"]
            if rule == "COMPLEMENT":
                return -values[0]
            op = jnp.minimum if rule == "MIN" else jnp.maximum
            out = values[0]
            for v in values[1:]:
                out = op(out, v)
            return out
        raise ValueError(f"node {i} is not a shape")

    return shape(model["root"], jnp.asarray(p))


def field(model: Model) -> Callable:
    """The model as ``f(theta, points)`` over a batch of points of any dimension ≤ 3."""

    def f(theta, points):
        points = jnp.asarray(points)
        pad = 3 - points.shape[-1]
        if pad:
            points = jnp.concatenate([points, jnp.zeros(points.shape[:-1] + (pad,))], axis=-1)
        return jax.vmap(lambda q: evaluate(model, theta, q))(points.reshape(-1, 3)).reshape(
            points.shape[:-1]
        )

    return f


def spatial_gradient(model: Model) -> Callable:
    """``grad_p f(theta, points)``, returned in the dimension of the points given."""

    def g(theta, points):
        points = jnp.asarray(points)
        dim = points.shape[-1]
        pad = 3 - dim
        if pad:
            points = jnp.concatenate([points, jnp.zeros(points.shape[:-1] + (pad,))], axis=-1)
        grad = jax.vmap(jax.grad(lambda q: evaluate(model, theta, q)))(points.reshape(-1, 3))
        return grad.reshape(points.shape)[..., :dim]

    return g


def parameter_jacobian(model: Model) -> Callable:
    """``∂f/∂θ(theta, points)``, of shape ``(*points.shape[:-1], len(theta))``."""

    f = field(model)

    def j(theta, points):
        flat = jnp.asarray(points).reshape(-1, jnp.asarray(points).shape[-1])
        return jax.vmap(jax.grad(lambda th, q: f(th, q[None])[0]), in_axes=(None, 0))(
            theta, flat
        ).reshape(jnp.asarray(points).shape[:-1] + (len(theta),))

    return j


# --------------------------------------------------------------------- lowering


class Table:
    """The node table and the grammar as methods that intern into it.

    Nodes are interned by structure, children always precede parents, and
    every method returns a node index. The JSON conventions are those of
    ``research/zeroset_test/lower.py``.
    """

    def __init__(self):
        self.nodes: list[dict] = []
        self.index: dict[str, int] = {}
        self.X, self.Y, self.Z = self.coord("X"), self.coord("Y"), self.coord("Z")
        self.point = (self.X, self.Y, self.Z)

    def add(self, node: dict) -> int:
        key = json.dumps(node, sort_keys=True)
        if key not in self.index:
            self.index[key] = len(self.nodes)
            self.nodes.append(node)
        return self.index[key]

    # -- expressions -------------------------------------------------------

    def lit(self, v) -> int:
        return self.add({"lit": float(v)})

    def coord(self, axis: str) -> int:
        return self.add({"coord": axis})

    def param(self, i: int) -> int:
        return self.add({"param": int(i)})

    def child(self, i: int) -> int:
        return self.add({"childValue": int(i)})

    def un(self, op: str, a: int) -> int:
        return self.add({"unary": {"op": op, "a": a}})

    def bi(self, op: str, a: int, b: int) -> int:
        return self.add({"binary": {"op": op, "a": a, "b": b}})

    def add_(self, a, b):
        return self.bi("ADD", a, b)

    def sub(self, a, b):
        return self.bi("SUB", a, b)

    def mul(self, a, b):
        return self.bi("MUL", a, b)

    def div(self, a, b):
        return self.bi("DIV", a, b)

    def mn(self, a, b):
        return self.bi("MIN", a, b)

    def mx(self, a, b):
        return self.bi("MAX", a, b)

    def neg(self, a):
        return self.un("NEG", a)

    def abs_(self, a):
        return self.un("ABS", a)

    def sqrt(self, a):
        return self.un("SQRT", a)

    def norm(self, v):
        acc = self.mul(v[0], v[0])
        for x in v[1:]:
            acc = self.add_(acc, self.mul(x, x))
        return self.sqrt(acc)

    def vsub(self, a, b):
        return tuple(self.sub(x, y) for x, y in zip(a, b))

    # -- shapes ------------------------------------------------------------

    def patch(self, e: int) -> int:
        return self.add({"patch": e})

    def owning(self, rule: str, children) -> int:
        return self.add({"combine": {"owning": rule, "children": list(children)}})

    def warp(self, image, child: int, scale: int | None = None) -> int:
        return self.add(
            {
                "warp": {
                    "imageX": image[0],
                    "imageY": image[1],
                    "imageZ": image[2],
                    "scale": self.lit(1) if scale is None else scale,
                    "child": child,
                }
            }
        )

    def blend(self, rule: int, children, transition: int | None = None) -> int:
        b = {"rule": rule} if transition is None else {"rule": rule, "transition": transition}
        return self.add({"combine": {"blend": b, "children": list(children)}})

    def smin(self, k: int, a: int, b: int) -> int:
        """cadjoint's smooth_min: k' = 4k; h = max(k' - |a-b|, 0); min(a,b) - h²/(4k')."""
        c0, c1 = self.child(0), self.child(1)
        k4 = self.mul(self.lit(4), k)
        gap = self.abs_(self.sub(c0, c1))
        h = self.mx(self.sub(k4, gap), self.lit(0))
        rule = self.sub(self.mn(c0, c1), self.div(self.mul(self.mul(h, h), self.lit(0.25)), k4))
        return self.blend(rule, [a, b], self.sub(k4, gap))

    def model(self, root: int, theta) -> Model:
        return {"node": self.nodes, "root": root, "theta": [float(t) for t in theta]}


def disc(radius: float = 0.6, centre=(0.0173, -0.0221)) -> tuple[Model, np.ndarray, list[str]]:
    """A disc: θ = (R, cx, cy), lowered as a warp of the unit-centred patch ‖p‖ − R."""
    t = Table()
    r, cx, cy = t.param(0), t.param(1), t.param(2)
    ball = t.patch(t.sub(t.norm(t.point), r))
    root = t.warp(t.vsub(t.point, (cx, cy, t.lit(0))), ball)
    theta = np.array([radius, *centre], dtype=float)
    return t.model(root, theta), theta, ["R", "cx", "cy"]


def sphere(
    radius: float = 0.6, centre=(0.0173, -0.0221, 0.0117)
) -> tuple[Model, np.ndarray, list[str]]:
    """A sphere: θ = (R, cx, cy, cz), the same lowering as the disc with a z offset."""
    t = Table()
    r, cx, cy, cz = t.param(0), t.param(1), t.param(2), t.param(3)
    ball = t.patch(t.sub(t.norm(t.point), r))
    root = t.warp(t.vsub(t.point, (cx, cy, cz)), ball)
    theta = np.array([radius, *centre], dtype=float)
    return t.model(root, theta), theta, ["R", "cx", "cy", "cz"]


def blended_union(
    radius: float = 0.4, blend: float = 0.08, box_half_height: float = 0.25
) -> tuple[Model, np.ndarray, list[str]]:
    """A disc smoothly unioned with a box: θ = (R, k, hy).

    The disc of radius R sits at (−0.35, 0); the box, a MAX of four
    half-planes with half-extents (0.35, hy), sits at (0.30, 0). The two are
    joined by cadjoint's smooth minimum with width k, so the blend's fillet
    band is a surface that belongs to neither child.
    """
    t = Table()
    r, k, hy = t.param(0), t.param(1), t.param(2)
    ball = t.patch(t.sub(t.norm(t.point), r))
    disc_node = t.warp(t.vsub(t.point, (t.lit(-0.35), t.lit(0), t.lit(0))), ball)
    hx = t.lit(0.35)
    planes = [
        t.sub(t.X, hx),
        t.sub(t.neg(t.X), hx),
        t.sub(t.Y, hy),
        t.sub(t.neg(t.Y), hy),
    ]
    box = t.owning("MAX", [t.patch(e) for e in planes])
    box_node = t.warp(t.vsub(t.point, (t.lit(0.30), t.lit(0), t.lit(0))), box)
    root = t.smin(k, disc_node, box_node)
    theta = np.array([radius, blend, box_half_height], dtype=float)
    return t.model(root, theta), theta, ["R", "k", "hy"]
