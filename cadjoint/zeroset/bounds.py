"""Bound a node table over a box, instead of sampling it at a point.

The mesher's octree prunes a block when the field provably cannot reach the
level set inside it, and today it proves that with a **Lipschitz constant
the caller supplies**: for a block of half-diagonal ρ, a zero needs
``|f(centre) − level| ≤ ρ·L`` (:mod:`cadjoint.meshing.adaptive`).  That is
only sound when ``L`` really bounds the gradient, which is why the module
tells callers to "pass a generous lipschitz" and why the viewer's overlay
refuses to prune at all — "user-written fields can exceed any assumed
gradient bound, and a hole in the viewer is worse than the ~100 ms this
costs" (:mod:`cadjoint.viewer._edge_overlay`).  Both are the same admission:
the bound is a guess, and the price of the guess is either unsoundness or
dense evaluation.

An *inclusion function* removes the guess.  Evaluate the field with interval
arithmetic over the whole box at once and you get ``[lo, hi]`` guaranteed to
contain every value the field takes there; if the level is outside that
interval, the block provably holds no surface, for any field, with no
constant to supply.  This is the classical idea behind interval branch and
bound (IBEX and its contractors are the mature form of it).

It is a third fold over the same table :mod:`cadjoint.zeroset.evaluate`
walks with JAX and :mod:`cadjoint.zeroset.wgsl` emits as shader code — the
node table is what makes it cheap, because there are only seventeen
operations to give an interval rule to, and every scene cadjoint can build
lowers to them.

**What it costs.**  Interval arithmetic is sound but not tight: it forgets
that the two ``x`` in ``x − x`` are the same variable, so a bound widens
with every reuse of a coordinate (the *dependency problem*).  A box's
interval is therefore wider than the field's true range over it, sometimes
much wider, and a loose bound prunes fewer blocks.  It never prunes a block
it should keep — that direction is guaranteed — so the failure mode is
wasted work, not a hole.  Affine arithmetic, which carries first-order
dependence on each input, is the usual next step when looseness is what
limits pruning; :func:`surface_cells` is written so that swapping the
carrier is the only change needed.

**The mean-value form was tried here and removed.**  Carrying gradient
intervals alongside the value and intersecting with ``f(c) ± Σ ρ|∇f|``
should tighten the bound, and it is the same test
:mod:`cadjoint.meshing.adaptive` performs with a caller-supplied constant.
Measured on ``scenes/starter.py`` and ``scenes/motor_shield.py`` it bought
about 6 % on the bound's width for roughly twenty times the cost, because
the polygon fields route through ``sign``, whose derivative is unbounded
across its jump, and every ``min``/``max`` hull widens the gradient again.
It also lost soundness on the coarsest boxes of an octree descent in a way
this author did not track down.  Affine arithmetic addresses the same
looseness at its actual cause, which is dependency, not linearisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from cadjoint.zeroset.table import Model

__all__ = ["Interval", "bound_field", "surface_cells"]


@dataclass(frozen=True)
class Interval:
    """A batch of intervals: ``lo <= hi`` elementwise, both shaped ``(N,)``."""

    lo: np.ndarray
    hi: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.hi - self.lo

    def contains(self, value: float) -> np.ndarray:
        """Which of the batch's intervals contain ``value``."""
        return (self.lo <= value) & (value <= self.hi)


def _constant(value: np.ndarray | float, n: int) -> Interval:
    array = np.broadcast_to(np.asarray(value, dtype=np.float64), (n,))
    return Interval(array, array)


def _neg(a: Interval) -> Interval:
    return Interval(-a.hi, -a.lo)


def _abs(a: Interval) -> Interval:
    """``|·|`` over an interval: 0 is the low end whenever the interval straddles it."""
    straddles = (a.lo <= 0.0) & (a.hi >= 0.0)
    magnitudes = np.maximum(np.abs(a.lo), np.abs(a.hi))
    lo = np.where(straddles, 0.0, np.minimum(np.abs(a.lo), np.abs(a.hi)))
    return Interval(lo, magnitudes)


def _sign(a: Interval) -> Interval:
    return Interval(np.sign(a.lo), np.sign(a.hi))


def _sqrt(a: Interval) -> Interval:
    return Interval(np.sqrt(np.maximum(a.lo, 0.0)), np.sqrt(np.maximum(a.hi, 0.0)))


def _log(a: Interval) -> Interval:
    with np.errstate(divide="ignore", invalid="ignore"):
        return Interval(np.log(np.maximum(a.lo, 0.0)), np.log(np.maximum(a.hi, 0.0)))


def _periodic(a: Interval, fn, critical: float) -> Interval:
    """A sine or cosine: exact unless the interval spans one of its turning points.

    Rather than reason about which quarter-periods are crossed, test the two
    endpoints and every turning point the interval contains — there are at
    most ``width / pi`` of them, and an interval wider than ``2 pi`` simply
    takes the whole range.
    """
    lo, hi = a.lo, a.hi
    values = [fn(lo), fn(hi)]
    # Turning points sit at `critical + k*pi`; include one on each side.
    first = np.ceil((lo - critical) / math.pi)
    for step in range(3):
        turn = critical + (first + step) * math.pi
        inside = (turn >= lo) & (turn <= hi)
        values.append(np.where(inside, fn(turn), fn(lo)))
        values.append(np.where(inside, fn(turn), fn(hi)))
    wide = (hi - lo) >= 2.0 * math.pi
    stack = np.stack(values)
    return Interval(
        np.where(wide, -1.0, stack.min(axis=0)),
        np.where(wide, 1.0, stack.max(axis=0)),
    )


def _tan(a: Interval) -> Interval:
    """``tan`` is unbounded across every odd multiple of ``pi/2``; say so."""
    lo, hi = a.lo, a.hi
    first = np.ceil((lo - math.pi / 2.0) / math.pi)
    pole = math.pi / 2.0 + first * math.pi
    crosses = ((hi - lo) >= math.pi) | ((pole >= lo) & (pole <= hi))
    with np.errstate(invalid="ignore"):
        return Interval(
            np.where(crosses, -np.inf, np.tan(lo)),
            np.where(crosses, np.inf, np.tan(hi)),
        )


_UNARY = {
    "NEG": _neg,
    "ABS": _abs,
    "SIGN": _sign,
    "SQRT": _sqrt,
    "EXP": lambda a: Interval(np.exp(a.lo), np.exp(a.hi)),
    "LOG": _log,
    "SIN": lambda a: _periodic(a, np.sin, math.pi / 2.0),
    "COS": lambda a: _periodic(a, np.cos, 0.0),
    "TAN": _tan,
}


def _mul(a: Interval, b: Interval) -> Interval:
    """Every corner of the product box; the extremes are always at two of them."""
    corners = np.stack([a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi])
    return Interval(corners.min(axis=0), corners.max(axis=0))


def _div(a: Interval, b: Interval) -> Interval:
    """Division, unbounded wherever the denominator can be zero — which is honest."""
    straddles = (b.lo <= 0.0) & (b.hi >= 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corners = np.stack([a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi])
        lo = np.where(straddles, -np.inf, np.nanmin(corners, axis=0))
        hi = np.where(straddles, np.inf, np.nanmax(corners, axis=0))
    return Interval(lo, hi)


def _pow(a: Interval, b: Interval) -> Interval:
    """``a**b`` for a constant exponent; anything else takes the whole line.

    The lowering only ever raises to a fixed power (a squared distance, a
    profile's taper), so the degenerate-exponent case is the one that
    matters and the rest may be conservative without costing anything.
    """
    constant = b.lo == b.hi
    positive = a.lo >= 0.0
    corners = np.stack([np.abs(a.lo), np.abs(a.hi)])
    with np.errstate(invalid="ignore"):
        lo = np.power(np.maximum(a.lo, 0.0), b.lo)
        hi = np.power(np.maximum(corners.max(axis=0), 0.0), b.lo)
    usable = constant & positive
    return Interval(np.where(usable, np.minimum(lo, hi), -np.inf), np.where(usable, hi, np.inf))


_BINARY = {
    "ADD": lambda a, b: Interval(a.lo + b.lo, a.hi + b.hi),
    "SUB": lambda a, b: Interval(a.lo - b.hi, a.hi - b.lo),
    "MUL": _mul,
    "DIV": _div,
    "MIN": lambda a, b: Interval(np.minimum(a.lo, b.lo), np.minimum(a.hi, b.hi)),
    "MAX": lambda a, b: Interval(np.maximum(a.lo, b.lo), np.maximum(a.hi, b.hi)),
    "POW": _pow,
    # Sound and deliberately blunt: atan2's range is all of [-pi, pi], and
    # tightening it only pays on a scene that revolves, where the angle is
    # one term of a distance the rest of which already bounds well.
    "ATAN2": lambda a, _b: Interval(np.full_like(a.lo, -math.pi), np.full_like(a.hi, math.pi)),
}

_AXIS = {"X": 0, "Y": 1, "Z": 2}


def _kind(node: dict[str, Any]) -> str:
    return next(iter(node))


class _Walker:
    """The table's fold, carrying intervals instead of point values."""

    def __init__(self, nodes: list[dict[str, Any]], theta: np.ndarray) -> None:
        self.nodes = nodes
        self.theta = np.asarray(theta, dtype=np.float64)
        # The memo is keyed by `id(box)`, and a box tuple that goes out of
        # scope frees its id for the next one — a lookup would then answer
        # with a value computed over a different box.  Holding every box
        # alive for the walk keeps the ids distinct.  (`evaluate.py` carries
        # the same guard for the same reason.)
        self._alive: list[Any] = []

    def expr(
        self, index: int, box: tuple[Interval, Interval, Interval], children, memo
    ) -> Interval:
        key = (index, id(box))
        if key in memo:
            return memo[key]
        node = self.nodes[index]
        kind = _kind(node)
        n = box[0].lo.shape[0]
        if kind == "lit":
            value = _constant(node["lit"], n)
        elif kind == "coord":
            value = box[_AXIS[node["coord"]]]
        elif kind == "param":
            value = _constant(self.theta[node["param"]], n)
        elif kind == "childValue":
            value = children[node["childValue"]]
        elif kind == "unary":
            value = _UNARY[node["unary"]["op"]](self.expr(node["unary"]["a"], box, children, memo))
        elif kind == "binary":
            b = node["binary"]
            value = _BINARY[b["op"]](
                self.expr(b["a"], box, children, memo), self.expr(b["b"], box, children, memo)
            )
        else:
            raise ValueError(f"node {index} ({kind}) is not an expression")
        memo[key] = value
        return value

    def shape(self, index: int, box: tuple[Interval, Interval, Interval]) -> Interval:
        self._alive.append(box)
        node = self.nodes[index]
        kind = _kind(node)
        memo: dict = {}
        if kind == "patch":
            return self.expr(node["patch"], box, None, memo)
        if kind == "warp":
            w = node["warp"]
            # The image of a box under the warp is bounded by evaluating the
            # point map's three coordinates over it — a box again, wider than
            # the true image, which is exactly what soundness requires.
            image = tuple(
                self.expr(w[axis], box, None, memo) for axis in ("imageX", "imageY", "imageZ")
            )
            inner = self.shape(w["child"], image)
            return _mul(inner, self.expr(w["scale"], box, None, memo))
        if kind == "combine":
            c = node["combine"]
            children = c["children"]
            if "blend" in c:
                # A blend rule names its operands by position, so every child
                # has to be live at once; there is no folding this one.
                values = [self.shape(child, box) for child in children]
                return self.expr(c["blend"]["rule"], box, values, memo)
            if c["owning"] == "COMPLEMENT":
                return _neg(self.shape(children[0], box))
            # A hard union or intersection folds pairwise, so only the running
            # bound and one child are ever live.  Scenes reach the hundreds of
            # patch fields under a single union, and holding all of their
            # intervals at once would cost that many arrays instead of two.
            reduce = _BINARY["MIN"] if c["owning"] == "MIN" else _BINARY["MAX"]
            out = self.shape(children[0], box)
            for child in children[1:]:
                out = reduce(out, self.shape(child, box))
            return out
        raise ValueError(f"node {index} ({kind}) is not a shape")


def bound_field(model: Model, theta: Any, lo: np.ndarray, hi: np.ndarray) -> Interval:
    """The field's guaranteed range over each box.

    Args:
        model: A lowered node table.
        theta: The design, in the table's parameter order.
        lo: ``(N, 3)`` lower corners.
        hi: ``(N, 3)`` upper corners.

    Returns:
        An :class:`Interval` of ``N`` guaranteed ranges: for every point in
        box ``k``, ``lo[k] <= f(p) <= hi[k]``.
    """
    lo = np.asarray(lo, dtype=np.float64).reshape(-1, 3)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1, 3)
    box = tuple(Interval(lo[:, axis], hi[:, axis]) for axis in range(3))
    return _Walker(model.nodes, theta).shape(model.root, box)


def surface_cells(
    model: Model,
    theta: Any,
    grid: Any,
    *,
    level: float = 0.0,
    stats: dict | None = None,
) -> np.ndarray:
    """Every lattice cell the level set can cross, by octree descent on bounds.

    The same descent as :func:`cadjoint.meshing.adaptive.surface_cells`, with
    the Lipschitz test replaced by the inclusion test: a block survives only
    if its interval contains ``level``.  No Lipschitz constant, and the
    result is a conservative superset for *any* field the table can express,
    not only for one whose gradient the caller guessed correctly.

    Args:
        model: A lowered node table.
        theta: The design.
        grid: A :class:`~cadjoint.meshing.GridSpec`.
        level: The level set to find.
        stats: Optional dict; filled with ``bounds`` (how many boxes were
            bounded) and ``levels`` (the octree depth walked).

    Returns:
        ``(cell_count, 3)`` lattice cell indices.
    """
    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    cells = np.asarray(grid.cells, dtype=np.int64)
    bounded = 0
    depth = 0
    block = 1 << int(np.ceil(np.log2(cells.max())))
    blocks = np.zeros((1, 3), dtype=np.int64)
    while True:
        low = blocks * block
        high = np.minimum(low + block, cells)
        interval = bound_field(model, theta, origin + low * spacing, origin + high * spacing)
        bounded += blocks.shape[0]
        depth += 1
        blocks = blocks[interval.contains(level)]
        if block == 1 or blocks.shape[0] == 0:
            break
        block //= 2
        offsets = np.array(list(np.ndindex(2, 2, 2)), dtype=np.int64)
        blocks = (blocks[:, None, :] * 2 + offsets[None]).reshape(-1, 3)
        blocks = blocks[np.all(blocks * block < cells, axis=1)]
    if stats is not None:
        stats.update(bounds=bounded, levels=depth)
    return blocks
