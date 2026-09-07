"""The node table, and the lowering of a cadjoint SDF graph into it.

The table is the wire form of ``research/zero-set.proto``: a list of nodes,
each an expression or a shape, referring to earlier nodes by index, so a
walker needs no recursion and a subtree used many times is stored once.
Nodes are interned by structure.  The dicts are the proto's JSON encoding,
so ``json.dumps(model.to_dict())`` parses on the other side with
``json_format.Parse``.

Facts about cadjoint's kernels the lowering depends on: ``Box.size`` and
``Cylinder.height`` are half-extents; ``smooth_min`` is Quilez's with ``k``
scaled by four; ``Rotate`` applies ``Rᵀp``; ``Mirror`` reflects (no union);
polygons use the even-odd signed distance of
:mod:`cadjoint.sdf.primitives.polygon`.  Where a formula here and a kernel
disagree, the kernel is normative: :mod:`tests.zeroset` compares them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cadjoint.extraction import extract_parameters
from cadjoint.geometry import Scalar

__all__ = ["Model", "Table", "lower"]

Index = int
Vec3 = tuple[Index, Index, Index]


@dataclass(frozen=True)
class Model:
    """A lowered model: the table, its root shape, and the design as data."""

    nodes: list[dict[str, Any]]
    root: Index
    theta: list[float]
    names: list[str] = field(default_factory=list)  #: one per θ entry, ``name`` or ``name[i]``

    def to_dict(self) -> dict[str, Any]:
        """The proto3 JSON encoding of ``zeroset.v1.Model``."""
        return {"node": self.nodes, "root": self.root, "theta": self.theta}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class Table:
    """The node table, with the grammar as methods that intern into it.

    Every method returns a node index.  Children always precede their
    parents, and identical subtrees (a pattern's copies, repeated literals)
    are stored once.
    """

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self._index: dict[str, Index] = {}
        self.X, self.Y, self.Z = self.coord("X"), self.coord("Y"), self.coord("Z")
        self.point: Vec3 = (self.X, self.Y, self.Z)

    def add(self, node: dict[str, Any]) -> Index:
        key = json.dumps(node, sort_keys=True)
        if key not in self._index:
            self._index[key] = len(self.nodes)
            self.nodes.append(node)
        return self._index[key]

    # -- expressions -------------------------------------------------------

    def lit(self, v: float) -> Index:
        return self.add({"lit": float(v)})

    def coord(self, axis: str) -> Index:
        return self.add({"coord": axis})

    def param(self, i: int) -> Index:
        return self.add({"param": int(i)})

    def child(self, i: int) -> Index:
        """A combine child's value, by position; meaningful only in a blend rule."""
        return self.add({"childValue": int(i)})

    def un(self, op: str, a: Index) -> Index:
        return self.add({"unary": {"op": op, "a": a}})

    def bi(self, op: str, a: Index, b: Index) -> Index:
        return self.add({"binary": {"op": op, "a": a, "b": b}})

    def add_(self, a: Index, b: Index) -> Index:
        return self.bi("ADD", a, b)

    def sub(self, a: Index, b: Index) -> Index:
        return self.bi("SUB", a, b)

    def mul(self, a: Index, b: Index) -> Index:
        return self.bi("MUL", a, b)

    def div(self, a: Index, b: Index) -> Index:
        return self.bi("DIV", a, b)

    def mn(self, a: Index, b: Index) -> Index:
        return self.bi("MIN", a, b)

    def mx(self, a: Index, b: Index) -> Index:
        return self.bi("MAX", a, b)

    def neg(self, a: Index) -> Index:
        return self.un("NEG", a)

    def abs_(self, a: Index) -> Index:
        return self.un("ABS", a)

    def sqrt(self, a: Index) -> Index:
        return self.un("SQRT", a)

    def sin(self, a: Index) -> Index:
        return self.un("SIN", a)

    def cos(self, a: Index) -> Index:
        return self.un("COS", a)

    def step(self, c: Index) -> Index:
        """1 where c > 0, 0 where c <= 0: a comparison in the grammar.

        Strict at 0 on purpose: a point on a face has the face's extent
        exactly 0, and a half-and-half there would blend an inside branch
        with an outside one at the one place a solver samples.
        """
        return self.mx(self.un("SIGN", c), self.lit(0))

    def same(self, a: Index, b: Index) -> Index:
        """Equality of two 0/1 values."""
        return self.sub(self.lit(1), self.abs_(self.sub(a, b)))

    def where(self, c: Index, a: Index, b: Index) -> Index:
        """``c > 0 ? a : b``."""
        s = self.step(c)
        return self.add_(self.mul(a, s), self.mul(b, self.sub(self.lit(1), s)))

    def clamp01(self, e: Index) -> Index:
        return self.mx(self.lit(0), self.mn(self.lit(1), e))

    def half(self, e: Index) -> Index:
        return self.div(e, self.lit(2))

    # -- vectors of expressions --------------------------------------------

    def dot(self, a: Vec3, b: Vec3) -> Index:
        return self.add_(
            self.add_(self.mul(a[0], b[0]), self.mul(a[1], b[1])), self.mul(a[2], b[2])
        )

    def vsub(self, a: Vec3, b: Vec3) -> Vec3:
        return tuple(self.sub(x, y) for x, y in zip(a, b))  # type: ignore[return-value]

    def vadd(self, a: Vec3, b: Vec3) -> Vec3:
        return tuple(self.add_(x, y) for x, y in zip(a, b))  # type: ignore[return-value]

    def vscale(self, s: Index, v: Vec3) -> Vec3:
        return tuple(self.mul(s, x) for x in v)  # type: ignore[return-value]

    def norm(self, v: Vec3) -> Index:
        return self.sqrt(self.dot(v, v))

    def unit(self, v: Vec3) -> Vec3:
        n = self.norm(v)
        return tuple(self.div(x, n) for x in v)  # type: ignore[return-value]

    def cross(self, a: Vec3, b: Vec3) -> Vec3:
        return (
            self.sub(self.mul(a[1], b[2]), self.mul(a[2], b[1])),
            self.sub(self.mul(a[2], b[0]), self.mul(a[0], b[2])),
            self.sub(self.mul(a[0], b[1]), self.mul(a[1], b[0])),
        )

    def rotation(self, k: Vec3, angle: Index) -> list[list[Index]]:
        """Rodrigues: ``R = c I + s [k]ₓ + (1 - c) k kᵀ`` for a unit axis, as a 3×3 of expressions."""
        c, s = self.cos(angle), self.sin(angle)
        t = self.sub(self.lit(1), c)
        kx, ky, kz = k
        zero = self.lit(0)
        skew = ((zero, self.neg(kz), ky), (kz, zero, self.neg(kx)), (self.neg(ky), kx, zero))
        return [
            [
                self.add_(
                    self.add_(self.mul(t, self.mul(k[i], k[j])), self.mul(s, skew[i][j])),
                    c if i == j else zero,
                )
                for j in range(3)
            ]
            for i in range(3)
        ]

    def apply_transposed(self, R: list[list[Index]], v: Vec3) -> Vec3:
        """``Rᵀ v``."""
        return tuple(
            self.add_(
                self.add_(self.mul(R[0][j], v[0]), self.mul(R[1][j], v[1])), self.mul(R[2][j], v[2])
            )
            for j in range(3)
        )  # type: ignore[return-value]

    def rotate_about(self, origin: Vec3, axis: Vec3, angle: Index) -> Vec3:
        """cadjoint's ``_rotate_about``: ``o + v c + (axis × v) s + axis (axis·v)(1 - c)``."""
        v = self.vsub(self.point, origin)
        c, s = self.cos(angle), self.sin(angle)
        par = self.dot(v, axis)
        cr = self.cross(axis, v)
        one_c = self.sub(self.lit(1), c)
        return tuple(
            self.add_(
                self.add_(self.add_(origin[i], self.mul(v[i], c)), self.mul(cr[i], s)),
                self.mul(self.mul(axis[i], par), one_c),
            )
            for i in range(3)
        )  # type: ignore[return-value]

    # -- shapes ------------------------------------------------------------

    def patch(self, e: Index) -> Index:
        return self.add({"patch": e})

    def owning(self, rule: str, children: list[Index]) -> Index:
        return self.add({"combine": {"owning": rule, "children": list(children)}})

    def complement(self, s: Index) -> Index:
        return self.owning("COMPLEMENT", [s])

    def warp(self, image: Vec3, child: Index, scale: Index | None = None) -> Index:
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

    def blend(
        self,
        rule: Index,
        children: list[Index],
        transition: Index | None = None,
        auxiliary: list[Index] | None = None,
    ) -> Index:
        b: dict[str, Any] = {"rule": rule}
        if transition is not None:
            b["transition"] = transition
        if auxiliary:
            b["auxiliary"] = list(auxiliary)
        return self.add({"combine": {"blend": b, "children": list(children)}})

    # -- composite fields ----------------------------------------------------

    def polygon2(self, px: Index, py: Index, verts: list[tuple[Index, Index]]) -> Index:
        """cadjoint's ``_polygon_distance``: even-odd sign times the distance to the nearest edge."""
        n = len(verts)
        v0x, v0y = verts[0]
        d = self.add_(
            self.mul(self.sub(px, v0x), self.sub(px, v0x)),
            self.mul(self.sub(py, v0y), self.sub(py, v0y)),
        )
        s = self.lit(1)
        for i in range(n):
            j = (i + n - 1) % n
            (vix, viy), (vjx, vjy) = verts[i], verts[j]
            ex, ey = self.sub(vjx, vix), self.sub(vjy, viy)
            wx, wy = self.sub(px, vix), self.sub(py, viy)
            t = self.clamp01(
                self.div(
                    self.add_(self.mul(wx, ex), self.mul(wy, ey)),
                    self.add_(self.mul(ex, ex), self.mul(ey, ey)),
                )
            )
            bx, by = self.sub(wx, self.mul(ex, t)), self.sub(wy, self.mul(ey, t))
            d = self.mn(d, self.add_(self.mul(bx, bx), self.mul(by, by)))
            c1 = self.step(self.sub(py, viy))  # py >= vi.y
            c2 = self.step(self.sub(vjy, py))  # py <  vj.y
            c3 = self.step(self.sub(self.mul(ex, wy), self.mul(ey, wx)))
            flip = self.mul(self.same(c1, c2), self.same(c2, c3))
            s = self.mul(s, self.sub(self.lit(1), self.mul(self.lit(2), flip)))
        return self.mul(s, self.sqrt(d))

    def euclid_extent(self, qs: list[Index], patches: list[Index]) -> Index:
        """cadjoint's box-style exact field over per-axis extents ``q_i``, as a blend.

        Inside (every ``q_i <= 0``) the field is ``max q_i``, one patch's own
        value, so the boundary is owned by the patches.  Outside near an edge
        or corner it is the Euclidean norm of the positive parts, which no
        patch owns; the transition is positive exactly there, off the boundary.
        """
        max_d = qs[0]
        for q in qs[1:]:
            max_d = self.mx(max_d, q)
        squared, positives, largest = self.lit(0), self.lit(0), self.lit(0)
        for q in qs:
            pos = self.mx(q, self.lit(0))
            squared = self.add_(squared, self.mul(pos, pos))
            positives = self.add_(positives, pos)
            largest = self.mx(largest, pos)
        # The sqrt's argument is guarded like the kernel's: inside, where it is
        # exactly zero, its derivative would be 0·∞ and poison the gradient.
        outside = self.where(max_d, self.sqrt(self.where(max_d, squared, self.lit(1))), max_d)
        return self.blend(outside, patches, self.sub(positives, largest))

    def smin(self, k: Index, a: Index, b: Index) -> Index:
        """cadjoint's ``smooth_min``: ``k' = 4k; h = max(k' - |a-b|, 0); min(a,b) - h²/(4k')``."""
        c0, c1 = self.child(0), self.child(1)
        k4 = self.mul(self.lit(4), k)
        gap = self.abs_(self.sub(c0, c1))
        h = self.mx(self.sub(k4, gap), self.lit(0))
        rule = self.sub(self.mn(c0, c1), self.div(self.mul(self.mul(h, h), self.lit(0.25)), k4))
        return self.blend(rule, [a, b], self.sub(k4, gap))


class _Lowering:
    """Maps a cadjoint graph onto a fresh table, and its free parameters onto θ."""

    def __init__(self, root: Any) -> None:
        self.t = Table()
        free, _fixed, _meta = extract_parameters(root)
        self.theta: list[float] = []
        self.names: list[str] = []
        self.index: dict[str, list[int]] = {}
        self._shapes: dict[int, Index] = {}
        for name, value in free.items():
            arr = np.atleast_1d(np.asarray(value, float))
            self.index[name] = list(range(len(self.theta), len(self.theta) + arr.size))
            self.theta.extend(arr.ravel().tolist())
            self.names.extend(
                [name] if arr.size == 1 else [f"{name}[{i}]" for i in range(arr.size)]
            )

    # -- cadjoint numbers -> expressions -------------------------------------

    def scalar(self, s: Any) -> Index:
        if isinstance(s, Scalar) and s.free:
            return self.t.param(self.index[s.name][0])
        return self.t.lit(float(np.asarray(s.value if hasattr(s, "value") else s)))

    def vector(self, v: Any) -> tuple[Index, ...]:
        if getattr(v, "free", False):
            return tuple(self.t.param(i) for i in self.index[v.name])
        return tuple(
            self.t.lit(float(x)) for x in np.asarray(v.value if hasattr(v, "value") else v, float)
        )

    def verts(self, p: dict[str, Any], prefix: str) -> list[tuple[Index, Index]]:
        keys = sorted(
            (k for k in p if k.startswith(prefix) and k[len(prefix) :].isdigit()),
            key=lambda k: int(k[len(prefix) :]),
        )
        return [self.vector(p[k]) for k in keys]  # type: ignore[misc]

    # -- cadjoint nodes -> shapes -------------------------------------------

    def shape(self, node: Any) -> Index:
        """Memoised per cadjoint node: a pattern's child lowers once."""
        if id(node) not in self._shapes:
            self._shapes[id(node)] = self._shape(node)
        return self._shapes[id(node)]

    def _shape(self, node: Any) -> Index:  # noqa: PLR0911, PLR0912, PLR0915 - one case per node kind
        t, p, name = self.t, node.params, type(node).__name__
        X, Y, Z, point = t.X, t.Y, t.Z, t.point

        if name == "Sphere":
            return t.patch(t.sub(t.norm(point), self.scalar(p["radius"])))
        if name == "Torus":  # q = (|p.xy| - R, z); |q| - r
            big, small = self.scalar(p["major_radius"]), self.scalar(p["minor_radius"])
            qxy = t.sub(t.sqrt(t.add_(t.mul(X, X), t.mul(Y, Y))), big)
            return t.patch(t.sub(t.sqrt(t.add_(t.mul(qxy, qxy), t.mul(Z, Z))), small))
        if name == "Capsule":  # a segment along z of half-length `height`, radius r
            r, h = self.scalar(p["radius"]), self.scalar(p["height"])
            dz = t.sub(Z, t.mx(t.neg(h), t.mn(h, Z)))
            return t.patch(
                t.sub(t.sqrt(t.add_(t.add_(t.mul(X, X), t.mul(Y, Y)), t.mul(dz, dz))), r)
            )
        if name == "Offset":
            return t.blend(t.sub(t.child(0), self.scalar(p["distance"])), [self.shape(node.sdf)])
        if name == "Scale":
            factors = np.asarray(p["scale"].value, float)
            uniform = bool(np.allclose(factors, factors[0]))
            s = self.vector(p["scale"])
            image = tuple(t.div(x, f) for x, f in zip(point, s))
            # cadjoint rescales the value only for a uniform scale; a non-uniform one is a bound
            return t.warp(image, self.shape(node.sdf), s[0] if uniform else None)  # type: ignore[arg-type]
        if name == "Box":
            hx, hy, hz = self.vector(p["size"])  # half-extents
            planes = [
                t.sub(X, hx),
                t.sub(t.neg(X), hx),
                t.sub(Y, hy),
                t.sub(t.neg(Y), hy),
                t.sub(Z, hz),
                t.sub(t.neg(Z), hz),
            ]
            qs = [
                t.mx(t.child(0), t.child(1)),
                t.mx(t.child(2), t.child(3)),
                t.mx(t.child(4), t.child(5)),
            ]
            return t.euclid_extent(qs, [t.patch(e) for e in planes])
        if name == "Cylinder":
            r, h = self.scalar(p["radius"]), self.scalar(p["height"])  # half-height
            side = t.sub(t.sqrt(t.add_(t.mul(X, X), t.mul(Y, Y))), r)
            return t.euclid_extent(
                [t.child(0), t.mx(t.child(1), t.child(2))],
                [t.patch(side), t.patch(t.sub(Z, h)), t.patch(t.sub(t.neg(Z), h))],
            )
        if name == "ExtrudedPolygon":
            vs, depth = self.verts(p, "v"), self.scalar(p["depth"])
            qx, qy = X, Y
            if "twist" in p:  # theta = deg2rad(twist) * z / depth; xy' = (x c + y s, y c - x s)
                ang = t.mul(t.mul(self.scalar(p["twist"]), t.lit(np.pi / 180.0)), t.div(Z, depth))
                c, s = t.cos(ang), t.sin(ang)
                qx, qy = t.add_(t.mul(X, c), t.mul(Y, s)), t.sub(t.mul(Y, c), t.mul(X, s))
            side = t.polygon2(qx, qy, vs)
            if "draft" in p:  # d2 += tan(deg2rad(draft)) * (z + depth/2)
                dr = t.mul(self.scalar(p["draft"]), t.lit(np.pi / 180.0))
                side = t.add_(side, t.mul(t.div(t.sin(dr), t.cos(dr)), t.add_(Z, t.half(depth))))
            return t.euclid_extent(
                [t.child(0), t.mx(t.child(1), t.child(2))],
                [
                    t.patch(side),
                    t.patch(t.sub(Z, t.half(depth))),
                    t.patch(t.sub(t.neg(Z), t.half(depth))),
                ],
            )
        if name == "RevolvedPolygon":
            vs, off = self.verts(p, "v"), self.scalar(p["offset"])
            radial = t.sub(t.sqrt(t.add_(t.mul(X, X), t.mul(Z, Z))), off)
            return t.patch(t.polygon2(radial, Y, vs))
        if name == "LoftedPolygon":
            vb, vt, height = self.verts(p, "v"), self.verts(p, "w"), self.scalar(p["height"])
            tt = t.clamp01(t.add_(t.div(Z, height), t.lit(0.5)))
            lerped = [
                (t.add_(bx, t.mul(t.sub(tx, bx), tt)), t.add_(by, t.mul(t.sub(ty, by), tt)))
                for (bx, by), (tx, ty) in zip(vb, vt)
            ]
            return t.euclid_extent(
                [t.child(0), t.mx(t.child(1), t.child(2))],
                [
                    t.patch(t.polygon2(X, Y, lerped)),
                    t.patch(t.sub(Z, t.half(height))),
                    t.patch(t.sub(t.neg(Z), t.half(height))),
                ],
            )
        if name == "Translate":
            return t.warp(t.vsub(point, self.vector(p["offset"])), self.shape(node.sdf))  # type: ignore[arg-type]
        if name == "Rotate":
            R = t.rotation(t.unit(self.vector(p["axis"])), self.scalar(p["angle"]))  # type: ignore[arg-type]
            return t.warp(t.apply_transposed(R, point), self.shape(node.sdf))
        if name == "Mirror":
            o, n = self.vector(p["origin"]), self.vector(p["normal"])
            twice = t.mul(t.lit(2), t.dot(t.vsub(point, o), n))  # type: ignore[arg-type]
            return t.warp(t.vsub(point, t.vscale(twice, n)), self.shape(node.sdf))  # type: ignore[arg-type]
        if name == "Shell":
            return t.blend(
                t.sub(t.abs_(t.child(0)), t.half(self.scalar(p["thickness"]))),
                [self.shape(node.sdf)],
            )
        if name in ("LinearPattern", "PolarPattern"):
            count, mask = int(np.asarray(p["count"].value)), int(np.asarray(p["skip_mask"].value))
            kept = [i for i in range(count) if not mask >> i & 1]
            axis = t.unit(self.vector(p["direction"]))  # type: ignore[arg-type]
            child = self.shape(node.sdf)
            copies = []
            for i in kept:
                if i == 0:
                    copies.append(child)
                elif name == "LinearPattern":
                    shift = t.vscale(t.mul(self.scalar(p["spacing"]), t.lit(i)), axis)
                    copies.append(t.warp(t.vsub(point, shift), child))
                else:
                    origin = self.vector(p["origin"])
                    copies.append(
                        t.warp(t.rotate_about(origin, axis, t.lit(-2 * np.pi * i / count)), child)
                    )  # type: ignore[arg-type]
            return copies[0] if len(copies) == 1 else t.owning("MIN", copies)
        if name in ("Union", "Intersection", "Difference"):
            kids = [self.shape(c) for c in node.sdfs]
            k = p["smoothness"]
            smooth = True if (isinstance(k, Scalar) and k.free) else float(np.asarray(k.value)) > 0
            if name == "Difference":
                kids, name = [kids[0]] + [t.complement(c) for c in kids[1:]], "Intersection"
            if not smooth:
                return t.owning("MIN" if name == "Union" else "MAX", kids)
            kk = self.scalar(k)
            fold = kids[0]
            for c in kids[1:]:
                if name == "Union":
                    fold = t.smin(kk, fold, c)
                else:  # smooth max = -smooth_min(-a, -b)
                    fold = t.complement(t.smin(kk, t.complement(fold), t.complement(c)))
            return fold
        raise NotImplementedError(f"no lowering for {name}")


def lower(root: Any) -> Model:
    """Lower a cadjoint SDF graph to a :class:`Model`.

    Args:
        root: The scene's root SDF node (``_execute_scene(source)["scene"]``,
            or any node built from :mod:`cadjoint.sdf`).

    Returns:
        The table, its root, and the free parameters flattened into ``theta``
        in :func:`cadjoint.extract_parameters` order.
    """
    low = _Lowering(root)
    top = low.shape(root)
    return Model(nodes=low.t.nodes, root=top, theta=low.theta, names=low.names)
