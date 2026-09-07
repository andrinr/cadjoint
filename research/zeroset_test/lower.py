"""Lower a cadjoint SDF graph into a zero-set Model, as proto JSON.

Runs in the cadjoint venv. Emits `model.json` (the Model in proto3 JSON,
the standard language-neutral encoding) and `reference.json`: cadjoint's
own field values and parameter gradients at random points, from JAX, for
the other side to check itself against.

    .venv/bin/python research/zeroset_test/lower.py OUT [scene.py] [--per-node]

Facts about cadjoint's kernels this depends on: Box.size and Cylinder.height
are half-extents; smooth_min is Quilez's with k scaled by four; Rotate applies
Rᵀp; Mirror reflects (no union); polygons use the even-odd signed distance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint import extract_parameters, functionalize
from cadjoint.geometry import Scalar, Vector
from cadjoint.sdf import boolean as B
from cadjoint.sdf import primitives as P
from cadjoint.sdf import transforms as T


class Table:
    """The node table, and the grammar as methods that intern into it.

    Nodes are interned by structure, so identical subtrees (a pattern's
    copies, repeated literals) are stored once and children always precede
    their parents. Every method returns a node index.
    """

    def __init__(self):
        self.nodes, self.index = [], {}
        self.X, self.Y, self.Z = self.coord("X"), self.coord("Y"), self.coord("Z")
        self.point = (self.X, self.Y, self.Z)

    def add(self, node):
        key = json.dumps(node, sort_keys=True)
        if key not in self.index:
            self.index[key] = len(self.nodes)
            self.nodes.append(node)
        return self.index[key]

    # -- expressions -------------------------------------------------------

    def lit(self, v):
        return self.add({"lit": float(v)})

    def coord(self, axis):
        return self.add({"coord": axis})

    def param(self, i):
        return self.add({"param": int(i)})

    def child(self, i):
        return self.add({"childValue": int(i)})

    def un(self, op, a):
        return self.add({"unary": {"op": op, "a": a}})

    def bi(self, op, a, b):
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

    def sin(self, a):
        return self.un("SIN", a)

    def cos(self, a):
        return self.un("COS", a)

    def step(self, c):
        """1 where c > 0, 0 where c < 0: a comparison in the grammar (½ at 0, a measure-zero set)."""
        return self.mul(self.add_(self.un("SIGN", c), self.lit(1)), self.lit(0.5))

    def same(self, a, b):
        """Equality of two 0/1 values."""
        return self.sub(self.lit(1), self.abs_(self.sub(a, b)))

    def where(self, c, a, b):
        """c > 0 ? a : b."""
        s = self.step(c)
        return self.add_(self.mul(a, s), self.mul(b, self.sub(self.lit(1), s)))

    def clamp01(self, e):
        return self.mx(self.lit(0), self.mn(self.lit(1), e))

    # -- vectors of expressions --------------------------------------------

    def dot(self, a, b):
        return self.add_(
            self.add_(self.mul(a[0], b[0]), self.mul(a[1], b[1])), self.mul(a[2], b[2])
        )

    def vsub(self, a, b):
        return tuple(self.sub(x, y) for x, y in zip(a, b))

    def vadd(self, a, b):
        return tuple(self.add_(x, y) for x, y in zip(a, b))

    def vscale(self, s, v):
        return tuple(self.mul(s, x) for x in v)

    def norm(self, v):
        return self.sqrt(self.dot(v, v))

    def unit(self, v):
        n = self.norm(v)
        return tuple(self.div(x, n) for x in v)

    def cross(self, a, b):
        return (
            self.sub(self.mul(a[1], b[2]), self.mul(a[2], b[1])),
            self.sub(self.mul(a[2], b[0]), self.mul(a[0], b[2])),
            self.sub(self.mul(a[0], b[1]), self.mul(a[1], b[0])),
        )

    def rotation(self, k, angle):
        """Rodrigues: R = c I + s [k]ₓ + (1 - c) k kᵀ for a unit axis k, as a 3×3 of expressions."""
        c, s = self.cos(angle), self.sin(angle)
        t = self.sub(self.lit(1), c)
        kx, ky, kz = k
        skew = (
            (self.lit(0), self.neg(kz), ky),
            (kz, self.lit(0), self.neg(kx)),
            (self.neg(ky), kx, self.lit(0)),
        )
        return [
            [
                self.add_(
                    self.add_(self.mul(t, self.mul(k[i], k[j])), self.mul(s, skew[i][j])),
                    c if i == j else self.lit(0),
                )
                for j in range(3)
            ]
            for i in range(3)
        ]

    def apply_transposed(self, R, v):
        """Rᵀ v."""
        return tuple(
            self.add_(
                self.add_(self.mul(R[0][j], v[0]), self.mul(R[1][j], v[1])), self.mul(R[2][j], v[2])
            )
            for j in range(3)
        )

    def rotate_about(self, origin, axis, angle):
        """cadjoint's _rotate_about: origin + v c + (axis × v) s + axis (axis·v)(1 - c), v = p - origin."""
        v = self.vsub(self.point, origin)
        c, s = self.cos(angle), self.sin(angle)
        par = self.dot(v, axis)
        cr = self.cross(axis, v)
        return tuple(
            self.add_(
                self.add_(self.add_(origin[i], self.mul(v[i], c)), self.mul(cr[i], s)),
                self.mul(self.mul(axis[i], par), self.sub(self.lit(1), c)),
            )
            for i in range(3)
        )

    # -- shapes ------------------------------------------------------------

    def patch(self, e):
        return self.add({"patch": e})

    def owning(self, rule, children):
        return self.add({"combine": {"owning": rule, "children": list(children)}})

    def warp(self, image, child, scale=None):
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

    def blend(self, rule, children, transition=None):
        b = {"rule": rule} if transition is None else {"rule": rule, "transition": transition}
        return self.add({"combine": {"blend": b, "children": list(children)}})

    def complement(self, s):
        return self.owning("COMPLEMENT", [s])

    # -- composite fields ----------------------------------------------------

    def polygon2(self, px, py, verts):
        """cadjoint's _polygon_distance: even-odd sign times the distance to the nearest edge."""
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

    def euclid_extent(self, qs, patches):
        """cadjoint's box-style exact field over per-axis extents q_i, as a Blend.

        Inside (every q_i <= 0) the field is max q_i, one patch's own value,
        so the boundary is owned by the patches. Outside near an edge or corner
        it is the Euclidean norm of the positive parts, which no patch owns;
        the transition is positive exactly there, off the boundary.
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
        rule = self.where(self.neg(max_d), max_d, self.sqrt(squared))
        return self.blend(rule, patches, self.sub(positives, largest))

    def smin(self, k, a, b):
        """cadjoint's smooth_min of two shapes: k' = 4k; h = max(k' - |a-b|, 0); min(a,b) - h*h*0.25/k'."""
        c0, c1 = self.child(0), self.child(1)
        k4 = self.mul(self.lit(4), k)
        gap = self.abs_(self.sub(c0, c1))
        h = self.mx(self.sub(k4, gap), self.lit(0))
        rule = self.sub(self.mn(c0, c1), self.div(self.mul(self.mul(h, h), self.lit(0.25)), k4))
        return self.blend(rule, [a, b], self.sub(k4, gap))


class Lowering:
    """Maps a cadjoint graph onto a fresh table, and its parameters onto θ."""

    def __init__(self, root):
        self.t = Table()
        free, self.fixed, self.meta = extract_parameters(root)
        self.theta, self.index, self._shapes = [], {}, {}
        for name, value in free.items():
            arr = np.atleast_1d(np.asarray(value, float))
            self.index[name] = list(range(len(self.theta), len(self.theta) + arr.size))
            self.theta.extend(arr.ravel().tolist())

    def model(self, root):
        top = self.shape(root)
        return {"node": self.t.nodes, "root": top, "theta": self.theta}

    # -- cadjoint numbers -> expressions -------------------------------------

    def scalar(self, s):
        if isinstance(s, Scalar) and s.free:
            return self.t.param(self.index[s.name][0])
        return self.t.lit(float(np.asarray(s.value if hasattr(s, "value") else s)))

    def vector(self, v):
        if getattr(v, "free", False):
            return tuple(self.t.param(i) for i in self.index[v.name])
        return tuple(
            self.t.lit(float(x)) for x in np.asarray(v.value if hasattr(v, "value") else v, float)
        )

    def verts(self, p, prefix):
        keys = sorted(
            (k for k in p if k.startswith(prefix) and k[len(prefix) :].isdigit()),
            key=lambda k: int(k[len(prefix) :]),
        )
        return [self.vector(p[k]) for k in keys]

    # -- cadjoint nodes -> shapes -------------------------------------------

    def shape(self, node):
        """Memoised per cadjoint node: a pattern's child lowers once."""
        if id(node) not in self._shapes:
            self._shapes[id(node)] = self._shape(node)
        return self._shapes[id(node)]

    def _shape(self, node):
        t, p, name = self.t, node.params, type(node).__name__
        X, Y, Z, point = t.X, t.Y, t.Z, t.point

        def half(e):
            return t.div(e, t.lit(2))

        if name == "Sphere":
            return t.patch(t.sub(t.norm(point), self.scalar(p["radius"])))
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
                side = t.add_(side, t.mul(t.div(t.sin(dr), t.cos(dr)), t.add_(Z, half(depth))))
            return t.euclid_extent(
                [t.child(0), t.mx(t.child(1), t.child(2))],
                [
                    t.patch(side),
                    t.patch(t.sub(Z, half(depth))),
                    t.patch(t.sub(t.neg(Z), half(depth))),
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
                    t.patch(t.sub(Z, half(height))),
                    t.patch(t.sub(t.neg(Z), half(height))),
                ],
            )
        if name == "Translate":
            return t.warp(t.vsub(point, self.vector(p["offset"])), self.shape(node.sdf))
        if name == "Rotate":
            R = t.rotation(t.unit(self.vector(p["axis"])), self.scalar(p["angle"]))
            return t.warp(t.apply_transposed(R, point), self.shape(node.sdf))
        if name == "Mirror":
            o, n = self.vector(p["origin"]), self.vector(p["normal"])
            return t.warp(
                t.vsub(point, t.vscale(t.mul(t.lit(2), t.dot(t.vsub(point, o), n)), n)),
                self.shape(node.sdf),
            )
        if name == "Shell":
            return t.blend(
                t.sub(t.abs_(t.child(0)), half(self.scalar(p["thickness"]))), [self.shape(node.sdf)]
            )
        if name in ("LinearPattern", "PolarPattern"):
            count, mask = int(np.asarray(p["count"].value)), int(np.asarray(p["skip_mask"].value))
            kept = [i for i in range(count) if not mask >> i & 1]
            axis = t.unit(self.vector(p["direction"]))
            child = self.shape(node.sdf)
            copies = []
            for i in kept:
                if i == 0:
                    copies.append(child)
                elif name == "LinearPattern":
                    copies.append(
                        t.warp(
                            t.vsub(
                                point, t.vscale(t.mul(self.scalar(p["spacing"]), t.lit(i)), axis)
                            ),
                            child,
                        )
                    )
                else:
                    copies.append(
                        t.warp(
                            t.rotate_about(
                                self.vector(p["origin"]), axis, t.lit(-2 * np.pi * i / count)
                            ),
                            child,
                        )
                    )
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
                fold = (
                    t.smin(kk, fold, c)
                    if name == "Union"
                    else t.complement(t.smin(kk, t.complement(fold), t.complement(c)))
                )
            return fold
        raise NotImplementedError(f"no lowering for {name}")


# ------------------------------------------------------------ references


def _children(n):
    cs = list(getattr(n, "sdfs", []) or [])
    c = getattr(n, "sdf", None)
    if c is not None and hasattr(c, "params"):
        cs.append(c)
    return cs


def _field(node):
    """cadjoint's own field of a node as a function of (θ flat, points), via JAX."""
    free, fixed, _ = extract_parameters(node)
    fn, names = functionalize(node), list(free.keys())

    def field(theta_flat, pts):
        fp, i = {}, 0
        for n in names:
            size = np.asarray(free[n]).size
            fp[n] = (
                theta_flat[i : i + size].reshape(np.asarray(free[n]).shape)
                if size > 1
                else theta_flat[i]
            )
            i += size
        return jax.vmap(fn(fp, fixed))(pts)

    return field


def sample_scene():
    """A bracket-like part: a plate, a boss, a smooth-unioned rib, a hole."""
    plate = P.Box(Vector([1.0, 0.7, 0.1], free=True, name="plate"))
    boss = T.Translate(P.Cylinder(Scalar(0.25, free=True, name="boss_r"), 0.3), [0.4, 0.0, 0.3])
    rib = T.Rotate(
        T.Translate(P.Box([0.05, 0.3, 0.25]), [-0.3, 0.0, 0.3]),
        [0, 0, 1],
        Scalar(0.3, free=True, name="tilt"),
    )
    body = B.Union(plate, boss, rib, smoothness=Scalar(0.03, free=True, name="fillet"))
    hole = T.Translate(P.Cylinder(0.08, 0.5), [-0.6, 0.3, 0.0])
    return B.Difference(body, hole)


def whole(out: Path, root):
    low = Lowering(root)
    model = low.model(root)
    (out / "model.json").write_text(json.dumps(model))
    field = _field(root)
    rng = np.random.default_rng(0)
    theta = jnp.asarray(low.theta)
    probe = jnp.asarray(rng.uniform(-4, 4, size=(20000, 3)))  # a coarse probe for the extent
    inside = np.asarray(field(theta, probe)) <= 0
    lo = np.asarray(probe)[inside].min(axis=0) - 0.25
    hi = np.asarray(probe)[inside].max(axis=0) + 0.25
    pts = jnp.asarray(rng.uniform(lo, hi, size=(400, 3)))
    (out / "reference.json").write_text(
        json.dumps(
            {
                "points": np.asarray(pts).tolist(),
                "values": np.asarray(field(theta, pts)).tolist(),
                "dtheta": np.asarray(jax.jacfwd(field)(theta, pts)).tolist(),
                "bounds": [lo.tolist(), hi.tolist()],
            }
        )
    )
    print(
        f"lowered: {len(low.theta)} parameters, {len(model['node'])} nodes; wrote {out}/model.json ({len(json.dumps(model))} bytes) and reference.json (400 points)"
    )


def per_node(out: Path, root):
    """Every node instance lowered alone with cadjoint's values, so a wrong lowering names itself."""
    instances = {}

    def walk(n, path):
        instances[f"{path}_{type(n).__name__}"] = n
        for i, c in enumerate(_children(n)):
            walk(c, f"{path}.{i}")

    walk(root, "r")
    rng = np.random.default_rng(0)
    for name, node in instances.items():
        low = Lowering(node)
        model = low.model(node)
        theta = jnp.asarray(low.theta) if low.theta else jnp.zeros(0)
        pts = jnp.asarray(rng.uniform(-2.5, 2.5, size=(300, 3)))
        d = out / "nodes" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.json").write_text(json.dumps(model))
        (d / "reference.json").write_text(
            json.dumps(
                {
                    "points": np.asarray(pts).tolist(),
                    "values": np.asarray(_field(node)(theta, pts)).tolist(),
                    "children": [type(c).__name__ for c in _children(node)],
                }
            )
        )
    print("per-node references:", len(instances), "instances")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    if len(args) > 1:
        from cadjoint.viewer.worker.scene import _execute_scene

        root = _execute_scene(Path(args[1]).read_text())["scene"]
    else:
        root = sample_scene()
    (per_node if "--per-node" in sys.argv else whole)(out, root)


if __name__ == "__main__":
    main()
