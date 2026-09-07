"""A reference backend for zero-set.proto: numpy only, no autodiff library.

One walker over the node table, parametrised by what it carries: values
alone for discovery's grid, or values with both Jacobians for solving and
derivatives. Discover samples a grid and classifies ownership; Solve is
Newton projection onto the incident surfaces in the minimum-norm gauge;
Evaluate and Vjp follow the formula in the schema. Serves over gRPC.

    python backend.py --port 8850
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
from concurrent import futures

import grpc
import numpy as np
import zero_set_pb2 as zs
import zero_set_pb2_grpc as zs_grpc

U, B, CO = zs.Node.Unary, zs.Node.Binary, zs.Node.Combine

# ----------------------------------------------------------------- numbers
#
# A `Val` is a batch of values over points with, optionally, dv/dp (n×3)
# and dv/dθ (n×P). With the Jacobians absent the same arithmetic is a
# plain evaluation, which is what discovery's grid pass needs.


class Val:
    __slots__ = ("v", "dp", "dt")

    def __init__(self, v, dp=None, dt=None):
        self.v, self.dp, self.dt = v, dp, dt

    @property
    def carries(self):
        return self.dp is not None

    def chain(self, d):
        """f(self) with f' = d, elementwise."""
        if not self.carries:
            return Val(None)
        return Val(None, self.dp * d[:, None], self.dt * d[:, None])

    @staticmethod
    def combine(a, b, da, db, v):
        """f(a, b) with partials da, db."""
        if not a.carries:
            return Val(v)
        return Val(
            v, a.dp * da[:, None] + b.dp * db[:, None], a.dt * da[:, None] + b.dt * db[:, None]
        )


def unary(op, a: Val) -> Val:
    v = a.v
    if op == U.NEG:
        f, d = -v, -np.ones_like(v)
    elif op == U.ABS:
        f, d = np.abs(v), np.sign(v)
    elif op == U.SIGN:
        f, d = np.sign(v), np.zeros_like(v)
    elif op == U.SQRT:
        f = np.sqrt(np.maximum(v, 0.0))
        d = np.where(f > 0, 0.5 / np.where(f > 0, f, 1.0), 0.0)
    elif op == U.EXP:
        f = np.exp(v)
        d = f
    elif op == U.LOG:
        f, d = np.log(v), 1.0 / v
    elif op == U.SIN:
        f, d = np.sin(v), np.cos(v)
    elif op == U.COS:
        f, d = np.cos(v), -np.sin(v)
    elif op == U.TAN:
        f = np.tan(v)
        d = 1.0 + f * f
    else:
        raise ValueError(f"unary op {op}")
    out = a.chain(d)
    out.v = f
    return out


def binary(op, a: Val, b: Val) -> Val:
    x, y = a.v, b.v
    if op == B.ADD:
        return Val.combine(a, b, np.ones_like(x), np.ones_like(y), x + y)
    if op == B.SUB:
        return Val.combine(a, b, np.ones_like(x), -np.ones_like(y), x - y)
    if op == B.MUL:
        return Val.combine(a, b, y, x, x * y)
    if op == B.DIV:
        inv = 1.0 / y
        f = x * inv
        return Val.combine(a, b, inv, -f * inv, f)
    if op in (B.MIN, B.MAX):
        pick = (x <= y) if op == B.MIN else (x >= y)
        return Val.combine(a, b, pick.astype(float), (~pick).astype(float), np.where(pick, x, y))
    if op == B.POW:
        f = np.power(x, y)
        return Val.combine(a, b, y * np.power(x, y - 1.0), f * np.log(np.where(x > 0, x, 1.0)), f)
    if op == B.ATAN2:
        r2 = x * x + y * y
        return Val.combine(a, b, y / r2, -x / r2, np.arctan2(x, y))
    raise ValueError(f"binary op {op}")


# ------------------------------------------------------------------ walker


class Walker:
    """Evaluates the node table at a batch of points.

    `derivatives` decides whether Vals carry dv/dp and dv/dθ. Shape results
    are memoised per (node, points) for the life of one `shape` call, with
    the point arrays kept alive so their ids stay unique; expressions get a
    memo scoped to one shape node, so nothing accumulates across point sets.
    """

    def __init__(self, model: zs.Model, theta: np.ndarray, derivatives: bool):
        self.nodes, self.theta, self.derivatives = list(model.node), theta, derivatives

    def lift(self, v, n):
        v = np.broadcast_to(np.asarray(v, float), (n,)).copy()
        return (
            Val(v, np.zeros((n, 3)), np.zeros((n, len(self.theta)))) if self.derivatives else Val(v)
        )

    def expr(self, i, pts, children, memo) -> Val:
        key = (i, id(children))
        if key in memo:
            return memo[key]
        e = self.nodes[i]
        kind = e.WhichOneof("kind")
        n = len(pts)
        if kind == "lit":
            v = self.lift(e.lit, n)
        elif kind == "coord":
            v = self.lift(pts[:, e.coord], n)
            if self.derivatives:
                v.dp[:, e.coord] = 1.0
        elif kind == "param":
            v = self.lift(self.theta[e.param], n)
            if self.derivatives:
                v.dt[:, e.param] = 1.0
        elif kind == "child_value":
            if children is None:
                raise ValueError("`child_value` outside a blend rule")
            v = children[e.child_value]
        elif kind == "external":
            raise ValueError("this backend accepts no externals")
        elif kind == "unary":
            v = unary(e.unary.op, self.expr(e.unary.a, pts, children, memo))
        elif kind == "binary":
            v = binary(
                e.binary.op,
                self.expr(e.binary.a, pts, children, memo),
                self.expr(e.binary.b, pts, children, memo),
            )
        else:
            raise ValueError(f"node {i} is not an expression")
        memo[key] = v
        return v

    def image(self, w, pts):
        """A warp's mapped points, with their Jacobians when carried, and its scale."""
        memo = {}
        img = [self.expr(k, pts, None, memo) for k in (w.image_x, w.image_y, w.image_z)]
        q = np.stack([c.v for c in img], axis=1)
        sc = self.expr(w.scale, pts, None, memo)
        if not self.derivatives:
            return q, None, None, sc
        return q, np.stack([c.dp for c in img], axis=1), np.stack([c.dt for c in img], axis=1), sc

    def shape(self, i, pts, memo=None) -> Val:
        memo = {} if memo is None else memo
        memo.setdefault("_alive", []).append(pts)
        key = ("s", i, id(pts))
        if key in memo:
            return memo[key]
        s = self.nodes[i]
        kind = s.WhichOneof("kind")
        if kind == "patch":
            v = self.expr(s.patch, pts, None, {})
        elif kind == "warp":
            q, dq_dp, dq_dt, sc = self.image(s.warp, pts)
            inner = self.shape(s.warp.child, q, memo)
            v = Val(sc.v * inner.v)
            if self.derivatives:
                dp = np.einsum("ni,nij->nj", inner.dp, dq_dp)
                dt = inner.dt + np.einsum("ni,nij->nj", inner.dp, dq_dt)
                v.dp = sc.v[:, None] * dp + sc.dp * inner.v[:, None]
                v.dt = sc.v[:, None] * dt + sc.dt * inner.v[:, None]
        elif kind == "combine":
            c = s.combine
            vals = [self.shape(ch, pts, memo) for ch in c.children]
            if c.WhichOneof("rule") == "blend":
                v = self.expr(c.blend.rule, pts, vals, {})
            elif c.owning == CO.COMPLEMENT:
                a = vals[0]
                v = Val(-a.v, None if a.dp is None else -a.dp, None if a.dt is None else -a.dt)
            else:
                v = vals[0]
                for b in vals[1:]:
                    v = binary(B.MIN if c.owning == CO.MIN else B.MAX, v, b)
        else:
            raise ValueError(f"node {i} is not a shape")
        memo[key] = v
        return v

    # -- surfaces ------------------------------------------------------------

    def surfaces(self, root):
        """Every surface in census order, each a function pts -> Val of its own
        field in the model's frame: patches and blends, a blend's rim after it,
        with the warps above composed on. A subtree reached twice is two sets."""
        out = []

        def go(i, warps):
            s = self.nodes[i]
            kind = s.WhichOneof("kind")
            if kind == "patch":
                out.append(("patch", self.through(warps, lambda pts, i=i: self.shape(i, pts))))
            elif kind == "warp":
                go(s.warp.child, warps + [s.warp])
            else:
                c = s.combine
                if c.WhichOneof("rule") == "blend":
                    out.append(("band", self.through(warps, lambda pts, i=i: self.shape(i, pts))))
                    if c.blend.HasField("transition"):

                        def rim(pts, c=c):
                            return self.expr(
                                c.blend.transition,
                                pts,
                                [self.shape(ch, pts, {}) for ch in c.children],
                                {},
                            )

                        out.append(("rim", self.through(warps, rim)))
                for ch in c.children:
                    go(ch, warps)

        go(root, [])
        return out

    def through(self, warps, leaf):
        """Compose the warps above a surface onto its field, scales included."""

        def f(pts):
            n = len(pts)
            q, scale = pts, self.lift(1.0, n)
            dq_dp = np.broadcast_to(np.eye(3), (n, 3, 3)).copy() if self.derivatives else None
            dq_dt = np.zeros((n, 3, len(self.theta))) if self.derivatives else None
            for w in warps:
                q2, J, T, sc = self.image(w, q)
                if self.derivatives:
                    sc = Val(
                        sc.v,
                        np.einsum("ni,nij->nj", sc.dp, dq_dp),
                        sc.dt + np.einsum("ni,nij->nj", sc.dp, dq_dt),
                    )
                    dq_dt = np.einsum("nij,njk->nik", J, dq_dt) + T
                    dq_dp = np.einsum("nij,njk->nik", J, dq_dp)
                scale = binary(B.MUL, scale, sc)
                q = q2
            inner = leaf(q)
            if self.derivatives:
                inner = Val(
                    inner.v,
                    np.einsum("ni,nij->nj", inner.dp, dq_dp),
                    inner.dt + np.einsum("ni,nij->nj", inner.dp, dq_dt),
                )
            return binary(B.MUL, scale, inner)

        return f


def eval_shape(model: zs.Model, pts: np.ndarray, theta: np.ndarray) -> Val:
    """The whole model at the points, with both Jacobians."""
    return Walker(model, theta, derivatives=True).shape(model.root, pts)


def census(model: zs.Model, theta: np.ndarray):
    """Surfaces with Jacobians, in census order."""
    return Walker(model, theta, derivatives=True).surfaces(model.root)


# ----------------------------------------------------------------- ownership


def owners(surfaces, F, whole, margin=0.0):
    """Which surfaces own which points: a patch where its value is the model's;
    a band only strictly inside its rim (transition > margin), since outside
    it its value is a child's and the child owns; a rim never."""
    active = np.abs(F - whole[:, None]) < 1e-9 * (1 + np.abs(whole[:, None]))
    for i, (kind, _) in enumerate(surfaces):
        if kind == "band":
            if i + 1 < len(surfaces) and surfaces[i + 1][0] == "rim":
                active[:, i] &= F[:, i + 1] > margin
        elif kind == "rim":
            active[:, i] = False
    return active


# --------------------------------------------------------------------- solve


def jacobians(surfaces, inc, pts):
    """values (N×K), J_p (N×K×3), J_θ (N×K×P) of each point's incident surfaces."""
    n, kmax = len(pts), max(len(i) for i in inc)
    by_surface: dict[int, list] = {}
    for a, row in enumerate(inc):
        for k, s in enumerate(row):
            by_surface.setdefault(s, []).append((a, k))
    V = JP = JT = None
    for s, where in by_surface.items():
        rows = np.array([a for a, _ in where])
        slots = np.array([k for _, k in where])
        val = surfaces[s][1](pts[rows])
        if V is None:
            V = np.zeros((n, kmax))
            JP = np.zeros((n, kmax, 3))
            JT = np.zeros((n, kmax, val.dt.shape[1]))
        V[rows, slots] = val.v
        JP[rows, slots] = val.dp
        JT[rows, slots] = val.dt
    return V, JP, JT


def project(surfaces, inc, pts, tol, iterations=40):
    """Minimum-norm Newton onto each point's incident surfaces: points, residuals."""
    pts = pts.copy()
    stuck = set()
    for _ in range(iterations):
        V, JP, _ = jacobians(surfaces, inc, pts)
        step = np.zeros_like(pts)
        for a, row in enumerate(inc):
            k = len(row)
            J = JP[a, :k]
            try:
                step[a] = -J.T @ np.linalg.solve(J @ J.T, V[a, :k])
            except np.linalg.LinAlgError:
                stuck.add(a)
        pts = pts + step
        if np.max(np.abs(step)) < tol * 1e-3:
            break
    V, _, _ = jacobians(surfaces, inc, pts)
    residual = np.array([np.max(np.abs(V[a, : len(row)])) for a, row in enumerate(inc)])
    for a in stuck:
        residual[a] = np.inf
    return pts, residual


def solve(model, topo, tol):
    theta = np.asarray(model.theta, float)
    inc = [list(i.surface) for i in topo.incidence]
    seeds = np.array([[s.x, s.y, s.z] for s in topo.seed], float)
    pts, residual = project(census(model, theta), inc, seeds, tol)
    bad = np.nonzero(residual > tol)[0]
    if len(bad):
        return zs.Refusal(
            reason=zs.Refusal.NONCONVERGENT,
            message=f"{len(bad)} points did not converge",
            point=bad.tolist(),
        )
    return zs.Solution(
        point=[zs.Point(x=q[0], y=q[1], z=q[2]) for q in pts],
        incidence=list(topo.incidence),
        residual=residual.tolist(),
    )


# ------------------------------------------------------------------ discover


def candidates(surfaces, grid, F, whole, spacing):
    """Seeds and incidences from grid points near the boundary: a face seed on
    each owner, an edge seed with every other patch passing nearby, and for a
    band, with its own rim."""
    active = owners(surfaces, F, whole)
    rim_of = {
        i: i + 1
        for i, (k, _) in enumerate(surfaces)
        if k == "band" and i + 1 < len(surfaces) and surfaces[i + 1][0] == "rim"
    }
    seeds, inc, seen = [], [], set()

    def emit(g, incidence):
        key = (g, tuple(incidence))
        if key not in seen:
            seen.add(key)
            seeds.append(grid[g])
            inc.append(list(incidence))

    for g in range(len(grid)):
        own = [int(i) for i in np.nonzero(active[g])[0]]
        if not own:
            continue
        o = own[0]
        emit(g, [o])
        for j in range(len(surfaces)):
            if j != o and surfaces[j][0] == "patch" and abs(F[g, j]) < spacing:
                emit(g, sorted([o, j]))
        if o in rim_of and abs(F[g, rim_of[o]]) < spacing:
            emit(g, [o, rim_of[o]])
    return seeds, inc


def verify(model, theta, surfaces, vsurf, seeds, inc, tol):
    """Project every candidate and keep the ones that land on the model's
    boundary, still owned by the surfaces they claim, with a well-conditioned
    system. A rim is kept only for a band that also owns face points."""
    values = Walker(model, theta, derivatives=False)
    proj, res = project(surfaces, inc, np.array(seeds), tol)
    Fq = np.stack([s(proj).v for _, s in vsurf], axis=1)
    wq = values.shape(model.root, proj).v
    owned = owners(surfaces, Fq, wq, margin=10 * tol)
    on_rim = owners(surfaces, Fq, wq, margin=-10 * tol)
    keep = []
    for a, (q, r) in enumerate(zip(proj, res)):
        if r > tol or abs(wq[a]) > 10 * tol:
            continue
        has_rim = any(surfaces[s][0] == "rim" for s in inc[a])
        if not all(surfaces[s][0] == "rim" or (on_rim if has_rim else owned)[a, s] for s in inc[a]):
            continue
        _, JP, _ = jacobians(surfaces, [inc[a]], q[None])
        J = JP[0, : len(inc[a])]
        if np.linalg.cond(J @ J.T) > 1e6:
            continue
        keep.append(a)
    real_bands = {inc[a][0] for a in keep if len(inc[a]) == 1 and surfaces[inc[a][0]][0] == "band"}
    keep = [
        a
        for a in keep
        if not (
            len(inc[a]) == 2 and surfaces[inc[a][1]][0] == "rim" and inc[a][0] not in real_bands
        )
    ]
    return [seeds[a] for a in keep], [inc[a] for a in keep]


def discover(model, domain, tol):
    theta = np.asarray(model.theta, float)
    lo = np.array([domain.min.x, domain.min.y, domain.min.z])
    hi = np.array([domain.max.x, domain.max.y, domain.max.z])
    h = domain.spacing
    grid = np.array(
        list(itertools.product(*[np.arange(lo[i], hi[i] + h * 0.5, h) for i in range(3)]))
    )
    values = Walker(model, theta, derivatives=False)
    whole_all = values.shape(model.root, grid).v
    grid = grid[np.abs(whole_all) < h]  # only near the boundary
    whole = values.shape(model.root, grid).v
    vsurf = values.surfaces(model.root)
    F = np.stack([s(grid).v for _, s in vsurf], axis=1)
    surfaces = census(model, theta)
    seeds, inc = candidates(surfaces, grid, F, whole, h)
    if seeds:
        seeds, inc = verify(model, theta, surfaces, vsurf, seeds, inc, tol)
    digest = hashlib.sha256(repr(sorted({tuple(i) for i in inc})).encode()).hexdigest()[:16]
    return zs.Topology(
        incidence=[zs.Incidence(surface=i) for i in inc],
        seed=[zs.Point(x=s[0], y=s[1], z=s[2]) for s in seeds],
        hash=digest,
    )


# ---------------------------------------------------------------- derivative


def evaluate(model, points, incidences):
    theta = np.asarray(model.theta, float)
    inc = [list(i.surface) for i in incidences]
    pts = np.array([[q.x, q.y, q.z] for q in points], float)
    V, JP, JT = jacobians(census(model, theta), inc, pts)
    return zs.FieldSamples(
        value=V.ravel().tolist(),
        point_jacobian=JP.ravel().tolist(),
        parameter_jacobian=JT.ravel().tolist(),
    )


def vjp(model, solution, cotangent):
    theta = np.asarray(model.theta, float)
    inc = [list(i.surface) for i in solution.incidence]
    pts = np.array([[q.x, q.y, q.z] for q in solution.point], float)
    pbar = np.array([[c.x, c.y, c.z] for c in cotangent], float)
    _, JP, JT = jacobians(census(model, theta), inc, pts)
    out, singular = np.zeros(len(theta)), []
    for a, row in enumerate(inc):
        k = len(row)
        J = JP[a, :k]
        try:
            out -= JT[a, :k].T @ np.linalg.solve(J @ J.T, J @ pbar[a])
        except np.linalg.LinAlgError:
            singular.append(a)
    if singular:
        return zs.Refusal(
            reason=zs.Refusal.NONCONVERGENT,
            message="tangency: J_p is rank deficient",
            point=singular,
        )
    return zs.ParameterCotangent(theta=out.tolist())


# ------------------------------------------------------------------- service


def _result(response_type, field, out):
    return (
        response_type(refusal=out) if isinstance(out, zs.Refusal) else response_type(**{field: out})
    )


class Servicer(zs_grpc.ExtractionServicer):
    def Describe(self, request, context):  # noqa: ARG002 - the gRPC signature
        return zs.Capabilities(
            contract_version=1,
            blends=True,
            unary=list(U.Op.values()),
            binary=list(B.Op.values()),
            evaluate=True,
            vjp=True,
            jvp=False,
        )

    def Discover(self, request, context):  # noqa: ARG002 - the gRPC signature
        try:
            return zs.DiscoverResponse(
                topology=discover(request.model, request.domain, request.tolerance)
            )
        except ValueError as e:
            return zs.DiscoverResponse(
                refusal=zs.Refusal(reason=zs.Refusal.UNSUPPORTED, message=str(e))
            )

    def Solve(self, request, context):  # noqa: ARG002 - the gRPC signature
        try:
            return _result(
                zs.SolveResponse,
                "solution",
                solve(request.model, request.topology, request.tolerance),
            )
        except ValueError as e:
            return zs.SolveResponse(
                refusal=zs.Refusal(reason=zs.Refusal.UNSUPPORTED, message=str(e))
            )

    def Evaluate(self, request, context):  # noqa: ARG002 - the gRPC signature
        return zs.EvaluateResponse(
            samples=evaluate(request.model, request.point, request.incidence)
        )

    def Vjp(self, request, context):  # noqa: ARG002 - the gRPC signature
        return _result(
            zs.VjpResponse, "cotangent", vjp(request.model, request.solution, request.cotangent)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8850)
    args = ap.parse_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    zs_grpc.add_ExtractionServicer_to_server(Servicer(), server)
    server.add_insecure_port(f"127.0.0.1:{args.port}")
    server.start()
    print(f"zero-set reference backend on 127.0.0.1:{args.port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
