"""The host side: does the protocol carry a cadjoint model to a backend it has
never heard of, and does the derivative survive the trip?

Runs in the scratch venv (numpy, grpc, protobuf; no JAX, no cadjoint).

    python host.py /tmp/zs --port 8850
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import backend as ref
import grpc
import numpy as np
import zero_set_pb2 as zs
import zero_set_pb2_grpc as zs_grpc
from google.protobuf import empty_pb2, json_format


def row(label, ok, detail):
    print(f"  {'ok ' if ok else 'FAIL'}  {label:<58} {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--port", type=int, default=8850)
    ap.add_argument("--cells", type=int, default=28)
    args = ap.parse_args()
    d = Path(args.dir)
    results = []

    # 1. the wire format: strict parse of the frontend's JSON into the schema
    model = json_format.Parse(d.joinpath("model.json").read_text(), zs.Model())
    wire = model.SerializeToString()
    results.append(
        row(
            "model.json parses into zeroset.v1.Model",
            True,
            f"{len(wire)} bytes binary, {len(model.node)} nodes, {len(model.theta)} parameters",
        )
    )

    # 2. fidelity: the lowered tree against cadjoint's own field and gradients
    refd = json.loads(d.joinpath("reference.json").read_text())
    pts = np.array(refd["points"])
    theta = np.array(model.theta)
    val = ref.eval_shape(model, pts, theta)
    fv = np.array(refd["values"])
    inside = fv <= 0
    results.append(
        row(
            "sign of the field agrees with cadjoint at 400 random points",
            np.all(np.sign(val.v) == np.sign(fv)),
            "the zero set is the same",
        )
    )
    err_v = np.max(np.abs(val.v - fv))
    results.append(
        row(
            "field values match cadjoint at all 400 points",
            err_v < 1e-5,
            f"max |Δf| = {err_v:.1e} ({int(inside.sum())} inside)",
        )
    )
    err_t = np.max(np.abs(val.dt - np.array(refd["dtheta"])))
    results.append(
        row(
            "∂f/∂θ matches JAX at all 400 points",
            err_t < 1e-4,
            f"max |Δ| = {err_t:.1e} over {theta.size} parameters",
        )
    )

    # 3. the service, in another process, over gRPC
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("backend.py")), "--port", str(args.port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(50):
            line = proc.stdout.readline()
            if "reference backend" in line:
                break
            time.sleep(0.1)
        chan = grpc.insecure_channel(f"127.0.0.1:{args.port}")
        stub = zs_grpc.ExtractionStub(chan)
        caps = stub.Describe(empty_pb2.Empty())
        results.append(
            row(
                "Describe answers over gRPC",
                caps.contract_version == 1,
                f"blends={caps.blends} evaluate={caps.evaluate} vjp={caps.vjp}",
            )
        )

        tol = 1e-8
        lo, hi = refd["bounds"]
        spacing = float(max(h2 - l2 for l2, h2 in zip(lo, hi))) / args.cells
        domain = zs.Domain(
            min=zs.Point(x=lo[0], y=lo[1], z=lo[2]),
            max=zs.Point(x=hi[0], y=hi[1], z=hi[2]),
            spacing=spacing,
        )
        t0 = time.perf_counter()
        disc = stub.Discover(zs.DiscoverRequest(model=model, domain=domain, tolerance=tol))
        topo = disc.topology
        n = len(topo.incidence)
        kinds = [len(i.surface) for i in topo.incidence]
        results.append(
            row(
                "Discover returns a topology",
                disc.WhichOneof("result") == "topology",
                f"{n} seeds: {kinds.count(1)} face, {kinds.count(2)} edge; hash {topo.hash}; {time.perf_counter()-t0:.1f}s",
            )
        )

        t0 = time.perf_counter()
        sol = stub.Solve(zs.SolveRequest(model=model, topology=topo, tolerance=tol)).solution
        res = np.array(sol.residual)
        results.append(
            row(
                "Solve: every residual within tolerance (backend's word)",
                np.all(res <= tol),
                f"max {res.max():.1e}; {time.perf_counter()-t0:.1f}s",
            )
        )

        # 4. admission: the host re-evaluates with its own walker
        pts_s = np.array([[p.x, p.y, p.z] for p in sol.point])
        inc = [list(i.surface) for i in sol.incidence]
        surfaces = ref.census(model, theta)
        V, JP, JT = ref.jacobians(surfaces, inc, pts_s)
        host_res = np.array([np.max(np.abs(V[a, : len(r)])) for a, r in enumerate(inc)])
        results.append(
            row(
                "residuals within tolerance when the host re-evaluates",
                np.all(host_res <= 1e-6),
                f"max {host_res.max():.1e}",
            )
        )
        results.append(
            row(
                "incidence unchanged by the solve",
                [list(i.surface) for i in topo.incidence] == inc,
                "",
            )
        )

        # 5. the derivative: backend's Vjp vs the host's own, and vs finite differences of Solve
        rng = np.random.default_rng(1)
        pbar = rng.normal(size=(n, 3))
        got = np.array(
            stub.Vjp(
                zs.VjpRequest(
                    model=model,
                    solution=sol,
                    cotangent=[zs.Point(x=c[0], y=c[1], z=c[2]) for c in pbar],
                )
            ).cotangent.theta
        )
        mine = np.zeros(theta.size)
        for a, r in enumerate(inc):
            k = len(r)
            J = JP[a, :k]
            lam = np.linalg.solve(J @ J.T, J @ pbar[a])
            mine -= JT[a, :k].T @ lam
        results.append(
            row(
                "backend Vjp equals the host's implicit-function Vjp",
                np.allclose(got, mine, rtol=1e-6, atol=1e-9),
                f"max |Δ| = {np.max(np.abs(got-mine)):.1e}",
            )
        )

        h = 1e-5
        dtheta = rng.normal(size=theta.size)
        # a refresh: the same topology, seeded with the solution just found,
        # which is the re-solve the derivative predicts
        refreshed = zs.Topology()
        refreshed.CopyFrom(topo)
        del refreshed.seed[:]
        refreshed.seed.extend(sol.point)
        refused = set()

        def solved(th):
            m2 = zs.Model()
            m2.CopyFrom(model)
            del m2.theta[:]
            m2.theta.extend(th.tolist())
            r2 = stub.Solve(zs.SolveRequest(model=m2, topology=refreshed, tolerance=tol))
            if r2.WhichOneof("result") == "refusal":
                # a partial refusal names its points; re-solve without them
                refused.update(r2.refusal.point)
                keep = [i for i in range(n) if i not in refused]
                t2 = zs.Topology(
                    incidence=[refreshed.incidence[i] for i in keep],
                    seed=[refreshed.seed[i] for i in keep],
                    hash="",
                )
                r2 = stub.Solve(zs.SolveRequest(model=m2, topology=t2, tolerance=tol))
                assert r2.WhichOneof("result") == "solution", r2.refusal.message
                full = np.full((n, 3), np.nan)
                full[keep] = [[p.x, p.y, p.z] for p in r2.solution.point]
                return full
            return np.array([[p.x, p.y, p.z] for p in r2.solution.point])

        fd = (solved(theta + h * dtheta) - solved(theta - h * dtheta)) / (2 * h)
        # forward form from the host's Jacobians, minimum-norm gauge
        jvp = np.zeros((n, 3))
        for a, r in enumerate(inc):
            k = len(r)
            J = JP[a, :k]
            lam = np.linalg.solve(J @ J.T, JT[a, :k] @ dtheta)
            jvp[a] = -J.T @ lam
        ok_rows = ~np.isnan(fd).any(axis=1)
        rel = np.linalg.norm((fd - jvp)[ok_rows]) / max(np.linalg.norm(jvp[ok_rows]), 1e-300)
        results.append(
            row(
                "forward derivative matches central differences of Solve",
                rel < 1e-4 and len(refused) <= n // 100,
                f"relative error {rel:.1e} over {int(ok_rows.sum())} points"
                + (f"; {len(refused)} refused on re-solve" if refused else ""),
            )
        )
        lp = pbar.ravel() @ jvp.ravel()
        ld = dtheta @ mine
        results.append(
            row(
                "⟨p̄, Jvp dθ⟩ = ⟨Vjp p̄, dθ⟩ (adjoint consistency)",
                abs(lp - ld) <= 1e-8 * max(1, abs(lp)),
                f"{lp:.6f} vs {ld:.6f}",
            )
        )

        # 6. a parameter change keeps the topology: refresh, don't rediscover
        m3 = zs.Model()
        m3.CopyFrom(model)
        m3.theta[1] += 0.05  # boss radius
        disc2 = stub.Discover(zs.DiscoverRequest(model=m3, domain=domain, tolerance=tol)).topology
        results.append(
            row(
                "a small parameter change leaves the topology hash unchanged",
                disc2.hash == topo.hash,
                f"{disc2.hash} vs {topo.hash}",
            )
        )
    finally:
        proc.terminate()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
