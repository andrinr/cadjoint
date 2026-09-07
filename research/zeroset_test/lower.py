"""Lower a cadjoint SDF graph into a zero-set Model, as proto JSON.

Runs in the cadjoint venv. Emits `model.json` (the Model in proto3 JSON,
the standard language-neutral encoding) and `reference.json`: cadjoint's
own field values and parameter gradients at random points, from JAX, for
the other side to check itself against.

    .venv/bin/python research/zeroset_test/lower.py OUT [scene.py] [--per-node]

The lowering itself is :func:`cadjoint.zeroset.lower`; this script only adds
the JAX reference the other side checks against.
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
from cadjoint.zeroset import lower

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
    low = lower(root)
    model = low.to_dict()
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
        low = lower(node)
        model = low.to_dict()
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
