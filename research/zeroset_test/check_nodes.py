"""Per-class fidelity: each node class lowered alone against cadjoint's values."""

import json
import sys
from pathlib import Path

import backend as ref
import numpy as np
import zero_set_pb2 as zs
from google.protobuf import json_format

rows = []
for d in sorted(Path(sys.argv[1]).joinpath("nodes").iterdir(), key=lambda d: (len(d.name), d.name)):
    model = json_format.Parse((d / "model.json").read_text(), zs.Model())
    r = json.loads((d / "reference.json").read_text())
    pts = np.array(r["points"])
    fv = np.array(r["values"])
    try:
        v = ref.Walker(model, np.array(model.theta), derivatives=False).shape(model.root, pts).v
    except Exception as e:  # noqa: BLE001
        print(f"  {d.name:18s} nodes {len(model.node):5d}  ERROR {type(e).__name__}: {e}")
        continue
    err = np.abs(v - fv)
    sign = np.mean(np.sign(v) == np.sign(fv))
    rows.append((d.name, len(model.node), err.max(), sign, r.get("children", [])))
bad = [row for row in rows if row[2] >= 1e-5]
print(f"{len(rows)} instances, {len(rows) - len(bad)} exact, {len(bad)} mismatching")
for name, n, e, _sgn, ch in bad:
    print(f"  {name:32s} nodes {n:5d}  max|Δf| {e:.1e}  children: {ch}")
