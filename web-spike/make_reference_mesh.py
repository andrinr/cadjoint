"""cadjoint's own extract_mesh on the lattice the browser page samples.

Origin (-42, -42, -42), 159 cells of 84/159 per axis: the 160^3 vertex
lattice in src/main.ts. Writes counts, volume and the edge check, so the
TS port can be compared number for number.
"""
import json
import sys
import warnings

import numpy as np
import jax.numpy as jnp

from cadjoint.meshing import GridSpec, extract_mesh

sys.path.insert(0, sys.argv[1])
src = open(f"{sys.argv[1]}/make_reference.py").read().split("rng = np.random")[0]
ns = {}
exec(src.replace('jax.config.update("jax_enable_x64", True)', ""), ns)
scene = ns["node"]()
grid = GridSpec.from_bounds((-42.0, -42.0, -42.0), (84.0, 84.0, 84.0), 159)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mesh = extract_mesh(lambda p: jnp.asarray(scene(p)), grid)
v = np.asarray(mesh.vertices, dtype=np.float64)
f = np.asarray(mesh.faces)
volume = float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6)
edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
_, counts = np.unique(edges, axis=0, return_counts=True)
out = {"vertices": int(v.shape[0]), "triangles": int(f.shape[0]), "volume": volume,
       "bad_edges": int((counts != 2).sum())}
json.dump(out, open(f"{sys.argv[1]}/reference_mesh.json", "w"))
print(out)
