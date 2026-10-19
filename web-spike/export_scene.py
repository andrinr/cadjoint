"""Export a cadjoint scene for the browser, with the Python references it is checked against.

Usage: python export_scene.py SCENE.py|SCENE.scene.json NAME [--cells N] [--bounds X0 Y0 Z0 X1 Y1 Z1 --spacing S]

Writes, next to this file:
  NAME.scene.json      the scene graph (cadjoint.backends.jaxjs.scene_graph)
  NAME.reference.json  field values at 2000 points in the part's bounds, and
                       d(sum)/d(param) by jax.grad for one scalar parameter
                       (the `free` entry names the node by type and ordinal)
  NAME.mesh.json       cadjoint's extract_mesh on a lattice: by default the
                       one a real export uses (fit_bounds, N cells along the
                       longest side); counts, volume, bad edges
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint.backends.jaxjs import graph_sdf, scene_graph
from cadjoint.backends.jaxjs.scene import _BY_CLASS, _children
from cadjoint.meshing import GridSpec, extract_mesh
from cadjoint.meshing.bounds import fit_bounds

HERE = Path(__file__).parent

#: Parameters to differentiate, in order of preference: the first node (in
#: depth-first order) of the first type here that has a positive, unshared
#: scalar parameter of that name.
FREE_CANDIDATES = [
    ("union", "smoothness"),
    ("difference", "smoothness"),
    ("cylinder", "radius"),
    ("extruded_polygon", "depth"),
    ("box", "size"),
]


def _nodes(root):
    out = [root]
    for child in _children(root):
        out.extend(_nodes(child))
    return out


def _choose_free(root):
    nodes = _nodes(root)
    uses: dict[int, int] = {}
    for n in nodes:
        for p in n.params.values():
            uses[id(p)] = uses.get(id(p), 0) + 1
    for type_name, name in FREE_CANDIDATES:
        ordinal = -1
        for n in nodes:
            if _BY_CLASS.get(type(n)) is None or _BY_CLASS[type(n)].name != type_name:
                continue
            ordinal += 1
            param = n.params.get(name)
            if param is None or uses[id(param)] != 1:
                continue
            value = np.asarray(param.value)
            if value.ndim == 0 and float(value) > 0:
                return {"type": type_name, "name": name, "index": ordinal}, param
    return None, None


class _GraphParam:
    """A graph node's parameter, with the `.value` interface of a cadjoint Parameter."""

    def __init__(self, params: dict, name: str):
        self.params, self.name = params, name

    @property
    def value(self):
        return self.params[self.name]

    @value.setter
    def value(self, v):
        self.params[self.name] = v


def _choose_free_in_graph(graph):
    def walk(node):
        yield node
        for child in node["children"]:
            yield from walk(child)

    for type_name, name in FREE_CANDIDATES:
        ordinal = -1
        for node in walk(graph["root"]):
            if node["type"] != type_name:
                continue
            ordinal += 1
            value = node["params"].get(name)
            if isinstance(value, float) and value > 0:
                return {"type": type_name, "name": name, "index": ordinal}, _GraphParam(
                    node["params"], name
                )
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("name")
    parser.add_argument("--cells", type=int, default=96, help="cells along the longest side")
    parser.add_argument("--bounds", type=float, nargs=6)
    parser.add_argument("--spacing", type=float)
    args = parser.parse_args()
    if args.scene.endswith(".json"):
        # A graph with no cadjoint object behind it (the spike's node and
        # clevis): the reference is the graph through cadjoint's own sdfs.
        scene = None
        graph = json.loads(Path(args.scene).read_text())
        field = graph_sdf(graph)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = runpy.run_path(args.scene, run_name="__export__")["scene"]
        field = lambda p: jnp.asarray(scene(p))  # noqa: E731
        graph = scene_graph(scene)
    (HERE / f"{args.name}.scene.json").write_text(json.dumps(graph))

    if args.bounds:
        low, high = np.array(args.bounds[:3]), np.array(args.bounds[3:])
        spacing = args.spacing
        cells = tuple(int(round(v)) for v in (high - low) / spacing)
    else:
        origin, size = fit_bounds(field)
        low, size = np.array(origin), np.array(size)
        spacing = float(np.float32(size.max() / args.cells))
        cells = tuple(int(math.ceil(v / spacing)) for v in size)
        high = low + np.array(cells) * spacing

    rng = np.random.default_rng(0)
    points = rng.uniform(low, high, size=(2000, 3)).astype(np.float32)
    values = np.asarray(field(jnp.asarray(points)), dtype=np.float64)

    # jax.grad through the node's own Parameter: the scene reads .value at
    # call time, so a tracer put there is what the scene differentiates.
    free, param = _choose_free(scene) if scene is not None else _choose_free_in_graph(graph)
    gradient = None
    if free is not None:
        original = param.value

        def total(k):
            param.value = k
            try:
                return jnp.sum(field(jnp.asarray(points)))
            finally:
                param.value = original

        free["value"] = float(original)
        free["d_sum"] = float(jax.grad(total)(jnp.float32(original)))
        gradient = free
    (HERE / f"{args.name}.reference.json").write_text(
        json.dumps(
            {
                "points": points.ravel().tolist(),
                "values": values.tolist(),
                "grad": gradient,
            }
        )
    )

    grid = GridSpec.from_bounds(tuple(low), tuple(np.array(cells) * spacing), cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = extract_mesh(field, grid)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v.astype(np.float32).tofile(HERE / f"{args.name}.mesh_vertices.bin")
    f = np.asarray(mesh.faces)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    (HERE / f"{args.name}.mesh.json").write_text(
        json.dumps(
            {
                "origin": [float(x) for x in low],
                "spacing": spacing,
                "cells": list(cells),
                "vertices": int(v.shape[0]),
                "triangles": int(f.shape[0]),
                "volume": float(
                    np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6
                ),
                "bad_edges": int((counts != 2).sum()),
            }
        )
    )
    print(
        args.name,
        "grad:",
        gradient and {k: gradient[k] for k in ("type", "name", "index", "d_sum")},
        "cells:",
        cells,
        "mesh:",
        int(v.shape[0]),
        int(f.shape[0]),
        "bad",
        int((counts != 2).sum()),
    )


if __name__ == "__main__":
    main()
