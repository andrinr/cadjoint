"""Export a cadjoint scene as a JSON scene graph, plus Python references.

Usage: python export_scene.py SCENE.py NAME --bounds X0 Y0 Z0 X1 Y1 Z1 --spacing S

Writes, next to this file:
  NAME.scene.json      the scene graph: node type (a kernel name), params,
                       children -- what src/runtime.ts composes kernels from
  NAME.reference.json  field values at 2000 points, and d(sum)/d(k) for the
                       first blended Union's smoothness by jax.grad
  NAME.mesh.json       cadjoint's extract_mesh on the given lattice: counts,
                       volume, bad edges
"""

from __future__ import annotations

import argparse
import json
import runpy
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint import extract_parameters, functionalize
from cadjoint.meshing import GridSpec, extract_mesh
from cadjoint.sdf.base import SDF

HERE = Path(__file__).parent

#: cadjoint class -> (kernel kind, parameter names in kernel order)
NODES = {
    "Sphere": ["radius"],
    "Box": ["size"],
    "RoundBox": ["size", "radius"],
    "Cylinder": ["radius", "height"],
    "Translate": ["offset"],
    "Rotate": ["axis", "angle"],
    "Offset": ["distance"],
    "Shell": ["thickness"],
    "Union": ["smoothness"],
    "Difference": ["smoothness"],
    "Intersection": ["smoothness"],
}
SNAKE = {"RoundBox": "round_box"}


def _children(node) -> list:
    for attr in ("sdf", "sdfs"):
        value = getattr(node, attr, None)
        if isinstance(value, SDF):
            return [value]
        if isinstance(value, (list, tuple)) and value and all(isinstance(c, SDF) for c in value):
            return list(value)
    return []


def _value(parameter):
    value = getattr(parameter, "xyz", None)
    value = parameter.value if value is None else value
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else array.tolist()


def to_graph(node) -> dict:
    kind = type(node).__name__
    if kind == "ExtrudedPolygon":
        for unused in ("draft", "twist"):
            if unused in node.params and float(node.params[unused].value) != 0.0:
                raise NotImplementedError(f"ExtrudedPolygon with {unused}")
        names = sorted((k for k in node.params if k[0] == "v" and k[1:].isdigit()), key=lambda k: int(k[1:]))
        kernel, order = f"extruded_polygon_{len(names)}", ["depth", *names]
    elif kind in NODES:
        kernel, order = SNAKE.get(kind, kind.lower()), NODES[kind]
    else:
        raise NotImplementedError(f"no kernel for {kind}")
    return {
        "type": kernel,
        "params": {name: _value(node.params[name]) for name in order},
        "children": [to_graph(child) for child in _children(node)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("name")
    parser.add_argument("--bounds", type=float, nargs=6, required=True)
    parser.add_argument("--spacing", type=float, required=True)
    args = parser.parse_args()
    scene = runpy.run_path(args.scene, run_name="__export__")["scene"]

    graph = to_graph(scene)
    (HERE / f"{args.name}.scene.json").write_text(json.dumps(graph))

    low, high = np.array(args.bounds[:3]), np.array(args.bounds[3:])
    rng = np.random.default_rng(0)
    points = rng.uniform(low, high, size=(2000, 3)).astype(np.float32)
    values = np.asarray(scene(jnp.asarray(points)), dtype=np.float64)

    # jax.grad w.r.t. the first blended Union's smoothness, through
    # cadjoint's own functional form of the scene.
    free, fixed, _ = extract_parameters(scene)
    sdf_of = functionalize(scene)
    blends = [k for k, v in fixed.items() if k.endswith("smoothness") and float(v) > 0]
    gradient = None
    if blends:
        key = blends[0]

        def total(k):
            return jnp.sum(sdf_of(free, {**fixed, key: k})(jnp.asarray(points)))

        gradient = {"param": key, "d_sum": float(jax.grad(total)(jnp.float32(fixed[key])))}
    (HERE / f"{args.name}.reference.json").write_text(json.dumps({
        "points": points.ravel().tolist(), "values": values.tolist(), "grad": gradient,
    }))

    cells = tuple(int(round(v)) for v in (high - low) / args.spacing)
    grid = GridSpec.from_bounds(tuple(low), tuple(np.array(cells) * args.spacing), cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = extract_mesh(lambda p: jnp.asarray(scene(p)), grid)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v.astype(np.float32).tofile(HERE / f"{args.name}.mesh_vertices.bin")
    np.asarray(mesh.cells, dtype=np.int32).tofile(HERE / f"{args.name}.mesh_cells.bin")
    f = np.asarray(mesh.faces)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    (HERE / f"{args.name}.mesh.json").write_text(json.dumps({
        "origin": low.tolist(), "spacing": args.spacing, "cells": list(cells),
        "vertices": int(v.shape[0]), "triangles": int(f.shape[0]),
        "volume": float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6),
        "bad_edges": int((counts != 2).sum()),
    }))
    print(args.name, "nodes:", json.dumps(graph).count('"type"'), "grad:", gradient,
          "mesh:", int(v.shape[0]), int(f.shape[0]), "bad", int((counts != 2).sum()))


if __name__ == "__main__":
    main()
