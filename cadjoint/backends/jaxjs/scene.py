"""A cadjoint SDF tree as a JSON scene graph: the contract with the browser runtime.

The graph is what a static page needs to evaluate a scene with the generated
kernels, and nothing else::

    {"format": "cadjoint-scene", "version": 1,
     "root": {"type": "union", "params": {"smoothness": 0.5},
              "children": [{"type": "sphere", "params": {"radius": 1.0}, "children": []},
                           ...]}}

``type`` is a :data:`~cadjoint.backends.jaxjs.kernels.NODE_TYPES` name,
``params`` hold plain numbers and lists under the kernel parameter names,
and profiles store their vertices as one ``vertices`` list of ``[x, y]``
pairs.  A browser-side scene API builds the same JSON, so this format, not
either language's classes, is what both sides test against.

Version 2 is a superset of version 1 that also carries what a renderer needs
and an evaluator does not: a stable id per node, the declared names of free
parameters, and each primitive's material.  It is written only on request
(``scene_graph(sdf, version=2)``), and :func:`graph_sdf` reads both.  The
format is specified in ``SCENE_FORMAT.md`` next to this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadjoint.sdf.base import SDF
from cadjoint.sdf.primitives.loft import LoftedPolygon
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon, RevolvedPolygon

from .kernels import NODE_TYPES, POLYGON_BUCKETS

__all__ = [
    "SCENE_FORMAT",
    "SCENE_VERSION",
    "SCENE_VERSIONS",
    "UnsupportedNode",
    "graph_sdf",
    "scene_graph",
]

SCENE_FORMAT = "cadjoint-scene"
#: The version :func:`scene_graph` writes by default, and the one the
#: generated jax-js runtime checks for.
SCENE_VERSION = 1
#: Every version :func:`graph_sdf` reads.  Each adds fields and changes none.
SCENE_VERSIONS = (1, 2)

#: The material properties a version 2 primitive carries: the two rows a
#: shader's material block holds.
MATERIAL_KEYS = ("color", "roughness", "metallic", "opacity", "ior", "reflectivity")

_BY_CLASS = {node.cls: node for node in NODE_TYPES.values()}


class UnsupportedNode(NotImplementedError):
    """A node the jax-js backend has no kernels for."""


def _children(node) -> list:
    # Instance attributes only: every node class also has an ``sdf`` staticmethod.
    for attr in ("sdfs", "sdf"):
        value = vars(node).get(attr)
        if value is None:
            continue
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        for child in values:
            if not isinstance(child, SDF):
                raise UnsupportedNode(
                    f"{type(node).__name__} has a child that is not an SDF node ({child!r})"
                )
        return values
    return []


def _plain(value) -> Any:
    xyz = getattr(value, "xyz", None)
    value = xyz if xyz is not None else getattr(value, "value", value)
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else array.tolist()


def _vertices(params: dict, prefix: str) -> list[list[float]]:
    names = sorted(
        (k for k in params if k.startswith(prefix) and k[len(prefix) :].isdigit()),
        key=lambda k: int(k[len(prefix) :]),
    )
    if len(names) > POLYGON_BUCKETS[-1]:
        raise UnsupportedNode(
            f"profile with {len(names)} vertices; the largest kernel takes {POLYGON_BUCKETS[-1]}"
        )
    return [_plain(params[k]) for k in names]


def _params(sdf, node) -> dict:
    params = sdf.params
    if isinstance(sdf, ExtrudedPolygon):
        out = {
            "depth": _plain(params["depth"]),
            "draft": _plain(params["draft"]) if "draft" in params else 0.0,
            "twist": _plain(params["twist"]) if "twist" in params else 0.0,
            "vertices": _vertices(params, "v"),
        }
    elif isinstance(sdf, RevolvedPolygon):
        out = {"offset": _plain(params["offset"]), "vertices": _vertices(params, "v")}
    elif isinstance(sdf, LoftedPolygon):
        out = {
            "height": _plain(params["height"]),
            "vertices": _vertices(params, "v"),
            "vertices_b": _vertices(params, "w"),
        }
    else:
        out = {}
        for name in node.params + node.static:
            if name not in params:
                raise UnsupportedNode(f"{type(sdf).__name__} has no parameter {name!r}")
            out[name] = _plain(params[name])
        for name in node.static:
            out[name] = int(round(out[name]))
    return out


def _free(params: dict) -> dict:
    """``{graph parameter: declared name}`` for the free parameters of a node.

    A profile's vertices map to a list with one entry per vertex, ``None``
    for a fixed one.
    """

    def declared(parameter):
        if getattr(parameter, "free", False) and getattr(parameter, "name", None):
            return str(parameter.name)
        return None

    out: dict = {}
    for prefix, key in (("v", "vertices"), ("w", "vertices_b")):
        names = sorted(
            (k for k in params if k.startswith(prefix) and k[len(prefix) :].isdigit()),
            key=lambda k: int(k[len(prefix) :]),
        )
        names_free = [declared(params[k]) for k in names]
        if any(names_free):
            out[key] = names_free
    for name, parameter in params.items():
        if name[:1] in "vw" and name[1:].isdigit():
            continue
        if declared(parameter) is not None:
            out[name] = declared(parameter)
    return out


def _material(sdf) -> dict:
    values = sdf.material_at(np.zeros(3, dtype=np.float64))
    out = {}
    for key in MATERIAL_KEYS:
        array = np.asarray(values[key], dtype=np.float64).reshape(-1)
        # JSON has no NaN; a reader turns null back into one.
        items = [float(v) if np.isfinite(v) else None for v in array]
        out[key] = items if key == "color" else items[0]
    return out


class _Writer:
    """Writes the nodes of one graph, numbering them in first-visit order."""

    def __init__(self, version: int) -> None:
        self.version = version
        self.ids: dict[int, str] = {}

    def node(self, sdf) -> dict:
        node = _BY_CLASS.get(type(sdf))
        if node is None:
            raise UnsupportedNode(f"no jax-js kernels for {type(sdf).__name__}")
        out = {"type": node.name, "params": _params(sdf, node)}
        if self.version >= 2:
            # A node reached twice keeps its first id, so a reader can tell a
            # shared subtree from two equal ones and emit it once.
            out["id"] = self.ids.setdefault(id(sdf), f"n{len(self.ids)}")
            free = _free(sdf.params)
            if free:
                out["free"] = free
            implicit = [
                name
                for name in ("draft", "twist")
                if isinstance(sdf, ExtrudedPolygon) and name not in sdf.params
            ]
            if implicit:
                out["implicit"] = implicit
            if node.kind == "primitive":
                out["material"] = _material(sdf)
        out["children"] = [self.node(c) for c in _children(sdf)]
        return out


def scene_graph(sdf, version: int = SCENE_VERSION) -> dict:
    """The scene graph of a cadjoint SDF tree.

    Args:
        sdf: The root node.
        version: The format version to write, one of :data:`SCENE_VERSIONS`.

    Returns:
        ``{"format", "version", "root"}``, JSON-serialisable.

    Raises:
        UnsupportedNode: If a node has no jax-js kernels, or a child is a
            plain callable rather than an SDF node.
    """
    if version not in SCENE_VERSIONS:
        raise ValueError(f"{SCENE_FORMAT} has versions {SCENE_VERSIONS}, not {version}")
    return {"format": SCENE_FORMAT, "version": version, "root": _Writer(version).node(sdf)}


def graph_sdf(graph: dict):
    """A scene graph as a Python field, through each node type's own ``sdf``.

    The same wiring the browser runtime does, in JAX: the reference a scene
    graph is checked against when no cadjoint object exists for it, and the
    proof (by comparison with the object) that :func:`scene_graph` loses
    nothing.  Parameter values are read when the field is called, so a
    tracer placed in ``params`` is differentiated through.

    Args:
        graph: A :func:`scene_graph` result.

    Returns:
        ``p (..., 3) -> distance (...)``.
    """
    import jax.numpy as jnp

    if graph.get("format") != SCENE_FORMAT or graph.get("version") not in SCENE_VERSIONS:
        raise ValueError(f"not a {SCENE_FORMAT} graph of a version in {SCENE_VERSIONS}")

    def build(node: dict):
        spec = NODE_TYPES[node["type"]]
        children = [build(c) for c in node["children"]]
        params = node["params"]
        arr = lambda name: jnp.asarray(params[name], dtype=jnp.float32)  # noqa: E731

        def profile(name: str, prefix: str) -> dict:
            return {
                f"{prefix}{i}": jnp.asarray(v, dtype=jnp.float32)
                for i, v in enumerate(params[name])
            }

        if spec.cls is ExtrudedPolygon:
            # Plain floats for a zero draft/twist, so the plain form is taken
            # exactly as the object takes it.
            def extra(name):
                value = params[name]
                return value if isinstance(value, float) and value == 0.0 else arr(name)

            return lambda p: ExtrudedPolygon.sdf(
                p,
                arr("depth"),
                draft=extra("draft"),
                twist=extra("twist"),
                **profile("vertices", "v"),
            )
        if spec.cls is RevolvedPolygon:
            return lambda p: RevolvedPolygon.sdf(p, arr("offset"), **profile("vertices", "v"))
        if spec.cls is LoftedPolygon:
            return lambda p: LoftedPolygon.sdf(
                p, arr("height"), **profile("vertices", "v"), **profile("vertices_b", "w")
            )
        if spec.kind == "primitive":
            return lambda p: spec.cls.sdf(p, *[arr(n) for n in spec.params])
        statics = [params[n] for n in spec.static]
        if spec.nary:
            return lambda p: spec.cls.sdf(tuple(children), p, *[arr(n) for n in spec.params])
        if spec.kind == "pattern":
            return lambda p: spec.cls.sdf(children[0], p, *_pattern_args(spec, arr, statics))
        return lambda p: spec.cls.sdf(children[0], p, *[arr(n) for n in spec.params])

    return build(graph["root"])


def _pattern_args(spec, arr, statics) -> list:
    count, skip_mask = statics
    if spec.name == "linear_pattern":
        return [arr("direction"), arr("spacing"), count, skip_mask]
    return [count, arr("origin"), arr("direction"), skip_mask]
