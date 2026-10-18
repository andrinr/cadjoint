"""The evaluated scene tree: what ``scene`` is built from, as the Objects panel shows it.

The construction payload lists the objects a program built and the element
payload lists the calls it wrote; neither says which boolean consumes which
solid.  That is only known once the program has run, by walking the SDF
graph ``scene`` evaluates to.  This module walks it and names every graph
node in the vocabulary the viewer already speaks:

- a **node** — the placed SDF a sketch feature or a ``Solid`` primitive
  generated — carries the construction node's id (``profile_3``, ``box_0``);
- a **boolean** carries the id of the boolean element the source map found
  on the line it was constructed on, when there is one;
- a **transform** (a pattern, a mirror, a shell) keeps its one operand;
- anything else is an **sdf** leaf, named after its class.

Placement wrappers (``Translate``/``Rotate``/``Scale``) are folded into what
they wrap: they are how a primitive or a hole is positioned, not an object
of their own, and the tree would otherwise be mostly ``Translate``.

The graph is a DAG, not a tree — ``Union(bolt, Mirror(bolt, "y"))`` reads
``bolt`` twice — so the result is a flat table of items keyed by id with
operand ids, and an object reached twice is one item referenced twice.
"""

from __future__ import annotations

from typing import Any


def _variable_names(namespace: dict[str, Any]) -> dict[int, str]:
    """The first module-level name each object is bound to, by ``id``."""
    names: dict[int, str] = {}
    for name, value in namespace.items():
        if name.startswith("_") or name == "scene":
            continue
        names.setdefault(id(value), name)
    return names


def build_scene_tree(
    scene: Any,
    profiles: list[tuple[object, int | None]],
    graph: dict[str, list],
    elements: list[dict[str, Any]],
    namespace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Walk the SDF graph *scene* evaluates to into the viewer's tree table.

    Args:
        scene: The program's ``scene`` value: one SDF, or a list/tuple of them.
        profiles: The ``(object, line)`` pairs from
            :func:`~cadjoint.viewer.source_map.capture.capture_profiles`, in
            the order that gives each construction node its id.
        graph: What :func:`~cadjoint.viewer.source_map.capture.capture_scene_graph`
            recorded while the program ran.
        elements: The element payload, for the boolean elements' ids.
        namespace: The program's namespace, so an object bound to a variable
            is labelled by it.

    Returns:
        ``{"roots": [...], "items": [...]}``: the ids ``scene`` evaluates to,
        and one item per graph node reachable from them, each with its
        ``operands`` as item ids in argument order.
    """
    from cadjoint.construction.solid import ConstructionPrimitive
    from cadjoint.sdf.base import SDF
    from cadjoint.sdf.boolean import BooleanOp
    from cadjoint.sdf.transforms import Rotate, Scale, Translate

    # Which construction node each generated solid belongs to, and — for a
    # loft — the other sketches it consumed, which nest under it.
    node_ids = {
        id(obj): (
            f"{obj.kind}_{index}" if isinstance(obj, ConstructionPrimitive) else f"profile_{index}"
        )
        for index, (obj, _) in enumerate(profiles)
    }
    node_of: dict[int, str] = {}
    consumed: dict[int, list[str]] = {}
    for primitive, placed in graph.get("primitives", []):
        if id(primitive) in node_ids:
            node_of.setdefault(id(placed), node_ids[id(primitive)])
    for obj, _ in profiles:
        if isinstance(obj, ConstructionPrimitive):
            continue
        for feature in getattr(obj, "features", None) or []:
            if feature.solid is None:
                continue
            node_of.setdefault(id(feature.solid), node_ids[id(obj)])
            others = [
                node_ids[id(p)] for p in feature.profiles if p is not obj and id(p) in node_ids
            ]
            if others:
                consumed.setdefault(id(feature.solid), others)

    # Which boolean element each runtime boolean was built by.  One call in a
    # loop builds several booleans; they all point at the same element.
    by_line: dict[tuple[str, int], list[str]] = {}
    for element in elements:
        if element.get("kind") == "boolean":
            by_line.setdefault((element["call"], element["line"]), []).append(element["id"])
    element_of: dict[int, str] = {}
    seen_on_line: dict[tuple[str, int], int] = {}
    for boolean, line in graph.get("booleans", []):
        if line is None:
            continue
        key = (type(boolean).__name__, line)
        candidates = by_line.get(key)
        if not candidates:
            continue
        # Two calls of one kind on one line: pair them in construction order,
        # which is inner-first, as the source order of nested calls is not.
        count = seen_on_line.get(key, 0)
        seen_on_line[key] = count + 1
        element_of[id(boolean)] = candidates[min(count, len(candidates) - 1)]

    variables = _variable_names(namespace or {})
    items: list[dict[str, Any]] = []
    item_of: dict[int, str] = {}
    used_ids: dict[str, int] = {}

    def unique(base: str) -> str:
        count = used_ids.get(base, 0)
        used_ids[base] = count + 1
        return base if count == 0 else f"{base}#{count + 1}"

    def visit(obj: Any, outer_name: str | None = None) -> str:
        if id(obj) in item_of:
            return item_of[id(obj)]
        name = variables.get(id(obj)) or outer_name
        node_id = node_of.get(id(obj))
        if node_id is None and isinstance(obj, (Translate, Rotate, Scale)):
            # Folded: the placement is not an object, the thing placed is.
            child_id = visit(obj.sdf, name)
            item_of[id(obj)] = child_id
            return child_id
        call = type(obj).__name__
        operands_of: list[Any] = []
        if node_id is not None:
            kind, element, base = "node", None, node_id
        elif isinstance(obj, BooleanOp):
            element = element_of.get(id(obj))
            kind, base = "boolean", element or f"boolean:{call}@{id(obj):x}"
            operands_of = list(obj.children())
        else:
            children = [child for child in obj.children() if isinstance(child, SDF)]
            element = None
            kind = "transform" if children else "sdf"
            base = f"{kind}:{name or call}"
            operands_of = children
        item_id = unique(base)
        item_of[id(obj)] = item_id
        user_name = getattr(obj, "name", None) if kind == "boolean" else None
        item: dict[str, Any] = {
            "id": item_id,
            "kind": kind,
            "call": call,
            "name": user_name if isinstance(user_name, str) else name,
            "node": node_id,
            "element": element,
            "operands": [],
        }
        items.append(item)
        item["operands"] = [visit(child) for child in operands_of]
        for other in consumed.get(id(obj), []):
            item["operands"].append(sketch(other))
        return item_id

    sketch_items: dict[str, str] = {}

    def sketch(node_id: str) -> str:
        """The item for a sketch reached only as a feature's second input."""
        if node_id not in sketch_items:
            item_id = unique(node_id)
            sketch_items[node_id] = item_id
            items.append(
                {
                    "id": item_id,
                    "kind": "node",
                    "call": "PolygonProfile",
                    "name": None,
                    "node": node_id,
                    "element": None,
                    "operands": [],
                }
            )
        return sketch_items[node_id]

    values = scene if isinstance(scene, (list, tuple)) else [scene]
    roots = [visit(value) for value in values if isinstance(value, SDF)]
    # Synthetic ids of unmatched booleans hold a memory address; renumber
    # them so the payload is deterministic for an unchanged program.
    renamed: dict[str, str] = {}
    counter = 0
    for item in items:
        if item["kind"] == "boolean" and item["element"] is None:
            renamed[item["id"]] = f"boolean:{item['call']}#{counter}"
            counter += 1
    if renamed:
        for item in items:
            item["id"] = renamed.get(item["id"], item["id"])
            item["operands"] = [renamed.get(op, op) for op in item["operands"]]
        roots = [renamed.get(root, root) for root in roots]
    return {"roots": roots, "items": items}


__all__ = ["build_scene_tree"]
