"""The evaluated scene tree: which boolean consumes which solid, by payload id."""

from __future__ import annotations

from cadjoint.viewer.schema.payloads import SceneTree
from cadjoint.viewer.source_map import PLAYGROUND_FILENAME, capture_profiles
from cadjoint.viewer.source_map.capture import capture_scene_graph
from cadjoint.viewer.source_map.elements import build_element_payload
from cadjoint.viewer.source_map.tree import build_scene_tree

PROGRAM = """from cadjoint.construction import PolygonProfile, Solid, extrude, loft
from cadjoint.sdf.boolean import Difference, Union
from cadjoint.sdf.transforms.fields import Mirror
from cadjoint.sdf.transforms.patterns import LinearPattern

plate = Solid.box(size=[2.0, 2.0, 0.2], name="plate")
pin_sketch = PolygonProfile([[0.5, 0.5], [0.7, 0.5], [0.7, 0.7]], name="pin")
pin = extrude(pin_sketch, depth=0.5)
pins = Union(pin, Mirror(pin, "y"))
foot = PolygonProfile([[0, 0], [1, 0], [1, 1], [0, 1]], name="foot")
top = PolygonProfile([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], name="top")
tower = loft(foot, top, height=1.0)
hole = Solid.cylinder(radius=0.1, height=1.0, name="hole")
holes = LinearPattern(hole, direction=[1.0, 0.0, 0.0], count=3, spacing=0.3)
spare = Solid.sphere(radius=0.2, name="spare")
body = Difference(Union(plate, pins, tower), holes)
scene = Union(body, plate | spare)
"""


def tree_of(source: str) -> dict:
    namespace = {"__builtins__": __builtins__, "__name__": "__cadjoint_playground__"}
    with (
        capture_profiles(PLAYGROUND_FILENAME) as profiles,
        capture_scene_graph(PLAYGROUND_FILENAME) as graph,
    ):
        exec(compile(source, PLAYGROUND_FILENAME, "exec"), namespace, namespace)
    tree = build_scene_tree(
        namespace["scene"], profiles, graph, build_element_payload(source), namespace
    )
    SceneTree.model_validate(tree)
    return tree


def test_booleans_list_their_operands_by_payload_id():
    tree = tree_of(PROGRAM)
    items = {item["id"]: item for item in tree["items"]}
    assert tree["roots"] == ["boolean:scene"]
    body = items[items["boolean:scene"]["operands"][0]]
    assert body["id"] == "boolean:body"
    assert body["element"] == "boolean:body"
    inner, holes = (items[operand] for operand in body["operands"])
    # The nested, unbound Union pairs with the element on its line.
    assert inner["kind"] == "boolean" and inner["element"] == "boolean@16"
    assert inner["operands"][0] == "box_0"
    assert items["box_0"] == {
        "id": "box_0",
        "kind": "node",
        "call": "Translate",
        "name": "plate",
        "node": "box_0",
        "element": None,
        "operands": [],
    }
    # A pattern keeps its operand; the cylinder's placement is folded away.
    assert holes["kind"] == "transform" and holes["name"] == "holes"
    assert items[holes["operands"][0]]["node"] == "cylinder_4"


def test_a_shared_operand_is_one_item_read_twice():
    tree = tree_of(PROGRAM)
    items = {item["id"]: item for item in tree["items"]}
    pins = items["boolean:pins"]
    pin, mirror = pins["operands"]
    assert items[pin]["node"] == "profile_1"
    assert items[mirror]["call"] == "Mirror"
    assert items[mirror]["operands"] == [pin]
    # The plate is read by the inner Union and by the `|` at the end.
    readers = [item for item in tree["items"] if "box_0" in item["operands"]]
    assert len(readers) == 2
    assert len([item for item in tree["items"] if item["node"] == "box_0"]) == 1


def test_a_loft_nests_its_second_sketch_and_operators_have_unmatched_ids():
    tree = tree_of(PROGRAM)
    items = {item["id"]: item for item in tree["items"]}
    tower = next(item for item in tree["items"] if item["node"] == "profile_2")
    assert [items[operand]["node"] for operand in tower["operands"]] == ["profile_3"]
    # `plate | spare` has no Union call in the source: a synthetic, stable id.
    operator_union = items[items["boolean:scene"]["operands"][1]]
    assert operator_union["element"] is None
    assert operator_union["id"] == "boolean:Union#0"
    assert [items[operand]["node"] for operand in operator_union["operands"]] == [
        "box_0",
        "sphere_5",
    ]


def test_ids_are_deterministic_across_runs():
    assert tree_of(PROGRAM) == tree_of(PROGRAM)
