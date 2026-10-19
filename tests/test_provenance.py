"""The construction-capture hook: which program line built which object."""

from __future__ import annotations

import inspect
import threading

from cadjoint.construction.sketch import PolygonProfile
from cadjoint.construction.solid import ConstructionPrimitive, Solid
from cadjoint.provenance import Site, capture_constructions
from cadjoint.sdf.boolean import Difference, Intersection, Union, Xor
from cadjoint.sdf.primitives import Box, Sphere

FILENAME = "<provenance-test>"

PROGRAM = """\
from cadjoint.construction.sketch import PolygonProfile
from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import Box, Sphere

profile = PolygonProfile([(0, 0), (1, 0), (0, 1)])
circle = PolygonProfile.circle(0.5)
both = Union(Box([0.5, 0.5, 0.5]), Sphere(0.5))
piped = Box([0.5, 0.5, 0.5]) | Sphere(0.5)
"""


def _run(program: str = PROGRAM):
    events: list[tuple[object, Site]] = []
    with capture_constructions(FILENAME, constructed=lambda obj, site: events.append((obj, site))):
        namespace: dict = {}
        exec(compile(program, FILENAME, "exec"), namespace)  # noqa: S102 - a test program
    return events, namespace


def test_each_marked_object_is_reported_with_its_line_in_construction_order():
    events, namespace = _run()
    by_object = {id(obj): site for obj, site in events}
    assert by_object[id(namespace["profile"])] == Site(5, True)
    # Built inside a library helper: the program line, but not a direct call.
    assert by_object[id(namespace["circle"])] == Site(6, False)
    assert by_object[id(namespace["both"])] == Site(7, True)
    assert by_object[id(namespace["piped"])] == Site(8, False)
    # Unmarked classes (the primitives) are not reported.
    assert {type(obj) for obj, _ in events} <= {PolygonProfile, Union}


def test_an_object_is_reported_once_fully_initialised():
    seen = []

    def constructed(obj, _site):
        seen.append((obj, dict(obj.params)))

    with capture_constructions(FILENAME, constructed=constructed):
        union = Union(Box([0.5, 0.5, 0.5]), Sphere(0.5), smoothness=0.0)
    ((obj, params),) = seen
    assert obj is union
    # The SDF wrapper's parameter cast had already run.
    assert hasattr(params["smoothness"], "value")


def test_nothing_is_reported_outside_a_block_or_from_another_file():
    events, _ = _run()
    assert events
    Union(Box([0.5, 0.5, 0.5]), Sphere(0.5))  # after the block: no listener left
    with capture_constructions("<elsewhere>", constructed=lambda o, s: events.append((o, s))):
        Difference(Box([0.5, 0.5, 0.5]), Sphere(0.5))
    assert events[-1][1] == Site(None, False)


def test_blocks_nest_and_each_receives_every_event():
    outer, inner = [], []
    with capture_constructions(FILENAME, constructed=lambda o, s: outer.append(o)):
        first = Intersection(Box([0.5, 0.5, 0.5]), Sphere(0.5))
        with capture_constructions(FILENAME, constructed=lambda o, s: inner.append(o)):
            second = Xor(Box([0.5, 0.5, 0.5]), Sphere(0.5))
    assert outer == [first, second]
    assert inner == [second]


def test_a_capture_belongs_to_its_own_thread():
    events = []
    with capture_constructions(FILENAME, constructed=lambda o, s: events.append(o)):
        worker = threading.Thread(target=lambda: Union(Box([0.5, 0.5, 0.5]), Sphere(0.5)))
        worker.start()
        worker.join()
    assert events == []


def test_placements_are_reported_with_the_primitive_that_built_them():
    placements = []
    with capture_constructions(FILENAME, placed=lambda p, sdf: placements.append((p, sdf))):
        solid = Solid.box(size=(1.0, 1.0, 1.0))
    ((primitive, placed),) = placements
    assert isinstance(primitive, ConstructionPrimitive)
    assert primitive.kind == "box"
    assert placed is solid


def test_marking_keeps_the_constructor_signature():
    assert "vertices" in inspect.signature(PolygonProfile).parameters
    assert "kind" in inspect.signature(ConstructionPrimitive).parameters
