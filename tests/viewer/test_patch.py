"""Tests for rewriting sketch vertex literals in user source."""

import ast

import pytest

from jaxcad.viewer._patch import (
    PatchError,
    apply_operation,
    delete_vertex,
    insert_vertex,
    set_vertex,
)

SIMPLE = """from jaxcad.construction import PolygonProfile, extrude

# keep this comment
profile = PolygonProfile([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]], name="tri")
scene = extrude(profile, depth=0.6)
"""

MULTILINE = """from jaxcad.construction import PolygonProfile, extrude
quad = PolygonProfile(
    [[0, 0], [1, 0], [1, 1], [0, 1]],
    name="quad",
)
scene = extrude(quad, depth=1.0)
"""


def vertices_of(source: str, line: int) -> list[list[float]]:
    """Read back the literal vertex list, so tests assert on parsed values."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "PolygonProfile":
            if node.lineno <= line <= (node.end_lineno or node.lineno):
                return ast.literal_eval(node.args[0])
    raise AssertionError("no PolygonProfile call found")


class TestSetVertex:
    def test_replaces_one_vertex(self):
        patched = set_vertex(SIMPLE, 4, 2, (1.25, 2.5))
        assert vertices_of(patched, 4) == [[0.0, 0.0], [2.0, 0.0], [1.25, 2.5]]

    def test_leaves_the_rest_of_the_file_untouched(self):
        patched = set_vertex(SIMPLE, 4, 0, (-1.0, -1.0))
        assert "# keep this comment" in patched
        assert patched.splitlines()[4] == SIMPLE.splitlines()[4]
        assert len(patched.splitlines()) == len(SIMPLE.splitlines())

    def test_works_inside_a_multiline_call(self):
        patched = set_vertex(MULTILINE, 2, 1, (3.0, -0.5))
        assert vertices_of(patched, 2) == [[0, 0], [3.0, -0.5], [1, 1], [0, 1]]
        assert patched.count("\n") == MULTILINE.count("\n")

    def test_formats_coordinates_compactly(self):
        patched = set_vertex(SIMPLE, 4, 0, (0.30000000000000004, -0.0))
        assert "[0.3, 0]" in patched

    def test_rejects_an_out_of_range_index(self):
        with pytest.raises(PatchError, match="out of range"):
            set_vertex(SIMPLE, 4, 7, (0.0, 0.0))

    def test_rejects_a_line_without_a_sketch(self):
        with pytest.raises(PatchError, match="No editable PolygonProfile"):
            set_vertex(SIMPLE, 1, 0, (0.0, 0.0))


class TestInsertVertex:
    def test_inserts_before_the_given_index(self):
        patched = insert_vertex(SIMPLE, 4, 1, (0.5, -0.5))
        assert vertices_of(patched, 4) == [[0.0, 0.0], [0.5, -0.5], [2.0, 0.0], [1.0, 1.5]]

    def test_appends_at_the_end(self):
        patched = insert_vertex(SIMPLE, 4, 3, (9.0, 9.0))
        assert vertices_of(patched, 4) == [[0.0, 0.0], [2.0, 0.0], [1.0, 1.5], [9.0, 9.0]]

    def test_inserts_at_the_front(self):
        patched = insert_vertex(SIMPLE, 4, 0, (-2.0, 0.0))
        assert vertices_of(patched, 4)[0] == [-2.0, 0.0]

    def test_stays_on_one_line_for_multiline_sketches(self):
        patched = insert_vertex(MULTILINE, 2, 2, (0.5, 0.5))
        assert len(vertices_of(patched, 2)) == 5
        assert patched.count("\n") == MULTILINE.count("\n")

    def test_rejects_an_index_past_the_end(self):
        with pytest.raises(PatchError, match="out of range"):
            insert_vertex(SIMPLE, 4, 9, (0.0, 0.0))


class TestDeleteVertex:
    def test_removes_a_middle_vertex(self):
        patched = delete_vertex(MULTILINE, 2, 1)
        assert vertices_of(patched, 2) == [[0, 0], [1, 1], [0, 1]]

    def test_removes_the_last_vertex(self):
        patched = delete_vertex(MULTILINE, 2, 3)
        assert vertices_of(patched, 2) == [[0, 0], [1, 0], [1, 1]]

    def test_removes_the_first_vertex(self):
        patched = delete_vertex(MULTILINE, 2, 0)
        assert vertices_of(patched, 2) == [[1, 0], [1, 1], [0, 1]]

    def test_keeps_at_least_a_triangle(self):
        with pytest.raises(PatchError, match="at least 3 vertices"):
            delete_vertex(SIMPLE, 4, 0)


class TestApplyOperation:
    def test_dispatches_by_name(self):
        patched = apply_operation(SIMPLE, "set_vertex", line=4, index=0, xy=(1.0, 1.0))
        assert vertices_of(patched, 4)[0] == [1.0, 1.0]

    def test_rejects_an_unknown_operation(self):
        with pytest.raises(PatchError, match="Unknown patch operation"):
            apply_operation(SIMPLE, "explode", line=4, index=0)

    def test_rejects_missing_arguments(self):
        with pytest.raises(PatchError, match="Invalid arguments"):
            apply_operation(SIMPLE, "set_vertex", line=4, index=0)


class TestRoundTrip:
    def test_patched_source_still_compiles_a_scene(self):
        from jaxcad.viewer._source_map import (
            PLAYGROUND_FILENAME,
            build_construction_payload,
            capture_profiles,
        )

        patched = insert_vertex(SIMPLE, 4, 1, (0.75, -0.25))
        patched = set_vertex(patched, 4, 0, (-0.5, -0.5))
        namespace = {"__builtins__": __builtins__}
        with capture_profiles(PLAYGROUND_FILENAME) as captured:
            exec(compile(patched, PLAYGROUND_FILENAME, "exec"), namespace, namespace)

        payload = build_construction_payload(captured, patched)
        assert payload[0]["editable"] is True
        assert [vertex["uv"] for vertex in payload[0]["vertices"]] == [
            [-0.5, -0.5],
            [0.75, -0.25],
            [2.0, 0.0],
            [1.0, 1.5],
        ]
        # Spans stay usable for the next edit.
        start, end = payload[0]["vertices"][1]["span"]
        assert patched[start:end] == "[0.75, -0.25]"


PRIMITIVES = """from jaxcad.construction import Solid
from jaxcad.sdf.boolean import Union

block = Solid.box(size=[0.5, 0.5, 0.5], position=[1.0, 0.0, 0.0], name="block")
scene = Union(block)
"""


def call_arguments(source: str, line: int, name: str) -> dict:
    """Read back a call's keyword arguments as Python values."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == name:
            if node.lineno <= line <= (node.end_lineno or node.lineno):
                return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    raise AssertionError(f"no {name}() call at line {line}")


class TestSetValue:
    def test_rewrites_an_existing_keyword(self):
        from jaxcad.viewer._patch import set_value

        patched = set_value(PRIMITIVES, 4, "box", "position", [1.5, 0.25, -0.5])
        assert call_arguments(patched, 4, "box")["position"] == [1.5, 0.25, -0.5]

    def test_adds_a_keyword_that_is_not_there_yet(self):
        from jaxcad.viewer._patch import set_value

        # A solid written without `rotation=` must still be rotatable.
        patched = set_value(PRIMITIVES, 4, "box", "rotation", [0, 0.5, 0])
        assert call_arguments(patched, 4, "box")["rotation"] == [0, 0.5, 0]
        # ...and rotating again updates rather than duplicating it.
        again = set_value(patched, 4, "box", "rotation", [0, 1.0, 0])
        assert again.count("rotation=") == 1
        assert call_arguments(again, 4, "box")["rotation"] == [0, 1.0, 0]

    def test_accepts_scalar_arguments(self):
        from jaxcad.viewer._patch import set_value

        source = "from jaxcad.construction import Solid\nball = Solid.sphere(radius=0.5, position=[0, 0, 0])\n"
        patched = set_value(source, 2, "sphere", "radius", 1.25)
        assert call_arguments(patched, 2, "sphere")["radius"] == 1.25

    def test_rejects_an_unknown_call(self):
        from jaxcad.viewer._patch import set_value

        with pytest.raises(PatchError, match="No editable"):
            set_value(PRIMITIVES, 1, "box", "position", [0, 0, 0])


class TestAddPrimitive:
    def test_extends_an_existing_union(self):
        from jaxcad.viewer._patch import add_primitive

        patched = add_primitive(PRIMITIVES, "sphere", [0.0, 1.0, 0.0], {"radius": 0.4})
        assert "sphere1 = Solid.sphere(radius=0.4, position=[0, 1, 0]" in patched
        assert "scene = Union(block, sphere1)" in patched

    def test_wraps_a_scene_that_is_not_a_union(self):
        from jaxcad.viewer._patch import add_primitive

        source = (
            "from jaxcad.construction import Solid\n"
            'scene = Solid.box(size=[1, 1, 1], position=[0, 0, 0], name="b")\n'
        )
        patched = add_primitive(source, "cylinder", [2.0, 0, 0], {"radius": 0.3, "height": 0.8})
        assert "from jaxcad.sdf.boolean import Union" in patched
        assert patched.rstrip().endswith("cylinder1)")

    def test_generates_names_that_do_not_collide(self):
        from jaxcad.viewer._patch import add_primitive

        once = add_primitive(PRIMITIVES, "sphere", [0, 0, 0], {"radius": 0.4})
        twice = add_primitive(once, "sphere", [1, 0, 0], {"radius": 0.4})
        assert "sphere1 =" in twice and "sphere2 =" in twice

    def test_adds_the_Solid_import_when_missing(self):
        from jaxcad.viewer._patch import add_primitive

        source = "from jaxcad.sdf.primitives import Sphere\nscene = Sphere(1.0)\n"
        patched = add_primitive(source, "box", [0, 0, 0], {"size": [0.5, 0.5, 0.5]})
        assert "from jaxcad.construction import Solid" in patched

    def test_requires_a_scene_assignment(self):
        from jaxcad.viewer._patch import add_primitive

        with pytest.raises(PatchError, match="scene = "):
            add_primitive("x = 1\n", "sphere", [0, 0, 0], {"radius": 0.5})
