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
