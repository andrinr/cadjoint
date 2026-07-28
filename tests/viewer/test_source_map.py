"""Tests for mapping construction objects back to their source text."""

from jaxcad.viewer._source_map import (
    PLAYGROUND_FILENAME,
    build_construction_payload,
    capture_profiles,
    locate_profile_call,
)

SIMPLE = """from jaxcad.construction import PolygonProfile, extrude
profile = PolygonProfile([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]], name="tri")
scene = extrude(profile, depth=0.6)
"""

MULTILINE = """from jaxcad.construction import PolygonProfile, SketchPlane, extrude
quad = PolygonProfile(
    [[0, 0], [1, 0], [1, 1], [0, 1]],
    plane=SketchPlane(origin=[0, 1, 0], normal=[1, 0, 0]),
    name="quad",
)
scene = extrude(quad, depth=1.0)
"""


def run(source: str):
    """Execute a program the way the compile worker does, capturing profiles."""
    namespace = {"__builtins__": __builtins__, "__name__": "__jaxcad_playground__"}
    with capture_profiles(PLAYGROUND_FILENAME) as captured:
        exec(compile(source, PLAYGROUND_FILENAME, "exec"), namespace, namespace)
    return captured, namespace


class TestCaptureProfiles:
    def test_records_construction_line(self):
        captured, _ = run(SIMPLE)
        assert len(captured) == 1
        assert captured[0][1] == 2

    def test_captures_profiles_never_bound_to_a_variable(self):
        source = (
            "from jaxcad.construction import PolygonProfile, extrude\n"
            "scene = extrude(PolygonProfile([[0, 0], [1, 0], [0, 1]]), depth=0.5)\n"
        )
        captured, _ = run(source)
        assert len(captured) == 1
        assert captured[0][1] == 2

    def test_restores_the_original_initialiser(self):
        from jaxcad.construction.sketch import PolygonProfile

        original = PolygonProfile.__init__
        with capture_profiles(PLAYGROUND_FILENAME):
            assert PolygonProfile.__init__ is not original
        assert PolygonProfile.__init__ is original


class TestLocateProfileCall:
    def test_finds_vertex_literal_spans(self):
        call = locate_profile_call(SIMPLE, 2)
        assert call is not None
        assert len(call.element_spans) == 3
        start, end = call.element_spans[1]
        assert SIMPLE[start:end] == "[2.0, 0.0]"

    def test_handles_multiline_calls(self):
        call = locate_profile_call(MULTILINE, 2)
        assert call is not None
        assert len(call.element_spans) == 4
        start, end = call.element_spans[3]
        assert MULTILINE[start:end] == "[0, 1]"

    def test_accepts_a_line_inside_a_multiline_call(self):
        # The captured frame line can point at any line of the call.
        assert locate_profile_call(MULTILINE, 3) is not None

    def test_handles_negative_coordinates(self):
        source = 'p = PolygonProfile([[-1.5, -0.25], [1, 0], [0, 1]], name="n")\n'
        call = locate_profile_call(source, 1)
        assert call is not None
        start, end = call.element_spans[0]
        assert source[start:end] == "[-1.5, -0.25]"

    def test_rejects_vertices_from_a_variable(self):
        source = "points = [[0, 0], [1, 0], [0, 1]]\np = PolygonProfile(points)\n"
        assert locate_profile_call(source, 2) is None

    def test_rejects_computed_coordinates(self):
        source = "p = PolygonProfile([[0, 0], [w, 0], [0, 1]])\n"
        assert locate_profile_call(source, 1) is None

    def test_rejects_two_calls_on_one_line(self):
        source = "a = PolygonProfile([[0, 0], [1, 0], [0, 1]]); b = PolygonProfile([[0, 0], [2, 0], [0, 2]])\n"
        assert locate_profile_call(source, 1) is None

    def test_returns_none_for_unparseable_source(self):
        assert locate_profile_call("def broken(:\n", 1) is None

    def test_returns_none_when_no_call_is_on_that_line(self):
        assert locate_profile_call(SIMPLE, 3) is None


class TestConstructionPayload:
    def test_describes_the_sketch(self):
        captured, _ = run(SIMPLE)
        payload = build_construction_payload(captured, SIMPLE)
        assert len(payload) == 1
        profile = payload[0]
        assert profile["id"] == "profile_0"
        assert profile["name"] == "tri"
        assert profile["line"] == 2
        assert profile["editable"] is True
        assert profile["plane"]["normal"] == [0.0, 0.0, 1.0]
        assert [vertex["uv"] for vertex in profile["vertices"]] == [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.5],
        ]

    def test_world_positions_follow_the_sketch_plane(self):
        captured, _ = run(MULTILINE)
        profile = build_construction_payload(captured, MULTILINE)[0]
        # Plane origin is (0, 1, 0), so every vertex sits on that offset plane.
        assert all(abs(vertex["world"][0]) < 1e-6 for vertex in profile["vertices"])

    def test_spans_point_at_the_literals(self):
        captured, _ = run(SIMPLE)
        profile = build_construction_payload(captured, SIMPLE)[0]
        start, end = profile["vertices"][2]["span"]
        assert SIMPLE[start:end] == "[1.0, 1.5]"

    def test_profiles_built_in_a_loop_are_not_editable(self):
        source = (
            "from jaxcad.construction import PolygonProfile, extrude\n"
            "loops = [PolygonProfile([[0, 0], [1, 0], [0, 1]], name=f'l{i}') for i in range(2)]\n"
            "scene = extrude(loops[0], depth=0.4)\n"
        )
        captured, _ = run(source)
        payload = build_construction_payload(captured, source)
        assert len(payload) == 2
        assert all(profile["editable"] is False for profile in payload)
        assert all(vertex["span"] is None for profile in payload for vertex in profile["vertices"])

    def test_variable_vertices_render_but_are_not_editable(self):
        source = (
            "from jaxcad.construction import PolygonProfile, extrude\n"
            "points = [[0, 0], [1, 0], [0, 1]]\n"
            "scene = extrude(PolygonProfile(points, name='v'), depth=0.4)\n"
        )
        captured, _ = run(source)
        payload = build_construction_payload(captured, source)
        assert len(payload) == 1
        assert payload[0]["editable"] is False
        assert len(payload[0]["vertices"]) == 3
