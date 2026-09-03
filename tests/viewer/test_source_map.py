"""Tests for mapping construction objects back to their source text."""

import pytest

from cadjoint.viewer._source_map import (
    PLAYGROUND_FILENAME,
    build_construction_payload,
    build_construction_relations,
    build_material_payload,
    capture_profiles,
    locate_call,
    locate_profile_call,
)

SIMPLE = """from cadjoint.construction import PolygonProfile, extrude
profile = PolygonProfile([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]], name="tri")
scene = extrude(profile, depth=0.6)
"""

MULTILINE = """from cadjoint.construction import PolygonProfile, SketchPlane, extrude
quad = PolygonProfile(
    [[0, 0], [1, 0], [1, 1], [0, 1]],
    plane=SketchPlane(origin=[0, 1, 0], normal=[1, 0, 0]),
    name="quad",
)
scene = extrude(quad, depth=1.0)
"""


def run(source: str):
    """Execute a program the way the compile worker does, capturing profiles."""
    namespace = {"__builtins__": __builtins__, "__name__": "__cadjoint_playground__"}
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
            "from cadjoint.construction import PolygonProfile, extrude\n"
            "scene = extrude(PolygonProfile([[0, 0], [1, 0], [0, 1]]), depth=0.5)\n"
        )
        captured, _ = run(source)
        assert len(captured) == 1
        assert captured[0][1] == 2

    def test_restores_the_original_initialiser(self):
        from cadjoint.construction.sketch import PolygonProfile

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

    def test_follows_vertices_to_a_named_list(self):
        source = "points = [[0, 0], [1, 0], [0, 1]]\np = PolygonProfile(points)\n"
        call = locate_profile_call(source, 2)
        assert call is not None
        start, end = call.element_spans[1]
        assert source[start:end] == "[1, 0]"

    def test_follows_vector2_parameters_to_their_values(self):
        source = (
            "from cadjoint.geometry import Vector2\n"
            "v0 = Vector2(value=[0, 0], free=True, name='v0')\n"
            "v1 = Vector2([1, 0], free=True, name='v1')\n"
            "v2 = Vector2([0, 1], free=True, name='v2')\n"
            "points = [v0, v1, v2]\n"
            "p = PolygonProfile(points)\n"
        )
        call = locate_profile_call(source, 6)
        assert call is not None
        assert [source[start:end] for start, end in call.element_spans] == [
            "[0, 0]",
            "[1, 0]",
            "[0, 1]",
        ]
        assert [source[start:end] for start, end in call.list_element_spans] == [
            "v0",
            "v1",
            "v2",
        ]

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
            "from cadjoint.construction import PolygonProfile, extrude\n"
            "loops = [PolygonProfile([[0, 0], [1, 0], [0, 1]], name=f'l{i}') for i in range(2)]\n"
            "scene = extrude(loops[0], depth=0.4)\n"
        )
        captured, _ = run(source)
        payload = build_construction_payload(captured, source)
        assert len(payload) == 2
        assert all(profile["editable"] is False for profile in payload)
        assert all(vertex["span"] is None for profile in payload for vertex in profile["vertices"])

    def test_variable_vertices_remain_editable_at_their_definition(self):
        source = (
            "from cadjoint.construction import PolygonProfile, extrude\n"
            "points = [[0, 0], [1, 0], [0, 1]]\n"
            "scene = extrude(PolygonProfile(points, name='v'), depth=0.4)\n"
        )
        captured, _ = run(source)
        payload = build_construction_payload(captured, source)
        assert len(payload) == 1
        assert payload[0]["editable"] is True
        assert len(payload[0]["vertices"]) == 3

    def test_default_plane_is_movable_by_adding_an_explicit_plane(self):
        captured, _ = run(SIMPLE)
        profile = build_construction_payload(captured, SIMPLE)[0]
        assert profile["transform"]["call"] == "PolygonProfile"
        assert profile["transform"]["positionArgument"] == "planeOrigin"

    def test_named_plane_parameter_maps_to_its_origin_definition(self):
        source = (
            "from cadjoint.construction import PolygonProfile, SketchPlane, extrude\n"
            "from cadjoint.geometry import Vector\n"
            "origin = Vector(value=[1, 2, 3], free=True, name='origin')\n"
            "plane = SketchPlane(origin=origin)\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]], plane=plane)\n"
            "scene = extrude(profile, depth=0.5)\n"
        )
        captured, _ = run(source)
        profile = build_construction_payload(captured, source)[0]
        assert profile["transform"]["call"] == "SketchPlane"
        start, end = locate_call(source, 4, {"SketchPlane"}).arguments["origin"]
        assert source[start:end] == "[1, 2, 3]"

    def test_reports_constraints_and_extrusion_history(self):
        source = (
            "from cadjoint.construction import PolygonProfile, extrude\n"
            "from cadjoint.constraints import DistanceConstraint, FixedConstraint\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]])\n"
            "FixedConstraint(profile.vertices[0], [0, 0])\n"
            "DistanceConstraint(profile.vertices[0], profile.vertices[1], 1.0)\n"
            "scene = extrude(profile, depth=0.5)\n"
        )
        captured, _ = run(source)
        profile = build_construction_payload(captured, source)[0]
        assert [item["kind"] for item in profile["constraints"]] == ["fixed", "distance"]
        assert profile["operators"] == [{"kind": "extrude", "line": 6}]

    def test_reports_revolve_history_and_material(self):
        source = (
            "from cadjoint.construction import PolygonProfile, revolve\n"
            "from cadjoint.render import Material\n"
            "copper = Material(color=[0.9, 0.4, 0.2], metallic=0.9)\n"
            "section = PolygonProfile([[0.7, -0.2], [1, -0.2], [1, 0.2], [0.7, 0.2]])\n"
            "scene = revolve(section, material=copper)\n"
        )
        captured, _ = run(source)
        profile = build_construction_payload(captured, source)[0]

        assert profile["operators"] == [{"kind": "revolve", "line": 5}]
        assert profile["material"] == "copper"

    def test_reports_named_materials_and_object_assignments(self):
        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.render import Material\n"
            "paint = Material(color=[0.2, 0.4, 0.8], roughness=0.3)\n"
            "scene = Solid.sphere(radius=0.5, material=paint, name='ball')\n"
        )
        captured, namespace = run(source)
        node = build_construction_payload(captured, source)[0]
        library = build_material_payload(namespace, source)

        assert node["material"] == "paint"
        assert len(library) == 1
        assert library[0]["name"] == "paint"
        assert library[0]["color"] == pytest.approx([0.2, 0.4, 0.8])
        start, end = library[0]["spans"]["color"]
        assert source[start:end] == "[0.2, 0.4, 0.8]"

    def test_reports_constraints_between_primitive_positions(self):
        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.constraints import DistanceConstraint, FixedConstraint\n"
            "from cadjoint.geometry import Vector\n"
            "from cadjoint.sdf.boolean import Union\n"
            "left_pos = Vector([-1, 0, 0], free=True, name='left_pos')\n"
            "right_pos = Vector([1, 0, 0], free=True, name='right_pos')\n"
            "left = Solid.sphere(radius=0.5, position=left_pos)\n"
            "right = Solid.sphere(radius=0.5, position=right_pos)\n"
            "FixedConstraint(left_pos, [-1, 0, 0])\n"
            "DistanceConstraint(left_pos, right_pos, 2.0)\n"
            "scene = Union(left, right)\n"
        )
        captured, _ = run(source)

        assert build_construction_relations(captured) == [
            {"kind": "fixed", "nodes": ["sphere_0"], "value": [-1.0, 0.0, 0.0]},
            {
                "kind": "distance",
                "nodes": ["sphere_0", "sphere_1"],
                "value": 2.0,
            },
        ]


BOUND = """from cadjoint.construction import PolygonProfile, Solid, extrude
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.sdf.boolean import Union

tip = Vector2(value=[1.0, 1.5], free=True, name="tip")
root = Vector2(value=[1.0, 0.0], name="root")
profile = PolygonProfile([[0.0, 0.0], root, tip], name="tri")
slab = extrude(profile, depth=Scalar(0.6, free=True, name="depth"))
seat = Vector([1.0, 0.0, 0.0], free=True, name="seat")
peg = Solid.cylinder(radius=Scalar(0.2), height=0.3, position=seat, name="peg")
scene = Union(slab, peg)
"""


class TestParameterBindings:
    """Which draggable values name a free parameter the shader can be told about.

    The binding is half of the join that lets a drag write the uniform buffer
    instead of recompiling; the other half is the shader's slot table, which
    the client checks the name against. What is pinned here is the half this
    package owns: a binding appears when, and only when, the source really
    does put a named free parameter behind the value.
    """

    def test_a_free_named_vertex_binds_to_its_parameter(self):
        captured, _ = run(BOUND)
        profile = build_construction_payload(captured, BOUND)[0]
        assert profile["vertices"][2]["binding"] == {
            "name": "tip",
            "components": 2,
            "index": None,
        }

    def test_a_pinned_vertex_binds_to_nothing(self):
        captured, _ = run(BOUND)
        profile = build_construction_payload(captured, BOUND)[0]
        # ``root`` is a ``Vector2`` the author declared and left pinned, so it
        # is a constant in the generated shader; promising a slot for it would
        # be a lie. A bare literal, by contrast, is wrapped as a free
        # ``{sketch}_v{i}`` and does get one.
        assert profile["vertices"][1]["binding"] is None
        assert profile["vertices"][0]["binding"] == {
            "name": "tri_v0",
            "components": 2,
            "index": None,
        }

    def test_a_primitive_position_binds_to_the_vector_it_was_given(self):
        captured, _ = run(BOUND)
        peg = next(
            entry for entry in build_construction_payload(captured, BOUND) if entry["name"] == "peg"
        )
        assert peg["transform"]["bindings"]["position"] == [
            {"name": "seat", "components": 3, "index": None}
        ]

    def test_an_unwrapped_dimension_binds_and_a_pinned_one_does_not(self):
        captured, _ = run(BOUND)
        peg = next(
            entry for entry in build_construction_payload(captured, BOUND) if entry["name"] == "peg"
        )
        bindings = peg["transform"]["bindings"]
        # A raw number becomes a named free parameter of the primitive; a
        # ``Scalar`` the author pinned is adopted as it stands.
        assert bindings["height"] == [{"name": "peg_height", "components": 1, "index": None}]
        assert "radius" not in bindings

    def test_rotation_binds_component_by_component(self):
        captured, _ = run(BOUND)
        peg = next(
            entry for entry in build_construction_payload(captured, BOUND) if entry["name"] == "peg"
        )
        assert peg["transform"]["bindings"]["rotation"] == [
            {"name": "peg_rx", "components": 1, "index": 0},
            {"name": "peg_ry", "components": 1, "index": 1},
            {"name": "peg_rz", "components": 1, "index": 2},
        ]

    def test_a_partly_free_value_binds_nothing(self):
        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.geometry import Scalar\n"
            "scene = Solid.cylinder(\n"
            "    radius=0.2, height=0.3, rotation=[Scalar(0.1), 0.2, 0.3], name='pin'\n"
            ")\n"
        )
        captured, _ = run(source)
        entry = build_construction_payload(captured, source)[0]
        # One pinned angle out of three: writing the other two live would draw
        # an orientation the source does not have, so the whole argument falls
        # back to the recompile.
        assert "rotation" not in entry["transform"]["bindings"]

    def test_a_sketch_plane_transform_binds_nothing(self):
        captured, _ = run(BOUND)
        profile = build_construction_payload(captured, BOUND)[0]
        assert profile["transform"]["bindings"] == {}


class TestStatementSpans:
    """Where the editor is sent when the viewport selects something.

    The payload used to offer only *argument* spans, so selecting a primitive
    revealed its position literal — three numbers — and selecting a sketch
    revealed nothing at all, because a profile publishes no argument spans.
    Every node now carries the span of the statement that declares it.
    """

    def test_a_sketch_declaration_is_published(self):
        captured, _ = run(SIMPLE)
        profile = build_construction_payload(captured, SIMPLE)[0]
        start, end = profile["statementSpan"]
        assert SIMPLE[start:end].startswith("profile = PolygonProfile(")

    def test_a_primitive_declaration_is_published(self):
        source = (
            "from cadjoint.construction import Solid\n"
            "block = Solid.box(size=[1, 1, 1], position=[0, 0, 0], name='block')\n"
            "scene = block\n"
        )
        captured, _ = run(source)
        entry = build_construction_payload(captured, source)[0]
        start, end = entry["statementSpan"]
        assert (
            source[start:end]
            == "block = Solid.box(size=[1, 1, 1], position=[0, 0, 0], name='block')"
        )

    def test_a_multiline_declaration_spans_all_of_it(self):
        captured, _ = run(MULTILINE)
        profile = build_construction_payload(captured, MULTILINE)[0]
        start, end = profile["statementSpan"]
        text = MULTILINE[start:end]
        assert text.startswith("quad = PolygonProfile(")
        assert text.rstrip().endswith(")")
        assert text.count("\n") >= 3

    def test_a_feature_call_resolves_to_its_own_statement(self):
        from cadjoint.viewer._source_map import statement_span

        # `extrude(...)` is a statement of its own, and the locator has to
        # find it by line the same way it finds a declaration.
        start, end = statement_span(SIMPLE, 3)
        assert SIMPLE[start:end] == "scene = extrude(profile, depth=0.6)"

    def test_the_innermost_statement_wins(self):
        source = (
            "from cadjoint.construction import PolygonProfile\n"
            "def build():\n"
            "    inner = PolygonProfile([[0, 0], [1, 0], [0, 1]], name='inner')\n"
            "    return inner\n"
        )
        from cadjoint.viewer._source_map import statement_span

        start, end = statement_span(source, 3)
        # The assignment, not the `def` that contains it.
        assert source[start:end].startswith("inner = PolygonProfile(")

    def test_an_unplaceable_object_has_no_statement(self):
        from cadjoint.viewer._source_map import statement_span

        assert statement_span(SIMPLE, None) is None
        assert statement_span("def broken(:\n", 1) is None


MESHES = """from cadjoint.fem import Dirichlet, Nodes, SimMesh, ThermalStudy
from cadjoint.sdf.primitives import Box

block = Box(size=[1.0, 1.0, 1.0])
scene = block
grid = SimMesh(name="grid", resolution=12, bounds=[-1.0, -1.0, -1.0], size=[2.0, 2.0, 2.0])
SimMesh(name="anonymous", resolution=8)
heat = ThermalStudy(
    name="bar",
    conductivity=1.0,
    mesh="grid",
    domain=block,
    bcs=[Dirichlet(Nodes.side("-x"), value=0.0)],
)
"""


class TestLocateMeshStatements:
    def test_locates_assigned_and_bare_constructors_in_order(self):
        from cadjoint.viewer._source_map import locate_mesh_statements

        statements = locate_mesh_statements(MESHES)
        assert [statement.index for statement in statements] == [0, 1]
        assert [statement.name for statement in statements] == ["grid", "anonymous"]
        assert [statement.variable for statement in statements] == ["grid", None]
        start, end = statements[0].call_span
        assert MESHES[start:end].startswith("SimMesh(")

    def test_skips_statements_with_more_than_one_constructor(self):
        from cadjoint.viewer._source_map import locate_mesh_statements

        two = "pair = (SimMesh(name='a', resolution=8), SimMesh(name='b', resolution=8))\n"
        assert locate_mesh_statements(two) == []
        # A loop body is not a top-level statement; nothing is located, and
        # the compile payload's count check marks the captured meshes
        # non-editable instead.
        loop = "for i in range(2):\n    SimMesh(name=str(i), resolution=8)\n"
        assert locate_mesh_statements(loop) == []

    def test_returns_none_for_unparsable_source(self):
        from cadjoint.viewer._source_map import locate_mesh_statements

        assert locate_mesh_statements("def broken(:\n") is None

    def test_positional_name_is_extracted(self):
        from cadjoint.viewer._source_map import locate_mesh_statements

        source = "grid = SimMesh('grid', 12)\n"
        statements = locate_mesh_statements(source)
        assert statements[0].name == "grid"


class TestStudyMeshSpans:
    def test_study_statements_carry_mesh_and_domain_value_spans(self):
        from cadjoint.viewer._source_map import locate_study_statements

        statement = locate_study_statements(MESHES)[0]
        start, end = statement.mesh_span
        assert MESHES[start:end] == '"grid"'
        start, end = statement.domain_span
        assert MESHES[start:end] == "block"

    def test_spans_are_none_when_the_keywords_are_absent(self):
        from cadjoint.viewer._source_map import locate_study_statements

        source = (
            "from cadjoint.fem import ThermalStudy\n"
            "scene = None\n"
            "heat = ThermalStudy(name='t', resolution=8, conductivity=1.0, bcs=[])\n"
        )
        statement = locate_study_statements(source)[0]
        assert statement.mesh_span is None
        assert statement.domain_span is None


OPTIMIZATIONS = """from cadjoint.optimize import Optimization
from cadjoint.sdf.primitives import Box

scene = Box(size=[1.0, 1.0, 1.0])

def volume(params):
    return params["size"].prod()

shrink = Optimization(name="min-volume", objective=volume, of=scene, steps=25, learning_rate=0.03)
Optimization("bare", volume, scene)
"""


class TestLocateOptimizationStatements:
    def test_locates_assigned_and_bare_constructors_in_order(self):
        from cadjoint.viewer._source_map import locate_optimization_statements

        statements = locate_optimization_statements(OPTIMIZATIONS)
        assert [statement.index for statement in statements] == [0, 1]
        assert [statement.name for statement in statements] == ["min-volume", "bare"]
        assert [statement.variable for statement in statements] == ["shrink", None]
        start, end = statements[0].call_span
        assert OPTIMIZATIONS[start:end].startswith("Optimization(")

    def test_carries_steps_and_learning_rate_value_spans(self):
        from cadjoint.viewer._source_map import locate_optimization_statements

        first, second = locate_optimization_statements(OPTIMIZATIONS)
        start, end = first.steps_span
        assert OPTIMIZATIONS[start:end] == "25"
        start, end = first.learning_rate_span
        assert OPTIMIZATIONS[start:end] == "0.03"
        # The bare declaration writes neither keyword.
        assert second.steps_span is None
        assert second.learning_rate_span is None

    def test_skips_statements_with_more_than_one_constructor(self):
        from cadjoint.viewer._source_map import locate_optimization_statements

        two = "pair = (Optimization('a', f, s), Optimization('b', f, s))\n"
        assert locate_optimization_statements(two) == []
        loop = "for i in range(2):\n    Optimization(str(i), f, s)\n"
        assert locate_optimization_statements(loop) == []

    def test_returns_none_for_unparsable_source(self):
        from cadjoint.viewer._source_map import locate_optimization_statements

        assert locate_optimization_statements("def broken(:\n") is None
