"""Tests for rewriting sketch vertex literals in user source."""

import ast

import pytest

from cadjoint.viewer._patch import (
    PatchError,
    apply_operation,
    delete_vertex,
    insert_vertex,
    set_vertex,
)

SIMPLE = """from cadjoint.construction import PolygonProfile, extrude

# keep this comment
profile = PolygonProfile([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]], name="tri")
scene = extrude(profile, depth=0.6)
"""

MULTILINE = """from cadjoint.construction import PolygonProfile, extrude
quad = PolygonProfile(
    [[0, 0], [1, 0], [1, 1], [0, 1]],
    name="quad",
)
scene = extrude(quad, depth=1.0)
"""

PARAMETERIZED = """from cadjoint.construction import PolygonProfile
from cadjoint.geometry import Vector2
v0 = Vector2(value=[0, 0], free=True, name="v0")
v1 = Vector2(value=[1, 0], free=True, name="v1")
v2 = Vector2(value=[0, 1], free=True, name="v2")
profile = PolygonProfile([v0, v1, v2])
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

    def test_inserts_into_parameterized_profile_list_not_vector_constructor(self):
        patched = insert_vertex(PARAMETERIZED, 6, 3, (0.4, 0.8))
        assert "PolygonProfile([v0, v1, v2, [0.4, 0.8]])" in patched
        assert "Vector2(value=[0, 1], free=True" in patched
        compile(patched, "<test>", "exec")

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

    def test_removes_parameter_reference_not_parameter_value(self):
        source = PARAMETERIZED.replace(
            "profile = PolygonProfile([v0, v1, v2])",
            "profile = PolygonProfile([v0, v1, v2, [1, 1]])",
        )
        patched = delete_vertex(source, 6, 1)
        assert "PolygonProfile([v0, v2, [1, 1]])" in patched
        assert "v1 = Vector2(value=[1, 0], free=True" in patched
        compile(patched, "<test>", "exec")

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
        from cadjoint.viewer._source_map import (
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


PRIMITIVES = """from cadjoint.construction import Solid
from cadjoint.sdf.boolean import Union

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
        from cadjoint.viewer._patch import set_value

        patched = set_value(PRIMITIVES, 4, "box", "position", [1.5, 0.25, -0.5])
        assert call_arguments(patched, 4, "box")["position"] == [1.5, 0.25, -0.5]

    def test_adds_a_keyword_that_is_not_there_yet(self):
        from cadjoint.viewer._patch import set_value

        # A solid written without `rotation=` must still be rotatable.
        patched = set_value(PRIMITIVES, 4, "box", "rotation", [0, 0.5, 0])
        assert call_arguments(patched, 4, "box")["rotation"] == [0, 0.5, 0]
        # ...and rotating again updates rather than duplicating it.
        again = set_value(patched, 4, "box", "rotation", [0, 1.0, 0])
        assert again.count("rotation=") == 1
        assert call_arguments(again, 4, "box")["rotation"] == [0, 1.0, 0]

    def test_accepts_scalar_arguments(self):
        from cadjoint.viewer._patch import set_value

        source = "from cadjoint.construction import Solid\nball = Solid.sphere(radius=0.5, position=[0, 0, 0])\n"
        patched = set_value(source, 2, "sphere", "radius", 1.25)
        assert call_arguments(patched, 2, "sphere")["radius"] == 1.25

    def test_rejects_an_unknown_call(self):
        from cadjoint.viewer._patch import set_value

        with pytest.raises(PatchError, match="No editable"):
            set_value(PRIMITIVES, 1, "box", "position", [0, 0, 0])

    def test_updates_a_named_vector_parameter_at_its_definition(self):
        from cadjoint.viewer._patch import set_value

        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.geometry import Vector\n"
            "location = Vector(value=[1, 2, 3], free=True, name='location')\n"
            "ball = Solid.sphere(radius=0.5, position=location)\n"
        )
        patched = set_value(source, 4, "sphere", "position", [4, 5, 6])
        assert "Vector(value=[4, 5, 6], free=True" in patched
        assert "position=location" in patched

    def test_updates_a_named_scalar_parameter_at_its_definition(self):
        from cadjoint.viewer._patch import set_value

        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.geometry import Scalar\n"
            "radius = Scalar(0.5, free=True, name='radius')\n"
            "ball = Solid.sphere(radius=radius, position=[0, 0, 0])\n"
        )
        patched = set_value(source, 4, "sphere", "radius", 1.25)
        assert "Scalar(1.25, free=True" in patched
        assert "radius=radius" in patched

    def test_makes_a_default_profile_plane_explicit_when_moved(self):
        from cadjoint.viewer._patch import set_value

        source = (
            "from cadjoint.construction import PolygonProfile\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]])\n"
        )
        patched = set_value(source, 2, "PolygonProfile", "planeOrigin", [2, 3, 4])
        assert "SketchPlane(origin=[2, 3, 4])" in patched
        assert "from cadjoint.construction import" in patched and "SketchPlane" in patched

    def test_makes_a_default_profile_plane_explicit_when_reoriented(self):
        from cadjoint.viewer._patch import set_value

        source = (
            "from cadjoint.construction import PolygonProfile\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]])\n"
        )
        patched = set_value(source, 2, "PolygonProfile", "planeNormal", [0, 1, 0])
        assert "SketchPlane(normal=[0, 1, 0])" in patched
        assert "from cadjoint.construction import" in patched and "SketchPlane" in patched

    def test_adds_a_normal_to_an_existing_sketch_plane(self):
        from cadjoint.viewer._patch import set_value

        source = (
            "from cadjoint.construction import PolygonProfile, SketchPlane\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]], "
            "plane=SketchPlane(origin=[1, 2, 3]), name='s')\n"
        )
        patched = set_value(source, 2, "PolygonProfile", "planeNormal", [0, 0.5, 0.5])
        assert "SketchPlane(origin=[1, 2, 3], normal=[0, 0.5, 0.5])" in patched
        # The profile gains no second `plane=` keyword.
        assert patched.count("plane=") == 1

    def test_rewrites_an_existing_sketch_plane_normal_in_place(self):
        from cadjoint.viewer._patch import set_value

        source = (
            "from cadjoint.construction import PolygonProfile, SketchPlane\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]], "
            "plane=SketchPlane(origin=[1, 2, 3], normal=[0, 0, 1]), name='s')\n"
        )
        patched = set_value(source, 2, "PolygonProfile", "planeNormal", [1, 0, 0])
        assert "normal=[1, 0, 0]" in patched
        assert "normal=[0, 0, 1]" not in patched
        assert "origin=[1, 2, 3]" in patched


class TestAddPrimitive:
    def test_extends_an_existing_union(self):
        from cadjoint.viewer._patch import add_primitive

        patched = add_primitive(PRIMITIVES, "sphere", [0.0, 1.0, 0.0], {"radius": 0.4})
        assert "sphere1 = Solid.sphere(radius=0.4, position=[0, 1, 0]" in patched
        assert "scene = Union(block, sphere1)" in patched

    def test_wraps_a_scene_that_is_not_a_union(self):
        from cadjoint.viewer._patch import add_primitive

        source = (
            "from cadjoint.construction import Solid\n"
            'scene = Solid.box(size=[1, 1, 1], position=[0, 0, 0], name="b")\n'
        )
        patched = add_primitive(source, "cylinder", [2.0, 0, 0], {"radius": 0.3, "height": 0.8})
        assert "from cadjoint.sdf.boolean import Union" in patched
        assert patched.rstrip().endswith("cylinder1)")

    def test_generates_names_that_do_not_collide(self):
        from cadjoint.viewer._patch import add_primitive

        once = add_primitive(PRIMITIVES, "sphere", [0, 0, 0], {"radius": 0.4})
        twice = add_primitive(once, "sphere", [1, 0, 0], {"radius": 0.4})
        assert "sphere1 =" in twice and "sphere2 =" in twice

    def test_adds_the_Solid_import_when_missing(self):
        from cadjoint.viewer._patch import add_primitive

        source = "from cadjoint.sdf.primitives import Sphere\nscene = Sphere(1.0)\n"
        patched = add_primitive(source, "box", [0, 0, 0], {"size": [0.5, 0.5, 0.5]})
        assert "from cadjoint.construction import Solid" in patched

    def test_requires_a_scene_assignment(self):
        from cadjoint.viewer._patch import add_primitive

        with pytest.raises(PatchError, match="scene = "):
            add_primitive("x = 1\n", "sphere", [0, 0, 0], {"radius": 0.5})


class TestMaterials:
    def test_creates_a_named_material_before_the_scene(self):
        from cadjoint.viewer._patch import add_material

        patched = add_material(PRIMITIVES, [0.2, 0.4, 0.8], roughness=0.25)
        assert "from cadjoint.render import Material" in patched
        assert "material1 = Material(" in patched
        assert patched.index("material1 =") < patched.index("scene =")
        assert ast.parse(patched)

    def test_assigns_and_replaces_a_primitive_material(self):
        from cadjoint.viewer._patch import assign_material

        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.render import Material\n"
            "red = Material(color=[1, 0, 0])\n"
            "blue = Material(color=[0, 0, 1])\n"
            "ball = Solid.sphere(radius=0.5, material=red)\n"
            "scene = ball\n"
        )
        patched = assign_material(source, 5, "blue")
        assert "Solid.sphere(radius=0.5, material=blue)" in patched

    def test_assigns_material_to_a_profiles_extrusion(self):
        from cadjoint.viewer._patch import assign_material

        source = (
            "from cadjoint.construction import PolygonProfile, extrude\n"
            "from cadjoint.render import Material\n"
            "paint = Material(color=[0.2, 0.4, 0.8])\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]])\n"
            "body = extrude(profile, depth=0.5)\n"
            "scene = body\n"
        )
        patched = assign_material(source, 4, "paint")
        assert "extrude(profile, depth=0.5, material=paint)" in patched

    def test_rejects_a_name_that_is_not_a_material_definition(self):
        from cadjoint.viewer._patch import assign_material

        with pytest.raises(PatchError, match="not a named Material"):
            assign_material(PRIMITIVES, 4, "block")


class TestSketchHistoryOperations:
    def test_adds_a_standalone_sketch_before_the_scene(self):
        from cadjoint.viewer._patch import add_sketch

        patched = add_sketch(PRIMITIVES, [2, 3, 0])
        assert "sketch1 = PolygonProfile(" in patched
        assert "SketchPlane(origin=[2, 3, 0])" in patched
        assert patched.index("sketch1 =") < patched.index("scene =")
        assert ast.parse(patched)

    def test_extrudes_a_named_sketch_into_the_scene(self):
        from cadjoint.viewer._patch import add_extrusion

        source = (
            "from cadjoint.construction import PolygonProfile\n"
            "from cadjoint.sdf.primitives import Sphere\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]])\n"
            "scene = Sphere(0.2)\n"
        )
        patched = add_extrusion(source, 3, 0.75)
        assert "profile_body = extrude(profile, depth=0.75)" in patched
        assert "scene = Union(Sphere(0.2), profile_body)" in patched
        assert ast.parse(patched)

    def test_adds_constraints_and_a_projection_step(self):
        from cadjoint.viewer._patch import add_constraint, solve_sketch

        source = (
            "from cadjoint.construction import PolygonProfile, extrude\n"
            "profile = PolygonProfile([[0, 0], [2, 0], [0, 1]])\n"
            "scene = extrude(profile, depth=0.5)\n"
        )
        patched = add_constraint(source, 2, "fixed", [0], [0, 0])
        # Imports land beside the constraint statements, so the profile's
        # line number is stable across repeated patches.
        patched = add_constraint(patched, 2, "distance", [0, 1], 1.0)
        patched = solve_sketch(patched, 2)
        assert "FixedConstraint(profile.vertices[0], [0, 0])" in patched
        assert "DistanceConstraint(profile.vertices[0], profile.vertices[1], 1)" in patched
        assert "satisfy_constraints(profile, method='newton', steps=8)" in patched
        assert patched.index("DistanceConstraint") < patched.index("satisfy_constraints(profile")
        assert ast.parse(patched)

    def test_solve_step_is_idempotent(self):
        from cadjoint.viewer._patch import solve_sketch

        source = (
            "from cadjoint.construction import PolygonProfile, extrude\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]])\n"
            "scene = extrude(profile, depth=0.5)\n"
        )
        once = solve_sketch(source, 2)
        twice = solve_sketch(once, 3, method="adam", iterations=24)
        assert twice.count("satisfy_constraints(profile") == 1
        assert "method='adam'" in twice
        assert "steps=24" in twice

    def test_rejects_invalid_solver_settings(self):
        from cadjoint.viewer._patch import solve_sketch

        source = "from cadjoint.construction import PolygonProfile\nprofile = PolygonProfile([[0, 0], [1, 0], [0, 1]])\n"
        with pytest.raises(PatchError, match="method"):
            solve_sketch(source, 2, method="bfgs")
        with pytest.raises(PatchError, match="iterations"):
            solve_sketch(source, 2, iterations=0)


class TestDeleteObject:
    def test_removes_a_solid_and_its_use_in_the_scene(self):
        from cadjoint.viewer._patch import delete_object

        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.sdf.boolean import Union\n"
            'ball = Solid.sphere(radius=0.5, position=[0, 0, 0], name="ball")\n'
            'block = Solid.box(size=[1, 1, 1], position=[2, 0, 0], name="block")\n'
            "scene = Union(ball, block)\n"
        )
        patched = delete_object(source, 3)

        assert "ball" not in patched
        assert "Solid.box" in patched
        assert ast.parse(patched)  # still valid Python
        scene = [line for line in patched.splitlines() if line.startswith("scene")][0]
        assert scene == "scene = Union(block)"

    def test_refuses_when_the_value_is_used_elsewhere(self):
        from cadjoint.viewer._patch import delete_object

        source = (
            "from cadjoint.construction import PolygonProfile, extrude\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]], name='p')\n"
            "scene = extrude(profile, depth=0.5)\n"
        )
        with pytest.raises(PatchError, match="used elsewhere"):
            delete_object(source, 2)

    def test_removes_constraints_owned_by_a_deleted_objects_position(self):
        from cadjoint.viewer._patch import delete_object

        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.constraints import DistanceConstraint\n"
            "from cadjoint.geometry import Vector\n"
            "from cadjoint.sdf.boolean import Union\n"
            "left_pos = Vector([-1, 0, 0], free=True, name='left_pos')\n"
            "right_pos = Vector([1, 0, 0], free=True, name='right_pos')\n"
            "left = Solid.sphere(radius=0.5, position=left_pos)\n"
            "right = Solid.sphere(radius=0.5, position=right_pos)\n"
            "DistanceConstraint(left_pos, right_pos, 2.0)\n"
            "scene = Union(left, right)\n"
        )
        patched = delete_object(source, 7)
        assert "left = Solid.sphere" not in patched
        assert "DistanceConstraint(left_pos" not in patched
        assert "scene = Union(right)" in patched

    def test_removes_an_inline_solid_from_the_scene(self):
        from cadjoint.viewer._patch import delete_object

        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.sdf.boolean import Union\n"
            "scene = Union(\n"
            "    Solid.sphere(radius=1.0),\n"
            "    Solid.box(size=[1, 1, 1]),\n"
            ")\n"
        )
        patched = delete_object(source, 4)
        assert patched.count("Solid.") == 1
        assert "Solid.box" in patched

    def test_refuses_two_objects_built_on_one_line(self):
        from cadjoint.viewer._patch import delete_object

        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.sdf.boolean import Union\n"
            "scene = Union(Solid.sphere(radius=1.0), Solid.box(size=[1, 1, 1]))\n"
        )
        with pytest.raises(PatchError, match="No single construction call"):
            delete_object(source, 3)


class TestConstraintEditing:
    """delete_constraint / set_constraint_value / relational add_constraint."""

    def _profile_line(self, source: str) -> int:
        return next(
            number + 1
            for number, text in enumerate(source.splitlines())
            if "PolygonProfile(" in text
        )

    def test_relational_kinds_emit_their_classes(self):
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        line = self._profile_line(EXAMPLE_SOURCE)
        cases = {
            "horizontal": ("HorizontalConstraint", [0, 1]),
            "vertical": ("VerticalConstraint", [0, 1]),
            "coincident": ("CoincidentConstraint", [0, 1]),
            "parallel": ("ParallelEdgesConstraint", [0, 1, 2, 3]),
            "perpendicular": ("PerpendicularEdgesConstraint", [0, 1, 2, 3]),
        }
        for kind, (symbol, indices) in cases.items():
            patched = apply_operation(
                EXAMPLE_SOURCE, "add_constraint", line=line, kind=kind, indices=indices
            )
            assert f"{symbol}(comb_profile.vertices[0]" in patched
            assert "from cadjoint.constraints import" in patched or symbol in patched

    def test_delete_constraint_removes_bare_name_statement(self):
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        line = self._profile_line(EXAMPLE_SOURCE)
        patched = apply_operation(EXAMPLE_SOURCE, "delete_constraint", line=line, index=0)
        assert "FixedConstraint(base_l," not in patched
        # Only that statement changed: removing its line reproduces the rest.
        removed = [
            text for text in EXAMPLE_SOURCE.splitlines() if "FixedConstraint(base_l," in text
        ]
        assert len(removed) == 1
        assert patched == EXAMPLE_SOURCE.replace(removed[0] + "\n", "")

    def test_delete_constraint_removes_subscript_statement(self):
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        line = self._profile_line(EXAMPLE_SOURCE)
        grown = apply_operation(
            EXAMPLE_SOURCE, "add_constraint", line=line, kind="horizontal", indices=[0, 1]
        )
        # Creation order appends: the new subscript statement sits after all
        # of the starter's bare-name constraint statements — the last index.
        from cadjoint.viewer._source_map import locate_constraint_statements

        index = len(locate_constraint_statements(grown, self._profile_line(grown))) - 1
        shrunk = apply_operation(
            grown, "delete_constraint", line=self._profile_line(grown), index=index
        )
        assert "HorizontalConstraint(comb_profile.vertices" not in shrunk

    def test_delete_constraint_rejects_out_of_range(self):
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        line = self._profile_line(EXAMPLE_SOURCE)
        with pytest.raises(PatchError, match="out of range"):
            apply_operation(EXAMPLE_SOURCE, "delete_constraint", line=line, index=99)

    def test_set_constraint_value_follows_scalar_indirection(self):
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        line = self._profile_line(EXAMPLE_SOURCE)
        patched = apply_operation(
            EXAMPLE_SOURCE, "set_constraint_value", line=line, index=1, value=2.5
        )
        assert "Scalar(2.5" in patched

    def test_set_constraint_value_rejects_relational(self):
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        line = self._profile_line(EXAMPLE_SOURCE)
        grown = apply_operation(
            EXAMPLE_SOURCE, "add_constraint", line=line, kind="coincident", indices=[0, 1]
        )
        with pytest.raises(PatchError, match="editable value"):
            apply_operation(
                grown, "set_constraint_value", line=self._profile_line(grown), index=2, value=1.0
            )


class TestAddRevolution:
    def test_revolves_a_fresh_sketch(self):
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        grown = apply_operation(EXAMPLE_SOURCE, "add_sketch", origin=[0.5, 0.5, 0.0])
        sketch_line = next(
            number + 1
            for number, text in enumerate(grown.splitlines())
            if text.startswith("sketch1 =")
        )
        patched = apply_operation(grown, "add_revolution", line=sketch_line, offset=0.3)
        assert "sketch1_body = revolve(sketch1, offset=0.3)" in patched
        assert ", sketch1_body" in patched

    def test_refuses_a_second_operator(self):
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        line = next(
            number + 1
            for number, text in enumerate(EXAMPLE_SOURCE.splitlines())
            if "PolygonProfile(" in text
        )
        with pytest.raises(PatchError, match="already has"):
            apply_operation(EXAMPLE_SOURCE, "add_revolution", line=line, offset=0.0)


def _sketch_line(source: str, variable: str) -> int:
    return next(
        number + 1
        for number, text in enumerate(source.splitlines())
        if text.startswith(f"{variable} =")
    )


class TestAddLoft:
    def _two_sketches(self) -> str:
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        grown = apply_operation(EXAMPLE_SOURCE, "add_sketch", origin=[0.5, 0.5, 0.0])
        return apply_operation(grown, "add_sketch", origin=[0.5, 0.5, 1.0])

    def test_lofts_two_fresh_sketches(self):
        from cadjoint.viewer.playground import compile_source

        source = self._two_sketches()
        patched = apply_operation(
            source,
            "add_loft",
            line_a=_sketch_line(source, "sketch1"),
            line_b=_sketch_line(source, "sketch2"),
            height=1.5,
        )
        assert "sketch1_body = loft(sketch1, sketch2, height=1.5)" in patched
        assert ", sketch1_body" in patched
        assert "from cadjoint.construction import" in patched and "loft" in patched
        assert patched.index("sketch1_body =") < patched.index("scene =")
        result = compile_source(patched)
        assert result["ok"], result.get("error")

    def test_refuses_unequal_vertex_counts(self):
        source = self._two_sketches()
        source = apply_operation(
            source,
            "insert_vertex",
            line=_sketch_line(source, "sketch2"),
            index=1,
            xy=[0.0, -0.8],
        )
        with pytest.raises(PatchError, match="equal vertex counts"):
            apply_operation(
                source,
                "add_loft",
                line_a=_sketch_line(source, "sketch1"),
                line_b=_sketch_line(source, "sketch2"),
            )

    def test_refuses_the_same_sketch_twice(self):
        source = self._two_sketches()
        line = _sketch_line(source, "sketch1")
        with pytest.raises(PatchError, match="two different sketches"):
            apply_operation(source, "add_loft", line_a=line, line_b=line)

    def test_refuses_a_sketch_with_an_operator(self):
        from cadjoint.viewer.playground import EXAMPLE_SOURCE

        grown = apply_operation(EXAMPLE_SOURCE, "add_sketch", origin=[0.5, 0.5, 0.0])
        extruded_line = next(
            number + 1
            for number, text in enumerate(grown.splitlines())
            if "PolygonProfile(" in text and not text.startswith("sketch1 =")
        )
        with pytest.raises(PatchError, match="already has an operator"):
            apply_operation(
                grown,
                "add_loft",
                line_a=extruded_line,
                line_b=_sketch_line(grown, "sketch1"),
            )

    def test_lofted_sketches_cannot_be_lofted_again(self):
        source = self._two_sketches()
        patched = apply_operation(
            source,
            "add_loft",
            line_a=_sketch_line(source, "sketch1"),
            line_b=_sketch_line(source, "sketch2"),
        )
        with pytest.raises(PatchError, match="already has an operator"):
            apply_operation(
                patched,
                "add_loft",
                line_a=_sketch_line(patched, "sketch1"),
                line_b=_sketch_line(patched, "sketch2"),
            )


STUDIES = """from cadjoint.construction import Solid
from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
from cadjoint.sdf.boolean import Union

block = Solid.box(size=[0.5, 0.5, 0.5], position=[1.0, 0.0, 0.0], name="block")
scene = Union(block)
heat = ThermalStudy(
    name="bar-conduction",
    resolution=12,
    conductivity=2.0,
    bcs=[Dirichlet(Nodes.side("-x"), value=1.0), Dirichlet(Nodes.side("+x"), value=0.0)],
)
"""

BOX_SELECTION = {"kind": "box", "min_corner": [0.0, 0.0, 0.0], "max_corner": [1.0, 1.0, 1.0]}


class TestAddStudy:
    def test_appends_a_thermal_study_after_the_scene(self):
        from cadjoint.viewer._patch import add_study

        patched = add_study(PRIMITIVES, "thermal")
        assert (
            "study1 = ThermalStudy(name='study1', resolution=20, conductivity=1.0, bcs=[])"
            in patched
        )
        assert "from cadjoint.fem import ThermalStudy" in patched
        assert patched.index("scene =") < patched.index("study1 =")
        assert ast.parse(patched)

    def test_appends_an_elastic_study_after_the_last_study(self):
        from cadjoint.viewer._patch import add_study

        patched = add_study(STUDIES, "elastic", name="cantilever")
        assert (
            "study1 = ElasticStudy(name='cantilever', resolution=20, "
            "youngs=200.0, poisson=0.3, bcs=[])" in patched
        )
        assert patched.index("heat = ") < patched.index("study1 =")
        # The existing cadjoint.fem import is extended in place, so every
        # original line keeps its number and only the new study line is added.
        assert "from cadjoint.fem import Dirichlet, Nodes, ThermalStudy, ElasticStudy" in patched
        assert patched.count("from cadjoint.fem import") == 1
        assert len(patched.splitlines()) == len(STUDIES.splitlines()) + 1

    def test_keeps_lines_above_the_insertion_untouched(self):
        from cadjoint.viewer._patch import add_study

        original = PRIMITIVES.splitlines()
        patched = add_study(PRIMITIVES, "thermal").splitlines()
        # A first-time import lands beside the study statement, below
        # everything else, so line-addressed operations stay valid.
        assert patched[: len(original)] == original

    def test_generates_names_that_do_not_collide(self):
        once = apply_operation(PRIMITIVES, "add_study", kind="thermal", name=None)
        twice = apply_operation(once, "add_study", kind="elastic", name=None)
        assert "study1 = ThermalStudy(" in twice
        assert "study2 = ElasticStudy(" in twice

    def test_rejects_a_duplicate_name(self):
        with pytest.raises(PatchError, match="already exists"):
            apply_operation(STUDIES, "add_study", kind="thermal", name="bar-conduction")

    def test_rejects_an_unknown_kind(self):
        with pytest.raises(PatchError, match="thermal"):
            apply_operation(PRIMITIVES, "add_study", kind="modal", name=None)

    def test_requires_a_scene_assignment(self):
        with pytest.raises(PatchError, match="scene = "):
            apply_operation("x = 1\n", "add_study", kind="thermal", name=None)

    def test_added_study_round_trips_through_exec(self):
        from cadjoint.fem import capture_studies

        patched = apply_operation(PRIMITIVES, "add_study", kind="thermal", name=None)
        patched = apply_operation(
            patched,
            "add_study_bc",
            study="study1",
            bc_type="dirichlet",
            selection=BOX_SELECTION,
            value=300.0,
        )
        with capture_studies() as studies:
            exec(compile(patched, "<test>", "exec"), {})
        assert [study.name for study in studies] == ["study1"]
        assert studies[0].describe()["bcs"][0]["value"] == 300.0


class TestDeleteStudy:
    def test_removes_a_named_study_statement(self):
        patched = apply_operation(STUDIES, "delete_study", study="bar-conduction")
        assert "ThermalStudy" not in patched.replace(
            "from cadjoint.fem import Dirichlet, Nodes, ThermalStudy", ""
        )
        assert "scene = Union(block)" in patched
        assert ast.parse(patched)

    def test_resolves_a_study_by_index(self):
        patched = apply_operation(STUDIES, "delete_study", study=0)
        assert "heat = " not in patched

    def test_refuses_when_the_study_is_used_elsewhere(self):
        source = STUDIES + "result = heat.solve(scene)\n"
        with pytest.raises(PatchError, match="used elsewhere"):
            apply_operation(source, "delete_study", study="heat")

    def test_rejects_an_unknown_reference(self):
        with pytest.raises(PatchError, match="No single study"):
            apply_operation(STUDIES, "delete_study", study="nope")
        with pytest.raises(PatchError, match="out of range"):
            apply_operation(STUDIES, "delete_study", study=4)


class TestStudyBc:
    def test_appends_the_literal_bc_source(self):
        patched = apply_operation(
            STUDIES,
            "add_study_bc",
            study="bar-conduction",
            bc_type="dirichlet",
            selection=BOX_SELECTION,
            value=300.0,
        )
        assert "Dirichlet(Nodes.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), value=300.0)" in patched
        # Appended after the existing conditions, inside the same list.
        assert patched.index('Nodes.side("+x")') < patched.index("Nodes.box(")
        assert ast.parse(patched)

    def test_fills_an_empty_bcs_list(self):
        grown = apply_operation(PRIMITIVES, "add_study", kind="elastic", name=None)
        patched = apply_operation(
            grown,
            "add_study_bc",
            study="study1",
            bc_type="fixed",
            selection={"kind": "side", "side": "-x"},
        )
        assert "bcs=[Fixed(Nodes.side('-x'))]" in patched

    def test_adds_a_missing_bcs_keyword(self):
        source = (
            "from cadjoint.fem import ThermalStudy\n"
            "scene = None\n"
            "study = ThermalStudy(name='t', resolution=8, conductivity=1.0)\n"
        )
        patched = apply_operation(
            source,
            "add_study_bc",
            study="t",
            bc_type="heat_flux",
            selection={"kind": "sphere", "center": [0.0, 0.0, 1.0], "radius": 0.5},
            value=2.5,
        )
        assert "bcs=[HeatFlux(Nodes.sphere([0.0, 0.0, 1.0], 0.5), flux=2.5)]" in patched
        assert ast.parse(patched)

    def test_creates_imports_beside_the_study_and_extends_them_in_place(self):
        source = (
            "from cadjoint.fem import ThermalStudy\n"
            "scene = None\n"
            "study = ThermalStudy(name='t', resolution=8, conductivity=1.0, bcs=[])\n"
        )
        once = apply_operation(
            source,
            "add_study_bc",
            study="t",
            bc_type="dirichlet",
            selection=BOX_SELECTION,
            value=1.0,
        )
        assert "from cadjoint.fem import ThermalStudy, Dirichlet, Nodes" in once
        # A second condition reuses the import line: no line drift.
        twice = apply_operation(
            once,
            "add_study_bc",
            study="t",
            bc_type="dirichlet",
            selection={"kind": "side", "side": "+x"},
            value=0.0,
        )
        assert twice.count("from cadjoint.fem import") == 1
        assert len(twice.splitlines()) == len(once.splitlines())

    def test_renders_composed_selections(self):
        patched = apply_operation(
            STUDIES,
            "add_study_bc",
            study="bar-conduction",
            bc_type="dirichlet",
            selection={
                "kind": "and",
                "operands": [
                    {"kind": "side", "side": "+x", "tol": 0.25},
                    {
                        "kind": "not",
                        "operand": {
                            "kind": "halfspace",
                            "point": [0.0, 0.0, 0.0],
                            "normal": [0.0, 0.0, -1.0],
                        },
                    },
                ],
            },
            value=5.0,
        )
        assert (
            "(Nodes.side('+x', tol=0.25) & "
            "~Nodes.halfspace([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]))" in patched
        )
        assert ast.parse(patched)

    def test_rejects_a_predicate_selection_description(self):
        with pytest.raises(PatchError, match="serializable"):
            apply_operation(
                STUDIES,
                "add_study_bc",
                study="bar-conduction",
                bc_type="dirichlet",
                selection={"kind": "predicate", "name": "hot_end"},
                value=1.0,
            )

    def test_rejects_a_kind_incompatible_bc_type(self):
        with pytest.raises(PatchError, match="thermal study accepts"):
            apply_operation(
                STUDIES,
                "add_study_bc",
                study="bar-conduction",
                bc_type="traction",
                selection=BOX_SELECTION,
                value=[0.0, 0.0, -1.0],
            )

    def test_rejects_bad_values(self):
        with pytest.raises(PatchError, match="numeric `value`"):
            apply_operation(
                STUDIES,
                "add_study_bc",
                study="bar-conduction",
                bc_type="dirichlet",
                selection=BOX_SELECTION,
                value=[1.0, 2.0],
            )
        elastic = apply_operation(PRIMITIVES, "add_study", kind="elastic", name=None)
        with pytest.raises(PatchError, match="three numbers"):
            apply_operation(
                elastic,
                "add_study_bc",
                study="study1",
                bc_type="traction",
                selection=BOX_SELECTION,
                value=1.0,
            )
        with pytest.raises(PatchError, match="takes no value"):
            apply_operation(
                elastic,
                "add_study_bc",
                study="study1",
                bc_type="fixed",
                selection=BOX_SELECTION,
                value=1.0,
            )

    def test_delete_removes_one_condition_and_its_separator(self):
        patched = apply_operation(STUDIES, "delete_study_bc", study="bar-conduction", bc=0)
        assert 'Nodes.side("-x")' not in patched
        assert 'Dirichlet(Nodes.side("+x"), value=0.0)' in patched
        assert ast.parse(patched)

    def test_delete_of_the_last_condition_leaves_an_empty_list(self):
        patched = apply_operation(STUDIES, "delete_study_bc", study="bar-conduction", bc=1)
        patched = apply_operation(patched, "delete_study_bc", study="bar-conduction", bc=0)
        assert "Dirichlet" not in patched.replace(
            "from cadjoint.fem import Dirichlet, Nodes, ThermalStudy", ""
        )
        assert "bcs=[" in patched
        assert ast.parse(patched)

    def test_delete_rejects_out_of_range(self):
        with pytest.raises(PatchError, match="out of range"):
            apply_operation(STUDIES, "delete_study_bc", study="bar-conduction", bc=5)

    def test_predicate_conditions_reject_edits(self):
        source = (
            "from cadjoint.fem import Dirichlet, Nodes, ThermalStudy\n"
            "def hot_end(points):\n"
            "    return points[:, 0] > 0.9\n"
            "scene = None\n"
            "study = ThermalStudy(name='t', resolution=8, conductivity=1.0,\n"
            "                     bcs=[Dirichlet(Nodes.predicate(hot_end), value=1.0)])\n"
        )
        with pytest.raises(PatchError, match="predicate"):
            apply_operation(source, "delete_study_bc", study="t", bc=0)
        with pytest.raises(PatchError, match="predicate"):
            apply_operation(source, "set_study_value", study="t", bc=0, value=2.0)


class TestSetStudyValue:
    def test_rewrites_a_bc_value_with_exact_repr(self):
        patched = apply_operation(
            STUDIES, "set_study_value", study="bar-conduction", bc=0, value=273.15
        )
        assert 'Dirichlet(Nodes.side("-x"), value=273.15)' in patched

    def test_rewrites_a_positional_bc_value(self):
        source = (
            "from cadjoint.fem import Dirichlet, Nodes, ThermalStudy\n"
            "scene = None\n"
            "study = ThermalStudy(name='t', resolution=8, conductivity=1.0,\n"
            "                     bcs=[Dirichlet(Nodes.side('-x'), 1.0)])\n"
        )
        patched = apply_operation(source, "set_study_value", study="t", bc=0, value=250.0)
        assert "Dirichlet(Nodes.side('-x'), 250.0)" in patched

    def test_rewrites_a_traction_vector(self):
        elastic = apply_operation(PRIMITIVES, "add_study", kind="elastic", name=None)
        elastic = apply_operation(
            elastic,
            "add_study_bc",
            study="study1",
            bc_type="traction",
            selection={"kind": "side", "side": "+x"},
            value=[0.0, 0.0, -1.0],
        )
        patched = apply_operation(
            elastic, "set_study_value", study="study1", bc=0, value=[0.0, 0.5, -2.0]
        )
        assert "vector=[0.0, 0.5, -2.0]" in patched

    def test_rejects_editing_a_fixed_bc(self):
        elastic = apply_operation(PRIMITIVES, "add_study", kind="elastic", name=None)
        elastic = apply_operation(
            elastic,
            "add_study_bc",
            study="study1",
            bc_type="fixed",
            selection={"kind": "side", "side": "-x"},
        )
        with pytest.raises(PatchError, match="no value to edit"):
            apply_operation(elastic, "set_study_value", study="study1", bc=0, value=1.0)

    def test_rewrites_a_study_keyword_in_place(self):
        patched = apply_operation(
            STUDIES, "set_study_value", study="bar-conduction", argument="conductivity", value=3.5
        )
        assert "conductivity=3.5" in patched
        assert "conductivity=2.0" not in patched

    def test_resolution_stays_integral(self):
        patched = apply_operation(
            STUDIES, "set_study_value", study="bar-conduction", argument="resolution", value=24
        )
        assert "resolution=24" in patched
        patched = apply_operation(
            STUDIES,
            "set_study_value",
            study="bar-conduction",
            argument="resolution",
            value=[10.0, 4.0, 4.0],
        )
        assert "resolution=[10, 4, 4]" in patched
        with pytest.raises(PatchError, match="whole numbers"):
            apply_operation(
                STUDIES,
                "set_study_value",
                study="bar-conduction",
                argument="resolution",
                value=10.5,
            )

    def test_adds_a_missing_keyword(self):
        patched = apply_operation(
            STUDIES, "set_study_value", study="bar-conduction", argument="source", value=0.5
        )
        assert "source=0.5" in patched
        assert ast.parse(patched)

    def test_follows_named_scalar_indirection(self):
        source = (
            "from cadjoint.fem import ThermalStudy\n"
            "scene = None\n"
            "k = 2.0\n"
            "study = ThermalStudy(name='t', resolution=8, conductivity=k, bcs=[])\n"
        )
        patched = apply_operation(
            source, "set_study_value", study="t", argument="conductivity", value=4.0
        )
        assert "k = 4.0" in patched
        assert "conductivity=k" in patched

    def test_rejects_arguments_of_the_other_kind(self):
        with pytest.raises(PatchError, match="thermal study's editable arguments"):
            apply_operation(
                STUDIES, "set_study_value", study="bar-conduction", argument="youngs", value=1.0
            )

    def test_rejects_both_or_neither_target(self):
        with pytest.raises(PatchError, match="exactly one"):
            apply_operation(
                STUDIES,
                "set_study_value",
                study="bar-conduction",
                bc=0,
                argument="conductivity",
                value=1.0,
            )
        with pytest.raises(PatchError, match="exactly one"):
            apply_operation(STUDIES, "set_study_value", study="bar-conduction", value=1.0)


# ── Simulation mesh operations ───────────────────────────────────────────────

MESHES = """from cadjoint.construction import Solid
from cadjoint.fem import Dirichlet, Nodes, SimMesh, ThermalStudy
from cadjoint.sdf.boolean import Union

block = Solid.box(size=[0.5, 0.5, 0.5], position=[1.0, 0.0, 0.0], name="block")
scene = Union(block)
grid = SimMesh(name="block-grid", resolution=12, bounds=[0.0, -0.5, -0.5], size=[2.0, 1.0, 1.0])
heat = ThermalStudy(
    name="bar-conduction",
    conductivity=2.0,
    mesh="block-grid",
    bcs=[Dirichlet(Nodes.side("-x"), value=1.0), Dirichlet(Nodes.side("+x"), value=0.0)],
)
"""


class TestAddMesh:
    def test_appends_a_mesh_after_the_scene(self):
        patched = apply_operation(PRIMITIVES, "add_mesh", name=None)
        assert "mesh1 = SimMesh(name='mesh1', resolution=20)" in patched
        assert "from cadjoint.fem import SimMesh" in patched
        assert patched.index("scene =") < patched.index("mesh1 =")
        assert ast.parse(patched)

    def test_lands_before_the_first_study(self):
        # A study resolves `mesh="name"` at construction time, so a mesh
        # declared after it would be invisible to it.
        patched = apply_operation(STUDIES, "add_mesh", name="grid")
        assert patched.index("mesh1 = SimMesh(") < patched.index("heat = ThermalStudy(")

    def test_appends_after_the_last_existing_mesh(self):
        patched = apply_operation(MESHES, "add_mesh", name="fine")
        assert patched.index("grid = SimMesh(") < patched.index("mesh1 = SimMesh(")
        assert patched.index("mesh1 = SimMesh(") < patched.index("heat = ThermalStudy(")

    def test_generates_names_that_do_not_collide(self):
        once = apply_operation(PRIMITIVES, "add_mesh", name=None)
        twice = apply_operation(once, "add_mesh", name=None)
        assert "mesh1 = SimMesh(" in twice
        assert "mesh2 = SimMesh(" in twice

    def test_rejects_a_duplicate_name(self):
        with pytest.raises(PatchError, match="already exists"):
            apply_operation(MESHES, "add_mesh", name="block-grid")

    def test_requires_a_scene_assignment(self):
        with pytest.raises(PatchError, match="scene = "):
            apply_operation("x = 1\n", "add_mesh", name=None)

    def test_added_mesh_round_trips_through_exec(self):
        from cadjoint.fem import capture_sim_meshes

        patched = apply_operation(PRIMITIVES, "add_mesh", name="grid")
        with capture_sim_meshes() as meshes:
            exec(compile(patched, "<test>", "exec"), {})
        assert [mesh.name for mesh in meshes] == ["grid"]
        assert meshes[0].resolution == 20


class TestDeleteMesh:
    def test_removes_an_unreferenced_mesh(self):
        source = MESHES.replace('mesh="block-grid",\n', "resolution=8,\n")
        patched = apply_operation(source, "delete_mesh", mesh="block-grid")
        assert "SimMesh(" not in patched.split("import")[-1]
        assert ast.parse(patched)

    def test_resolves_a_mesh_by_index_and_variable(self):
        source = MESHES.replace('mesh="block-grid",\n', "resolution=8,\n")
        assert "grid = SimMesh" not in apply_operation(source, "delete_mesh", mesh=0)
        assert "grid = SimMesh" not in apply_operation(source, "delete_mesh", mesh="grid")

    def test_refuses_while_a_study_references_the_name(self):
        with pytest.raises(PatchError, match="referenced by a study"):
            apply_operation(MESHES, "delete_mesh", mesh="block-grid")

    def test_refuses_while_the_variable_is_used(self):
        source = MESHES.replace('mesh="block-grid",\n', "mesh=grid,\n")
        with pytest.raises(PatchError, match="used elsewhere"):
            apply_operation(source, "delete_mesh", mesh="block-grid")

    def test_rejects_an_unknown_reference(self):
        with pytest.raises(PatchError, match="No single mesh"):
            apply_operation(MESHES, "delete_mesh", mesh="nope")
        with pytest.raises(PatchError, match="out of range"):
            apply_operation(MESHES, "delete_mesh", mesh=4)


class TestSetMeshValue:
    def test_rewrites_numeric_arguments_in_place(self):
        patched = apply_operation(
            MESHES, "set_mesh_value", mesh="block-grid", argument="resolution", value=[24, 12, 12]
        )
        assert "resolution=[24, 12, 12]" in patched
        patched = apply_operation(
            patched, "set_mesh_value", mesh=0, argument="bounds", value=[-1.0, -1.0, -1.0]
        )
        assert "bounds=[-1.0, -1.0, -1.0]" in patched
        patched = apply_operation(
            patched, "set_mesh_value", mesh="grid", argument="size", value=[2.0, 2.0, 2.0]
        )
        assert "size=[2.0, 2.0, 2.0]" in patched

    def test_adds_a_missing_keyword(self):
        patched = apply_operation(
            MESHES, "set_mesh_value", mesh="block-grid", argument="padding", value=0.25
        )
        assert "padding=0.25" in patched
        assert ast.parse(patched)

    def test_padding_must_be_non_negative(self):
        with pytest.raises(PatchError, match="non-negative"):
            apply_operation(
                MESHES, "set_mesh_value", mesh="block-grid", argument="padding", value=-0.1
            )

    def test_resolution_stays_integral(self):
        with pytest.raises(PatchError, match="whole numbers"):
            apply_operation(
                MESHES, "set_mesh_value", mesh="block-grid", argument="resolution", value=10.5
            )

    def test_sets_the_domain_to_a_named_object(self):
        patched = apply_operation(
            MESHES, "set_mesh_value", mesh="block-grid", argument="domain", value="block"
        )
        assert "domain=block" in patched
        assert ast.parse(patched)

    def test_rejects_a_domain_that_is_not_assigned_above(self):
        with pytest.raises(PatchError, match="not assigned before"):
            apply_operation(
                MESHES, "set_mesh_value", mesh="block-grid", argument="domain", value="mystery"
            )

    def test_sets_the_meshing_method_as_a_string_literal(self):
        patched = apply_operation(
            MESHES, "set_mesh_value", mesh="block-grid", argument="method", value="tet10"
        )
        assert "method='tet10'" in patched
        assert ast.parse(patched)
        rewritten = apply_operation(
            patched, "set_mesh_value", mesh="block-grid", argument="method", value="hex"
        )
        assert "method='hex'" in rewritten
        assert "tet10" not in rewritten

    def test_rejects_an_unknown_meshing_method(self):
        with pytest.raises(PatchError, match="hex, tet4, tet10"):
            apply_operation(
                MESHES, "set_mesh_value", mesh="block-grid", argument="method", value="voxel"
            )

    def test_rejects_unknown_arguments(self):
        with pytest.raises(PatchError, match="editable arguments"):
            apply_operation(MESHES, "set_mesh_value", mesh="block-grid", argument="name", value=1.0)


class TestSetStudyMeshAndDomain:
    def test_points_a_study_at_a_declared_mesh_by_name(self):
        patched = apply_operation(
            STUDIES.replace(
                "from cadjoint.fem import Dirichlet, Nodes, ThermalStudy",
                "from cadjoint.fem import Dirichlet, Nodes, SimMesh, ThermalStudy",
            ).replace(
                "scene = Union(block)\n",
                "scene = Union(block)\ngrid = SimMesh(name='block-grid', resolution=8)\n",
            ),
            "set_study_value",
            study="bar-conduction",
            argument="mesh",
            value="block-grid",
        )
        assert "mesh='block-grid'" in patched
        # Meshing intent moves onto the SimMesh: the study's own resolution
        # keyword is removed as part of the same edit.
        assert "resolution=12" not in patched
        assert ast.parse(patched)

    def test_accepts_a_mesh_variable_reference(self):
        patched = apply_operation(
            MESHES, "set_study_value", study="bar-conduction", argument="mesh", value="grid"
        )
        assert "mesh=grid" in patched

    def test_rejects_an_undeclared_mesh(self):
        with pytest.raises(PatchError, match="No single SimMesh"):
            apply_operation(
                STUDIES, "set_study_value", study="bar-conduction", argument="mesh", value="nope"
            )

    def test_sets_the_domain_of_an_implicit_study(self):
        patched = apply_operation(
            STUDIES, "set_study_value", study="bar-conduction", argument="domain", value="block"
        )
        assert "domain=block" in patched
        assert ast.parse(patched)

    def test_domain_of_a_mesh_backed_study_lives_on_the_mesh(self):
        with pytest.raises(PatchError, match="set the mesh's `domain`"):
            apply_operation(
                MESHES, "set_study_value", study="bar-conduction", argument="domain", value="block"
            )

    def test_rejects_a_domain_that_is_not_assigned_above(self):
        with pytest.raises(PatchError, match="not assigned before"):
            apply_operation(
                STUDIES, "set_study_value", study="bar-conduction", argument="domain", value="ghost"
            )


OPTIMIZATIONS = """from cadjoint.geometry import Scalar, Vector2
from cadjoint.optimize import Optimization
from cadjoint.sdf.primitives import Box

wall_width = Scalar(0.4, free=True, name="wall_width")
anchor = Vector2(value=[0.5, 0.85], free=True, name="anchor")
scene = Box(size=[1.0, 1.0, 1.0])


def volume(params):
    return params["wall_width"]


shrink = Optimization(name="min-volume", objective=volume, of=scene, steps=25, learning_rate=0.03)
"""


class TestDeleteOptimization:
    def test_removes_a_named_optimization_statement(self):
        patched = apply_operation(OPTIMIZATIONS, "delete_optimization", optimization="min-volume")
        assert "shrink = " not in patched
        assert "def volume(params):" in patched
        assert ast.parse(patched)

    def test_resolves_by_index_and_by_variable(self):
        assert "shrink = " not in apply_operation(
            OPTIMIZATIONS, "delete_optimization", optimization=0
        )
        assert "shrink = " not in apply_operation(
            OPTIMIZATIONS, "delete_optimization", optimization="shrink"
        )

    def test_keeps_every_line_above_the_deletion_untouched(self):
        patched = apply_operation(OPTIMIZATIONS, "delete_optimization", optimization="min-volume")
        original = OPTIMIZATIONS.splitlines()
        assert patched.splitlines() == original[:-1]

    def test_refuses_when_the_optimization_is_used_elsewhere(self):
        source = OPTIMIZATIONS + "result = shrink.run()\n"
        with pytest.raises(PatchError, match="used elsewhere"):
            apply_operation(source, "delete_optimization", optimization="shrink")

    def test_rejects_an_unknown_reference(self):
        with pytest.raises(PatchError, match="No single optimization"):
            apply_operation(OPTIMIZATIONS, "delete_optimization", optimization="nope")
        with pytest.raises(PatchError, match="out of range"):
            apply_operation(OPTIMIZATIONS, "delete_optimization", optimization=4)


class TestSetOptimizationValue:
    def test_rewrites_steps_keeping_them_integral(self):
        patched = apply_operation(
            OPTIMIZATIONS,
            "set_optimization_value",
            optimization="min-volume",
            argument="steps",
            value=40,
        )
        assert "steps=40" in patched
        assert len(patched.splitlines()) == len(OPTIMIZATIONS.splitlines())

    def test_rewrites_the_learning_rate_with_exact_repr(self):
        patched = apply_operation(
            OPTIMIZATIONS,
            "set_optimization_value",
            optimization=0,
            argument="learning_rate",
            value=0.005,
        )
        assert "learning_rate=0.005" in patched

    def test_adds_an_absent_keyword(self):
        bare = OPTIMIZATIONS.replace(", steps=25, learning_rate=0.03", "")
        patched = apply_operation(
            bare,
            "set_optimization_value",
            optimization="min-volume",
            argument="steps",
            value=8,
        )
        assert "of=scene, steps=8)" in patched

    def test_rejects_unknown_arguments_and_bad_values(self):
        with pytest.raises(PatchError, match="editable arguments"):
            apply_operation(
                OPTIMIZATIONS,
                "set_optimization_value",
                optimization=0,
                argument="method",
                value=1,
            )
        with pytest.raises(PatchError, match="whole number"):
            apply_operation(
                OPTIMIZATIONS,
                "set_optimization_value",
                optimization=0,
                argument="steps",
                value=2.5,
            )
        with pytest.raises(PatchError, match="positive"):
            apply_operation(
                OPTIMIZATIONS,
                "set_optimization_value",
                optimization=0,
                argument="learning_rate",
                value=-0.1,
            )


class TestSetParameterValues:
    def test_rewrites_named_scalar_and_vector2_literals_exactly(self):
        from cadjoint.viewer._patch import set_parameter_values

        patched = set_parameter_values(
            OPTIMIZATIONS,
            {"wall_width": 0.9750000238418579, "anchor": [0.51, 0.8399999737739563]},
        )
        assert "Scalar(0.9750000238418579, free=True" in patched
        assert "value=[0.51, 0.8399999737739563]" in patched
        assert len(patched.splitlines()) == len(OPTIMIZATIONS.splitlines())
        assert ast.parse(patched)

    def test_rejects_a_parameter_without_a_single_declaration(self):
        from cadjoint.viewer._patch import set_parameter_values

        with pytest.raises(PatchError, match="exactly one"):
            set_parameter_values(OPTIMIZATIONS, {"ghost": 1.0})
        doubled = OPTIMIZATIONS + 'again = Scalar(0.5, name="wall_width")\n'
        with pytest.raises(PatchError, match="exactly one"):
            set_parameter_values(doubled, {"wall_width": 1.0})
