"""One test per defect the editing-operations spec exposed, each pinned to its fix.

``research/editing-operations.md`` states what every operation must do;
``test_patch_properties.py`` checks the invariants over generated requests.
This module keeps the individual findings: the exact request that used to
produce a program that would not compile, crash the server, or edit the
wrong thing, and the refusal or rewrite it produces now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cadjoint.viewer._patch_requests import patch_source
from cadjoint.viewer._source_map import PLAYGROUND_FILENAME, capture_profiles, identity_index
from cadjoint.viewer._worker_scene import _execute_scene
from cadjoint.viewer.patch import PatchError, apply_operation

SCENES_DIR = Path(__file__).resolve().parents[2] / "scenes"
STARTER = (SCENES_DIR / "starter.py").read_text()
END_CAP = (SCENES_DIR / "end_cap.py").read_text()
BRACKET = (SCENES_DIR / "bracket.py").read_text()

PLATE = """\
from cadjoint.constraints import DistanceConstraint, FixedConstraint
from cadjoint.construction import PolygonProfile, extrude
plate = PolygonProfile([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], name="plate")
FixedConstraint(plate.vertices[0], [0.0, 0.0])
DistanceConstraint(plate.vertices[2], plate.vertices[3], 1.0)
scene = extrude(plate, depth=0.5)
"""


def compiles(source: str) -> None:
    _execute_scene(
        source, capture=(("__profiles__", lambda: capture_profiles(PLAYGROUND_FILENAME)),)
    )


def refused(source: str, **request) -> str:
    result = patch_source({"source": source, **request})
    assert result["ok"] is False, "expected a refusal"
    return result["error"]


def accepted(source: str, **request) -> str:
    result = patch_source({"source": source, **request})
    assert result["ok"] is True, result.get("error")
    return result["source"]


class TestSetValueWritesOnlyWhatTheCallTakes:
    """``set_value`` used to append a second keyword — or an unknown one — verbatim."""

    def test_a_keyword_bound_to_an_expression_is_refused_not_duplicated(self):
        # ``bolt_head``'s position is ``[_corner, _corner, 0.215]``: not a literal.
        error = refused(
            END_CAP,
            op="set_value",
            id="assign:bolt_head",
            name="cylinder",
            argument="position",
            value=[1, 1, 1],
        )
        assert error == "The cylinder's `position` is not an editable literal; edit it in the code."

    def test_a_face_referenced_sketch_plane_is_refused_not_duplicated(self):
        error = refused(
            END_CAP,
            op="set_value",
            id="assign:pad_profile",
            name="PolygonProfile",
            argument="planeOrigin",
            value=[0, 0, 1],
        )
        assert error.startswith("The sketch's `plane` is an expression over other geometry")

    def test_a_plane_reached_through_a_name_is_rewritten_at_its_definition(self):
        source = (
            "from cadjoint.construction import PolygonProfile, SketchPlane\n"
            "p = SketchPlane(origin=[0, 0, 0], normal=[0, 0, 1])\n"
            "profile = PolygonProfile([[0, 0], [1, 0], [0, 1]], plane=p)\n"
        )
        patched = apply_operation(
            source,
            "set_value",
            line=3,
            name="PolygonProfile",
            argument="planeOrigin",
            value=[1, 2, 3],
        )
        assert "p = SketchPlane(origin=[1, 2, 3], normal=[0, 0, 1])" in patched
        assert patched.count("plane=") == 1

    def test_the_plane_call_is_addressable_as_the_viewport_addresses_it(self):
        # Dragging a sketch as an object targets its plane, and the gizmo names
        # the call and keyword as SketchPlane names them, not as the profile's
        # planeOrigin alias does; both must land in the same SketchPlane(...).
        patched = patch_source(
            {
                "source": STARTER,
                "op": "set_value",
                "line": 151,
                "name": "SketchPlane",
                "argument": "origin",
                "value": [0, 0.527380126689968, 0],
            }
        )
        assert patched["ok"], patched
        assert "SketchPlane(origin=[0, 0.5274, 0]" in patched["source"]

    def test_an_argument_the_call_does_not_take_is_refused(self):
        error = refused(
            STARTER, op="set_value", id="assign:board", name="box", argument="foo", value=1
        )
        assert error == "`box` has no editable argument `foo`; expected: position, rotation, size."

    def test_a_call_outside_the_vocabulary_is_refused(self):
        error = refused(
            STARTER, op="set_value", id="assign:board", name="Union", argument="smoothness", value=1
        )
        assert error.startswith(
            "`set_value` edits one of these calls: Material, PolygonProfile, SketchPlane, box"
        )

    def test_the_value_shape_follows_the_argument(self):
        assert (
            refused(
                STARTER, op="set_value", id="assign:board", name="box", argument="size", value=1
            )
            == "`size` needs 3 numbers."
        )
        assert (
            refused(
                STARTER,
                op="set_value",
                id="assign:bush_a",
                name="cylinder",
                argument="radius",
                value=[1, 2],
            )
            == "`radius` needs one number."
        )

    def test_a_material_scalar_keeps_the_brackets_the_inspector_enforces(self):
        error = refused(
            STARTER,
            op="set_value",
            id="assign:aluminum",
            name="Material",
            argument="roughness",
            value=5,
        )
        assert error == "`roughness` must be a number from 0 to 1 (dimensionless)."
        patched = accepted(
            STARTER,
            op="set_value",
            id="assign:aluminum",
            name="Material",
            argument="color",
            value=[0.1, 0.2, 0.3],
        )
        assert "color=[0.1, 0.2, 0.3]" in patched

    def test_non_finite_and_empty_values_are_refused(self):
        for value in ([float("nan"), 1, 1], [float("inf"), 1, 1], []):
            error = refused(
                STARTER, op="set_value", id="assign:board", name="box", argument="size", value=value
            )
            assert error == "The patch request needs `value` as a number or numbers."


class TestAddPrimitiveWritesOnlyWhatSolidBuilds:
    """An unknown kind compiled to ``Solid.torus``; empty dimensions crashed the server."""

    def test_an_unknown_kind_is_refused(self):
        error = refused(
            STARTER, op="add_primitive", kind="torus", position=[0, 0, 0], dimensions={"radius": 1}
        )
        assert error == "Primitive `kind` must be one of: box, cylinder, sphere."

    def test_missing_or_foreign_dimensions_are_refused(self):
        for dimensions in ({}, {"radius": 1}, {"size": [1, 1, 1], "radius": 1}):
            error = refused(
                STARTER, op="add_primitive", kind="box", position=[0, 0, 0], dimensions=dimensions
            )
            assert error == "A `box` takes exactly these dimensions: `size`."

    def test_the_operation_itself_refuses_too(self):
        with pytest.raises(PatchError, match="Primitive `kind`"):
            apply_operation(
                STARTER, "add_primitive", kind="torus", position=[0, 0, 0], dimensions={}
            )
        with pytest.raises(PatchError, match="exactly these dimensions"):
            apply_operation(STARTER, "add_primitive", kind="box", position=[0, 0, 0], dimensions={})


class TestConstraintsNameVerticesTheSketchHas:
    """``profile.vertices[99]`` was written verbatim and raised on the next compile."""

    def test_an_index_past_the_end_is_refused(self):
        error = refused(
            BRACKET,
            op="add_constraint",
            id="assign:rib_profile",
            kind="horizontal",
            indices=[0, 99],
        )
        assert error == "Vertex index 99 is out of range; the sketch has 3 vertices."

    def test_an_edge_needs_two_different_vertices(self):
        error = refused(
            BRACKET, op="add_constraint", id="assign:rib_profile", kind="horizontal", indices=[1, 1]
        )
        assert error == "A constraint edge needs two different vertices, not 1 twice."

    def test_a_pin_is_a_point_and_a_distance_is_a_length(self):
        assert (
            refused(
                BRACKET,
                op="add_constraint",
                id="assign:rib_profile",
                kind="fixed",
                indices=[0],
                value=0.5,
            )
            == "A `fixed` constraint needs `value` as two numbers."
        )
        assert (
            refused(
                BRACKET,
                op="add_constraint",
                id="assign:rib_profile",
                kind="distance",
                indices=[0, 1],
                value=-1,
            )
            == "A `distance` constraint needs `value` as a non-negative number."
        )
        pinned = accepted(
            BRACKET,
            op="add_constraint",
            id="assign:rib_profile",
            kind="fixed",
            indices=[0],
            value=[0.5, 0.1],
        )
        compiles(pinned)
        # ``set_constraint_value`` keeps the same shapes.
        assert (
            refused(pinned, op="set_constraint_value", id="constraint:rib_profile[1]", value=0.5)
            == "A `fixed` constraint needs `value` as two numbers."
        )
        compiles(
            accepted(
                pinned, op="set_constraint_value", id="constraint:rib_profile[1]", value=[0.2, 0.3]
            )
        )

    def test_a_kind_that_is_not_a_string_is_refused_not_raised(self):
        error = refused(
            STARTER, op="add_constraint", id="assign:comb_profile", kind=["fixed"], indices=[0]
        )
        assert error.startswith("Constraint `kind` must be one of:")


class TestVertexEditsKeepConstraintsPointingAtTheirVertices:
    """Deleting vertex 1 used to leave ``vertices[2]`` naming what was vertex 3."""

    def test_deleting_a_vertex_renumbers_the_subscripts_after_it(self):
        patched = accepted(PLATE, op="delete_vertex", id="vertex:plate[1]")
        assert "FixedConstraint(plate.vertices[0], [0.0, 0.0])" in patched
        assert "DistanceConstraint(plate.vertices[1], plate.vertices[2], 1.0)" in patched
        compiles(patched)

    def test_deleting_a_constrained_vertex_deletes_its_constraints(self):
        patched = accepted(PLATE, op="delete_vertex", id="vertex:plate[0]")
        assert "FixedConstraint(" not in patched
        assert "DistanceConstraint(plate.vertices[1], plate.vertices[2], 1.0)" in patched
        compiles(patched)

    def test_inserting_a_vertex_renumbers_the_subscripts_from_it(self):
        patched = accepted(PLATE, op="insert_vertex", id="vertex:plate[1]", xy=[0.5, 0])
        assert "FixedConstraint(plate.vertices[0], [0.0, 0.0])" in patched
        assert "DistanceConstraint(plate.vertices[3], plate.vertices[4], 1.0)" in patched
        compiles(patched)
        # And deleting it again is the exact inverse, constraints included.
        assert accepted(patched, op="delete_vertex", id="vertex:plate[1]") == PLATE

    def test_a_vertex_named_by_parameter_takes_its_constraints_with_it(self):
        # ``base_r`` is a Vector2 in the comb; four constraints name it.
        patched = accepted(STARTER, op="delete_vertex", id="vertex:comb_profile[1]")
        assert "base_r,\n" not in patched
        assert "DistanceConstraint(base_l, base_r, base_width)" not in patched
        assert "HorizontalConstraint(base_l, base_r)" not in patched
        assert "base_r = Vector2(" in patched, "the declaration stays; only its uses go"
        compiles(patched)

    def test_an_anonymous_sketch_has_no_constraints_to_renumber(self):
        source = "from cadjoint.construction import PolygonProfile\nPolygonProfile([[0, 0], [1, 0], [1, 1], [0, 1]])\n"
        patched = apply_operation(source, "delete_vertex", line=2, index=1)
        assert "PolygonProfile([[0, 0], [1, 1], [0, 1]])" in patched


class TestFeaturesAreObjectsToo:
    """``_ID_TARGETS`` promised a feature could be deleted and given a material."""

    def test_a_feature_that_only_a_union_uses_is_deleted(self):
        patched = accepted(BRACKET, op="delete_object", id="assign:plate")
        assert "plate = extrude(" not in patched
        assert "body = Union(web, rib, smoothness=0.05)" in patched
        assert "plate_profile = PolygonProfile(" in patched, "the sketch it consumed stays"
        compiles(patched)

    def test_a_feature_used_elsewhere_is_refused(self):
        error = refused(STARTER, op="delete_object", id="assign:sink")
        assert error.startswith("`sink` is used elsewhere in the program")

    def test_a_material_lands_on_the_feature_itself(self):
        patched = accepted(STARTER, op="assign_material", id="assign:slug", material="steel")
        assert "slug = revolve(slug_profile, material=steel)" in patched

    def test_a_revolved_sketch_takes_a_material_through_its_revolve(self):
        patched = accepted(
            STARTER, op="assign_material", id="assign:slug_profile", material="steel"
        )
        assert "slug = revolve(slug_profile, material=steel)" in patched

    def test_a_bare_sketch_says_what_it_needs(self):
        with pytest.raises(PatchError, match="needs one operator"):
            apply_operation(
                BRACKET.replace(
                    "plate = extrude(plate_profile, depth=plate_thickness, material=steel)\n", ""
                ).replace("Union(plate, web, rib", "Union(web, rib"),
                "assign_material",
                line=identity_index(BRACKET)["assign:plate_profile"].line,
                material="steel",
            )


class TestOneSketchOneSolid:
    """``add_extrusion`` accepted a sketch that a revolve already consumed."""

    def test_every_operator_refuses_a_consumed_sketch_the_same_way(self):
        message = "`slug_profile` already has an operator."
        assert refused(STARTER, op="add_extrusion", id="assign:slug_profile") == message
        assert refused(STARTER, op="add_revolution", id="assign:slug_profile") == message
        assert (
            refused(STARTER, op="add_loft", id_a="assign:slug_profile", id_b="assign:comb_profile")
            == message
        )


class TestUnionsNeverGoEmpty:
    """Deleting the last operand left ``body = Union(smoothness=0.05)``."""

    def test_the_last_operand_is_refused(self):
        source = accepted(BRACKET, op="delete_object", id="assign:plate")
        source = accepted(source, op="delete_object", id="assign:web")
        error = refused(source, op="delete_object", id="assign:rib")
        assert error == (
            "`rib` is the last operand of `body = Union(...)`, so deleting it would leave an "
            "empty union. Remove that union in the code first."
        )

    def test_a_trailing_operand_takes_its_separator_with_it(self):
        source = (
            "from cadjoint.construction import Solid\n"
            "from cadjoint.sdf.boolean import Union\n"
            'a = Solid.box(size=[1, 1, 1], name="a")\n'
            'b = Solid.sphere(radius=0.5, name="b")\n'
            "scene = Union(a, b)\n"
        )
        patched = apply_operation(source, "delete_object", line=4)
        assert "scene = Union(a)\n" in patched


class TestDeclarationsStayConsistent:
    def test_a_mesh_domain_is_replaced_whatever_it_named(self):
        # ``sink_mesh`` already has ``domain=thermal_body``; the edit used to be refused.
        patched = accepted(
            STARTER, op="set_mesh_value", id="assign:sink_mesh", argument="domain", value="board"
        )
        assert "    domain=board,\n" in patched
        assert "domain=thermal_body" not in patched.split("sink_mesh = SimMesh(")[1].split(")")[0]
        compiles(patched)

    def test_meshing_intent_stays_on_the_mesh_of_a_mesh_backed_study(self):
        for argument, value in (("resolution", 10), ("bounds", [0, 0, 0]), ("size", [1, 1, 1])):
            error = refused(
                STARTER,
                op="set_study_value",
                id="assign:heat_study",
                argument=argument,
                value=value,
            )
            assert error == (
                f"This study solves on a SimMesh; set the mesh's `{argument}` instead (set_mesh_value)."
            )

    def test_bounds_and_size_are_a_pair(self):
        grown = accepted(STARTER, op="add_mesh", name="probe")
        error = refused(
            grown, op="set_mesh_value", mesh="probe", argument="bounds", value=[0, 0, 0]
        )
        assert error == (
            "`bounds` and `size` are stated together or not at all; this mesh states neither, "
            "so add both in the code first."
        )
        study = accepted(STARTER, op="add_study", kind="thermal", name="probe")
        error = refused(
            study, op="set_study_value", study="probe", argument="size", value=[1, 1, 1]
        )
        assert "this study states neither" in error

    def test_the_last_boundary_condition_leaves_an_empty_list(self):
        once = accepted(STARTER, op="delete_study_bc", id="bc:heat_study[0]")
        twice = accepted(once, op="delete_study_bc", id="bc:heat_study[0]")
        assert "    bcs=[],\n" in twice
        compiles(twice)


class TestFaceReferencesNameFacesTheSolidDeclares:
    """``SketchPlane.on(plate.face('+z'))`` compiled to a KeyError."""

    def test_an_extrusion_has_caps_and_sides_not_axis_faces(self):
        sketch = identity_index(BRACKET)["assign:web_profile"]
        assert sketch.line > identity_index(BRACKET)["assign:plate"].line
        error = refused(
            BRACKET,
            op="set_sketch_plane",
            id=sketch.id,
            reference={"kind": "face", "owner": "assign:plate", "key": "+z"},
        )
        assert (
            error
            == "A extrude has no face '+z'; `plate` declares: cap+, cap-, side0, side1, side2, side3."
        )
        error = refused(
            BRACKET,
            op="set_sketch_plane",
            id=sketch.id,
            reference={"kind": "side", "owner": "assign:plate", "edge": 4},
        )
        assert error == "`plate` has 4 sides, so `edge` must be from 0 to 3."
        compiles(
            accepted(
                BRACKET,
                op="set_sketch_plane",
                id=sketch.id,
                reference={"kind": "side", "owner": "assign:plate", "edge": 3},
            )
        )

    def test_a_revolve_declares_no_faces(self):
        error = refused(
            END_CAP,
            op="set_sketch_plane",
            id="assign:rib_profile",
            reference={"kind": "cap", "owner": "assign:seat_cut", "sign": "+"},
        )
        assert error == "A revolve declares no cap faces; `seat_cut` has no `cap`."
        error = refused(
            END_CAP,
            op="set_sketch_plane",
            id="assign:rib_profile",
            reference={"kind": "side", "owner": "assign:seat_cut", "edge": 0},
        )
        assert error == "A revolve declares no side faces; `seat_cut` has no `side`."

    def test_a_box_declares_axis_faces_and_a_cylinder_caps(self):
        error = refused(
            STARTER,
            op="set_sketch_plane",
            id="assign:slug_profile",
            reference={"kind": "cap", "owner": "assign:bush_a", "sign": "+"},
        )
        # ``bush_a`` is declared after the slug sketch, so ordering answers first.
        assert "a sketch can only sit on geometry built before it" in error
        source = STARTER.replace(
            "scene = Union(",
            'probe = PolygonProfile([[0, 0], [1, 0], [1, 1]], name="probe")\nscene = Union(',
        )
        assert (
            refused(
                source,
                op="set_sketch_plane",
                id="assign:probe",
                reference={"kind": "face", "owner": "assign:board", "key": "cap+"},
            )
            == "A box has no face 'cap+'; `board` declares: +x, +y, +z, -x, -y, -z."
        )
        compiles(
            accepted(
                source,
                op="set_sketch_plane",
                id="assign:probe",
                reference={"kind": "face", "owner": "assign:board", "key": "+z"},
            )
        )
        compiles(
            accepted(
                source,
                op="set_sketch_plane",
                id="assign:probe",
                reference={"kind": "cap", "owner": "assign:bush_a", "sign": "+"},
            )
        )


class TestMaterialPropertyRemovalIsExact:
    def test_a_wrapped_keyword_leaves_no_dangling_comma(self):
        grown = accepted(
            STARTER,
            op="set_material_property",
            id="assign:black_oxide",
            property="density",
            value=8500.0,
        )
        assert (
            "metallic=0.85,\n    density=8500.0)" in grown
        ), "the keyword wrapped onto its own line"
        assert (
            accepted(
                grown,
                op="set_material_property",
                id="assign:black_oxide",
                property="density",
                value=None,
            )
            == STARTER
        )


class TestTheRequestSurfaceIsExact:
    def test_a_stray_field_is_refused(self):
        error = refused(STARTER, op="delete_object", id="assign:board", line=231, force=True)
        assert error == (
            "The patch operation `delete_object` does not take `force`. "
            "If you updated cadjoint, restart the playground server."
        )

    def test_the_id_checks_answer_before_the_field_check(self):
        assert refused(STARTER, op="add_sketch", id="assign:board", origin=[0, 0, 0]) == (
            "The patch operation `add_sketch` creates a new object, so it takes no `id`."
        )
        assert refused(STARTER, op="add_loft", id="assign:comb_profile") == (
            "`add_loft` names its two sketches with `id_a` and `id_b`, not `id`."
        )

    def test_every_field_the_frontend_sends_is_a_model_field(self):
        # The shapes ``frontend/src`` builds: id and line together, study and
        # mesh by id and index together.
        line = identity_index(STARTER)["assign:board"].line
        assert patch_source(
            {"source": STARTER, "op": "delete_object", "id": "assign:board", "line": line}
        )["ok"]
        assert patch_source(
            {
                "source": STARTER,
                "op": "delete_study_bc",
                "id": "assign:heat_study",
                "study": 0,
                "bc": 0,
            }
        )["ok"]
