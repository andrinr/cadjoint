"""Face references end to end: payload, locators, and the plane patch.

The viewer resolves a raymarch hit to a face client-side, against the polygons
this payload carries, and then asks the server to write the reference into the
program. These tests pin both halves of that contract — the shape of a face
entry, and the source a ``set_sketch_plane`` request produces — plus the round
trip that proves the written source rebuilds the same plane.
"""

from __future__ import annotations

import pytest

from cadjoint.viewer._patch import PatchError, apply_operation, set_sketch_plane
from cadjoint.viewer._patch_requests import patch_source
from cadjoint.viewer._source_map import (
    PLAYGROUND_FILENAME,
    build_construction_payload,
    capture_profiles,
    locate_feature_call,
    locate_feature_calls,
    locate_plane_reference,
)

STACK = """from cadjoint.construction import PolygonProfile, Solid, extrude
from cadjoint.sdf.boolean import Union

base = PolygonProfile([[-1, -1], [1, -1], [1, 1], [-1, 1]], name="base")
body = extrude(base, depth=0.6)
boss = PolygonProfile([[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]], name="boss")
boss_body = extrude(boss, depth=0.4)
block = Solid.box(size=[0.5, 0.5, 0.5], position=[2.0, 0.0, 0.0], name="block")
scene = Union(body, boss_body, block)
"""

PRIMITIVE_FIRST = """from cadjoint.construction import PolygonProfile, Solid, extrude
from cadjoint.sdf.boolean import Union

block = Solid.box(size=[0.5, 0.5, 0.5], position=[2.0, 0.0, 0.0], name="block")
lid = PolygonProfile([[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]], name="lid")
scene = Union(block, extrude(lid, depth=0.2))
"""

ANONYMOUS = """from cadjoint.construction import PolygonProfile, extrude
from cadjoint.sdf.boolean import Union

base = PolygonProfile([[-1, -1], [1, -1], [1, 1], [-1, 1]], name="base")
scene = Union(extrude(base, depth=0.6))
"""


def run(source: str):
    """Execute a program the way the compile worker does, capturing profiles."""
    namespace = {"__builtins__": __builtins__, "__name__": "__cadjoint_playground__"}
    with capture_profiles(PLAYGROUND_FILENAME) as captured:
        exec(compile(source, PLAYGROUND_FILENAME, "exec"), namespace, namespace)
    return build_construction_payload(captured, source)


def faces_of(source: str, node_id: str) -> dict[str, dict]:
    entry = next(node for node in run(source) if node["id"] == node_id)
    return {face["id"]: face for face in entry["faces"]}


# ── Payload ──────────────────────────────────────────────────────────────────


class TestFacePayload:
    def test_a_sketch_lists_the_faces_of_the_feature_it_generated(self):
        assert set(faces_of(STACK, "profile_0")) == {
            "profile_0:cap+",
            "profile_0:cap-",
            "profile_0:side0",
            "profile_0:side1",
            "profile_0:side2",
            "profile_0:side3",
        }

    def test_a_face_carries_its_plane_and_a_polygon_to_highlight(self):
        cap = faces_of(STACK, "profile_0")["profile_0:cap+"]
        assert cap["origin"] == pytest.approx([0.0, 0.0, 0.3], abs=1e-6)
        assert cap["normal"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
        assert cap["xAxis"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
        assert cap["yAxis"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
        assert len(cap["polygon"]) == 4
        assert all(point[2] == pytest.approx(0.3, abs=1e-6) for point in cap["polygon"])
        assert cap["tolerance"] > 0.0

    def test_a_face_names_the_owner_a_patch_would_write(self):
        cap = faces_of(STACK, "profile_0")["profile_0:cap+"]
        assert cap["owner"] == {"kind": "extrude", "line": 5, "variable": "body"}
        assert cap["reference"] == {"call": "cap", "args": ["+"]}
        assert cap["usable"] is True

    def test_a_primitive_lists_its_six_faces(self):
        faces = faces_of(STACK, "box_2")
        assert set(faces) == {f"box_2:{key}" for key in ("+x", "-x", "+y", "-y", "+z", "-z")}
        assert faces["box_2:+x"]["origin"] == pytest.approx([2.5, 0.0, 0.0], abs=1e-6)
        assert faces["box_2:+x"]["owner"]["variable"] == "block"
        assert faces["box_2:+x"]["reference"] == {"call": "face", "args": ["+x"]}

    def test_faces_of_an_unnamed_feature_are_drawn_but_not_usable(self):
        faces = faces_of(ANONYMOUS, "profile_0")
        assert set(faces)  # still highlightable
        assert all(face["usable"] is False for face in faces.values())

    def test_a_sketch_with_no_feature_lists_no_faces(self):
        source = 'from cadjoint.construction import PolygonProfile\nscene = PolygonProfile([[0, 0], [1, 0], [0, 1]], name="p")\n'
        assert run(source)[0]["faces"] == []

    def test_the_payload_reports_the_reference_a_sketch_sits_on(self):
        planted = set_sketch_plane(STACK, 6, {"kind": "cap", "owner": 5, "sign": "+"})
        entry = next(node for node in run(planted) if node["id"] == "profile_1")
        assert entry["plane"]["reference"] == {
            "constructor": "on",
            "owner": "body",
            "accessor": "cap",
            "argument": "'+'",
        }

    def test_a_world_plane_reports_no_reference(self):
        entry = next(node for node in run(STACK) if node["id"] == "profile_0")
        assert entry["plane"]["reference"] is None


# ── Locators ─────────────────────────────────────────────────────────────────


class TestFeatureLocators:
    def test_finds_every_feature_call_with_its_binding(self):
        calls = locate_feature_calls(STACK)
        assert [(call.line, call.kind, call.variable) for call in calls] == [
            (5, "extrude", "body"),
            (7, "extrude", "boss_body"),
            (8, "box", "block"),
        ]

    def test_a_nested_call_binds_no_variable(self):
        assert locate_feature_calls(ANONYMOUS)[0].variable is None

    def test_a_line_without_a_feature_resolves_to_nothing(self):
        assert locate_feature_call(STACK, 4) is None

    def test_unparseable_source_is_refused(self):
        assert locate_feature_calls("def (") is None
        assert locate_plane_reference("def (", 1) is None

    def test_reads_back_a_tangent_reference(self):
        planted = set_sketch_plane(
            STACK, 6, {"kind": "tangent", "owner": 5, "near": [0.5, 0.0, 0.3]}
        )
        reference = locate_plane_reference(planted, 6)
        assert (reference.constructor, reference.owner) == ("tangent", "body")

    def test_reads_back_a_plain_plane(self):
        planted = set_sketch_plane(
            STACK, 6, {"kind": "world", "origin": [0, 0, 1], "normal": [0, 0, 1]}
        )
        assert locate_plane_reference(planted, 6).constructor == "plain"


# ── The patch operation ──────────────────────────────────────────────────────


def plane_argument(source: str, line: int) -> str:
    reference = locate_plane_reference(source, line)
    start, end = reference.span
    return source[start:end]


class TestSetSketchPlane:
    def test_writes_a_cap_reference_and_its_import(self):
        patched = set_sketch_plane(STACK, 6, {"kind": "cap", "owner": 5, "sign": "+"})
        assert plane_argument(patched, 6) == "SketchPlane.on(body.cap('+'))"
        assert "SketchPlane" in patched.splitlines()[0]

    def test_writes_a_side_reference(self):
        patched = set_sketch_plane(STACK, 6, {"kind": "side", "owner": 5, "edge": 2})
        assert plane_argument(patched, 6) == "SketchPlane.on(body.side(2))"

    def test_writes_a_primitive_face_reference(self):
        patched = set_sketch_plane(PRIMITIVE_FIRST, 5, {"kind": "face", "owner": 4, "key": "+x"})
        assert plane_argument(patched, 5) == "SketchPlane.on(block.face('+x'))"

    def test_writes_a_tangent_reference_from_a_hit_point(self):
        patched = set_sketch_plane(
            STACK, 6, {"kind": "tangent", "owner": 5, "near": [0.5, 0.0, 0.3]}
        )
        assert plane_argument(patched, 6) == "SketchPlane.tangent(body, near=[0.5, 0, 0.3])"

    def test_carries_flip_x_axis_and_offset(self):
        patched = set_sketch_plane(
            STACK,
            6,
            {"kind": "cap", "owner": 5, "sign": "-"},
            x_axis=[0.0, 1.0, 0.0],
            flip=True,
            offset=0.25,
        )
        assert plane_argument(patched, 6) == (
            "SketchPlane.offset(SketchPlane.on(body.cap('-'), x_axis=[0, 1, 0], flip=True), 0.25)"
        )

    def test_replaces_an_existing_plane_argument(self):
        once = set_sketch_plane(STACK, 6, {"kind": "cap", "owner": 5, "sign": "+"})
        twice = set_sketch_plane(once, 6, {"kind": "cap", "owner": 5, "sign": "-"})
        assert plane_argument(twice, 6) == "SketchPlane.on(body.cap('-'))"
        assert twice.count("SketchPlane.on") == 1

    def test_leaves_line_numbers_and_the_rest_of_the_file_alone(self):
        patched = set_sketch_plane(STACK, 6, {"kind": "cap", "owner": 5, "sign": "+"})
        assert len(patched.splitlines()) == len(STACK.splitlines())
        assert patched.splitlines()[3] == STACK.splitlines()[3]

    def test_refuses_a_feature_defined_after_the_sketch(self):
        with pytest.raises(PatchError, match="only sit on geometry built before it"):
            set_sketch_plane(STACK, 6, {"kind": "cap", "owner": 7, "sign": "+"})

    def test_refuses_a_feature_with_no_name_to_write(self):
        with pytest.raises(PatchError, match="not assigned to a variable"):
            set_sketch_plane(ANONYMOUS, 4, {"kind": "cap", "owner": 5, "sign": "+"})

    def test_refuses_a_line_that_is_not_a_sketch(self):
        with pytest.raises(PatchError, match="No single PolygonProfile"):
            set_sketch_plane(STACK, 5, {"kind": "cap", "owner": 5, "sign": "+"})

    def test_is_reachable_through_the_operation_registry(self):
        patched = apply_operation(
            STACK, "set_sketch_plane", line=6, reference={"kind": "cap", "owner": 5, "sign": "+"}
        )
        assert "SketchPlane.on(body.cap('+'))" in patched


class TestSetSketchPlaneRequests:
    def request(self, **fields):
        return patch_source({"source": STACK, "op": "set_sketch_plane", **fields})

    def test_accepts_a_well_formed_request(self):
        result = self.request(line=6, reference={"kind": "cap", "owner": 5, "sign": "+"})
        assert result["ok"] is True
        assert "SketchPlane.on(body.cap('+'))" in result["source"]

    def test_needs_a_line(self):
        result = self.request(reference={"kind": "cap", "owner": 5, "sign": "+"})
        assert result["error"] == "The patch request needs an integer `line`."

    def test_needs_a_reference_object(self):
        assert self.request(line=6, reference=4)["error"] == (
            "The patch request needs `reference` as an object."
        )

    def test_names_the_reference_kinds_it_knows(self):
        result = self.request(line=6, reference={"kind": "blend"})
        assert (
            result["error"]
            == "Plane `reference.kind` must be one of: cap, face, side, tangent, world."
        )

    def test_checks_each_kinds_own_field(self):
        assert self.request(line=6, reference={"kind": "cap", "owner": 5, "sign": 1})["error"] == (
            "A cap reference needs `sign` as `+` or `-`."
        )
        assert (
            self.request(line=6, reference={"kind": "side", "owner": 5, "edge": -1})["error"]
            == "A side reference needs a non-negative `edge` index."
        )
        assert (
            self.request(line=6, reference={"kind": "face", "owner": 5, "key": " "})["error"]
            == "A face reference needs a non-empty `key`."
        )
        assert (
            self.request(line=6, reference={"kind": "tangent", "owner": 5, "near": [0, 0]})["error"]
            == "A tangent reference needs `near` as three numbers."
        )

    def test_needs_an_owner_line(self):
        assert self.request(line=6, reference={"kind": "cap", "sign": "+"})["error"] == (
            "The plane reference needs an integer `owner` line."
        )

    def test_rejects_a_degenerate_x_axis(self):
        result = self.request(
            line=6, reference={"kind": "cap", "owner": 5, "sign": "+"}, x_axis=[0, 0, 0]
        )
        assert result["error"] == "`x_axis` must be three numbers and must not be zero."

    def test_rejects_a_non_boolean_flip(self):
        result = self.request(
            line=6, reference={"kind": "cap", "owner": 5, "sign": "+"}, flip="yes"
        )
        assert result["error"] == "The patch request needs `flip` as a boolean."

    def test_rejects_a_non_numeric_offset(self):
        result = self.request(
            line=6, reference={"kind": "cap", "owner": 5, "sign": "+"}, offset="far"
        )
        assert result["error"] == "The patch request needs a numeric `offset`."

    def test_a_world_plane_needs_a_non_zero_normal(self):
        result = self.request(
            line=6, reference={"kind": "world", "origin": [0, 0, 0], "normal": [0, 0, 0]}
        )
        assert result["error"] == "A sketch-plane normal must not be zero."

    def test_reports_a_failed_edit_rather_than_raising(self):
        result = self.request(line=6, reference={"kind": "cap", "owner": 7, "sign": "+"})
        assert result["ok"] is False
        assert "built before it" in result["error"]


# ── Round trip ───────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_the_written_reference_rebuilds_the_plane_it_named(self):
        patched = set_sketch_plane(STACK, 6, {"kind": "cap", "owner": 5, "sign": "+"})
        entry = next(node for node in run(patched) if node["id"] == "profile_1")
        # The boss now sits on the base's top cap: depth 0.6 -> z = 0.3.
        assert entry["plane"]["origin"] == pytest.approx([0.0, 0.0, 0.3], abs=1e-6)
        assert entry["plane"]["normal"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)

    def test_the_plane_follows_the_parent_when_the_depth_changes(self):
        patched = set_sketch_plane(STACK, 6, {"kind": "cap", "owner": 5, "sign": "+"})
        deeper = patched.replace("depth=0.6", "depth=1.4")
        entry = next(node for node in run(deeper) if node["id"] == "profile_1")
        assert entry["plane"]["origin"] == pytest.approx([0.0, 0.0, 0.7], abs=1e-6)

    def test_the_scene_still_compiles_for_the_viewer(self):
        from cadjoint.viewer.playground import compile_source

        patched = set_sketch_plane(STACK, 6, {"kind": "cap", "owner": 5, "sign": "+"})
        result = compile_source(patched)
        assert result["ok"] is True
        assert result["shader"]
        faces = [face for node in result["construction"] for face in node["faces"]]
        assert {face["id"] for face in faces} >= {"profile_0:cap+", "box_2:+z"}
