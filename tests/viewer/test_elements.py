"""The construction-element payload the properties window reads.

Every element is built statically from the source, so these tests never run
a scene: they check what the extractor says about the shipped programs and
about the edge cases the window has to render — a sketch with no plane, a
plane derived from a face, a boolean addressed by line — and that every
patch address an element publishes is one the request pipeline accepts and
the compile worker then reports back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cadjoint.viewer._compile_worker import _compile_source
from cadjoint.viewer._patch_requests import patch_source
from cadjoint.viewer.patch.geometry import EDITABLE_CALLS
from cadjoint.viewer.schema import CompilePayload, ConstructionElement
from cadjoint.viewer.source_map.elements import build_element_payload

SCENES_DIR = Path(__file__).resolve().parents[2] / "scenes"
STARTER = (SCENES_DIR / "starter.py").read_text()
BRACKET = (SCENES_DIR / "bracket.py").read_text()

SMALL = """\
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude
from cadjoint.geometry import Scalar
from cadjoint.sdf.boolean import Union

depth = Scalar(0.5, free=True, name="depth")
sketch = PolygonProfile([[0, 0], [1, 0], [1, 1]], name="s")
body = extrude(sketch, depth=depth, material=None)
boss = PolygonProfile([[0, 0], [1, 0], [1, 1]], plane=SketchPlane.on(body.cap("+")), name="t")
block = Solid.box(size=[1, 1, 1], position=[2, 0, 0], name="block")
scene = Union(body, block, smoothness=0.05)
"""


def elements(source: str) -> dict[str, dict]:
    listed = build_element_payload(source)
    for entry in listed:
        ConstructionElement.model_validate(entry)
    return {entry["id"]: entry for entry in listed}


def argument(element: dict, name: str) -> dict:
    return next(item for item in element["arguments"] if item["name"] == name)


class TestWhatTheStarterDeclares:
    def test_every_construction_call_is_listed_in_source_order(self):
        listed = build_element_payload(STARTER)
        kinds = [entry["kind"] for entry in listed]
        assert kinds[:5] == ["sketch", "plane", "feature", "sketch", "plane"]
        assert kinds.count("boolean") == 2
        assert kinds.count("primitive") == 8
        # Source order, planes aside: a plane sits inside its sketch's statement.
        lines = [entry["line"] for entry in listed if entry["kind"] != "plane"]
        assert lines == sorted(lines)

    def test_a_sketch_and_its_plane_share_the_stable_identity_scheme(self):
        by_id = elements(STARTER)
        sketch = by_id["assign:comb_profile"]
        plane = by_id["plane:comb_profile"]
        assert sketch["stableId"] == "assign:comb_profile"
        assert plane["owner"] == "assign:comb_profile"
        assert plane["stableId"] == "plane:comb_profile"
        assert plane["call"] == "SketchPlane"
        assert plane["line"] == 151

    def test_a_stated_plane_is_editable_through_its_sketch(self):
        plane = elements(STARTER)["plane:comb_profile"]
        origin = argument(plane, "origin")
        normal = argument(plane, "normal")
        assert origin["kind"] == "vector" and origin["value"] == [0.0, 0.0, 0.0]
        assert normal["value"] == [0.0, 1.0, 0.0]
        assert origin["patch"] == {
            "op": "set_value",
            "name": "PolygonProfile",
            "argument": "planeOrigin",
            "line": 132,
            "id": "assign:comb_profile",
        }
        assert normal["patch"]["argument"] == "planeNormal"

    def test_a_parameter_backed_depth_names_its_parameter_and_stays_editable(self):
        sink = elements(STARTER)["assign:sink"]
        depth = argument(sink, "depth")
        assert depth["kind"] == "number"
        assert depth["value"] == pytest.approx(1.2)
        assert depth["parameter"] == "fin_depth"
        assert depth["patch"]["name"] == "extrude" and depth["patch"]["id"] == "assign:sink"

    def test_a_material_is_a_reference_with_an_assign_address(self):
        sink = elements(STARTER)["assign:sink"]
        material = argument(sink, "material")
        assert material["kind"] == "reference" and material["value"] == "aluminum"
        assert material["patch"]["op"] == "assign_material"
        assert material["patch"]["line"] == sink["line"]

    def test_a_primitive_lists_its_dimensions_and_placement(self):
        board = elements(STARTER)["assign:board"]
        assert [item["name"] for item in board["arguments"]] == [
            "size",
            "position",
            "material",
        ]
        assert argument(board, "size")["patch"]["argument"] == "size"
        assert board["name"] == "board" and board["variable"] == "board"

    def test_a_boolean_is_addressed_by_line_and_lists_its_operands(self):
        body = elements(STARTER)["boolean:thermal_body"]
        assert body["stableId"] is None
        assert argument(body, "operands")["value"] == "sink, slug, bush_a, bush_b"
        smoothness = argument(body, "smoothness")
        assert smoothness["value"] == pytest.approx(0.03)
        assert smoothness["patch"] == {
            "op": "set_value",
            "name": "Union",
            "argument": "smoothness",
            "line": body["line"],
            "id": None,
        }


class TestTheEdgeCasesTheWindowRenders:
    def test_a_sketch_without_a_plane_reports_the_default_world_plane(self):
        plane = elements(SMALL)["plane:sketch"]
        origin = argument(plane, "origin")
        assert origin["kind"] == "default" and origin["value"] == [0.0, 0.0, 0.0]
        assert argument(plane, "normal")["value"] == [0.0, 0.0, 1.0]
        # Still editable: set_value synthesises the SketchPlane keyword.
        assert origin["patch"]["argument"] == "planeOrigin"

    def test_a_derived_plane_is_shown_as_its_expression_and_not_editable(self):
        plane = elements(SMALL)["plane:boss"]
        assert plane["call"] == "SketchPlane.on"
        (reference,) = plane["arguments"]
        assert reference["kind"] == "expression"
        assert reference["text"] == 'SketchPlane.on(body.cap("+"))'
        assert reference["patch"] is None

    def test_a_none_material_has_no_assign_address(self):
        body = elements(SMALL)["assign:body"]
        material = argument(body, "material")
        assert material["kind"] == "expression" and material["patch"] is None

    def test_a_named_scalar_argument_resolves_to_its_literal(self):
        body = elements(SMALL)["assign:body"]
        depth = argument(body, "depth")
        assert depth["value"] == 0.5 and depth["parameter"] == "depth"
        assert depth["span"] is not None

    def test_an_unparseable_program_yields_nothing(self):
        assert build_element_payload("scene = Union(") == []

    def test_the_bracket_lists_its_difference(self):
        by_id = elements(BRACKET)
        assert by_id["boolean:bracket"]["call"] == "Difference"
        assert argument(by_id["boolean:bracket"], "operands")["value"] == (
            "body, hole_left, hole_right"
        )


class TestEveryPublishedAddressIsAccepted:
    @pytest.mark.parametrize(
        "source", [SMALL, STARTER, BRACKET], ids=["small", "starter", "bracket"]
    )
    def test_each_set_value_address_rewrites_the_program(self, source: str):
        for element in build_element_payload(source):
            for item in element["arguments"]:
                patch = item["patch"]
                if patch is None or patch["op"] != "set_value":
                    continue
                value = item["value"] if item["kind"] != "default" else item["value"]
                request = {
                    "source": source,
                    "op": "set_value",
                    "line": patch["line"],
                    "name": patch["name"],
                    "argument": patch["argument"],
                    "value": value,
                }
                if patch["id"] is not None:
                    request["id"] = patch["id"]
                result = patch_source(request)
                assert result["ok"] is True, (element["id"], item["name"], result.get("error"))

    def test_each_assign_address_is_accepted(self):
        for element in build_element_payload(STARTER):
            for item in element["arguments"]:
                patch = item["patch"]
                if patch is None or patch["op"] != "assign_material":
                    continue
                result = patch_source(
                    {
                        "source": STARTER,
                        "op": "assign_material",
                        "line": patch["line"],
                        "id": patch["id"],
                        "material": "steel",
                    }
                )
                assert result["ok"] is True, (element["id"], result.get("error"))

    def test_the_contract_table_is_what_marks_an_argument_editable(self):
        for element in build_element_payload(STARTER):
            for item in element["arguments"]:
                patch = item["patch"]
                if patch is None or patch["op"] != "set_value":
                    continue
                assert patch["argument"] in EDITABLE_CALLS[patch["name"]]


class TestBooleanSmoothnessThroughSetValue:
    def test_a_boolean_blend_is_rewritten_by_line(self):
        result = patch_source(
            {
                "source": SMALL,
                "op": "set_value",
                "line": 10,
                "name": "Union",
                "argument": "smoothness",
                "value": 0.2,
            }
        )
        assert result["ok"] is True
        assert "scene = Union(body, block, smoothness=0.2)" in result["source"]

    def test_a_negative_blend_is_refused(self):
        result = patch_source(
            {
                "source": SMALL,
                "op": "set_value",
                "line": 10,
                "name": "Difference",
                "argument": "smoothness",
                "value": -0.1,
            }
        )
        assert result["ok"] is False
        assert result["error"] == "`smoothness` needs a number of at least 0."

    def test_a_missing_blend_keyword_is_added(self):
        source = SMALL.replace("Union(body, block, smoothness=0.05)", "Union(body, block)")
        result = patch_source(
            {
                "source": source,
                "op": "set_value",
                "line": 10,
                "name": "Union",
                "argument": "smoothness",
                "value": 0.1,
            }
        )
        assert result["ok"] is True
        assert "Union(body, block, smoothness=0.1)" in result["source"]

    def test_a_draft_or_twist_is_shown_but_not_editable(self):
        # A drafted or twisted extrusion has no analytic faces, so rewriting
        # either would break every sketch planted on the feature's cap.
        source = SMALL.replace("material=None)", "material=None, draft=2.0)")
        body = elements(source)["assign:body"]
        draft = argument(body, "draft")
        assert draft["kind"] == "number" and draft["value"] == 2.0
        assert draft["patch"] is None


class TestThePayloadCarriesTheElements:
    def test_the_compile_payload_lists_them(self):
        payload = _compile_source(SMALL)
        assert CompilePayload.model_validate(payload).ok is True
        ids = [entry["id"] for entry in payload["elements"]]
        assert ids == [
            "assign:sketch",
            "plane:sketch",
            "assign:body",
            "assign:boss",
            "plane:boss",
            "assign:block",
            "boolean:scene",
        ]
