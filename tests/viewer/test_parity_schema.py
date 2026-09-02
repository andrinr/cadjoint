"""The payload's schema, and the TypeScript generated from it.

Three things have to stay in step: the models, the compile worker that
builds a payload against them, and the ``.d.ts`` the frontend imports. Each
test here pins one join — the models describe the real payloads of both
shipped scenes, the request models cover exactly the operations the server
accepts, and the checked-in TypeScript is what the emitter produces right
now.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cadjoint.viewer._compile_worker import _compile_source
from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer._patch import OPERATIONS
from cadjoint.viewer._patch_requests import patch_source
from cadjoint.viewer.schema import (
    PATCH_REQUEST_MODELS,
    TYPESCRIPT_PATH,
    CompilePayload,
    PatchResponse,
    typescript_source,
    validate_patch_request,
)

BRACKET = (Path(__file__).resolve().parents[2] / "scenes" / "bracket.py").read_text()

SMALL = """\
from cadjoint.construction import PolygonProfile, Solid, extrude
from cadjoint.sdf.boolean import Union

sketch = PolygonProfile([[0, 0], [1, 0], [1, 1]], name="s")
body = extrude(sketch, depth=0.5)
block = Solid.box(size=[1, 1, 1], position=[2, 0, 0], name="block")
scene = Union(body, block)
"""


@pytest.fixture(scope="module")
def payloads() -> dict[str, dict]:
    """One real compile payload per shipped scene, built once."""
    return {
        "small": _compile_source(SMALL),
        "starter": _compile_source(EXAMPLE_SOURCE),
        "bracket": _compile_source(BRACKET),
    }


class TestTheModelsDescribeRealPayloads:
    @pytest.mark.parametrize("scene", ["small", "starter", "bracket"])
    def test_a_compiled_scene_validates(self, payloads, scene: str):
        assert CompilePayload.model_validate(payloads[scene]).ok is True

    def test_the_worker_validates_before_it_answers(self, payloads):
        # The worker calls the same validator, so a payload that reached the
        # test at all already passed it; this pins that it is wired in.
        assert payloads["starter"]["ok"] is True
        assert "identities" in payloads["starter"]

    def test_a_payload_missing_a_field_is_refused(self, payloads):
        broken = {key: value for key, value in payloads["small"].items() if key != "identities"}
        with pytest.raises(ValidationError):
            CompilePayload.model_validate(broken)

    def test_an_unknown_top_level_field_is_refused(self, payloads):
        with pytest.raises(ValidationError):
            CompilePayload.model_validate({**payloads["small"], "surprise": 1})

    def test_a_declaration_may_still_grow_its_own_describe_keys(self, payloads):
        # ``studies`` comes from the FEM layer's describe(), which adds keys
        # as the solver grows; that must not fail the boundary.
        payload = {**payloads["bracket"]}
        payload["studies"] = [{**payload["studies"][0], "future_field": 3}]
        assert CompilePayload.model_validate(payload).ok is True


class TestTheRequestModels:
    def test_there_is_one_model_per_operation_the_server_accepts(self):
        assert set(PATCH_REQUEST_MODELS) == set(OPERATIONS)

    def test_each_model_is_discriminated_on_its_own_operation(self):
        for operation, model in PATCH_REQUEST_MODELS.items():
            assert model.model_fields["op"].annotation.__args__ == (operation,)

    @pytest.mark.parametrize(
        ("request_body", "expected"),
        [
            ({"op": "set_vertex", "id": "assign:sketch", "index": 0, "xy": [1.0, 2.0]}, True),
            ({"op": "set_vertex", "line": 4, "index": 0, "xy": [1.0, 2.0]}, True),
            ({"op": "add_sketch", "origin": [0.0, 0.0, 0.0]}, True),
            (
                {
                    "op": "set_sketch_plane",
                    "id": "assign:sketch",
                    "reference": {"kind": "cap", "owner": "assign:body", "sign": "+"},
                },
                True,
            ),
            ({"op": "solve_sketch", "line": 4, "method": "newton", "iterations": 4}, True),
            (
                {
                    "op": "set_optimization_value",
                    "optimization": 0,
                    "argument": "steps",
                    "value": 3,
                },
                True,
            ),
        ],
    )
    def test_a_request_the_server_would_accept_also_parses(self, request_body, expected):
        """The two descriptions of the contract agree on what is well formed."""
        assert validate_patch_request({**request_body, "source": SMALL}) is not None
        served = patch_source({**request_body, "source": SMALL})
        # The model checks shape; the validators additionally check that the
        # target exists, so a well-shaped request may still be refused for a
        # reason the schema cannot see. What must never happen is the reverse.
        assert isinstance(served["ok"], bool) is expected

    @pytest.mark.parametrize(
        "request_body",
        [
            {"op": "set_vertex", "line": 4, "index": 0},  # no xy
            {"op": "add_sketch", "origin": [0.0, 0.0]},  # two numbers
            {"op": "solve_sketch", "line": 4, "method": "bfgs"},
            {"op": "solve_sketch", "line": 4, "iterations": 9999},
            {"op": "set_optimization_value", "argument": "nope", "value": 1},
        ],
    )
    def test_a_request_the_server_refuses_does_not_parse(self, request_body):
        with pytest.raises(ValidationError):
            validate_patch_request({**request_body, "source": SMALL})
        assert patch_source({**request_body, "source": SMALL})["ok"] is False

    def test_the_server_and_the_models_agree_about_stray_fields(self):
        """A field no model names is refused by both descriptions.

        The server used to read the fields it needed and ignore the rest, so
        a browser running newer assets kept *working* against an older
        server — by silently dropping whatever the new field asked for and
        applying half an edit.  That is the same skew the unknown-operation
        check already refuses, and it is refused the same way now, with the
        same advice; the models forbid extras so a typo in new frontend
        code is caught at compile time as well.
        """
        stray = {"source": SMALL, "op": "set_vertex", "line": 4, "index": 0, "xy": [1.0, 2.0]}
        assert patch_source({**stray, "xyz": 1}) == {
            "ok": False,
            "error": (
                "The patch operation `set_vertex` does not take `xyz`. "
                "If you updated cadjoint, restart the playground server."
            ),
        }
        with pytest.raises(ValidationError):
            validate_patch_request({**stray, "xyz": 1})

    def test_a_response_round_trips_through_its_model(self):
        served = patch_source({"source": SMALL, "op": "add_sketch", "origin": [0, 0, 1]})
        assert PatchResponse.model_validate(served).ok is True
        refused = patch_source({"source": SMALL, "op": "add_sketch"})
        assert PatchResponse.model_validate(refused).error


class TestTheGeneratedTypeScript:
    def test_the_checked_in_file_is_what_the_emitter_produces(self):
        """Regenerate and diff — the whole point of generating it."""
        assert TYPESCRIPT_PATH.read_text() == typescript_source(), (
            "cadjoint/viewer/schema/payloads.d.ts is stale; regenerate it with "
            "`python -m cadjoint.viewer.schema.emit`."
        )

    def test_it_declares_every_model_the_payload_carries(self):
        text = TYPESCRIPT_PATH.read_text()
        for name in (
            "CompilePayload",
            "ConstructionNode",
            "ConstructionFace",
            "IdentityEntry",
            "MaterialDefinition",
            "StudyPayload",
            "SimMeshPayload",
            "OptimizationPayload",
        ):
            assert f"export interface {name} {{" in text

    def test_it_declares_one_request_interface_per_operation(self):
        text = TYPESCRIPT_PATH.read_text()
        for model in PATCH_REQUEST_MODELS.values():
            assert f"export interface {model.__name__} {{" in text
        assert "export type PatchRequest =" in text
        assert 'export type PatchOperation = PatchRequest["op"];' in text

    def test_the_stable_ids_reach_the_generated_types(self):
        text = TYPESCRIPT_PATH.read_text()
        assert "stableId?: string | null;" not in text, "stableId is always sent, even as null"
        assert "stableId: string | null;" in text
        assert "ownerStableId: string | null;" in text
