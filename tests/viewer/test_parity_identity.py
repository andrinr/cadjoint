"""Stable identities: what they name, and that a line move cannot break them.

The payload used to address everything by the line the last compile saw it
on, so any edit between a compile and a click could send a patch to the
wrong statement. These tests pin the replacement: the id scheme itself, the
proof that inserting lines above every declaration moves every line and no
id, and the resolution path that lets ``/patch`` take an id wherever it used
to take a line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer._patch_requests import patch_source
from cadjoint.viewer._source_map import (
    PLAYGROUND_FILENAME,
    build_construction_payload,
    build_identities,
    capture_profiles,
    describe_identities,
    identity_index,
)
from cadjoint.viewer._worker_declarations import (
    _mesh_entries,
    _optimization_entries,
    _study_entries,
)
from cadjoint.viewer._worker_scene import _execute_scene

BRACKET = (Path(__file__).resolve().parents[2] / "scenes" / "bracket.py").read_text()

SOURCE = """\
from cadjoint.constraints import DistanceConstraint, FixedConstraint
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude
from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy
from cadjoint.geometry import Scalar, Vector2
from cadjoint.optimize import Optimization
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union

width = Scalar(1.0, free=True, name="width")
steel = Material(name="steel", color=[0.5, 0.5, 0.5], roughness=0.4, metallic=0.8)
a = Vector2(value=[0.0, 0.0], free=True, name="a")
b = Vector2(value=[1.0, 0.0], free=True, name="b")
c = Vector2(value=[1.0, 1.0], free=True, name="c")
profile = PolygonProfile([a, b, c], name="p")
FixedConstraint(a, [0.0, 0.0])
DistanceConstraint(a, b, width)
body = extrude(profile, depth=width, material=steel)
block = Solid.box(size=[0.4, 0.4, 0.4], position=[2.0, 0.0, 0.0], name="block")
scene = Union(body, block)
grid = SimMesh(name="grid", resolution=(4, 4, 4), bounds=(-1.0, -1.0, -1.0), size=(2.0, 2.0, 2.0))
study = ThermalStudy(
    name="sink-conduction",
    conductivity=1.0,
    bcs=[Dirichlet(Nodes.sphere([0, 0, 0], 1.0), 0.0), HeatFlux(Nodes.side("+z"), 5.0)],
    mesh=grid,
)
opt = Optimization(
    name="o", study="sink-conduction", metric="max", steps=3, learning_rate=0.01
)
"""


def namespace_of(source: str) -> dict:
    """Run a program the way the compile worker does, capture registries and all."""
    return _execute_scene(
        source, capture=(("__profiles__", lambda: capture_profiles(PLAYGROUND_FILENAME)),)
    )


def payload_of(source: str) -> list[dict]:
    """Build the construction payload the way the compile worker does."""
    return build_construction_payload(namespace_of(source)["__profiles__"], source)


def ids(source: str) -> set[str]:
    return set(identity_index(source))


class TestTheIdScheme:
    def test_names_a_bound_call_after_the_variable_it_is_assigned_to(self):
        index = identity_index(SOURCE)
        assert index["assign:profile"].kind == "sketch"
        assert index["assign:body"].kind == "feature"
        assert index["assign:block"].kind == "primitive"
        assert index["assign:steel"].kind == "material"
        assert index["assign:study"].kind == "study"
        assert index["assign:grid"].kind == "mesh"
        assert index["assign:opt"].kind == "optimization"

    def test_names_an_ordinal_inside_its_owner_for_children(self):
        index = identity_index(SOURCE)
        # The prompt's shape: the study's own name, then the position.
        assert index["bc:study[0]"].kind == "bc"
        assert index["bc:study[1]"].index == 1
        assert index["constraint:profile[0]"].call == "FixedConstraint"
        assert index["constraint:profile[1]"].call == "DistanceConstraint"
        assert [index[f"vertex:profile[{i}]"].index for i in range(3)] == [0, 1, 2]
        assert index["plane:profile"].kind == "plane"

    def test_names_an_unbound_feature_after_the_sketch_it_consumes(self):
        source = SOURCE.replace(
            "body = extrude(profile, depth=width, material=steel)\n", ""
        ).replace("Union(body, block)", "Union(extrude(profile, depth=width), block)")
        assert "call:extrude@profile" in ids(source)

    def test_names_an_unbound_construction_call_after_its_name_argument(self):
        source = "from cadjoint.construction import PolygonProfile\nPolygonProfile([[0, 0]], name='rim')\n"
        assert "sketch:rim" in ids(source)

    def test_falls_back_to_an_ordinal_when_there_is_no_name_at_all(self):
        source = "from cadjoint.construction import PolygonProfile\nPolygonProfile([[0, 0]])\n"
        assert "sketch:#0" in ids(source)

    def test_a_duplicate_id_resolves_to_nothing_rather_than_to_a_guess(self):
        source = SOURCE + "block = Solid.box(size=[1, 1, 1], position=[0, 0, 0])\n"
        assert "assign:block" not in ids(source)

    def test_unparseable_source_declares_no_identities(self):
        assert build_identities("def (") == []

    def test_a_node_selection_is_not_mistaken_for_a_solid(self):
        # ``Nodes.sphere(...)`` shares a name with a construction primitive.
        assert not [item for item in build_identities(SOURCE) if item.token.startswith("#")]


class TestStabilityUnderLineMoves:
    """Insert lines above every declaration; every line moves, no id does."""

    @pytest.mark.parametrize(
        "source", [SOURCE, EXAMPLE_SOURCE, BRACKET], ids=["small", "starter", "bracket"]
    )
    def test_every_id_survives_a_line_inserted_above_every_statement(self, source: str):
        before = {item.id: item.line for item in build_identities(source)}
        # A comment above every line: nothing an id is derived from changes,
        # and every line number does.
        spaced = "".join(f"# spacer\n{line}\n" for line in source.splitlines())
        after = {item.id: item.line for item in build_identities(spaced)}

        assert set(before) == set(after)
        assert before, "the fixture should declare something to move"
        moved = [key for key in before if before[key] != after[key]]
        assert moved == list(before), "every line should have moved"

    def test_an_added_sketch_does_not_renumber_the_ids_below_it(self):
        patched = patch_source({"source": SOURCE, "op": "add_sketch", "origin": [0.0, 0.0, 1.0]})
        assert patched["ok"] is True
        # The new sketch is the only new id; every old one still resolves.
        assert ids(SOURCE) <= ids(patched["source"])
        before = identity_index(SOURCE)["assign:study"].line
        after = identity_index(patched["source"])["assign:study"].line
        assert after != before

    def test_a_stale_line_and_a_stable_id_disagree_after_an_edit(self):
        """The whole point: the id still hits, the remembered line does not."""
        stale_line = identity_index(SOURCE)["assign:profile"].line
        # Two comments typed at the top of the file — the cheapest edit there
        # is, and enough to make every remembered line wrong.
        grown = "# a note\n# and another\n" + SOURCE
        by_line = patch_source(
            {"source": grown, "op": "set_vertex", "line": stale_line, "index": 0, "xy": [9.0, 9.0]}
        )
        by_id = patch_source(
            {"source": grown, "op": "set_vertex", "id": "vertex:profile[0]", "xy": [9.0, 9.0]}
        )
        assert by_id["ok"] is True
        assert "a = Vector2(value=[9, 9]" in by_id["source"]
        # The remembered line now points two statements too high.
        assert by_line["ok"] is False


class TestThePayloadPublishesThem:
    def test_every_construction_node_carries_its_stable_id(self):
        nodes = payload_of(SOURCE)
        assert [node["stableId"] for node in nodes] == ["assign:profile", "assign:block"]

    def test_a_sketch_stamps_its_plane_vertices_and_constraints(self):
        sketch = payload_of(SOURCE)[0]
        assert sketch["plane"]["stableId"] == "plane:profile"
        assert [item["stableId"] for item in sketch["vertices"]] == [
            f"vertex:profile[{i}]" for i in range(3)
        ]
        assert [item["stableId"] for item in sketch["constraints"]] == [
            "constraint:profile[0]",
            "constraint:profile[1]",
        ]

    def test_a_face_carries_its_own_id_and_its_owners(self):
        faces = {face["key"]: face for face in payload_of(SOURCE)[0]["faces"]}
        assert faces["cap+"]["stableId"] == "face:body:cap+"
        assert faces["cap+"]["ownerStableId"] == "assign:body"

    def test_declarations_carry_theirs(self):
        namespace = namespace_of(SOURCE)
        meshes = _mesh_entries(namespace["__sim_meshes__"], SOURCE)
        studies = _study_entries(namespace["__studies__"], SOURCE)
        optimizations = _optimization_entries(namespace["__optimizations__"], SOURCE)
        assert [entry["stableId"] for entry in meshes] == ["assign:grid"]
        assert [entry["stableId"] for entry in optimizations] == ["assign:opt"]
        assert [entry["stableId"] for entry in studies] == ["assign:study"]
        assert [bc["stableId"] for bc in studies[0]["bcs"]] == ["bc:study[0]", "bc:study[1]"]

    def test_the_whole_table_is_published_for_the_lines_nothing_stamps(self):
        table = {entry["id"]: entry for entry in describe_identities(SOURCE)}
        operator = payload_of(SOURCE)[0]["operators"][0]
        named = [entry for entry in table.values() if entry["line"] == operator["line"]]
        assert [entry["id"] for entry in named] == ["assign:body"]


class TestPatchRequestsResolveThem:
    def test_a_vertex_id_supplies_both_the_line_and_the_index(self):
        result = patch_source(
            {"source": SOURCE, "op": "set_vertex", "id": "vertex:profile[1]", "xy": [3.5, 4.25]}
        )
        assert result["ok"] is True
        assert "b = Vector2(value=[3.5, 4.25]" in result["source"]

    def test_a_constraint_id_supplies_the_sketch_and_the_ordinal(self):
        result = patch_source(
            {"source": SOURCE, "op": "delete_constraint", "id": "constraint:profile[0]"}
        )
        assert result["ok"] is True
        assert "FixedConstraint(a, [0.0, 0.0])" not in result["source"]

    def test_a_bc_id_supplies_the_study_and_the_position(self):
        result = patch_source({"source": SOURCE, "op": "delete_study_bc", "id": "bc:study[1]"})
        assert result["ok"] is True
        assert "HeatFlux(Nodes.side" not in result["source"]

    def test_a_declaration_id_stands_in_for_its_index(self):
        result = patch_source(
            {
                "source": SOURCE,
                "op": "set_optimization_value",
                "id": "assign:opt",
                "argument": "steps",
                "value": 9,
            }
        )
        assert result["ok"] is True
        assert "steps=9" in result["source"]

    def test_a_loft_names_its_two_sketches_by_id(self):
        source = SOURCE.replace(
            "body = extrude(profile, depth=width, material=steel)\n",
            "other = PolygonProfile([[0, 0], [1, 0], [1, 1]], name='q')\n",
        ).replace("Union(body, block)", "Union(block)")
        result = patch_source(
            {"source": source, "op": "add_loft", "id_a": "assign:profile", "id_b": "assign:other"}
        )
        assert result["ok"] is True
        assert "loft(profile, other" in result["source"]

    def test_a_face_reference_names_its_owner_by_id(self):
        # A face can only be planted on a feature declared above the sketch.
        source = SOURCE.replace(
            "scene = Union(body, block)",
            "boss = PolygonProfile([[0, 0], [1, 0], [1, 1]], name='boss')\n"
            "scene = Union(body, block)",
        )
        result = patch_source(
            {
                "source": source,
                "op": "set_sketch_plane",
                "id": "assign:boss",
                "reference": {"kind": "cap", "owner": "assign:body", "sign": "+"},
            }
        )
        assert result["ok"] is True
        assert "SketchPlane.on(body.cap" in result["source"]

    def test_the_legacy_line_form_still_works_unchanged(self):
        line = identity_index(SOURCE)["assign:profile"].line
        by_line = patch_source(
            {"source": SOURCE, "op": "set_vertex", "line": line, "index": 1, "xy": [3.5, 4.25]}
        )
        by_id = patch_source(
            {"source": SOURCE, "op": "set_vertex", "id": "vertex:profile[1]", "xy": [3.5, 4.25]}
        )
        assert by_line == by_id


class TestRejectedIds:
    def test_an_id_that_names_nothing_says_so(self):
        result = patch_source({"source": SOURCE, "op": "set_vertex", "id": "assign:nope"})
        assert result == {
            "ok": False,
            "error": "No statement in this program has the id 'assign:nope'.",
        }

    def test_an_id_of_the_wrong_kind_names_the_kind_and_the_operation(self):
        result = patch_source({"source": SOURCE, "op": "delete_study", "id": "vertex:profile[0]"})
        assert result == {
            "ok": False,
            "error": (
                "The id 'vertex:profile[0]' names a vertex, which `delete_study` cannot address."
            ),
        }

    def test_an_operation_that_creates_something_takes_no_id(self):
        result = patch_source(
            {"source": SOURCE, "op": "add_sketch", "id": "assign:profile", "origin": [0, 0, 0]}
        )
        assert result == {
            "ok": False,
            "error": "The patch operation `add_sketch` creates a new object, so it takes no `id`.",
        }

    def test_a_loft_says_which_two_fields_it_wants(self):
        result = patch_source({"source": SOURCE, "op": "add_loft", "id": "assign:profile"})
        assert result == {
            "ok": False,
            "error": "`add_loft` names its two sketches with `id_a` and `id_b`, not `id`.",
        }

    def test_an_id_must_be_a_non_empty_string(self):
        result = patch_source({"source": SOURCE, "op": "set_vertex", "id": 7})
        assert result == {
            "ok": False,
            "error": "The patch request needs `id` as a non-empty string.",
        }
