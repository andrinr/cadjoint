"""The round-trip invariant: what a patch says it did, the payload shows.

Every viewer action is an AST-span rewrite, so ``patch`` and ``compile`` are
two halves of one claim — the user dragged a vertex *there*, set that value,
added that boundary condition. Until now nothing checked the two halves
against each other: the patch tests read the source text, the payload tests
read a payload, and no test made a patch and then looked.

This module closes that loop for all 27 operations. For each generated
request it asserts, in order:

1. the patched text is still valid Python;
2. patching twice from the same input is byte-identical (no hidden state);
3. addressing by stable id and by the legacy line produce the same text;
4. compiling the patched program shows exactly the change that was asked
   for — the vertex moved, the value set, the declaration added or gone;
5. operations that describe a *state* rather than a step are idempotent:
   applying one to its own output changes nothing further;
6. no operation loses the identity of anything it did not delete.

The generator is seeded rather than random, so a failure is reproducible
and the parametrized ids are stable. ``hypothesis`` is not in the
environment; a seeded generator over a table of shapes covers the same
ground for a contract whose whole input space is enumerable.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer._patch import OPERATIONS
from cadjoint.viewer._patch_requests import _ID_TARGETS, patch_source
from cadjoint.viewer._source_map import (
    PLAYGROUND_FILENAME,
    build_construction_payload,
    build_material_payload,
    capture_profiles,
    identity_index,
)
from cadjoint.viewer._worker_declarations import (
    _mesh_entries,
    _optimization_entries,
    _study_entries,
)
from cadjoint.viewer._worker_scene import _execute_scene

SEED = 20260902

BRACKET = (Path(__file__).resolve().parents[2] / "scenes" / "bracket.py").read_text()

SCENE = """\
from cadjoint.constraints import DistanceConstraint, FixedConstraint
from cadjoint.construction import PolygonProfile, Solid, extrude
from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy
from cadjoint.geometry import Scalar
from cadjoint.optimize import Optimization
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union

depth = Scalar(0.6, free=True, name="depth")
steel = Material(name="steel", color=[0.5, 0.5, 0.5], roughness=0.4, metallic=0.8)
brass = Material(name="brass", color=[0.7, 0.6, 0.2], roughness=0.3, metallic=0.9)
plate = PolygonProfile([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], name="plate")
FixedConstraint(plate.vertices[0], [0.0, 0.0])
DistanceConstraint(plate.vertices[0], plate.vertices[1], 1.0)
slab = extrude(plate, depth=depth, material=steel)
post = PolygonProfile([[0.0, 0.0], [0.4, 0.0], [0.4, 0.4]], name="post")
rim = PolygonProfile([[0.0, 0.0], [0.3, 0.0], [0.3, 0.3]], name="rim")
block = Solid.box(size=[0.4, 0.4, 0.4], position=[2.0, 0.0, 0.0], name="block")
scene = Union(slab, block)
grid = SimMesh(name="grid", resolution=(6, 6, 6), bounds=(-1.0, -1.0, -1.0), size=(3.0, 3.0, 3.0))
spare = SimMesh(name="spare", resolution=(4, 4, 4), bounds=(-1.0, -1.0, -1.0), size=(2.0, 2.0, 2.0))
heat = ThermalStudy(
    name="heat",
    conductivity=45.0,
    bcs=[Dirichlet(Nodes.side("-z"), 300.0), HeatFlux(Nodes.side("+z"), 500.0)],
    mesh=grid,
)
idle = ThermalStudy(
    name="idle", resolution=8, conductivity=12.0, bcs=[Dirichlet(Nodes.side("-x"), 290.0)]
)
tune = Optimization(name="tune", study="heat", metric="max", steps=3, learning_rate=0.01)
"""


# ── The payload half of the round trip ──────────────────────────────────────


def compile_payload(source: str) -> dict[str, Any]:
    """What the compile worker would report for this program.

    The shader half of a real compile is expensive and irrelevant here, so
    this runs the same execution and the same payload builders and stops
    there.
    """
    namespace = _execute_scene(
        source, capture=(("__profiles__", lambda: capture_profiles(PLAYGROUND_FILENAME)),)
    )
    return {
        "construction": build_construction_payload(namespace["__profiles__"], source),
        "materials": build_material_payload(namespace, source),
        "studies": _study_entries(namespace["__studies__"], source),
        "meshes": _mesh_entries(namespace["__sim_meshes__"], source),
        "optimizations": _optimization_entries(
            namespace["__optimizations__"], source, namespace["scene"]
        ),
    }


def node(payload: dict, stable_id: str) -> dict:
    """The one construction node with this stable id."""
    matches = [item for item in payload["construction"] if item["stableId"] == stable_id]
    assert len(matches) == 1, f"{stable_id} names {len(matches)} nodes"
    return matches[0]


def named(entries: list[dict], stable_id: str) -> dict:
    matches = [item for item in entries if item.get("stableId") == stable_id]
    assert len(matches) == 1, f"{stable_id} names {len(matches)} entries"
    return matches[0]


def counts(payload: dict) -> dict[str, int]:
    return {key: len(value) for key, value in payload.items()}


# ── The cases ───────────────────────────────────────────────────────────────


@dataclass
class Case:
    """One generated request and what the payload must say afterwards."""

    name: str
    request: dict[str, Any]
    expect: Callable[[dict, dict], None]
    idempotent: bool = False
    legacy: dict[str, Any] | None = None
    """The same edit addressed the old way, for the id/line parity check."""
    removes: tuple[str, ...] = field(default_factory=tuple)
    """Stable ids this operation is expected to take out of the program."""

    @property
    def op(self) -> str:
        return self.request["op"]


def _grew(key: str, by: int = 1) -> Callable[[dict, dict], None]:
    """The payload gained (or lost) exactly *by* entries under *key*."""

    def check(before: dict, after: dict) -> None:
        assert counts(after)[key] == counts(before)[key] + by

    return check


def _cases() -> list[Case]:
    """Every operation, exercised against :data:`SCENE` with seeded values."""
    rng = random.Random(SEED)
    index = identity_index(SCENE)
    plate, post, rim = index["assign:plate"], index["assign:post"], index["assign:rim"]
    block, heat = index["assign:block"], index["assign:heat"]
    grid, spare, tune = index["assign:grid"], index["assign:spare"], index["assign:tune"]

    def number() -> float:
        # Viewer-generated coordinates are written with ``%.4g``, so a
        # generated value has to be one that survives four significant
        # digits — otherwise the test would be pinning the formatter, not
        # the round trip.
        return round(rng.uniform(-2.0, 2.0), 3)

    def point(n: int = 3) -> list[float]:
        return [number() for _ in range(n)]

    cases: list[Case] = []

    # ── sketch vertices ────────────────────────────────────────────────────
    for vertex in (0, 2):
        xy = point(2)
        cases.append(
            Case(
                name=f"set_vertex[{vertex}]",
                request={"op": "set_vertex", "id": f"vertex:plate[{vertex}]", "xy": xy},
                legacy={"op": "set_vertex", "line": plate.line, "index": vertex, "xy": xy},
                expect=lambda before, after, i=vertex, xy=xy: (
                    _approx(node(after, "assign:plate")["vertices"][i]["uv"], xy)
                ),
                idempotent=True,
            )
        )
    xy = point(2)
    cases.append(
        Case(
            name="insert_vertex",
            request={"op": "insert_vertex", "id": "vertex:plate[1]", "xy": xy},
            legacy={"op": "insert_vertex", "line": plate.line, "index": 1, "xy": xy},
            expect=lambda before, after, xy=xy: (
                len(node(after, "assign:plate")["vertices"])
                == len(node(before, "assign:plate")["vertices"]) + 1
                # ``insert_vertex`` inserts *before* the index it is given.
                and _approx(node(after, "assign:plate")["vertices"][1]["uv"], xy)
            )
            or _fail("the inserted vertex is not where it was asked for"),
        )
    )
    cases.append(
        Case(
            name="delete_vertex",
            request={"op": "delete_vertex", "id": "vertex:plate[3]"},
            legacy={"op": "delete_vertex", "line": plate.line, "index": 3},
            expect=lambda before, after: (
                len(node(after, "assign:plate")["vertices"])
                == len(node(before, "assign:plate")["vertices"]) - 1
            )
            or _fail("the vertex was not removed"),
            removes=("vertex:plate[3]",),
        )
    )

    # ── values on construction calls ───────────────────────────────────────
    size = [abs(number()) + 0.1 for _ in range(3)]
    cases.append(
        Case(
            name="set_value[box.size]",
            request={
                "op": "set_value",
                "id": "assign:block",
                "name": "box",
                "argument": "size",
                "value": size,
            },
            legacy={
                "op": "set_value",
                "line": block.line,
                "name": "box",
                "argument": "size",
                "value": size,
            },
            expect=lambda before, after, size=size: _approx(
                node(after, "assign:block")["transform"]["dimensions"]["size"], size
            ),
            idempotent=True,
        )
    )
    origin = point()
    cases.append(
        Case(
            name="set_value[planeOrigin]",
            request={
                "op": "set_value",
                "id": "assign:post",
                "name": "PolygonProfile",
                "argument": "planeOrigin",
                "value": origin,
            },
            legacy={
                "op": "set_value",
                "line": post.line,
                "name": "PolygonProfile",
                "argument": "planeOrigin",
                "value": origin,
            },
            expect=lambda before, after, origin=origin: _approx(
                node(after, "assign:post")["plane"]["origin"], origin
            ),
            idempotent=True,
        )
    )

    # ── creating construction objects ──────────────────────────────────────
    for kind, dimensions in (
        ("box", {"size": [0.3, 0.3, 0.3]}),
        ("sphere", {"radius": 0.25}),
        ("cylinder", {"radius": 0.2, "height": 0.5}),
    ):
        cases.append(
            Case(
                name=f"add_primitive[{kind}]",
                request={
                    "op": "add_primitive",
                    "kind": kind,
                    "position": point(),
                    "dimensions": dimensions,
                },
                expect=_grew("construction"),
            )
        )
    cases.append(
        Case(
            name="add_sketch",
            request={"op": "add_sketch", "origin": point()},
            expect=_grew("construction"),
        )
    )
    cases.append(
        Case(
            name="add_material",
            request={"op": "add_material", "color": [0.2, 0.4, 0.6], "roughness": 0.5},
            expect=_grew("materials"),
        )
    )
    cases.append(
        Case(
            name="assign_material",
            request={"op": "assign_material", "id": "assign:block", "material": "brass"},
            legacy={"op": "assign_material", "line": block.line, "material": "brass"},
            expect=lambda before, after: node(after, "assign:block")["material"] == "brass"
            or _fail("the material was not assigned"),
            idempotent=True,
        )
    )
    cases.append(
        Case(
            name="delete_object",
            request={"op": "delete_object", "id": "assign:block"},
            legacy={"op": "delete_object", "line": block.line},
            expect=_grew("construction", -1),
            removes=("assign:block",),
        )
    )

    # ── operators and planes ───────────────────────────────────────────────
    cases.append(
        Case(
            name="add_extrusion",
            request={"op": "add_extrusion", "id": "assign:post", "depth": 0.35},
            legacy={"op": "add_extrusion", "line": post.line, "depth": 0.35},
            expect=lambda before, after: (
                [item["kind"] for item in node(after, "assign:post")["operators"]] == ["extrude"]
            )
            or _fail("no extrusion appeared on the sketch"),
        )
    )
    cases.append(
        Case(
            name="add_revolution",
            request={"op": "add_revolution", "id": "assign:post", "offset": 0.1},
            legacy={"op": "add_revolution", "line": post.line, "offset": 0.1},
            expect=lambda before, after: (
                [item["kind"] for item in node(after, "assign:post")["operators"]] == ["revolve"]
            )
            or _fail("no revolution appeared on the sketch"),
        )
    )
    cases.append(
        Case(
            name="add_loft",
            request={"op": "add_loft", "id_a": "assign:post", "id_b": "assign:rim"},
            legacy={"op": "add_loft", "line_a": post.line, "line_b": rim.line},
            expect=lambda before, after: (
                [item["kind"] for item in node(after, "assign:post")["operators"]] == ["loft"]
            )
            or _fail("no loft appeared on the sketch"),
        )
    )
    cases.append(
        Case(
            name="set_sketch_plane",
            request={
                "op": "set_sketch_plane",
                "id": "assign:post",
                "reference": {"kind": "cap", "owner": "assign:slab", "sign": "+"},
            },
            legacy={
                "op": "set_sketch_plane",
                "line": post.line,
                "reference": {"kind": "cap", "owner": index["assign:slab"].line, "sign": "+"},
            },
            expect=lambda before, after: (
                node(after, "assign:post")["plane"]["reference"]["constructor"] == "on"
                and node(after, "assign:post")["plane"]["reference"]["owner"] == "slab"
            )
            or _fail("the sketch was not planted on the face"),
            idempotent=True,
        )
    )

    # ── constraints ────────────────────────────────────────────────────────
    for kind, indices, value in (
        ("horizontal", [1, 2], None),
        ("coincident", [0, 2], None),
        ("fixed", [2], [0.5, 0.5]),
        ("distance", [1, 2], 0.75),
        ("parallel", [0, 1, 2, 3], None),
    ):
        body = {"op": "add_constraint", "id": "assign:plate", "kind": kind, "indices": indices}
        if value is not None:
            body["value"] = value
        cases.append(
            Case(
                name=f"add_constraint[{kind}]",
                request=body,
                legacy={**_without_id(body), "line": plate.line},
                expect=lambda before, after, kind=kind, indices=indices: (
                    node(after, "assign:plate")["constraints"][-1]["kind"] == kind
                    and node(after, "assign:plate")["constraints"][-1]["vertices"] == indices
                )
                or _fail("the constraint the payload reports is not the one asked for"),
            )
        )
    cases.append(
        Case(
            name="delete_constraint",
            request={"op": "delete_constraint", "id": "constraint:plate[0]"},
            legacy={"op": "delete_constraint", "line": plate.line, "index": 0},
            expect=lambda before, after: (
                len(node(after, "assign:plate")["constraints"])
                == len(node(before, "assign:plate")["constraints"]) - 1
            )
            or _fail("the constraint was not removed"),
            removes=("constraint:plate[1]",),
        )
    )
    distance = round(abs(number()) + 0.2, 4)
    cases.append(
        Case(
            name="set_constraint_value",
            request={
                "op": "set_constraint_value",
                "id": "constraint:plate[1]",
                "value": distance,
            },
            legacy={
                "op": "set_constraint_value",
                "line": plate.line,
                "index": 1,
                "value": distance,
            },
            expect=lambda before, after, value=distance: _approx(
                node(after, "assign:plate")["constraints"][1]["value"], value
            ),
            idempotent=True,
        )
    )
    cases.append(
        Case(
            name="solve_sketch",
            request={
                "op": "solve_sketch",
                "id": "assign:plate",
                "method": "newton",
                "iterations": 4,
            },
            legacy={
                "op": "solve_sketch",
                "line": plate.line,
                "method": "newton",
                "iterations": 4,
            },
            # ``solve_sketch`` rewrites the sketch's one ``satisfy_constraints``
            # call rather than stacking another, so it sets state too.  The
            # payload check is that the solve *ran* — reaching it at all means
            # the patched program executed — and left the model it solved
            # intact rather than moving pinned geometry.
            expect=lambda before, after: (
                [item["kind"] for item in node(after, "assign:plate")["constraints"]]
                == [item["kind"] for item in node(before, "assign:plate")["constraints"]]
                and _approx(node(after, "assign:plate")["vertices"][0]["uv"], [0.0, 0.0])
            )
            or _fail("solving the sketch disturbed the constraints it solved"),
            idempotent=True,
        )
    )

    # ── studies ────────────────────────────────────────────────────────────
    cases.append(
        Case(
            name="add_study",
            request={"op": "add_study", "kind": "elastic", "name": "pull"},
            expect=_grew("studies"),
        )
    )
    idle = index["assign:idle"]
    cases.append(
        Case(
            name="delete_study",
            request={"op": "delete_study", "id": "assign:idle"},
            legacy={"op": "delete_study", "study": idle.index},
            expect=_grew("studies", -1),
            removes=("assign:idle", "bc:idle[0]"),
        )
    )
    for bc_type, value in (
        ("dirichlet", 275.0),
        ("heat_flux", 120.0),
    ):
        body = {
            "op": "add_study_bc",
            "id": "assign:heat",
            "bc_type": bc_type,
            "selection": {"kind": "side", "side": "+x", "tol": None},
            "value": value,
        }
        cases.append(
            Case(
                name=f"add_study_bc[{bc_type}]",
                request=body,
                legacy={**_without_id(body), "study": heat.index},
                expect=lambda before, after, bc_type=bc_type: (
                    named(after["studies"], "assign:heat")["bcs"][-1]["type"] == bc_type
                    and len(named(after["studies"], "assign:heat")["bcs"])
                    == len(named(before["studies"], "assign:heat")["bcs"]) + 1
                )
                or _fail("the boundary condition the payload reports is not the one added"),
            )
        )
    cases.append(
        Case(
            name="delete_study_bc",
            request={"op": "delete_study_bc", "id": "bc:heat[1]"},
            legacy={"op": "delete_study_bc", "study": heat.index, "bc": 1},
            expect=lambda before, after: (
                len(named(after["studies"], "assign:heat")["bcs"])
                == len(named(before["studies"], "assign:heat")["bcs"]) - 1
            )
            or _fail("the boundary condition was not removed"),
            removes=("bc:heat[1]",),
        )
    )
    temperature = round(abs(number()) * 100 + 10, 4)
    cases.append(
        Case(
            name="set_study_value[bc]",
            request={"op": "set_study_value", "id": "bc:heat[0]", "value": temperature},
            legacy={"op": "set_study_value", "study": heat.index, "bc": 0, "value": temperature},
            expect=lambda before, after, value=temperature: _approx(
                named(after["studies"], "assign:heat")["bcs"][0]["value"], value
            ),
            idempotent=True,
        )
    )
    conductivity = round(abs(number()) * 50 + 1, 4)
    cases.append(
        Case(
            name="set_study_value[argument]",
            request={
                "op": "set_study_value",
                "id": "assign:heat",
                "argument": "conductivity",
                "value": conductivity,
            },
            legacy={
                "op": "set_study_value",
                "study": heat.index,
                "argument": "conductivity",
                "value": conductivity,
            },
            expect=lambda before, after, value=conductivity: _approx(
                named(after["studies"], "assign:heat")["material"]["conductivity"], value
            ),
            idempotent=True,
        )
    )

    # ── meshes ─────────────────────────────────────────────────────────────
    cases.append(
        Case(
            name="add_mesh",
            request={"op": "add_mesh", "name": "coarse"},
            expect=_grew("meshes"),
        )
    )
    cases.append(
        Case(
            name="delete_mesh",
            request={"op": "delete_mesh", "id": "assign:spare"},
            legacy={"op": "delete_mesh", "mesh": spare.index},
            expect=_grew("meshes", -1),
            removes=("assign:spare",),
        )
    )
    for argument, value in (("resolution", [8, 8, 8]), ("padding", 0.15), ("method", "tet4")):
        body = {"op": "set_mesh_value", "id": "assign:grid", "argument": argument, "value": value}
        cases.append(
            Case(
                name=f"set_mesh_value[{argument}]",
                request=body,
                legacy={**_without_id(body), "mesh": grid.index},
                expect=lambda before, after, argument=argument, value=value: _approx(
                    named(after["meshes"], "assign:grid")[argument], value
                ),
                idempotent=True,
            )
        )

    # ── optimizations ──────────────────────────────────────────────────────
    for argument, value in (("steps", 12), ("learning_rate", 0.05)):
        body = {
            "op": "set_optimization_value",
            "id": "assign:tune",
            "argument": argument,
            "value": value,
        }
        cases.append(
            Case(
                name=f"set_optimization_value[{argument}]",
                request=body,
                legacy={**_without_id(body), "optimization": tune.index},
                expect=lambda before, after, argument=argument, value=value: _approx(
                    named(after["optimizations"], "assign:tune")[argument], value
                ),
                idempotent=True,
            )
        )
    cases.append(
        Case(
            name="delete_optimization",
            request={"op": "delete_optimization", "id": "assign:tune"},
            legacy={"op": "delete_optimization", "optimization": tune.index},
            expect=_grew("optimizations", -1),
            removes=("assign:tune",),
        )
    )
    return cases


def _without_id(request: dict[str, Any]) -> dict[str, Any]:
    """The same request with its stable id dropped, for the legacy form."""
    return {key: value for key, value in request.items() if key != "id"}


def _fail(message: str) -> bool:
    raise AssertionError(message)


def _approx(actual: Any, expected: Any) -> bool:
    """Compare a payload value to what the patch asked for."""
    if isinstance(expected, str):
        assert actual == expected, f"{actual!r} != {expected!r}"
        return True
    # Payload numbers come back through float32 parameters.
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), f"{actual!r} != {expected!r}"
    return True


CASES = _cases()


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return compile_payload(SCENE)


class TestCoverage:
    def test_every_operation_the_server_accepts_is_exercised(self):
        assert {case.op for case in CASES} == set(OPERATIONS)

    def test_the_scene_itself_compiles(self, baseline):
        assert counts(baseline) == {
            "construction": 4,
            "materials": 2,
            "studies": 2,
            "meshes": 2,
            "optimizations": 1,
        }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
class TestTheRoundTripInvariant:
    def test_the_patch_applies_and_still_parses(self, case: Case):
        result = patch_source({"source": SCENE, **case.request})
        assert result["ok"] is True, result.get("error")
        ast.parse(result["source"])

    def test_patching_twice_from_the_same_text_is_byte_identical(self, case: Case):
        first = patch_source({"source": SCENE, **case.request})
        second = patch_source({"source": SCENE, **case.request})
        assert first == second

    def test_the_stable_id_and_the_legacy_line_agree(self, case: Case):
        if case.legacy is None:
            # Nothing to compare — and the server agrees this operation has
            # no existing target, which is why it refuses an `id` at all.
            assert not _ID_TARGETS[case.op]
            return
        by_id = patch_source({"source": SCENE, **case.request})
        by_line = patch_source({"source": SCENE, **case.legacy})
        assert by_id == by_line

    def test_the_payload_shows_exactly_what_was_patched(self, case: Case, baseline):
        patched = patch_source({"source": SCENE, **case.request})["source"]
        case.expect(baseline, compile_payload(patched))

    def test_repeating_it_does_what_repeating_it_should(self, case: Case):
        """Setting a value twice changes nothing; adding twice does not vanish.

        Both halves matter. An operation that sets state has to be safe to
        replay — a drag sends the same edit many times — and an operation
        that adds or removes has to actually happen again, or be refused,
        rather than silently doing nothing.
        """
        once = patch_source({"source": SCENE, **case.request})["source"]
        twice = patch_source({"source": once, **case.request})
        if case.idempotent:
            assert twice["ok"] is True, twice.get("error")
            assert twice["source"] == once
        else:
            assert twice["ok"] is False or twice["source"] != once

    def test_it_keeps_the_identity_of_everything_it_did_not_remove(self, case: Case):
        patched = patch_source({"source": SCENE, **case.request})["source"]
        before = set(identity_index(SCENE))
        after = set(identity_index(patched))
        # Deleting the last vertex of a sketch renumbers the ones after it,
        # and deleting a study takes its boundary conditions with it; each
        # case states what it expects to lose, and nothing else may go.
        assert before - after == set(case.removes), before - after


class TestTheBugsTheInvariantCaught:
    """Each of these failed the round trip before the fix beside it."""

    def test_deleting_a_study_an_optimization_drives_is_refused(self):
        """``delete_study`` used to leave a dangling ``study="..."``.

        ``delete_mesh`` already refused the mirror image — a mesh named by a
        study's ``mesh="..."`` — but a study named by an optimization's
        ``study="..."`` went through and left a program that raises on the
        very next compile, with no way back from the viewer.
        """
        result = patch_source({"source": SCENE, "op": "delete_study", "id": "assign:heat"})
        assert result == {
            "ok": False,
            "error": (
                "Study 'heat' is referenced by an optimization, so it cannot be "
                "deleted from the viewer. Point the optimization at another study first."
            ),
        }
        # The study nothing names still deletes.
        assert patch_source({"source": SCENE, "op": "delete_study", "id": "assign:idle"})["ok"]

    def test_deleting_a_mesh_a_study_names_is_still_refused(self):
        """The guard the study one now mirrors, sharing one helper.

        ``grid`` is reached by variable here, so the older guard answers
        first; the name-literal guard is what catches the form the viewer's
        own ``set_study_value(mesh=...)`` writes.
        """
        result = patch_source({"source": SCENE, "op": "delete_mesh", "id": "assign:grid"})
        assert result["ok"] is False
        assert "used elsewhere in the program" in result["error"]

        by_name = SCENE.replace("    mesh=grid,\n", '    mesh="grid",\n')
        result = patch_source({"source": by_name, "op": "delete_mesh", "id": "assign:grid"})
        assert result["ok"] is False
        assert "referenced by a study" in result["error"]


SHIPPED_OPERATIONS = [
    ("set_vertex", {"xy": [0.25, 0.125]}),
    ("insert_vertex", {"xy": [0.25, 0.125]}),
    ("delete_vertex", {}),
    ("solve_sketch", {"method": "newton", "iterations": 4}),
    ("add_constraint", {"kind": "horizontal", "indices": [0, 1]}),
]


def _shipped_cases(source: str) -> list[tuple[str, dict]]:
    """One request per operation per sketch of a real scene, addressed by id."""
    index = identity_index(source)
    sketches = [item for item in index.values() if item.kind == "sketch"]
    cases = []
    for sketch in sketches:
        for operation, extra in SHIPPED_OPERATIONS:
            identifier = f"vertex:{sketch.token}[0]" if "vertex" in operation else sketch.id
            if identifier not in index:
                continue
            cases.append(
                (f"{operation}@{sketch.token}", {"op": operation, "id": identifier, **extra})
            )
    return cases


@pytest.mark.parametrize(
    ("scene", "name", "request_body"),
    [
        (scene, name, body)
        for scene, source in (("starter", EXAMPLE_SOURCE), ("bracket", BRACKET))
        for name, body in _shipped_cases(source)
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
class TestTheShippedScenes:
    """The same invariants on the two programs users actually open.

    Compiling these is expensive, so the payload half is left to
    :class:`TestTheRoundTripInvariant`; what is checked here is that every
    id resolves against real, large, hand-written source and that the two
    addressing forms still agree there.
    """

    def test_the_patch_applies_or_explains_itself(self, scene, name, request_body):
        source = EXAMPLE_SOURCE if scene == "starter" else BRACKET
        result = patch_source({"source": source, **request_body})
        if result["ok"]:
            ast.parse(result["source"])
        else:
            # A refusal is fine — a sketch may already carry an operator —
            # but it must never be the id failing to resolve.
            assert "has the id" not in result["error"], result["error"]

    def test_the_stable_id_and_the_legacy_line_agree(self, scene, name, request_body):
        source = EXAMPLE_SOURCE if scene == "starter" else BRACKET
        index = identity_index(source)
        identity = index[request_body["id"]]
        legacy = {key: value for key, value in request_body.items() if key != "id"}
        legacy["line"] = index[identity.owner].line if identity.owner else identity.line
        if identity.index is not None and identity.kind == "vertex":
            legacy["index"] = identity.index
        assert patch_source({"source": source, **request_body}) == patch_source(
            {"source": source, **legacy}
        )
