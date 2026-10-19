"""The B-rep inspector's payload (:mod:`cadjoint.viewer._brep_inspector`).

Most of this runs on stand-ins for the private tier — a duck-typed B-rep and
a ``feature_edges`` component — so the public gate checks the reading of the
B-rep, the edge kinds and the off-surface measure without ``diff-brep``.
The last test runs the real thing through the worker when it is installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cadjoint import tier
from cadjoint.geometry import Vector
from cadjoint.plugins.contracts import EdgeSet
from cadjoint.sdf.primitives import Box
from cadjoint.viewer import _brep_inspector
from cadjoint.viewer._brep_inspector import OFF_SURFACE_CELLS, brep_inspection
from cadjoint.viewer.schema.payloads import BRepInspectionPayload


def _box() -> Box:
    # `size` is the half-extent: faces at ±0.5.
    return Box(size=Vector([0.5, 0.5, 0.5]))


def _rail(x: float, y: float) -> np.ndarray:
    """A vertical polyline at (x, y): on the unit box's edge when x = y = 0.5."""
    return np.stack([np.full(9, x), np.full(9, y), np.linspace(-0.4, 0.4, 9)], axis=1)


def _fake_brep() -> SimpleNamespace:
    """Two quads' worth of B-rep: three faces, four edges, one vertex.

    Edge 0 is the box's own vertical edge (the overlay draws it, traced);
    edge 1 is a crossing the overlay did not draw; edge 2 is a blend boundary
    floating off the box; edge 3 carries a state the inspector has never
    heard of.
    """
    face = lambda index, kind, analytic, patch: SimpleNamespace(  # noqa: E731
        index=index, kind=kind, analytic=analytic, patch=patch, leaf=0, area=1.0
    )
    edge = lambda index, faces, patches, polyline, state: SimpleNamespace(  # noqa: E731
        index=index,
        faces=faces,
        patches=patches,
        polyline=polyline,
        vertices=(-1, -1),
        closed=False,
        analytic=state != "blend",
        residual=0.0,
        crossing=1.0,
        state=state,
    )
    points = np.array(
        [
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        faces=[face(0, "plane", True, 0), face(1, "plane", False, 1), face(2, "blend", False, -1)],
        edges=[
            edge(0, (0, 1), (0, 1), _rail(0.5, 0.5), "crossing"),
            edge(1, (0, 2), (0, 2), _rail(0.5, -0.5), "crossing"),
            edge(2, (1, 2), (1, -1), _rail(0.7, 0.7), "blend"),
            edge(3, (1, 2), (1, 3), _rail(-0.5, 0.5), "a_state_from_the_future"),
        ],
        vertices=[
            SimpleNamespace(
                index=0,
                kind="corner",
                point=np.array([0.5, 0.5, 0.5]),
                faces=(0, 1, 2),
                residual=0.0,
            )
        ],
        points=points,
        # No triangles: the inspector must split the lattice quads itself.
        triangles=np.zeros((0, 3), dtype=np.int32),
        triangle_face=np.zeros(0, dtype=np.int64),
        mesh=SimpleNamespace(quads=np.array([[0, 1, 2, 3], [1, 4, 5, 2]])),
        quad_face=np.array([0, 1]),
        decomposition=SimpleNamespace(leaves=[_box()]),
        stats={"source": "dual_contouring", "ineligible_reason": "a blend", "quads": 2},
    )


def _drawn_curves() -> EdgeSet:
    """The overlay drew edge 0 only, slightly re-sampled (as its trace would)."""
    rail = _rail(0.5, 0.5)
    return EdgeSet(
        polylines=(rail[::2] + 1e-6,),
        closed=np.array([False]),
        patches=np.array([[1, 0]]),
        kind=("traced",),
        residual=np.array([0.0]),
        vertices=np.array([[-1, -1]]),
    )


@pytest.fixture
def fake_tier(monkeypatch):
    calls = {}

    class Extractor:
        def extract(self, scene, grid, **options):
            calls["grid"] = grid
            return _fake_brep()

    class Drawer:
        def feature_edges(self, scene, grid, **options):
            return _drawn_curves()

    components = {"brep": Extractor(), "feature_edges": Drawer()}
    monkeypatch.setattr(tier, "component", lambda kind: components.get(str(kind)))
    return calls


def test_without_the_private_tier_the_inspector_says_why(monkeypatch):
    monkeypatch.setattr(tier, "component", lambda kind: None)

    payload = brep_inspection(_box())

    assert payload["available"] is False
    assert "diff-brep" in payload["reason"]
    BRepInspectionPayload.model_validate(payload)


def test_a_failed_extraction_is_reported_not_raised(monkeypatch):
    class Broken:
        def extract(self, scene, grid, **options):
            raise IndexError("list index out of range")

    monkeypatch.setattr(tier, "component", lambda kind: Broken() if kind == "brep" else None)

    payload = brep_inspection(_box())

    assert payload["available"] is True
    assert payload["error"].startswith("IndexError: list index out of range (at extract")
    BRepInspectionPayload.model_validate(payload)


def test_faces_are_named_by_certified_surface_and_owning_patch(fake_tier):
    payload = brep_inspection(_box())
    BRepInspectionPayload.model_validate(payload)

    surfaces = {face["id"]: face["surface"] for face in payload["faces"]}
    # Certified plane, uncertified plane, blend.
    assert surfaces == {0: "plane", 1: "faceted", 2: "blend"}
    assert payload["faces"][2]["patch"] == -1
    assert payload["faces"][0]["leaf_type"] == "Box"


def test_the_lattice_quads_are_split_into_a_tessellation_with_face_ids(fake_tier):
    payload = brep_inspection(_box())

    triangles = np.asarray(payload["triangles"]).reshape(-1, 3)
    assert triangles.shape == (4, 3)
    assert sorted(payload["triangle_face"]) == [0, 0, 1, 1]
    assert len(payload["positions"]) == 6 * 3


def test_the_b_reps_own_corner_normals_are_shipped_for_smooth_shading(monkeypatch):
    """A B-rep that knows its surfaces' normals sends them; split lattice quads send none.

    The viewer shades each face with the normal at every triangle corner, so
    a smooth blend shades smooth instead of showing its facets.  The normals
    are the private tier's (``brep.corner_normals(scene)``), one per corner
    in triangle order, and only for its own triangles: the quads this module
    splits itself have none, and are shaded flat as before.
    """
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.tile(np.array([1.0, 0.0, 0.0]), (2, 3, 1))
    asked = []

    def corner_normals(scene):
        asked.append(scene)
        return normals

    brep = _fake_brep()
    brep.triangles = triangles
    brep.triangle_face = np.array([0, 0])
    brep.corner_normals = corner_normals

    class Extractor:
        def extract(self, scene, grid, **options):
            return brep

    monkeypatch.setattr(
        tier, "component", lambda kind: Extractor() if str(kind) == "brep" else None
    )
    scene = _box()
    payload = brep_inspection(scene)
    BRepInspectionPayload.model_validate(payload)
    assert asked == [scene]
    assert payload["corner_normals"] == [1.0, 0.0, 0.0] * 6

    lattice = _fake_brep()
    lattice.corner_normals = corner_normals
    monkeypatch.setattr(
        tier,
        "component",
        lambda kind: SimpleNamespace(extract=lambda *a, **k: lattice)
        if str(kind) == "brep"
        else None,
    )
    assert "corner_normals" not in brep_inspection(_box())


def test_edges_are_kinded_by_how_they_reached_the_screen(fake_tier):
    payload = brep_inspection(_box())

    kinds = {edge["id"]: (edge["kind"], edge["state"]) for edge in payload["edges"]}
    assert kinds == {
        0: ("traced", "crossing"),  # the overlay's curve, matched back by geometry
        1: ("undrawn", "crossing"),  # a crossing the overlay dropped
        2: ("blend", "blend"),
        3: ("a_state_from_the_future", "a_state_from_the_future"),  # passed through
    }
    # The drawn curve replaces the graph's seeds for the edge it matched.
    assert len(payload["edges"][0]["points"]) == 5 * 3
    assert payload["diagnostics"]["edge_kinds"]["traced"] == 1


def test_an_edge_off_the_solid_is_measured_by_the_scene_itself(fake_tier):
    payload = brep_inspection(_box())
    diagnostics = payload["diagnostics"]

    off = {edge["id"]: edge["off_surface"] for edge in payload["edges"]}
    # float32 evaluation, and the drawn curve sits 1e-6 off the rail.
    assert off[0] < 1e-5 and off[1] < 1e-5
    # (0.7, 0.7) is 0.2 outside both faces: |f| is the corner distance.
    assert off[2] == pytest.approx(0.2 * np.sqrt(2.0), abs=1e-5)
    assert diagnostics["off_surface_tolerance"] == pytest.approx(
        OFF_SURFACE_CELLS * diagnostics["cell"]
    )
    assert diagnostics["off_surface_edges"] == 1


def test_the_diagnostics_name_the_path_and_the_reason(fake_tier):
    diagnostics = brep_inspection(_box())["diagnostics"]

    assert diagnostics["exact"] is False
    assert diagnostics["source"] == "dual_contouring"
    assert diagnostics["fallback_reason"] == "a blend"
    assert (diagnostics["faces"], diagnostics["edges"], diagnostics["vertices"]) == (3, 4, 1)
    assert diagnostics["stats"] == {"quads": 2}


def test_the_inspector_extracts_on_the_overlays_grid(fake_tier):
    from cadjoint.viewer._edge_overlay import _overlay_grid

    brep_inspection(_box())

    assert np.allclose(fake_tier["grid"].spacing, _overlay_grid().spacing)


def test_curve_matching_takes_the_nearest_edge_between_the_same_patches():
    near = SimpleNamespace(patches=(0, 1), polyline=_rail(0.5, 0.5))
    far = SimpleNamespace(patches=(1, 0), polyline=_rail(0.5, 0.5) + [0.0, 0.3, 0.0])
    other = SimpleNamespace(patches=(0, 2), polyline=_rail(0.5, 0.5))

    matched = _brep_inspector._match_curves([far, near, other], _drawn_curves(), cell=0.1)

    assert matched == {1: 0}


@pytest.mark.skipif(not tier.available("brep"), reason="the private tier is not installed")
def test_the_worker_serves_an_exact_brep_of_hard_csg():
    from cadjoint.viewer._worker_client import brep_source

    source = (
        "from cadjoint.construction import Solid\n"
        "from cadjoint.sdf.boolean import Difference\n"
        "block = Solid.box(size=[0.9, 0.6, 0.3], position=[0.0, 0.0, 0.0], name='block')\n"
        "hole = Solid.cylinder(radius=0.1, height=0.8, position=[0.2, 0.0, 0.0], name='hole')\n"
        "scene = Difference(block, hole, smoothness=0)\n"
    )

    result = brep_source(source)

    assert result["ok"] is True, result.get("error")
    payload = result["brep"]
    assert payload["available"] is True and "error" not in payload
    diagnostics = payload["diagnostics"]
    assert diagnostics["exact"] is True
    assert diagnostics["surfaces"] == {"cylinder": 1, "plane": 6}
    assert set(diagnostics["edge_kinds"]) == {"traced"}
    assert diagnostics["off_surface_edges"] == 0
    assert len(payload["triangle_face"]) * 3 == len(payload["triangles"])
    assert set(payload["triangle_face"]) == {face["id"] for face in payload["faces"]}


def test_each_face_says_how_the_step_export_writes_it(monkeypatch):
    """With the ``step_export`` kind, every face carries its export surface and the card the file's count.

    The plan is the B-rep's own (``brep.step_plan(scene)``): an exact surface
    as it is, a blend on one certified B-spline with its errors, a face no
    surface carries as facets with the reason.
    """
    brep = _fake_brep()
    brep.step_plan = lambda scene: {
        "length_unit": "metre",
        "fit_tolerance": 1e-4,
        "surfaces": {"plane": 1, "bspline": 1, "faceted": 1, "cylinder": 0},
        "faces": [
            {
                "face": 0,
                "surface": "plane",
                "fit_error": None,
                "field_error": None,
                "control_points": 0,
                "reason": "",
            },
            {
                "face": 1,
                "surface": "faceted",
                "fit_error": 3e-3,
                "field_error": 2e-2,
                "control_points": 900,
                "reason": "a crease inside the face",
            },
            {
                "face": 2,
                "surface": "bspline",
                "fit_error": 2e-5,
                "field_error": 4e-5,
                "control_points": 1200,
                "reason": "",
            },
        ],
    }
    components = {
        "brep": SimpleNamespace(extract=lambda *a, **k: brep),
        "step_export": SimpleNamespace(),
    }
    monkeypatch.setattr(tier, "component", lambda kind: components.get(str(kind)))

    payload = brep_inspection(_box())
    BRepInspectionPayload.model_validate(payload)

    exports = {face["id"]: face["export"] for face in payload["faces"]}
    assert exports[0]["surface"] == "plane" and exports[0]["fit_error"] is None
    assert exports[1]["reason"] == "a crease inside the face"
    assert exports[2] == {
        "surface": "bspline",
        "fit_error": 2e-5,
        "field_error": 4e-5,
        "control_points": 1200,
        "reason": "",
    }
    step = payload["diagnostics"]["step"]
    assert step == {
        "faces": 3,
        "surfaces": {"plane": 1, "bspline": 1, "faceted": 1},
        "fit_tolerance": 1e-4,
        "fit_error_max": 3e-3,
        "length_unit": "metre",
    }
    assert payload["diagnostics"]["step_note"] is None


def test_without_the_step_export_kind_there_is_no_plan_and_the_card_says_why(fake_tier):
    payload = brep_inspection(_box())
    BRepInspectionPayload.model_validate(payload)

    assert all("export" not in face for face in payload["faces"])
    assert payload["diagnostics"]["step"] is None
    assert "diff-brep" in payload["diagnostics"]["step_note"]
