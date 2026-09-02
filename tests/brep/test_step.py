"""Export from the graph: exact STEP, and the tessellation beside it.

The plate is the case where the whole model has a closed form, so the file
should contain no facets at all — six ``PLANE``s and one
``CYLINDRICAL_SURFACE``, sewing into one solid whose volume is the analytic
one.  The thermal body is the mixed case: analytic where the patches own the
surface, faceted across the blends, and still one closed shell.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.brep import (
    brep_loops,
    brep_triangles,
    brep_volume,
    save_brep_obj,
    save_brep_step,
    save_brep_stl,
    simplify_loop,
)
from tests.brep.conftest import plate_volume


def test_the_plate_exports_with_no_facets(plate_brep, tmp_path):
    report = save_brep_step(plate_brep, tmp_path / "plate.step")
    assert report["faces"]["facet"] == 0
    assert report["faces"]["dropped"] == 0
    assert report["faces"]["plane"] == 6
    assert report["faces"]["cylinder"] == 1
    keywords = report["keywords"]
    assert keywords["PLANE"] == 6
    assert keywords["CYLINDRICAL_SURFACE"] == 1
    assert keywords["CIRCLE"] == 2, "the two rims are one shared circle each"
    assert keywords["ADVANCED_FACE"] == 7
    assert keywords["EDGE_CURVE"] == 14
    assert keywords["VERTEX_POINT"] == 10, "8 box corners + 2 circle seams"


def test_loops_collapse_onto_their_exact_edges(plate_brep):
    loops = brep_loops(plate_brep)
    sides = [
        face
        for face in plate_brep.faces
        if face.kind == "plane" and abs(float(face.surface.axis[2])) < 0.5
    ]
    assert len(sides) == 4
    for face in sides:
        assert len(face.loops[0]) > 20, "the raw dual-contour boundary walks per cell"
        assert len(loops[face.index][0]) == 4, "a rectangle has four corners"


def test_simplify_loop_keeps_protected_vertices():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    loop = [0, 1, 2, 3]
    assert simplify_loop(points, loop, 1e-9) == [0, 2, 3]
    assert simplify_loop(points, loop, 1e-9, {1}) == [0, 1, 2, 3]


def test_the_tessellation_is_watertight_and_encloses_the_plate(plate_brep):
    triangles, face_ids = brep_triangles(plate_brep)
    assert face_ids.shape[0] == triangles.shape[0]
    usage: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for index in range(3):
            a, b = int(triangle[index]), int(triangle[(index + 1) % 3])
            key = (a, b) if a < b else (b, a)
            usage[key] = usage.get(key, 0) + 1
    assert all(count == 2 for count in usage.values()), "the surface must close"
    assert brep_volume(plate_brep) == pytest.approx(plate_volume(), rel=5e-3)


def test_obj_and_stl_round_trip(plate_brep, tmp_path):
    obj_path = tmp_path / "plate.obj"
    save_brep_obj(plate_brep, obj_path)
    lines = obj_path.read_text().splitlines()
    vertices = [line for line in lines if line.startswith("v ")]
    faces = [line for line in lines if line.startswith("f ")]
    assert len(vertices) == plate_brep.points.shape[0]
    quads = [line for line in faces if len(line.split()) == 5]
    assert quads, "the four side planes become single quads"

    stl_path = tmp_path / "plate.stl"
    save_brep_stl(plate_brep, stl_path)
    payload = stl_path.read_bytes()
    count = int(np.frombuffer(payload[80:84], dtype="<u4")[0])
    assert count == brep_triangles(plate_brep)[0].shape[0]
    assert len(payload) == 84 + 50 * count


def test_the_thermal_body_is_analytic_where_it_can_be(thermal_brep, tmp_path):
    report = save_brep_step(thermal_brep, tmp_path / "sink.step")
    assert report["faces"]["plane"] > 0, "the fin walls and decks are exact"
    assert report["faces"]["facet"] > 0, "the blends have no closed form"
    assert report["faces"]["dropped"] == 0
    assert report["step_faces"] == sum(
        value for key, value in report["faces"].items() if key != "dropped"
    )


def test_disabling_analytic_output_facets_everything(plate_brep, tmp_path):
    report = save_brep_step(plate_brep, tmp_path / "faceted.step", analytic=False)
    assert report["faces"]["plane"] == 0
    assert report["faces"]["cylinder"] == 0
    assert report["faces"]["facet"] == brep_triangles(plate_brep)[0].shape[0]


def test_the_analytic_file_is_far_smaller_than_the_faceted_one(plate_brep, tmp_path):
    analytic = save_brep_step(plate_brep, tmp_path / "a.step")
    faceted = save_brep_step(plate_brep, tmp_path / "b.step", analytic=False)
    assert analytic["entities"] * 20 < faceted["entities"]
