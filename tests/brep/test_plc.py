"""The meshing spike: a PLC from the graph, and what it costs and buys.

Two claims are measured here rather than asserted in prose: re-projecting the
dual-contour surface onto its owner patches does not hurt the tet mesh, and
coarsening the planar faces to their own outlines is a large reduction that
only applies where no blend pins the boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.brep import brep_plc, extract_brep, plc_quality
from cadjoint.brep.plc import _ear_clip, recompute_plc_points
from tests.brep.conftest import plate_volume

tetgen = pytest.importorskip("tetgen")


@pytest.fixture(scope="module")
def comb_brep(starter_namespace, starter_grid):
    return starter_namespace["sink"], extract_brep(starter_namespace["sink"], starter_grid)


def _watertight(triangles: np.ndarray) -> bool:
    usage: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for index in range(3):
            a, b = int(triangle[index]), int(triangle[(index + 1) % 3])
            key = (a, b) if a < b else (b, a)
            usage[key] = usage.get(key, 0) + 1
    return all(count == 2 for count in usage.values())


def test_the_uncoarsened_plc_is_the_projected_surface(plate_brep):
    plc = brep_plc(plate_brep, coarsen=False)
    assert plc.stats["plc_triangles"] == plc.stats["dc_triangles"]
    assert plc.stats["coarsened_faces"] == 0
    assert _watertight(plc.triangles)
    assert plc.owner_arity.shape[0] == plc.points.shape[0]
    assert plc.owner_patches.shape == (plc.points.shape[0], 3)


def test_coarsening_collapses_a_wholly_planar_body(comb_brep):
    """The fin comb is eighteen planes: its PLC is its outline, not its lattice."""
    _scene, brep = comb_brep
    plc = brep_plc(brep, coarsen=True)
    assert plc.stats["coarsen_blocked_faces"] == 0
    assert plc.stats["coarsened_faces"] == len(brep.faces)
    assert plc.stats["plc_triangles"] * 20 < plc.stats["dc_triangles"]
    assert _watertight(plc.triangles)


def test_a_blend_pins_the_boundary_of_every_face_it_touches(thermal_brep):
    """Coarsening is all-or-nothing across an edge, so blends block it."""
    plc = brep_plc(thermal_brep, coarsen=True)
    assert plc.stats["coarsen_blocked_faces"] > 0
    assert plc.stats["coarsened_faces"] == 0
    assert plc.stats["plc_triangles"] == plc.stats["dc_triangles"]


def test_a_mixed_body_cannot_be_partly_coarsened(plate_brep):
    """The plate's caps have holes and its bore is curved, so nothing coarsens."""
    plc = brep_plc(plate_brep, coarsen=True)
    assert plc.stats["coarsened_faces"] == 0
    assert plc.stats["coarsen_blocked_faces"] == 4


def test_the_plc_tet_mesh_matches_the_dual_contour_path(plate_brep):
    plc = brep_plc(plate_brep, coarsen=False)
    from cadjoint.brep import plc_tet_mesh

    mesh = plc_tet_mesh(plate_brep, plc)
    quality = plc_quality(mesh.points, mesh.cells)
    assert quality["volume"] == pytest.approx(plate_volume(), rel=5e-3)
    assert quality["radius_ratio_min"] > 0.0
    assert mesh.num_surface == plc.points.shape[0]
    assert np.allclose(mesh.points[: mesh.num_surface], plc.points, atol=1e-12)


def test_nodes_move_by_their_own_arity_under_a_parameter_change(plate_brep):
    """The mesh's boundary follows the graph, corner by corner."""
    from cadjoint.brep import plc_tet_mesh

    plc = brep_plc(plate_brep, coarsen=False)
    mesh = plc_tet_mesh(plate_brep, plc)
    moved = recompute_plc_points(plate_brep, plc, mesh)
    assert moved.shape == mesh.points[: mesh.num_corner_points].shape
    # At the nominal design the projection is a fixed point of itself.
    assert np.allclose(moved[: mesh.num_surface], plc.points, atol=1e-6)
    assert np.allclose(moved[mesh.num_surface :], mesh.base_points[mesh.num_surface :])


def test_ear_clipping_triangulates_a_non_convex_polygon():
    polygon = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    triangles = _ear_clip(polygon)
    assert len(triangles) == polygon.shape[0] - 2
    area = 0.0
    for a, b, c in triangles:
        pa, pb, pc = polygon[a], polygon[b], polygon[c]
        area += 0.5 * float((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]))
    assert area == pytest.approx(4.0, abs=1e-9)  # 3x1 arm plus a 1x1 foot


def test_ear_clipping_refuses_a_degenerate_polygon():
    assert _ear_clip(np.zeros((2, 2))) == []
    assert _ear_clip(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])) == []
