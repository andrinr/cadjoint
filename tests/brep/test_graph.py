"""The ownership graph on a hard solid and on a blended one.

The plate is the exactness test: its whole boundary lies on patch zero sets,
so the derived graph must be the textbook one — six planes, one cylinder,
twelve edges, eight corners, every corner solved to float precision.  The
starter's thermal body is the ambiguity test: smooth unions produce faces
that lie on no patch, and what matters there is that they are *identified*
rather than mistaken for geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.brep import extract_brep
from cadjoint.brep.graph import _revolved_edge_kind
from tests.brep.conftest import BORE_RADIUS, PLATE_GRID, PLATE_SIZE


def test_the_plate_graph_is_the_textbook_one(plate_brep):
    report = plate_brep.report()
    assert report["faces"] == 7
    assert report["face_kinds"] == {"plane": 6, "cylinder": 1}
    assert report["edges"] == 14  # 12 box edges + 2 bore rims
    assert report["vertices"] == 8
    assert report["blend_faces"] == 0
    assert report["blend_quads"] == 0
    assert report["ambiguous_vertices"] == 0
    assert report["non_simple_faces"] == 0
    assert report["analytic_faces"] == 7


def test_the_plate_corners_are_solved_exactly(plate_brep):
    corners = np.array(sorted(tuple(np.round(v.point, 9)) for v in plate_brep.vertices))
    expected = np.array(
        sorted(
            (sx * PLATE_SIZE[0], sy * PLATE_SIZE[1], sz * PLATE_SIZE[2])
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        )
    )
    assert np.allclose(corners, expected, atol=1e-6)
    assert max(vertex.residual for vertex in plate_brep.vertices) < 1e-9
    assert all(vertex.analytic for vertex in plate_brep.vertices)


def test_the_bore_is_a_cylinder_of_the_right_radius(plate_brep):
    cylinder = next(face for face in plate_brep.faces if face.kind == "cylinder")
    assert cylinder.surface.kind == "cylinder"
    assert cylinder.surface.radius == pytest.approx(BORE_RADIUS, abs=1e-5)
    assert abs(float(cylinder.surface.axis[2])) == pytest.approx(1.0, abs=1e-6)
    assert cylinder.surface.residual < 1e-6
    # A bore: the solid is outside the fitted cylinder.
    assert cylinder.surface.sense == -1.0
    assert len(cylinder.loops) == 2


def test_the_caps_keep_their_hole_as_a_second_loop(plate_brep):
    caps = [
        face
        for face in plate_brep.faces
        if face.kind == "plane" and abs(float(face.surface.axis[2])) > 0.9
    ]
    assert len(caps) == 2
    for cap in caps:
        assert len(cap.loops) == 2, "a cap is a plane with a hole, not two faces"


def test_every_plate_edge_lies_on_both_its_patches(plate_brep):
    assert all(edge.analytic for edge in plate_brep.edges)
    assert max(edge.residual for edge in plate_brep.edges) < 1e-6
    closed = [edge for edge in plate_brep.edges if edge.closed]
    assert len(closed) == 2, "the two bore rims are the only closed chains"


def test_planes_are_fitted_to_the_exact_faces(plate_brep):
    planes = [face for face in plate_brep.faces if face.kind == "plane"]
    assert max(face.surface.residual for face in planes) < 1e-9
    offsets = sorted(round(float(face.surface.origin @ face.surface.axis), 6) for face in planes)
    assert offsets == [0.4, 0.4, 0.6, 0.6, 0.6, 0.6]


def test_every_mesh_vertex_carries_its_owner(plate_brep):
    arity = plate_brep.owner_arity
    assert arity.min() >= 1, "a hard solid has no blend neighbourhoods"
    assert arity.max() == 3
    assert np.count_nonzero(arity == 3) == 8
    for row in range(plate_brep.points.shape[0]):
        owners = plate_brep.owner_patches[row, : arity[row]]
        assert (owners >= 0).all()
        fields = [plate_brep.patches[index].field for index in owners]
        from cadjoint.brep.project import field_residuals

        assert float(field_residuals(fields, plate_brep.points[row : row + 1])[0]) < 1e-5


def test_projection_repairs_a_smooth_extraction(plate):
    """The graph does not inherit the extractor's vertex placement.

    Extracted with the Tikhonov (``sharp=False``) placement, the corners of
    the plate carry the regularizer's mass-point bias — visibly off the
    geometry.  Re-solving them against their own three patches removes it
    entirely, which is the claim that dual contouring is a discovery tool
    and not the geometry.
    """
    from cadjoint.brep.project import field_residuals
    from cadjoint.meshing.dual_contouring import extract_mesh

    smooth = extract_mesh(plate, PLATE_GRID, sharp=False)
    brep = extract_brep(plate, PLATE_GRID, mesh=smooth)
    raw = np.asarray(smooth.vertices, dtype=np.float64)
    corners = np.flatnonzero(brep.owner_arity == 3)
    assert corners.size >= 8
    fields = [
        [brep.patches[index].field for index in brep.owner_patches[row, :3]] for row in corners
    ]
    before = max(
        float(field_residuals(f, raw[row : row + 1])[0]) for f, row in zip(fields, corners)
    )
    after = max(
        float(field_residuals(f, brep.points[row : row + 1])[0]) for f, row in zip(fields, corners)
    )
    # Measured on this grid: 4.4e-5 before, below 1e-6 after — the Tikhonov
    # bias is of order ``regularization x cell size`` and it goes away.
    assert before > 1e-5, "the smooth placement really is off the corner"
    assert after < 1e-6
    assert after * 10 < before


def test_the_thermal_body_separates_blends_from_analytic_faces(thermal_brep):
    report = thermal_brep.report()
    assert report["blend_faces"] > 0, "the starter unions at smoothness 0.03"
    assert report["analytic_faces"] > report["blend_faces"]
    kinds = report["face_kinds"]
    assert kinds.get("plane", 0) > 0
    assert kinds.get("cylinder", 0) > 0, "the bushings and the slug bore"
    # Every non-blend face carries a certified surface; every blend face
    # carries none, by construction.
    for face in thermal_brep.faces:
        if face.kind == "blend":
            assert not face.analytic
            assert face.surface.kind == "freeform"
        else:
            assert face.analytic
            assert face.surface.residual < 1e-5


def test_blend_neighbourhoods_are_reported_not_solved(thermal_brep):
    blend_rows = np.flatnonzero(thermal_brep.owner_arity == 0)
    assert blend_rows.size > 0
    raw = np.asarray(thermal_brep.mesh.vertices, dtype=np.float64)
    assert np.allclose(
        thermal_brep.points[blend_rows], raw[blend_rows]
    ), "a point with no owning patch must keep its dual-contour position"
    assert thermal_brep.stats["ambiguous_vertices"] > 0
    assert thermal_brep.stats["tangent_or_blend_edges"] > 0


def test_the_thermal_body_finds_the_bushings_and_the_slug(thermal_brep):
    cylinders = [face for face in thermal_brep.faces if face.kind == "cylinder"]
    radii = sorted(round(face.surface.radius, 4) for face in cylinders)
    assert 0.07 in radii or 0.05 in radii or 0.26 in radii, radii
    for face in cylinders:
        assert face.surface.residual < 1e-5


def test_revolved_profile_edges_get_their_swept_surface_type():
    """A profile edge's direction decides cylinder / plane / cone."""
    from cadjoint.geometry import Vector2
    from cadjoint.sdf.primitives.polygon import RevolvedPolygon

    profile = RevolvedPolygon(
        [
            Vector2(value=[0.2, -0.1]),  # 0 -> 1: constant radius  => cylinder
            Vector2(value=[0.2, 0.1]),  # 1 -> 2: constant height  => plane
            Vector2(value=[0.4, 0.1]),  # 2 -> 3: neither          => cone
            Vector2(value=[0.5, -0.1]),  # 3 -> 0: constant height => plane
        ]
    )
    kinds = [_revolved_edge_kind(profile, index) for index in range(4)]
    assert kinds == ["cylinder", "plane", "cone", "plane"]


def test_extraction_refuses_a_grid_that_misses_the_surface(plate):
    from cadjoint.meshing.edge_detection import GridSpec

    empty = GridSpec.from_bounds((5.0, 5.0, 5.0), (1.0, 1.0, 1.0), 4)
    with pytest.raises(ValueError, match="no quads"):
        extract_brep(plate, empty)


def test_the_graph_is_stable_against_the_grid(plate):
    """Topology is discovered, so a different lattice must find the same graph."""
    shifted = extract_brep(
        plate,
        type(PLATE_GRID)(
            origin=(-0.871, -0.869, -0.657),
            spacing=PLATE_GRID.spacing,
            cells=PLATE_GRID.cells,
        ),
    )
    report = shifted.report()
    assert report["faces"] == 7
    assert report["edges"] == 14
    assert report["vertices"] == 8
