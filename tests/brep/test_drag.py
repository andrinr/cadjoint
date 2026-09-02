"""Dragging a handle: solve for the design, keep the constraints, refuse topology.

The headline is the starter's fin comb, because it is the case a stored
B-rep cannot do: the corner being dragged is not stored anywhere, it is the
solution of three patch equations whose parameters are sketch points under a
production-style constraint system.  Moving it has to come out as a *sketch
edit*.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.brep import drag_handle, extract_brep
from cadjoint.brep.drag import handle_position
from tests.brep.conftest import PLATE_GRID, plate_scene


@pytest.fixture(scope="module")
def comb(starter_namespace, starter_grid):
    """The starter's fin comb alone: eighteen planar faces, no blends."""
    scene = starter_namespace["sink"]
    return scene, extract_brep(scene, starter_grid)


def _corner_near(brep, position):
    candidates = [vertex for vertex in brep.vertices if vertex.analytic]
    distances = [float(np.linalg.norm(vertex.point - position)) for vertex in candidates]
    return candidates[int(np.argmin(distances))], min(distances)


def test_dragging_a_fin_tip_corner_edits_exactly_one_sketch_point(comb):
    scene, brep = comb
    # World x is the profile's -x, world z its y, and the comb is extruded
    # along world y; fin1's outer tip corner is profile (0.75, 0.85).
    vertex, distance = _corner_near(brep, np.array([-0.75, 0.6, 0.85]))
    assert distance < 1e-4, "the extraction should land a corner on the fin tip"

    result = drag_handle(
        scene,
        brep,
        "vertex",
        vertex.index,
        vertex.point + np.array([0.05, 0.0, 0.0]),
        apply=False,
    )
    assert not result.topology_changed
    assert result.error < 1e-6
    assert result.constraint_residual < 1e-6
    assert [name for name, _ in result.moved] == ["fin1_tip_r"]
    assert result.moved[0][1] == pytest.approx(0.05, abs=1e-6)


def test_a_drag_that_would_change_topology_is_reported_not_solved(comb):
    scene, brep = comb
    vertex, _ = _corner_near(brep, np.array([-0.75, 0.6, 0.85]))
    buried = vertex.point + np.array([0.0, 0.0, -0.9])
    result = drag_handle(scene, brep, "vertex", vertex.index, buried, apply=False)
    assert result.topology_changed
    assert not result.applied
    assert "boundary" in result.reason
    assert result.error > 0.1, "the handle cannot reach a target that is not on the solid"


def test_the_scene_is_left_untouched_when_a_drag_is_not_applied(comb):
    from cadjoint.extraction import extract_parameters

    scene, brep = comb
    before = {name: np.asarray(value) for name, value in extract_parameters(scene)[0].items()}
    vertex, _ = _corner_near(brep, np.array([-0.75, 0.6, 0.85]))
    drag_handle(
        scene,
        brep,
        "vertex",
        vertex.index,
        vertex.point + np.array([0.02, 0.0, 0.0]),
        apply=False,
    )
    after = extract_parameters(scene)[0]
    for name, value in before.items():
        assert np.array_equal(value, np.asarray(after[name])), name


def test_applying_a_drag_writes_the_solved_design_back():
    """A local scene, so the mutation cannot leak into another test."""
    from cadjoint.extraction import extract_parameters
    from cadjoint.geometry import Scalar, Vector
    from cadjoint.sdf.boolean import Difference
    from cadjoint.sdf.primitives import Box, Cylinder
    from cadjoint.sdf.transforms import Translate

    size = Vector([0.6, 0.6, 0.4], free=True, name="plate_size")
    scene = Difference(
        (
            Box(size=size),
            Translate(Cylinder(radius=Scalar(0.25), height=Scalar(0.9)), Vector([0.0, 0.0, 0.0])),
        ),
        smoothness=0.0,
    )
    brep = extract_brep(scene, PLATE_GRID)
    vertex, _ = _corner_near(brep, np.array([0.6, 0.6, 0.4]))
    result = drag_handle(
        scene, brep, "vertex", vertex.index, np.array([0.66, 0.6, 0.4]), apply=True
    )
    assert result.applied
    assert result.error < 1e-6
    assert np.asarray(extract_parameters(scene)[0]["plate_size"])[0] == pytest.approx(
        0.66, abs=1e-5
    )


def test_restricting_the_parameters_restricts_the_edit(comb):
    scene, brep = comb
    vertex, _ = _corner_near(brep, np.array([-0.75, 0.6, 0.85]))
    result = drag_handle(
        scene,
        brep,
        "vertex",
        vertex.index,
        vertex.point + np.array([0.0, 0.05, 0.0]),
        parameters=["fin_depth"],
        apply=False,
    )
    assert {name for name, _ in result.moved} <= {"fin_depth"}
    assert result.error < 1e-6


def test_an_edge_handle_moves_across_its_own_curve(comb):
    """An edge point can only be moved off its curve, never along it."""
    scene, brep = comb
    edge = max(
        (edge for edge in brep.edges if edge.analytic),
        key=lambda edge: edge.polyline.shape[0],
    )
    middle = edge.polyline.shape[0] // 2
    station = edge.polyline[middle]
    tangent = edge.polyline[middle + 1] - edge.polyline[middle - 1]
    tangent /= np.linalg.norm(tangent)
    normal = np.eye(3)[int(np.argmin(np.abs(tangent)))]
    normal = normal - float(normal @ tangent) * tangent
    normal /= np.linalg.norm(normal)

    across = drag_handle(scene, brep, "edge", edge.index, station + 0.01 * normal, apply=False)
    assert across.handle.startswith("edge:")
    assert across.error < 1e-5
    assert across.moved

    # Along the curve the design cannot help: the projection puts the handle
    # straight back, so the drag is a measured no-op rather than a wrong edit.
    along = drag_handle(scene, brep, "edge", edge.index, station + 0.01 * tangent, apply=False)
    assert np.allclose(along.achieved, station, atol=1e-5)


def test_handle_position_reproduces_the_graph_at_the_nominal_design(comb):
    from cadjoint.extraction import extract_parameters

    scene, brep = comb
    free = extract_parameters(scene)[0]
    vertex = next(v for v in brep.vertices if v.analytic)
    solved = handle_position(scene, vertex.patches, vertex.point, free, max_step=0.1)
    assert np.allclose(np.asarray(solved), vertex.point, atol=1e-6)


def test_unusable_handles_are_refused(comb, thermal_brep, starter_namespace):
    scene, brep = comb
    with pytest.raises(ValueError, match="handle must be"):
        drag_handle(scene, brep, "face", 0, np.zeros(3))
    blended = next((vertex for vertex in thermal_brep.vertices if not vertex.analytic), None)
    if blended is not None:
        with pytest.raises(ValueError, match="not a clean triple point"):
            drag_handle(
                starter_namespace["thermal_body"],
                thermal_brep,
                "vertex",
                blended.index,
                np.zeros(3),
            )


def test_a_scene_with_no_free_parameters_cannot_be_dragged():
    scene = plate_scene()
    brep = extract_brep(scene, PLATE_GRID)
    vertex = next(v for v in brep.vertices if v.analytic)
    with pytest.raises(ValueError, match="no free parameters"):
        drag_handle(scene, brep, "vertex", vertex.index, vertex.point)
