"""Tests for the viewer-facing FEM surface payload (jaxcad.fem.render_payload)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from jaxcad.fem.hexmesh import GridSpec, sdf_to_hex_mesh
from jaxcad.fem.render_payload import (
    boundary_render_payload,
    cell_to_node_scalar,
    face_group_catalog,
)
from jaxcad.geometry.parameters import Vector
from jaxcad.sdf.primitives import Box, Sphere


def _box_mesh():
    box = Box(Vector([0.5, 0.5, 0.5], free=True, name="size"))
    grid = GridSpec.from_bounds((-0.75, -0.75, -0.75), (1.5, 1.5, 1.5), 12)
    return sdf_to_hex_mesh(box, grid)


def _sphere_mesh():
    grid = GridSpec.from_bounds((-0.8, -0.8, -0.8), (1.6, 1.6, 1.6), 10)
    return sdf_to_hex_mesh(Sphere(0.6), grid)


class TestFaceGroupCatalog:
    def test_box_has_six_axis_groups(self):
        catalog = face_group_catalog(_box_mesh())
        assert [entry["id"] for entry in catalog] == ["+x", "+y", "+z", "-x", "-y", "-z"]

    def test_entry_shape_and_geometry(self):
        catalog = face_group_catalog(_box_mesh())
        for entry in catalog:
            assert set(entry) == {"id", "axis", "side", "center", "area", "faces"}
            assert entry["axis"] in "xyz"
            assert entry["side"] in "+-"
            assert entry["id"] == entry["side"] + entry["axis"]
            assert len(entry["center"]) == 3
            assert entry["area"] > 0.0
            assert entry["faces"] > 0
        # The +x group's center sits on the +x face of the unit box.
        plus_x = next(entry for entry in catalog if entry["id"] == "+x")
        assert plus_x["center"][0] == pytest.approx(0.5, abs=1e-6)
        # One face of the unit box has area 1; each group must cover it.
        assert plus_x["area"] == pytest.approx(1.0, rel=1e-6)


class TestCellToNodeScalar:
    def test_constant_field_is_preserved(self):
        mesh = _box_mesh()
        nodal = cell_to_node_scalar(mesh, np.full(mesh.num_cells, 3.5))
        assert nodal.shape == (mesh.num_points,)
        assert np.allclose(nodal, 3.5)

    def test_wrong_length_is_rejected(self):
        mesh = _box_mesh()
        with pytest.raises(ValueError, match="per cell"):
            cell_to_node_scalar(mesh, np.zeros(mesh.num_cells + 1))


class TestBoundaryRenderPayload:
    def test_indices_in_range_and_scalar_per_vertex(self):
        mesh = _sphere_mesh()
        scalar = mesh.points[:, 0]  # Any nodal field.
        payload = boundary_render_payload(mesh, scalar)
        vertex_count = payload["vertex_count"]
        assert len(payload["positions"]) == vertex_count * 3
        assert len(payload["scalars"]) == vertex_count
        assert len(payload["indices"]) % 3 == 0
        indices = np.asarray(payload["indices"])
        assert indices.min() >= 0
        assert indices.max() < vertex_count
        low, high = payload["range"]
        assert low == pytest.approx(min(payload["scalars"]), abs=1e-5)
        assert high == pytest.approx(max(payload["scalars"]), abs=1e-5)

    def test_boundary_surface_is_watertight(self):
        mesh = _sphere_mesh()
        payload = boundary_render_payload(mesh, np.zeros(mesh.num_points))
        triangles = np.asarray(payload["indices"]).reshape(-1, 3)
        edges: Counter = Counter()
        for a, b, c in triangles:
            for u, v in ((a, b), (b, c), (c, a)):
                edges[(min(u, v), max(u, v))] += 1
        # The boundary of a voxel volume is closed: every edge of the
        # triangulated surface is shared by exactly two triangles.
        assert set(edges.values()) == {2}

    def test_group_ranges_tile_the_index_buffer(self):
        mesh = _box_mesh()
        payload = boundary_render_payload(mesh, np.zeros(mesh.num_points))
        offset = 0
        for group in payload["groups"]:
            assert group["start"] == offset
            assert group["count"] > 0
            assert group["count"] % 6 == 0  # Two triangles per quad.
            assert group["count"] == group["faces"] * 6
            offset += group["count"]
        assert offset == len(payload["indices"])

    def test_group_triangles_lie_on_their_face(self):
        mesh = _box_mesh()
        payload = boundary_render_payload(mesh, np.zeros(mesh.num_points))
        positions = np.asarray(payload["positions"]).reshape(-1, 3)
        indices = np.asarray(payload["indices"])
        plus_x = next(group for group in payload["groups"] if group["id"] == "+x")
        used = indices[plus_x["start"] : plus_x["start"] + plus_x["count"]]
        assert np.allclose(positions[used][:, 0], 0.5, atol=1e-4)

    def test_wrong_scalar_length_is_rejected(self):
        mesh = _box_mesh()
        with pytest.raises(ValueError, match="per node"):
            boundary_render_payload(mesh, np.zeros(mesh.num_points - 1))
