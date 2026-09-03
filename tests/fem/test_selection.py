"""Tests for cadjoint.fem.selection (programmatic node selection)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cadjoint.fem import (
    GridSpec,
    Nodes,
    faces_from_nodes,
    sdf_to_hex_mesh,
    select_faces,
    selection_from_description,
)
from cadjoint.fem.selection import boundary_node_mask
from cadjoint.geometry.parameters import Vector
from cadjoint.sdf.primitives import Box

# Bar of 2.0 x 0.3 x 0.3 along x on a face-aligned grid (spacing 0.1).
_BOUNDS = (-1.1, -0.25, -0.25)
_SIZE = (2.2, 0.5, 0.5)
_RESOLUTION = (22, 5, 5)


@pytest.fixture(scope="module")
def bar_mesh():
    bar = Box(Vector([1.0, 0.15, 0.15], free=True, name="size"))
    return sdf_to_hex_mesh(bar, GridSpec.from_bounds(_BOUNDS, _SIZE, _RESOLUTION))


class TestPrimitives:
    def test_box_selects_the_end_face(self, bar_mesh):
        indices = Nodes.box([0.95, -1.0, -1.0], [1.05, 1.0, 1.0]).resolve(bar_mesh)
        assert np.allclose(bar_mesh.points[indices, 0], 1.0)
        # 3x3 quads on the end face -> 4x4 nodes.
        assert indices.size == 16

    def test_sphere_selects_around_a_corner(self, bar_mesh):
        selection = Nodes.sphere([1.0, 0.15, 0.15], 0.05)
        indices = selection.resolve(bar_mesh)
        distances = np.linalg.norm(bar_mesh.points[indices] - [1.0, 0.15, 0.15], axis=-1)
        assert distances.max() <= 0.05

    def test_halfspace_selects_one_side(self, bar_mesh):
        indices = Nodes.halfspace([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]).resolve(bar_mesh)
        assert bar_mesh.points[indices, 0].min() >= 0.5
        # Interior nodes at x >= 0.5 are excluded: only surface nodes.
        assert boundary_node_mask(bar_mesh)[indices].all()

    def test_side_selects_the_axis_extreme_plane(self, bar_mesh):
        for side, axis, value in (("+x", 0, 1.0), ("-x", 0, -1.0), ("+z", 2, 0.15)):
            indices = Nodes.side(side).resolve(bar_mesh)
            assert np.allclose(bar_mesh.points[indices, axis], value), side

    def test_side_custom_tol_widens_the_capture(self, bar_mesh):
        strict = Nodes.side("+x", tol=1e-6).resolve(bar_mesh)
        wide = Nodes.side("+x", tol=0.15).resolve(bar_mesh)
        assert strict.size < wide.size
        assert np.isin(strict, wide).all()

    def test_predicate_is_vectorized_over_positions(self, bar_mesh):
        indices = Nodes.predicate(lambda points: points[:, 0] > 0.99).resolve(bar_mesh)
        expected = Nodes.side("+x").resolve(bar_mesh)
        assert np.array_equal(indices, expected)

    def test_non_vectorized_predicate_gets_a_clear_error(self, bar_mesh):
        with pytest.raises(ValueError, match="vectorized"):
            Nodes.predicate(lambda points: bool(points[0, 0] > 0.0)).resolve(bar_mesh)

    def test_only_boundary_nodes_are_ever_selected(self, bar_mesh):
        everything = Nodes.predicate(lambda points: np.ones(len(points), dtype=bool))
        indices = everything.resolve(bar_mesh)
        boundary = np.flatnonzero(boundary_node_mask(bar_mesh))
        assert np.array_equal(indices, boundary)
        assert indices.size < bar_mesh.num_points


class TestValidation:
    def test_box_corner_ordering(self):
        with pytest.raises(ValueError, match="min_corner"):
            Nodes.box([1.0, 0.0, 0.0], [0.0, 1.0, 1.0])

    def test_sphere_radius_positive(self):
        with pytest.raises(ValueError, match="radius"):
            Nodes.sphere([0.0, 0.0, 0.0], 0.0)

    def test_halfspace_normal_nonzero(self):
        with pytest.raises(ValueError, match="normal"):
            Nodes.halfspace([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def test_side_name(self):
        with pytest.raises(ValueError, match="side must be one of"):
            Nodes.side("+w")

    def test_side_tol_non_negative(self):
        with pytest.raises(ValueError, match="tol"):
            Nodes.side("+x", tol=-0.1)

    def test_predicate_needs_a_callable(self):
        with pytest.raises(ValueError, match="callable"):
            Nodes.predicate(None)

    def test_combining_with_non_selection_raises(self):
        with pytest.raises(TypeError, match="NodeSelection"):
            Nodes.side("+x") & (lambda points: points)

    def test_empty_selection_raises_on_resolve(self, bar_mesh):
        with pytest.raises(ValueError, match="matched no boundary nodes"):
            Nodes.sphere([50.0, 0.0, 0.0], 0.1).resolve(bar_mesh)

    def test_nodes_namespace_is_not_instantiable(self):
        with pytest.raises(TypeError, match="factory namespace"):
            Nodes()


class TestAlgebra:
    def test_intersection(self, bar_mesh):
        top_of_end = (Nodes.side("+x") & Nodes.halfspace([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])).mask(
            bar_mesh
        )
        expected = Nodes.side("+x").mask(bar_mesh) & (bar_mesh.points[:, 2] >= 0.0)
        assert np.array_equal(top_of_end, expected)

    def test_union(self, bar_mesh):
        both_ends = (Nodes.side("+x") | Nodes.side("-x")).mask(bar_mesh)
        assert np.array_equal(
            both_ends, Nodes.side("+x").mask(bar_mesh) | Nodes.side("-x").mask(bar_mesh)
        )

    def test_complement_stays_on_the_boundary(self, bar_mesh):
        rest = (~Nodes.side("-x")).mask(bar_mesh)
        boundary = boundary_node_mask(bar_mesh)
        assert not (rest & ~boundary).any()
        assert np.array_equal(rest, boundary & ~Nodes.side("-x").mask(bar_mesh))

    def test_de_morgan(self, bar_mesh):
        a, b = Nodes.side("+x"), Nodes.side("+z")
        lhs = (~(a | b)).mask(bar_mesh)
        rhs = (~a & ~b).mask(bar_mesh)
        assert np.array_equal(lhs, rhs)


class TestDescribe:
    def test_primitive_schemas(self):
        assert Nodes.box([0, 0, 0], [1, 1, 1]).describe() == {
            "kind": "box",
            "min_corner": [0.0, 0.0, 0.0],
            "max_corner": [1.0, 1.0, 1.0],
        }
        assert Nodes.sphere([1, 0, 0], 0.5).describe() == {
            "kind": "sphere",
            "center": [1.0, 0.0, 0.0],
            "radius": 0.5,
        }
        assert Nodes.halfspace([0, 0, 0], [0, 0, 1]).describe() == {
            "kind": "halfspace",
            "point": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
        }
        assert Nodes.side("+x").describe() == {"kind": "side", "side": "+x", "tol": None}
        assert Nodes.side("-z", tol=0.02).describe() == {"kind": "side", "side": "-z", "tol": 0.02}

    def test_composite_schema_and_json_round_trip(self):
        selection = (Nodes.side("+x") | Nodes.sphere([0, 0, 0], 1.0)) & ~Nodes.box(
            [0, 0, 0], [1, 1, 1]
        )
        payload = selection.describe()
        assert payload["kind"] == "and"
        assert payload["operands"][0]["kind"] == "or"
        assert payload["operands"][1] == {
            "kind": "not",
            "operand": {
                "kind": "box",
                "min_corner": [0.0, 0.0, 0.0],
                "max_corner": [1.0, 1.0, 1.0],
            },
        }
        assert json.loads(json.dumps(payload)) == payload

    def test_round_trip_through_from_description(self, bar_mesh):
        selection = (Nodes.side("+x", tol=0.02) | Nodes.sphere([1.0, 0.0, 0.0], 0.2)) & ~Nodes.box(
            [0.9, -1.0, -1.0], [1.1, 1.0, 0.0]
        )
        rebuilt = selection_from_description(json.loads(json.dumps(selection.describe())))
        assert rebuilt.describe() == selection.describe()
        assert np.array_equal(rebuilt.mask(bar_mesh), selection.mask(bar_mesh))

    def test_serializable_flag(self):
        def loaded(points):
            return points[:, 0] > 0.99

        assert Nodes.side("+x").serializable
        assert not Nodes.predicate(loaded).serializable
        assert not (Nodes.side("+x") & Nodes.predicate(loaded)).serializable
        assert not (~Nodes.predicate(loaded)).serializable
        assert Nodes.predicate(loaded).describe() == {"kind": "predicate", "name": "loaded"}

    def test_predicate_description_does_not_reconstruct(self):
        payload = Nodes.predicate(lambda points: points[:, 0] > 0.0).describe()
        with pytest.raises(ValueError, match="not\\s+serializable"):
            selection_from_description(payload)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown selection kind"):
            selection_from_description({"kind": "torus"})


class TestFacesFromNodes:
    def test_full_side_selection_spans_the_side_faces(self, bar_mesh):
        nodes = Nodes.side("+x").resolve(bar_mesh)
        group = faces_from_nodes(bar_mesh, nodes)
        by_hand = select_faces(bar_mesh, lambda center: center[0] > 0.99)
        assert {tuple(row) for row in group.nodes} == {tuple(row) for row in by_hand.nodes}

    def test_partial_selection_spans_no_face(self, bar_mesh):
        # A single node spans no complete quad.
        nodes = Nodes.side("+x").resolve(bar_mesh)[:1]
        group = faces_from_nodes(bar_mesh, nodes)
        assert group.nodes.shape == (0, 4)

    def test_composite_selection_derives_a_sub_patch(self, bar_mesh):
        upper = Nodes.side("+x") & Nodes.halfspace([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        group = faces_from_nodes(bar_mesh, upper.resolve(bar_mesh))
        # Top row of the 3x3 end face (both quad corners at z >= 0): 3 quads.
        assert group.nodes.shape[0] == 3
        assert group.centers[:, 2].min() > 0.0


class TestCylinder:
    """`Nodes.cylinder`: an annular, optionally finite selector about a line."""

    def test_full_cylinder_selects_around_the_axis(self, bar_mesh):
        # The bar's own axis, radius just past its half-width: every node.
        everything = Nodes.cylinder([0, 0, 0], [1, 0, 0], 0.30).resolve(bar_mesh)
        assert everything.size == boundary_node_mask(bar_mesh).sum()
        # Radius under the half-width: only the nodes near the axis line.
        near = Nodes.cylinder([0, 0, 0], [1, 0, 0], 0.16).resolve(bar_mesh)
        radial = np.linalg.norm(bar_mesh.points[near][:, 1:], axis=-1)
        assert radial.max() <= 0.16
        assert 0 < near.size < everything.size

    def test_annulus_excludes_the_core(self, bar_mesh):
        ring = Nodes.cylinder([0, 0, 0], [1, 0, 0], 0.30, inner=0.16).resolve(bar_mesh)
        radial = np.linalg.norm(bar_mesh.points[ring][:, 1:], axis=-1)
        assert radial.min() >= 0.16
        full = Nodes.cylinder([0, 0, 0], [1, 0, 0], 0.30).resolve(bar_mesh)
        core = Nodes.cylinder([0, 0, 0], [1, 0, 0], 0.16).resolve(bar_mesh)
        assert ring.size == full.size - core.size

    def test_half_length_bounds_the_axial_extent(self, bar_mesh):
        slab = Nodes.cylinder([0.5, 0, 0], [1, 0, 0], 0.30, half_length=0.25).resolve(bar_mesh)
        assert bar_mesh.points[slab, 0].min() >= 0.25 - 1e-9
        assert bar_mesh.points[slab, 0].max() <= 0.75 + 1e-9

    def test_axis_need_not_be_unit_or_aligned(self, bar_mesh):
        scaled = Nodes.cylinder([0, 0, 0], [3, 0, 0], 0.16).resolve(bar_mesh)
        unit = Nodes.cylinder([0, 0, 0], [1, 0, 0], 0.16).resolve(bar_mesh)
        assert np.array_equal(scaled, unit)

    def test_validation(self):
        with pytest.raises(ValueError, match="axis"):
            Nodes.cylinder([0, 0, 0], [0, 0, 0], 1.0)
        with pytest.raises(ValueError, match="radius"):
            Nodes.cylinder([0, 0, 0], [0, 0, 1], -1.0)
        with pytest.raises(ValueError, match="inner"):
            Nodes.cylinder([0, 0, 0], [0, 0, 1], 1.0, inner=1.0)
        with pytest.raises(ValueError, match="half_length"):
            Nodes.cylinder([0, 0, 0], [0, 0, 1], 1.0, half_length=0.0)

    def test_describe_round_trips(self, bar_mesh):
        selection = Nodes.cylinder([0.5, 0, 0], [1, 0, 0], 0.30, inner=0.16, half_length=0.25)
        payload = json.loads(json.dumps(selection.describe()))
        assert payload == {
            "kind": "cylinder",
            "center": [0.5, 0.0, 0.0],
            "axis": [1.0, 0.0, 0.0],
            "radius": 0.30,
            "inner": 0.16,
            "half_length": 0.25,
        }
        rebuilt = selection_from_description(payload)
        assert selection.serializable
        assert np.array_equal(rebuilt.resolve(bar_mesh), selection.resolve(bar_mesh))
        # The optional fields default when absent, as an older payload would omit them.
        plain = selection_from_description(
            {"kind": "cylinder", "center": [0, 0, 0], "axis": [1, 0, 0], "radius": 0.3}
        )
        assert plain.describe()["inner"] == 0.0 and plain.describe()["half_length"] is None
