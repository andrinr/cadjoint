"""Tests for per-primitive patch fields and scene surface signatures.

The protocol under test: every hard primitive decomposes into smooth patch
fields whose ``argmin |field|`` ownership is exact on the surface, feature
edges are exactly where ownership switches, and transforms forward the
decomposition by mapping queries into the child frame exactly as their sdf
does.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.geometry.parameters import Vector2
from cadjoint.meshing import (
    GridSpec,
    exact_feature_mask,
    extract_mesh,
    patch_signatures,
    scene_patch_fields,
    signature_function,
    world_frame_leaves,
)
from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import (
    Box,
    Cylinder,
    ExtrudedPolygon,
    RevolvedPolygon,
    Sphere,
    Torus,
)
from cadjoint.sdf.transforms import Rotate, Scale, Translate, Twist

# Box patch order is [+x, -x, +y, -y, +z, -z]: index 2*axis + side.
BOX_SIZE = np.array([0.8, 0.6, 0.5])

# The playground house profile: a convex pentagon, counterclockwise.
HOUSE_VERTICES = [
    [-1.1, -0.7],
    [1.1, -0.7],
    [1.1, 0.3],
    [0.0, 1.0],
    [-1.1, 0.3],
]
HOUSE_DEPTH = 1.2


def _house() -> ExtrudedPolygon:
    return ExtrudedPolygon([jnp.array(v) for v in HOUSE_VERTICES], depth=HOUSE_DEPTH)


@contextlib.contextmanager
def _vertices_set(polygon, values):
    """Swap a polygon's profile vertices for ``values`` (tracers welcome)."""
    names = [f"v{i}" for i in range(polygon.num_vertices)]
    original = [polygon.params[name].value for name in names]
    for index, name in enumerate(names):
        polygon.params[name].value = values[index]
    try:
        yield
    finally:
        for name, value in zip(names, original):
            polygon.params[name].value = value


def _patch_id(fields, point) -> int:
    magnitudes = jnp.stack([jnp.abs(field(jnp.asarray(point))) for field in fields])
    return int(jnp.argmin(magnitudes))


def _box_face_points(face: int, count: int = 5, margin: float = 0.05) -> np.ndarray:
    """A grid of points on the interior of one box face.

    ``margin`` keeps samples away from the face's bounding edges, where
    ownership legitimately switches.
    """
    axis, side = divmod(face, 2)
    sign = 1.0 if side == 0 else -1.0
    other = [a for a in range(3) if a != axis]
    spans = [np.linspace(-BOX_SIZE[a] + margin, BOX_SIZE[a] - margin, count) for a in other]
    grid_u, grid_v = np.meshgrid(*spans, indexing="ij")
    points = np.zeros((count * count, 3))
    points[:, axis] = sign * BOX_SIZE[axis]
    points[:, other[0]] = grid_u.ravel()
    points[:, other[1]] = grid_v.ravel()
    return points


class TestBoxPatchFields:
    def test_six_fields(self):
        assert len(Box(size=jnp.asarray(BOX_SIZE)).patch_fields()) == 6

    @pytest.mark.parametrize("face", range(6))
    def test_face_ownership(self, face):
        fields = Box(size=jnp.asarray(BOX_SIZE)).patch_fields()
        for point in _box_face_points(face):
            assert _patch_id(fields, point) == face
            # The owning field is exactly the face plane distance: zero here.
            assert float(fields[face](jnp.asarray(point))) == pytest.approx(0.0, abs=1e-6)

    def test_max_composition_reproduces_interior_sdf(self):
        """Inside/on the box, max over the six fields is exactly the sdf."""
        box = Box(size=jnp.asarray(BOX_SIZE))
        fields = box.patch_fields()
        rng = np.random.default_rng(0)
        points = jnp.asarray(rng.uniform(-1.0, 1.0, size=(64, 3)) * BOX_SIZE)
        stacked = jnp.stack([field(points) for field in fields])
        composed = jnp.max(stacked, axis=0)
        reference = box(points)
        inside = np.asarray(reference) <= 0.0
        np.testing.assert_allclose(
            np.asarray(composed)[inside], np.asarray(reference)[inside], atol=1e-6
        )


class TestTransformForwarding:
    AXIS = jnp.array([0.3, 1.0, -0.2])
    ANGLE = 0.7
    OFFSET = jnp.array([1.5, -0.4, 2.0])

    def _world(self, local: np.ndarray) -> jnp.ndarray:
        rotation = Rotate._rotation_matrix(self.AXIS, self.ANGLE)
        return jnp.einsum("ij,nj->ni", rotation, jnp.asarray(local, jnp.float32)) + self.OFFSET

    def test_rotated_translated_box_stays_exact(self):
        shape = Translate(
            Rotate(Box(size=jnp.asarray(BOX_SIZE)), self.AXIS, self.ANGLE), self.OFFSET
        )
        fields = shape.patch_fields()
        assert len(fields) == 6
        for face in range(6):
            world = self._world(_box_face_points(face))
            for point in world:
                assert _patch_id(fields, point) == face
                # Exactness through the transform plumbing: the owning field
                # still evaluates to the face-plane distance, zero on-face.
                assert float(fields[face](point)) == pytest.approx(0.0, abs=1e-5)

    def test_uniform_scale_forwards_and_rescales(self):
        shape = Scale(Box(size=jnp.asarray(BOX_SIZE)), 2.0)
        fields = shape.patch_fields()
        assert len(fields) == 6
        point = jnp.array([2.0 * BOX_SIZE[0], 0.0, 0.0])
        assert _patch_id(fields, point) == 0
        assert float(fields[0](point)) == pytest.approx(0.0, abs=1e-6)
        # Distances stay metric: one unit outside the +x face reads 1.
        assert float(fields[0](point + jnp.array([1.0, 0.0, 0.0]))) == pytest.approx(1.0, 1e-5)

    def test_nonuniform_scale_reports_none(self):
        assert (
            Scale(Box(size=jnp.asarray(BOX_SIZE)), jnp.array([1.0, 2.0, 3.0])).patch_fields()
            is None
        )

    def test_unsupported_transform_reports_none(self):
        assert Twist(Box(size=jnp.asarray(BOX_SIZE)), 1.0).patch_fields() is None

    def test_transform_of_unsupported_child_reports_none(self):
        loft_like = Twist(Sphere(1.0), 1.0)
        assert Translate(loft_like, self.OFFSET).patch_fields() is None


class TestCylinderPatchFields:
    RADIUS = 0.5
    HEIGHT = 0.7

    def test_side_and_caps(self):
        fields = Cylinder(radius=self.RADIUS, height=self.HEIGHT).patch_fields()
        assert len(fields) == 3
        for angle in np.linspace(0.0, 2.0 * np.pi, 9):
            side = [self.RADIUS * np.cos(angle), self.RADIUS * np.sin(angle), 0.3]
            assert _patch_id(fields, side) == 0
        assert _patch_id(fields, [0.2, 0.1, self.HEIGHT]) == 1
        assert _patch_id(fields, [0.2, 0.1, -self.HEIGHT]) == 2

    def test_ownership_switches_at_rim(self):
        fields = Cylinder(radius=self.RADIUS, height=self.HEIGHT).patch_fields()
        eps = 1e-3
        # Just below the rim on the side wall vs just inside the top cap.
        assert _patch_id(fields, [self.RADIUS, 0.0, self.HEIGHT - 2 * eps]) == 0
        assert _patch_id(fields, [self.RADIUS - 2 * eps, 0.0, self.HEIGHT]) == 1


class TestSmoothPrimitives:
    def test_single_patch(self):
        assert len(Sphere(1.0).patch_fields()) == 1
        assert len(Torus(1.0, 0.3).patch_fields()) == 1


class TestExtrudedPolygonPatchFields:
    def test_edge_ids_match_nearest_profile_edge(self):
        house = _house()
        fields = house.patch_fields()
        vertices = np.asarray(HOUSE_VERTICES)
        count = len(vertices)
        assert len(fields) == count + 2
        for k in range(count):
            a, b = vertices[k], vertices[(k + 1) % count]
            for t in (0.2, 0.5, 0.8):
                for z in (-0.4, 0.0, 0.4):
                    xy = a + t * (b - a)
                    point = [xy[0], xy[1], z]
                    assert _patch_id(fields, point) == k
                    assert float(fields[k](jnp.asarray(point))) == pytest.approx(0.0, abs=1e-6)

    def test_cap_ids(self):
        fields = _house().patch_fields()
        count = len(HOUSE_VERTICES)
        assert _patch_id(fields, [0.0, 0.0, -HOUSE_DEPTH / 2]) == count
        assert _patch_id(fields, [0.0, 0.0, HOUSE_DEPTH / 2]) == count + 1

    def test_winding_independent(self):
        reversed_house = ExtrudedPolygon(
            [jnp.array(v) for v in reversed(HOUSE_VERTICES)], depth=HOUSE_DEPTH
        )
        fields = reversed_house.patch_fields()
        # Outward fields stay positive outside regardless of input winding.
        outside = jnp.array([2.0, -0.2, 0.0])
        assert (
            min(float(f(outside)) for f in fields[:-2])
            < 0.0
            < max(float(f(outside)) for f in fields[:-2])
        )
        assert float(jnp.max(jnp.stack([f(outside) for f in fields]))) > 0.0

    def test_draft_and_twist_report_none(self):
        verts = [jnp.array(v) for v in HOUSE_VERTICES]
        assert ExtrudedPolygon(verts, depth=1.0, draft=5.0).patch_fields() is None
        assert ExtrudedPolygon(verts, depth=1.0, twist=30.0).patch_fields() is None

    def test_jacrev_through_a_traced_profile_matches_finite_differences(self):
        """Patch fields rebuilt with the sketch vertices traced stay differentiable.

        The B-rep handle solver re-reads ``patch_fields()`` with the profile's
        parameters swapped for tracers (``cadjoint.brep.drag.patch_field_fn``).
        The only discrete reading in the rebuild — the profile's shoelace
        winding — is taken once at construction, so nothing inside needs a
        concrete value and ``jax.jacrev`` goes straight through.
        """
        house = ExtrudedPolygon(
            [Vector2(value=list(vertex)) for vertex in HOUSE_VERTICES], depth=HOUSE_DEPTH
        )
        point = jnp.asarray([0.7, 0.1, 0.2])

        def walls(profile):
            with _vertices_set(house, profile):
                fields = house.patch_fields()
            return jnp.stack([jnp.reshape(field(point), ()) for field in fields])

        nominal = jnp.asarray(HOUSE_VERTICES)
        analytic = np.asarray(jax.jacrev(walls)(nominal))
        assert analytic.shape == (len(HOUSE_VERTICES) + 2, len(HOUSE_VERTICES), 2)
        assert np.isfinite(analytic).all()
        assert np.abs(analytic).max() > 0.1, "the walls must actually move with the profile"

        # float32 pipeline: 1e-3 is the sweet spot between truncation and
        # cancellation for these (almost linear) half-plane fields.
        step = 1e-3
        numeric = np.zeros_like(analytic)
        for vertex in range(len(HOUSE_VERTICES)):
            for axis in range(2):
                shift = np.zeros((len(HOUSE_VERTICES), 2))
                shift[vertex, axis] = step
                plus = np.asarray(walls(nominal + jnp.asarray(shift)))
                minus = np.asarray(walls(nominal - jnp.asarray(shift)))
                numeric[:, vertex, axis] = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(analytic, numeric, atol=2e-3, rtol=2e-3)


class TestRevolvedPolygonPatchFields:
    def test_edge_ids_in_radial_height_coords(self):
        square = [jnp.array(v) for v in [[1.0, -0.5], [2.0, -0.5], [2.0, 0.5], [1.0, 0.5]]]
        rev = RevolvedPolygon(square)
        fields = rev.patch_fields()
        assert len(fields) == 4
        for angle in (0.0, 1.1, 2.7):
            c, s = np.cos(angle), np.sin(angle)
            # Outer wall (radial = 2) is edge 1; inner wall (radial = 1) is edge 3.
            assert _patch_id(fields, [2.0 * c, 0.0, 2.0 * s]) == 1
            assert _patch_id(fields, [1.0 * c, 0.0, 1.0 * s]) == 3
            # Bottom (height = -0.5) is edge 0; top is edge 2.
            assert _patch_id(fields, [1.5 * c, -0.5, 1.5 * s]) == 0
            assert _patch_id(fields, [1.5 * c, 0.5, 1.5 * s]) == 2


class TestSceneSignatures:
    def _scene(self):
        return Union(
            (
                Box(size=jnp.asarray(BOX_SIZE)),
                Translate(Sphere(0.5), jnp.array([2.0, 0.0, 0.0])),
            )
        )

    def test_leaf_and_patch_ids(self):
        scene = self._scene()
        decomposition = scene_patch_fields(scene)
        assert decomposition.leaf_ids == [0, 1]
        assert decomposition.exact == [True, True]
        assert len(world_frame_leaves(scene)) == 2
        points = np.array(
            [
                [BOX_SIZE[0], 0.0, 0.0],  # box +x face
                [0.0, -BOX_SIZE[1], 0.0],  # box -y face
                [2.5, 0.0, 0.0],  # sphere surface
            ]
        )
        leaf_ids, patch_ids = patch_signatures(scene, points)
        np.testing.assert_array_equal(leaf_ids, [0, 0, 1])
        np.testing.assert_array_equal(patch_ids, [0, 3, 0])

    def test_fallback_leaf_is_single_opaque_patch(self):
        scene = Union((Box(size=jnp.asarray(BOX_SIZE)), Twist(Sphere(0.5), 1.0)))
        decomposition = scene_patch_fields(scene)
        assert decomposition.exact == [True, False]
        assert len(decomposition.fields[1]) == 1

    def test_signature_constant_on_face_interiors(self):
        """Dense on-face sampling: the signature changes NOWHERE on a face."""
        scene = Box(size=jnp.asarray(BOX_SIZE))
        for face in range(6):
            points = _box_face_points(face, count=17, margin=1e-3)
            leaf_ids, patch_ids = patch_signatures(scene, points)
            assert np.all(leaf_ids == 0)
            assert np.all(patch_ids == face), f"face {face} interior not uniform"

    def test_signature_changes_exactly_across_analytic_edges(self):
        """Point pairs straddling each box edge flip patch id; parallel pairs don't."""
        scene = Box(size=jnp.asarray(BOX_SIZE))
        eps = 1e-4
        sx, sy, sz = BOX_SIZE
        # Straddle the +x/+y edge on the surface: one point on each face.
        on_x = [[sx, sy - eps, z] for z in np.linspace(-sz + 0.05, sz - 0.05, 7)]
        on_y = [[sx - eps, sy, z] for z in np.linspace(-sz + 0.05, sz - 0.05, 7)]
        _, patch_x = patch_signatures(scene, np.asarray(on_x))
        _, patch_y = patch_signatures(scene, np.asarray(on_y))
        assert np.all(patch_x == 0)
        assert np.all(patch_y == 2)
        # Pairs along the same face (same offsets, no edge between them) agree.
        along = [[sx, sy - eps, 0.0], [sx, sy - 0.3, 0.0]]
        _, patch_along = patch_signatures(scene, np.asarray(along))
        assert patch_along[0] == patch_along[1] == 0

    def test_vmap_and_jit(self):
        scene = self._scene()
        signature = signature_function(scene)
        points = jnp.asarray(
            [[BOX_SIZE[0], 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, BOX_SIZE[1], 0.0]],
            dtype=jnp.float32,
        )
        eager = jax.vmap(signature)(points)
        jitted = jax.jit(jax.vmap(signature))(points)
        np.testing.assert_array_equal(np.asarray(eager[0]), np.asarray(jitted[0]))
        np.testing.assert_array_equal(np.asarray(eager[1]), np.asarray(jitted[1]))
        np.testing.assert_array_equal(np.asarray(jitted[0]), [0, 1, 0])
        np.testing.assert_array_equal(np.asarray(jitted[1]), [0, 0, 2])

    def test_exact_feature_mask(self):
        leaf_ids = np.array([0, 0, 1, 0])
        patch_ids = np.array([0, 2, 0, 0])
        adjacency = np.array([[0, 1], [0, 2], [0, 3], [1, 1]])
        mask = exact_feature_mask(leaf_ids, patch_ids, adjacency)
        np.testing.assert_array_equal(mask, [True, True, False, False])


def _house_analytic_edges() -> list[tuple[np.ndarray, np.ndarray]]:
    """Every analytic feature segment of the extruded house pentagon.

    Vertical edges at each profile vertex, plus the two cap rims (one
    segment per profile edge at each cap plane).
    """
    vertices = np.asarray(HOUSE_VERTICES)
    count = len(vertices)
    half = HOUSE_DEPTH / 2.0
    segments = []
    for k in range(count):
        x, y = vertices[k]
        segments.append((np.array([x, y, -half]), np.array([x, y, half])))
        nxt = vertices[(k + 1) % count]
        for z in (-half, half):
            segments.append((np.array([*vertices[k], z]), np.array([*nxt, z])))
    return segments


def _distance_to_segments(points: np.ndarray, segments) -> np.ndarray:
    """Min distance from each point to a set of 3D segments."""
    best = np.full(points.shape[0], np.inf)
    for start, end in segments:
        direction = end - start
        t = np.clip((points - start) @ direction / float(direction @ direction), 0.0, 1.0)
        closest = start + t[:, None] * direction
        best = np.minimum(best, np.linalg.norm(points - closest, axis=1))
    return best


def _segment_cells(grid: GridSpec, segments) -> np.ndarray:
    """Lattice indices of every grid cell an analytic segment passes through.

    Dense sampling at a fraction of the smallest spacing cannot skip cells.
    """
    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    cells = []
    for start, end in segments:
        steps = int(np.ceil(np.linalg.norm(end - start) / (0.25 * spacing.min()))) + 1
        samples = start + np.linspace(0.0, 1.0, steps)[:, None] * (end - start)
        cells.append(np.floor((samples - origin) / spacing).astype(np.int64))
    return np.unique(np.concatenate(cells), axis=0)


class TestHouseDemonstration:
    """Signature-based edge cells on the example house match the analytic edges."""

    def test_signature_edges_match_analytic_profile_edges(self):
        house = _house()
        grid = GridSpec.from_bounds((-1.6, -1.2, -1.0), (3.2, 2.6, 2.0), 26)
        mesh = extract_mesh(lambda p: jnp.asarray(house(p)), grid)
        adjacency = np.unique(
            np.sort(
                np.concatenate(
                    [
                        mesh.quads[:, [0, 1]],
                        mesh.quads[:, [1, 2]],
                        mesh.quads[:, [2, 3]],
                        mesh.quads[:, [3, 0]],
                    ]
                ),
                axis=1,
            ),
            axis=0,
        )
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        leaf_ids, patch_ids = patch_signatures(house, vertices)
        assert np.all(leaf_ids == 0)
        mask = exact_feature_mask(leaf_ids, patch_ids, adjacency)
        assert mask.any(), "the house must produce signature-change edges"

        segments = _house_analytic_edges()
        curve_cells = _segment_cells(grid, segments)

        def chebyshev_to_curve(cells: np.ndarray) -> np.ndarray:
            return np.min(
                np.max(np.abs(cells[:, None, :] - curve_cells[None, :, :]), axis=2), axis=1
            )

        # Exactness, direction 1: every signature-change adjacency locates
        # the analytic curve within one cell — the endpoint on the curve's
        # side sits in a cell the curve passes through or its immediate
        # neighbor (the far endpoint legitimately lies one cell further).
        flagged_edges = adjacency[mask]
        near = np.minimum(
            chebyshev_to_curve(mesh.cells[flagged_edges[:, 0]]),
            chebyshev_to_curve(mesh.cells[flagged_edges[:, 1]]),
        )
        assert (
            int(near.max()) <= 1
        ), f"a signature edge strays {near.max()} cells from the analytic edges"

        # Direction 2: every analytic-curve cell is matched — it has a
        # signature-change vertex within one cell.
        flagged_cells = mesh.cells[np.unique(flagged_edges)]
        coverage = np.min(
            np.max(np.abs(curve_cells[:, None, :] - flagged_cells[None, :, :]), axis=2),
            axis=1,
        )
        assert int(coverage.max()) <= 1, f"analytic edge cell uncovered by {coverage.max()} cells"

        # Geometrically, the on-curve endpoint of each flagged adjacency
        # hugs the analytic curve: sharp placement lands most exactly on it,
        # and even where a feature plane coincides with a lattice plane (the
        # base at y = -0.7 here) the endpoint stays within half a cell.
        tolerance = 0.5 * float(max(grid.spacing))
        residuals = np.minimum(
            _distance_to_segments(vertices[flagged_edges[:, 0]], segments),
            _distance_to_segments(vertices[flagged_edges[:, 1]], segments),
        )
        assert float(residuals.max()) <= tolerance

        # And nowhere else: unflagged adjacencies keep one signature, so any
        # edge fully interior to a face never triggers.
        interior = adjacency[~mask]
        assert np.all(patch_ids[interior[:, 0]] == patch_ids[interior[:, 1]])
