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
from cadjoint.sdf.boolean import Difference, Intersection, Union
from cadjoint.sdf.primitives import (
    Box,
    Cylinder,
    ExtrudedPolygon,
    RevolvedPolygon,
    Sphere,
    Torus,
)
from cadjoint.sdf.transforms import Rotate, Scale, Translate, Twist
from cadjoint.sdf.transforms.fields import Mirror, Shell
from cadjoint.sdf.transforms.patterns import LinearPattern, PolarPattern

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

    def test_twist_reports_none(self):
        """A twisted wall is a helicoid — none of the surfaces a fitter knows."""
        verts = [jnp.array(v) for v in HOUSE_VERTICES]
        assert ExtrudedPolygon(verts, depth=1.0, twist=30.0).patch_fields() is None
        # Twist decides on its own, whatever the draft alongside it does.
        assert ExtrudedPolygon(verts, depth=1.0, draft=5.0, twist=30.0).patch_fields() is None


class TestDraftedExtrusionPatchFields:
    """A draft only tilts the walls, so they stay planes and stay declared."""

    DRAFT = 8.0

    def _drafted(self) -> ExtrudedPolygon:
        return ExtrudedPolygon(
            [jnp.array(v) for v in HOUSE_VERTICES], depth=HOUSE_DEPTH, draft=self.DRAFT
        )

    def _wall_point(self, edge: int, t: float, z: float) -> np.ndarray:
        """A point on drafted wall ``edge``, built from the draft's own geometry.

        The profile is exact at ``z = -depth/2``; above it the wall has moved
        *inward* along the edge's outward normal by ``tan(δ)·(z + depth/2)``,
        which is what offsetting the 2D distance by that amount means.
        """
        vertices = np.asarray(HOUSE_VERTICES)
        a, b = vertices[edge], vertices[(edge + 1) % len(vertices)]
        direction = b - a
        # HOUSE_VERTICES wind counterclockwise, so the outward normal is the
        # edge direction turned by -90 degrees.
        outward = np.array([direction[1], -direction[0]]) / np.linalg.norm(direction)
        inset = np.tan(np.radians(self.DRAFT)) * (z + HOUSE_DEPTH / 2.0)
        xy = a + t * direction - inset * outward
        return np.array([xy[0], xy[1], z])

    def test_declares_walls_and_caps(self):
        assert len(self._drafted().patch_fields()) == len(HOUSE_VERTICES) + 2

    def test_each_wall_field_vanishes_on_its_drafted_wall(self):
        shape = self._drafted()
        fields = shape.patch_fields()
        for edge in range(len(HOUSE_VERTICES)):
            for t in (0.3, 0.5, 0.7):
                # Strictly between the caps: on a cap rim the wall and the cap
                # both vanish, and ownership there is a tie by construction.
                for z in (-0.55, -0.2, 0.0, 0.55):
                    point = jnp.asarray(self._wall_point(edge, t, z), jnp.float32)
                    # The sample really is on the drafted solid's surface...
                    assert float(shape(point)) == pytest.approx(0.0, abs=1e-6)
                    # ...its own wall's field vanishes there...
                    assert float(fields[edge](point)) == pytest.approx(0.0, abs=1e-6)
                    # ...and no other patch claims it.
                    assert _patch_id(fields, point) == edge

    def test_cap_ids_are_unchanged_by_the_draft(self):
        """A draft tapers the walls; the caps stay the planes they were."""
        fields = self._drafted().patch_fields()
        count = len(HOUSE_VERTICES)
        assert _patch_id(fields, [0.0, 0.0, -HOUSE_DEPTH / 2]) == count
        assert _patch_id(fields, [0.0, 0.0, HOUSE_DEPTH / 2]) == count + 1

    def test_wall_fields_are_planes_with_unit_gradient(self):
        """The point of the ``cos(δ)`` scaling, pinned.

        Untouched, a drafted wall field reads ``sec(δ)`` times the true
        distance — it would then win ``argmin |f_i|`` against neighbours it
        should lose to.  Scaled, its gradient is a unit vector, and being
        *constant* over the wall is what makes it a plane rather than merely
        a surface through the right points.
        """
        fields = self._drafted().patch_fields()
        gradient = jax.grad(lambda p, f=fields[0]: jnp.reshape(f(p), ()))
        first = np.asarray(gradient(jnp.asarray(self._wall_point(0, 0.4, 0.0), jnp.float32)))
        assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-6)
        # The tilt is the draft angle, off vertical, about the profile edge.
        assert float(first[2]) == pytest.approx(np.sin(np.radians(self.DRAFT)), abs=1e-6)
        for wall in range(len(HOUSE_VERTICES)):
            gradient = jax.grad(lambda p, f=fields[wall]: jnp.reshape(f(p), ()))
            for t, z in ((0.2, -0.5), (0.8, 0.5)):
                other = np.asarray(gradient(jnp.asarray(self._wall_point(wall, t, z), jnp.float32)))
                assert np.linalg.norm(other) == pytest.approx(1.0, abs=1e-6)

    def test_zero_draft_measured_distances_are_metric(self):
        """One unit outside a drafted wall reads one unit, not ``sec(δ)``."""
        fields = self._drafted().patch_fields()
        surface = self._wall_point(0, 0.5, 0.1)
        normal = np.asarray(
            jax.grad(lambda p, f=fields[0]: jnp.reshape(f(p), ()))(
                jnp.asarray(surface, jnp.float32)
            )
        )
        for distance in (0.1, 0.5, 1.0):
            probe = jnp.asarray(surface + distance * normal, jnp.float32)
            assert float(fields[0](probe)) == pytest.approx(distance, abs=1e-5)

    def test_composition_agrees_in_sign_but_not_in_distance(self):
        """The documented cost of normalising: same verdict, rescaled wall term.

        ``sdf`` composes the *unscaled* wall term, so ``max_i f_i`` is not
        the drafted sdf value — it is ``cos(δ)`` times it wherever a wall
        dominates.  Inside/outside still agree everywhere, which is what
        makes the zero sets the same surface.
        """
        shape = self._drafted()
        fields = shape.patch_fields()
        rng = np.random.default_rng(11)
        points = jnp.asarray(rng.uniform(-1.4, 1.4, size=(256, 3)))
        composed = np.asarray(jnp.max(jnp.stack([field(points) for field in fields]), axis=0))
        reference = np.asarray(shape(points))
        assert np.all(np.sign(composed) == np.sign(reference))
        assert not np.allclose(composed, reference, atol=1e-3), "expected the cos(δ) rescale"

    def test_jacrev_through_a_traced_profile_matches_finite_differences(self):
        """Patch fields rebuilt with the sketch vertices traced stay differentiable.

        The private tier's handle solver (the ``drag`` plugin kind) re-reads
        ``patch_fields()`` with the profile's parameters swapped for tracers.
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


class _Plane:
    """The duck type ``Mirror`` accepts as a mirror plane: origin + normal.

    ``cadjoint.sdf`` must not import ``cadjoint.construction``, so a mirror
    plane is read off whatever carries the two attributes — a ``Face``, a
    ``SketchPlane``, or this.  Used here to mirror across a plane that is
    *not* through the origin, which is the case the ``origin`` term exists
    for and the one a named axis cannot express.
    """

    def __init__(self, origin, normal):
        self.origin = jnp.asarray(origin, jnp.float32)
        self.normal = jnp.asarray(normal, jnp.float32)


# The pin the pattern tests copy: small enough that neighbouring instances
# stay disjoint at the spacings below.  Its patch order is the cylinder's own:
# [side, +z cap, -z cap].
PIN_RADIUS = 0.15
PIN_HEIGHT = 0.25
PIN_PATCHES = 3


def _pin() -> Cylinder:
    return Cylinder(radius=PIN_RADIUS, height=PIN_HEIGHT)


def _pin_patch_points() -> np.ndarray:
    """One interior surface point per patch of ``_pin()``, in its own frame.

    Row ``j`` sits on patch ``j`` and nowhere near patch ``j``'s rims, so
    ownership there is unambiguous for the seed copy.
    """
    return np.array(
        [
            [PIN_RADIUS, 0.0, 0.3 * PIN_HEIGHT],
            [0.4 * PIN_RADIUS, 0.0, PIN_HEIGHT],
            [0.4 * PIN_RADIUS, 0.0, -PIN_HEIGHT],
        ]
    )


def _rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``(..., 3)`` points by ``angle`` about the world z axis."""
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(points, dtype=float) @ matrix.T


class TestMirrorPatchFields:
    """A mirror reflects only, so it forwards the child's patches one for one."""

    OFFSET = jnp.array([0.9, 0.2, -0.3])

    def _child(self) -> Translate:
        return Translate(Box(size=jnp.asarray(BOX_SIZE)), self.OFFSET)

    def _reflected(self, face: int, plane_x: float) -> np.ndarray:
        """Points on one box face, carried into the mirrored copy's frame."""
        world = np.asarray(_box_face_points(face)) + np.asarray(self.OFFSET)
        world[:, 0] = 2.0 * plane_x - world[:, 0]
        return world

    def test_count_is_unchanged(self):
        assert len(Mirror(self._child(), "x").patch_fields()) == 6

    def test_reflected_faces_keep_the_child_patch_ids(self):
        """On the mirrored surface, the child's own patch still owns the point."""
        shape = Mirror(self._child(), "x")
        fields = shape.patch_fields()
        for face in range(6):
            for point in self._reflected(face, 0.0):
                q = jnp.asarray(point, jnp.float32)
                # The sample really is on this node's zero set...
                assert float(shape(q)) == pytest.approx(0.0, abs=1e-5)
                # ...and the decomposition covers it, with the child's id.
                assert _patch_id(fields, q) == face
                assert float(fields[face](q)) == pytest.approx(0.0, abs=1e-5)

    def test_mirror_plane_origin_is_honoured(self):
        """A plane off the origin moves the copy, and the fields with it."""
        plane_x = 1.4
        shape = Mirror(self._child(), _Plane([plane_x, 0.0, 0.0], [1.0, 0.0, 0.0]))
        fields = shape.patch_fields()
        for face in range(6):
            for point in self._reflected(face, plane_x):
                q = jnp.asarray(point, jnp.float32)
                assert float(shape(q)) == pytest.approx(0.0, abs=1e-5)
                assert _patch_id(fields, q) == face

    def test_ownership_switches_at_the_mirrored_edges(self):
        """Across a mirrored box edge the id switches; along a face it does not.

        The ids are the *child's*: the reflected +x face still reports patch
        0, because a mirror renames nothing.
        """
        shape = Mirror(self._child(), "x")
        fields = shape.patch_fields()
        eps = 1e-3
        sx, sy, _ = BOX_SIZE
        on_x = np.array([[sx, sy - eps, 0.0], [sx, sy - 0.2, 0.0]]) + np.asarray(self.OFFSET)
        on_y = np.array([[sx - eps, sy, 0.0]]) + np.asarray(self.OFFSET)
        on_x[:, 0] *= -1.0
        on_y[:, 0] *= -1.0
        assert [_patch_id(fields, p) for p in on_x] == [0, 0]
        assert [_patch_id(fields, p) for p in on_y] == [2]

    def test_child_without_a_decomposition_reports_none(self):
        assert Mirror(Twist(Sphere(1.0), 1.0), "x").patch_fields() is None

    def test_plain_callable_child_reports_none(self):
        """A bare lambda declares no patches, so the mirror has none to forward."""
        assert Mirror(lambda p: jnp.linalg.norm(p, axis=-1) - 1.0, "x").patch_fields() is None


class TestLinearPatternPatchFields:
    """``n_kept * child`` fields, instance-major, in kept order."""

    # A diagonal direction on purpose: patterning a cylinder along x would
    # leave every copy's two cap half-spaces *coincident* (they depend on z
    # alone), and ownership between identical fields is not a meaningful
    # thing to assert.  Along [1, 0, 1] no two instances share a patch.
    DIRECTION = jnp.array([1.0, 0.0, 1.0])
    SPACING = 0.6
    COUNT = 4
    SKIP = (2,)
    KEPT = (0, 1, 3)

    def _pattern(self, skip=SKIP) -> LinearPattern:
        return LinearPattern(_pin(), self.DIRECTION, self.COUNT, self.SPACING, skip=skip)

    def _instance_points(self, instance: int) -> np.ndarray:
        step = np.asarray(self.DIRECTION) / np.linalg.norm(np.asarray(self.DIRECTION))
        return _pin_patch_points() + self.SPACING * instance * step

    def test_field_count_is_kept_instances_times_child(self):
        assert len(self._pattern().patch_fields()) == len(self.KEPT) * PIN_PATCHES
        assert len(self._pattern(skip=()).patch_fields()) == self.COUNT * PIN_PATCHES

    def test_instance_major_order(self):
        """Slot ``n``, child patch ``j`` is index ``3n + j`` — not ``3j + n``."""
        pattern = self._pattern()
        fields = pattern.patch_fields()
        for slot, instance in enumerate(self.KEPT):
            for patch, point in enumerate(self._instance_points(instance)):
                q = jnp.asarray(point, jnp.float32)
                assert float(pattern(q)) == pytest.approx(0.0, abs=1e-6)
                assert _patch_id(fields, q) == PIN_PATCHES * slot + patch
                index = PIN_PATCHES * slot + patch
                assert float(fields[index](q)) == pytest.approx(0.0, abs=1e-6)

    def test_seed_instance_is_the_child_untouched(self):
        """Instance 0 is not displaced, so its fields are the child's own."""
        pattern = self._pattern()
        fields = pattern.patch_fields()
        child_fields = _pin().patch_fields()
        for point in ([0.05, -0.02, 0.1], [0.4, 0.3, -0.2]):
            q = jnp.asarray(point, jnp.float32)
            for patch, child_field in enumerate(child_fields):
                assert float(fields[patch](q)) == float(child_field(q))

    def test_a_suppressed_instance_contributes_no_fields(self):
        """The declared count follows the *kept* instances, not ``count``."""
        skipped, full = self._pattern(), self._pattern(skip=())
        point = jnp.asarray(self._instance_points(2)[0], jnp.float32)
        # Instance 2's surface is a patch of the full pattern...
        assert min(abs(float(f(point))) for f in full.patch_fields()) == pytest.approx(
            0.0, abs=1e-6
        )
        # ...and not on, nor even near, anything the skipping one declares.
        assert float(skipped(point)) > 0.1
        assert min(abs(float(f(point))) for f in skipped.patch_fields()) > 0.05

    def test_min_of_max_composition_is_the_sdf_inside(self):
        """The node's sdf agrees with the composition its patches describe.

        A pattern is a ``min`` over instances of a child that is itself a
        ``max`` over its patches; the reshape below only lands on the right
        groups because the layout is instance-major.
        """
        pattern = LinearPattern(Box(size=jnp.asarray(BOX_SIZE)), self.DIRECTION, 3, 2.5)
        fields = pattern.patch_fields()
        rng = np.random.default_rng(1)
        points = jnp.asarray(rng.uniform(-1.0, 1.0, size=(96, 3)) * BOX_SIZE)
        stacked = jnp.stack([field(points) for field in fields])
        composed = jnp.min(jnp.max(stacked.reshape(3, 6, -1), axis=1), axis=0)
        reference = pattern(points)
        inside = np.asarray(reference) <= 0.0
        assert inside.any()
        np.testing.assert_allclose(
            np.asarray(composed)[inside], np.asarray(reference)[inside], atol=1e-6
        )


class TestPolarPatternPatchFields:
    """``n_kept * child`` fields, instance-major, seed copy unrotated."""

    COUNT = 6
    SKIP = (2, 4)
    KEPT = (0, 1, 3, 5)
    BOLT_CIRCLE = 0.8
    # Tilted for the same reason the linear pattern runs diagonally: an
    # upright pin rotated about z would leave every copy's cap half-spaces
    # coincident, since rotation about z does not move a plane z = const.
    TILT = 0.4

    def _child(self) -> Translate:
        return Translate(Rotate(_pin(), "y", self.TILT), jnp.array([self.BOLT_CIRCLE, 0.0, 0.0]))

    def _pattern(self, skip=SKIP) -> PolarPattern:
        return PolarPattern(self._child(), self.COUNT, skip=skip)

    def _instance_points(self, instance: int) -> np.ndarray:
        """Surface points of copy ``instance``, one per child patch."""
        rotation = np.asarray(Rotate._rotation_matrix(jnp.array([0.0, 1.0, 0.0]), self.TILT))
        seed = _pin_patch_points() @ rotation.T + np.array([self.BOLT_CIRCLE, 0.0, 0.0])
        return _rotate_z(seed, 2.0 * np.pi * instance / self.COUNT)

    def test_field_count_is_kept_instances_times_child(self):
        assert len(self._pattern().patch_fields()) == len(self.KEPT) * PIN_PATCHES
        assert len(self._pattern(skip=()).patch_fields()) == self.COUNT * PIN_PATCHES

    def test_instance_major_order(self):
        """Slot ``n``, child patch ``j`` is index ``3n + j`` — not ``3j + n``."""
        pattern = self._pattern()
        fields = pattern.patch_fields()
        for slot, instance in enumerate(self.KEPT):
            for patch, point in enumerate(self._instance_points(instance)):
                q = jnp.asarray(point, jnp.float32)
                assert float(pattern(q)) == pytest.approx(0.0, abs=1e-5)
                assert _patch_id(fields, q) == PIN_PATCHES * slot + patch
                index = PIN_PATCHES * slot + patch
                assert float(fields[index](q)) == pytest.approx(0.0, abs=1e-5)

    def test_seed_instance_is_the_child_untouched(self):
        """Copy 0 is evaluated unrotated, exactly as ``sdf`` leaves it."""
        pattern = self._pattern()
        fields = pattern.patch_fields()
        child_fields = self._child().patch_fields()
        for point in ([0.7, 0.1, 0.05], [0.2, -0.4, 0.3]):
            q = jnp.asarray(point, jnp.float32)
            for patch, child_field in enumerate(child_fields):
                assert float(fields[patch](q)) == float(child_field(q))

    def test_a_suppressed_instance_contributes_no_fields(self):
        skipped, full = self._pattern(), self._pattern(skip=())
        point = jnp.asarray(self._instance_points(2)[0], jnp.float32)
        assert min(abs(float(f(point))) for f in full.patch_fields()) == pytest.approx(
            0.0, abs=1e-5
        )
        assert float(skipped(point)) > 0.1
        assert min(abs(float(f(point))) for f in skipped.patch_fields()) > 0.05

    def test_min_of_max_composition_is_the_sdf_inside(self):
        pattern = PolarPattern(
            Translate(Box(size=jnp.asarray(BOX_SIZE)), jnp.array([3.0, 0.0, 0.0])), 4
        )
        fields = pattern.patch_fields()
        rng = np.random.default_rng(2)
        points = jnp.asarray(
            rng.uniform(-1.0, 1.0, size=(96, 3)) * BOX_SIZE + np.array([3.0, 0.0, 0.0])
        )
        stacked = jnp.stack([field(points) for field in fields])
        composed = jnp.min(jnp.max(stacked.reshape(4, 6, -1), axis=1), axis=0)
        reference = pattern(points)
        inside = np.asarray(reference) <= 0.0
        assert inside.any()
        np.testing.assert_allclose(
            np.asarray(composed)[inside], np.asarray(reference)[inside], atol=1e-5
        )

    def test_child_without_a_decomposition_reports_none(self):
        assert PolarPattern(Twist(Sphere(1.0), 1.0), 4).patch_fields() is None


class TestShellPatchFields:
    """Two offsets per child patch, adjacent, outward before inward."""

    RADIUS = 0.5
    HEIGHT = 0.7
    THICKNESS = 0.2

    @property
    def half(self) -> float:
        return self.THICKNESS / 2.0

    def _tube(self) -> Shell:
        return Shell(Cylinder(radius=self.RADIUS, height=self.HEIGHT), self.THICKNESS)

    def test_field_count_is_twice_the_child(self):
        assert len(self._tube().patch_fields()) == 2 * 3
        assert len(Shell(Box(size=jnp.asarray(BOX_SIZE)), self.THICKNESS).patch_fields()) == 12

    def test_outward_then_inward_per_child_patch(self):
        """Index ``2j`` is ``f_j - t/2``; index ``2j + 1`` is ``-(f_j + t/2)``."""
        child = Cylinder(radius=self.RADIUS, height=self.HEIGHT)
        fields = Shell(child, self.THICKNESS).patch_fields()
        child_fields = child.patch_fields()
        for point in ([0.3, -0.1, 0.05], [0.0, 0.0, 0.9], [0.62, 0.02, -0.4]):
            q = jnp.asarray(point, jnp.float32)
            for j, child_field in enumerate(child_fields):
                value = float(child_field(q))
                assert float(fields[2 * j](q)) == pytest.approx(value - self.half, abs=1e-6)
                assert float(fields[2 * j + 1](q)) == pytest.approx(-(value + self.half), abs=1e-6)

    def test_both_walls_are_on_a_declared_patch(self):
        """Analytic points on the node's zero set, and who owns each.

        Patch order for a shelled cylinder is
        ``[side out, side in, +cap out, +cap in, -cap out, -cap in]``.  The
        cap samples sit at mid-wall radius so they are clear of both side
        surfaces, and the outward cap sample stays inside the child's radius
        so it is on the flat offset rather than the rounded rim.
        """
        shell = self._tube()
        fields = shell.patch_fields()
        mid = 0.4 * self.RADIUS
        expected = [
            ([self.RADIUS + self.half, 0.0, 0.0], 0),
            ([self.RADIUS - self.half, 0.0, 0.0], 1),
            ([mid, 0.0, self.HEIGHT + self.half], 2),
            ([mid, 0.0, self.HEIGHT - self.half], 3),
            ([mid, 0.0, -self.HEIGHT - self.half], 4),
            ([mid, 0.0, -self.HEIGHT + self.half], 5),
        ]
        for point, patch in expected:
            q = jnp.asarray(point, jnp.float32)
            assert float(shell(q)) == pytest.approx(0.0, abs=1e-6)
            magnitudes = [abs(float(f(q))) for f in fields]
            assert min(magnitudes) == pytest.approx(0.0, abs=1e-6)
            assert _patch_id(fields, q) == patch

    def test_ownership_switches_at_both_offset_rims(self):
        """The wall has two rims — outer and inner — and each is a switch."""
        shell = self._tube()
        fields = shell.patch_fields()
        eps = 1e-3
        outer_r, outer_z = self.RADIUS + self.half, self.HEIGHT + self.half
        inner_r, inner_z = self.RADIUS - self.half, self.HEIGHT - self.half
        assert _patch_id(fields, [outer_r, 0.0, outer_z - 2 * eps]) == 0
        assert _patch_id(fields, [outer_r - 2 * eps, 0.0, outer_z]) == 2
        assert _patch_id(fields, [inner_r, 0.0, inner_z - 2 * eps]) == 1
        assert _patch_id(fields, [inner_r - 2 * eps, 0.0, inner_z]) == 3

    def test_max_composition_is_the_sdf_for_a_smooth_child(self):
        """One child patch, so the two offsets compose to ``|f| - t/2`` exactly."""
        shell = Shell(Sphere(0.6), self.THICKNESS)
        fields = shell.patch_fields()
        assert len(fields) == 2
        rng = np.random.default_rng(3)
        points = jnp.asarray(rng.uniform(-1.5, 1.5, size=(128, 3)))
        composed = jnp.max(jnp.stack([field(points) for field in fields]), axis=0)
        np.testing.assert_allclose(np.asarray(composed), np.asarray(shell(points)), atol=1e-6)

    def test_both_offset_spheres_are_on_a_declared_patch(self):
        shell = Shell(Sphere(0.6), self.THICKNESS)
        fields = shell.patch_fields()
        rng = np.random.default_rng(4)
        directions = rng.normal(size=(24, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for radius, patch in ((0.6 + self.half, 0), (0.6 - self.half, 1)):
            for point in radius * directions:
                q = jnp.asarray(point, jnp.float32)
                assert float(shell(q)) == pytest.approx(0.0, abs=1e-6)
                assert _patch_id(fields, q) == patch

    def test_the_sharp_corner_of_two_outward_patches_is_off_the_surface(self):
        """The documented caveat, pinned rather than only described.

        ``|f| - t/2`` rounds the outside of a convex child edge, while the
        max over two outward offset patches continues both to a sharp corner.
        The corner is therefore a point where two patch fields vanish and the
        node's own sdf does not — the surfaces are still the right ones, and
        the consumer re-derives the trim between them, but ``argmin |f_i|``
        is not exact ownership in that sliver.
        """
        shell = self._tube()
        fields = shell.patch_fields()
        corner = jnp.asarray([self.RADIUS + self.half, 0.0, self.HEIGHT + self.half])
        assert float(fields[0](corner)) == pytest.approx(0.0, abs=1e-6)
        assert float(fields[2](corner)) == pytest.approx(0.0, abs=1e-6)
        # The rounded surface sits (sqrt(2) - 1) * t/2 inside that corner.
        assert float(shell(corner)) == pytest.approx((np.sqrt(2.0) - 1.0) * self.half, abs=1e-6)

    def test_child_without_a_decomposition_reports_none(self):
        assert Shell(Twist(Sphere(1.0), 1.0), self.THICKNESS).patch_fields() is None


class TestBooleanPatchFields:
    """The operands' fields concatenated, operand-major, for hard booleans."""

    # Far enough apart that the two boxes are disjoint, so all twelve patches
    # bound something and every face point has one unambiguous owner.  Offset
    # on *all three* axes on purpose: a translation along x alone would leave
    # the two copies' four y/z face planes coincident, and ownership between
    # two identical fields is not a meaningful thing to assert.
    APART = jnp.array([3.0, 1.7, 1.1])

    def _boxes(self):
        return Box(size=jnp.asarray(BOX_SIZE)), Translate(
            Box(size=jnp.asarray(BOX_SIZE)), self.APART
        )

    def test_hard_union_declares_both_operands(self):
        first, second = self._boxes()
        assert len(Union((first, second), smoothness=0.0).patch_fields()) == 12

    def test_operand_major_order(self):
        """Operand 0's six patches come first, then operand 1's — not interleaved."""
        first, second = self._boxes()
        union = Union((first, second), smoothness=0.0)
        fields = union.patch_fields()
        for face in range(6):
            for point in _box_face_points(face):
                near = jnp.asarray(point, jnp.float32)
                far = near + self.APART
                assert float(union(near)) == pytest.approx(0.0, abs=1e-6)
                assert float(union(far)) == pytest.approx(0.0, abs=1e-6)
                assert _patch_id(fields, near) == face
                assert _patch_id(fields, far) == 6 + face

    def test_min_of_max_composition_is_the_sdf(self):
        """The node's sdf agrees with the composition its patches describe.

        The reshape only lands on the right groups because the layout is
        operand-major.  Compared inside, where a box's own ``max`` over its
        face half-spaces is its exact distance.
        """
        first, second = self._boxes()
        union = Union((first, second), smoothness=0.0)
        fields = union.patch_fields()
        rng = np.random.default_rng(21)
        points = jnp.asarray(rng.uniform(-1.2, 1.2, size=(96, 3)) * BOX_SIZE)
        stacked = jnp.stack([field(points) for field in fields])
        composed = jnp.min(jnp.max(stacked.reshape(2, 6, -1), axis=1), axis=0)
        reference = union(points)
        inside = np.asarray(reference) <= 0.0
        assert inside.any()
        np.testing.assert_allclose(
            np.asarray(composed)[inside], np.asarray(reference)[inside], atol=1e-6
        )

    def test_smooth_union_declares_nothing(self):
        """A blended seam is on no operand's zero set, so there is nothing to say."""
        first, second = self._boxes()
        assert Union((first, second), smoothness=0.1).patch_fields() is None
        # The default constructor is a *smooth* union, and declines too.
        assert Union((first, second)).patch_fields() is None
        assert Intersection((first, second)).patch_fields() is None
        assert Difference((first, second)).patch_fields() is None

    def test_nesting_flattens_operand_major(self):
        """A union of a union: the inner operands keep their relative order."""
        first, second = self._boxes()
        third = Translate(Box(size=jnp.asarray(BOX_SIZE)), 2.0 * self.APART)
        inner = Union((first, second), smoothness=0.0)
        outer = Union((inner, third), smoothness=0.0)
        fields = outer.patch_fields()
        assert len(fields) == 18
        for face in range(6):
            for point in _box_face_points(face):
                base = jnp.asarray(point, jnp.float32)
                assert _patch_id(fields, base) == face
                assert _patch_id(fields, base + self.APART) == 6 + face
                assert _patch_id(fields, base + 2.0 * self.APART) == 12 + face

    def test_an_operand_without_a_decomposition_declines_the_whole_node(self):
        """Partial cover is not on offer: the contract is the whole surface."""
        first, _ = self._boxes()
        opaque = Translate(Twist(Sphere(0.5), 1.0), self.APART)
        assert Union((first, opaque), smoothness=0.0).patch_fields() is None
        assert Difference((first, opaque), smoothness=0.0).patch_fields() is None

    def test_intersection_owns_the_surviving_faces(self):
        """Two overlapping boxes: each keeps the faces that bound the overlap."""
        offset = jnp.array([0.9, 0.0, 0.0])
        first = Box(size=jnp.asarray(BOX_SIZE))
        second = Translate(Box(size=jnp.asarray(BOX_SIZE)), offset)
        node = Intersection((first, second), smoothness=0.0)
        fields = node.patch_fields()
        assert len(fields) == 12
        # The overlap runs x in [0.9 - 0.8, 0.8]; its +x wall is operand 0's
        # +x face (patch 0) and its -x wall is operand 1's -x face (patch 7).
        plus = jnp.asarray([BOX_SIZE[0], 0.1, 0.05], jnp.float32)
        minus = jnp.asarray([float(offset[0]) - BOX_SIZE[0], 0.1, 0.05], jnp.float32)
        for point, patch in ((plus, 0), (minus, 7)):
            assert float(node(point)) == pytest.approx(0.0, abs=1e-6)
            assert _patch_id(fields, point) == patch

    def test_difference_negates_its_tools(self):
        """A cut surface must read positive outside the *result*, not the tool.

        The tool's own field is positive outside the tool — which is inside
        the material it was cutting.  Negating puts the sign back the way
        round the consumer expects, without moving the zero set.
        """
        body = Box(size=jnp.asarray(BOX_SIZE))
        tool = Translate(Box(size=jnp.asarray([0.3, 0.3, 0.3])), jnp.array([BOX_SIZE[0], 0.0, 0.0]))
        node = Difference((body, tool), smoothness=0.0)
        fields = node.patch_fields()
        assert len(fields) == 12
        tool_fields = tool.patch_fields()
        # Deep inside the cavity: outside the result, so every declared field
        # of the tool must agree by reading positive there.
        cavity = jnp.asarray([BOX_SIZE[0], 0.0, 0.0], jnp.float32)
        assert float(node(cavity)) > 0.0
        for index in range(6):
            assert float(fields[6 + index](cavity)) == pytest.approx(
                -float(tool_fields[index](cavity)), abs=1e-6
            )
        assert min(float(fields[6 + index](cavity)) for index in range(6)) > 0.0
        # The cut's own back wall (the tool's -x face) is on the result, and
        # the negation leaves that zero set exactly where it was.
        wall = jnp.asarray([BOX_SIZE[0] - 0.3, 0.1, 0.05], jnp.float32)
        assert float(node(wall)) == pytest.approx(0.0, abs=1e-6)
        assert _patch_id(fields, wall) == 6 + 1

    def test_a_boolean_under_a_pattern_is_forwarded(self):
        """The case the protocol exists for: a nested boolean the scene keeps.

        ``world_frame_leaves`` splits a scene at its booleans, so a boolean
        only reaches ``patch_fields`` from *under* something else — here a
        polar pattern of a two-cylinder tool, which is exactly the shape of
        the motor shield leaf this unblocked.
        """
        # Both pins are tilted, and staggered in radius and height.  Upright
        # coaxial pins would leave every cap patch coincident with every
        # other's — a cap field depends on z alone, and rotating about z does
        # not move a plane z = const — and ownership between two identical
        # fields is not a meaningful thing to assert.
        places = (np.array([0.8, 0.0, 0.0]), np.array([1.35, 0.0, 0.6]))
        tilts = (0.4, -0.3)
        tool = Union(
            tuple(
                Translate(Rotate(_pin(), "y", tilt), jnp.asarray(place))
                for place, tilt in zip(places, tilts)
            ),
            smoothness=0.0,
        )
        pattern = PolarPattern(tool, 4)
        fields = pattern.patch_fields()
        assert len(fields) == 4 * 2 * PIN_PATCHES
        for instance in range(4):
            angle = 2.0 * np.pi * instance / 4
            for operand, (place, tilt) in enumerate(zip(places, tilts)):
                rotation = np.asarray(Rotate._rotation_matrix(jnp.array([0.0, 1.0, 0.0]), tilt))
                seed = _pin_patch_points() @ rotation.T + place
                for patch, point in enumerate(_rotate_z(seed, angle)):
                    q = jnp.asarray(point, jnp.float32)
                    assert float(pattern(q)) == pytest.approx(0.0, abs=1e-5)
                    expected = instance * 2 * PIN_PATCHES + operand * PIN_PATCHES + patch
                    assert _patch_id(fields, q) == expected


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
