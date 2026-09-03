"""Tests for cadjoint.sdf.operations (Shell, Offset, Mirror, patterns)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint import extract_parameters, functionalize
from cadjoint.geometry.parameters import Scalar
from cadjoint.meshing import GridSpec, extract_mesh
from cadjoint.sdf.operations import LinearPattern, Mirror, Offset, PolarPattern, Shell
from cadjoint.sdf.primitives import Sphere
from cadjoint.sdf.transforms import Translate
from tests.meshing.test_dual_contouring import (
    euler_characteristic,
    signed_volume,
    undirected_edge_counts,
)


def sample_points(count: int = 500, extent: float = 1.5, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(-extent, extent, size=(count, 3)), dtype=jnp.float32)


def face_component_count(faces: np.ndarray) -> int:
    """Number of connected components of the faces' vertex graph (union-find)."""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for face in faces:
        anchor = find(int(face[0]))
        for vertex in face[1:]:
            parent[find(int(vertex))] = anchor
    return len({find(v) for v in parent})


class TestFieldValues:
    def test_shell_is_abs_minus_half_thickness(self):
        sphere = Sphere(1.0)
        points = sample_points()
        expected = np.abs(np.asarray(sphere(points))) - 0.1
        np.testing.assert_allclose(np.asarray(Shell(sphere, 0.2)(points)), expected, atol=1e-6)

    def test_offset_subtracts_distance(self):
        sphere = Sphere(1.0)
        points = sample_points()
        expected = np.asarray(sphere(points)) - 0.3
        np.testing.assert_allclose(np.asarray(Offset(sphere, 0.3)(points)), expected, atol=1e-6)

    def test_negative_offset_shrinks(self):
        sphere = Sphere(1.0)
        boundary = jnp.array([0.9, 0.0, 0.0])
        assert float(sphere(boundary)) < 0.0
        assert float(Offset(sphere, -0.2)(boundary)) > 0.0

    @pytest.mark.parametrize("axis,flip", [("x", 0), ("y", 1), ("z", 2)])
    def test_mirror_flips_the_chosen_axis(self, axis, flip):
        shape = Translate(Sphere(0.5), offset=jnp.array([0.7, 0.2, -0.3]))
        points = sample_points()
        sign = np.ones(3, dtype=np.float32)
        sign[flip] = -1.0
        expected = np.asarray(shape(points * jnp.asarray(sign)))
        np.testing.assert_allclose(np.asarray(Mirror(shape, axis)(points)), expected, atol=1e-6)

    def test_linear_pattern_is_min_over_translates(self):
        sphere = Sphere(0.4)
        pattern = LinearPattern(sphere, direction=[1.0, 0.0, 0.0], count=3, spacing=1.2)
        points = sample_points(extent=3.0)
        translates = [np.asarray(sphere(points - jnp.array([1.2 * i, 0.0, 0.0]))) for i in range(3)]
        np.testing.assert_allclose(
            np.asarray(pattern(points)), np.minimum.reduce(translates), atol=1e-6
        )

    def test_linear_pattern_normalizes_direction(self):
        sphere = Sphere(0.4)
        pattern = LinearPattern(sphere, direction=[2.0, 0.0, 0.0], count=2, spacing=1.0)
        unit = LinearPattern(sphere, direction=[1.0, 0.0, 0.0], count=2, spacing=1.0)
        points = sample_points(extent=2.0, seed=1)
        np.testing.assert_allclose(np.asarray(pattern(points)), np.asarray(unit(points)), atol=1e-6)

    def test_polar_pattern_is_min_over_rotations(self):
        shape = Translate(Sphere(0.5), offset=jnp.array([0.7, 0.2, -0.3]))
        pattern = PolarPattern(shape, count=4)
        points = sample_points(extent=2.0, seed=2)
        rotated = []
        for i in range(4):
            theta = 2.0 * math.pi * i / 4
            c, s = math.cos(theta), math.sin(theta)
            q = jnp.stack(
                [
                    points[:, 0] * c + points[:, 1] * s,
                    points[:, 1] * c - points[:, 0] * s,
                    points[:, 2],
                ],
                axis=-1,
            )
            rotated.append(np.asarray(shape(q)))
        np.testing.assert_allclose(
            np.asarray(pattern(points)), np.minimum.reduce(rotated), atol=1e-6
        )


class TestTreeStructure:
    def test_children_return_the_wrapped_shape(self):
        sphere = Sphere(1.0)
        wrappers = [
            Shell(sphere, 0.2),
            Offset(sphere, 0.3),
            Mirror(sphere, "x"),
            LinearPattern(sphere, direction=[1.0, 0.0, 0.0], count=3, spacing=1.2),
            PolarPattern(sphere, count=4),
        ]
        for wrapper in wrappers:
            assert wrapper.children() == [sphere]

    def test_plain_callable_child_evaluates_but_has_no_children(self):
        def field(p):
            return jnp.linalg.norm(p, axis=-1) - 1.0

        shell = Shell(field, 0.2)
        assert shell.children() == []
        np.testing.assert_allclose(float(shell(jnp.array([1.05, 0.0, 0.0]))), -0.05, atol=1e-6)

    def test_validation_errors(self):
        sphere = Sphere(1.0)
        with pytest.raises(ValueError, match="axis"):
            Mirror(sphere, "w")
        with pytest.raises(ValueError, match="count"):
            LinearPattern(sphere, direction=[1.0, 0.0, 0.0], count=0, spacing=1.0)
        with pytest.raises(ValueError, match="count"):
            PolarPattern(sphere, count=0)
        with pytest.raises(ValueError, match="axis"):
            PolarPattern(sphere, count=3, axis="x")


class TestMeshingComposition:
    OPS_GRID = GridSpec.from_bounds((-1.55, -1.55, -1.55), (3.1, 3.1, 3.1), 31)

    def assert_watertight(self, mesh) -> None:
        assert mesh.faces.shape[0] > 0
        np.testing.assert_array_equal(np.unique(undirected_edge_counts(mesh.faces)), [2])
        assert signed_volume(np.asarray(mesh.vertices, dtype=np.float64), mesh.faces) > 0

    def test_linear_pattern_meshes_three_components(self):
        pattern = LinearPattern(Sphere(0.4), direction=[1.0, 0.0, 0.0], count=3, spacing=1.2)
        grid = GridSpec.from_bounds((-0.65, -0.65, -0.65), (3.7, 1.3, 1.3), (37, 13, 13))
        mesh = extract_mesh(pattern, grid)
        self.assert_watertight(mesh)
        # Three disjoint sphere shells: Euler characteristic 2 per component.
        assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == 6
        assert face_component_count(mesh.faces) == 3

    def test_shell_meshes_two_nested_components(self):
        mesh = extract_mesh(Shell(Sphere(1.0), 0.2), self.OPS_GRID)
        self.assert_watertight(mesh)
        # Inner and outer sphere surfaces: Euler characteristic 2 + 2.
        assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == 4
        assert face_component_count(mesh.faces) == 2

    def test_offset_meshes_grown_sphere(self):
        mesh = extract_mesh(Offset(Sphere(1.0), 0.3), self.OPS_GRID)
        self.assert_watertight(mesh)
        assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == 2
        volume = signed_volume(np.asarray(mesh.vertices, dtype=np.float64), mesh.faces)
        np.testing.assert_allclose(volume, 4.0 / 3.0 * np.pi * 1.3**3, rtol=2e-2)

    def test_mirror_meshes_reflected_sphere(self):
        shape = Mirror(Translate(Sphere(0.6), offset=jnp.array([0.8, 0.0, 0.0])), "x")
        mesh = extract_mesh(shape, self.OPS_GRID)
        self.assert_watertight(mesh)
        assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == 2
        centroid = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
        np.testing.assert_allclose(centroid, [-0.8, 0.0, 0.0], atol=5e-2)

    def test_polar_pattern_meshes_four_components(self):
        shape = PolarPattern(Translate(Sphere(0.45), offset=jnp.array([0.9, 0.0, 0.0])), count=4)
        mesh = extract_mesh(shape, self.OPS_GRID)
        self.assert_watertight(mesh)
        assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == 8
        assert face_component_count(mesh.faces) == 4


class TestGradients:
    def loss_of(self, shape):
        compiled = functionalize(shape)
        free, fixed, _ = extract_parameters(shape)
        points = sample_points(200, seed=3)

        def loss(free_params):
            sdf = compiled(free_params, fixed)
            return jnp.sum(jax.vmap(sdf)(points) ** 2)

        return loss, free

    def test_shell_radius_gradient_is_finite_and_nonzero(self):
        loss, free = self.loss_of(Shell(Sphere(Scalar(1.0, free=True, name="radius")), 0.2))
        gradient = float(jax.grad(loss)(free)["radius"])
        assert np.isfinite(gradient)
        assert abs(gradient) > 0.0
        eps = 1e-3
        upper = float(loss({"radius": jnp.asarray(free["radius"]) + eps}))
        lower = float(loss({"radius": jnp.asarray(free["radius"]) - eps}))
        np.testing.assert_allclose(gradient, (upper - lower) / (2 * eps), rtol=1e-2)

    def test_offset_radius_gradient_is_finite_and_nonzero(self):
        loss, free = self.loss_of(Offset(Sphere(Scalar(1.0, free=True, name="radius")), 0.3))
        gradient = float(jax.grad(loss)(free)["radius"])
        assert np.isfinite(gradient)
        assert abs(gradient) > 0.0
        eps = 1e-3
        upper = float(loss({"radius": jnp.asarray(free["radius"]) + eps}))
        lower = float(loss({"radius": jnp.asarray(free["radius"]) - eps}))
        np.testing.assert_allclose(gradient, (upper - lower) / (2 * eps), rtol=1e-2)


class TestSuppressedInstances:
    """``skip`` leaves a station of a pattern empty without breaking the pattern.

    A ring of ribs or a row of holes is a pattern right up until something else
    has to occupy one of its stations. The suppressed copies keep the angles
    and offsets they would have had, so a gap stays a gap.
    """

    @staticmethod
    def unit_seed():
        return Translate(Sphere(0.25), offset=jnp.array([1.0, 0.0, 0.0]))

    @staticmethod
    def occupied(shape, x, y):
        return float(shape(jnp.asarray([x, y, 0.0], dtype=jnp.float32))) < 0.0

    def test_polar_pattern_omits_only_the_named_instances(self):
        seed = self.unit_seed()
        thinned = PolarPattern(seed, count=8, skip=(3, 5))
        for index in range(8):
            angle = 2.0 * math.pi * index / 8
            here = self.occupied(thinned, math.cos(angle), math.sin(angle))
            assert here is (index not in (3, 5)), index

    def test_the_kept_instances_do_not_move_up(self):
        """Suppression leaves a gap; it does not respace the ring."""
        seed = self.unit_seed()
        full = PolarPattern(seed, count=8)
        thinned = PolarPattern(seed, count=8, skip=(3,))
        for index in (0, 1, 2, 4, 5, 6, 7):
            angle = 2.0 * math.pi * index / 8
            point = jnp.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=jnp.float32)
            assert float(thinned(point)) == pytest.approx(float(full(point)), abs=1e-6)

    def test_linear_pattern_omits_only_the_named_instances(self):
        seed = self.unit_seed()
        row = LinearPattern(seed, direction=[1.0, 0.0, 0.0], count=5, spacing=0.5, skip=(2,))
        assert [self.occupied(row, 1.0 + 0.5 * i, 0.0) for i in range(5)] == [
            True,
            True,
            False,
            True,
            True,
        ]

    def test_the_suppressed_set_is_readable_back(self):
        seed = self.unit_seed()
        assert PolarPattern(seed, count=8, skip=(5, 3, 3)).skip == (3, 5)
        assert PolarPattern(seed, count=8).skip == ()
        assert LinearPattern(seed, [1.0, 0.0, 0.0], 5, 0.5, skip=[2]).skip == (2,)

    def test_suppression_survives_functionalize(self):
        seed = self.unit_seed()
        thinned = PolarPattern(seed, count=8, skip=(3, 5))
        free, fixed, _ = extract_parameters(thinned)
        sdf = functionalize(thinned)(free, fixed)
        for index in range(8):
            angle = 2.0 * math.pi * index / 8
            point = jnp.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=jnp.float32)
            assert (float(sdf(point)) < 0.0) is (index not in (3, 5)), index

    @pytest.mark.parametrize(
        ("skip", "message"),
        [
            ((0,), "cannot skip instance 0"),
            ((8,), "outside 1 .. 7"),
            ((-1,), "outside 1 .. 7"),
        ],
    )
    def test_an_unusable_index_is_refused(self, skip, message):
        with pytest.raises(ValueError, match=message):
            PolarPattern(self.unit_seed(), count=8, skip=skip)

    def test_a_count_the_mask_cannot_address_is_refused(self):
        with pytest.raises(ValueError, match="only up to count 24"):
            PolarPattern(self.unit_seed(), count=40, skip=(2,))
        # Without suppression the count is not capped.
        assert PolarPattern(self.unit_seed(), count=40).skip == ()
