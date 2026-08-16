"""Tests for the loft construction function and LoftedPolygon primitive."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcad import extract_parameters, functionalize
from jaxcad.construction import PolygonProfile, SketchPlane, extrude, loft
from jaxcad.sdf.primitives import LoftedPolygon

SQUARE = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
SMALL_SQUARE = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
TRIANGLE = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]


def sample_points(count: int = 1000, extent: float = 1.5, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(-extent, extent, size=(count, 3)), dtype=jnp.float32)


class TestValidation:
    def test_vertex_count_mismatch_raises(self):
        a = PolygonProfile(SQUARE, name="a")
        b = PolygonProfile(TRIANGLE, name="b")
        with pytest.raises(ValueError, match="vertex counts"):
            loft(a, b, height=1.0)

    def test_primitive_vertex_count_mismatch_raises(self):
        square = [jnp.asarray(v, dtype=jnp.float32) for v in SQUARE]
        triangle = [jnp.asarray(v, dtype=jnp.float32) for v in TRIANGLE]
        with pytest.raises(ValueError, match="equal vertex counts"):
            LoftedPolygon(square, triangle, height=1.0)

    def test_too_few_vertices_raises(self):
        segment = [jnp.zeros(2), jnp.ones(2)]
        with pytest.raises(ValueError, match="at least 3"):
            LoftedPolygon(segment, segment, height=1.0)


class TestFieldValues:
    def test_constant_loft_matches_extrude(self):
        # Lofting a profile onto itself is a plain extrusion of that profile.
        solid_l = loft(PolygonProfile(SQUARE, name="la"), PolygonProfile(SQUARE, name="lb"), 1.2)
        solid_e = extrude(PolygonProfile(SQUARE, name="e"), depth=1.2)
        points = sample_points()
        np.testing.assert_allclose(
            np.asarray(solid_l(points)), np.asarray(solid_e(points)), atol=1e-6
        )

    def test_profiles_are_recovered_at_the_caps(self):
        # Just inside each cap the slice polygon is the corresponding profile.
        solid = loft(
            PolygonProfile(SQUARE, name="ca"), PolygonProfile(SMALL_SQUARE, name="cb"), 2.0
        )
        near_bottom, near_top = -0.999, 0.999
        # (0.75, 0, z) is inside the bottom square but outside the top square.
        assert float(solid(jnp.array([0.75, 0.0, near_bottom]))) < 0.0
        assert float(solid(jnp.array([0.75, 0.0, near_top]))) > 0.0
        # The centroid stays inside through the whole loft.
        assert float(solid(jnp.array([0.0, 0.0, near_bottom]))) < 0.0
        assert float(solid(jnp.array([0.0, 0.0, near_top]))) < 0.0

    def test_returns_lofted_polygon_on_identity_plane(self):
        solid = loft(PolygonProfile(SQUARE, name="ia"), PolygonProfile(SQUARE, name="ib"), 1.0)
        assert isinstance(solid, LoftedPolygon)


class TestParameterSharing:
    def test_mutating_profile_vertex_changes_field(self):
        profile_a = PolygonProfile(SQUARE, name="sa")
        profile_b = PolygonProfile(SMALL_SQUARE, name="sb")
        solid = loft(profile_a, profile_b, height=1.2)
        points = sample_points()
        before = np.asarray(solid(points))
        profile_a.vertices[0].value = jnp.array([-2.0, -2.0], dtype=jnp.float32)
        after = np.asarray(solid(points))
        assert np.abs(after - before).max() > 1e-3

    def test_profile_parameters_are_shared_by_identity(self):
        profile_a = PolygonProfile(SQUARE, name="ida")
        profile_b = PolygonProfile(SMALL_SQUARE, name="idb")
        solid = loft(profile_a, profile_b, height=1.2)
        for i, vertex in enumerate(profile_a.vertices):
            assert solid.params[f"v{i}"] is vertex
        for i, vertex in enumerate(profile_b.vertices):
            assert solid.params[f"w{i}"] is vertex


class TestPlanePlacement:
    PLANE_ORIGIN = (1.0, 2.0, 3.0)
    PLANE_NORMAL = (0.0, 1.0, 0.0)

    def test_placed_on_profile_a_plane_like_extrude(self):
        plane = SketchPlane(origin=self.PLANE_ORIGIN, normal=self.PLANE_NORMAL)
        profile_a = PolygonProfile(SQUARE, plane=plane, name="pa")
        profile_b = PolygonProfile(SQUARE, name="pb")
        solid_l = loft(profile_a, profile_b, height=1.2)
        solid_e = extrude(PolygonProfile(SQUARE, plane=plane, name="pe"), depth=1.2)
        rng = np.random.default_rng(1)
        points = jnp.asarray(rng.uniform(-1.0, 4.0, size=(500, 3)), dtype=jnp.float32)
        np.testing.assert_allclose(
            np.asarray(solid_l(points)), np.asarray(solid_e(points)), atol=1e-6
        )

    def test_profile_b_plane_is_ignored(self):
        plane = SketchPlane(origin=self.PLANE_ORIGIN, normal=self.PLANE_NORMAL)
        other = SketchPlane(origin=(9.0, -9.0, 9.0), normal=(1.0, 0.0, 0.0))
        profile_a = PolygonProfile(SQUARE, plane=plane, name="ba")
        on_default = loft(profile_a, PolygonProfile(SMALL_SQUARE, name="bd"), height=1.2)
        on_other = loft(profile_a, PolygonProfile(SMALL_SQUARE, plane=other, name="bo"), 1.2)
        rng = np.random.default_rng(2)
        points = jnp.asarray(rng.uniform(-1.0, 4.0, size=(500, 3)), dtype=jnp.float32)
        np.testing.assert_allclose(
            np.asarray(on_default(points)), np.asarray(on_other(points)), atol=1e-6
        )


class TestGradients:
    def test_vertex_gradient_is_finite_and_nonzero(self):
        profile_a = PolygonProfile(SQUARE, name="ga")
        profile_b = PolygonProfile(SMALL_SQUARE, name="gb")
        solid = loft(profile_a, profile_b, height=1.2)
        compiled = functionalize(solid)
        free, fixed, _ = extract_parameters(solid)
        assert "ga_v0" in free
        points = sample_points(200, seed=3)

        def loss(free_params):
            sdf = compiled(free_params, fixed)
            return jnp.sum(jax.vmap(sdf)(points) ** 2)

        gradient = np.asarray(jax.grad(loss)(free)["ga_v0"])
        assert np.isfinite(gradient).all()
        assert np.abs(gradient).max() > 0.0

    def test_vertex_gradient_matches_finite_differences(self):
        profile_a = PolygonProfile(SQUARE, name="fa")
        profile_b = PolygonProfile(SMALL_SQUARE, name="fb")
        solid = loft(profile_a, profile_b, height=1.2)
        compiled = functionalize(solid)
        free, fixed, _ = extract_parameters(solid)
        points = sample_points(200, seed=4)

        def loss(free_params):
            sdf = compiled(free_params, fixed)
            return jnp.sum(jax.vmap(sdf)(points) ** 2)

        gradient = np.asarray(jax.grad(loss)(free)["fa_v0"])
        eps = 1e-3
        base = np.asarray(free["fa_v0"], dtype=np.float64)
        for component in range(2):
            delta = np.zeros(2)
            delta[component] = eps
            upper = float(loss({**free, "fa_v0": jnp.asarray(base + delta, dtype=jnp.float32)}))
            lower = float(loss({**free, "fa_v0": jnp.asarray(base - delta, dtype=jnp.float32)}))
            np.testing.assert_allclose(gradient[component], (upper - lower) / (2 * eps), rtol=5e-2)
