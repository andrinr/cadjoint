"""Tests for the draft/twist options on extrude and ExtrudedPolygon."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.construction import PolygonProfile, extrude
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.geometry.primitives import Rectangle
from cadjoint.sdf.primitives import ExtrudedPolygon

SQUARE = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
# Elongated in x so a rotated slice is distinguishable from the plain one.
BAR = [[-1.0, -0.2], [1.0, -0.2], [1.0, 0.2], [-1.0, 0.2]]


def sample_points(count: int = 1000, extent: float = 1.5, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(-extent, extent, size=(count, 3)), dtype=jnp.float32)


def square_vertex_kwargs() -> dict[str, jnp.ndarray]:
    return {f"v{i}": jnp.asarray(v, dtype=jnp.float32) for i, v in enumerate(SQUARE)}


class TestZeroOptionsAreThePlainPath:
    def test_explicit_zeros_are_bit_identical_to_plain_extrude(self):
        plain = extrude(PolygonProfile(SQUARE, name="pz"), depth=1.2)
        zeroed = extrude(PolygonProfile(SQUARE, name="zz"), depth=1.2, draft=0.0, twist=0.0)
        points = sample_points()
        np.testing.assert_array_equal(np.asarray(plain(points)), np.asarray(zeroed(points)))

    def test_explicit_zeros_are_bit_identical_on_the_primitive(self):
        vertices = [jnp.asarray(v, dtype=jnp.float32) for v in SQUARE]
        plain = ExtrudedPolygon(vertices, depth=1.2)
        zeroed = ExtrudedPolygon(vertices, depth=1.2, draft=0.0, twist=0.0)
        points = sample_points()
        np.testing.assert_array_equal(np.asarray(plain(points)), np.asarray(zeroed(points)))

    def test_concrete_zeros_keep_the_plain_parameter_layout(self):
        vertices = [jnp.asarray(v, dtype=jnp.float32) for v in SQUARE]
        zeroed = ExtrudedPolygon(vertices, depth=1.2, draft=0.0, twist=0.0)
        assert "draft" not in zeroed.params
        assert "twist" not in zeroed.params
        assert zeroed.is_exact

    def test_scalar_zero_parameters_are_kept(self):
        # A Scalar draft/twist stays a live parameter even at value zero.
        vertices = [jnp.asarray(v, dtype=jnp.float32) for v in SQUARE]
        solid = ExtrudedPolygon(
            vertices,
            depth=1.2,
            draft=Scalar(0.0, name="draft"),
            twist=Scalar(0.0, name="twist"),
        )
        assert "draft" in solid.params
        assert "twist" in solid.params
        assert not solid.is_exact


class TestDraft:
    def test_draft_narrows_the_section_toward_positive_z(self):
        solid = extrude(PolygonProfile(SQUARE, name="dn"), depth=1.0, draft=20.0)
        # Near the walls: inside at the bottom cap, drafted away at the top.
        assert float(solid(jnp.array([0.9, 0.0, -0.45]))) < 0.0
        assert float(solid(jnp.array([0.9, 0.0, 0.45]))) > 0.0
        # The center column survives the taper at this draft angle.
        assert float(solid(jnp.array([0.0, 0.0, 0.45]))) < 0.0

    def test_profile_is_exact_at_the_bottom_cap(self):
        # At z = -depth/2 the drafted 2D offset is zero: the in-slice field
        # matches the plain extrusion on the bottom cap plane.
        plain = extrude(PolygonProfile(SQUARE, name="pb"), depth=1.0)
        drafted = extrude(PolygonProfile(SQUARE, name="db"), depth=1.0, draft=20.0)
        rng = np.random.default_rng(5)
        xy = rng.uniform(-1.4, 1.4, size=(200, 2))
        points = jnp.asarray(np.concatenate([xy, np.full((200, 1), -0.5)], axis=1), jnp.float32)
        inside = np.asarray(plain(points)) < -1e-3
        np.testing.assert_allclose(
            np.asarray(drafted(points))[inside], np.asarray(plain(points))[inside], atol=1e-6
        )

    def test_drafted_solid_is_flagged_inexact(self):
        solid = extrude(PolygonProfile(SQUARE, name="df"), depth=1.0, draft=20.0)
        assert not solid.is_exact


class TestTwist:
    def test_twist_matches_rotating_the_query_by_theta_of_z(self):
        # Documented form: the query XY is rotated by twist * z / depth
        # (zero at mid-depth, +/- twist/2 at the caps).
        depth, twist = 1.0, 90.0
        twisted = extrude(PolygonProfile(BAR, name="tq"), depth=depth, twist=twist)
        plain = extrude(PolygonProfile(BAR, name="pq"), depth=depth)
        points = sample_points(seed=6)
        theta = jnp.deg2rad(twist) * points[:, 2] / depth
        c, s = jnp.cos(theta), jnp.sin(theta)
        rotated = jnp.stack(
            [
                points[:, 0] * c + points[:, 1] * s,
                points[:, 1] * c - points[:, 0] * s,
                points[:, 2],
            ],
            axis=-1,
        )
        np.testing.assert_allclose(
            np.asarray(twisted(points)), np.asarray(plain(rotated)), atol=1e-6
        )

    def test_caps_are_rotated_by_half_the_twist(self):
        # With twist=90 the section at the top cap is the bar rotated +45deg
        # (counterclockwise), at the bottom cap -45deg. A probe on the +45deg
        # diagonal is inside near the top only; its mirror diagonal is inside
        # near the bottom only.
        solid = extrude(PolygonProfile(BAR, name="tc"), depth=1.0, twist=90.0)
        r = 0.7 / np.sqrt(2.0)
        on_plus_diagonal = jnp.array([r, r, 0.499])
        assert float(solid(on_plus_diagonal)) < 0.0
        assert float(solid(jnp.array([r, r, -0.499]))) > 0.0
        on_minus_diagonal = jnp.array([r, -r, -0.499])
        assert float(solid(on_minus_diagonal)) < 0.0
        assert float(solid(jnp.array([r, -r, 0.499]))) > 0.0

    def test_mid_depth_section_is_unrotated(self):
        twisted = extrude(PolygonProfile(BAR, name="tm"), depth=1.0, twist=90.0)
        plain = extrude(PolygonProfile(BAR, name="pm"), depth=1.0)
        rng = np.random.default_rng(7)
        xy = rng.uniform(-1.4, 1.4, size=(200, 2))
        points = jnp.asarray(np.concatenate([xy, np.zeros((200, 1))], axis=1), jnp.float32)
        np.testing.assert_allclose(
            np.asarray(twisted(points)), np.asarray(plain(points)), atol=1e-6
        )

    def test_twisted_solid_is_flagged_inexact(self):
        solid = extrude(PolygonProfile(SQUARE, name="tf"), depth=1.0, twist=45.0)
        assert not solid.is_exact


class TestDifferentiability:
    POINTS = None

    @classmethod
    def points(cls) -> jnp.ndarray:
        if cls.POINTS is None:
            cls.POINTS = sample_points(200, seed=8)
        return cls.POINTS

    def test_draft_gradient_is_finite_and_nonzero(self):
        points = self.points()

        def loss(draft):
            return jnp.sum(
                ExtrudedPolygon.sdf(
                    points,
                    jnp.float32(1.0),
                    draft=draft,
                    twist=jnp.float32(0.0),
                    **square_vertex_kwargs(),
                )
                ** 2
            )

        gradient = float(jax.grad(loss)(jnp.float32(10.0)))
        assert np.isfinite(gradient)
        assert abs(gradient) > 0.0
        eps = 1e-2
        finite = (float(loss(jnp.float32(10.0 + eps))) - float(loss(jnp.float32(10.0 - eps)))) / (
            2 * eps
        )
        np.testing.assert_allclose(gradient, finite, rtol=5e-2)

    def test_twist_gradient_is_finite_and_nonzero(self):
        points = self.points()

        def loss(twist):
            return jnp.sum(
                ExtrudedPolygon.sdf(
                    points,
                    jnp.float32(1.0),
                    draft=jnp.float32(0.0),
                    twist=twist,
                    **square_vertex_kwargs(),
                )
                ** 2
            )

        gradient = float(jax.grad(loss)(jnp.float32(30.0)))
        assert np.isfinite(gradient)
        assert abs(gradient) > 0.0

    def test_vertex_gradient_with_draft_and_twist_is_finite_and_nonzero(self):
        points = self.points()

        def loss(v0):
            vertices = square_vertex_kwargs()
            vertices["v0"] = v0
            return jnp.sum(
                ExtrudedPolygon.sdf(
                    points,
                    jnp.float32(1.0),
                    draft=jnp.float32(15.0),
                    twist=jnp.float32(40.0),
                    **vertices,
                )
                ** 2
            )

        gradient = np.asarray(jax.grad(loss)(jnp.asarray(SQUARE[0], dtype=jnp.float32)))
        assert np.isfinite(gradient).all()
        assert np.abs(gradient).max() > 0.0


class TestLegacyRectanglePath:
    def rectangle(self) -> Rectangle:
        return Rectangle(
            center=Vector([0, 0, 0], name="center"),
            width=Scalar(2.0, name="width"),
            height=Scalar(1.0, name="height"),
            normal=Vector([0, 0, 1], name="normal"),
        )

    def test_rectangle_rejects_draft_and_twist(self):
        with pytest.raises(ValueError, match="PolygonProfile"):
            extrude(self.rectangle(), depth=1.0, draft=5.0)
        with pytest.raises(ValueError, match="PolygonProfile"):
            extrude(self.rectangle(), depth=1.0, twist=5.0)

    def test_rectangle_accepts_explicit_zeros(self):
        box = extrude(self.rectangle(), depth=1.0, draft=0.0, twist=0.0)
        assert box is not None
