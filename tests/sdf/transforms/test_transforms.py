"""Tests for transformation operations."""

import jax.numpy as jnp
import pytest

from jaxcad.sdf.boolean import Union
from jaxcad.sdf.primitives import Box, Sphere
from jaxcad.sdf.transforms import Rotate, Scale, Translate, Twist


def test_translate():
    """Test translation transformation."""
    sphere = Sphere(radius=1.0)
    offset = jnp.array([1.0, 0.0, 0.0])
    translated = Translate(sphere, offset)

    # Point at (1, 0, 0) should be at center of translated sphere
    assert translated(jnp.array([1.0, 0.0, 0.0])) == pytest.approx(-1.0, abs=1e-5)

    # Point at (2, 0, 0) should be on surface
    assert translated(jnp.array([2.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-5)


def test_uniform_scale():
    """Test uniform scaling."""
    sphere = Sphere(radius=1.0)
    scaled = Scale(sphere, 2.0)

    # Sphere with radius 1 scaled by 2 should have radius 2
    assert scaled(jnp.array([2.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-5)
    assert scaled(jnp.array([0.0, 0.0, 0.0])) == pytest.approx(-2.0, abs=1e-5)


def test_rotate_z():
    """Test rotation around Z axis."""
    box = Box(size=jnp.array([2.0, 1.0, 1.0]))
    rotated = Rotate(box, "z", jnp.pi / 2)

    # Box has half-extents [2.0, 1.0, 1.0], so it extends ±2 in X, ±1 in Y/Z
    # After 90° rotation around Z, the X axis becomes Y axis
    # Point at (0, 2.0, 0) should be on surface (at +Y boundary)
    assert rotated(jnp.array([0.0, 2.0, 0.0])) == pytest.approx(0.0, abs=1e-4)


def test_fluent_transform_chain_matches_explicit_transforms():
    fluent = Sphere(1.0).translate([1.0, 0.0, 0.0]).rotate("z", jnp.pi / 2).scale(2.0)
    explicit = Scale(Rotate(Translate(Sphere(1.0), [1.0, 0.0, 0.0]), "z", jnp.pi / 2), 2.0)
    point = jnp.array([0.0, 2.0, 0.0])

    assert jnp.isclose(fluent(point), explicit(point))


def test_fluent_twist_returns_twist():
    assert isinstance(Box([1.0, 1.0, 1.0]).twist(0.5, "y"), Twist)


def test_exactness_propagates_through_composed_tree():
    exact = Sphere(1.0).translate([1.0, 0.0, 0.0])
    approximate = Box([1.0, 1.0, 1.0]).twist(0.5).translate([0.0, 1.0, 0.0])
    nonuniform = Sphere(1.0).scale([1.0, 2.0, 3.0])

    assert exact.is_exact
    assert not approximate.is_exact
    assert not nonuniform.is_exact
    assert not Union(exact, approximate).is_exact


@pytest.mark.parametrize("transform", [Rotate, Twist])
@pytest.mark.parametrize("axis", ["invalid", [0.0, 0.0, 0.0]])
def test_axis_transforms_reject_invalid_axes(transform, axis):
    args = (Sphere(1.0), axis, 0.5) if transform is Rotate else (Sphere(1.0), 0.5, axis)

    with pytest.raises(ValueError, match="axis"):
        transform(*args)


def test_scale_accepts_jax_scalar_and_preserves_sign_of_distance():
    scaled = Scale(Sphere(1.0), jnp.array(-2.0))

    assert scaled(jnp.zeros(3)) == pytest.approx(-2.0)
    assert scaled(jnp.array([2.0, 0.0, 0.0])) == pytest.approx(0.0)


@pytest.mark.parametrize("scale", [0.0, [1.0, 0.0, 1.0], [1.0, 2.0]])
def test_scale_rejects_degenerate_values(scale):
    with pytest.raises(ValueError, match="Scale"):
        Scale(Sphere(1.0), scale)
