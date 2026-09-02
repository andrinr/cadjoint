"""The stacked and unrolled polygon distances must be the same function.

The vertex loop is emitted two ways — once as an ``(N, 2)`` array reduced in
one pass, once as N individual ``vec2`` chains for the shader backend, which
has no type wider than a ``mat4``.  Both are the same arithmetic in a different
order, so they must agree in value and in gradient, not merely be close.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.geometry.parameters import Vector2
from cadjoint.sdf._lowering import scalar_lowering
from cadjoint.sdf.primitives.polygon import (
    ExtrudedPolygon,
    RevolvedPolygon,
    _polygon_distance,
    polygon_sdf_2d,
)

COMB = jnp.array(
    [
        [-0.9, 0.0],
        [0.9, 0.0],
        [0.9, 0.18],
        [0.55, 0.18],
        [0.5, 0.62],
        [0.32, 0.62],
        [0.27, 0.18],
        [-0.27, 0.18],
        [-0.32, 0.62],
        [-0.5, 0.62],
        [-0.55, 0.18],
        [-0.9, 0.18],
    ]
)


def _queries(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    return jax.random.uniform(
        jax.random.PRNGKey(seed), shape, minval=-1.4, maxval=1.4, dtype=jnp.float32
    )


@pytest.mark.parametrize("shape", [(2,), (64, 2), (4, 8, 2)])
def test_stacked_and_unrolled_distances_are_identical(shape):
    points = _queries(0, shape)
    vertices = [COMB[i] for i in range(COMB.shape[0])]

    stacked = _polygon_distance(points, vertices)
    with scalar_lowering():
        unrolled = _polygon_distance(points, vertices)

    np.testing.assert_array_equal(np.asarray(stacked), np.asarray(unrolled))


def test_gradients_match_through_the_profile_vertices():
    points = _queries(1, (128, 2))

    def loss(vertices):
        return jnp.sum(polygon_sdf_2d(points, vertices) ** 2)

    stacked = jax.grad(loss)(COMB)
    with scalar_lowering():
        unrolled = jax.grad(loss)(COMB)

    scale = float(jnp.max(jnp.abs(stacked)))
    assert float(jnp.max(jnp.abs(stacked - unrolled))) <= 1e-6 * max(scale, 1.0)


def test_extruded_and_revolved_solids_agree_in_both_forms():
    profile = [Vector2(value=list(map(float, COMB[i])), name=f"v{i}") for i in range(6)]
    points = _queries(2, (64, 3))

    for solid in (
        ExtrudedPolygon(profile, depth=0.8),
        RevolvedPolygon(profile, offset=1.4),
    ):
        stacked = jax.vmap(solid)(points)
        with scalar_lowering():
            unrolled = jax.vmap(solid)(points)
        np.testing.assert_allclose(np.asarray(stacked), np.asarray(unrolled), rtol=0, atol=1e-6)
