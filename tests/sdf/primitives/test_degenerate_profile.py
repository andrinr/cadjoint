"""A repeated profile vertex must not poison the gradient.

Both polygon distance forms project the query onto each edge with
``w·e / e·e``, so a repeated vertex makes that an exact ``0/0``.  The value
survives, because the dead edge never wins the ``min`` — but reverse mode
visits every branch, and ``min`` propagates the NaN it finds in the one that
lost.  The result was a profile with a perfectly ordinary field and a NaN
derivative at **every point in space**, which nothing downstream notices
until an optimizer quietly stops moving.

The check that matters is not "no NaN" on its own: a repeated vertex adds no
geometry, so the degenerate profile must be the *same function* as the clean
one.  These assert that, in value and in gradient.

Deliberately small — five vertices and 64 points — so it stays in the fast
loop.  The whole-profile sweeps live in ``test_polygon_lowering``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.geometry.parameters import Scalar, Vector2
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon, polygon_sdf_2d

CLEAN = [(0.2, 0.0), (0.4, 0.0), (0.4, 0.3), (0.2, 0.3)]
#: The same rectangle with its second vertex written twice.
REPEATED = [(0.2, 0.0), (0.4, 0.0), (0.4, 0.0), (0.4, 0.3), (0.2, 0.3)]


def _solid(vertices):
    return ExtrudedPolygon([Vector2([x, y]) for x, y in vertices], depth=Scalar(0.5))


def _points(n=64, seed=0):
    return jnp.asarray(np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, 3)))


class TestTheGradientSurvivesARepeatedVertex:
    def test_no_gradient_is_nan(self):
        grads = np.asarray(jax.vmap(jax.grad(_solid(REPEATED)))(_points()))
        bad = int(np.isnan(grads).any(axis=1).sum())
        assert bad == 0, f"{bad} of {len(grads)} gradients are NaN"

    def test_the_value_is_unchanged_by_the_repeat(self):
        """A repeated vertex adds no geometry, so the field must be identical."""
        pts = _points()
        clean = np.asarray(jax.vmap(_solid(CLEAN))(pts))
        repeated = np.asarray(jax.vmap(_solid(REPEATED))(pts))
        np.testing.assert_array_equal(clean, repeated)

    def test_the_gradient_is_unchanged_by_the_repeat(self):
        pts = _points()
        clean = np.asarray(jax.vmap(jax.grad(_solid(CLEAN)))(pts))
        repeated = np.asarray(jax.vmap(jax.grad(_solid(REPEATED)))(pts))
        np.testing.assert_allclose(clean, repeated, atol=1e-6)

    @pytest.mark.parametrize("where", [0, 2, 4], ids=["first", "middle", "closing"])
    def test_a_repeat_anywhere_in_the_loop_is_survivable(self, where):
        """Including the wrap: the last edge closes onto vertex 0."""
        loop = list(CLEAN)
        loop.insert(where, loop[where - 1])
        grads = np.asarray(jax.vmap(jax.grad(_solid(loop)))(_points()))
        assert not np.isnan(grads).any()


class TestTheTwoDimensionalFormToo:
    """`polygon_sdf_2d` shares the projection, so it shares the trap."""

    def test_no_gradient_is_nan(self):
        verts = jnp.asarray(REPEATED)
        pts = jnp.asarray(np.random.default_rng(1).uniform(-1.0, 1.0, size=(64, 2)))
        grads = np.asarray(jax.vmap(jax.grad(lambda q: polygon_sdf_2d(q, verts)))(pts))
        assert not np.isnan(grads).any()

    def test_it_agrees_with_the_clean_profile(self):
        pts = jnp.asarray(np.random.default_rng(1).uniform(-1.0, 1.0, size=(64, 2)))
        clean = np.asarray(jax.vmap(lambda q: polygon_sdf_2d(q, jnp.asarray(CLEAN)))(pts))
        repeated = np.asarray(jax.vmap(lambda q: polygon_sdf_2d(q, jnp.asarray(REPEATED)))(pts))
        np.testing.assert_allclose(clean, repeated, atol=1e-6)
