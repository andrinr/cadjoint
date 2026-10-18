"""Primitives built on a Euclidean norm keep finite gradients where it is zero.

``jnp.linalg.norm`` has a NaN gradient at the zero vector, and a NaN
gradient survives a ``max``/``where`` even where its branch is not selected,
so one primitive evaluated at such a point poisons every boolean around it
(issue #57).  For RoundBox that point set is its whole inner box; for the
others it is a centre, an axis or a core circle that a lattice can hit
exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.sdf.primitives.base import safe_norm
from cadjoint.sdf.primitives.capsule import Capsule
from cadjoint.sdf.primitives.round_box import RoundBox
from cadjoint.sdf.primitives.sphere import Sphere
from cadjoint.sdf.primitives.torus import Torus

SIZE = jnp.array([57.0, 27.0, 2.0])

#: (name, field, point where a bare norm's argument is zero)
DEGENERATE = [
    ("round_box inside", lambda p: RoundBox.sdf(p, SIZE, 3.0), [-54.0, 8.0, -1.8]),
    ("sphere centre", lambda p: Sphere.sdf(p, 1.0), [0.0, 0.0, 0.0]),
    ("capsule axis", lambda p: Capsule.sdf(p, 1.0, 2.0), [0.0, 0.0, 0.5]),
    ("torus z axis", lambda p: Torus.sdf(p, 3.0, 1.0), [0.0, 0.0, 0.5]),
    ("torus core", lambda p: Torus.sdf(p, 3.0, 1.0), [3.0, 0.0, 0.0]),
]


@pytest.mark.parametrize("name,field,point", DEGENERATE, ids=[d[0] for d in DEGENERATE])
def test_gradient_is_finite_where_the_norm_is_zero(name, field, point):
    gradient = jax.grad(field)(jnp.array(point))
    assert np.isfinite(np.asarray(gradient)).all(), name


def test_round_box_interior_gradient_points_at_the_nearest_face():
    # 0.2 below the top of the inner box in z, far from the x and y faces.
    gradient = jax.grad(lambda p: RoundBox.sdf(p, SIZE, 3.0))(jnp.array([-54.0, 8.0, -1.8]))
    np.testing.assert_allclose(np.asarray(gradient), [0.0, 0.0, -1.0], atol=1e-6)


def test_a_nan_no_longer_leaks_through_a_boolean_that_does_not_select_it():
    """The issue's case: a cut whose plane wins, around a round box."""

    def cut(p):
        return jnp.maximum(RoundBox.sdf(p, SIZE, 3.0), -(p[0] + 54.0))

    gradient = jax.grad(cut)(jnp.array([-54.5, 8.0, -1.8]))
    np.testing.assert_allclose(np.asarray(gradient), [-1.0, 0.0, 0.0], atol=1e-6)


def test_safe_norm_values_match_the_plain_norm_under_jit():
    """Bit for bit when compiled, as fields are; eagerly the unfused sum can
    round one ulp differently, which is not a property of the guard."""
    rng = np.random.default_rng(0)
    vectors = jnp.asarray(rng.normal(size=(1000, 3)).astype(np.float32))
    vectors = vectors.at[0].set(0.0)
    guarded = jax.jit(safe_norm)(vectors)
    plain = jax.jit(lambda v: jnp.linalg.norm(v, axis=-1))(vectors)
    np.testing.assert_array_equal(np.asarray(guarded), np.asarray(plain))
