"""One projection kernel: onto a surface, an edge, a corner — and refused at a tangency."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.primitives import Box, Sphere
from cadjoint.zeroset import lower
from cadjoint.zeroset.evaluate import surfaces
from cadjoint.zeroset.project import classify, project, project_table

jax.config.update("jax_enable_x64", True)


def _sphere(center, radius):
    c = jnp.asarray(center)
    return lambda p: jnp.linalg.norm(p - c) - radius


def test_one_field_lands_on_the_surface_and_the_step_is_clamped():
    seeds = np.random.default_rng(0).normal(size=(64, 3)) * 0.4 + [0.3, 0.0, 0.0]
    on = np.asarray(project([_sphere([0.3, 0, 0], 0.5)], seeds))
    np.testing.assert_allclose(np.linalg.norm(on - [0.3, 0, 0], axis=1), 0.5, atol=1e-9)
    clamped = np.asarray(project([_sphere([0.3, 0, 0], 0.5)], seeds, max_step=1e-3))
    assert np.linalg.norm(clamped - seeds, axis=1).max() <= 1e-3 + 1e-12


def test_two_fields_land_on_their_intersection_and_three_on_a_point():
    a, b = _sphere([0, 0, 0], 0.7), _sphere([0.6, 0, 0], 0.7)
    seeds = np.random.default_rng(1).normal(size=(32, 3)) * 0.2 + [0.3, 0.0, 0.5]
    on = np.asarray(project([a, b], seeds))
    np.testing.assert_allclose(np.linalg.norm(on, axis=1), 0.7, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(on - [0.6, 0, 0], axis=1), 0.7, atol=1e-9)
    c = _sphere([0.3, 0.5, 0.2], 0.7)
    corner = np.asarray(project([a, b, c], seeds))
    for centre in ([0, 0, 0], [0.6, 0, 0], [0.3, 0.5, 0.2]):
        np.testing.assert_allclose(np.linalg.norm(corner - centre, axis=1), 0.7, atol=1e-8)


def test_a_tangency_is_refused_and_the_points_stay():
    # two spheres touching at the origin: their gradients are parallel there
    a, b = _sphere([0.5, 0, 0], 0.5), _sphere([-0.5, 0, 0], 0.5)
    seeds = np.array([[0.0, 0.01, 0.0], [0.0, -0.02, 0.01]])
    assert np.allclose(np.asarray(project([a, b], seeds)), seeds)


def test_the_kernel_differentiates_through_the_design():
    def moved(radius):
        return project([_sphere([0, 0, 0], radius)], jnp.asarray([[0.2, 0.3, 0.1]]))[0]

    dp = jax.jacfwd(moved)(0.5)
    p = np.asarray(moved(0.5))
    np.testing.assert_allclose(dp, p / np.linalg.norm(p), atol=1e-9)  # the point moves radially


def test_a_table_projects_by_incidence_and_classifies_its_own_points():
    model = lower(Box(size=Vector([0.5, 0.4, 0.3], free=True, name="s")))
    theta = jnp.asarray(model.theta)
    kinds = [k for k, _ in surfaces(model)]
    faces = [i for i, k in enumerate(kinds) if k == "patch"]  # +x, -x, +y, -y, +z, -z
    seeds = np.array([[0.7, 0.1, 0.0], [0.6, 0.5, 0.1], [0.4, 0.5, 0.4], [0.0, 0.0, 0.0]])
    incidence = [[faces[0]], [faces[0], faces[2]], [faces[0], faces[2], faces[4]], []]
    on = np.asarray(project_table(model, theta, seeds, incidence))
    np.testing.assert_allclose(on[0][0], 0.5, atol=1e-9)
    np.testing.assert_allclose(on[1][:2], [0.5, 0.4], atol=1e-9)
    np.testing.assert_allclose(on[2], [0.5, 0.4, 0.3], atol=1e-9)
    np.testing.assert_allclose(on[3], seeds[3])
    found = classify(model, theta, on, tolerance=1e-6)
    assert [sorted(row) for row in found] == [sorted(row) for row in incidence]
    # the box's extent blend owns nothing on the boundary: no band, no rim
    assert not any(kinds[s] in ("band", "rim") for row in found for s in row)
    # and the design derivative reaches the projected points
    slope = jax.jacfwd(lambda th: project_table(model, th, seeds[:1], incidence[:1])[0, 0])(theta)
    np.testing.assert_allclose(slope, [1.0, 0.0, 0.0], atol=1e-9)


def test_a_sphere_in_a_table_matches_the_plain_kernel():
    model = lower(Sphere(radius=Scalar(0.6, free=True, name="r")))
    seeds = np.random.default_rng(2).normal(size=(16, 3)) * 0.3
    via_table = np.asarray(project_table(model, jnp.asarray(model.theta), seeds, [[0]] * 16))
    plain = np.asarray(project([_sphere([0, 0, 0], 0.6)], seeds))
    np.testing.assert_allclose(via_table, plain, atol=1e-9)
