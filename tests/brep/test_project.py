"""The one projection kernel at all three arities, and its implicit adjoint.

The claims under test are the ones the rest of the package rests on:

- at one field the kernel *is* :func:`cadjoint.fem.motion.project_points`,
  bit for bit, guard included;
- at two fields it agrees with the viewer's seam projection;
- at all three arities the ``custom_vjp`` matches central differences, which
  is the whole point of writing the adjoint by the implicit-function theorem
  instead of unrolling the loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.brep.project import (
    field_residuals,
    project,
    project_batched,
    project_fields,
    stacked_fields,
    transversal,
)
from cadjoint.fem.motion import project_points

# Two planes and a sphere whose common solutions are easy to write down.
_PARAMS = {
    "radius": jnp.asarray(1.3),
    "height": jnp.asarray(0.4),
    "offset": jnp.asarray(-0.2),
}


def _sphere(params, point):
    return jnp.linalg.norm(point) - params["radius"]


def _plane_z(params, point):
    return point[2] - params["height"]


def _plane_x(params, point):
    return point[0] - params["offset"]


def _stack(*fields):
    def field_fn(params, point):
        return jnp.stack([field(params, point).reshape(()) for field in fields])

    return field_fn


def _seeds(count: int, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return jnp.asarray(directions * (1.3 + 0.05 * rng.normal(size=(count, 1))), dtype=jnp.float32)


def test_one_field_reproduces_project_points():
    """The same Newton step, to float32 round-off.

    ``project_points`` divides by ``|grad f|^2``; the kernel solves the same
    one-by-one system with ``jnp.linalg.solve``, which is the same arithmetic
    in a different order and lands within an ulp.
    """
    points = _seeds(8)
    ours = project(_stack(_sphere), _PARAMS, points, max_step=1.0)
    theirs = project_points(lambda p: jnp.linalg.norm(p) - _PARAMS["radius"], points, 1.0)
    assert np.allclose(np.asarray(ours), np.asarray(theirs), rtol=1e-6, atol=1e-7)


def test_one_field_gradient_matches_project_points():
    points = _seeds(8)

    def ours(radius):
        return jnp.sum(
            project(_stack(_sphere), {**_PARAMS, "radius": radius}, points, max_step=1.0) ** 2
        )

    def theirs(radius):
        return jnp.sum(project_points(lambda p: jnp.linalg.norm(p) - radius, points, 1.0) ** 2)

    radius = jnp.asarray(1.3)
    assert float(jax.grad(ours)(radius)) == pytest.approx(float(jax.grad(theirs)(radius)), rel=1e-6)


def test_two_fields_reproduce_the_viewer_seam_projection():
    """The overlay's seam solver and the kernel land on the same curve."""
    from cadjoint.viewer._edge_overlay import _project_to_seam

    sphere = lambda p: jnp.linalg.norm(p) - 1.3  # noqa: E731
    plane = lambda p: p[2] - 0.4  # noqa: E731
    # Seeds a fifth of a cell off the true seam circle, as dual contouring
    # would leave them.
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    seam_radius = np.sqrt(1.3**2 - 0.4**2)
    rng = np.random.default_rng(3)
    seeds = np.stack(
        [seam_radius * np.cos(angles), seam_radius * np.sin(angles), np.full(12, 0.4)], axis=1
    ) + 0.02 * rng.normal(size=(12, 3))

    ours = project_fields([sphere, plane], seeds, max_step=0.4)
    theirs = _project_to_seam([sphere, plane], seeds, 0.4)
    assert np.allclose(ours, theirs, atol=1e-5)
    # Both are on the circle x^2 + y^2 = 1.3^2 - 0.4^2 at z = 0.4.
    assert np.allclose(np.linalg.norm(ours[:, :2], axis=1), seam_radius, atol=1e-5)
    assert np.allclose(ours[:, 2], 0.4, atol=1e-6)


def test_three_fields_solve_the_triple_point():
    seeds = np.array([[0.0, 1.2, 0.5], [0.0, 1.1, 0.3]])
    solved = project_fields([_wrap(_sphere), _wrap(_plane_z), _wrap(_plane_x)], seeds, max_step=1.0)
    expected_y = np.sqrt(1.3**2 - 0.4**2 - 0.2**2)
    assert np.allclose(solved[:, 0], -0.2, atol=1e-6)
    assert np.allclose(solved[:, 2], 0.4, atol=1e-6)
    assert np.allclose(np.abs(solved[:, 1]), expected_y, atol=1e-5)
    assert (
        float(field_residuals([_wrap(_sphere), _wrap(_plane_z), _wrap(_plane_x)], solved).max())
        < 1e-6
    )


def _wrap(field):
    return lambda point: field(_PARAMS, point)


@pytest.mark.parametrize(
    ("arity", "fields", "seeds"),
    [
        (1, (_sphere,), np.array([[0.9, 0.4, 0.6], [-0.7, 0.8, 0.3]])),
        (2, (_sphere, _plane_z), np.array([[1.1, 0.4, 0.41], [-1.0, -0.5, 0.39]])),
        (3, (_sphere, _plane_z, _plane_x), np.array([[-0.2, 1.2, 0.41], [-0.2, -1.2, 0.39]])),
    ],
)
def test_the_adjoint_matches_central_differences(arity, fields, seeds):
    """The implicit-function VJP against finite differences, per arity."""
    field_fn = _stack(*fields)
    points = jnp.asarray(seeds, dtype=jnp.float32)
    weights = jnp.asarray(np.array([[0.3, -0.7, 1.1], [0.5, 0.2, -0.9]]), dtype=jnp.float32)

    def loss(params):
        return jnp.sum(weights * project(field_fn, params, points, max_step=1.0))

    gradient = jax.grad(loss)(_PARAMS)
    table = {}
    for name in _PARAMS:
        step = 1e-3
        plus = loss({**_PARAMS, name: _PARAMS[name] + step})
        minus = loss({**_PARAMS, name: _PARAMS[name] - step})
        table[name] = (float(gradient[name]), float((plus - minus) / (2.0 * step)))
    for name, (analytic, numeric) in table.items():
        assert analytic == pytest.approx(numeric, abs=2e-3, rel=2e-3), (arity, name, table)


def test_the_guard_suppresses_a_dead_gradient_in_both_passes():
    """A field with no usable gradient moves nothing and carries no derivative."""

    def dead(params, point):
        return jnp.stack([params["radius"] * 0.0 + 0.0 * point[0]])

    points = jnp.asarray([[0.3, 0.4, 0.5]], dtype=jnp.float32)
    solved = project(dead, _PARAMS, points, max_step=1.0)
    assert np.array_equal(np.asarray(solved), np.asarray(points))
    gradient = jax.grad(lambda p: jnp.sum(project(dead, p, points, max_step=1.0)))(_PARAMS)
    assert float(gradient["radius"]) == 0.0


def test_tangent_patches_are_not_transversal():
    """Two coincident spheres have no intersection curve to project onto."""
    sphere = lambda p: jnp.linalg.norm(p) - 1.3  # noqa: E731
    seeds = np.asarray(_seeds(4, seed=7), dtype=np.float64)
    assert not transversal([sphere, sphere], seeds).any()
    assert np.allclose(project_fields([sphere, sphere], seeds, max_step=0.4), seeds)


def test_batched_projection_matches_one_call_per_subset():
    """``project_batched`` is the same answer as looping, in one program."""
    fields = [_wrap(_sphere), _wrap(_plane_z), _wrap(_plane_x)]
    seeds = np.array([[0.9, 0.9, 0.41], [-0.2, 1.2, 0.39], [-0.2, 0.9, 0.41]])
    members = np.array([[0, 1], [0, 2], [1, 2]])
    batched = project_batched(fields, members, seeds, max_step=1.0)
    for row, member in enumerate(members):
        subset = [fields[index] for index in member.tolist()]
        single = project_fields(subset, seeds[row : row + 1], max_step=1.0)
        assert np.allclose(batched[row], single[0], atol=1e-6)


def test_project_validates_its_inputs():
    with pytest.raises(ValueError, match="shaped"):
        project(_stack(_sphere), _PARAMS, jnp.zeros((3,)), max_step=1.0)
    with pytest.raises(ValueError, match="max_step"):
        project(_stack(_sphere), _PARAMS, jnp.zeros((2, 3)), max_step=0.0)
    with pytest.raises(ValueError, match="steps"):
        project(_stack(_sphere), _PARAMS, jnp.zeros((2, 3)), max_step=1.0, steps=0)
    with pytest.raises(ValueError, match="1 to 3"):
        project_fields([_wrap(_sphere)] * 4, np.zeros((1, 3)), max_step=1.0)


def test_stacked_fields_ignores_its_parameter_slot():
    field_fn = stacked_fields([_wrap(_sphere)])
    value = field_fn(None, jnp.asarray([1.3, 0.0, 0.0], dtype=jnp.float32))
    assert value.shape == (1,)
    assert float(value[0]) == pytest.approx(0.0, abs=1e-6)
