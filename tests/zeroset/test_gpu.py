"""``Solve`` on the GPU: seeds land on their surfaces, and where the CPU lands them.

The GPU projector runs the dual-number WGSL; the check is the host's own
oracle (:mod:`cadjoint.zeroset.evaluate`) re-evaluating the returned
points, exactly as the protocol's certificate does, plus agreement with a
CPU Newton from the same seeds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import Box, Sphere
from cadjoint.sdf.transforms import Translate
from cadjoint.zeroset import lower
from cadjoint.zeroset.evaluate import surfaces

pytest.importorskip("wgpu")
from cadjoint.zeroset.gpu import Projector  # noqa: E402


def _cpu_newton(model, seeds, incidence, iterations=32):
    """Minimum-norm Newton with JAX, the reference for the GPU kernel."""
    theta = jnp.asarray(model.theta)
    fields = [f for _, f in surfaces(model)]
    pts = np.array(seeds, float)
    for a, row in enumerate(incidence):
        p = jnp.asarray(pts[a])
        for _ in range(iterations):
            vals, grads = [], []
            for s in row:
                f = lambda q, s=s: fields[s](theta, q[None])[0]  # noqa: E731
                vals.append(float(f(p)))
                grads.append(np.asarray(jax.grad(f)(p)))
            J = np.array(grads)
            p = p - J.T @ np.linalg.solve(J @ J.T, np.array(vals))
        pts[a] = np.asarray(p)
    return pts


def _residual(model, points, incidence):
    theta = jnp.asarray(model.theta)
    fields = [f for _, f in surfaces(model)]
    return np.array(
        [
            max(abs(float(fields[s](theta, jnp.asarray(p)[None])[0])) for s in row)
            for p, row in zip(points, incidence)
        ]
    )


def test_face_points_land_on_a_sphere():
    model = lower(
        Translate(Sphere(radius=Scalar(0.8, free=True, name="r")), Vector([0.3, -0.2, 0.1]))
    )
    rng = np.random.default_rng(0)
    seeds = np.array([0.3, -0.2, 0.1]) + rng.normal(size=(256, 3)) * 0.6
    proj = Projector(model).solve(model.theta, seeds, [[0]] * len(seeds))
    assert proj.residual.max() < 1e-6
    np.testing.assert_allclose(
        np.linalg.norm(proj.points - [0.3, -0.2, 0.1], axis=1), 0.8, atol=1e-6
    )
    assert _residual(model, proj.points, [[0]] * len(seeds)).max() < 1e-6


def test_edge_and_vertex_points_land_on_a_box():
    model = lower(Box(size=Vector([0.5, 0.4, 0.3], free=True, name="s")))
    kinds = [k for k, _ in surfaces(model)]
    faces = [i for i, k in enumerate(kinds) if k == "patch"]  # +x, -x, +y, -y, +z, -z
    seeds = np.array([[0.7, 0.6, 0.0], [0.6, 0.5, 0.4], [-0.1, 0.5, -0.4]])
    incidence = [[faces[0], faces[2]], [faces[0], faces[2], faces[4]], [faces[2], faces[5]]]
    proj = Projector(model).solve(model.theta, seeds, incidence)
    assert proj.residual.max() < 1e-6
    np.testing.assert_allclose(proj.points[0][:2], [0.5, 0.4], atol=1e-6)  # the +x/+y edge, z free
    np.testing.assert_allclose(proj.points[1], [0.5, 0.4, 0.3], atol=1e-6)  # the corner
    np.testing.assert_allclose(proj.points[2][1:], [0.4, -0.3], atol=1e-6)  # the +y/-z edge


def test_the_gpu_lands_where_a_cpu_newton_does():
    model = lower(
        Union(
            Sphere(radius=Scalar(0.7, free=True, name="a")),
            Translate(Box(size=[0.4, 0.4, 0.4]), Vector([0.6, 0.0, 0.0], free=True, name="o")),
            smoothness=Scalar(0.05, free=True, name="k"),
        )
    )
    kinds = [k for k, _ in surfaces(model)]
    band = kinds.index("band")
    sphere = [i for i, k in enumerate(kinds) if k == "patch"][0]
    rng = np.random.default_rng(3)
    incidence = [[sphere]] * 32 + [[band]] * 32
    # seeds near the surface, where the basin is unambiguous: a far seed may
    # legitimately land elsewhere on the same surface, in either implementation
    near = _cpu_newton(model, rng.normal(size=(64, 3)) * 0.7, incidence)
    seeds = near + rng.normal(size=near.shape) * 0.03
    proj = Projector(model).solve(model.theta, seeds, incidence)
    cpu = _cpu_newton(model, seeds, incidence)
    assert proj.residual.max() < 1e-5
    np.testing.assert_allclose(proj.points, cpu, atol=2e-5)


def test_a_parameter_change_is_a_buffer_write():
    model = lower(Sphere(radius=Scalar(0.5, free=True, name="r")))
    solver = Projector(model)
    seeds = np.random.default_rng(1).normal(size=(128, 3))
    for r in (0.5, 0.9, 1.3):
        proj = solver.solve([r], seeds, [[0]] * len(seeds))
        np.testing.assert_allclose(np.linalg.norm(proj.points, axis=1), r, atol=1e-6)
