"""A refresh moves an extraction's points to a new design at fixed topology."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import Box, Sphere
from cadjoint.sdf.transforms import Translate
from cadjoint.zeroset import lower
from cadjoint.zeroset.evaluate import field

pytest.importorskip("wgpu")
from cadjoint.zeroset.gpu import Projector  # noqa: E402
from cadjoint.zeroset.refresh import Overlay, theta_from_values  # noqa: E402


def _box_points(size, n=400, seed=0):
    """Points on a box's faces and edges, from its half-extents."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(n, 3)) * size
    axis = rng.integers(0, 3, n)
    pts[np.arange(n), axis] = np.sign(pts[np.arange(n), axis]) * size[axis]
    quarter = np.arange(n // 4)
    pts[quarter, (axis[quarter] + 1) % 3] = size[(axis[quarter] + 1) % 3]  # a quarter on edges
    return pts


def test_theta_from_values_writes_whole_parameters():
    model = lower(
        Translate(
            Sphere(radius=Scalar(0.5, free=True, name="r")),
            Vector([0, 0, 0], free=True, name="o"),
        )
    )
    theta = theta_from_values(model, {"o": [1.0, 2.0, 3.0], "unknown": 9.0})
    assert dict(zip(model.names, theta)) == {"r": 0.5, "o[0]": 1.0, "o[1]": 2.0, "o[2]": 3.0}


def test_a_box_follows_its_size():
    size = np.array([0.5, 0.4, 0.3])
    model = lower(Box(size=Vector(size.tolist(), free=True, name="s")))
    overlay = Overlay(model, _box_points(size))
    assert overlay.placed.all()
    new = np.array([0.7, 0.5, 0.45])  # every face grows, so no point falls off its face
    moved = overlay.refresh(theta_from_values(model, {"s": new}))
    assert moved.stale == 0.0
    # every point is on the new box's boundary: on a face in at least one axis, inside in the rest
    on_face = np.isclose(np.abs(moved.points), new, atol=1e-5)
    assert on_face.any(axis=1).all()
    assert (np.abs(moved.points) <= new + 1e-5).all()
    # the quarter seeded on edges stayed on edges
    assert (on_face[:100].sum(axis=1) >= 2).all()
    # a face that shrinks drops the points beyond its new extent: refused, not moved off the boundary
    shrunk = overlay.refresh(theta_from_values(model, {"s": [0.7, 0.2, 0.45]}))
    assert 0.05 < shrunk.stale < 0.5
    kept = np.abs(shrunk.points[shrunk.certified])
    assert (kept <= np.array([0.7, 0.2, 0.45]) + 1e-5).all()


def test_a_refresh_certifies_against_the_new_topology():
    model = lower(
        Union(
            Sphere(radius=Scalar(0.7, free=True, name="r")),
            Translate(Box(size=[0.3, 0.3, 0.3]), Vector([0.8, 0.0, 0.0], free=True, name="o")),
            smoothness=0.05,
        )
    )
    whole = field(model)
    rng = np.random.default_rng(2)
    seeds = rng.normal(size=(300, 3)) * 0.8
    seeds[:, 0] += 0.3
    # boundary points: the band's own field is the whole model, so projecting onto it lands anywhere
    kinds = [k for k, _w, _x in Projector(model).surfaces]
    band = kinds.index("band")
    on = Projector(model).solve(model.theta, seeds, [[band]] * len(seeds)).points
    overlay = Overlay(model, on)
    assert overlay.placed.mean() > 0.95
    # a small move keeps almost every point's surfaces: only those within the
    # fillet's own width of its rim change surface, and they are refused
    small = overlay.refresh(theta_from_values(model, {"r": 0.71, "o": [0.82, 0.0, 0.0]}))
    assert small.stale < 0.1
    certified = jnp.asarray(small.points[small.certified])
    np.testing.assert_array_less(np.abs(whole(jnp.asarray(small.theta), certified)), 1e-4)
    # pulling the box clear of the sphere changes the topology: the fillet's points are refused
    far = overlay.refresh(theta_from_values(model, {"r": 0.71, "o": [2.5, 0.0, 0.0]}))
    assert far.stale > 0.1
