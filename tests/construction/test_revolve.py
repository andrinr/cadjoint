"""A revolved profile with an edge on the axis: the axis edge is not surface (#50)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cadjoint.construction import PolygonProfile, SketchPlane, revolve
from cadjoint.sdf import Shell
from cadjoint.sdf._lowering import scalar_lowering
from cadjoint.sdf.primitives.polygon import RevolvedPolygon

# A cup's cross-section: the left edge, (0, 50) -> (0, 0), lies on the axis.
CUP = [(0.0, 0.0), (30.0, 0.0), (40.0, 50.0), (0.0, 50.0)]
# The same solid drawn across the axis: no edge on it, so the polygon
# distance at (r, h) is the 3D distance with nothing to correct.
CUP_MIRRORED = [(-30.0, 0.0), (30.0, 0.0), (40.0, 50.0), (-40.0, 50.0)]


def _cup(vertices):
    return revolve(PolygonProfile(vertices, plane=SketchPlane(normal=(0, 1, 0)), name="cup"))


def test_a_shelled_cup_is_empty_on_its_axis():
    """The issue's repro: Shell must not grow a wall along the axis."""
    scene = Shell(_cup(CUP), 2.0)
    axis = jnp.array([[0.0, 0.0, z] for z in (5.0, 25.0, 45.0)])
    assert np.all(np.asarray(scene(axis)) > 1.0)


def test_the_axis_edge_reads_the_mirrored_profiles_distance():
    """Inside and out, the half profile and the mirrored one are one solid."""
    rng = np.random.default_rng(0)
    points = jnp.asarray(rng.uniform([-50, -50, -10], [50, 50, 60], size=(2000, 3)))
    # Points right next to the axis, inside the cup, where the old field read -r.
    near = rng.uniform([-0.5, -0.5, 1.0], [0.5, 0.5, 49.0], size=(200, 3))
    points = jnp.concatenate([points, jnp.asarray(near)])
    half, mirrored = _cup(CUP), _cup(CUP_MIRRORED)
    np.testing.assert_allclose(half(points), mirrored(points), atol=1e-4)


def test_the_unrolled_form_excludes_the_axis_edge_too():
    """The shader backends trace the unrolled loop; it must agree with the stacked one."""
    rev = RevolvedPolygon([jnp.array(v) for v in CUP])
    rng = np.random.default_rng(1)
    points = jnp.asarray(rng.uniform([-45, -5, -45], [45, 55, 45], size=(500, 3)))
    stacked = rev(points)
    with scalar_lowering():
        unrolled = rev(points)
    np.testing.assert_allclose(stacked, unrolled, atol=1e-5)
    # On the axis, mid-height, the nearest surface is the bottom or the top.
    assert float(rev(jnp.array([0.0, 25.0, 0.0]))) < -20.0


def test_an_offset_moves_the_axis_with_the_profile():
    """With an offset the axis is at x = -offset; an edge there is still interior."""
    shifted = [(x - 5.0, y) for x, y in CUP]
    rev = RevolvedPolygon([jnp.array(v) for v in shifted], offset=5.0)
    reference = RevolvedPolygon([jnp.array(v) for v in CUP])
    points = jnp.asarray(np.random.default_rng(2).uniform(-45, 55, size=(500, 3)))
    np.testing.assert_allclose(rev(points), reference(points), atol=1e-4)
