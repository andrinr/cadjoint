"""A redundant constraint must not turn the model into NaN.

Real sketches are over-constrained all the time: a corner is stated to be
perpendicular even though a horizontal and a vertical already force it. That
makes the constraint Jacobian rank-deficient, ``J Jᵀ`` singular, and the exact
solve the Newton projection used to do returned NaN — which then propagated
into *every* free parameter in the program, not just the sketch's, and left a
scene whose geometry was NaN everywhere.

The failure was also dtype-dependent, which is the worst part: float32
roundoff hid the singularity, so a scene could render correctly and only go
NaN once the FEM path turned x64 on to mesh it. These tests pin both halves —
redundancy is tolerated, and it is tolerated in float64 too.
"""

from __future__ import annotations

import contextlib

import jax
import numpy as np
import pytest

from cadjoint.constraints import (
    DistanceConstraint,
    FixedConstraint,
    HorizontalConstraint,
    PerpendicularEdgesConstraint,
    PointOnLineConstraint,
    VerticalConstraint,
    satisfy_constraints,
)
from cadjoint.construction import PolygonProfile, SketchPlane, extrude
from cadjoint.geometry import Scalar, Vector2


@contextlib.contextmanager
def _x64(enabled: bool):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _gusset(redundant: bool, tag: str):
    """A gusset sketch, optionally carrying one constraint too many."""
    heel = Vector2(value=[-0.95, 0.16], free=True, name=f"{tag}_heel")
    toe = Vector2(value=[-0.60, 0.16], free=True, name=f"{tag}_toe")
    crest = Vector2(value=[-0.60, 0.56], free=True, name=f"{tag}_crest")
    slope = Vector2(value=[-0.78, 0.36], free=True, name=f"{tag}_slope")
    profile = PolygonProfile(
        [heel, toe, crest, slope],
        plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
        name=tag,
    )
    solid = extrude(profile, depth=0.11)

    FixedConstraint(heel, [-0.95, 0.16])
    HorizontalConstraint(heel, toe)
    PerpendicularEdgesConstraint(heel, toe, toe, crest)
    DistanceConstraint(toe, crest, Scalar(0.40))
    PointOnLineConstraint(slope, heel, crest)
    if redundant:
        # Implied by the horizontal root plus the perpendicular corner.
        VerticalConstraint(toe, crest)
    return solid, (heel, toe, crest, slope)


def _values(points):
    return np.stack([np.asarray(point.value) for point in points])


@pytest.mark.parametrize("x64", [False, True])
def test_a_redundant_constraint_still_solves(x64):
    with _x64(x64):
        solid, points = _gusset(redundant=True, tag=f"red{int(x64)}")
        satisfy_constraints(solid, steps=3)
        assert np.isfinite(_values(points)).all()


@pytest.mark.parametrize("x64", [False, True])
def test_redundancy_does_not_change_the_answer(x64):
    """The extra statement is consistent, so it must be a no-op."""
    with _x64(x64):
        lean, lean_points = _gusset(redundant=False, tag=f"lean{int(x64)}")
        satisfy_constraints(lean, steps=3)
        fat, fat_points = _gusset(redundant=True, tag=f"fat{int(x64)}")
        satisfy_constraints(fat, steps=3)
        np.testing.assert_allclose(_values(lean_points), _values(fat_points), atol=1e-5)


def test_the_constraints_are_actually_satisfied():
    """Guard against 'no NaN' passing on a solver that simply did nothing."""
    solid, (heel, toe, crest, _slope) = _gusset(redundant=True, tag="check")
    satisfy_constraints(solid, steps=4)
    heel_v, toe_v, crest_v = (np.asarray(p.value) for p in (heel, toe, crest))
    assert heel_v[1] == pytest.approx(toe_v[1], abs=1e-4)  # horizontal root
    assert np.linalg.norm(crest_v - toe_v) == pytest.approx(0.40, abs=1e-3)  # driving distance
    assert np.dot(toe_v - heel_v, crest_v - toe_v) == pytest.approx(0.0, abs=1e-4)  # square corner


def test_a_free_parameter_outside_the_sketch_is_left_finite():
    """The NaN used to escape the sketch and poison unrelated parameters."""
    with _x64(True):
        depth = Scalar(0.18, free=True, name="unrelated_depth")
        solid, _ = _gusset(redundant=True, tag="escape")
        square = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
        plate = extrude(PolygonProfile(square, name="plate"), depth=depth)
        satisfy_constraints(solid | plate, steps=3)
        assert np.isfinite(float(depth.value))
