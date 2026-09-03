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


# ── rank deficiency, stated exactly rather than by implication ──────────────
#
# The gusset above is *redundant*: the extra statement is implied by the
# others, so ``J`` loses rank through the geometry.  The two cases below make
# the deficiency structural instead — a literally duplicated row, and a row
# that is identically zero — so ``J Jᵀ`` is singular by construction in every
# dtype, with no roundoff to hide behind.  Both must give a finite,
# minimum-norm correction rather than NaN, in float32 and float64 alike.


def _duplicated(tag: str):
    """The same relation stated twice: two identical rows in ``J``."""
    left = Vector2(value=[0.0, 0.30], free=True, name=f"{tag}_left")
    right = Vector2(value=[0.8, 0.05], free=True, name=f"{tag}_right")
    profile = PolygonProfile([left, right, [0.4, 0.9]], name=tag)
    FixedConstraint(left, [0.0, 0.0])
    HorizontalConstraint(left, right)
    HorizontalConstraint(left, right)  # the duplicate
    return profile, (left, right)


def _degenerate(tag: str):
    """A relation between a point and itself: a row of ``J`` that is all zero."""
    lone = Vector2(value=[0.0, 0.30], free=True, name=f"{tag}_lone")
    mate = Vector2(value=[0.8, 0.05], free=True, name=f"{tag}_mate")
    profile = PolygonProfile([lone, mate, [0.4, 0.9]], name=tag)
    FixedConstraint(lone, [0.0, 0.0])
    HorizontalConstraint(lone, mate)
    HorizontalConstraint(lone, lone)  # residual and gradient both identically 0
    return profile, (lone, mate)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("build", [_duplicated, _degenerate], ids=["duplicated", "degenerate"])
def test_a_rank_deficient_system_stays_finite(build, x64):
    with _x64(x64):
        profile, points = build(f"rank_{build.__name__}_{int(x64)}")
        satisfy_constraints(profile, steps=3)
        assert np.isfinite(_values(points)).all()


@pytest.mark.parametrize("x64", [False, True])
def test_a_duplicated_relation_is_still_enforced(x64):
    """Rank deficiency must not be survived by declining to solve."""
    with _x64(x64):
        profile, (left, right) = _duplicated(f"dup_solve_{int(x64)}")
        satisfy_constraints(profile, steps=3)
        left_v, right_v = np.asarray(left.value), np.asarray(right.value)
        assert np.isfinite(left_v).all() and np.isfinite(right_v).all()
        assert left_v[1] == pytest.approx(right_v[1], abs=1e-4)
        np.testing.assert_allclose(left_v, [0.0, 0.0], atol=1e-4)


@pytest.mark.parametrize("x64", [False, True])
def test_a_duplicated_relation_gives_the_same_answer_as_one_copy(x64):
    """Stating a relation twice must not move the sketch."""
    with _x64(x64):
        once_left = Vector2(value=[0.0, 0.30], free=True, name=f"once_l{int(x64)}")
        once_right = Vector2(value=[0.8, 0.05], free=True, name=f"once_r{int(x64)}")
        once = PolygonProfile([once_left, once_right, [0.4, 0.9]], name=f"once{int(x64)}")
        FixedConstraint(once_left, [0.0, 0.0])
        HorizontalConstraint(once_left, once_right)
        satisfy_constraints(once, steps=3)

        twice, twice_points = _duplicated(f"twice{int(x64)}")
        satisfy_constraints(twice, steps=3)

        np.testing.assert_allclose(
            _values((once_left, once_right)), _values(twice_points), atol=1e-5
        )
