"""Tests for 2D sketch constraints (horizontal, vertical, coincident, edges, ...)."""

import jax
import jax.numpy as jnp
import pytest

from jaxcad.constraints import (
    CoincidentConstraint,
    DistanceConstraint,
    EqualLengthConstraint,
    FixedConstraint,
    HorizontalConstraint,
    ParallelEdgesConstraint,
    PerpendicularEdgesConstraint,
    PointOnLineConstraint,
    VerticalConstraint,
    build_residual_fn,
    satisfy_constraints,
)
from jaxcad.constraints.residual import compute_param_vector
from jaxcad.constraints.solve import solve_constraints
from jaxcad.construction import PolygonProfile
from jaxcad.geometry.parameters import Vector2


def _points(*xys):
    return [Vector2(xy, free=True, name=f"skp{i}") for i, xy in enumerate(xys)]


def _edge_len(profile, i, j):
    return jnp.linalg.norm(profile.vertices[i].value - profile.vertices[j].value)


def _cross2(u, w):
    return u[0] * w[1] - u[1] * w[0]


# ---------------------------------------------------------------------------
# Residuals: zero when satisfied, nonzero otherwise
# ---------------------------------------------------------------------------


def test_horizontal_residual():
    p1, p2 = _points([0, 1], [3, 1])
    c = HorizontalConstraint(p1, p2)
    values = {"skp0": p1.xy, "skp1": p2.xy}

    assert jnp.abs(c.compute_residual(values)) < 1e-6

    values["skp1"] = jnp.array([3.0, 1.5])
    assert jnp.abs(c.compute_residual(values) - (-0.5)) < 1e-6


def test_vertical_residual():
    p1, p2 = _points([2, 0], [2, 5])
    c = VerticalConstraint(p1, p2)
    values = {"skp0": p1.xy, "skp1": p2.xy}

    assert jnp.abs(c.compute_residual(values)) < 1e-6

    values["skp0"] = jnp.array([2.75, 0.0])
    assert jnp.abs(c.compute_residual(values) - 0.75) < 1e-6


def test_coincident_residual():
    p1, p2 = _points([1, 2], [1, 2])
    c = CoincidentConstraint(p1, p2)
    values = {"skp0": p1.xy, "skp1": p2.xy}

    residual = c.compute_residual(values)
    assert residual.shape == (2,)
    assert jnp.allclose(residual, jnp.zeros(2), atol=1e-6)

    values["skp1"] = jnp.array([0.0, 3.0])
    assert jnp.allclose(c.compute_residual(values), jnp.array([1.0, -1.0]), atol=1e-6)


def test_equal_length_residual():
    a1, a2, b1, b2 = _points([0, 0], [3, 4], [1, 1], [6, 1])
    c = EqualLengthConstraint(a1, a2, b1, b2)
    values = {"skp0": a1.xy, "skp1": a2.xy, "skp2": b1.xy, "skp3": b2.xy}

    assert jnp.abs(c.compute_residual(values)) < 1e-6  # both length 5

    values["skp3"] = jnp.array([3.0, 1.0])  # second edge length 2
    assert jnp.abs(c.compute_residual(values) - 3.0) < 1e-6


def test_point_on_line_residual():
    p, l1, l2 = _points([1, 1], [0, 0], [2, 2])
    c = PointOnLineConstraint(p, l1, l2)
    values = {"skp0": p.xy, "skp1": l1.xy, "skp2": l2.xy}

    assert jnp.abs(c.compute_residual(values)) < 1e-6  # collinear

    values["skp0"] = jnp.array([1.0, 2.0])
    assert jnp.abs(c.compute_residual(values)) > 0.5


def test_parallel_edges_residual():
    a1, a2, b1, b2 = _points([0, 0], [2, 1], [1, 1], [5, 3])
    c = ParallelEdgesConstraint(a1, a2, b1, b2)
    values = {"skp0": a1.xy, "skp1": a2.xy, "skp2": b1.xy, "skp3": b2.xy}

    assert jnp.abs(c.compute_residual(values)) < 1e-6  # (2,1) parallel (4,2)

    values["skp3"] = jnp.array([5.0, 4.0])
    assert jnp.abs(c.compute_residual(values)) > 0.5


def test_perpendicular_edges_residual():
    a1, a2, b1, b2 = _points([0, 0], [2, 0], [1, 0], [1, 3])
    c = PerpendicularEdgesConstraint(a1, a2, b1, b2)
    values = {"skp0": a1.xy, "skp1": a2.xy, "skp2": b1.xy, "skp3": b2.xy}

    assert jnp.abs(c.compute_residual(values)) < 1e-6

    values["skp3"] = jnp.array([2.0, 3.0])
    assert jnp.abs(c.compute_residual(values)) > 0.5


@pytest.mark.parametrize(
    "make_constraint,expected_dof",
    [
        (lambda ps: HorizontalConstraint(ps[0], ps[1]), 1),
        (lambda ps: VerticalConstraint(ps[0], ps[1]), 1),
        (lambda ps: CoincidentConstraint(ps[0], ps[1]), 2),
        (lambda ps: EqualLengthConstraint(ps[0], ps[1], ps[2], ps[3]), 1),
        (lambda ps: PointOnLineConstraint(ps[0], ps[1], ps[2]), 1),
        (lambda ps: ParallelEdgesConstraint(ps[0], ps[1], ps[2], ps[3]), 1),
        (lambda ps: PerpendicularEdgesConstraint(ps[0], ps[1], ps[2], ps[3]), 1),
    ],
)
def test_dof_reduction(make_constraint, expected_dof):
    ps = _points([0, 0], [2, 1], [1, 3], [4, 2])
    c = make_constraint(ps)

    assert c.dof_reduction() == expected_dof
    for param in c.get_parameters():
        assert any(param is p for p in ps)


# ---------------------------------------------------------------------------
# satisfy_constraints drives each kind to satisfaction on a PolygonProfile
# ---------------------------------------------------------------------------


def test_satisfy_horizontal_on_profile():
    profile = PolygonProfile([[0, 0], [2, 0.7], [1, 2]], name="sk_h")
    HorizontalConstraint(profile.vertices[0], profile.vertices[1])

    satisfy_constraints(profile, method="newton")

    y0 = profile.vertices[0].value[1]
    y1 = profile.vertices[1].value[1]
    assert jnp.abs(y0 - y1) < 1e-4


def test_satisfy_vertical_on_profile():
    profile = PolygonProfile([[0, 0], [2, 0], [1.2, 2]], name="sk_v")
    VerticalConstraint(profile.vertices[0], profile.vertices[2])

    satisfy_constraints(profile, method="newton")

    x0 = profile.vertices[0].value[0]
    x2 = profile.vertices[2].value[0]
    assert jnp.abs(x0 - x2) < 1e-4


def test_satisfy_coincident_on_profile():
    profile = PolygonProfile([[0, 0], [2, 0], [2, 2], [0.5, 1.8]], name="sk_c")
    CoincidentConstraint(profile.vertices[2], profile.vertices[3])

    satisfy_constraints(profile, method="newton")

    assert jnp.allclose(profile.vertices[2].value, profile.vertices[3].value, atol=1e-4)


def test_satisfy_equal_length_on_profile():
    profile = PolygonProfile([[0, 0], [3, 0], [3, 1], [0, 1]], name="sk_el")
    EqualLengthConstraint(
        profile.vertices[0], profile.vertices[1], profile.vertices[1], profile.vertices[2]
    )

    satisfy_constraints(profile, method="newton")

    assert jnp.abs(_edge_len(profile, 0, 1) - _edge_len(profile, 1, 2)) < 1e-4


def test_satisfy_point_on_line_on_profile():
    profile = PolygonProfile([[0, 0], [2, 0], [1, 1]], name="sk_pol")
    PointOnLineConstraint(profile.vertices[2], profile.vertices[0], profile.vertices[1])

    satisfy_constraints(profile, method="newton")

    u = profile.vertices[2].value - profile.vertices[0].value
    w = profile.vertices[1].value - profile.vertices[0].value
    assert jnp.abs(_cross2(u, w)) < 1e-4


def test_satisfy_parallel_edges_on_profile():
    profile = PolygonProfile([[0, 0], [2, 0], [2.5, 1], [0, 1.4]], name="sk_pe")
    ParallelEdgesConstraint(
        profile.vertices[0], profile.vertices[1], profile.vertices[3], profile.vertices[2]
    )

    satisfy_constraints(profile, method="newton")

    u = profile.vertices[1].value - profile.vertices[0].value
    w = profile.vertices[2].value - profile.vertices[3].value
    assert jnp.abs(_cross2(u, w)) < 1e-4


def test_satisfy_perpendicular_edges_on_profile():
    profile = PolygonProfile([[0, 0], [2, 0.3], [2.4, 2], [0, 2]], name="sk_qe")
    PerpendicularEdgesConstraint(
        profile.vertices[0], profile.vertices[1], profile.vertices[1], profile.vertices[2]
    )

    satisfy_constraints(profile, method="newton")

    u = profile.vertices[1].value - profile.vertices[0].value
    w = profile.vertices[2].value - profile.vertices[1].value
    assert jnp.abs(jnp.dot(u, w)) < 1e-4


# ---------------------------------------------------------------------------
# solve_constraints DOF accounting (exact-DOF setups)
# ---------------------------------------------------------------------------


def test_solve_exact_dof_horizontal_vertical():
    """Triangle: 6 DOF, two anchors (4) + horizontal (1) + vertical (1) = 6."""
    profile = PolygonProfile([[0.2, 0.1], [2.0, 1.2], [0.4, 0.9]], name="exact_hv")
    v0, v1, v2 = profile.vertices
    FixedConstraint(v0, [0.0, 0.0])
    FixedConstraint(v1, [2.0, 1.0])
    HorizontalConstraint(v1, v2)
    VerticalConstraint(v0, v2)

    solved = solve_constraints(profile)

    assert jnp.allclose(solved["exact_hv_v0"], jnp.array([0.0, 0.0]), atol=1e-5)
    assert jnp.allclose(solved["exact_hv_v1"], jnp.array([2.0, 1.0]), atol=1e-5)
    assert jnp.allclose(solved["exact_hv_v2"], jnp.array([0.0, 1.0]), atol=1e-5)


def test_solve_exact_dof_perpendicular_edges():
    """Triangle: anchors (4) + perpendicular edges (1) + distance (1) = 6."""
    profile = PolygonProfile([[0.1, 0.0], [2.1, 0.1], [2.2, 0.8]], name="exact_pe")
    v0, v1, v2 = profile.vertices
    FixedConstraint(v0, [0.0, 0.0])
    FixedConstraint(v1, [2.0, 0.0])
    PerpendicularEdgesConstraint(v0, v1, v1, v2)
    DistanceConstraint(v1, v2, 1.0)

    solved = solve_constraints(profile)

    assert jnp.allclose(solved["exact_pe_v2"], jnp.array([2.0, 1.0]), atol=1e-5)


def test_solve_exact_dof_coincident():
    """Triangle: anchors (4) + coincident (2) = 6."""
    profile = PolygonProfile([[0.0, 0.0], [2.0, 0.0], [1.5, 0.5]], name="exact_co")
    v0, v1, v2 = profile.vertices
    FixedConstraint(v0, [0.0, 0.0])
    FixedConstraint(v1, [2.0, 1.0])
    CoincidentConstraint(v2, v1)

    solved = solve_constraints(profile)

    assert jnp.allclose(solved["exact_co_v2"], solved["exact_co_v1"], atol=1e-5)


def test_solve_under_constrained_raises():
    profile = PolygonProfile([[0, 0], [2, 0], [1, 1]], name="under_sk")
    HorizontalConstraint(profile.vertices[0], profile.vertices[1])

    with pytest.raises(ValueError, match="Under-constrained"):
        solve_constraints(profile)


# ---------------------------------------------------------------------------
# Gradients are finite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_constraint,n_points",
    [
        (lambda ps: HorizontalConstraint(ps[0], ps[1]), 2),
        (lambda ps: VerticalConstraint(ps[0], ps[1]), 2),
        (lambda ps: CoincidentConstraint(ps[0], ps[1]), 2),
        (lambda ps: EqualLengthConstraint(ps[0], ps[1], ps[2], ps[3]), 4),
        (lambda ps: PointOnLineConstraint(ps[0], ps[1], ps[2]), 3),
        (lambda ps: ParallelEdgesConstraint(ps[0], ps[1], ps[2], ps[3]), 4),
        (lambda ps: PerpendicularEdgesConstraint(ps[0], ps[1], ps[2], ps[3]), 4),
    ],
)
def test_gradients_finite(make_constraint, n_points):
    coords = [[0.3, 0.1], [2.1, 0.9], [1.4, 2.2], [3.7, 1.5]]
    ps = _points(*coords[:n_points])
    c = make_constraint(ps)

    meta = {p.name: p for p in ps}
    flat_fn = build_residual_fn([c], meta)
    x0 = compute_param_vector(meta)
    jacobian = jax.jacobian(flat_fn)(x0)

    assert jacobian.shape == (c.dof_reduction(), 2 * n_points)
    assert bool(jnp.all(jnp.isfinite(jacobian)))
