"""In-place projection used by the browser sketch workflow."""

import jax.numpy as jnp
import pytest

from jaxcad.constraints import DistanceConstraint, FixedConstraint, satisfy_constraints
from jaxcad.constraints.solve import capture_constraint_solves
from jaxcad.construction import PolygonProfile


def test_satisfy_constraints_updates_an_underconstrained_profile_in_place():
    profile = PolygonProfile([[0, 0], [2, 0], [0, 1]], name="ui")
    FixedConstraint(profile.vertices[0], [0, 0])
    DistanceConstraint(profile.vertices[0], profile.vertices[1], 1.0)

    solved = satisfy_constraints(profile)

    assert set(solved) == {"ui_v0", "ui_v1", "ui_v2"}
    assert jnp.allclose(profile.vertices[0].value, jnp.array([0.0, 0.0]), atol=1e-5)
    assert jnp.allclose(jnp.linalg.norm(profile.vertices[1].value), 1.0, atol=1e-5)


@pytest.mark.parametrize(
    ("method", "steps"),
    [("newton", 2), ("adam", 48), ("sgd", 32)],
)
def test_satisfy_constraints_records_decreasing_loss(method, steps):
    profile = PolygonProfile([[0, 0], [2, 0], [0, 1]], name=f"ui_{method}")
    FixedConstraint(profile.vertices[0], [0, 0])
    DistanceConstraint(profile.vertices[0], profile.vertices[1], 1.0)

    with capture_constraint_solves() as reports:
        satisfy_constraints(profile, method=method, steps=steps)

    assert len(reports) == 1
    assert reports[0]["method"] == method
    assert reports[0]["iterations"] == steps
    assert len(reports[0]["losses"]) == steps + 1
    assert reports[0]["losses"][-1] < reports[0]["losses"][0]


def test_satisfy_constraints_rejects_invalid_controls():
    profile = PolygonProfile([[0, 0], [1, 0], [0, 1]], name="invalid")
    with pytest.raises(ValueError, match="positive integer"):
        satisfy_constraints(profile, steps=0)
    with pytest.raises(ValueError, match="method"):
        satisfy_constraints(profile, method="bfgs")
