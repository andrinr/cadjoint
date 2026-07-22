"""Tests for optimization-space parameter mappings."""

import jax
import jax.numpy as jnp
import pytest

from jaxcad.geometry.parameters import Scalar
from jaxcad.parametrization import (
    compute_param_scales,
    from_normalized,
    to_constrained,
    to_normalized,
    to_unconstrained,
)


@pytest.mark.parametrize(
    ("bounds", "value"),
    [
        ((-2.0, 3.0), 0.75),
        ((1.0, None), 4.5),
        ((None, 5.0), 1.5),
        ((None, None), -2.5),
        (None, 8.0),
    ],
)
def test_bounded_mappings_round_trip(bounds, value):
    metadata = {"value": Scalar(value, free=True, name="value", bounds=bounds)}
    params = {"value": jnp.array(value)}

    unconstrained = to_unconstrained(params, metadata)
    restored = to_constrained(unconstrained, metadata)

    assert jnp.allclose(restored["value"], params["value"], rtol=1e-5, atol=1e-5)


def test_upper_bound_mapping_enforces_bound_and_is_jittable():
    metadata = {"value": Scalar(2.0, free=True, name="value", bounds=(None, 3.0))}
    constrain = jax.jit(lambda value: to_constrained({"value": value}, metadata)["value"])

    values = jax.vmap(constrain)(jnp.array([-100.0, 0.0, 100.0]))

    assert jnp.all(values <= 3.0)
    assert jnp.isfinite(values).all()


def test_softplus_inverse_stays_finite_for_large_values_and_gradients():
    metadata = {"value": Scalar(1.0, free=True, name="value", bounds=(0.0, None))}

    def inverse(value):
        return to_unconstrained({"value": value}, metadata)["value"]

    assert jnp.isfinite(inverse(jnp.array(1e4)))
    assert jnp.isfinite(jax.grad(inverse)(jnp.array(1e4)))


def test_normalized_mapping_round_trip_with_upper_bound():
    metadata = {"value": Scalar(1.5, free=True, name="value", bounds=(None, 5.0))}
    params = {"value": jnp.array(1.5)}
    scales = compute_param_scales(metadata, scene_scale=2.0)

    normalized = to_normalized(params, metadata, scales)

    assert jnp.allclose(from_normalized(normalized, metadata, scales)["value"], 1.5)


@pytest.mark.parametrize("scene_scale", [0.0, -1.0, jnp.inf, jnp.nan])
def test_compute_param_scales_rejects_invalid_scene_scale(scene_scale):
    metadata = {"value": Scalar(1.0, free=True, name="value")}

    with pytest.raises(ValueError, match="positive, finite"):
        compute_param_scales(metadata, scene_scale=scene_scale)


@pytest.mark.parametrize("bounds", [(1.0, 1.0), (2.0, -1.0), (0.0,)])
def test_invalid_bounds_fail_with_parameter_context(bounds):
    metadata = {"radius": Scalar(1.0, free=True, name="radius", bounds=bounds)}

    with pytest.raises(ValueError, match="radius"):
        to_unconstrained({"radius": jnp.array(1.0)}, metadata)
