"""Reparametrization and normalization utilities for optimization.

Two composable layers sit between raw parameters and the optimizer:

1. **Reparametrization** (``to_unconstrained`` / ``to_constrained``): enforces
   bounds via smooth invertible maps (sigmoid for fully-bounded, softplus for
   lower-bounded) so the optimizer never violates constraints.

2. **Normalization** (``normalize`` / ``unnormalize``): divides unconstrained
   params by per-param scales so everything is O(1) in optimizer space.  This
   makes weight decay affect all params proportionally regardless of their
   natural magnitude.

Typical usage::

    from jaxcad import extract_parameters
    from jaxcad.parametrization import compute_param_scales, to_normalized, from_normalized

    free_params, fixed_params, metadata = extract_parameters(scene)

    # One-time Python-level setup
    scales = compute_param_scales(metadata, scene_scale=2.0)

    # Enter optimizer space
    params = to_normalized(free_params, metadata, scales)
    opt_state = optimizer.init(params)

    # Inside loss_fn (JAX-traceable)
    def loss_fn(normalized_p):
        p = from_normalized(normalized_p, metadata, scales)
        return my_loss(render_fn(p))

    # Round-trip identity (up to float32 precision)
    # from_normalized(to_normalized(free_params, metadata, scales), metadata, scales)
    # ≈ free_params
"""

import jax
import jax.numpy as jnp

__all__ = [
    "to_unconstrained",
    "to_constrained",
    "compute_param_scales",
    "normalize",
    "unnormalize",
    "to_normalized",
    "from_normalized",
]


# ---------------------------------------------------------------------------
# Reparametrization
# ---------------------------------------------------------------------------


def _softplus_inverse(y):
    """Numerically stable inverse of softplus: log(exp(y) - 1)."""
    return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(jnp.maximum(y, 1e-6))))


def to_unconstrained(params, metadata):
    """Map constrained ``free_params`` to an unconstrained dict.

    Applies per-param invertible transforms based on ``metadata[k].bounds``:

    * ``[lo, hi]`` — logit:      ``u = log(v / (1 - v))`` where ``v = (param - lo) / (hi - lo)``
    * ``[lo, ∞)``  — softplus⁻¹: ``u = log(exp(param - lo) - 1)``
    * unbounded    — identity:   ``u = param``

    Args:
        params: ``dict[str, Array]`` of raw parameter values (in constrained space).
        metadata: ``dict[str, Parameter]`` as returned by :func:`extract_parameters`.

    Returns:
        ``dict[str, Array]`` with the same keys, values in unconstrained space.
    """
    result = {}
    for k, v in params.items():
        meta = metadata.get(k)
        bounds = meta.bounds if meta is not None else None
        if bounds is None:
            result[k] = v
        else:
            lo, hi = bounds
            if lo is not None and hi is not None:
                v_norm = jnp.clip((v - lo) / (hi - lo), 1e-6, 1 - 1e-6)
                result[k] = jnp.log(v_norm / (1 - v_norm))  # logit
            elif lo is not None:
                result[k] = _softplus_inverse(jnp.maximum(v - lo, 1e-6))
            else:
                result[k] = v
    return result


def to_constrained(unconstrained, metadata):
    """Map unconstrained params back to constrained space.

    Inverse of :func:`to_unconstrained`.

    Args:
        unconstrained: ``dict[str, Array]`` in unconstrained space.
        metadata: ``dict[str, Parameter]`` as returned by :func:`extract_parameters`.

    Returns:
        ``dict[str, Array]`` with values clamped/mapped to their declared bounds.
    """
    result = {}
    for k, v in unconstrained.items():
        meta = metadata.get(k)
        bounds = meta.bounds if meta is not None else None
        if bounds is None:
            result[k] = v
        else:
            lo, hi = bounds
            if lo is not None and hi is not None:
                result[k] = lo + (hi - lo) * jax.nn.sigmoid(v)
            elif lo is not None:
                result[k] = lo + jax.nn.softplus(v)
            else:
                result[k] = v
    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def compute_param_scales(metadata, scene_scale=1.0):
    """Compute per-parameter normalization scales.

    Pure Python — reads ``metadata[k].bounds`` and returns a dict of JAX
    arrays (same shape as each parameter value).  Call once before
    :func:`jax.jit`; the returned scales are static constants.

    Scale assignment:

    * ``[lo, hi]`` fully-bounded → sigmoid maps to ``(0, 1)``, logit maps
      back to ``(-∞, +∞)`` and is already O(1–4) → scale = ``1.0``
    * ``[lo, ∞)`` lower-bounded or unbounded → values are O(``scene_scale``)
      after the softplus⁻¹/identity → scale = ``scene_scale``

    Args:
        metadata: ``dict[str, Parameter]`` as returned by :func:`extract_parameters`.
        scene_scale: Characteristic length of the scene (default ``1.0``).
            Set to the typical coordinate magnitude, e.g. ``2.0`` for a scene
            where objects span ±2 units.

    Returns:
        ``dict[str, Array]`` mapping each parameter name to its scale array.
    """
    scales = {}
    for k, meta in metadata.items():
        bounds = meta.bounds
        if bounds is not None:
            lo, hi = bounds
            if lo is not None and hi is not None:
                scales[k] = jnp.ones_like(meta.value)
            else:
                scales[k] = jnp.full_like(meta.value, float(scene_scale))
        else:
            scales[k] = jnp.full_like(meta.value, float(scene_scale))
    return scales


def normalize(unconstrained, scales):
    """Divide unconstrained params by their scales → O(1) optimizer space.

    Args:
        unconstrained: ``dict[str, Array]`` in unconstrained space.
        scales: ``dict[str, Array]`` from :func:`compute_param_scales`.

    Returns:
        ``dict[str, Array]`` with each value divided by its scale.
    """
    return {k: v / scales[k] for k, v in unconstrained.items()}


def unnormalize(normalized, scales):
    """Multiply normalized params by their scales → unconstrained space.

    Inverse of :func:`normalize`.

    Args:
        normalized: ``dict[str, Array]`` in normalized O(1) space.
        scales: ``dict[str, Array]`` from :func:`compute_param_scales`.

    Returns:
        ``dict[str, Array]`` in unconstrained space.
    """
    return {k: v * scales[k] for k, v in normalized.items()}


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def to_normalized(free_params, metadata, scales):
    """``to_unconstrained`` followed by ``normalize``.

    Convenience wrapper for entering optimizer space from raw param values.

    Args:
        free_params: ``dict[str, Array]`` of raw (constrained) parameter values.
        metadata: ``dict[str, Parameter]`` from :func:`extract_parameters`.
        scales: ``dict[str, Array]`` from :func:`compute_param_scales`.

    Returns:
        ``dict[str, Array]`` ready to pass to the optimizer.
    """
    return normalize(to_unconstrained(free_params, metadata), scales)


def from_normalized(normalized, metadata, scales):
    """``unnormalize`` followed by ``to_constrained``.

    Convenience wrapper for recovering constrained param values from
    optimizer space, e.g. inside a loss function.

    Args:
        normalized: ``dict[str, Array]`` in normalized O(1) optimizer space.
        metadata: ``dict[str, Parameter]`` from :func:`extract_parameters`.
        scales: ``dict[str, Array]`` from :func:`compute_param_scales`.

    Returns:
        ``dict[str, Array]`` of constrained parameter values, suitable for
        passing directly to the render function.
    """
    return to_constrained(unnormalize(normalized, scales), metadata)
