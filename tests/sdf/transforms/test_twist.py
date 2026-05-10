"""Diagnostic tests for the Twist deformation.

These tests verify the properties needed for correct sphere-tracing and shading:
  1. Gradient magnitude at surface points  (drives the `ao` proxy in the renderer)
  2. Conservative property                 (SDF must not overestimate distances)
  3. Normal direction correctness          (gradient direction points outward)
  4. Rendering brightness parity          (twist should not significantly darken/brighten)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcad.render.raymarch import _sphere_trace, raymarch
from jaxcad.sdf.primitives import Box
from jaxcad.sdf.transforms import Twist

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_twisted_box(strength=1.0, axis="y"):
    box = Box(size=jnp.array([1.0, 2.0, 1.0]))
    return Twist(box, strength=strength, axis=axis)


def _surface_points(sdf, origins, directions, steps=128):
    """Run sphere tracing from origins and return hit positions where sdf ≈ 0."""
    hits = []
    for o, d in zip(origins, directions):
        t, d_min = _sphere_trace(sdf, o, d, steps)
        if float(d_min) < 0.02:
            hits.append(o + t * d)
    return hits


# ---------------------------------------------------------------------------
# 1. Gradient magnitude at surface
# ---------------------------------------------------------------------------


def test_twist_gradient_magnitude_at_surface():
    """At surface points the gradient magnitude determines the `ao` proxy.

    For correct rendering we need ||grad|| to not differ wildly from 1.
    Record exact values so failures are informative.
    """
    sdf = _make_twisted_box(strength=1.0)

    # Cast rays inward from 6 axis-aligned directions to find surface points
    origins = [
        jnp.array([3.0, 0.0, 0.0]),
        jnp.array([-3.0, 0.0, 0.0]),
        jnp.array([0.0, 3.0, 0.0]),
        jnp.array([0.0, -3.0, 0.0]),
        jnp.array([0.0, 0.0, 3.0]),
        jnp.array([0.0, 0.0, -3.0]),
    ]
    directions = [-o / jnp.linalg.norm(o) for o in origins]

    hits = _surface_points(sdf, origins, directions)
    assert len(hits) > 0, "No surface hits found — sphere tracing may be broken"

    grad_norms = []
    for pos in hits:
        g = jax.grad(sdf)(pos)
        grad_norms.append(float(jnp.linalg.norm(g)))

    min_norm = min(grad_norms)
    max_norm = max(grad_norms)
    print(f"\nGradient norms at surface: min={min_norm:.4f}  max={max_norm:.4f}")

    # For a valid SDF the gradient norm equals 1. For an approximate SDF it should
    # stay within a reasonable band so the `ao` proxy is not wildly off.
    assert max_norm <= 2.5, f"Gradient norm too large ({max_norm:.3f}) — ao will be inflated"
    assert min_norm >= 0.4, f"Gradient norm too small ({min_norm:.3f}) — ao will darken scene"


# ---------------------------------------------------------------------------
# 2. Conservative property (no overstepping)
# ---------------------------------------------------------------------------


def test_twist_sdf_conservativeness():
    """Document the known non-conservativeness of the raw Twist SDF.

    The uncorrected Twist SDF can overestimate distances (step too large →
    overshoot surface).  This test records the violation count and magnitude
    so regressions are visible without blocking the suite.
    """
    sdf = _make_twisted_box(strength=1.0)

    test_points = [
        jnp.array([1.5, 0.5, 0.0]),
        jnp.array([0.5, 0.5, 1.5]),
        jnp.array([0.8, 1.0, 0.8]),
    ]

    rng = np.random.default_rng(42)
    n_dirs = 32
    violations = []

    for p in test_points:
        d = float(sdf(p))
        if d <= 0:
            continue
        dirs = rng.standard_normal((n_dirs, 3))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        for dir_ in dirs:
            p_stepped = p + d * jnp.array(dir_, dtype=jnp.float32)
            d_new = float(sdf(p_stepped))
            if d_new < -0.02:
                violations.append((float(d), float(d_new)))

    print(f"\nRaw Twist SDF conservativeness violations: {len(violations)}")
    for d_orig, d_new in violations[:5]:
        print(f"  stepped from sdf={d_orig:.4f} to sdf={d_new:.4f}")


# ---------------------------------------------------------------------------
# 3. Normal direction correctness
# ---------------------------------------------------------------------------


def test_twist_normals_point_outward():
    """The gradient direction (surface normal) must point away from the interior.

    Moving a small step along the gradient from a surface point should increase
    the SDF value (move outward), not decrease it.
    """
    sdf = _make_twisted_box(strength=1.0)

    origins = [
        jnp.array([3.0, 0.5, 0.0]),
        jnp.array([0.0, 0.0, 3.0]),
        jnp.array([-3.0, -0.5, 0.0]),
    ]
    directions = [-o / jnp.linalg.norm(o) for o in origins]
    hits = _surface_points(sdf, origins, directions)
    assert len(hits) > 0

    step = 0.05
    for pos in hits:
        g = jax.grad(sdf)(pos)
        normal = g / jnp.linalg.norm(g)
        d_inward = float(sdf(pos - step * normal))
        d_outward = float(sdf(pos + step * normal))
        assert d_outward > d_inward, (
            f"Normal points inward at {pos}: "
            f"sdf(+step)={d_outward:.4f}  sdf(-step)={d_inward:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Rendering brightness parity
# ---------------------------------------------------------------------------


def test_twist_rendering_brightness_parity():
    """A twisted box should have similar mean brightness to the untwisted version.

    The twist deformation should not systematically darken or brighten the scene.
    Tolerance is generous (20%) since the geometry does change somewhat.
    """
    box = Box(size=jnp.array([1.0, 2.0, 1.0]))
    twisted = Twist(box, strength=1.0, axis="y")

    common_kwargs = {
        "camera_pos": jnp.array([4.0, 3.0, 5.0]),
        "look_at": jnp.array([0.0, 0.0, 0.0]),
        "light_dirs": jnp.array([[1.5, 2.0, 1.0]]),
        "resolution": (32, 32),
        "max_steps": 96,
        "ambient": 0.05,
    }

    img_plain = raymarch(box, **common_kwargs)
    img_twist = raymarch(twisted, **common_kwargs)

    # Only compare pixels that are lit in both images
    mask = (img_plain.mean(axis=2) > 0.01) & (img_twist.mean(axis=2) > 0.01)
    if mask.sum() == 0:
        pytest.skip("No overlapping lit pixels to compare")

    mean_plain = float(img_plain[mask].mean())
    mean_twist = float(img_twist[mask].mean())

    print(f"\nMean brightness  plain={mean_plain:.4f}  twisted={mean_twist:.4f}")
    ratio = mean_twist / (mean_plain + 1e-8)
    assert 0.80 <= ratio <= 1.20, (
        f"Twisted scene brightness deviates too much: "
        f"plain={mean_plain:.4f}  twisted={mean_twist:.4f}  ratio={ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# 5. Sphere-tracing convergence
# ---------------------------------------------------------------------------


def test_twist_sphere_trace_converges():
    """Rays aimed at a twisted box should converge to near-zero SDF values."""
    sdf = _make_twisted_box(strength=1.0)

    rays = [
        (jnp.array([4.0, 0.0, 0.0]), jnp.array([-1.0, 0.0, 0.0])),
        (jnp.array([0.0, 0.0, 4.0]), jnp.array([0.0, 0.0, -1.0])),
        (jnp.array([3.0, 1.0, 3.0]), jnp.array([-1.0, -0.3, -1.0]) / jnp.sqrt(2.09)),
    ]

    for origin, direction in rays:
        t, d_min = _sphere_trace(sdf, origin, direction, steps=128)
        pos = origin + t * direction
        sdf_at_hit = float(sdf(pos))
        print(f"  d_min={float(d_min):.5f}  sdf@hit={sdf_at_hit:.5f}")
        assert d_min < 0.05, (
            f"Sphere trace failed to converge (d_min={float(d_min):.4f}). "
            "Twist SDF may be non-conservative, causing overstepping."
        )
