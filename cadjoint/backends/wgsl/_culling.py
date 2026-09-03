"""The scene's distance field with every far-away node skipped.

:func:`cadjoint.functionalize.functionalize_scene` lowers the tree faithfully:
a union evaluates every operand at every query point, and a sphere trace then
pays for all forty leaves of a part at every step of every ray.  Almost all of
that is wasted — at any point most leaves are far away and lose the ``min``
by a wide margin — but a ``min`` over a value is not a skipped value.  This
module traces the same tree with one addition: before a boolean evaluates an
operand, it compares the distance to the operand's bounding box (see
:mod:`cadjoint.sdf._lowering`) against what it already holds, and when the
operand cannot change the result it is not evaluated at all.  ``lax.cond``
lowers to a ``stablehlo.case``, which the WGSL emitter turns into a real
``if``, so the skipped branch costs nothing on the GPU.

The result is not an approximation.  Each skip is taken only when the exact
value is provably what the running result already is:

* **Smooth union.**  ``smin(m, d) = min(m, d) - h²/(4K)`` with
  ``h = max(K - |m - d|, 0)`` and ``K`` the scaled band.  When ``d >= m + K``
  the band term is exactly zero and ``smin(m, d) = m``.  The box gives
  ``d >= box(p)``, so ``box(p) >= m + K`` suffices.
* **Smooth difference.**  ``smax(m, -d) = -smin(-m, d)``, which is ``m`` when
  ``d >= K - m``; again ``box(p) >= K - m`` suffices.  Outside the body
  ``m > 0`` and the condition is nearly always met, so every hole of a part
  is skipped for every exterior ray step that is not inside the hole's box.
* **Pattern instance.**  A pattern is a plain ``min``; ``min(m, d) = m`` when
  ``d >= m``.

Each test also requires ``box(p) > 0`` — the point is outside the box — which
is where the lower bound holds, and adds :data:`CULL_MARGIN` so that float
rounding in the box distance can never flip a test the exact arithmetic
would not.  ``tests/backends/test_wgsl_culling.py`` checks the two forms
agree to the last bit on random points in every shipped scene.

Intersections and XORs are not culled: a lower bound on an operand cannot
show that the *maximum* is unchanged.  Their operands are still culled
inside, where they are unions.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp

from cadjoint.functionalize import _collect, _outlined, _repeated_nodes, _static_params
from cadjoint.sdf._lowering import (
    Bounds,
    box_distance,
    is_scalar_lowering,
    node_bounds,
    rotation_rows,
)
from cadjoint.sdf._lowering import (
    _apply_rows as apply_rows,
)

#: World-unit slack on every skip test.  A bound violated by rounding is on
#: the order of 1e-7 of the scene's extent; this is three orders above it.
CULL_MARGIN = 1e-4


def _skip(condition, held, evaluate: Callable):
    """``held`` if ``condition``, else ``evaluate(held)`` — as a real branch."""
    return jax.lax.cond(condition, lambda m: m, evaluate, held)


def culled_scene_sdf(geometry, *, margin: float = CULL_MARGIN) -> Callable:
    """Compile a scene's distance field with bounding-box culling.

    Mirrors :func:`cadjoint.functionalize.functionalize_scene` node for node
    — the same DFS numbering, so path-keyed fixed parameters line up; the
    same sharing and outlining — and returns only the distance half of it.
    Every node type is evaluated through its own static ``sdf`` exactly as
    there; unions, differences and the two patterns are the exception,
    re-spelled with a ``lax.cond`` around each operand.

    Args:
        geometry: Root SDF node.
        margin: Slack added to every skip test, in world units. A *traced*
            value here is what lets the viewer switch culling off without a
            recompile: every test is ``box_distance(p, bounds) >= threshold +
            margin``, so an infinite margin makes all of them false and the
            module computes the flat field (see
            :data:`cadjoint.backends.wgsl.CULL_DISABLED_MARGIN`).

    Returns:
        ``(free_params, fixed_params, margin=None) -> (point -> distance)``.
            The optional third argument replaces the build-time margin, which
            is how a traced one gets in: the skip tests read it late, at trace
            time, so binding it here reaches every one of them.
    """
    # One cell, read by every skip test at trace time, so `bound` can replace
    # the margin after the tree is built.
    nonlocal_margin = [margin]

    from cadjoint.render.material import Material
    from cadjoint.sdf.boolean.base import BooleanOp
    from cadjoint.sdf.boolean.difference import Difference
    from cadjoint.sdf.boolean.smooth import smooth_max, smooth_min
    from cadjoint.sdf.boolean.union import Union
    from cadjoint.sdf.primitives.base import Primitive
    from cadjoint.sdf.transforms.patterns import (
        LinearPattern,
        PolarPattern,
        _kept_instances,
        _rotate_about,
    )

    node_counter = {"count": 0}
    repeated = _repeated_nodes(geometry)
    memo: dict[int, tuple] = {}

    def skip(obj) -> None:
        node_counter["count"] += 1
        for child in obj.children():
            skip(child)

    def build(obj):
        """DFS builder — returns ``(sdf_eval | None, bounds_fn | None)``.

        ``sdf_eval(p, free, fixed) -> distance``;
        ``bounds_fn(free, fixed) -> Bounds | None`` in the node's own frame.
        """
        if id(obj) in memo:
            skip(obj)
            return memo[id(obj)]

        node_id = f"{obj.__class__.__name__.lower()}_{node_counter['count']}"
        node_counter["count"] += 1
        ps = obj.params
        static = _static_params(obj)

        if not hasattr(obj.__class__, "sdf") or isinstance(obj, Material):
            for child in obj.children():
                build(child)
            memo[id(obj)] = (None, None)
            return memo[id(obj)]

        pure_sdf = obj.__class__.sdf
        child_results = [build(child) for child in obj.children()]
        sdf_children = [(s, b) for s, b in child_results if s is not None]
        outline = id(obj) in repeated

        def collect(free, fixed, _nid=node_id, _ps=ps, _static=static):
            return _collect(_nid, _ps, free, fixed, _static)

        def bounds_fn(free, fixed, _obj=obj, _ch=sdf_children):
            values = collect(free, fixed)
            return node_bounds(_obj, values, [b(free, fixed) if b else None for _, b in _ch])

        # ── primitive ─────────────────────────────────────────────────────
        if isinstance(obj, Primitive):

            def sdf_eval(p, free, fixed, _fn=pure_sdf):
                return _fn(p, **collect(free, fixed))

            memo[id(obj)] = (_outlined(sdf_eval, outline), bounds_fn)
            return memo[id(obj)]

        # ── union / difference: one branch per operand ────────────────────
        if isinstance(obj, (Union, Difference)):
            is_union = isinstance(obj, Union)

            def sdf_eval(p, free, fixed, _ch=sdf_children, _union=is_union):
                values = collect(free, fixed)
                smoothness = values["smoothness"]
                band = jnp.maximum(smoothness * 4.0, 1e-10)
                result = _ch[0][0](p, free, fixed)
                for child_sdf, child_bounds in _ch[1:]:
                    bounds = child_bounds(free, fixed) if child_bounds else None

                    if _union:

                        def combine(m, _s=child_sdf):
                            return smooth_min(m, _s(p, free, fixed), smoothness)

                        threshold = jnp.maximum(result + band, 0.0) + nonlocal_margin[0]
                    else:

                        def combine(m, _s=child_sdf):
                            d = _s(p, free, fixed)
                            return jnp.where(
                                smoothness > 0,
                                smooth_max(m, -d, smoothness),
                                jnp.maximum(m, -d),
                            )

                        threshold = jnp.maximum(band - result, 0.0) + nonlocal_margin[0]

                    if bounds is None:
                        result = combine(result)
                    else:
                        result = _skip(box_distance(p, bounds) >= threshold, result, combine)
                return result

            memo[id(obj)] = (_outlined(sdf_eval, outline), bounds_fn)
            return memo[id(obj)]

        # ── other booleans: verbatim ──────────────────────────────────────
        if isinstance(obj, BooleanOp):

            def sdf_eval(p, free, fixed, _fn=pure_sdf, _ch=sdf_children):
                evals = tuple(lambda p_, s=s: s(p_, free, fixed) for s, _ in _ch)
                return _fn(evals, p, **collect(free, fixed))

            memo[id(obj)] = (_outlined(sdf_eval, outline), bounds_fn)
            return memo[id(obj)]

        assert len(sdf_children) == 1, f"Transform {obj.__class__.__name__} must have 1 SDF child"
        child_sdf, child_bounds = sdf_children[0]

        # ── patterns: one branch per instance (the shader's unrolled form) ─
        if isinstance(obj, (PolarPattern, LinearPattern)):
            is_polar = isinstance(obj, PolarPattern)

            def sdf_eval(p, free, fixed, _fn=pure_sdf, _polar=is_polar):
                values = collect(free, fixed)
                child = lambda q: child_sdf(q, free, fixed)  # noqa: E731
                count = int(values["count"])
                bounds = child_bounds(free, fixed) if child_bounds else None
                # A pattern may leave instances out — the row of holes a part
                # interrupts where something else passes through. The flat
                # field emits `kept`, so this has to emit exactly `kept` too;
                # unrolling `range(1, count)` instead puts back the geometry
                # the scene asked to omit.
                kept = _kept_instances(count, values["skip_mask"])
                if not is_scalar_lowering() or len(kept) == 1 or bounds is None:
                    return _fn(child, p, **values)
                result = child(p)
                if _polar:
                    origin, direction = values["origin"], values["direction"]
                    for i in kept[1:]:
                        theta = 2.0 * math.pi * i / count
                        rows = rotation_rows(direction, theta)
                        instance = Bounds(
                            center=origin + apply_rows(rows, bounds.center - origin),
                            half=apply_rows([jnp.abs(row) for row in rows], bounds.half),
                            scale=bounds.scale,
                        )

                        def combine(m, _theta=theta):
                            return jnp.minimum(
                                m, child(_rotate_about(p, origin, direction, -_theta))
                            )

                        threshold = jnp.maximum(result, 0.0) + nonlocal_margin[0]
                        result = _skip(box_distance(p, instance) >= threshold, result, combine)
                    return result
                direction, spacing = values["direction"], values["spacing"]
                axis = direction / jnp.linalg.norm(direction)
                for i in kept[1:]:
                    shift = axis * (spacing * i)
                    instance = Bounds(
                        center=bounds.center + shift, half=bounds.half, scale=bounds.scale
                    )

                    def combine(m, _shift=shift):
                        return jnp.minimum(m, child(p - _shift))

                    threshold = jnp.maximum(result, 0.0) + nonlocal_margin[0]
                    result = _skip(box_distance(p, instance) >= threshold, result, combine)
                return result

            memo[id(obj)] = (_outlined(sdf_eval, outline), bounds_fn)
            return memo[id(obj)]

        # ── transforms and field operations: verbatim ─────────────────────
        def sdf_eval(p, free, fixed, _fn=pure_sdf):
            return _fn(lambda q: child_sdf(q, free, fixed), p, **collect(free, fixed))

        memo[id(obj)] = (_outlined(sdf_eval, outline), bounds_fn)
        return memo[id(obj)]

    inner_sdf, _ = build(geometry)
    if inner_sdf is None:
        raise ValueError("The scene root is not an SDF node")

    def bound(free, fixed, margin=None):
        # Every `threshold = ... + margin` above is evaluated when the tree is
        # traced, not when it is built, and Python closures read their
        # enclosing scope late — so rebinding the name here reaches all of
        # them without threading a fourth argument through the outlining.
        if margin is not None:
            nonlocal_margin[0] = margin
        return lambda p: inner_sdf(p, free, fixed)

    return bound


def scene_bounds(geometry, free: dict, fixed: dict) -> Bounds | None:
    """The whole scene's bounds in world space, for reporting and tests."""
    from cadjoint.render.material import Material

    node_counter = {"count": 0}
    memo: dict[int, Bounds | None] = {}

    def skip(obj) -> None:
        node_counter["count"] += 1
        for child in obj.children():
            skip(child)

    def walk(obj):
        if id(obj) in memo:
            skip(obj)
            return memo[id(obj)]
        node_id = f"{obj.__class__.__name__.lower()}_{node_counter['count']}"
        node_counter["count"] += 1
        if not hasattr(obj.__class__, "sdf") or isinstance(obj, Material):
            for child in obj.children():
                walk(child)
            memo[id(obj)] = None
            return None
        children = []
        for child in obj.children():
            bounds = walk(child)
            # Material children advance the counter too, exactly as build() does.
            if hasattr(child.__class__, "sdf"):
                children.append(bounds)
        values = _collect(node_id, obj.params, free, fixed, _static_params(obj))
        memo[id(obj)] = node_bounds(obj, values, children)
        return memo[id(obj)]

    return walk(geometry)
