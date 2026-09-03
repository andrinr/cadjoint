"""Compile parameterized SDF trees to pure JAX functions.

The SDF tree *is* cadjoint's intermediate representation: primitives, booleans,
transforms and patterns, each with named parameter slots.  This module lowers
it to JAX, and how it does that decides how big the compiled program is.

Traced naively, the tree **flattens**: a pattern of eight ribs emits eight
copies of the rib, a subtree used by two booleans is emitted twice, and every
parameter value is folded in as a literal — so a slider edit produces a program
that is structurally identical to the one before it and yet compiles from
scratch.  Three things keep that from happening here.

* **Sharing.** A node is built once per *object*, not once per occurrence, so a
  tool cut from several bodies is one function.
* **Outlining.** A node that is evaluated more than once — a pattern's child, a
  shared subtree — is wrapped in :func:`jax.jit`, which StableHLO keeps as a
  ``func.func`` and a ``func.call`` per instance.  The WGSL backend already
  emits one shader function per ``func.func``, so the same move shrinks the
  shader.
* **Parameters as arguments.** :func:`functionalize_parametric` hands the
  parameter dicts to the *jitted* function rather than closing over them, so
  the lowered HLO is byte-identical across value edits and the persistent
  compilation cache hits.

Everything is opt-out through :func:`cadjoint.sdf._lowering.scalar_lowering`,
which the shader backend holds while it traces: WGSL has no type wider than a
``mat4``, so the array-shaped forms inside the primitives cannot be lowered
there and the flat emission is still the right one.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

# ── helpers ───────────────────────────────────────────────────────────────────


def _static_value(candidate, param):
    """A concrete value for a structural parameter, never a tracer.

    ``count`` on a pattern decides how many instances are *emitted*, so it has
    to be a Python-readable number even when every other parameter has been
    lifted to an argument.  A caller may still override it with a concrete
    value; a traced one falls back to the value the node was built with.

    Args:
        candidate: The value found in the free/fixed dicts, or ``None``.
        param: The node's own :class:`~cadjoint.geometry.parameters.Parameter`.

    Returns:
        The candidate when it is concrete, else the parameter's stored value.
    """
    if candidate is None or isinstance(candidate, jax.core.Tracer):
        return param.value
    return candidate


def _collect(node_id: str, params_snapshot: dict, free: dict, fixed: dict, static=()) -> dict:
    """Resolve a node's params from free/fixed dicts into plain arrays.

    Args:
        node_id: The node's DFS-numbered identity, used to key fixed params.
        params_snapshot: The node's ``params`` dict.
        free: Name-keyed free parameter values.
        fixed: Path-keyed fixed parameter values.
        static: Attribute names that must stay concrete (see
            :attr:`~cadjoint.sdf.transforms.patterns.LinearPattern.static_params`).

    Returns:
        ``dict[str, Array]`` of the values the node's pure ``sdf`` expects.
    """
    result = {}
    for attr, param in params_snapshot.items():
        if param.free:
            value = free[param.name]
        else:
            path = f"{node_id}.{attr}"
            if path not in fixed and attr not in static:
                continue
            value = fixed.get(path)
        result[attr] = _static_value(value, param) if attr in static else value
    return result


def _resolve(param, free_params: dict):
    """Return the current value of a Parameter (free → from dict, fixed → stored)."""
    return free_params[param.name] if param.free else param.value


def _static_params(obj) -> tuple[str, ...]:
    """The attribute names this node needs concrete at trace time."""
    return tuple(getattr(obj.__class__, "static_params", ()))


# ── structured lowering ───────────────────────────────────────────────────────


def _instance_count(obj) -> int:
    """How many times this node evaluates its child, when that is knowable.

    Args:
        obj: A tree node.

    Returns:
        The child evaluation count for a pattern with a concrete ``count``,
        else ``1``.
    """
    count = obj.params.get("count") if getattr(obj, "params", None) else None
    if count is None:
        return 1
    try:
        return int(getattr(count, "value", count))
    except (TypeError, ValueError):
        return 1


def _repeated_nodes(root) -> set[int]:
    """Ids of nodes whose evaluation appears more than once in the trace.

    Two things make a node repeat: being the child of a pattern that emits
    several instances of it, and being reachable from more than one parent —
    the same tool subtracted from two bodies, or the dowel that a ``Mirror``
    also reflects.  Those, and only those, are worth outlining into their own
    function; a node used once would occupy exactly as much program either way.

    Args:
        root: Root of the tree to scan.

    Returns:
        The set of ``id(node)`` values to outline.
    """
    in_degree: dict[int, int] = {}
    repeated: set[int] = set()
    seen: set[int] = set()

    def walk(obj) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        multiplier = _instance_count(obj)
        for child in obj.children():
            key = id(child)
            in_degree[key] = in_degree.get(key, 0) + 1
            if multiplier > 1 or in_degree[key] > 1:
                repeated.add(key)
            walk(child)

    walk(root)
    return repeated


def _outlined(fn: Callable, enabled: bool) -> Callable:
    """Wrap a node evaluation so a trace emits it as one shared function.

    ``jax.jit`` inside a trace lowers to a StableHLO ``func.func`` plus one
    ``func.call`` per use, and JAX prunes the parameter dict entries the callee
    does not read, so the shared function's signature stays small.  Outside a
    trace the inner function is called directly: eager evaluation gains nothing
    from a call boundary and would pay a dispatch for it.

    Args:
        fn: ``(p, free, fixed) -> value``.
        enabled: Whether this node repeats.

    Returns:
        A callable with the same signature.
    """
    if not enabled:
        return fn
    jitted = jax.jit(fn)

    def call(p, free: dict, fixed: dict):
        if isinstance(p, jax.core.Tracer):
            return jitted(p, free, fixed)
        return fn(p, free, fixed)

    return call


# ── functionalize ─────────────────────────────────────────────────────────────


def functionalize(sdf) -> Callable:
    """Compile an SDF to a pure function with free and fixed parameters.

    Returns a curried function with signature:
        sdf_fn(free_params, fixed_params) -> (point -> distance)

    Nodes are built once per object and repeated nodes are outlined, so a
    pattern of ``N`` instances and a subtree used by two parents each cost one
    copy in the emitted program rather than ``N`` and two.

    Args:
        sdf: The SDF to compile

    Returns:
        Callable: Curried function ``sdf_fn(free_params, fixed_params) -> (point -> distance)``
            mapping parameter dicts to a callable ``point: Array (3,) -> distance: Array ()``.

    Example:
        ```python
        radius = Scalar(value=1.0, free=True, name='radius')
        sphere = Sphere(radius=radius)
        sdf_fn = functionalize(sphere)
        distance = sdf_fn({'sphere_0.radius': 2.0}, {})(jnp.array([0., 0., 0.]))
        ```
    """
    from cadjoint.sdf.boolean.base import BooleanOp

    node_counter = {"count": 0}
    repeated = _repeated_nodes(sdf)
    memo: dict[int, Callable | None] = {}

    def skip(obj) -> None:
        """Advance the node counter over an already-built subtree.

        ``extract_parameters`` numbers every *occurrence*, so a shared node is
        walked twice there and the counter has to agree.
        """
        node_counter["count"] += 1
        for child in obj.children():
            skip(child)

    def build_function(obj) -> Callable | None:
        if id(obj) in memo:
            skip(obj)
            return memo[id(obj)]

        node_id = f"{obj.__class__.__name__.lower()}_{node_counter['count']}"
        node_counter["count"] += 1

        if not hasattr(obj.__class__, "sdf"):
            for c in obj.children():
                build_function(c)
            memo[id(obj)] = None
            return None

        pure_sdf = obj.__class__.sdf
        params_snapshot = obj.params
        static = _static_params(obj)
        raw_child_fns = [build_function(c) for c in obj.children()]
        child_fns = [f for f in raw_child_fns if f is not None]

        def eval_fn(
            p,
            free_params: dict,
            fixed_params: dict,
            _nid=node_id,
            _ps=params_snapshot,
            _fn=pure_sdf,
            _ch=child_fns,
            _static=static,
        ):
            param_values = _collect(_nid, _ps, free_params, fixed_params, _static)
            child_evals = [lambda p_, fn=fn: fn(p_, free_params, fixed_params) for fn in _ch]
            if isinstance(obj, BooleanOp):
                return _fn(tuple(child_evals), p, **param_values)
            elif child_evals:
                return _fn(child_evals[0], p, **param_values)
            else:
                return _fn(p, **param_values)

        shared = _outlined(eval_fn, id(obj) in repeated)
        memo[id(obj)] = shared
        return shared

    inner = build_function(sdf)
    return lambda free_params, fixed_params: lambda p: inner(p, free_params, fixed_params)


def functionalize_parametric(sdf) -> Callable:
    """Compile an SDF to a function whose *parameters* are arguments.

    :func:`functionalize` closes over the parameter values, so ``jax.jit`` of
    the result folds every one of them into the program as a literal: two
    designs that differ only in a slider value lower to two different HLO
    modules, compile separately, and miss each other in the persistent
    compilation cache.  This form takes them as arrays instead, so the lowered
    text is byte-identical for every value of every parameter and one compiled
    executable serves them all.

    Args:
        sdf: The SDF to compile.

    Returns:
        ``fn(free_params, fixed_params, point) -> distance``, safe to wrap in
            :func:`jax.jit` directly.

    Example:
        ```python
        free, fixed, _ = extract_parameters(scene)
        evaluate = jax.jit(functionalize_parametric(scene))
        distance = evaluate(free, fixed, jnp.zeros(3))
        ```
    """
    compiled = functionalize(sdf)

    def evaluate(free_params: dict, fixed_params: dict, point):
        return compiled(free_params, fixed_params)(point)

    return evaluate


# ── functionalize_scene ───────────────────────────────────────────────────────


def functionalize_scene(geometry) -> Callable:
    """Compile a geometry SDF tree to pure (sdf, material_fn) closures.

    Returns a curried function::

        scene_fn = functionalize_scene(geometry)
        sdf, material_fn = scene_fn(free_params, fixed_params)
        distance = sdf(point)
        mat_dict = material_fn(point)

    Both ``sdf`` and ``material_fn`` are pure JAX functions. This helper is for
    geometry and material evaluation; the forward image renderer deliberately
    has a separate, non-differentiable API.

    The node counter matches ``extract_parameters`` exactly (same DFS order,
    same counter increments for every Fluent node), so path-keyed fixed params
    align correctly — including for a node reached twice, which is numbered
    twice there and built only once here.

    Args:
        geometry: Root SDF node of the geometry tree.

    Returns:
        ``(free_params, fixed_params) -> (sdf_fn, material_fn)``
    """
    from cadjoint.render.material import Material
    from cadjoint.sdf.boolean.base import BooleanOp
    from cadjoint.sdf.boolean.smooth import smooth_min
    from cadjoint.sdf.primitives.base import Primitive

    node_counter = {"count": 0}
    repeated = _repeated_nodes(geometry)
    memo: dict[int, tuple] = {}

    def skip(obj) -> None:
        """Advance the node counter over an already-built subtree."""
        node_counter["count"] += 1
        for child in obj.children():
            skip(child)

    def build(obj):
        """DFS builder — returns (sdf_eval | None, mat_eval).

        For SDF nodes:   sdf_eval(p, free, fixed) -> distance
                         mat_eval(p, free, fixed) -> material dict
        For non-SDF nodes (Material): sdf_eval is None,
                         mat_eval(free, fixed) -> raw param dict  ← no p
        """
        if id(obj) in memo:
            skip(obj)
            return memo[id(obj)]

        node_id = f"{obj.__class__.__name__.lower()}_{node_counter['count']}"
        node_counter["count"] += 1
        ps = obj.params  # params snapshot
        static = _static_params(obj)

        # ── non-SDF Fluent (Material) ─────────────────────────────────────────
        if not hasattr(obj.__class__, "sdf"):
            for c in obj.children():
                build(c)

            def mat_params(free, fixed, _nid=node_id, _ps=ps):
                return _collect(_nid, _ps, free, fixed)

            memo[id(obj)] = (None, mat_params)
            return memo[id(obj)]

        # ── SDF node ─────────────────────────────────────────────────────────
        pure_sdf = obj.__class__.sdf
        child_res = [build(c) for c in obj.children()]

        # SDF children: have sdf_eval (s is not None)
        sdf_ch = [(s, m) for s, m in child_res if s is not None]
        # Non-SDF children: material param collectors (s is None)
        mat_pfs = [m for s, m in child_res if s is None and m is not None]

        outline = id(obj) in repeated

        # ── Primitive ─────────────────────────────────────────────────────────
        if isinstance(obj, Primitive):
            mat_pf = mat_pfs[0] if mat_pfs else None

            def sdf_eval(p, free, fixed, _nid=node_id, _ps=ps, _fn=pure_sdf, _static=static):
                return _fn(p, **_collect(_nid, _ps, free, fixed, _static))

            def mat_eval(_p, free, fixed, _mpf=mat_pf):
                mp = _mpf(free, fixed) if _mpf is not None else {}
                return {
                    "color": mp.get("color", jnp.ones(3)),
                    "roughness": mp.get("roughness", jnp.array(0.5)),
                    "metallic": mp.get("metallic", jnp.array(0.0)),
                    "opacity": mp.get("opacity", jnp.array(1.0)),
                    "ior": mp.get("ior", jnp.array(1.0)),
                    "reflectivity": mp.get("reflectivity", jnp.array(0.0)),
                }

            memo[id(obj)] = (_outlined(sdf_eval, outline), mat_eval)
            return memo[id(obj)]

        # ── BooleanOp (Union, Intersection, …) ───────────────────────────────
        if isinstance(obj, BooleanOp):

            def sdf_eval(
                p, free, fixed, _nid=node_id, _ps=ps, _fn=pure_sdf, _ch=sdf_ch, _static=static
            ):
                pv = _collect(_nid, _ps, free, fixed, _static)
                evals = tuple(lambda p_, s=s: s(p_, free, fixed) for s, _ in _ch)
                return _fn(evals, p, **pv)

            def mat_eval(p, free, fixed, _nid=node_id, _ps=ps, _ch=sdf_ch, _static=static):
                pv = _collect(_nid, _ps, free, fixed, _static)
                smoothness = pv.get("smoothness", jnp.array(0.1))
                k = jnp.maximum(smoothness * 4.0, 1e-10)

                d0 = _ch[0][0](p, free, fixed)
                mat0 = _ch[0][1](p, free, fixed)
                for s_fn, m_fn in _ch[1:]:
                    d = s_fn(p, free, fixed)
                    m = m_fn(p, free, fixed)
                    t = jnp.clip(0.5 + 0.5 * (d - d0) / k, 0.0, 1.0)
                    mat0 = Material.blend(mat0, m, t)
                    d0 = smooth_min(d0, d, smoothness)
                return mat0

            memo[id(obj)] = (_outlined(sdf_eval, outline), mat_eval)
            return memo[id(obj)]

        # ── Transform (Translate, Rotate, Scale, …) ──────────────────────────
        assert len(sdf_ch) == 1, f"Transform {obj.__class__.__name__} must have 1 SDF child"
        child_sdf_fn, child_mat_fn = sdf_ch[0]

        def sdf_eval(
            p, free, fixed, _nid=node_id, _ps=ps, _fn=pure_sdf, _csdf=child_sdf_fn, _static=static
        ):
            pv = _collect(_nid, _ps, free, fixed, _static)
            return _fn(lambda p_: _csdf(p_, free, fixed), p, **pv)

        def mat_eval(
            p,
            free,
            fixed,
            _nid=node_id,
            _ps=ps,
            _cmat=child_mat_fn,
            _cls=obj.__class__,
            _static=static,
        ):
            pv = _collect(_nid, _ps, free, fixed, _static)
            tp = _cls._transform_point(p, **pv)
            return _cmat(tp, free, fixed)

        memo[id(obj)] = (_outlined(sdf_eval, outline), mat_eval)
        return memo[id(obj)]

    inner_sdf, inner_mat = build(geometry)

    def scene_fn(free_params: dict, fixed_params: dict):
        return (
            lambda p: inner_sdf(p, free_params, fixed_params),
            lambda p: inner_mat(p, free_params, fixed_params),
        )

    return scene_fn


def functionalize_scene_parametric(geometry) -> Callable:
    """Compile a scene so its *parameters* are arguments, not literals.

    The :func:`functionalize_parametric` of :func:`functionalize_scene`: one
    lowered program serves every value of every parameter, which is what makes
    a shader's uniform block possible and what makes the on-disk compilation
    cache hit across edits and processes.

    Args:
        geometry: Root SDF node of the geometry tree.

    Returns:
        ``fn(free_params, fixed_params, point) -> (distance, material dict)``.
    """
    compiled = functionalize_scene(geometry)

    def evaluate(free_params: dict, fixed_params: dict, point):
        sdf, material = compiled(free_params, fixed_params)
        return sdf(point), material(point)

    return evaluate
