"""
Iterative StableHLO → GLSL compilation tests.

Each group extends coverage by one op class:
  1. jax.export / StableHLO  — verify the JAX IR is produced correctly
  2. Type mapping             — tensor<3xf32> → vec3, etc.
  3. Scalar arithmetic        — add, sub, mul, div
  4. Math ops                 — sqrt, abs, max, min
  5. Vector norm              — exercises reduce(add) + sqrt
  6. Sphere SDF               — full end-to-end for a primitive
  7. Box SDF                  — abs, broadcast, reduce_max, sqrt
  8. Union (smooth_min)       — composed scene
  9. Translate transform      — vector subtraction of a constant
 10. Syntax sanity            — balanced braces for all of the above
"""

import jax
import jax.numpy as jnp
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _compile(fn, point=None):
    from jaxcad.backends.glsl import compile_sdf_to_glsl

    return compile_sdf_to_glsl(fn, example_point=point)


def _valid(code: str) -> bool:
    return "return" in code and "float sdf(" in code


def _balanced(code: str) -> bool:
    return code.count("{") == code.count("}")


# ── 1. StableHLO export ───────────────────────────────────────────────────────


def test_stable_hlo_export_sphere():
    from jax.export import export

    from jaxcad.sdf.primitives.sphere import Sphere

    sphere = Sphere(radius=1.0)
    exported = export(jax.jit(sphere))(jnp.zeros(3))
    mlir = exported.mlir_module()

    assert "stablehlo" in mlir
    assert "func.func" in mlir
    assert "tensor<3xf32>" in mlir
    assert "tensor<f32>" in mlir


def test_stable_hlo_export_box():
    from jax.export import export

    from jaxcad.sdf.primitives.box import Box

    box = Box(size=[0.5, 0.5, 0.5])
    exported = export(jax.jit(box))(jnp.zeros(3))
    mlir = exported.mlir_module()

    assert "stablehlo.abs" in mlir
    assert "stablehlo.reduce" in mlir


def test_stable_hlo_gradients():
    """jax.grad of an SDF also exports cleanly to StableHLO."""
    from jax.export import export

    from jaxcad.sdf.primitives.sphere import Sphere

    sphere = Sphere(radius=1.0)
    grad_fn = jax.grad(sphere)
    exported = export(jax.jit(grad_fn))(jnp.ones(3))
    assert "stablehlo" in exported.mlir_module()


# ── 2. Type mapping ───────────────────────────────────────────────────────────


def test_type_mapping():
    from jaxcad.backends._type_utils import mlir_type_to_glsl

    assert mlir_type_to_glsl("tensor<f32>") == "float"
    assert mlir_type_to_glsl("tensor<2xf32>") == "vec2"
    assert mlir_type_to_glsl("tensor<3xf32>") == "vec3"
    assert mlir_type_to_glsl("tensor<4xf32>") == "vec4"
    assert mlir_type_to_glsl("tensor<i1>") == "bool"
    assert mlir_type_to_glsl("tensor<i32>") == "int"
    assert mlir_type_to_glsl("tensor<3x3xf32>") == "mat3"


# ── 3. Scalar arithmetic ──────────────────────────────────────────────────────


def test_scalar_add():
    def fn(p):
        return p[0] + p[1]

    code = _compile(fn)
    assert _valid(code) and "+" in code


def test_scalar_sub():
    def fn(p):
        return p[0] - p[1]

    code = _compile(fn)
    assert _valid(code) and "-" in code


def test_scalar_mul_div():
    def fn(p):
        return p[0] * p[1] / (p[2] + 1e-6)

    code = _compile(fn)
    assert _valid(code) and "*" in code and "/" in code


# ── 4. Math ops ───────────────────────────────────────────────────────────────


def test_sqrt():
    def fn(p):
        return jnp.sqrt(jnp.sum(p * p))

    code = _compile(fn)
    assert "sqrt" in code and _valid(code)


def test_abs_max_min():
    def fn(p):
        return jnp.minimum(jnp.abs(p[0]), jnp.maximum(p[1], p[2]))

    code = _compile(fn)
    assert "abs" in code and _valid(code)


# ── 5. Vector norm (reduce + sqrt) ───────────────────────────────────────────


def test_vector_norm():
    def fn(p):
        return jnp.linalg.norm(p)

    code = _compile(fn)
    assert _valid(code)
    # Should use dot or sqrt — either reduce(add) or dot_general path
    assert "sqrt" in code or "dot" in code


def test_dot_product():
    axis = jnp.array([0.0, 1.0, 0.0])

    def fn(p):
        return jnp.dot(p, axis)

    code = _compile(fn)
    assert "dot" in code and _valid(code)


# ── 6. Sphere SDF ─────────────────────────────────────────────────────────────


def test_sphere_glsl():
    from jaxcad.sdf.primitives.sphere import Sphere

    sphere = Sphere(radius=1.5)
    code = _compile(sphere)
    assert _valid(code)
    assert "1.500000" in code or "1.5" in code


def test_sphere_glsl_balanced():
    from jaxcad.sdf.primitives.sphere import Sphere

    code = _compile(Sphere(radius=1.0))
    assert _balanced(code)


# ── 7. Box SDF ────────────────────────────────────────────────────────────────


def test_box_glsl():
    from jaxcad.sdf.primitives.box import Box

    box = Box(size=[0.5, 1.0, 0.5])
    code = _compile(box)
    assert _valid(code)
    assert "abs" in code
    assert _balanced(code)


# ── 8. Smooth union ───────────────────────────────────────────────────────────


def test_union_glsl():
    from jaxcad.sdf.boolean.union import Union
    from jaxcad.sdf.primitives.box import Box
    from jaxcad.sdf.primitives.sphere import Sphere

    scene = Union(Sphere(1.0), Box([0.5, 0.5, 0.5]), smoothness=0.1)
    code = _compile(scene)
    assert _valid(code)
    assert _balanced(code)
    # smooth_min uses arithmetic on distances
    assert "min" in code or "-" in code


# ── 9. Translate transform ────────────────────────────────────────────────────


def test_translate_glsl():
    from jaxcad.sdf.primitives.sphere import Sphere
    from jaxcad.sdf.transforms.affine.translate import Translate

    scene = Translate(Sphere(1.0), offset=jnp.array([1.0, 0.0, 0.0]))
    code = _compile(scene)
    assert _valid(code)
    assert _balanced(code)
    assert "1.000000" in code  # offset constant is embedded


# ── 10. Syntax sanity across all primitives ───────────────────────────────────


def test_cylinder_glsl():
    from jaxcad.sdf.primitives.cylinder import Cylinder

    cyl = Cylinder(radius=0.5, height=1.0)
    code = _compile(cyl)
    assert _valid(code) and _balanced(code)


def test_capsule_glsl():
    from jaxcad.sdf.primitives.capsule import Capsule

    cap = Capsule(radius=0.3, height=1.5)
    code = _compile(cap)
    assert _valid(code) and _balanced(code)


# ── backend class API ─────────────────────────────────────────────────────────


def test_glsl_backend_api():
    from jaxcad.backends import GLSLBackend
    from jaxcad.sdf.primitives.sphere import Sphere

    backend = GLSLBackend()
    assert backend.name == "glsl"
    code = backend.compile_sdf(Sphere(1.0))
    assert _valid(code)


def test_wgsl_backend():
    from jaxcad.backends import WGSLBackend
    from jaxcad.sdf.primitives.sphere import Sphere

    backend = WGSLBackend()
    assert backend.name == "wgsl"
    code = backend.compile_sdf(Sphere(1.0))
    assert "fn sdf(" in code
    assert "-> f32" in code
    assert code.count("{") == code.count("}")


# ── renderer (requires moderngl) ──────────────────────────────────────────────


def test_glsl_renderer():
    pytest.importorskip("moderngl")
    from jaxcad.backends.glsl.renderer import GLSLRenderer

    try:
        GLSLRenderer()
    except Exception:
        pytest.skip("no usable OpenGL context on this system")
