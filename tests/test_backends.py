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
import numpy as np
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _compile(fn, point=None):
    from jaxcad.backends.glsl import compile_sdf_to_glsl

    return compile_sdf_to_glsl(fn, example_point=point)


def _valid(code: str) -> bool:
    return "return" in code and "float sdf(" in code


def _balanced(code: str) -> bool:
    return code.count("{") == code.count("}")


def _compile_target(target, fn, point=None):
    if target == "glsl":
        from jaxcad.backends.glsl import compile_sdf_to_glsl

        return compile_sdf_to_glsl(fn, example_point=point)
    from jaxcad.backends.wgsl import compile_sdf_to_wgsl

    return compile_sdf_to_wgsl(fn, example_point=point)


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
    from jaxcad.backends.wgsl._wgsl_emitter import mlir_type_to_wgsl

    assert mlir_type_to_glsl("tensor<f32>") == "float"
    assert mlir_type_to_glsl("tensor<2xf32>") == "vec2"
    assert mlir_type_to_glsl("tensor<3xf32>") == "vec3"
    assert mlir_type_to_glsl("tensor<4xf32>") == "vec4"
    assert mlir_type_to_glsl("tensor<i1>") == "bool"
    assert mlir_type_to_glsl("tensor<i32>") == "int"
    assert mlir_type_to_glsl("tensor<3x3xf32>") == "mat3"
    assert mlir_type_to_glsl("tensor<2x3xf32>") == "mat3x2"
    assert mlir_type_to_glsl("tensor<1x3xf32>") == "vec3"
    assert mlir_type_to_wgsl("tensor<2x3xf32>") == "mat3x2<f32>"
    assert mlir_type_to_wgsl("tensor<3x1xf32>") == "vec3<f32>"


def test_type_mapping_rejects_unsupported_types():
    from jaxcad.backends._type_utils import mlir_type_to_glsl

    with pytest.raises(ValueError, match="ranked MLIR tensor"):
        mlir_type_to_glsl("f32")
    with pytest.raises(ValueError, match="f16"):
        mlir_type_to_glsl("tensor<f16>")
    with pytest.raises(ValueError, match="static tensor shapes"):
        mlir_type_to_glsl("tensor<?xf32>")


def test_matrix_literals_use_shader_column_order():
    from jaxcad.backends._type_utils import glsl_literal
    from jaxcad.backends.wgsl._wgsl_emitter import wgsl_literal

    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    values = "1.000000, 4.000000, 2.000000, 5.000000, 3.000000, 6.000000"
    assert glsl_literal(value, (2, 3), np.float32) == f"mat3x2({values})"
    assert wgsl_literal(value, (2, 3), np.float32) == f"mat3x2<f32>({values})"


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


def test_float_remainder_preserves_truncating_semantics():
    def fn(point):
        return jnp.remainder(point[0], 2.0)

    glsl = _compile_target("glsl", fn)
    wgsl = _compile_target("wgsl", fn)

    assert "trunc(" in glsl and "mod(" not in glsl
    assert "%" in wgsl


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


# ── complete built-in catalog ────────────────────────────────────────────────


def _builtin_scenes():
    from jaxcad.sdf.boolean import Difference, Intersection, Union, Xor
    from jaxcad.sdf.primitives import (
        Box,
        Capsule,
        Cylinder,
        Plane,
        RoundBox,
        Sphere,
        Torus,
    )
    from jaxcad.sdf.transforms.affine import Rotate, Scale, Translate
    from jaxcad.sdf.transforms.deformations import Twist

    sphere = Sphere(1.0)
    box = Box(jnp.array([0.7, 0.8, 0.9]))
    return {
        "sphere": sphere,
        "box": box,
        "capsule": Capsule(0.5, 1.0),
        "cylinder": Cylinder(0.5, 1.0),
        "plane": Plane(-1.0),
        "round_box": RoundBox(jnp.ones(3), 0.2),
        "torus": Torus(2.0, 0.5),
        "union": Union(sphere, box),
        "intersection": Intersection(sphere, box),
        "difference": Difference(sphere, box),
        "xor": Xor(sphere, box),
        "translate": Translate(sphere, jnp.array([1.0, 2.0, 3.0])),
        "rotate": Rotate(box, "z", 0.4),
        "scale": Scale(sphere, 2.0),
        "twist": Twist(box, 1.0, "y"),
    }


@pytest.mark.parametrize("target", ["glsl", "wgsl"])
@pytest.mark.parametrize("scene_name", list(_builtin_scenes()))
def test_builtin_scene_compiles_for_shader_targets(target, scene_name):
    code = _compile_target(target, _builtin_scenes()[scene_name])

    assert "?UNKNOWN?" not in code
    assert _balanced(code)
    assert "float sdf(" in code if target == "glsl" else "fn sdf(" in code


@pytest.mark.parametrize("target", ["glsl", "wgsl"])
def test_compiler_enforces_sdf_signature(target):
    with pytest.raises(ValueError, match="shape \\(3,\\)"):
        _compile_target(target, jnp.sum, point=jnp.zeros(2, dtype=jnp.float32))

    with pytest.raises(ValueError, match="scalar float32 distance"):
        _compile_target(target, lambda point: point)


def test_rotate_and_scale_emit_required_operations():
    scenes = _builtin_scenes()
    rotate_glsl = _compile_target("glsl", scenes["rotate"])
    rotate_wgsl = _compile_target("wgsl", scenes["rotate"])
    scale_glsl = _compile_target("glsl", scenes["scale"])
    scale_wgsl = _compile_target("wgsl", scenes["scale"])

    assert "transpose(" in rotate_glsl and "mat3" in rotate_glsl
    assert "transpose(" in rotate_wgsl and "mat3x3<f32>" in rotate_wgsl
    assert "all(" in scale_glsl and "isinf(" in scale_glsl
    assert "all(" in scale_wgsl and "3.402823e+38" in scale_wgsl


def test_fragment_shader_options_are_validated():
    from jaxcad.backends.glsl import build_fragment_shader
    from jaxcad.sdf.primitives import Sphere

    with pytest.raises(ValueError, match="max_steps"):
        build_fragment_shader(Sphere(1.0), max_steps=0)
    with pytest.raises(ValueError, match="max_dist"):
        build_fragment_shader(Sphere(1.0), max_dist=float("inf"))
    with pytest.raises(ValueError, match="surf_eps"):
        build_fragment_shader(Sphere(1.0), surf_eps=0.0)


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
