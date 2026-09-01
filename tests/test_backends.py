"""
Iterative StableHLO → WGSL compilation tests.

Each group extends coverage by one op class:
  1. jax.export / StableHLO  — verify the JAX IR is produced correctly
  2. Type mapping             — tensor<3xf32> → vec3<f32>, etc.
  3. Scalar arithmetic        — add, sub, mul, div
  4. Math ops                 — sqrt, abs, max, min, remainder
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
    from cadjoint.backends import compile_sdf_to_wgsl

    return compile_sdf_to_wgsl(fn, example_point=point)


def _valid(code: str) -> bool:
    return "return" in code and "fn sdf(" in code


def _balanced(code: str) -> bool:
    return code.count("{") == code.count("}")


# ── 1. StableHLO export ───────────────────────────────────────────────────────


def test_stable_hlo_export_sphere():
    from jax.export import export

    from cadjoint.sdf.primitives.sphere import Sphere

    sphere = Sphere(radius=1.0)
    exported = export(jax.jit(sphere))(jnp.zeros(3))
    mlir = exported.mlir_module()

    assert "stablehlo" in mlir
    assert "func.func" in mlir
    assert "tensor<3xf32>" in mlir
    assert "tensor<f32>" in mlir


def test_stable_hlo_export_box():
    from jax.export import export

    from cadjoint.sdf.primitives.box import Box

    box = Box(size=[0.5, 0.5, 0.5])
    exported = export(jax.jit(box))(jnp.zeros(3))
    mlir = exported.mlir_module()

    assert "stablehlo.abs" in mlir
    assert "stablehlo.reduce" in mlir


def test_stable_hlo_gradients():
    """jax.grad of an SDF also exports cleanly to StableHLO."""
    from jax.export import export

    from cadjoint.sdf.primitives.sphere import Sphere

    sphere = Sphere(radius=1.0)
    grad_fn = jax.grad(sphere)
    exported = export(jax.jit(grad_fn))(jnp.ones(3))
    assert "stablehlo" in exported.mlir_module()


# ── 2. Type mapping ───────────────────────────────────────────────────────────


def test_type_mapping():
    from cadjoint.backends.wgsl._wgsl_emitter import mlir_type_to_wgsl

    assert mlir_type_to_wgsl("tensor<f32>") == "f32"
    assert mlir_type_to_wgsl("tensor<2xf32>") == "vec2<f32>"
    assert mlir_type_to_wgsl("tensor<3xf32>") == "vec3<f32>"
    assert mlir_type_to_wgsl("tensor<4xf32>") == "vec4<f32>"
    assert mlir_type_to_wgsl("tensor<i1>") == "bool"
    assert mlir_type_to_wgsl("tensor<i32>") == "i32"
    assert mlir_type_to_wgsl("tensor<3x3xf32>") == "mat3x3<f32>"
    assert mlir_type_to_wgsl("tensor<2x3xf32>") == "mat3x2<f32>"
    assert mlir_type_to_wgsl("tensor<1x3xf32>") == "vec3<f32>"
    assert mlir_type_to_wgsl("tensor<3x1xf32>") == "vec3<f32>"


def test_type_mapping_rejects_unsupported_types():
    from cadjoint.backends.wgsl._wgsl_emitter import mlir_type_to_wgsl

    with pytest.raises(ValueError, match="ranked MLIR tensor"):
        mlir_type_to_wgsl("f32")
    with pytest.raises(ValueError, match="f16"):
        mlir_type_to_wgsl("tensor<f16>")
    with pytest.raises(ValueError, match="static tensor shapes"):
        mlir_type_to_wgsl("tensor<?xf32>")


def test_matrix_literals_use_shader_column_order():
    from cadjoint.backends.wgsl._wgsl_emitter import wgsl_literal

    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    values = "1.000000, 4.000000, 2.000000, 5.000000, 3.000000, 6.000000"
    assert wgsl_literal(value, (2, 3), np.float32) == f"mat3x2<f32>({values})"


def test_wgsl_literals_preserve_small_nonzero_values():
    from cadjoint.backends.wgsl._wgsl_emitter import wgsl_literal

    assert wgsl_literal(np.float32(1e-10), (), np.float32) == "1e-10"
    assert wgsl_literal(np.float32(5e-7), (), np.float32) == "5e-07"


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


def test_float_remainder_uses_wgsl_operator():
    def fn(point):
        return jnp.remainder(point[0], 2.0)

    code = _compile(fn)
    assert "%" in code


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


def test_sphere_wgsl():
    from cadjoint.sdf.primitives.sphere import Sphere

    sphere = Sphere(radius=1.5)
    code = _compile(sphere)
    assert _valid(code)
    assert "1.500000" in code or "1.5" in code


def test_sphere_wgsl_balanced():
    from cadjoint.sdf.primitives.sphere import Sphere

    code = _compile(Sphere(radius=1.0))
    assert _balanced(code)


# ── 7. Box SDF ────────────────────────────────────────────────────────────────


def test_box_wgsl():
    from cadjoint.sdf.primitives.box import Box

    box = Box(size=[0.5, 1.0, 0.5])
    code = _compile(box)
    assert _valid(code)
    assert "abs" in code
    assert _balanced(code)


# ── 8. Smooth union ───────────────────────────────────────────────────────────


def test_union_wgsl():
    from cadjoint.sdf.boolean.union import Union
    from cadjoint.sdf.primitives.box import Box
    from cadjoint.sdf.primitives.sphere import Sphere

    scene = Union(Sphere(1.0), Box([0.5, 0.5, 0.5]), smoothness=0.1)
    code = _compile(scene)
    assert _valid(code)
    assert _balanced(code)
    # smooth_min uses arithmetic on distances
    assert "min" in code or "-" in code


# ── 9. Translate transform ────────────────────────────────────────────────────


def test_translate_wgsl():
    from cadjoint.sdf.primitives.sphere import Sphere
    from cadjoint.sdf.transforms.affine.translate import Translate

    scene = Translate(Sphere(1.0), offset=jnp.array([1.0, 0.0, 0.0]))
    code = _compile(scene)
    assert _valid(code)
    assert _balanced(code)
    assert "1.000000" in code  # offset constant is embedded


# ── 10. Syntax sanity across all primitives ───────────────────────────────────


def test_cylinder_wgsl():
    from cadjoint.sdf.primitives.cylinder import Cylinder

    cyl = Cylinder(radius=0.5, height=1.0)
    code = _compile(cyl)
    assert _valid(code) and _balanced(code)


def test_capsule_wgsl():
    from cadjoint.sdf.primitives.capsule import Capsule

    cap = Capsule(radius=0.3, height=1.5)
    code = _compile(cap)
    assert _valid(code) and _balanced(code)


# ── complete built-in catalog ────────────────────────────────────────────────


def _builtin_scenes():
    from cadjoint.sdf.boolean import Difference, Intersection, Union, Xor
    from cadjoint.sdf.primitives import (
        Box,
        Capsule,
        Cylinder,
        Plane,
        RoundBox,
        Sphere,
        Torus,
    )
    from cadjoint.sdf.transforms.affine import Rotate, Scale, Translate
    from cadjoint.sdf.transforms.deformations import Twist

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


@pytest.mark.parametrize("scene_name", list(_builtin_scenes()))
def test_builtin_scene_compiles_to_wgsl(scene_name):
    code = _compile(_builtin_scenes()[scene_name])

    assert "?UNKNOWN?" not in code
    assert _balanced(code)
    assert "fn sdf(" in code


def test_compiler_enforces_sdf_signature():
    with pytest.raises(ValueError, match="shape \\(3,\\)"):
        _compile(jnp.sum, point=jnp.zeros(2, dtype=jnp.float32))

    with pytest.raises(ValueError, match="scalar float32 distance"):
        _compile(lambda point: point)


def test_rotate_and_scale_emit_required_operations():
    scenes = _builtin_scenes()
    rotate_wgsl = _compile(scenes["rotate"])
    scale_wgsl = _compile(scenes["scale"])

    assert "transpose(" in rotate_wgsl and "mat3x3<f32>" in rotate_wgsl
    assert "all(" in scale_wgsl and "3.402823e+38" in scale_wgsl
