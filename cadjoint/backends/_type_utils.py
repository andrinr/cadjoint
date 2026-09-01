"""JAX/MLIR shape and dtype helpers shared by the shader backends."""

from __future__ import annotations

import re

import numpy as np


def _shader_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Collapse singleton dimensions that shaders represent as scalars/vectors."""
    return tuple(dimension for dimension in shape if dimension != 1)


def shape_dtype_to_glsl(shape: tuple[int, ...], dtype) -> str:
    base = _base(dtype)
    shape = _shader_shape(shape)
    if not shape:
        return base
    if len(shape) == 1:
        n = shape[0]
        if n == 1:
            return base
        if 2 <= n <= 4:
            prefix = {
                "float": "vec",
                "int": "ivec",
                "uint": "uvec",
                "bool": "bvec",
            }[base]
            return f"{prefix}{n}"
        raise ValueError(f"GLSL vectors must contain 2 to 4 elements, got shape {shape}")
    if len(shape) == 2:
        m, n = shape
        if base == "float" and 2 <= m <= 4 and 2 <= n <= 4:
            return f"mat{m}" if m == n else f"mat{n}x{m}"
        raise ValueError(f"Cannot map matrix shape {shape} with dtype {np.dtype(dtype)} to GLSL")
    raise ValueError(f"Cannot map shape {shape} to a GLSL type")


def glsl_literal(val, shape: tuple[int, ...], dtype) -> str:
    value = np.asarray(val, dtype=np.dtype(dtype))
    shader_shape = _shader_shape(shape)
    # GLSL matrix constructors consume columns, while NumPy/StableHLO values
    # are flattened in row-major order.
    arr = value.T.ravel() if len(shader_shape) == 2 else value.ravel()
    base = _base(dtype)

    def fmt(v) -> str:
        if base == "float":
            fv = float(v)
            if np.isposinf(fv):
                return "3.402823e38"
            if np.isneginf(fv):
                return "-3.402823e38"
            if np.isnan(fv):
                raise ValueError("NaN constants cannot be represented portably in GLSL")
            return f"{fv:.6f}"
        if base == "int":
            return str(int(v))
        if base == "uint":
            return f"{int(v)}u"
        return "true" if v else "false"

    if not shader_shape:
        return fmt(arr[0] if len(arr) else 0)

    glsl_type = shape_dtype_to_glsl(shader_shape, dtype)
    return f"{glsl_type}({', '.join(fmt(v) for v in arr)})"


_MLIR_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "i1": np.bool_,
    "ui32": np.uint32,
}

_MLIR_DTYPE_RE = re.compile(r"^tensor<(.+)>$")


def parse_mlir_tensor_type(mlir_type_str: str) -> tuple[tuple[int, ...], np.dtype]:
    """Parse ``'tensor<3xf32>'`` → ``((3,), np.float32)``."""
    m = _MLIR_DTYPE_RE.match(mlir_type_str)
    if not m:
        raise ValueError(f"Expected a ranked MLIR tensor type, got {mlir_type_str!r}")
    inner = m.group(1)
    parts = inner.split("x")
    dtype = _MLIR_DTYPE_MAP.get(parts[-1])
    if dtype is None:
        raise ValueError(f"Shader backends do not support MLIR dtype {parts[-1]!r}")
    try:
        shape = tuple(int(dimension) for dimension in parts[:-1])
    except ValueError as exc:
        raise ValueError(
            f"Shader backends require static tensor shapes: {mlir_type_str!r}"
        ) from exc
    return shape, np.dtype(dtype)


def mlir_type_to_glsl(mlir_type_str: str) -> str:
    """Convert an MLIR tensor type string to the corresponding GLSL type."""
    shape, dtype = parse_mlir_tensor_type(mlir_type_str)
    return shape_dtype_to_glsl(shape, dtype)


def _base(dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return "float"
    if dtype == np.int32:
        return "int"
    if dtype == np.uint32:
        return "uint"
    if dtype == np.bool_:
        return "bool"
    raise ValueError(f"Shader backends do not support dtype {dtype}")
