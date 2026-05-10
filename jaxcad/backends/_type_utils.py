"""JAX/MLIR shape+dtype → GLSL type mapping and literal formatting."""

from __future__ import annotations

import re

import numpy as np


def shape_dtype_to_glsl(shape: tuple[int, ...], dtype) -> str:
    base = _base(dtype)
    if not shape:
        return base
    if len(shape) == 1:
        n = shape[0]
        if n == 1:
            return base
        if n <= 4:
            prefix = {"float": "vec", "int": "ivec", "bool": "bvec"}.get(base, "vec")
            return f"{prefix}{n}"
        return f"{base}[{n}]"
    if len(shape) == 2:
        m, n = shape
        if base == "float" and m <= 4 and n <= 4:
            return f"mat{m}" if m == n else f"mat{n}x{m}"
        return f"{base}[{m * n}]"
    raise ValueError(f"Cannot map shape {shape} to a GLSL type")


def glsl_literal(val, shape: tuple[int, ...], dtype) -> str:
    arr = np.asarray(val, dtype=np.dtype(dtype)).ravel()
    base = _base(dtype)

    def fmt(v) -> str:
        if base == "float":
            fv = float(v)
            if np.isposinf(fv):
                return "(1.0 / 0.0)"
            if np.isneginf(fv):
                return "(-1.0 / 0.0)"
            if np.isnan(fv):
                return "(0.0 / 0.0)"
            return f"{fv:.6f}"
        if base == "int":
            return str(int(v))
        return "true" if v else "false"

    if not shape or (len(shape) == 1 and shape[0] == 1):
        return fmt(arr[0] if len(arr) else 0)

    glsl_type = shape_dtype_to_glsl(shape, dtype)
    return f"{glsl_type}({', '.join(fmt(v) for v in arr)})"


_MLIR_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "i1": np.bool_,
    "ui32": np.uint32,
}

_MLIR_DTYPE_RE = re.compile(r"tensor<(.+)>")


def parse_mlir_tensor_type(mlir_type_str: str) -> tuple[tuple[int, ...], np.dtype]:
    """Parse ``'tensor<3xf32>'`` → ``((3,), np.float32)``."""
    m = _MLIR_DTYPE_RE.match(mlir_type_str)
    if not m:
        return (), np.float32
    inner = m.group(1)
    parts = inner.split("x")
    dtype = _MLIR_DTYPE_MAP.get(parts[-1], np.float32)
    shape = tuple(int(d) for d in parts[:-1]) if len(parts) > 1 else ()
    return shape, dtype


def mlir_type_to_glsl(mlir_type_str: str) -> str:
    """Convert an MLIR tensor type string to the corresponding GLSL type."""
    shape, dtype = parse_mlir_tensor_type(mlir_type_str)
    return shape_dtype_to_glsl(shape, dtype)


def _base(dtype) -> str:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return "float"
    if np.issubdtype(dtype, np.integer):
        return "int"
    if dtype == np.bool_:
        return "bool"
    return "float"
