"""StableHLO MLIR → WGSL compiler.

Same pipeline as the GLSL emitter but targeting WebGPU Shading Language.
Key differences from GLSL:
  - Types parameterised: f32, vec3<f32>, mat3x3<f32>
  - Function syntax: fn name(p: vec3<f32>) -> f32 { ... }
  - Immutable bindings: let _v0: f32 = expr;
  - atan2(y, x) instead of atan(y, x) for two-arg arctangent
  - inverseSqrt (capital S)
  - % for float remainder instead of mod()
"""

from __future__ import annotations

import numpy as np

from .._stablehlo_emitter import _strip_locs
from .._type_utils import parse_mlir_tensor_type

# ── WGSL type mapping ─────────────────────────────────────────────────────────

_WGSL_BASE = {
    np.float32: "f32",
    np.float64: "f32",
    np.int32: "i32",
    np.int64: "i32",
    np.bool_: "bool",
    np.uint32: "u32",
}


def _wgsl_base(dtype) -> str:
    return _WGSL_BASE.get(np.dtype(dtype).type, "f32")


def wgsl_type(shape, dtype) -> str:
    base = _wgsl_base(dtype)
    if not shape:
        return base
    if len(shape) == 1:
        n = shape[0]
        if n == 1:
            return base
        if n <= 4:
            return f"vec{n}<{base}>"
        return f"array<{base},{n}>"
    if len(shape) == 2:
        m, n = shape
        return f"mat{m}x{n}<f32>"
    raise ValueError(f"Cannot map shape {shape} to WGSL type")


def wgsl_literal(val, shape, dtype) -> str:
    arr = np.asarray(val, dtype=np.dtype(dtype)).ravel()
    base = _wgsl_base(dtype)

    def fmt(v) -> str:
        if base == "f32":
            fv = float(v)
            if np.isposinf(fv):
                return "1e38"
            if np.isneginf(fv):
                return "-1e38"
            if np.isnan(fv):
                return "0.0"
            return f"{fv:.6f}"
        if base in ("i32", "u32"):
            return str(int(v))
        return "true" if v else "false"

    if not shape or (len(shape) == 1 and shape[0] == 1):
        return fmt(arr[0] if len(arr) else 0)
    t = wgsl_type(shape, dtype)
    return f"{t}({', '.join(fmt(v) for v in arr)})"


def mlir_type_to_wgsl(mlir_type_str: str) -> str:
    shape, dtype = parse_mlir_tensor_type(mlir_type_str)
    return wgsl_type(shape, dtype)


# ── op tables ─────────────────────────────────────────────────────────────────

_BINARY: dict[str, str] = {
    "stablehlo.add": "{0} + {1}",
    "stablehlo.subtract": "{0} - {1}",
    "stablehlo.multiply": "{0} * {1}",
    "stablehlo.divide": "{0} / {1}",
    "stablehlo.remainder": "{0} % {1}",
    "stablehlo.maximum": "max({0}, {1})",
    "stablehlo.minimum": "min({0}, {1})",
    "stablehlo.pow": "pow({0}, {1})",
    "stablehlo.atan2": "atan2({0}, {1})",
    "stablehlo.and": "{0} && {1}",
    "stablehlo.or": "{0} || {1}",
    "stablehlo.xor": "{0} != {1}",
}

_UNARY: dict[str, str] = {
    "stablehlo.negate": "-({0})",
    "stablehlo.abs": "abs({0})",
    "stablehlo.sqrt": "sqrt({0})",
    "stablehlo.rsqrt": "inverseSqrt({0})",
    "stablehlo.exp": "exp({0})",
    "stablehlo.exp2": "exp2({0})",
    "stablehlo.log": "log({0})",
    "stablehlo.log1p": "log({0} + 1.0)",
    "stablehlo.sine": "sin({0})",
    "stablehlo.cosine": "cos({0})",
    "stablehlo.tanh": "tanh({0})",
    "stablehlo.sign": "sign({0})",
    "stablehlo.floor": "floor({0})",
    "stablehlo.ceil": "ceil({0})",
    "stablehlo.round_nearest_afz": "round({0})",
    "stablehlo.not": "!({0})",
}

_COMPARE: dict[str, str] = {
    "LT": "<",
    "LE": "<=",
    "GT": ">",
    "GE": ">=",
    "EQ": "==",
    "NE": "!=",
}


# ── per-function emitter ──────────────────────────────────────────────────────


class _WGSLFuncEmitter:
    def __init__(self) -> None:
        self._names: dict = {}
        self._counter = 0
        self._lines: list[str] = []

    def _fresh(self) -> str:
        name = f"_v{self._counter}"
        self._counter += 1
        return name

    def _resolve(self, val) -> str:
        return self._names.get(val, "?UNKNOWN?")

    def emit(self, func_op, fn_name: str) -> str:
        self._names = {}
        self._counter = 0
        self._lines = []

        blk = func_op.regions[0].blocks[0]

        # ── arguments ─────────────────────────────────────────────────────────
        params = []
        for i, arg in enumerate(blk.arguments):
            name = "p" if (i == 0 and fn_name == "sdf") else f"_arg{i}"
            self._names[arg] = name
            params.append(f"{name}: {mlir_type_to_wgsl(str(arg.type))}")

        # ── ops ───────────────────────────────────────────────────────────────
        ret_name, ret_type = "MISSING", "f32"
        for op in blk.operations:
            if op.name in ("func.return", "stablehlo.return"):
                if op.operands:
                    ret_name = self._resolve(op.operands[0])
                    ret_type = mlir_type_to_wgsl(str(op.operands[0].type))
                break

            expr = self._dispatch(op)
            for i, res in enumerate(op.results):
                name = self._fresh()
                self._names[res] = name
                if expr is not None and i == 0:
                    t = mlir_type_to_wgsl(str(res.type))
                    self._lines.append(f"    let {name}: {t} = {expr};")

        sig = f"fn {fn_name}({', '.join(params)}) -> {ret_type}"
        body = "\n".join(self._lines)
        return f"{sig} {{\n{body}\n    return {ret_name};\n}}"

    def _dispatch(self, op) -> str | None:
        from jaxlib.mlir import ir as mlir_ir

        name = op.name
        a = [self._resolve(o) for o in op.operands]

        if name in _BINARY:
            return _BINARY[name].format(*a)
        if name in _UNARY:
            return _UNARY[name].format(*a)

        if name == "stablehlo.convert":
            t = mlir_type_to_wgsl(str(op.results[0].type))
            return f"{t}({a[0]})"

        if name == "stablehlo.constant":
            val = np.array(op.attributes["value"])
            shape, dtype = parse_mlir_tensor_type(str(op.results[0].type))
            return wgsl_literal(val, shape, dtype)

        if name == "func.call":
            callee = mlir_ir.FlatSymbolRefAttr(op.attributes["callee"]).value
            return f"{callee}({', '.join(a)})"

        if name in ("stablehlo.broadcast_in_dim", "stablehlo.reshape"):
            in_t = str(op.operands[0].type)
            out_t = str(op.results[0].type)
            if in_t == out_t:
                return a[0]
            t = mlir_type_to_wgsl(out_t)
            return f"{t}({a[0]})"

        if name == "stablehlo.select":
            return f"select({a[2]}, {a[1]}, {a[0]})"  # WGSL: select(true_val, false_val, cond)

        if name == "stablehlo.clamp":
            return f"clamp({a[1]}, {a[0]}, {a[2]})"

        if name == "stablehlo.compare":
            direction = str(op.attributes["comparison_direction"])
            for k, v in _COMPARE.items():
                if k in direction:
                    return f"{a[0]} {v} {a[1]}"
            return f"{a[0]} == {a[1]}"

        if name == "stablehlo.dot_general":
            dnums = str(op.attributes["dot_dimension_numbers"])
            s0, _ = parse_mlir_tensor_type(str(op.operands[0].type))
            s1, _ = parse_mlir_tensor_type(str(op.operands[1].type))
            if (
                len(s0) == 1
                and len(s1) == 1
                and "lhs_contracting_dimensions = [0]" in dnums
                and "rhs_contracting_dimensions = [0]" in dnums
            ):
                return f"dot({a[0]}, {a[1]})"
            return f"{a[0]} * {a[1]}"

        if name == "stablehlo.reduce":
            shape, _ = parse_mlir_tensor_type(str(op.operands[0].type))
            n = shape[0] if shape else 1
            inner = list(op.regions[0].blocks[0].operations)
            compute = inner[0].name if inner else ""
            x, comps = a[0], list("xyzw"[:n])
            if n == 1:
                return f"{x}.x"
            if "add" in compute:
                ones = ", ".join(["1.0"] * n)
                return f"dot({x}, vec{n}<f32>({ones}))"
            if "maximum" in compute:
                expr = f"{x}.{comps[0]}"
                for c in comps[1:]:
                    expr = f"max({expr}, {x}.{c})"
                return expr
            if "minimum" in compute:
                expr = f"{x}.{comps[0]}"
                for c in comps[1:]:
                    expr = f"min({expr}, {x}.{c})"
                return expr
            raise NotImplementedError(f"stablehlo.reduce with '{compute}'")

        if name == "stablehlo.slice":
            starts = list(op.attributes["start_indices"])
            limits = list(op.attributes["limit_indices"])
            if len(starts) == 1:
                return f"{a[0]}.{'xyzw'[starts[0] : limits[0]]}"
            raise NotImplementedError("multi-dim slice")

        if name == "stablehlo.concatenate":
            t = mlir_type_to_wgsl(str(op.results[0].type))
            return f"{t}({', '.join(a)})"

        if name == "stablehlo.iota":
            shape, _ = parse_mlir_tensor_type(str(op.results[0].type))
            n = shape[0] if shape else 1
            t = mlir_type_to_wgsl(str(op.results[0].type))
            return f"{t}({', '.join(f'{float(i):.1f}' for i in range(n))})"

        raise NotImplementedError(
            f"StableHLO op '{name}' not yet supported in the WGSL backend.\n"
            f"Add it in jaxcad/backends/wgsl/_wgsl_emitter.py"
        )


# ── public API ────────────────────────────────────────────────────────────────


class StableHLOToWGSL:
    """Compiles JAX functions to WGSL via StableHLO."""

    def compile(self, fn, *example_args) -> str:
        import jax
        from jax.export import export

        exported = export(jax.jit(fn))(*example_args)
        return self.convert(exported.mlir_module())

    def convert(self, mlir_text: str) -> str:
        from jax._src.interpreters.mlir import make_ir_context
        from jaxlib.mlir import ir

        clean = _strip_locs(mlir_text)
        with make_ir_context():
            module = ir.Module.parse(clean)
            funcs = list(module.body.operations)
            parts: list[str] = []
            for func_op in reversed(funcs):
                fn_name = func_op.name.value
                if fn_name == "main":
                    fn_name = "sdf"
                parts.append(_WGSLFuncEmitter().emit(func_op, fn_name))
        return "\n\n".join(parts)
