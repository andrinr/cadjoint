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

import re

import numpy as np

from .._stablehlo_emitter import _strip_locs, _validate_point_shader_export
from .._type_utils import _shader_shape, parse_mlir_tensor_type

# ── WGSL type mapping ─────────────────────────────────────────────────────────

_WGSL_BASE = {
    np.float32: "f32",
    np.int32: "i32",
    np.bool_: "bool",
    np.uint32: "u32",
}


def _wgsl_base(dtype) -> str:
    try:
        return _WGSL_BASE[np.dtype(dtype).type]
    except KeyError as exc:
        raise ValueError(f"WGSL does not support dtype {np.dtype(dtype)}") from exc


def wgsl_type(shape, dtype) -> str:
    base = _wgsl_base(dtype)
    shape = _shader_shape(shape)
    if not shape:
        return base
    if len(shape) == 1:
        n = shape[0]
        if n == 1:
            return base
        if 2 <= n <= 4:
            return f"vec{n}<{base}>"
        raise ValueError(f"WGSL vectors must contain 2 to 4 elements, got shape {shape}")
    if len(shape) == 2:
        m, n = shape
        if base == "f32" and 2 <= m <= 4 and 2 <= n <= 4:
            # WGSL spells matrices as columns x rows.
            return f"mat{n}x{m}<f32>"
        raise ValueError(f"Cannot map matrix shape {shape} with dtype {np.dtype(dtype)} to WGSL")
    raise ValueError(f"Cannot map shape {shape} to WGSL type")


def wgsl_literal(val, shape, dtype) -> str:
    value = np.asarray(val, dtype=np.dtype(dtype))
    shader_shape = _shader_shape(shape)
    # WGSL matrix constructors consume columns, while StableHLO constants are
    # flattened in row-major order.
    arr = value.T.ravel() if len(shader_shape) == 2 else value.ravel()
    base = _wgsl_base(dtype)

    def fmt(v) -> str:
        if base == "f32":
            fv = float(v)
            if np.isposinf(fv):
                return "3.402823e38"
            if np.isneginf(fv):
                return "-3.402823e38"
            if np.isnan(fv):
                raise ValueError("NaN constants cannot be represented portably in WGSL")
            return f"{fv:.6f}"
        if base == "i32":
            return str(int(v))
        if base == "u32":
            return f"{int(v)}u"
        return "true" if v else "false"

    if not shader_shape:
        return fmt(arr[0] if len(arr) else 0)
    t = wgsl_type(shader_shape, dtype)
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
}

_COMPARE: dict[str, str] = {
    "LT": "<",
    "LE": "<=",
    "GT": ">",
    "GE": ">=",
    "EQ": "==",
    "NE": "!=",
}

_WGSL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validate_entry_point(entry_point: str) -> None:
    if _WGSL_IDENTIFIER.fullmatch(entry_point) is None:
        raise ValueError(f"WGSL entry point must be a valid identifier, got {entry_point!r}")


# ── per-function emitter ──────────────────────────────────────────────────────


class _WGSLFuncEmitter:
    def __init__(self, symbol_names: dict[str, str]) -> None:
        self._symbol_names = symbol_names
        self._names: dict = {}
        self._counter = 0
        self._lines: list[str] = []

    def _fresh(self) -> str:
        name = f"_v{self._counter}"
        self._counter += 1
        return name

    def _resolve(self, val) -> str:
        try:
            return self._names[val]
        except KeyError as exc:
            raise RuntimeError(f"StableHLO value {val} was used before it was emitted") from exc

    def emit(self, func_op, fn_name: str, *, point_entry: bool = False) -> str:
        self._names = {}
        self._counter = 0
        self._lines = []

        blk = func_op.regions[0].blocks[0]

        # ── arguments ─────────────────────────────────────────────────────────
        params = []
        for i, arg in enumerate(blk.arguments):
            name = "p" if (i == 0 and point_entry) else f"_arg{i}"
            self._names[arg] = name
            params.append(f"{name}: {mlir_type_to_wgsl(str(arg.type))}")
        # JAX may remove an unused point from the lowered main function when,
        # for example, a scene has one constant material. Keep the public
        # point-query contract stable even though the generated body ignores it.
        if point_entry and not params:
            params.append("p: vec3<f32>")

        # ── ops ───────────────────────────────────────────────────────────────
        ret_name, ret_type = "MISSING", "f32"
        for op in blk.operations:
            if op.name in ("func.return", "stablehlo.return"):
                if len(op.operands) != 1:
                    raise NotImplementedError("Shader SDF functions must return exactly one value")
                ret_name = self._resolve(op.operands[0])
                ret_type = mlir_type_to_wgsl(str(op.operands[0].type))
                break

            if len(op.results) > 1:
                raise NotImplementedError(
                    f"StableHLO op '{op.name}' returns multiple values, which is not supported"
                )
            expr = self._dispatch(op)
            for res in op.results:
                name = self._fresh()
                self._names[res] = name
                if expr is not None:
                    t = mlir_type_to_wgsl(str(res.type))
                    self._lines.append(f"    let {name}: {t} = {expr};")

        sig = f"fn {fn_name}({', '.join(params)}) -> {ret_type}"
        body = "\n".join(self._lines)
        return f"{sig} {{\n{body}\n    return {ret_name};\n}}"

    def _dispatch(self, op) -> str | None:
        from jaxlib.mlir import ir as mlir_ir

        name = op.name
        a = [self._resolve(o) for o in op.operands]
        result_shape, result_dtype = (
            parse_mlir_tensor_type(str(op.results[0].type))
            if op.results
            else ((), np.dtype(np.float32))
        )

        if name in _BINARY:
            return _BINARY[name].format(*a)
        if name in _UNARY:
            return _UNARY[name].format(*a)

        if name in ("stablehlo.and", "stablehlo.or", "stablehlo.xor"):
            operator = {
                "stablehlo.and": "&&" if result_dtype == np.bool_ else "&",
                "stablehlo.or": "||" if result_dtype == np.bool_ else "|",
                "stablehlo.xor": "!=" if result_dtype == np.bool_ else "^",
            }[name]
            if result_dtype != np.bool_ or not result_shape:
                return f"{a[0]} {operator} {a[1]}"
            return self._componentwise_binary(a[0], a[1], result_shape, operator)

        if name == "stablehlo.not":
            if result_dtype != np.bool_:
                return f"~({a[0]})"
            if not result_shape:
                return f"!({a[0]})"
            size = self._vector_size(result_shape)
            components = "xyzw"[:size]
            values = ", ".join(f"!({a[0]}.{component})" for component in components)
            return f"vec{size}<bool>({values})"

        if name == "stablehlo.convert":
            t = mlir_type_to_wgsl(str(op.results[0].type))
            return f"{t}({a[0]})"

        if name == "stablehlo.constant":
            val = np.array(op.attributes["value"])
            shape, dtype = parse_mlir_tensor_type(str(op.results[0].type))
            return wgsl_literal(val, shape, dtype)

        if name == "func.call":
            callee = mlir_ir.FlatSymbolRefAttr(op.attributes["callee"]).value
            try:
                emitted_callee = self._symbol_names[callee]
            except KeyError as exc:
                raise RuntimeError(f"StableHLO calls unknown function '{callee}'") from exc
            return f"{emitted_callee}({', '.join(a)})"

        if name in ("stablehlo.broadcast_in_dim", "stablehlo.reshape"):
            in_t = mlir_type_to_wgsl(str(op.operands[0].type))
            out_t = str(op.results[0].type)
            t = mlir_type_to_wgsl(out_t)
            if in_t == t:
                return a[0]
            return f"{t}({a[0]})"

        if name == "stablehlo.select":
            # WGSL select(false_value, true_value, condition).
            return f"select({a[2]}, {a[1]}, {a[0]})"

        if name == "stablehlo.clamp":
            return f"clamp({a[1]}, {a[0]}, {a[2]})"

        if name == "stablehlo.compare":
            direction = str(op.attributes["comparison_direction"])
            for k, v in _COMPARE.items():
                if k in direction:
                    return f"{a[0]} {v} {a[1]}"
            raise NotImplementedError(f"Unknown StableHLO comparison direction: {direction}")

        if name == "stablehlo.is_finite":
            # WGSL has no isfinite/isinf/isnan builtin. Comparisons with NaN
            # are false, and infinities exceed the largest finite f32 value.
            largest_finite = "3.402823e+38"
            if not result_shape:
                return f"abs({a[0]}) <= {largest_finite}"
            size = self._vector_size(result_shape)
            components = "xyzw"[:size]
            values = ", ".join(
                f"abs({a[0]}.{component}) <= {largest_finite}" for component in components
            )
            return f"vec{size}<bool>({values})"

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
            if len(s0) == 2 and len(s1) == 1 and "lhs_contracting_dimensions = [1]" in dnums:
                return f"{a[0]} * {a[1]}"
            if len(s0) == 1 and len(s1) == 2 and "rhs_contracting_dimensions = [0]" in dnums:
                return f"{a[0]} * {a[1]}"
            if len(s0) == 2 and len(s1) == 2:
                return f"{a[0]} * {a[1]}"
            raise NotImplementedError(f"Unsupported stablehlo.dot_general: {dnums}")

        if name == "stablehlo.transpose":
            permutation = self._integer_list(op.attributes["permutation"])
            if len(result_shape) == 2 and permutation == [1, 0]:
                return f"transpose({a[0]})"
            raise NotImplementedError(
                f"Only two-dimensional matrix transpose is supported, got {permutation}"
            )

        if name == "stablehlo.reduce":
            shape, dtype = parse_mlir_tensor_type(str(op.operands[0].type))
            dimensions = self._integer_list(op.attributes["dimensions"])
            if len(shape) != 1 or dimensions != [0]:
                raise NotImplementedError(
                    f"Only one-dimensional reductions are supported, got shape={shape}, "
                    f"dimensions={dimensions}"
                )
            n = shape[0]
            inner = list(op.regions[0].blocks[0].operations)
            compute = inner[0].name if inner else ""
            x, initial = a
            comps = list("xyzw"[:n])
            if "add" in compute:
                if n == 1:
                    reduced = x
                elif dtype == np.float32:
                    ones = ", ".join(["1.0"] * n)
                    reduced = f"dot({x}, vec{n}<f32>({ones}))"
                else:
                    reduced = " + ".join(f"{x}.{component}" for component in comps)
                return f"{initial} + {reduced}"
            if "maximum" in compute or "minimum" in compute:
                function = "max" if "maximum" in compute else "min"
                reduced = x if n == 1 else f"{x}.{comps[0]}"
                for c in comps[1:]:
                    reduced = f"{function}({reduced}, {x}.{c})"
                return f"{function}({initial}, {reduced})"
            if "and" in compute:
                reduced = x if n == 1 else f"all({x})"
                return f"{initial} && {reduced}"
            if "or" in compute:
                reduced = x if n == 1 else f"any({x})"
                return f"{initial} || {reduced}"
            raise NotImplementedError(f"stablehlo.reduce with '{compute}'")

        if name == "stablehlo.slice":
            starts = self._integer_list(op.attributes["start_indices"])
            limits = self._integer_list(op.attributes["limit_indices"])
            strides = self._integer_list(op.attributes["strides"])
            if len(starts) == 1 and strides == [1] and limits[0] <= 4:
                return f"{a[0]}.{'xyzw'[starts[0] : limits[0]]}"
            raise NotImplementedError("multi-dim slice")

        if name == "stablehlo.concatenate":
            t = mlir_type_to_wgsl(str(op.results[0].type))
            if len(result_shape) == 2:
                dimension = self._integer_attribute(op.attributes["dimension"])
                rows, columns = result_shape
                if dimension == 0:
                    row_matrix = f"mat{rows}x{columns}<f32>({', '.join(a)})"
                    return f"transpose({row_matrix})"
                if dimension == 1:
                    return f"{t}({', '.join(a)})"
                raise NotImplementedError(f"Invalid matrix concatenate dimension {dimension}")
            return f"{t}({', '.join(a)})"

        if name == "stablehlo.iota":
            shape, dtype = parse_mlir_tensor_type(str(op.results[0].type))
            n = shape[0] if shape else 1
            return wgsl_literal(np.arange(n), shape, dtype)

        raise NotImplementedError(
            f"StableHLO op '{name}' not yet supported in the WGSL backend.\n"
            f"Add it in jaxcad/backends/wgsl/_wgsl_emitter.py"
        )

    @staticmethod
    def _integer_list(attribute) -> list[int]:
        return [int(str(value)) for value in attribute]

    @staticmethod
    def _integer_attribute(attribute) -> int:
        return int(str(attribute).split()[0])

    @staticmethod
    def _vector_size(shape: tuple[int, ...]) -> int:
        dimensions = [dimension for dimension in shape if dimension != 1]
        if len(dimensions) != 1 or not 2 <= dimensions[0] <= 4:
            raise NotImplementedError(f"Expected a shader vector shape, got {shape}")
        return dimensions[0]

    def _componentwise_binary(
        self,
        lhs: str,
        rhs: str,
        shape: tuple[int, ...],
        operator: str,
    ) -> str:
        size = self._vector_size(shape)
        components = "xyzw"[:size]
        values = ", ".join(
            f"{lhs}.{component} {operator} {rhs}.{component}" for component in components
        )
        return f"vec{size}<bool>({values})"


# ── public API ────────────────────────────────────────────────────────────────


class StableHLOToWGSL:
    """Compiles JAX functions to WGSL via StableHLO."""

    def compile(
        self,
        fn,
        *example_args,
        entry_point: str = "sdf",
        output_shape: tuple[int, ...] = (),
        output_description: str = "scalar float32 distance",
    ) -> str:
        import jax
        from jax.export import export

        _validate_entry_point(entry_point)
        exported = export(jax.jit(fn))(*example_args)
        _validate_point_shader_export(
            exported,
            shader_description=(
                "An SDF shader"
                if entry_point == "sdf" and output_shape == ()
                else f"The {entry_point!r} shader"
            ),
            output_shape=output_shape,
            output_description=output_description,
        )
        return self.convert(exported.mlir_module(), entry_point=entry_point)

    def convert(self, mlir_text: str, *, entry_point: str = "sdf") -> str:
        from jax._src.interpreters.mlir import make_ir_context
        from jaxlib.mlir import ir

        _validate_entry_point(entry_point)

        clean = _strip_locs(mlir_text)
        with make_ir_context():
            module = ir.Module.parse(clean)
            funcs = list(module.body.operations)
            symbol_names = {
                func_op.name.value: (
                    entry_point
                    if func_op.name.value == "main"
                    else f"{entry_point}__{func_op.name.value}"
                )
                for func_op in funcs
            }
            parts: list[str] = []
            for func_op in reversed(funcs):
                original_name = func_op.name.value
                parts.append(
                    _WGSLFuncEmitter(symbol_names).emit(
                        func_op,
                        symbol_names[original_name],
                        point_entry=original_name == "main",
                    )
                )
        return "\n\n".join(parts)
