"""StableHLO MLIR → GLSL compiler.

Pipeline:
  jax.export(jax.jit(fn))(*args).mlir_module()   →  StableHLO MLIR text
  StableHLOToGLSL.convert(mlir_text)              →  GLSL function string(s)

Each ``func.func`` in the module becomes a GLSL function.
``@main`` is renamed ``sdf``; helper functions keep their names.
Helpers are emitted before ``sdf`` so forward declarations are not needed.

Supported StableHLO ops are listed in ``_BINARY_OPS`` / ``_UNARY_OPS`` and
``_FuncEmitter._dispatch``. Unsupported ops raise ``NotImplementedError``
with a clear message pointing to this file.
"""

from __future__ import annotations

import re

import numpy as np

from ._type_utils import (
    glsl_literal,
    mlir_type_to_glsl,
    parse_mlir_tensor_type,
)

# ── loc-annotation pattern ───────────────────────────────────────────────────
_LOC_RE = re.compile(r"\s+loc\([^)]*(?:\([^)]*\)[^)]*)*\)")


def _strip_locs(text: str) -> str:
    """Remove MLIR location annotations that require extra dialect infra."""
    text = _LOC_RE.sub("", text)
    text = re.sub(r"^#loc.*\n", "", text, flags=re.MULTILINE)
    return text


# ── op dispatch tables ────────────────────────────────────────────────────────

_BINARY_OPS: dict[str, str] = {
    "stablehlo.add": "{0} + {1}",
    "stablehlo.subtract": "{0} - {1}",
    "stablehlo.multiply": "{0} * {1}",
    "stablehlo.divide": "{0} / {1}",
    "stablehlo.remainder": "mod({0}, {1})",
    "stablehlo.maximum": "max({0}, {1})",
    "stablehlo.minimum": "min({0}, {1})",
    "stablehlo.pow": "pow({0}, {1})",
    "stablehlo.atan2": "atan({0}, {1})",
    "stablehlo.and": "{0} && {1}",
    "stablehlo.or": "{0} || {1}",
    "stablehlo.xor": "{0} != {1}",
}

_UNARY_OPS: dict[str, str] = {
    "stablehlo.negate": "-({0})",
    "stablehlo.abs": "abs({0})",
    "stablehlo.sqrt": "sqrt({0})",
    "stablehlo.rsqrt": "inversesqrt({0})",
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

_COMPARE_DIRS: dict[str, str] = {
    "LT": "<",
    "LE": "<=",
    "GT": ">",
    "GE": ">=",
    "EQ": "==",
    "NE": "!=",
}


# ── per-function emitter ──────────────────────────────────────────────────────


class _FuncEmitter:
    """Converts one StableHLO ``func.func`` to a GLSL function string."""

    def __init__(self) -> None:
        self._val_names: dict = {}  # ir.Value → GLSL variable name
        self._counter = 0
        self._lines: list[str] = []

    def _fresh(self) -> str:
        name = f"_v{self._counter}"
        self._counter += 1
        return name

    def _resolve(self, val) -> str:
        return self._val_names.get(val, "?UNKNOWN?")

    def emit(self, func_op, fn_name: str) -> str:
        self._val_names = {}
        self._counter = 0
        self._lines = []

        blk = func_op.regions[0].blocks[0]

        # ── function arguments ───────────────────────────────────────────────
        arg_decls = []
        for i, arg in enumerate(blk.arguments):
            name = "p" if (i == 0 and fn_name == "sdf") else f"_arg{i}"
            self._val_names[arg] = name
            glsl_type = mlir_type_to_glsl(str(arg.type))
            arg_decls.append(f"{glsl_type} {name}")

        # ── ops in SSA order ─────────────────────────────────────────────────
        ret_name = "MISSING_RETURN"
        ret_type = "float"
        for op in blk.operations:
            if op.name in ("func.return", "stablehlo.return"):
                if op.operands:
                    ret_name = self._resolve(op.operands[0])
                    ret_type = mlir_type_to_glsl(str(op.operands[0].type))
                break  # always last op — stop here

            expr = self._dispatch(op)

            for i, res in enumerate(op.results):
                name = self._fresh()
                self._val_names[res] = name
                if expr is not None and i == 0:
                    glsl_type = mlir_type_to_glsl(str(res.type))
                    self._lines.append(f"    {glsl_type} {name} = {expr};")

        sig = f"{ret_type} {fn_name}({', '.join(arg_decls)})"
        body = "\n".join(self._lines)
        return f"{sig} {{\n{body}\n    return {ret_name};\n}}"

    # ── op dispatch ───────────────────────────────────────────────────────────

    def _dispatch(self, op) -> str | None:
        from jaxlib.mlir import ir as mlir_ir

        name = op.name
        a = [self._resolve(o) for o in op.operands]

        if name in _BINARY_OPS:
            return _BINARY_OPS[name].format(*a)

        if name in _UNARY_OPS:
            return _UNARY_OPS[name].format(*a)

        # ── type conversion ───────────────────────────────────────────────────
        if name == "stablehlo.convert":
            out = mlir_type_to_glsl(str(op.results[0].type))
            return f"{out}({a[0]})"

        # ── constant ─────────────────────────────────────────────────────────
        if name == "stablehlo.constant":
            val = np.array(op.attributes["value"])
            shape, dtype = parse_mlir_tensor_type(str(op.results[0].type))
            return glsl_literal(val, shape, dtype)

        # ── function call ─────────────────────────────────────────────────────
        if name == "func.call":
            callee = mlir_ir.FlatSymbolRefAttr(op.attributes["callee"]).value
            return f"{callee}({', '.join(a)})"

        # ── shape ops (broadcast, reshape) ────────────────────────────────────
        if name in ("stablehlo.broadcast_in_dim", "stablehlo.reshape"):
            in_type = str(op.operands[0].type)
            out_type_str = str(op.results[0].type)
            if in_type == out_type_str:
                return a[0]
            out = mlir_type_to_glsl(out_type_str)
            return f"{out}({a[0]})"

        # ── select / clamp ────────────────────────────────────────────────────
        if name == "stablehlo.select":
            return f"({a[0]} ? {a[1]} : {a[2]})"

        if name == "stablehlo.clamp":
            return f"clamp({a[1]}, {a[0]}, {a[2]})"

        # ── compare ───────────────────────────────────────────────────────────
        if name == "stablehlo.compare":
            direction = str(op.attributes["comparison_direction"])
            for k, v in _COMPARE_DIRS.items():
                if k in direction:
                    return f"{a[0]} {v} {a[1]}"
            return f"{a[0]} == {a[1]}"

        # ── dot product ───────────────────────────────────────────────────────
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

        # ── reduce (sum / max / min over a dimension) ─────────────────────────
        if name == "stablehlo.reduce":
            shape, _ = parse_mlir_tensor_type(str(op.operands[0].type))
            n = shape[0] if shape else 1
            inner_ops = list(op.regions[0].blocks[0].operations)
            compute = inner_ops[0].name if inner_ops else ""
            x, comps = a[0], list("xyzw"[:n])

            if n == 1:
                return f"{x}.x"
            if "add" in compute:
                ones = ", ".join(["1.0"] * n)
                return f"dot({x}, vec{n}({ones}))"
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
            raise NotImplementedError(
                f"stablehlo.reduce with inner computation '{compute}' "
                f"is not supported. Add it in jaxcad/backends/_stablehlo_emitter.py."
            )

        # ── slice (extract component range) ──────────────────────────────────
        if name == "stablehlo.slice":
            starts = list(op.attributes["start_indices"])
            limits = list(op.attributes["limit_indices"])
            if len(starts) == 1:
                swizzle = "xyzw"[starts[0] : limits[0]]
                return f"{a[0]}.{swizzle}"
            raise NotImplementedError("stablehlo.slice on multi-dim tensor not yet supported.")

        # ── concatenate ───────────────────────────────────────────────────────
        if name == "stablehlo.concatenate":
            out = mlir_type_to_glsl(str(op.results[0].type))
            return f"{out}({', '.join(a)})"

        # ── iota ──────────────────────────────────────────────────────────────
        if name == "stablehlo.iota":
            shape, _ = parse_mlir_tensor_type(str(op.results[0].type))
            n = shape[0] if shape else 1
            out = mlir_type_to_glsl(str(op.results[0].type))
            vals = ", ".join(f"{float(i):.1f}" for i in range(n))
            return f"{out}({vals})"

        raise NotImplementedError(
            f"StableHLO op '{name}' is not yet supported in the GLSL backend.\n"
            f"To add support, implement a case for it in:\n"
            f"  jaxcad/backends/_stablehlo_emitter.py  (_FuncEmitter._dispatch)"
        )


# ── public API ────────────────────────────────────────────────────────────────


class StableHLOToGLSL:
    """Compiles JAX functions to GLSL via StableHLO."""

    def compile(self, fn, *example_args) -> str:
        """Trace *fn* through ``jax.export`` and emit GLSL function(s).

        Args:
            fn: JAX-traceable callable ``(p: f32[3]) -> f32[]``.
            example_args: Example inputs for shape inference.

        Returns:
            Complete GLSL source for all emitted functions, ready to embed
            in a fragment shader.  The entry point is always named ``sdf``.
        """
        import jax
        from jax.export import export

        exported = export(jax.jit(fn))(*example_args)
        return self.convert(exported.mlir_module())

    def convert(self, mlir_text: str) -> str:
        """Convert raw StableHLO MLIR text to GLSL function string(s).

        Helper functions are emitted before the ``sdf`` entry point so no
        forward declarations are needed.
        """
        from jax._src.interpreters.mlir import make_ir_context
        from jaxlib.mlir import ir

        clean = _strip_locs(mlir_text)
        with make_ir_context():
            module = ir.Module.parse(clean)
            funcs = list(module.body.operations)

            glsl_parts: list[str] = []
            for func_op in reversed(funcs):  # helpers first, @main last
                fn_name = func_op.name.value
                if fn_name == "main":
                    fn_name = "sdf"
                emitter = _FuncEmitter()
                glsl_parts.append(emitter.emit(func_op, fn_name))

        return "\n\n".join(glsl_parts)
