"""StableHLO to jax-js TypeScript, one ``func.func`` at a time.

The translation is op for op: every StableHLO op becomes one call into
jax-js's numpy namespace, so the emitted function computes what the traced
JAX function computed, in the same order.  jax-js arrays are reference
counted -- an operation consumes its inputs -- so the emitter counts every
SSA value's uses and takes ``.ref`` on all but the last.

An op without a mapping raises :class:`UnsupportedOp`, naming it; that is
how a node type that lowers to something new (control flow, gathers) reports
what the translator would still need.
"""

from __future__ import annotations

import re

import jax
import numpy as np
from jax.export import export

from cadjoint.backends.wgsl._wgsl_emitter import _strip_locs, parse_mlir_tensor_type

__all__ = ["UnsupportedOp", "function_to_ts"]


class UnsupportedOp(NotImplementedError):
    """A StableHLO op the jax-js translator has no mapping for."""


_DTYPES = {
    np.dtype(np.float32): "np.float32",
    np.dtype(np.bool_): "np.bool",
    np.dtype(np.int32): "np.int32",
}
_BINARY = {
    "stablehlo.add": "np.add",
    "stablehlo.subtract": "np.subtract",
    "stablehlo.multiply": "np.multiply",
    "stablehlo.divide": "np.divide",
    "stablehlo.maximum": "np.maximum",
    "stablehlo.minimum": "np.minimum",
    "stablehlo.power": "np.power",
    "stablehlo.atan2": "np.atan2",
    # C semantics (the sign of the dividend), as StableHLO defines it.
    "stablehlo.remainder": "np.fmod",
}
_UNARY = {
    "stablehlo.negate": "np.negative",
    "stablehlo.abs": "np.abs",
    "stablehlo.sqrt": "np.sqrt",
    "stablehlo.sine": "np.sin",
    "stablehlo.cosine": "np.cos",
    "stablehlo.exponential": "np.exp",
    "stablehlo.log": "np.log",
    "stablehlo.tanh": "np.tanh",
    "stablehlo.sign": "np.sign",
    "stablehlo.floor": "np.floor",
    "stablehlo.ceil": "np.ceil",
    "stablehlo.is_finite": "np.isfinite",
}
_COMPARE = {
    "EQ": "np.equal",
    "NE": "np.notEqual",
    "GE": "np.greaterEqual",
    "GT": "np.greater",
    "LE": "np.lessEqual",
    "LT": "np.less",
}
_LOGICAL = {"stablehlo.and": "np.logicalAnd", "stablehlo.or": "np.logicalOr"}


def _ints(attribute) -> list[int]:
    """Integers of an `array<i64: 0, 1>` or `dense<[0, 1]> : tensor<2xi64>` attribute."""
    text = str(attribute)
    if text.startswith("array<"):
        body = text[len("array<") : -1]
        return [int(v) for v in re.findall(r"-?\d+", body.split(":", 1)[1])] if ":" in body else []
    scalar = re.match(r"^(-?\d+) : i\d+$", text)
    if scalar:
        return [int(scalar.group(1))]
    match = re.match(r"dense<(.*?)>", text)
    return [int(v) for v in re.findall(r"-?\d+", match.group(1))] if match else []


def _shape(value) -> tuple[int, ...]:
    return parse_mlir_tensor_type(str(value.type))[0]


class _FunctionEmitter:
    """One func.func -> one TS function, with jax-js ownership made explicit."""

    def __init__(self, ts_names: dict[str, str]):
        self.ts_names = ts_names

    def emit(self, func_op, name: str) -> str:
        block = func_op.regions[0].blocks[0]
        uses: dict = {}
        for op in block.operations:
            for operand in op.operands:
                uses[operand] = uses.get(operand, 0) + 1
        self.remaining = dict(uses)
        self.names = {}
        self.lines: list[str] = []
        args = []
        for i, arg in enumerate(block.arguments):
            self.names[arg] = f"a{i}"
            args.append(f"a{i}: np.Array")
            if uses.get(arg, 0) == 0:
                self.lines.append(f"  a{i}.dispose();")
        counter = 0
        for op in block.operations:
            if op.name in ("func.return", "stablehlo.return"):
                values = [self.use(v) for v in op.operands]
                self.lines.append(
                    f"  return {values[0] if len(values) == 1 else '[' + ', '.join(values) + ']'};"
                )
                continue
            expression = self.expression(op)
            result = op.results[0]
            var = f"t{counter}"
            counter += 1
            self.names[result] = var
            self.lines.append(f"  const {var} = {expression};")
            if uses.get(result, 0) == 0:
                self.lines.append(f"  {var}.dispose();")
        return f"function {name}({', '.join(args)}): np.Array {{\n" + "\n".join(self.lines) + "\n}"

    def use(self, value) -> str:
        """The value's name; .ref on every use but the last."""
        name = self.names[value]
        self.remaining[value] -= 1
        return name if self.remaining[value] == 0 else f"{name}.ref"

    def expression(self, op) -> str:
        name = op.name
        result_shape, result_dtype = parse_mlir_tensor_type(str(op.results[0].type))
        if name == "stablehlo.constant":
            value = np.array(op.attributes["value"])
            dtype = _DTYPES[np.dtype(value.dtype)]
            if value.dtype == np.bool_:
                literal = repr(value.tolist()).replace("True", "true").replace("False", "false")
            else:
                literal = (
                    repr(value.astype(float).tolist())
                    .replace("inf", "Infinity")
                    .replace("nan", "NaN")
                )
            return f"np.array({literal}, {{ dtype: {dtype} }})"
        if name in _BINARY:
            a, b = (self.use(v) for v in op.operands)
            return f"{_BINARY[name]}({a}, {b})"
        if name in _UNARY:
            return f"{_UNARY[name]}({self.use(op.operands[0])})"
        if name == "stablehlo.rsqrt":
            return f"np.reciprocal(np.sqrt({self.use(op.operands[0])}))"
        if name == "stablehlo.compare":
            direction = re.search(
                r"comparison_direction (\w+)", str(op.attributes["comparison_direction"])
            ).group(1)
            a, b = (self.use(v) for v in op.operands)
            return f"{_COMPARE[direction]}({a}, {b})"
        if name in _LOGICAL:
            a, b = (self.use(v) for v in op.operands)
            return f"{_LOGICAL[name]}({a}, {b})"
        if name == "stablehlo.not":
            return f"np.logicalNot({self.use(op.operands[0])})"
        if name == "stablehlo.select":
            pred, a, b = (self.use(v) for v in op.operands)
            return f"np.where({pred}, {a}, {b})"
        if name == "stablehlo.clamp":
            lo, x, hi = (self.use(v) for v in op.operands)
            return f"np.clip({x}, {lo}, {hi})"
        if name == "stablehlo.convert":
            return f"np.astype({self.use(op.operands[0])}, {_DTYPES[np.dtype(result_dtype)]})"
        if name == "stablehlo.reshape":
            return f"np.reshape({self.use(op.operands[0])}, {list(result_shape)})"
        if name == "stablehlo.broadcast_in_dim":
            operand = op.operands[0]
            dims = _ints(op.attributes["broadcast_dimensions"])
            in_shape = _shape(operand)
            staged = [1] * len(result_shape)
            for source, target in enumerate(dims):
                staged[target] = in_shape[source]
            x = self.use(operand)
            if tuple(staged) == tuple(result_shape):
                return f"np.reshape({x}, {list(result_shape)})"
            return f"np.broadcastTo(np.reshape({x}, {staged}), {list(result_shape)})"
        if name == "stablehlo.slice":
            starts = _ints(op.attributes["start_indices"])
            limits = _ints(op.attributes["limit_indices"])
            strides = _ints(op.attributes["strides"])
            if any(s != 1 for s in strides):
                raise UnsupportedOp("strided slice")
            index = ", ".join(f"[{s}, {e}]" for s, e in zip(starts, limits))
            return f"{self.use(op.operands[0])}.slice({index})"
        if name == "stablehlo.concatenate":
            axis = _ints(op.attributes["dimension"])[0]
            parts = ", ".join(self.use(v) for v in op.operands)
            return f"np.concatenate([{parts}], {axis})"
        if name == "stablehlo.transpose":
            return (
                f"np.transpose({self.use(op.operands[0])}, {_ints(op.attributes['permutation'])})"
            )
        if name == "stablehlo.reduce":
            body = op.regions[0].blocks[0].operations[0].name
            reducer = {
                "stablehlo.add": "np.sum",
                "stablehlo.multiply": "np.prod",
                "stablehlo.maximum": "np.max",
                "stablehlo.minimum": "np.min",
                "stablehlo.and": "np.all",
                "stablehlo.or": "np.any",
            }[body]
            axes = _ints(op.attributes["dimensions"])
            x = self.use(op.operands[0])
            init = self.use(op.operands[1])  # identity of the reducer; consumed, not used
            self.lines.append(f"  {init}.dispose();")
            return f"{reducer}({x}, {axes if len(axes) > 1 else axes[0]})"
        if name == "stablehlo.dot_general":
            dims = str(op.attributes["dot_dimension_numbers"])
            lhs_c = [
                int(v)
                for v in re.search(r"lhs_contracting_dimensions = \[([^\]]*)\]", dims)
                .group(1)
                .split(",")
                if v.strip()
            ]
            rhs_c = [
                int(v)
                for v in re.search(r"rhs_contracting_dimensions = \[([^\]]*)\]", dims)
                .group(1)
                .split(",")
                if v.strip()
            ]
            if "batching" in dims and re.search(r"batching_dimensions = \[\d", dims):
                raise UnsupportedOp("batched dot_general")
            lhs_rank, rhs_rank = len(_shape(op.operands[0])), len(_shape(op.operands[1]))
            letters = iter("abcdefgh")
            lhs = [next(letters) for _ in range(lhs_rank)]
            rhs = [next(letters) for _ in range(rhs_rank)]
            for left, right in zip(lhs_c, rhs_c):
                rhs[right] = lhs[left]
            out = [c for i, c in enumerate(lhs) if i not in lhs_c] + [
                c for i, c in enumerate(rhs) if i not in rhs_c
            ]
            a, b = (self.use(v) for v in op.operands)
            return f'np.einsum("{"".join(lhs)},{"".join(rhs)}->{"".join(out)}", {a}, {b})'
        if name == "func.call":
            callee = str(op.attributes["callee"]).lstrip("@")
            return f"{self.ts_names[callee]}({', '.join(self.use(v) for v in op.operands)})"
        raise UnsupportedOp(f"no jax-js translation for {name}")


def function_to_ts(name: str, fn, example_args) -> str:
    """Trace ``fn`` on ``example_args`` and translate it to a TypeScript function.

    Args:
        name: The emitted function's name.  Callees StableHLO split out
            (``func.call``) are emitted first, as ``{name}__{callee}``.
        fn: A JAX-traceable function of arrays.
        example_args: Arrays (or ``jax.ShapeDtypeStruct``) fixing shapes and dtypes.

    Returns:
        TypeScript source: the callees, then ``function name(a0, a1, ...)``.
    """
    from jax._src.interpreters.mlir import make_ir_context
    from jaxlib.mlir import ir

    text = export(jax.jit(fn))(*example_args).mlir_module()
    with make_ir_context():
        module = ir.Module.parse(_strip_locs(text))
        funcs = list(module.body.operations)
        ts_names = {
            f.name.value: (name if f.name.value == "main" else f"{name}__{f.name.value}")
            for f in funcs
        }
        # Callees first, so every function is defined before it is called.
        ordered = [f for f in funcs if f.name.value != "main"]
        ordered += [f for f in funcs if f.name.value == "main"]
        return "\n\n".join(
            _FunctionEmitter(ts_names).emit(f, ts_names[f.name.value]) for f in ordered
        )
