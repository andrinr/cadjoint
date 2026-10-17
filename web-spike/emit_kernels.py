"""Emit jax-js kernels from cadjoint's own SDF code, through StableHLO.

The single source of truth for every distance formula stays the Python in
cadjoint.  Each node type's per-point function -- a primitive's ``sdf``, a
transform's ``_transform_point``, a field op's or boolean's ``sdf`` -- is
traced with JAX, exported to StableHLO, and translated op by op into a
TypeScript function on jax-js.  Nothing about the maths is written twice.

Writes ``src/kernels.gen.ts``. The emitted code follows jax-js's ownership
rules mechanically: every SSA value's uses are counted, and every use but
the last takes ``.ref``.

Usage: python emit_kernels.py
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.export import export

from cadjoint.backends.wgsl._wgsl_emitter import _strip_locs, parse_mlir_tensor_type
from cadjoint.sdf._lowering import scalar_lowering
from cadjoint.sdf.boolean.difference import Difference
from cadjoint.sdf.boolean.intersection import Intersection
from cadjoint.sdf.boolean.union import Union
from cadjoint.sdf.primitives.box import Box
from cadjoint.sdf.primitives.cylinder import Cylinder
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon
from cadjoint.sdf.primitives.round_box import RoundBox
from cadjoint.sdf.primitives.sphere import Sphere
from cadjoint.sdf.transforms.affine.rotate import Rotate
from cadjoint.sdf.transforms.affine.translate import Translate
from cadjoint.sdf.transforms.fields import Offset, Shell

HERE = Path(__file__).parent
F32 = jnp.float32
POINT = jnp.zeros(3, F32)
S = jnp.zeros((), F32)
V2 = jnp.zeros(2, F32)
V3 = jnp.zeros(3, F32)


def _polygon(count: int):
    names = [f"v{i}" for i in range(count)]

    def kernel(p, depth, *vertices):
        return ExtrudedPolygon.sdf(p, depth, **dict(zip(names, vertices)))

    return kernel, ["depth", *names], [POINT, S, *([V2] * count)]


def _field_op(cls):
    return lambda d, value: cls.sdf(lambda _p: d, POINT, value)


def _boolean(cls):
    return lambda a, b, k: cls.sdf([lambda _p: a, lambda _p: b], POINT, k)


#: (kernel name, kind, cadjoint function, parameter names, example arguments)
KERNELS = [
    ("sphere", "primitive", Sphere.sdf, ["radius"], [POINT, S]),
    ("box", "primitive", Box.sdf, ["size"], [POINT, V3]),
    ("round_box", "primitive", RoundBox.sdf, ["size", "radius"], [POINT, V3, S]),
    ("cylinder", "primitive", Cylinder.sdf, ["radius", "height"], [POINT, S, S]),
    *[(f"extruded_polygon_{n}", "primitive", *_polygon(n)) for n in (3, 4, 5, 6)],
    ("translate", "transform", Translate._transform_point, ["offset"], [POINT, V3]),
    ("rotate", "transform", Rotate._transform_point, ["axis", "angle"], [POINT, V3, S]),
    ("offset", "field", _field_op(Offset), ["distance"], [S, S]),
    ("shell", "field", _field_op(Shell), ["thickness"], [S, S]),
    ("union", "boolean", _boolean(Union), ["smoothness"], [S, S, S]),
    ("difference", "boolean", _boolean(Difference), ["smoothness"], [S, S, S]),
    ("intersection", "boolean", _boolean(Intersection), ["smoothness"], [S, S, S]),
]

_DTYPES = {np.dtype(np.float32): "np.float32", np.dtype(np.bool_): "np.bool", np.dtype(np.int32): "np.int32"}
_BINARY = {
    "stablehlo.add": "np.add", "stablehlo.subtract": "np.subtract", "stablehlo.multiply": "np.multiply",
    "stablehlo.divide": "np.divide", "stablehlo.maximum": "np.maximum", "stablehlo.minimum": "np.minimum",
    "stablehlo.power": "np.power", "stablehlo.atan2": "np.atan2",
}
_UNARY = {
    "stablehlo.negate": "np.negative", "stablehlo.abs": "np.abs", "stablehlo.sqrt": "np.sqrt",
    "stablehlo.sine": "np.sin", "stablehlo.cosine": "np.cos", "stablehlo.exponential": "np.exp",
    "stablehlo.log": "np.log", "stablehlo.tanh": "np.tanh", "stablehlo.sign": "np.sign",
    "stablehlo.floor": "np.floor", "stablehlo.ceil": "np.ceil",
}
_COMPARE = {"EQ": "np.equal", "NE": "np.notEqual", "GE": "np.greaterEqual",
            "GT": "np.greater", "LE": "np.lessEqual", "LT": "np.less"}
_LOGICAL = {"stablehlo.and": "np.logicalAnd", "stablehlo.or": "np.logicalOr"}


def _ints(attribute) -> list[int]:
    """Integers of an `array<i64: 0, 1>` or `dense<[0, 1]> : tensor<2xi64>` attribute."""
    text = str(attribute)
    if text.startswith("array<"):
        body = text[len("array<"):-1]
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
                self.lines.append(f"  return {values[0] if len(values) == 1 else '[' + ', '.join(values) + ']'};")
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
                literal = repr(value.astype(float).tolist()).replace("inf", "Infinity").replace("nan", "NaN")
            return f"np.array({literal}, {{ dtype: {dtype} }})"
        if name in _BINARY:
            a, b = (self.use(v) for v in op.operands)
            return f"{_BINARY[name]}({a}, {b})"
        if name in _UNARY:
            return f"{_UNARY[name]}({self.use(op.operands[0])})"
        if name == "stablehlo.rsqrt":
            return f"np.reciprocal(np.sqrt({self.use(op.operands[0])}))"
        if name == "stablehlo.compare":
            direction = re.search(r"comparison_direction (\w+)", str(op.attributes["comparison_direction"])).group(1)
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
                raise NotImplementedError("strided slice")
            index = ", ".join(f"[{s}, {e}]" for s, e in zip(starts, limits))
            return f"{self.use(op.operands[0])}.slice({index})"
        if name == "stablehlo.concatenate":
            axis = _ints(op.attributes["dimension"])[0]
            parts = ", ".join(self.use(v) for v in op.operands)
            return f"np.concatenate([{parts}], {axis})"
        if name == "stablehlo.transpose":
            return f"np.transpose({self.use(op.operands[0])}, {_ints(op.attributes['permutation'])})"
        if name == "stablehlo.reduce":
            body = op.regions[0].blocks[0].operations[0].name
            reducer = {"stablehlo.add": "np.sum", "stablehlo.maximum": "np.max", "stablehlo.minimum": "np.min"}[body]
            axes = _ints(op.attributes["dimensions"])
            x = self.use(op.operands[0])
            init = self.use(op.operands[1])  # identity of the reducer; consumed, not used
            self.lines.append(f"  {init}.dispose();")
            return f"{reducer}({x}, {axes if len(axes) > 1 else axes[0]})"
        if name == "stablehlo.dot_general":
            dims = str(op.attributes["dot_dimension_numbers"])
            lhs_c = [int(v) for v in re.search(r"lhs_contracting_dimensions = \[([^\]]*)\]", dims).group(1).split(",") if v.strip()]
            rhs_c = [int(v) for v in re.search(r"rhs_contracting_dimensions = \[([^\]]*)\]", dims).group(1).split(",") if v.strip()]
            if "batching" in dims and re.search(r"batching_dimensions = \[\d", dims):
                raise NotImplementedError("batched dot_general")
            lhs_rank, rhs_rank = len(_shape(op.operands[0])), len(_shape(op.operands[1]))
            letters = iter("abcdefgh")
            lhs = [next(letters) for _ in range(lhs_rank)]
            rhs = [next(letters) for _ in range(rhs_rank)]
            for l, r in zip(lhs_c, rhs_c):
                rhs[r] = lhs[l]
            out = [c for i, c in enumerate(lhs) if i not in lhs_c] + [c for i, c in enumerate(rhs) if i not in rhs_c]
            a, b = (self.use(v) for v in op.operands)
            return f"np.einsum(\"{''.join(lhs)},{''.join(rhs)}->{''.join(out)}\", {a}, {b})"
        if name == "func.call":
            callee = str(op.attributes["callee"]).lstrip("@")
            return f"{self.ts_names[callee]}({', '.join(self.use(v) for v in op.operands)})"
        raise NotImplementedError(f"no jax-js translation for {name}")


def emit_kernel(kernel_name: str, fn, example_args) -> str:
    from jax._src.interpreters.mlir import make_ir_context
    from jaxlib.mlir import ir

    with scalar_lowering():
        text = export(jax.jit(fn))(*example_args).mlir_module()
    with make_ir_context():
        module = ir.Module.parse(_strip_locs(text))
        funcs = list(module.body.operations)
        ts_names = {f.name.value: (kernel_name if f.name.value == "main" else f"{kernel_name}__{f.name.value}") for f in funcs}
        # Callees first, so every function is defined before it is called.
        ordered = [f for f in funcs if f.name.value != "main"] + [f for f in funcs if f.name.value == "main"]
        return "\n\n".join(_FunctionEmitter(ts_names).emit(f, ts_names[f.name.value]) for f in ordered)


def main() -> None:
    parts = [
        "// GENERATED by emit_kernels.py from cadjoint's own SDF code via StableHLO.",
        "// Do not edit: change the Python and re-run the generator.",
        "/* eslint-disable */",
        'import { numpy as np } from "@jax-js/jax";',
        "",
        "// max/min follow jax-js's tie-gradient convention. Up to 0.1.26 that is",
        "// all to the first argument; JAX splits a tie 0.5 / 0.5. Values are",
        "// identical, gradients differ only at exact ties. ekzhang/jax-js#231",
        "// (merged after 0.1.26) makes jax-js split like JAX. A composite",
        "// min(a, b) + min(b, a) here would double the jit-fused expression per",
        "// boolean: ekzhang/jax-js#233.",
        "",
    ]
    registry = []
    for kernel_name, kind, fn, params, example in KERNELS:
        parts.append(emit_kernel(kernel_name, fn, example))
        parts.append("")
        registry.append(f'  {kernel_name}: {{ kind: "{kind}", params: {params!r}, fn: {kernel_name} }},')
    parts.append("export type KernelKind = \"primitive\" | \"transform\" | \"field\" | \"boolean\";")
    parts.append("export const kernels: Record<string, { kind: KernelKind; params: string[]; fn: (...args: np.Array[]) => np.Array }> = {")
    parts.extend(registry)
    parts.append("};")
    out = HERE / "src" / "kernels.gen.ts"
    out.write_text("\n".join(parts).replace("'", '"') + "\n")
    print(f"wrote {out} ({sum(1 for _ in out.read_text().splitlines())} lines, {len(KERNELS)} kernels)")


if __name__ == "__main__":
    main()
