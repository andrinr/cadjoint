# cadjoint's field in the browser on jax-js

Can a static page evaluate, differentiate, mesh and export a cadjoint scene
with no server, without writing cadjoint's maths a second time? Yes, for
every shipped scene that loads, on Wasm and WebGPU.

Measured 2026-10-18, Apple Silicon, jax-js 0.1.27: Node (Wasm) for the
headless check, the Claude desktop browser pane (Chromium) for the page.

## How it works

```
cadjoint (Python)  --python -m cadjoint.backends.jaxjs kernels-->  src/kernels.gen.ts
cadjoint scene     --python -m cadjoint.backends.jaxjs scene---->  NAME.scene.json (cadjoint-scene v1)
src/runtime.ts     wires the kernels from the scene graph (no formulas)
src/dc.ts          cadjoint's dual contouring, ported (extract_mesh, sharp=True)
src/chunk.ts       bounded-batch evaluation (jax-js's Wasm heap is 32-bit)
```

The generator lives in the package, `cadjoint/backends/jaxjs/`:

- `kernels.py` traces each node type's own `sdf` with JAX, exports it to
  StableHLO, and `_stablehlo.py` translates that op by op into jax-js.
  Primitives become one kernel. Operations are traced with *recording*
  children: the **query** kernel is the point a child is asked about
  (skipped when it is `p`), the **combine** kernel is what the node makes
  of the children's distances (skipped when it returns the child's value);
  n-ary nodes fold it left. Patterns get a per-instance query (one kernel
  for any count and skip mask) and a folded `min`. Profiles are emitted at
  4/8/16/32/64 vertices and padded with their last vertex, which the polygon
  distance treats exactly.
- `scene.py`: `scene_graph(sdf)` writes the JSON contract; `graph_sdf(graph)`
  evaluates a graph through the same `sdf`s in Python.
- `tests/backends/test_jaxjs.py`: one scene using all 23 node types; its
  graph evaluates bit-identically to the object, and the generated TS
  matches Python within 2e-5 fused and per-node (needs `npm install` here).

## Coverage

23 node types: sphere, box, round_box, cylinder, capsule, torus, plane,
extruded_polygon (with draft and twist), revolved_polygon, lofted_polygon,
translate, rotate, scale, twist, mirror, offset, shell, union, difference,
intersection, xor, linear_pattern, polar_pattern. That is every concrete
`cadjoint.sdf` node class. Not covered: profiles over 64 vertices, and
plain Python callables as children (both raise `UnsupportedNode`).

## Results: every shipped scene, headless (Wasm)

`npx tsx check.mts node clevis bracket cylinder_block duct_fairing duct_sink end_cap motor_shield`

Each scene's export lattice is the one a real export uses (`fit_bounds`, 96
cells on the longest side); node and clevis keep the spike's lattices.

| scene | nodes | value err (rel) | grad err (rel) | vertices / triangles vs `extract_mesh` | vertex positions | sample + mesh, fused |
| --- | --- | --- | --- | --- | --- | --- |
| node | 30 | 8.7e-8 | 2.5e-7 | 121,960 / 243,936, equal | ≤ 1.2e-4 cells | 0.6 + 1.5 s |
| clevis | 40 | 5.1e-9 | 3.9e-7 | 45,556 / 91,140, equal | 18 differ, ≤ 0.28 cells | 0.4 + 0.9 s |
| bracket | 17 | 1.1e-7 | 8.5e-8 | 20,492 / 40,980, equal | ≤ 2.1e-5 cells | 0.1 + 0.4 s |
| cylinder_block | 144 | 9.4e-8 | 4.0e-7 | 63,600 / 127,776, equal | ≤ 8.3e-5 cells | 5.2 + 18.8 s |
| duct_fairing | 2 | 0 | (no blend) | 38,108 / 76,212, equal | exact | 0.0 + 0.2 s |
| duct_sink | 9 | 2.7e-8 | 5.3e-7 | 30,826 / 61,648, equal | ≤ 2.8e-6 cells | 0.0 + 0.2 s |
| end_cap | 51 | 1.8e-7 | 8.0e-8 | 31,932 / 63,860, equal | ≤ 9.0e-5 cells | 1.4 + 3.3 s |
| motor_shield | 154 | 3.2e-7 | 4.1e-7 | 65,619 / 131,408, equal | 4 differ, ≤ 0.37 cells | 4.9 + 15.3 s |

Bad edges equal Python's everywhere (0, and 3 on motor_shield, which
Python's mesh has too). The gradient is `jax.grad` of the summed field at
2000 points w.r.t. one blend smoothness, through the cadjoint object's own
Parameter. `scenes/starter.py` does not load stand-alone (it needs a declared
study), so it is not exported.

The differing vertices are QEF solutions where the Hermite data differ by
rounding: the clevis's at creases (cadjoint's sharp booleans floor k at
1e-10, so which side's normal wins is float noise), motor_shield's at
points where the jax-js vertex is no farther from the zero set than
Python's (|f| 3.6e-4 vs 6.6e-2 cells for the worst). Not traced further.

Per-node composition gives the same values within 1.2e-6 relative and the
same gradients within 6.3e-7 on all eight; its meshes equal the fused ones
except on the clevis, whose comparison lattice puts 10,909 points exactly on
faces where the sign at zero is rounding.

## Browser

`npm run dev`, open http://localhost:5191/ (`?export=NAME[&webgpu][&pernode]`
for one scene; `?split=NAME`, `?kernels` for timings). `?export` runs values,
grad, the export lattice and dual contouring, and offers the STL.

- **WebGPU works** since jax-js 0.1.27 (ekzhang/jax-js#234, the `maxArgs`
  infinite loop). Every kernel compiles in 3-165 ms, `?export=end_cap&webgpu`
  (fused) and `?export=motor_shield&webgpu&pernode` (154 nodes) match
  Python: vertex, triangle and bad-edge counts equal, volume within 1e-7
  and 1e-6 relative, gradient within 2.4e-7 and 8.3e-8. motor_shield's
  export takes 4.1 s sampling + 12.0 s meshing there.
- **WebGPU bug, `scale` only**: `Scale.sdf`'s `jnp.allclose` becomes a
  boolean `all` over an `or` expression, and jax-js's WebGPU reduction emits
  `acc = acc && a || b` (it strips the item's parentheses), which WGSL
  rejects. Minimal repro: `jit((a, b, c) => np.all(np.logicalOr(np.less(a,
  b), np.logicalAnd(np.less(b, c), np.less(a, c)))))` on WebGPU. Fix: keep
  the parentheses in `src/backend/webgpu.ts`'s boolean min/max reduction
  (WebGL does). Wasm is unaffected.
- **Memory**: one fused call over 550k points of the 144-node cylinder block
  exceeds jax-js's 4 GiB Wasm heap, so evaluation runs in 16k-point chunks.

Timings (fused, compile + first call at 256 points; warm 1M points through
`sample`). Taken with the browser pane in the background, so treat them as
upper bounds; an earlier unchunked, foreground run of the clevis gave 153 ms
/ 30 ms on WebGPU.

| | clevis Wasm | clevis WebGPU | motor_shield Wasm | motor_shield WebGPU |
| --- | --- | --- | --- | --- |
| fused, first 256 | 153 ms | 308 ms | 1192 ms | 2423 ms |
| per-node, first 256 | 38 ms | 106 ms | 201 ms | 259 ms |
| fused, warm 1M | 308 ms | 376 ms | 5.1 s | 3.1 s |
| per-node, warm 1M | 493 ms | 567 ms | 7.6 s | 5.3 s |

Per-node compiles much less (one kernel per node type and role, shared);
fused runs faster once compiled. 16k-point chunks with a synchronous
readback each cost WebGPU most of its advantage; a larger chunk on WebGPU
(its memory is not the Wasm heap) is the obvious next step.

## Files

`check.mts` (headless checks), `eval.mts` (graph + points to values, for
pytest), `export_scene.py` (graph + Python references for a scene file or a
graph), `src/runtime.ts`, `src/dc.ts`, `src/chunk.ts`, `src/mesh.ts` (checks,
STL), `src/main.ts` (the page), `src/kernels.gen.ts` (generated),
`src/sdf.ts` and `make_reference*.py` (the first hand-written port, kept for
comparison).
