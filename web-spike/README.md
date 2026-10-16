# Spike: cadjoint's field in the browser on jax-js

Can a static page evaluate, differentiate, mesh and export a cadjoint scene
with no server, without writing cadjoint's maths a second time?

Measured 2026-10-16, Apple Silicon, jax-js 0.1.26: Node (Wasm) for the
headless check, and the Claude desktop browser pane (Chromium 152) for the
page.

## One source of truth for the maths

```
cadjoint (Python, jnp)  --emit_kernels.py-->  StableHLO  -->  src/kernels.gen.ts (jax-js)
cadjoint scene object   --export_scene.py-->  NAME.scene.json  (node type, params, children)
src/runtime.ts          composes the generated kernels from the scene graph
src/dc.ts               cadjoint's dual contouring, ported (extract_mesh, sharp=True)
```

- `emit_kernels.py` lowers each node type's per-point function from
  cadjoint's own code -- `Box.sdf`, `Cylinder.sdf`, `ExtrudedPolygon.sdf`,
  `Rotate._transform_point`, `Union.sdf`, ... -- to StableHLO and translates
  it op by op into jax-js (15 kernels, ~1400 lines, never edited by hand).
  It counts every SSA value's uses and emits jax-js's `.ref` ownership
  mechanically.
- `src/runtime.ts` is the only hand-written part of the field: four wiring
  patterns (primitive, transform, field op, boolean fold), ~60 lines,
  mirroring each node's `sdf(child_sdf, p, ...)`.
- `export_scene.py` writes the scene graph plus Python references: values at
  2000 points, `jax.grad` of their sum w.r.t. the first blend's smoothness
  through cadjoint's functional form, and cadjoint's `extract_mesh` on a
  lattice.
- `check.mts` runs all four checks headless: `npx tsx check.mts`.

## Results (generated path, Wasm)

| | space-frame node | clevis bracket |
| --- | --- | --- |
| field vs cadjoint, 2000 points | 3.8e-6 | 3.8e-6 |
| `grad` w.r.t. blend vs Python `jax.grad`, relative | 3.3e-7 | 3.9e-7 |
| vertices / triangles vs `extract_mesh` | 121,960 / 243,936, exact | 45,556 / 91,140, exact |
| bad edges (also after a position weld) | 0 | 0 |
| volume, browser vs Python | 25,128.0 vs 25,128.0 mm^3 | 106,573.5 vs 106,588.8 mm^3 |

The clevis mesh has the same topology as Python's; 1.7 % of its vertices sit
up to 0.22 mm apart (99 % within 0.05 mm). Traced cell by cell, none of it is
the generated formulas:

- cadjoint's `RoundBox` has a NaN gradient everywhere inside its inner box
  (`norm` of a zero vector), which leaks through every boolean around it.
  Python drops those normals and places the cutout-wall vertices by mass
  point; jax-js has finite gradients and puts them on the wall (x = -54.000
  vs Python's -53.984). Filed as andrinr/cadjoint#57.
- On creases (a pin bore meeting a lug face) cadjoint's sharp booleans are a
  smooth max with k floored at 1e-10, so which side's normal wins is decided
  by float noise: XLA and jax-js pick differently, both valid.
- At exact max/min ties jax-js gives the gradient to the first argument, JAX
  splits it 0.5 / 0.5 (ekzhang/jax-js#230). A composite workaround reuses its
  operands, and jax-js's jit fusion inlines that exponentially on chained
  booleans (clevis compile 163 ms -> 66 s, then OOM), so the fix belongs in
  jax-js's JVP rule.

## Performance

| | Wasm (Node) | Wasm (browser) | WebGPU (browser) |
| --- | --- | --- | --- |
| clevis field, first jitted call (compile) | 153 ms | 178 ms | see below |
| 1 M points, warm | 342 ms | | |
| node: sample 160^3 / mesh | 0.6 s / 1.1 s | 1.7 s / 1.8 s | 0.4 s / 0.6 s |
| clevis: sample 1.1 M / mesh | 0.4 s / 0.8 s | 2.2 s / 3.2 s | |

WebGPU compiles the node scene's fused kernel in ~0.1 s but the 40-node
clevis kernel takes far longer (still compiling after a minute in this
pane); Wasm is the dependable backend for export today, and per-node
kernels instead of one fused scene kernel are the obvious next step for
WebGPU.

## Files

`emit_kernels.py`, `export_scene.py`, `check.mts`, `src/kernels.gen.ts`
(generated), `src/runtime.ts`, `src/dc.ts`, `src/mesh.ts` (checks, STL),
`src/main.ts` (the page; `?quick`, `?split[=node][&webgpu]` time single
steps), `src/sdf.ts` (the earlier hand-written node port, kept for
comparison), `make_reference*.py` (the first, hand-port references).

Run: `npm install && npm run dev`, open http://localhost:5191/.
