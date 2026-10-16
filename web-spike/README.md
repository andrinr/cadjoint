# Spike: cadjoint's field in the browser on jax-js

Can a static page evaluate, differentiate, mesh and export a cadjoint scene
with no server? Measured on 2026-10-16 in Chromium (Claude desktop browser
pane), Apple Silicon, jax-js 0.1.26.

The scene is `skills/cadjoint/references/examples/space_frame_node.py` from
the text-to-cad skill: six cylinders, two rotations each, a smooth union
(k = 4), a 2 mm Shell, a sharp intersection with a ball. `src/sdf.ts` ports
those operations from the cadjoint source; `make_reference.py` evaluates
the same scene with Python cadjoint.

| check | Wasm | WebGPU |
| --- | --- | --- |
| field vs Python cadjoint, 2000 points | max error 5.8e-6 | 7.3e-6 |
| `grad` of sum(field) w.r.t. hub radius | -102.47504 | -102.47503 |
| Python `jax.grad`, float32 | -102.47495 | |
| 160^3 grid (4.1 M points), `jit`, warm | 102-214 ms | 47-169 ms |
| dual contouring (`src/dc.ts`) on that grid | 1.1-1.9 s | 1.2-2.0 s |

A float64 central difference gives -102.614; it differs from both analytic
gradients only where sample points sit on the field's kinks (abs, max,
where). Timings vary run to run; ranges are over three page loads.

## The mesh matches cadjoint's, number for number

`src/dc.ts` ports `extract_mesh(sdf, grid, sharp=True)` stage by stage:
`find_crossing_edges`, `edge_hermite_data` (16 bisections, a secant step,
one guarded Newton step, with gradients from jax-js `grad`),
`manifold_cell_incidence` (one vertex per inside-corner component per cell),
`sharp_qef_vertices` (rank-revealing QEF via the eigenvalues of N^T N, every
rank tried and clamped to the cell) and `dual_faces`. On the lattice the page
samples, `make_reference_mesh.py` runs Python cadjoint's `extract_mesh`:

| | browser (Wasm and WebGPU) | Python `extract_mesh` |
| --- | --- | --- |
| vertices | 121,960 | 121,960 |
| triangles | 243,936 | 243,936 |
| edges not shared by two triangles | 0 | 0 |
| ... after merging vertices by position, as an STL reader does | 0 (0 slivers) | |
| enclosed volume | 25,128.0 mm^3 | 25,127.99 mm^3 |

The page offers both meshes as binary STL downloads.

The cost: jax-js arrays are reference counted and every function consumes
its arguments, so a body that reuses an array must take `.ref`. Passing
points as three columns and keeping every transform constant a JS number
confines that to the coordinates; the six SDF operations are ~130 lines,
the mesher ~330.

Run: `npm install && npm run dev`, open http://localhost:5191/.
