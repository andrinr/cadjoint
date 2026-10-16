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
| 160^3 grid (4.1 M points), `jit`, warm | 102 ms | 65 ms |
| surface nets (CPU, TS) on that grid | 0.30 s | 0.35 s |
| enclosed volume | 25,080 mm^3 | 25,080 mm^3 |

cadjoint's own dual contouring at the same spacing encloses 25,128 mm^3
(0.19 % apart). A float64 central difference gives -102.614; it differs
from both analytic gradients only where sample points sit on the field's
kinks (abs, max, where).

Not yet good enough: the surface-nets mesh has 44 edges not shared by two
triangles. That is the stand-in mesher, not jax-js: naive surface nets
gives one vertex per cell, and a 2 mm wall at 0.53 mm cells puts two
sheets in some cells. cadjoint's dual contouring keeps one vertex per sheet
per cell and is watertight on the same scene.

The cost: jax-js arrays are reference counted and every function consumes
its arguments, so a body that reuses an array must take `.ref`. Passing
points as three columns and keeping every transform constant a JS number
confines that to the coordinates; the six operations here are ~130 lines.

Run: `npm install && npm run dev`, open http://localhost:5191/.
