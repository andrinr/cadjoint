# Where the playground's seconds actually go — a measured evaluation of every speed-up avenue

Status: measured profile + prototypes (2026-09-02). Nothing in `cadjoint/**` was
changed; every number below comes from a profile or a working prototype in the
scratch workspace listed in §11.

**Machine**: Apple Silicon arm64, macOS 25.4, CPython 3.14.5, jax 0.8.2 (CPU
backend), numpy 2.4.1, scipy 1.17.0, petsc4py 3.25.4, native Rust mesher built
(`native/target/release/libcadjoint_native_mesher.dylib`).

**Scenes**: two versions of `scenes/starter.py` are used, because it grew
mid-measurement and the growth is itself a finding.

| label | what it is | world-frame leaves |
|---|---|---:|
| `starter@d42d800` | the heat sink alone (`Union(sink, slug, bush_a, bush_b, k=0.03)`) | 4 |
| `starter@current` | plus board / die / screw heads / caps in an outer `Union(..., k=0.005)` | 10 |

**Cache discipline**: every "warm" number is with `CADJOINT_CACHE_DIR` pointed at a
populated directory (`cadjoint.cache`). "Cold" means an *empty* cache directory.
Run every benchmark twice and quote the second.

---

## 0. The one-sentence answer

Every worker mode is dominated by **JAX in eager (op-by-op) mode re-tracing and
re-dispatching the user's SDF on every request**, and by **XLA compiling
shape-specific programs that no two requests share**. There is no algorithm, no
library and no language in the hot path: the Rust core's entire discrete
pipeline is **2–8 ms**, the FEM linear solve is **47 ms**, TetGen is **11 ms**,
JSON serialisation is **1–5 ms**. Everything else is Python-side JAX overhead.

---

## 1. First question: why did `mode=mesh` explode when the starter gained context geometry?

### 1.1 End to end, fresh subprocess per request, warm compilation cache

| mode | `starter@d42d800` | `starter@current` |
|---|---:|---:|
| `compile` | **1.07 s** | **1.59–1.65 s** |
| `mesh` | **5.65 s** | **12.4–12.8 s** |
| `mesh_inspect` | 3.25 s | — |
| `simulate` (design already seen) | 4.90 s | — |
| `simulate` (**novel** design) | **31–77 s** | — |
| `optimize` (8 steps) | 229 s | — |
| process start + all imports | 0.39 s | 0.39 s |

Cold compilation cache, `mesh`, `starter@current`: **45–53 s** (isolated empty
cache; the 160–230 s figures seen earlier were the untrimmed scene).

### 1.2 Stage breakdown of `_mesh_edge_payload` (in-process, `starter@current`)

Instrumented copy of `cadjoint/viewer/_edge_overlay.py` with section timers.
"cold" = first call in a fresh process (shared warm disk cache); "warm" = second
call in the same process.

| section (source order) | cold | warm |
|---|---:|---:|
| `sample_grid` (65³ = 274 625 lattice points) | 0.465 s | 0.088 s |
| `find_crossing_edges` | 0.002 s | 0.001 s |
| `manifold_cell_incidence` | 0.001 s | 0.001 s |
| **`edge_hermite_data`** (16 bisections + 1 Newton, 1 492 edges) | **20.81 s** | **0.921 s** |
| `sharp_qef_vertices` | 0.920 s | 0.001 s |
| `dual_faces` + quad edge list | <0.001 s | <0.001 s |
| `classify_feature_cells` | 4.061 s | 0.001 s |
| subgradient verification (`jax.vmap(jax.grad(sdf))` probes) | 14.46 s | 0.178 s |
| **seam grouping + Newton projection** (`_project_to_seam` ×15) | **149.70 s** | **5.902 s** |
| `feature_cell_links` | 0.002 s | 0.002 s |
| seam tangents + `jax.grad` per operand pair | 40.09 s | 0.548 s |
| junction-shortcut prune (pure-Python loop) | 0.002 s | 0.002 s |
| chain building (pure-Python loop) | 0.001 s | 0.001 s |
| debris pruning (union-find, pure-Python loop) | 0.001 s | 0.001 s |
| `np.unique` on wire edges | 0.001 s | 0.001 s |
| `segments()` rounding to 3 dp (3 052 + 377 segments) | 0.005 s | 0.005 s |
| **total** | **230.5 s** | **7.65 s** |

Same table for `starter@d42d800` (4 leaves): total 4.64 s cold / 2.79 s warm,
with `edge_hermite_data` 1.035/0.704 s and seam projection 2.400/1.575 s.

**Read the last four rows.** Every pure-Python and NumPy loop in this 541-line
file, together, costs **~10 ms**. The 449-line `_mesh_edge_payload` is not slow
because of its Python; it is slow because it evaluates the SDF through eager JAX
in many separate programs.

### 1.3 The mechanism: cost scales with the number of *seam groups*, not with work

`_mesh_edge_payload` groups seam vertices by the set of CSG operands meeting
there and calls `_project_to_seam` once per group. Measured, `starter@current`:

| group | fields | points | cold | warm |
|---:|---:|---:|---:|---:|
| 0 | 2 | 12 | 0.917 s | 0.540 s |
| 5 | 2 | 14 | 0.668 s | 0.172 s |
| 10 | 2 | 36 | 1.052 s | 0.580 s |
| **14** | **2** | **1** | **0.868 s** | **0.483 s** |
| … 15 groups total | | 108 points | **8.59 s** | **5.59 s** |

**Projecting a single point costs 0.48 s.** The cost is entirely fixed per-call
overhead — building `jax.vmap(jax.value_and_grad(...))` evaluators and running
four Newton iterations op-by-op — and is independent of the point count. Going
from 4 leaves to 10 leaves took the group count from 3 to 15, and that is the
whole of the mesh-mode regression.

`starter@d42d800`: 3 groups → 2.40 s cold / 1.58 s warm.
`starter@current`: 15 groups → 8.59 s cold / 5.59 s warm.

### 1.4 Answering the four candidate causes named in the brief

| candidate | verdict |
|---|---|
| hermite sampling of the lattice | 0.088 s warm — **not it** (1 % of the request) |
| `jax.grad` of the SDF for normals | 0.178 s warm — **not it** |
| the edge extractor's Newton projections | **yes, but not because they iterate** — see §2 |
| XLA compile of the enlarged expression | **yes, for the cold number**: 45–53 s cold vs 12.4 s warm, and 149.7 s vs 5.9 s in-process for the seam block |

So: **cold cost is XLA compiling one program per seam group; warm cost is Python
tracing and eager dispatch, again once per seam group.** The persistent
compilation cache removes the first and cannot touch the second.

---

## 2. Second question: does the hard union (`smoothness=0`) really cost 3.7×?

**No.** The 32.7 / 13.0 / 8.8 s spread is a compilation-cache artifact. Measured
with a *separate, empty* cache directory per variant, `starter@current`,
in-process:

| outer `Union` smoothness | seam groups | cold (own empty cache) | warm (2nd call) | sharp segments |
|---:|---:|---:|---:|---:|
| 0.0 (hard `min`) | **15** | 53.1 s | **8.37 s** | 377 |
| 0.005 (current) | 11 | 45.3 s | 7.15 s | 382 |
| 0.01 | 11 | 42.6 s | 6.42 s | 373 |

The real effect of `k = 0` is **+17 % to +30 %**, and its mechanism is the one in
§1.3: the hard union shifts `owners = argmin |leaf|` on the dual vertices, which
produces 4 extra operand-set groups, each carrying the same ~0.4–0.5 s fixed
overhead. It is not a convergence problem.

### 2.1 The non-convergence hypothesis is disproven structurally

Every root-finding loop in this pipeline is **fixed-length**, not
convergence-driven:

- `_edge_overlay._project_to_seam` — `for _ in range(4)` (4 Newton steps, always).
- `meshing.edge_detection.edge_hermite_data` — `jax.lax.fori_loop(0, 16, halve, …)`
  plus `newton_steps=1`, all fixed.
- `fem.motion.project_points` — `for _ in range(steps)` with `steps=8`, fixed.

There is no iteration cap to hit and no early exit to add: a non-convergent
point costs exactly as much as a convergent one. Clamping or detecting
non-convergence would save **zero** time. What non-convergence *does* affect is
acceptance: `genuine = residual < 0.1 * max(grid.spacing)` drops a whole group
whose vertices did not land on the operand zero sets, and that check costs one
extra field evaluation per group. Sharp-segment counts across k (377 / 382 / 373)
show no quality cliff at `k = 0`.

**Recommendation on (1)**: do not add iteration control. The wasted time is
per-group *fixed overhead*; the fix is to batch the groups (§6.2), which removes
it whether or not anything converges.

**Recommendation on (2)**: do not enforce or warn about a minimum blend on
performance grounds — 30 % does not justify constraining the modelling language,
and `Union(a, b)` already defaults to `smoothness=0.1`. If a warning is wanted it
should be about *seam-group count*, which is the quantity that actually costs
(15 groups ≈ 6 s of the 12.4 s request), not about `k`.

### 2.2 What budget does `mesh` mode actually need?

| cache state | `starter@current` | `starter@d42d800` |
|---|---:|---:|
| warm | 12.4–12.8 s | 5.65 s |
| cold (empty cache) | 45–53 s | 12–17 s |

`COMPILE_TIMEOUT_SECONDS = 20` was indeed tripped by the warm path on a slightly
larger scene. The `MESH_TIMEOUT_SECONDS = 90` now in `_worker_client.py` is the
right call: ~7× headroom warm, ~1.8× cold. It is not generous — a 15-leaf scene on
a cold cache would exceed it. Two ways to stop chasing the timeout: warm the
compilation cache at server start by issuing one background `mesh` request for
the scene the editor opens with, or remove the per-group program explosion (§6.2),
which brings the request under 4 s warm and under 10 s cold.

---

## 3. `compile` mode

In-process, `starter@d42d800`, warm (total 0.338 s):

| stage | warm |
|---|---:|
| scene `exec` (the user's Python, constraints solved) | 0.069 s |
| **`compile_scene_to_wgsl`** (jax export → StableHLO → WGSL emit) | **0.235 s** |
| `build_construction_payload` | 0.028 s |
| `build_material_payload` | 0.003 s |
| study / mesh / optimization declaration entries | 0.003 s |
| `build_viewer_shader` + `build_path_tracer_shader` + relations | <0.001 s |
| `json.dumps` of the 803 kB response | 0.001 s |

`compile_scene_to_wgsl` has no hotspot — 255 ms spread over jax's exporter
(`_module_to_bytecode` 35 ms, `mlir_module` 19 ms, lowering 50 ms) and
`_wgsl_emitter._dispatch` (3 655 calls, 30 ms). It is cacheable by source hash
but not obviously optimisable in place.

### 3.1 The response payload is 37 % literal duplication

| field | size |
|---|---:|
| `path_shader` | 163.7 kB |
| `shader` | 157.1 kB |
| `preview_shader` | 157.1 kB — **byte-identical to `shader`** |
| `sdf` | 143.0 kB |
| `scene_wgsl` | 143.0 kB — **byte-identical to `sdf`** |
| `construction` | 35.0 kB |
| everything else | 2 kB |
| **total** | **803.1 kB** (1 478 kB on `starter@current`) |

`scene_wgsl` also appears verbatim *inside* both `preview_shader` and
`path_shader`. Dropping the two duplicate keys and substituting the scene body
out of the two shaders: **803 kB → 217 kB**; gzipped, **151 kB → 39 kB**. The
server (`_http.py`) sends no `Content-Encoding`.

On loopback this is worth **~1–5 ms**, so it is a tidiness fix, not a speed fix —
but see §8, because the same observation has a much larger consequence.

---

## 4. `simulate` / `mesh_inspect` — and the finding that matters most here

In-process, `starter@d42d800`, design already compiled (total 1.94 s):

| stage | warm |
|---|---:|
| **meshing** (`study._solve_mesh` → `SimMesh.build`) | **1.757 s** |
|  ├ DC `extract_mesh` | 0.751 s |
|  │   └ of which `edge_hermite_data` | 0.680 s |
|  │   └ of which `sample_grid` | 0.067 s |
|  ├ **`project_points`** (8 Newton steps, eager JAX) | **1.040 s** |
|  ├ **TetGen** `surface_to_tet_mesh` | **0.011 s** |
|  └ `tet10_from_tet4` | 0.004 s |
| BC resolve check | 0.001 s |
| `tet_thermal_solve` (jax-fem assembly + solve) | 0.110 s |
|  └ of which the **linear solve** (PETSc LU + residual check) | **0.046 s** |
| response payload build (`_study_payload`, render surface, edges) | 0.006 s |
| `json.dumps` (86 kB) | 0.001 s |

Mesh: 5 726 nodes, 2 957 TET10 elements.

### 4.1 A novel design costs 30 s of XLA compile, every time

Every design edit changes the DC surface, so TetGen returns a **different node
and element count**, so every JAX program downstream gets **new shapes** and is
compiled from scratch. Measured on `starter@d42d800`, varying `fin_depth`:

| design | nodes | total | meshing | `tet_thermal_solve` |
|---|---:|---:|---:|---:|
| `fin_depth=1.31` (first time) | 5 801 | **33.8 s** | 3.2 s | **30.1 s** |
| `fin_depth=1.42` (first time) | 6 721 | **77.2 s** | **49.1 s** | 28.0 s |
| `fin_depth=1.53` (first time) | 6 709 | **31.2 s** | 2.2 s | 28.9 s |
| the *same three* designs, repeated | " | **3.3–4.7 s** | 2.5–3.1 s | **0.7–1.0 s** |

The linear solve inside those 28–30 s is **47 ms**. The rest is XLA compiling the
jax-fem assembly kernels for one more node count that will never recur.

This is the same disease as §1.3 in a different organ, and it is the largest
single number in this document: **a user simulating a design they just edited
waits ~30 s for a compile they can never reuse.**

---

## 5. `optimize`

`starter@d42d800`, `cool-sink`, 8 steps, `remesh_every=6`:

| | eager (today) | prototype: `jax.jit` on the frozen objective |
|---|---:|---:|
| first evaluation | 23.2 s | 48.3 s (trace + lower; **not** helped by the disk cache) |
| steps 2–6, each | **12.7 s** | **0.55 s** |
| refreeze at step 6 | 94.2 s | 48.5 s |
| final re-mesh + solve + patch | ~38 s | ~19 s |
| **8-step total** | **229.0 s** | **119.5 s** |

`cadjoint/optimize.py:896` builds `value_and_grad = jax.value_and_grad(frozen[1])`
and calls it once per step **without `jax.jit`**, even though the topology is
already frozen and the function is stable for `remesh_every` steps. Wrapping it
in `jax.jit` is a one-line change; objective values matched the eager run to 13
significant digits over the first six steps (1.6130011303450 vs 1.6130011303450);
steps 7–8 diverge at 1e-4 because the *refreeze* re-runs TetGen, not because of
the jit.

The 48 s first-evaluation cost is **tracing and lowering**, which the persistent
compilation cache does not touch (re-running with a fully warm cache reproduced
48.3 s exactly). That is the ceiling on this lever without a warm worker.

---

## 6. The avenues, each with its measured verdict

### 6.1 Warm worker / process reuse — **worth doing, and it is the enabler for everything else**

Prototype: `warm_worker.py`, a stdin/stdout NDJSON loop importing the real
`_compile_worker` and dispatching the same five modes, state retained.

| | fresh subprocess (today) | warm worker |
|---|---:|---:|
| worker boot | 0.39 s per request | 0.28–0.31 s **once** |
| `compile`, first request | 1.07 s | 0.70 s |
| `compile`, subsequent edits | 1.07 s | **0.336 s** |
| `mesh`, first request | 5.65 s | 4.94 s |
| `mesh`, subsequent edits | 5.65 s | **3.02–3.15 s** (one 5.99 s outlier when the topology changed) |
| `mesh_inspect`, subsequent | 3.25 s | 2.29–2.34 s |

Measured saving: **0.73 s on `compile` (3.2×), 2.6 s on `mesh` (1.9×)**. It is
made of 0.39 s of process start + imports and ~0.3–2.2 s of first-call-in-process
JAX warm-up (jaxpr caches, lowering caches, the pjit executable cache — none of
which the disk cache substitutes for).

The larger reason to want it: **every jit win in §6.5 requires a process that
outlives one request.** A jitted program traced in a disposable worker is thrown
away before its second call.

**What it costs in isolation.** Today's contract is strong: a fresh process per
request means a runaway `exec` cannot outlive its timeout, cannot leak globals
into the next request, and cannot corrupt shared state. A warm worker must
reconstruct that:

- Keep the supervisor and the per-mode timeout in `_worker_client.py`; on
  timeout, **kill and replace** the worker rather than waiting on it.
- Recycle a worker after any exception and after every *K* requests, so leaked
  module-level state (a user program that monkeypatches `numpy`, an `atexit`
  hook, a thread it started) has a bounded lifetime.
- Run a small pool so one slow `mesh` does not block a `compile`.
- Accept that `exec`'d user code shares an address space with the accumulated
  JAX caches. On a **localhost single-user playground** that is a fair trade; it
  would not be on a shared server.

A safe intermediate that keeps full isolation: keep the disposable worker for
`compile` (already 1.07 s) and make only `mesh`/`simulate`/`optimize` warm.

### 6.2 Batch the seam projections — **the single best fix for `mesh` mode**

Prototype `proto_seam_batch.py`: one program that evaluates value + gradient of
**every** world-frame leaf at **every** seam point, four Newton iterations, group
membership carried as a `(P, 3)` index array with a one-hot gather, replacing the
15 separate `_project_to_seam` calls.

| | current (15 calls, 108 points) | batched (1 call) |
|---|---:|---:|
| eager, steady | **5.59 s** | **0.685 s** |
| jitted, steady | — | **0.0003 s** |
| jitted, first call | — | 1.88 s |

**8.2× on the seam block eagerly**, with no warm worker and no jit required.
Projected effect on the whole warm `mesh` path, `starter@current`:
7.65 s − 5.59 s + 0.69 s ≈ **2.8 s** (arithmetic from measured parts, not an
end-to-end measurement). The same batching applies to the seam-tangent gradients
(0.548 s warm, two `jax.grad` evaluations per group) for another ~0.5 s.

Risk: the masked Newton system must reproduce today's per-group numerics,
including the `transversal` eigenvalue test and the `genuine` residual
acceptance. Both are per-point quantities and survive masking, but this needs the
existing edge-view regression images (`research/edge-view/*.png`) to confirm.

### 6.3 Restrict feature-edge extraction to the design geometry — **cheap, do it**

`_world_frame_leaves` descends every `BooleanOp`, so the board, die, screw heads
and capacitors each become a seam-projection operand. Restricting the *sharp*
layer to a designated design subtree (`thermal_body`) takes the group count from
15 to 3 — measured on the two scenes, that is **5.59 s → 1.58 s** of seam work.
The *wire* layer comes from `dual_faces` over the whole scene (<1 ms) and is
unaffected, so context geometry keeps its mesh wireframe and loses only its
feature curves — arguably the desired behaviour for geometry the physics never
sees.

### 6.4 Rust / porting hot loops — **not worth doing; there is nothing left**

`cadjoint/meshing/native.py` already binds a rayon-parallel cdylib *(Retired 2026-09-02: the Rust core measured 5 ms faster over a 5,650 ms request and was removed; see `research/native-mesher.md`.)*
(`native/src/{lib,core}.rs`, 1 082 lines, ctypes ABI) for crossing detection,
manifold incidence, QEF placement and dual faces. Measured on the 65³ lattice of
`starter@d42d800` (1 492 edges, 1 494 cells):

| stage | NumPy reference | Rust |
|---|---:|---:|
| `find_crossing_edges` | 1.60 ms | 0.93 ms |
| `manifold_cell_incidence` | 0.71 ms | 0.44 ms |
| `sharp_qef_vertices` | 3.56 ms | 0.35 ms |
| `dual_faces` | 0.58 ms | 0.19 ms |
| `classify_feature_cells` | 1.11 ms | (Python only) |
| `feature_cell_links` | 0.26 ms | (Python only) |
| **whole discrete pipeline** | **7.8 ms** | **~3 ms** |

The Rust core saves **~5 ms** of a 5 650 ms request. The remaining pure-Python
loops in `_edge_overlay.py` — the junction-shortcut prune with its
`adjacency` dict, the greedy chain builder, the union-find debris pruner — measure
**1–2 ms each** (§1.2). A `tottime` cProfile of the warm `mesh` path shows no
cadjoint function in the top 30; the list is `jax/_src/util.py:wrapper`
(1 017 547 calls), `pjit._pjit_call_impl` (33 524), `core.get_aval` (668 085).

**Verdict: do not port anything else to Rust.** The existing port was correct and
is finished. A new port would have to be of the *SDF evaluator itself*, which
would mean giving up JAX autodiff — see §6.6.

### 6.5 `jax.jit` — **the largest lever in the codebase, and it is nearly free to apply**

The pipeline is written to be traceable but is called eagerly. Measured, all on
`starter@d42d800` unless noted:

| block | eager, steady | jitted, steady | jitted, first call |
|---|---:|---:|---:|
| `edge_hermite_data` (1 492 edges) | 0.69 s | **0.003 s** | 0.94 s (warm disk cache) / 2.10 s (cold) |
| `_project_to_seam` (2 fields, 300 pts) | 0.53 s | **0.0004 s** | 1.37 s |
| batched all-leaf seam projection (§6.2) | 0.685 s | **0.0003 s** | 1.88 s |
| `sample_grid`, parameters traced | 0.071 s | **0.001 s** | 0.56 s |
| 3 000 gradient probes, parameters traced | — | **0.0006 s** | 0.53 s |
| optimize frozen objective, per step | 12.7 s | **0.55 s** | 48 s |

Two conditions turn these into real wins:

1. **The process must survive** (§6.1). In a disposable worker the first call is
   all you get, and jitting is roughly break-even (0.94 s jitted-first vs 1.04 s
   eager for `edge_hermite_data`).
2. **The traced function must be reused across edits.** Today the scene is
   re-`exec`'d per request, producing a fresh closure with the design constants
   baked in, so the jit cache misses. The fix already exists in the codebase:
   `extract_parameters` / `functionalize` turn design values into *arguments*.
   Measured with the topology frozen and parameters traced, `edge_hermite_data`
   re-run at four different designs: **2.17 s, then 0.003 s, 0.003 s, 0.003 s**.

The catch is shape stability. `find_crossing_edges` returns a variable-length
edge set, so a topology change recompiles. Padding the crossing-edge and
seam-point arrays to bucketed capacities keeps shapes constant across ordinary
edits — the standard JAX idiom, and the same fix §6.7 needs for FEM.

### 6.6 WebAssembly / moving work into the browser — **feasible for the mesh overlay only, and it is the biggest project here**

What could move, and what forbids it:

| work | needs `jax.grad`? | portable? |
|---|---|---|
| viewer mesh overlay (`_mesh_edge_payload`) — output is **display line segments** | **no** | **yes** |
| DC discrete stages (crossings, incidence, QEF, faces) | no | yes — already Rust, 3 ms, `wasm32` is a target flip |
| feature classification, link filtering, chain building | no | yes — 10 ms of Python/NumPy |
| SDF sampling + Hermite refinement + seam Newton | no *for the overlay* | yes, **on the GPU** — see below |
| FEM sim mesh (`recompute_tet_points` under `jax.grad`) | **yes** | no — stays in Python |
| optimize chain, FEM adjoint, constraint projection | **yes** | no — stays in Python |

The decisive fact is that **the SDF is already compiled to WGSL and already
evaluated on the GPU far harder than meshing needs**. `_webgpu.py` marches 96
steps per primary ray; at 1200×800 that is ~92 M SDF evaluations *per frame*, at
interactive rates. The mesh overlay needs 274 625 lattice evaluations plus
~30 000 refinement evaluations — **≈ 0.3 % of one rendered frame**. A WebGPU
compute pass filling the value lattice and the Hermite data with the existing
`scene_wgsl`, feeding a `wasm32` build of `native/` for the discrete stages, would
make the mesh overlay a **zero-round-trip, sub-frame operation** and would delete
`mode=mesh` from the server entirely.

Cost: a second implementation of the feature-extraction semantics (the 449-line
`_mesh_edge_payload` is intricate — seam identity grouping, tangent estimation,
junction-shortcut rules, chain degree limits, debris pruning) with no shared
tests, plus gradient-free Newton projection on the GPU. That is a large project
whose *speed* payoff §6.2 + §6.1 + §6.5 already deliver at a fraction of the cost.
Its unique payoff is **latency**: nothing else gets the overlay to interactive.

**Verdict: not now. Revisit if the mesh overlay must follow a dragged sketch
point in real time.** If it is done, do §6.2 first anyway — the batched form is
what a GPU implementation would have to be.

### 6.7 Libraries: is jax-fem / PETSc the right solver? — **the solver is irrelevant; do not touch it**

Same assembled system (`n = 5 726`, `nnz = 133 352`, TET10 thermal), same RHS:

| solver | time | relative residual |
|---|---:|---:|
| **PETSc LU + residual verify (current, `_tet_direct_linear_solver`)** | **47.1 ms** | — |
| `scipy.sparse.linalg.splu` (SuperLU, COLAMD) | **32.4 ms** | 3.3e-14 |
| `scipy.sparse.linalg.spsolve` (SuperLU) | 32.1 ms | 3.3e-14 |
| CG + Jacobi, `rtol=1e-10` | 37.4 ms | 8.3e-11 |
| CHOLMOD (`scikit-sparse`) | not installed | — |
| pypardiso | not installed | — |

Best case saves **15 ms of a 4 900 ms request — 0.3 %.** The layered PETSc→SuperLU
fallback exists for a documented robustness reason (`research/tet-vs-hex.md`:
sliver tets defeat every single solver somewhere); trading that for 15 ms would be
a bad deal. The CalculiX backend is not a forward-solve speed play either.

**The FEM cost is not in the solve. It is 1.8 s of eager JAX meshing and up to
30 s of XLA compile per novel mesh shape (§4.1).**

### 6.8 Algorithms: resolution, redundancy, caching, payload

**Lattice resolution and adaptivity — not a speed lever.** The overlay grid is
64³ over a 6×6×6 box (`_MESH_EDGE_RESOLUTION`, `_MESH_EDGE_SIZE`) while the part
spans ~1.8 × 1.2 × 1.0, so only 1 492 of ~786 000 lattice edges cross the surface
(**0.19 % occupancy**). But `sample_grid` is 0.088 s of a 12.4 s request (**0.7 %**),
and every expensive stage downstream scales with the 1 492 crossings, not with
the lattice. Narrowing the bounds to the scene AABB buys **resolution at equal
cost** — a quality win worth having — but saves at most 0.07 s.

**Redundant recomputation across modes.** `mesh` does *not* redo `compile`'s WGSL
work (it only re-`exec`s the scene: 0.065 s warm). But `mesh_inspect` and
`simulate` each rebuild the *same* `SimMesh` from scratch — 1.8–3.0 s each — and
`SimMesh._cache` cannot help, because every request `exec`s the program afresh
and gets a new instance. A **cross-request artefact cache keyed by
(source hash, mesh name)** in a warm worker makes the second of that pair
~1.5 s instead of ~3.9 s. Worth doing once §6.1 exists; worthless without it.

**Payload format.** `mesh` returns 160 kB of JSON (3 052 wire + 377 sharp
segments, floats rounded to 3 dp); `json.dumps` is 2 ms and `segments()` rounding
is 5 ms. `compile` returns 803 kB (1 478 kB on `starter@current`), 37 % of it
literal duplication (§3.1), `json.dumps` 1 ms. Binary vertex/index buffers would
save single-digit milliseconds on loopback. **Not a speed lever** — but the
duplication is worth removing on principle, and gzip would take `compile` from
803 kB to 39 kB for a few ms of CPU.

---

## 7. Ranked recommendations

| # | change | measured effect | effort | risk |
|---|---|---|---|---|
| **1** | **`jax.jit` the frozen study objective** in `optimize.py:896`/`:900` | 8 steps **229 s → 119.5 s**; marginal step **12.7 s → 0.55 s** (23×) | **one line** | **low** — objective identical to 13 digits; first eval and each refreeze pay 48 s of trace+lower |
| **2** | **Shape-stable FEM**: reuse the frozen tet topology across parameter-only edits (`motion.recompute_tet_points` already exists), or pad node/element counts to buckets | novel-design `simulate` **31–77 s → 3.3–4.7 s** | medium | medium — must not perturb BC resolution or quality reporting |
| **3** | **Batch the seam projections** into one all-leaf program (§6.2) | seam block **5.59 s → 0.685 s** (8.2×); warm `mesh` ≈ **7.65 s → 2.8 s** | medium (one function) | medium — re-verify against `research/edge-view/*.png` |
| **4** | **Restrict the sharp layer to the design subtree** (§6.3) | seam block **5.59 s → 1.58 s**; wire layer unaffected | low | low — behavioural choice, arguably a fix |
| **5** | **Warm worker pool** with kill-on-timeout and recycle-on-error (§6.1) | `compile` **1.07 → 0.34 s**, `mesh` **5.65 → 3.0 s**; unlocks #6 | medium | **medium-high** — trades today's per-request isolation; see the mitigations in §6.1 |
| **6** | **Parameter-keyed JIT with frozen topology** for the mesh path, on top of #5 (§6.5) | per-edit JAX work **~2.8 s → ~0.01 s**; whole warm `mesh` request ≈ **0.15 s** (projected from measured parts) | high | medium — needs `functionalize` plumbing + padded shapes |
| **7** | **Warm the compilation cache at server start** (one background `mesh` for the opened scene) | removes the 45–53 s cold cliff from the user's first overlay | low | low |
| **8** | **Cross-request artefact cache** keyed by source hash (built `SimMesh`, DC Hermite data) — requires #5 | second of `mesh_inspect`+`simulate` **3.9 s → ~1.5 s** | low-medium | low |
| **9** | **Lift free/named parameters to a WGSL uniform** (§8) | a slider drag becomes a uniform write instead of a 1.6 s round trip + browser shader recompile | high (WGSL backend + frontend) | medium — biggest *perceived* win in the app |
| **10** | **De-duplicate the `compile` payload + gzip** (§3.1) | 803 kB → 217 kB → 39 kB; **~1–5 ms** wall clock | low (frontend-coupled) | low |
| **11** | Narrow the DC lattice to the scene AABB | ≤ 0.07 s; real gain is **resolution at equal cost** | low | low |
| — | **Port more of the mesher to Rust** | whole discrete pipeline is **3–8 ms**; nothing to win | — | **not worth doing** |
| — | **Swap the sparse solver** (SuperLU / CHOLMOD / pypardiso / CalculiX) | 47 ms → 32 ms best case = **0.3 %** of a `simulate` | — | **not worth doing** |
| — | **Binary instead of JSON payloads** | single-digit ms on loopback | — | **not worth doing for speed** |
| — | **Adaptive DC / Lipschitz pruning for speed** | `sample_grid` is 0.7 % of the request | — | **not worth doing for speed** |
| — | **WASM/WebGPU client-side mesh overlay** (§6.6) | only path to *interactive* overlay; #3+#5+#6 give the same seconds far cheaper | very high | **defer** |

Doing **#1 + #3 + #4 + #7** — no architectural change, no isolation trade, a
handful of days — takes warm `mesh` from 12.4 s to roughly 4 s, the cold cliff
from 50 s to a background warm-up, and 8-step `optimize` from 229 s to 120 s.
**#2** is the one that removes the worst single wait in the app (30 s of
throwaway XLA compile per novel simulated design). **#5 + #6** are what a
genuinely interactive playground eventually needs.

---

## 8. One structural observation worth its own heading

Design parameters are **baked as float literals into the generated WGSL**. Editing
`fin_depth` from 1.200 to 1.250 changes **3 lines out of 139 297 bytes**:

```
-    let _v26: f32 = 1.200000;
+    let _v26: f32 = 1.250000;
```

and the shader carries **zero** `@group` / `var<uniform>` declarations. So every
slider nudge pays: 1.6 s of server round trip (0.235 s of WGSL codegen inside it),
1.5 MB of transfer, and a full browser shader recompilation — to change three
constants.

Lifting the free and named parameters into a uniform buffer would make a
parameter edit a **uniform write and a redraw**, with no server involvement at
all, and would simultaneously make the WGSL cacheable by *structure* rather than
by *values*. It is the highest-value interactivity change in the app, and it is
independent of everything else in this document.

---

## 9. Corrections to the assumptions this study started from

- *"`simulate` ≈ 9.2 s, unchanged by the compilation cache."* — Measured 4.90 s
  warm vs 10.42 s cold: the cache **does** help `simulate`, because the DC meshing
  in front of the solve is XLA work. What it cannot help is a **novel** mesh
  shape (§4.1), which costs 31–77 s.
- *"jax-fem's solve runs in PETSc outside XLA, so nothing to cache."* — True of
  the 47 ms linear solve, and irrelevant: the assembly around it is XLA and is
  ~30 s of compile on a new shape.
- *"`_mesh_edge_payload` is 449 lines and worth its own line."* — It is, but not
  for its Python: every loop in it totals ~10 ms. It is worth its own line
  because it launches 15 separate JAX programs.
- *"Imports ≈ 0.35 s, NOT the problem."* — Confirmed at 0.39 s; it is 36 % of a
  warm `compile` request but 3 % of a `mesh` request.

## 10. Loose ends worth a look

- `_tet_direct_linear_solver` builds its PETSc vectors with
  `vec.setValues(range(len(rhs)), x)` — a Python `range` over every DOF, three
  times per residual check. Immaterial at 5 726 DOF; it will not stay immaterial.
- `sdf_to_tet_mesh` failed with *"TetGen rejected the surface"* on 1 of 4 randomly
  perturbed `fin_depth` values. That is a robustness issue, not a performance one,
  but it will surface as a mysterious `simulate` failure during optimization.
- The `mesh` warm-worker run showed one 5.99 s request among 3.0 s neighbours —
  a topology change re-tracing. Padded shapes (§6.5) would remove that jitter.

## 11. Reproducing this

Scripts live in the scratch workspace
the session scratchpad (not kept)
(ephemeral). `bench_all.py` dispatches all of them; each is standalone and takes
the scene through `BENCH_SCENE`.

| script | produces |
|---|---|
| `bench_e2e.py <modes…>` | §1.1 — subprocess wall clock per mode (`REPS`, `STEPS` env) |
| `run_instr.py` + `edge_overlay_instrumented.py` | §1.2 — `_mesh_edge_payload` section timings |
| `prof_seam_groups.py` | §1.3, §2 — per-group `_project_to_seam` cost, smoothness sweep |
| `prof_compile.py`, `cprof_wgsl.py`, `payload_dup.py` | §3 — `compile` stages and payload |
| `prof_sim.py`, `prof_simmesh.py`, `prof_sim_novel.py` | §4, §4.1 — `simulate` stages, novel-design compile cost |
| `prof_opt.py`, `proto_opt_jit.py [eager\|jit]` | §5 — `optimize`, eager vs jitted objective |
| `warm_worker.py` + `bench_warm.py <modes…>` | §6.1 — persistent-worker prototype |
| `proto_seam_batch.py` | §6.2 — batched all-leaf seam projection |
| `proto_rust_vs_py.py` | §6.4 — Rust vs NumPy discrete stages |
| `proto_jit_hermite.py`, `proto_seam_jit.py`, `proto_param_jit.py`, `proto_frozen_mesh.py` | §6.5 — jit prototypes |
| `proto_solver.py` | §6.7 — sparse solver shootout on the real assembled system |
| `starter_baseline.py`, `starter_current.py`, `starter_k{0.0,0.005,0.01}.py` | the pinned scenes |

Protocol for any number here:

```sh
export CADJOINT_CACHE_DIR=/tmp/cadjoint-jax-cache      # or a fresh dir for a cold number
.venv/bin/python bench_e2e.py mesh    # run once to populate the cache
.venv/bin/python bench_e2e.py mesh    # quote this one
```

---

# 12. The compiled program is enormous — where the size came from, and what structured lowering recovered

Status: **implemented and measured** (2026-09-02). Unlike §1–§11 above, this
section changes `cadjoint/**`: `cadjoint/functionalize.py`,
`cadjoint/sdf/primitives/polygon.py`, `cadjoint/sdf/primitives/loft.py`,
`cadjoint/sdf/operations.py`, a new `cadjoint/sdf/_lowering.py`, and the WGSL
backend.

Same machine as §0. Every number below uses an isolated `CADJOINT_CACHE_DIR`;
"cold" is an empty one, "warm" is the second run against a populated one. The
**flat** column is the pre-change lowering, reproduced in-process by restoring
the two behaviours that were removed (no node outlining, scalar/unrolled
emission) — it reproduces the pre-change StableHLO byte for byte, which is how
it was validated.

## 12.1 The question

> *"I think part of the performance problem is that the compiled code is
> insanely big, can this somewhat be reduced? do we need another IR between
> compiling to StableHLO?"*

The compiled code **was** insanely big, and the diagnosis was right about the
mechanism: cadjoint already has an IR — the SDF object graph — and the trace
was **flattening** it. But the fix is not a second IR. It is to stop discarding
the structure the first one already carries.

## 12.2 Where the size came from

Four separate flattenings, each measurable on its own:

| # | What flattened | Why the program grew |
|---|---|---|
| 1 | **Profile vertices** | `_polygon_distance` looped in Python over N vertices, emitting ~20 operations per vertex. The starter's fin comb is 12 vertices; `research/complex-scene.md` measured mesh cost scaling with the *total* profile vertex count (168 verts → 112 s, 98 → 54 s). |
| 2 | **Pattern instances** | `LinearPattern`/`PolarPattern` called `child_sdf` once per instance in a Python loop. `scenes/end_cap.py` has 8 ribs + 4 bolt holes + 3 port screws + 4 bolt heads + 2 pad taps = **21 copies** of geometry that is written once. |
| 3 | **Shared subtrees** | `build_function` recursed per *occurrence*, so a node reachable from two parents was traced twice. The end cap's `dowel` is both a body and the child of a `Mirror`. |
| 4 | **Parameter values** | `functionalize(sdf)(free, fixed)` closes over the values, so `jax.jit` folds every one of them in as a literal. Two designs differing in one slider lower to two different modules, compile separately, and miss each other in the persistent cache. |

## 12.3 What was changed

No new IR. Four changes to how the existing graph lowers itself.

1. **Vectorised profile distance.** The vertex loop is stacked into one
   `(N, 2)` array; the nearest-point search becomes one `min` and the even-odd
   crossing test becomes a parity count. Both reductions are exact, so the two
   forms agree **bit for bit** (§12.6). `LoftedPolygon` shares the same kernel.
2. **Vectorised patterns.** `LinearPattern` maps its child over an array of
   offsets; `PolarPattern` maps copies 1..N-1 over an array of angles and keeps
   copy 0 as its own unrotated evaluation, because copy 0 is what the child's
   face references are declared against.
3. **Sharing and outlining.** Nodes are built once per *object* (the DFS counter
   still advances per occurrence, so `extract_parameters`' path keys are
   unchanged). A node that is evaluated more than once — a pattern's child, a
   subtree with in-degree > 1 — is wrapped in `jax.jit`, which StableHLO keeps
   as a `func.func` plus one `func.call` per use, and JAX prunes the parameter
   entries the callee does not read.
4. **Parameters as arguments.** `functionalize_parametric` /
   `functionalize_scene_parametric` hand the dicts to the jitted function.
   `count` is the one exception: it decides how much program is emitted, so
   patterns declare `static_params = ("count",)` and it stays concrete.

**The shader keeps the flat form**, under a `scalar_lowering()` context the WGSL
backend holds while it traces: WGSL has no type wider than a `mat4`, so an
`(N, 2)` vertex array or a batched instance axis is untranslatable there.
Outlining is *not* mode-dependent, and the emitter already maps one `func.func`
to one WGSL function — which is where the shader's own saving comes from.

## 12.4 StableHLO: before and after

`jax.jit(sdf).lower(p).as_text()` on the scene root, plus the same under
`vmap` over 4 096 points and under `jax.grad` of a sum-of-squares over the free
parameters — the three shapes the viewer, the mesher and the optimizer
actually compile.

### `scenes/starter.py` (35 free / 147 fixed parameters)

| program | metric | flat | structured | change |
|---|---|---:|---:|---:|
| point query | HLO bytes | 114 220 | **58 662** | −49 % |
| | HLO ops | 1 588 | **832** | −48 % |
| | XLA compile, cold | 0.113 s | **0.049 s** | −57 % |
| | first call, warm | 0.0002 s | 0.0002 s | — |
| `vmap`, 4 096 pts | HLO bytes | 160 397 | **77 020** | −52 % |
| | HLO ops | 1 978 | **992** | −50 % |
| | XLA compile, cold | 0.207 s | **0.078 s** | −62 % |
| `grad` over params | HLO bytes | 499 784 | **313 578** | −37 % |
| | HLO ops | 4 844 | **2 620** | −46 % |
| | XLA compile, cold | 0.799 s | **0.271 s** | −66 % |
| WGSL | bytes | 264 694 | 264 694 | — |
| | `let` statements | 6 793 | 6 793 | — |

The starter has no patterns and no shared subtree, so its shader is untouched;
its HLO halves purely from the vectorised comb profile.

### `scenes/end_cap.py` (11 free / 421 fixed parameters, 21 pattern instances)

| program | metric | flat | structured | change |
|---|---|---:|---:|---:|
| point query | HLO bytes | 774 291 | **244 063** | −68 % |
| | HLO ops | 10 384 | **3 241** | −69 % |
| | XLA compile, cold | 5.06 s | **0.168 s** | **30×** |
| | first call, warm | 0.0006 s | 0.0004 s | — |
| `vmap`, 4 096 pts | HLO bytes | 1 051 898 | **311 840** | −70 % |
| | HLO ops | 12 626 | **3 798** | −70 % |
| | XLA compile, cold | 7.94 s | **0.330 s** | **24×** |
| `grad` over params | HLO bytes | 1 615 183 | **584 771** | −64 % |
| | HLO ops | 18 090 | **5 771** | −68 % |
| | XLA compile, cold | **25.38 s** | **0.675 s** | **38×** |
| WGSL | bytes | 2 419 438 | **1 450 816** | −40 % |
| | `let` statements | 58 593 | **35 630** | −39 % |
| | functions emitted | 24 | 42 | (the shared ones) |

That 25.4 s gradient compile is §4.1's *"a novel design costs 30 s of XLA
compile, every time"*, and it is now 0.68 s.

## 12.5 End to end, through the real compile worker

Fresh subprocess per request, `cadjoint/viewer/_compile_worker.py` driven on
stdin exactly as the viewer drives it. Three runs; cold is the first against an
empty cache, warm is the median of the rest. (The machine was shared during
these runs — the wall clocks carry a few seconds of noise; the HLO figures in
§12.4 do not.)

| scene | mode | flat cold | flat warm | structured cold | structured warm |
|---|---|---:|---:|---:|---:|
| starter | `compile` | 3.32 s | 1.24 s | 3.50 s | 1.24 s |
| starter | `mesh` | 13.9 s | 5.9 s | 12.2 s | **4.77 s** |
| end_cap | `compile` | 7.51 s | 4.48 s | 6.28 s | **3.21 s** |
| end_cap | `mesh` | 106.7 s | 37.8–41.6 s | **42.4 s** | **15.2–16.8 s** |

`end_cap` `mesh` is **2.5× faster warm and 2.5× faster cold**. The `compile`
response payload for `end_cap` — the shader that crosses the wire to the
browser — drops from 12.63 MB to 7.67 MB.

## 12.6 Numerical invariants

512 pseudorandom points in the scene's bounding box, structured vs flat
lowering, float32 (eps ≈ 1.2 × 10⁻⁷):

| tree | max abs Δ value | value scale | max abs Δ grad | grad scale | relative |
|---|---:|---:|---:|---:|---:|
| starter `sink` (the comb) | **0.0** | 1.44 | 1.53e−5 | 195.0 | 7.8e−8 |
| starter `scene` | **0.0** | 1.12 | 4.77e−7 | 240.3 | 2.0e−9 |
| end_cap `scene` | 1.64e−7 | 1.32 | 5.72e−6 | 153.0 | 3.7e−8 |

The two polygon forms are **bit-identical**: `min` and a parity count are exact
reductions of the sequential `minimum` and sign flips they replace. The end
cap's 1.6e−7 is the polar pattern alone — the vectorised form computes
`cos`/`sin` of a traced angle where the unrolled form folded a Python float, and
`origin + (p − origin)` is `p` only to within a rounding step. Every relative
difference is below 1e−7, i.e. at float32 rounding.

Also asserted in `tests/sdf/primitives/test_polygon_lowering.py`,
`tests/test_functionalize.py` and `tests/backends/test_wgsl_uniforms.py`.

## 12.7 Parameters as arguments: the cache proof

Three free parameters edited by a constant; each row is a **fresh process**
sharing one `CADJOINT_CACHE_DIR`. `sha` is over the lowered StableHLO text.

### `scenes/starter.py`

| edit | form | StableHLO sha (16) | bytes | XLA compile | cache entries before → after |
|---|---|---|---:|---:|---|
| +0.00 | literal | `56054bd1a38858d9` | 58 662 | 0.083 s | 0 → 6 |
| +0.05 | literal | `92412225b5465668` | 58 661 | 0.080 s | 6 → **8** |
| +0.11 | literal | `67074d8bcb07a884` | 58 663 | 0.085 s | 8 → **10** |
| +0.00 | parametric | `d867a2cb08270c1a` | 55 412 | 0.060 s | 0 → 6 |
| +0.05 | parametric | `d867a2cb08270c1a` | 55 412 | **0.007 s** | 6 → **6** |
| +0.11 | parametric | `d867a2cb08270c1a` | 55 412 | **0.007 s** | 6 → **6** |

### `scenes/end_cap.py`

| edit | form | StableHLO sha (16) | bytes | XLA compile | cache entries before → after |
|---|---|---|---:|---:|---|
| +0.00 | literal | `78e97e3604128452` | 244 042 | 0.304 s | 0 → 6 |
| +0.05 | literal | `2163df79fca6e6d1` | 244 040 | 0.316 s | 6 → **8** |
| +0.11 | literal | `4d9cb41cbace1f15` | 244 040 | 0.342 s | 8 → **10** |
| +0.00 | parametric | `eb83c4f5381c14e7` | 232 387 | 0.219 s | 0 → 6 |
| +0.05 | parametric | `eb83c4f5381c14e7` | 232 387 | **0.027 s** | 6 → **6** |
| +0.11 | parametric | `eb83c4f5381c14e7` | 232 387 | **0.027 s** | 6 → **6** |

The literal form writes two new cache entries per edit and never hits; the
parametric form is byte-identical across all three values and hits from a cold
process. One caveat: byte-identity needs matching **avals**, not just shapes —
a weakly-typed Python float and a `float32` array lower differently. Values
that come from `extract_parameters` / `apply_parameters` are always `float32`
arrays, which is why the scenes above are stable.

## 12.8 WGSL: the uniform contract (§8, ranked item 9 — done)

`compile_scene_to_wgsl(scene, uniforms=True)` — equivalently
`compile_scene_with_uniforms(scene)` — returns a `ShaderProgram` instead of a
string. Literal inlining stays the default until the frontend adopts it; nothing
in `frontend/` was touched.

**Buffer layout.** One `vec4<f32>` slot per parameter — the only element type a
WGSL uniform array carries without per-field alignment rules — so slot `i` sits
at byte `16·i` and a 1-, 2- or 3-component parameter uses `.x` / `.xy` / `.xyz`
of it. The module declares:

```wgsl
struct SdfParameters { values: array<vec4<f32>, N>, };
@group(3) @binding(0) var<uniform> sdf_parameters: SdfParameters;
```

`@group(3)` is free: the preview shader, the path tracer, the overlay, the
graticule and the simulation shader all bind at `@group(0)`. Both indices are
arguments (`group=`, `binding=`) and are reported back on the program.

**Names.** Exactly the names `extract_parameters` returns — a free parameter's
declared name (`fin_depth`, `base_l`), a fixed one's `node.attribute` path
(`extrudedpolygon_1.depth`). `ShaderParameter` carries
`{name, offset, components, value, free}`; `ShaderProgram.buffer()` packs the
current values into the `float32` array to upload, padding included.

**Entry points are unchanged.** `sdf(p) -> f32`, `material_base(p) -> vec4<f32>`,
`material_optics(p) -> vec4<f32>`, all three reading the same buffer. Internally
each is a thin wrapper over an `*_impl` that takes the parameters as arguments.

**What it costs.** Values can no longer be constant-folded, so the module grows:

| scene | literal WGSL | uniform WGSL | parameters | buffer |
|---|---:|---:|---:|---:|
| starter | 264 694 B | 276 383 B (+4 %) | 143 | 2 288 B |
| end_cap | 1 450 816 B | 2 031 352 B (+40 %) | 325 | 5 200 B |

In exchange the source is **byte-identical across every parameter edit**
(verified: literal sha changes, uniform sha does not), so a slider drag becomes
a 2–5 kB buffer write and a redraw instead of a 1.6 s round trip, a multi-MB
transfer and a full browser shader recompile. Both modules compile through
wgpu-native/Naga.

Two footnotes for whoever wires the frontend: a pattern's `count` keeps a slot
it never reads (the instance count decides how much shader is emitted, so it
cannot be edited without a recompile), and a parameter wider than four floats
stays a literal rather than distorting the layout.

## 12.9 Two fixes the shader backend needed on the way

- **Callee ordering.** WGSL has no forward declarations, and outlining nests
  helpers arbitrarily deep. `convert` now emits functions in topological order
  rather than reversed declaration order.
- **NaN constants.** XLA leaves a NaN behind in the untaken branch of the
  guarded-`sqrt` idiom. It used to be folded away with the parameter values; as
  arguments it survives to the emitter, which raised. It is now emitted as
  `bitcast<f32>(0x7fc00000u)` — exact and portable — and a dead-code pass drops
  the ones nothing reads.

## 12.10 So: do we need another IR?

**No.** Every measurement above came from lowering the *existing* graph better,
and the two structures a second IR would have been built to provide already
exist in the stack:

- **Function-level sharing** is `func.func` + `func.call` in StableHLO, reached
  from Python with a nested `jax.jit`, and the WGSL emitter already maps one to
  one. This is what a "one function per primitive type" IR would have bought,
  without a second lowering to maintain.
- **Loop-level sharing** is `vmap` over an instance or vertex axis. XLA is a
  tensor compiler; giving it a `(N, 2)` array is telling it the same thing a
  loop-carrying IR would.

A second IR would also have to be kept honest against `patch_fields`,
`extract_parameters`' path keys, materials, face references and the constraint
system — all of which read the object graph directly. That is the real cost, and
nothing measured here justifies paying it.

Two things do still argue for *more* structure, and neither needs a new IR:

1. **Spatial culling by bounding box.** Everything above shrinks the program by
   removing duplication; none of it removes *work*. A sphere trace still
   evaluates all 42 leaves of the end cap at every step, and a `min` over a
   bounding-box-rejected branch is a `select`, not a skipped branch. A
   conservative bounding volume per node, emitted as an early-out, is the next
   order-of-magnitude lever — and it is a property computed *on the existing
   graph*, not a new representation of it.
2. **A shader that is not a straight line.** The remaining 1.45 MB of end-cap
   WGSL is three entry points each holding the whole tree, with the profiles
   unrolled because WGSL cannot type an `(N, 2)` array. A hand-written WGSL
   kernel per primitive type — reading its vertices from a storage buffer, with
   the CSG tree as data — would collapse it to a few kilobytes. That is a
   *second backend*, not a second IR: the graph it walks is the same one.

## 12.11 Reproducing §12

Scripts in the ephemeral scratch workspace
`…/scratchpad/ir/`: `measure2.py <scene> [--wgsl]` (§12.4, `FLAT=1` for the
before column), `runworker.py <scene> <mode>` (§12.5, same `FLAT` switch),
`invariants.py` (§12.6), `parametric.py <scene> <edit>` (§12.7, `FORM=literal`
for the control), `uniformcheck.py <scene>` and `shadercheck.py <scene>`
(§12.8). Each takes `CADJOINT_CACHE_DIR` from the environment; every "warm"
number is the second run against a populated one.

# 13. The shader: what a parameter edit costs the GPU, and why folding decides it

Status: shipped (2026-09-03). §12.8 built the uniform form and left it unused;
this section is the frontend adopting it, the 21× regression that adoption
exposed, and what the evidence said to do about it.

**Machine**: the same Apple Silicon host as §0. **Adapter**: `apple metal-3`
through Chromium's WebGPU (`--use-angle=metal`). **Frames**: 1200 × 800, median
of 8 after 2 warm-up frames, two repetitions, the second quoted. Every
`createShaderModule` / `getCompilationInfo` / `createRenderPipelineAsync` is
timed with `performance.now()` in the page.

**Pixel check**: every frame table below was taken with a coverage and
mean-luminance probe on the same rendered image. Unless a row says otherwise
its probe is identical to the literal row's to every digit — that is what
makes the frame times comparable at all.

## 13.1 Before: the literal form, as the viewer shipped it

Each scene compiled through the real worker (`mode: "compile"`), warm cache.

| scene | worker wall | payload | preview WGSL | path WGSL |
|---|---:|---:|---:|---:|
| `starter` | 1.85 s | 1.71 MB | 312 088 B | 302 680 B |
| `end_cap` | 4.92 s | 8.64 MB | 1 671 539 B | 1 662 131 B |
| `motor_shield` | 12.01 s | 23.83 MB | 4 567 621 B | 4 558 213 B |

Browser-side, per compile — and this is paid **on every edit**, because in the
literal form every design parameter is a float constant in the source:

| scene | modules (create + info) | pipelines (4) |
|---|---:|---:|
| `starter` | 4.1 ms | 3.8 ms |
| `end_cap` | 27.3 ms | 18.8 ms |
| `motor_shield` | 648.4 ms | 745.2 ms |

Frame time by display mode, Ultra:

| scene | default | pbr | slice | gradient | normal | depth | path/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| `starter` | 0.9 | 0.9 | 1.3 | 1.3 | 0.9 | 0.9 | 6.1 |
| `end_cap` | 3.8 | 4.6 | 5.3 | 5.3 | 4.4 | 3.7 | 41.9 |
| `motor_shield` | 15.7 | 15.9 | 19.4 | 26.1 | 24.9 | 21.2 | 258.9 |

So a slider drag on `motor_shield` was a 12 s round trip followed by 1.4 s of
browser compilation, per edit. That is what the uniform form was built to
remove.

## 13.2 The regression: the uniform form was correct and 21× slower

Switching the worker to `compile_scene_to_wgsl(scene, uniforms=True)` — every
parameter in a `@group(3)` buffer, source byte-identical across edits — worked,
drew the identical image, and cost this:

| scene | literal | every parameter buffered | ratio |
|---|---:|---:|---:|
| `starter` (143 params) | 0.9 ms | 1.1 ms | 1.2× |
| `end_cap` (330 params) | 3.8 ms | 105.8 ms | **28×** |
| `motor_shield` (889 params) | 15.7 ms | 605.4 ms | **39×** |

The ratio is wildly non-linear in the parameter count, which rules out any
per-parameter cost and points at a cliff.

### 13.2.1 Four candidate causes, tested rather than assumed

All four variants below are built from the **same** all-uniform module for
`end_cap`, by textual substitution, so structure is held constant and only the
spelling of a parameter read changes. All four draw the identical image
(coverage 0.036617, mean luma 225.5668, matching the literal build exactly).

| variant | what it is | uniform reads | default frame |
|---|---|---:|---:|
| A | the literal build (control) | 0 | 4.0 ms |
| B | all-uniform module, reads replaced by their literal values | 3 | **3.4 ms** |
| C | all-uniform module, reads hoisted to one `let` per function | 671 | 108.1 ms |
| D | all-uniform module, as emitted | 4 560 | 105.6 ms |
| E | only the 11 **free** parameters left as reads | 255 | **3.3 ms** |
| F | only **one** parameter left as a read | 10 | 3.5 ms |

Read across, this settles it:

- **Not the number of loads.** C cuts them 6.8× and changes nothing (108.1
  against 105.6 — within noise, and on the wrong side of it).
- **Not the shape of the emitted code.** B has D's exact function list,
  argument counts and expression tree; substituting the values back recovers
  the full speed, and then some.
- **Not the driver deoptimising on a hot-loop uniform read.** B, E and F all
  read the same buffer in the same loop and are all fast.
- **Not a benchmark artefact.** Same harness, same warm-up, same frame count,
  same pixels, both directions of the substitution.

**It is constant folding, and nothing else.** `scenes/end_cap.py` is 1.7 MB of
WGSL *because* it is mostly foldable: 21 pattern instances, unrolled, each
carrying its own transform algebra that collapses to a few instructions once
the transform is a constant and runs in full when it is not. The module size
is not the cost; the module size is the *evidence* of how much the compiler
normally deletes.

B being slightly faster than A is the same fact seen from the other side: the
uniform form's outlining happens to give Metal a marginally better program to
fold than the literal lowering does.

### 13.2.2 Two corollaries worth recording

The reserved NaN slot cannot be substituted away. Every constant spelling of a
NaN — `bitcast<f32>(0x7fc00000u)` included — is const-evaluated and rejected
("value nan cannot be represented as 'f32'"), which is why §12.8 put one in the
buffer. Variant B keeps exactly that one read.

Hoisting is not a hidden win either. 31.6 % of the emitted bindings are
invocation-invariant (they depend only on parameters, never on the point) and
sit inside the marched `sdf` call being recomputed at every step — 13 323 of
`sdf_impl`'s 25 221 for `motor_shield`. Lifting them out is textbook LICM, but
the *frontier* — the invariant values that point-dependent code actually reads,
and so the values that would have to stay live across the march loop — is 2 712
for `motor_shield` and 919 for `end_cap`. An Apple GPU thread has on the order
of a hundred registers before occupancy collapses. Hoisting would trade ALU for
spill traffic, which is the wrong direction, and variant C is the small-scale
measurement that says so.

## 13.3 What shipped: only the free parameters get a slot

`compile_scene_with_uniforms(..., scope="free")` is now the default, and the
worker's default. A free parameter — declared, named, optimizable, and the only
kind a handle drags or an optimizer moves — gets a `vec4` slot. A fixed one — a
node attribute, a material property, a bare float literal — stays a constant in
the source and still costs a recompile when it changes.

The ratio is what makes this work: `end_cap` has 11 free parameters against
319 fixed, `motor_shield` 41 against 848. The default leaves 95–97 % of the
scene's numbers foldable.

| scene | free / all | buffer | preview WGSL vs literal |
|---|---:|---:|---:|
| `starter` | 35 / 143 | 576 B | +2.5 % |
| `end_cap` | 11 / 330 | 192 B | +0.3 % |
| `motor_shield` | 41 / 889 | 672 B | +0.2 % |

### After: frame time by display mode, Ultra

| scene | form | default | pbr | slice | gradient | normal | depth | path/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `starter` | literal | 0.9 | 0.9 | 1.3 | 1.3 | 0.9 | 0.9 | 6.1 |
| `starter` | **free (shipped)** | 1.1 | 1.1 | 1.5 | 1.5 | 1.1 | 1.0 | 8.3 |
| `end_cap` | literal | 3.8 | 4.6 | 5.3 | 5.3 | 4.4 | 3.7 | 41.9 |
| `end_cap` | **free (shipped)** | **3.3** | **3.4** | **4.7** | **4.7** | **3.3** | **2.8** | **37.2** |
| `motor_shield` | literal | 15.7 | 15.9 | 19.4 | 26.1 | 24.9 | 21.2 | 258.9 |
| `motor_shield` | **free (shipped)** | 35.8 | 36.3 | 51.3 | 51.1 | 35.9 | 35.2 | — |

`end_cap` is *faster* than the literal build it replaces. `starter` is 0.2 ms
slower, which is a fifth of a frame at 1200 × 800 and inside the run-to-run
spread of the harness. `motor_shield` is 2.3× slower and that is a real cost,
recorded in §13.6 rather than explained away: 41 live parameters, most of them
sketch-profile vertices feeding polygon SDFs whose per-edge algebra is exactly
what folding used to delete.

Against what it buys — the alternative on `motor_shield` is not 15.7 ms, it is
15.7 ms *plus a 12 s round trip and 1.4 s of browser compilation for every
edit* — 35.8 ms is the right trade. It is also the only form in which the drag
exists at all.

### The drag

A pointer move during a drag is now `queue.writeBuffer` of ≤ 672 bytes and a
redraw. Measured end to end through the real app, server and GPU
(`frontend/e2e/shader.spec.ts`), 60 consecutive dragged frames:

| counter | before drag | after 60 frames |
|---|---:|---:|
| pipelines built | 8 | **8** |
| shader modules compiled | 5 | **5** |
| parameter uploads | 0 | **60** |

**Zero pipeline rebuilds per drag**, one buffer write per frame, and the image
demonstrably follows the buffer (the test reads the canvas back at two
different values and requires them to differ, so a renderer that ignored the
overrides could not pass).

## 13.4 Sparseness: conservative bounds and a real branch

`cadjoint/backends/wgsl/_culling.py` traces the same tree as
`functionalize_scene`, node for node, with one addition: before a boolean
evaluates an operand it compares the distance to that operand's bounding box
against the value it already holds, and skips the operand when it provably
cannot change the answer. `lax.cond` lowers to `stablehlo.case`, which the
emitter turns into a real `if`, so the skipped branch costs nothing.

Bounds are computed in `cadjoint/sdf/_lowering.py` from the *traced* parameter
values, so they follow a parameter edit rather than being baked at compile
time — `test_the_bound_follows_a_parameter_edit` pins that, and it is what
keeps culling correct in the uniform form. A pattern is bounded per instance,
a smooth union is grown by its blend band `4k`, and a node that cannot promise
the bound (a drafted extrusion, say) reports `None` and is never skipped.

**It is not an approximation.** Each skip is taken only where the exact value
is what the running result already is: a smooth union's band term is *exactly*
zero when `d >= m + K`, and the box distance is a lower bound on `d`, so
`box(p) >= m + K` suffices. The module docstring carries the algebra per node
family, and `CULL_MARGIN = 1e-4` covers float rounding in the box distance
three orders above its magnitude.

| verification | scope | result |
|---|---|---|
| culled field vs flat field | every node family, 20 k points each | ≤ 1e-6 |
| culled field vs flat field | every shipped scene, 100 k points each | ≤ 1e-6 |
| box distance ≤ node distance outside the box | every node family, 40 k points | holds at the root, which inherits every child's error |

### Where it helps, and where it does not

| scene | mode | culling off | culling on | speed-up |
|---|---|---:|---:|---:|
| `starter` | default | 1.1 | 1.1 | 1.0× |
| `starter` | slice | 2.3 | 1.5 | 1.5× |
| `end_cap` | default | 6.9 | 3.3 | **2.1×** |
| `end_cap` | slice | 32.2 | 4.7 | **6.9×** |
| `end_cap` | path/sample | 61.3 | 37.2 | 1.6× |
| `motor_shield` | default | 110.5 | 35.8 | **3.1×** |
| `motor_shield` | slice | 166.2 | 51.3 | **3.2×** |

It does nothing for `starter` in the default view and everything for the two
large parts, which is the expected shape: culling removes work proportional to
how much of the tree is far from the ray, and a four-leaf scene has none to
remove. The slice views gain most, because a slice plane marches through empty
space where nearly every leaf is skippable.

The cost is source size — the branch is emitted per operand — at +9 % preview
WGSL for `end_cap` and +13 % for `motor_shield`, and it is worth it several
times over.

Intersections and XORs are not culled: a lower bound on an operand cannot show
that a *maximum* is unchanged. Their operands are still culled inside, where
they are unions.

## 13.5 Caching, both sides

### Browser: modules by source

`ShaderModuleCache` (`frontend/src/viewer/shaderProgram.ts`) keys compiled
`GPUShaderModule`s by their own source, LRU, capacity 8 — bounded because the
keys *are* the sources and a scene's shaders are megabytes of string. Above it
sits the renderer's own short-circuit: when a payload's sources are identical
to the installed ones it never asks for a module at all.

A four-step scripted session (compile → free-parameter edit → topology edit →
undo), measured through the real app:

| step | pipelines built | module hits | module misses |
|---|---:|---:|---:|
| initial compile | 8 | 1 | 5 |
| free-parameter edit | **8** | 1 | 5 |
| topology edit | 12 | 2 | 7 |
| undo (back to the first source) | 16 | **5** | **7** |

**Hit rate 41.7 %** over the session. The two rows that matter: a
free-parameter edit adds *nothing* to either counter, and the undo installs
three modules while compiling none of them.

### Worker: what the persistent XLA cache actually holds

Fresh process per request, private cache directory, counting files gained:

| step | wall | cache entries | shader hash |
|---|---:|---:|---|
| 1. cold cache | 4.41 s | 0 → 482 (+482) | `3b7c86d03fed` |
| 2. same source again | 2.05 s | 482 → 482 (+0) | `3b7c86d03fed` |
| 3. free-parameter edit | 2.04 s | 482 → 482 (+0) | `3b7c86d03fed` |
| 4. fixed-parameter edit | 2.03 s | 482 → 482 (+0) | `d2e3a26046ab` |
| 5. topology edit (new leaf) | 2.09 s | 482 → 482 (+0) | `60fdc6203c7f` |
| 6. back to the original | 2.05 s | 482 → 482 (+0) | `3b7c86d03fed` |

Two findings.

**A free-parameter edit produces a byte-identical shader** (rows 1–3 share a
hash), which is the whole contract, confirmed end to end through the worker
rather than in a unit test. A fixed-parameter edit does not, by design.

**The persistent XLA cache misses nothing on a topology edit, because it is
not involved.** `compile` mode traces to StableHLO and emits text; it never
asks XLA for an executable, so a new leaf adds no cache entry and costs the
same 2.0 s as a no-op. The 482 entries are laid down once by the *constraint
solver*, and they scale with sketch content rather than with the CSG tree:

| scene | wall | entries |
|---|---:|---:|
| two spheres, no sketch | 0.57 s | 36 |
| one unconstrained sketch | 1.08 s | 156 |
| `starter` (constrained sketches) | 3.80 s | 482 |

So the 2.0 s warm floor of a `compile` request is Python-side tracing and WGSL
emission, not compilation. Cutting it is §6.1 and §6.5's problem, not this
section's.

### Outlined `func.func` bodies across scenes

**They are not reused, and nothing in the current design could reuse them.**
An outlined body is produced by a nested `jax.jit` inside
`functionalize_scene`, closed over that scene's parameter dicts and named by
its DFS index (`sdf_impl__sdf_eval_122`). Two scenes sharing a subtree get two
separately traced, separately named, separately emitted copies, and one scene
edited twice gets new names as soon as the DFS numbering shifts.

Reuse would need three things that do not exist: a **content hash** of a
subtree's shape and static attributes to name bodies by, in place of the
positional index; a **parameter-passing convention** so a shared body takes
its values as arguments rather than closing over one scene's dicts — which is
exactly the machinery `_uniform_bindings` had to defeat to stay under WGSL's
255-argument limit, so it would have to be a struct or a buffer slice; and a
**cross-request store** for the emitted WGSL, since the worker is a fresh
process per request. That is a real project, and §12.10's conclusion stands:
nothing measured here justifies it ahead of the two levers that are already
paying — culling, and not compiling at all.

## 13.6 What is left undone

- **`motor_shield` in the free form is 2.3× the literal frame time**
  (35.8 ms against 15.7 ms). Intrinsic to having 41 live parameters, most of
  them sketch vertices feeding polygon SDFs. The two obvious attacks are both
  measured and both rejected above: buffering fewer parameters is what the
  free scope already does, and hoisting the invariants would spill (§13.2.2).
  The remaining route is §12.10's second backend — a hand-written WGSL kernel
  per primitive type reading its vertices from a storage buffer — which would
  make profile vertices data rather than code and collapse the whole question.
- **Path-trace timings for `motor_shield` in the uniform forms are missing.**
  The harness returned 0.1 ms, which is not a measurement; at 4.6 MB the path
  pipeline appears not to survive the run. The literal figure (258.9 ms) is
  sound. Worth a look, but the path tracer is not the interactive path.
- **`scope="all"` is kept and tested but must never ship.** It exists because
  it is the control the 31× is measured against and the form that stresses the
  emitter's argument-binding pass to WGSL's 255-parameter limit.
- **The gizmo's own drag does not yet drive the buffer.** The mechanism is in
  place and tested (`Renderer.setParameterOverrides`, 60 frames, zero
  rebuilds), but wiring the transform gizmo to it needs a link from a
  construction node's transform to the free parameter backing it, and the
  construction payload does not carry one. A gizmo drag on a node whose
  placement *is* a free parameter already takes the values-only path on
  commit; what is missing is the frame-rate preview for it.

## 13.7 Reproducing §13

Scripts in the ephemeral scratch workspace, `shader-` prefixed:
`shader-compile.py <scene> <label>` (worker walls, payload sizes, shader
sources; honours `CADJOINT_SHADER_FORM`, `CADJOINT_SHADER_SCOPE` and
`CADJOINT_SHADER_CULL`), `shader-bench.mjs <label.json>...` (module,
pipeline and per-mode frame timings in Chromium with the pixel probe),
`shader-variants.py <label> <mode> <out>` (§13.2.1's A–F), `shader-invariant.py`
(§13.2.2's invariant and frontier counts), `shader-xlacache.py` and
`shader-xlaorigin.py` (§13.5). The e2e counters come from
`npx playwright test e2e/shader.spec.ts --reporter=json`, whose attachments
carry the tables in §13.3 and §13.5 verbatim.

## 13.8 The march, as settings — three controls, each with its price

Status: shipped (2026-09-03). §13.3 made the *scene's* parameters editable
without a recompile. This does the same for the *renderer's*: the step budget,
hit refinement and bounds culling are now controls in the Render panel, and
none of them touches the source, triggers a compile or is lost by one.

**How they reach the shader.** Two ride in the viewer's uniform block — the
budget in `path_settings.w`, refinement as a new bit (32) in the `display.z`
bitfield — so they are part of the per-frame write that already happens.
Culling cannot: its skip tests are inside the *generated* scene module, which
reads no uniform but its own parameter buffer, so the cull margin takes a
reserved slot there beside the NaN. Toggling it is a `writeBuffer` of a few
hundred bytes, exactly like a handle drag.

**Method.** Same harness and machine as §13, 1200 × 800, median of 21 frames
after 2 warm-up, on the shipped `scope="free"` shaders. Every row renders the
setting off and on and diffs the two frames pixel for pixel, because a control
that changes no pixel and a control that changes half the frame are both
bugs. **Pixel columns are exact and reproduced identically across runs; the
timing column on `motor_shield` is not** — a 4.6 MB shader throttles, and its
ratios there moved by ±40 % between runs. Read `starter` and `end_cap` for
cost, all three for effect.

### The step budget (`path_settings.w`, 16…512, default: the quality tier)

| scene | change | frame | pixels changed | worst level |
|---|---|---:|---:|---:|
| `starter` | 192 → 64 | ×1.00 | 0.00 % | 0 |
| `starter` | 192 → 384 | ×1.00 | 0.00 % | 0 |
| `end_cap` | 192 → 96 | ×0.97 | 0.00 % | **187** |
| `end_cap` | 192 → 64 | ×0.96 | 0.02 % | **188** |
| `end_cap` | 192 → 384 | ×1.00 | 0.00 % | 0 |
| `motor_shield` | 192 → 96 | (noise) | 0.02 % | 85 |
| `motor_shield` | 192 → 64 | (noise) | 0.11 % | 175 |
| `motor_shield` | 192 → 384 | (noise) | 0.00 % | **16** |

**It is a cap, not a cost.** Nearly every ray converges and returns long
before the budget, so raising it is close to free — 192 → 384 is inside the
noise on every scene — and lowering it does not reliably buy time either. What
lowering it does is drop the rays that *needed* the steps, and those are
whole-pixel failures: a "worst level" of 187 on a 0–255 channel is a pixel
that went from surface to background. `end_cap` at 96 changes too few pixels
to round above 0.00 % and still breaks them completely.

Two consequences worth naming. `starter` is converged by 64, so the Draft tier
costs it nothing. `motor_shield` is *not* converged at 192 — a handful of
pixels still move at 384 — so the Ultra tier is a floor for that part, not a
ceiling. This is why the control exists at all, and why the panel calls it a
cap rather than a quality slider.

### Hit refinement (`DISPLAY_REFINE_HIT`, default off)

The march stops at the first sample inside `SURFACE_EPS`, and takes its `t`.
On a ray meeting the surface squarely that is accurate to the band; on a
grazing one the step is `0.9|f|` and |f| falls slowly along a tangent, so the
march creeps and stops up to a whole epsilon short. `refine_hit` re-solves for
the crossing by clamped secant iteration from the two samples the march
already holds — the first correction needs no field evaluation at all, and
`REFINE_STEPS` (4) further ones cost one each, only on pixels that hit.

| scene | mode | frame | pixels changed | worst level |
|---|---|---:|---:|---:|
| `starter` | default | ×1.00 | 0.10 % | 133 |
| `starter` | pbr | ×1.00–1.10 | 0.12 % | 87 |
| `starter` | normal | ×1.00 | 0.44 % | 87 |
| `starter` | depth | ×1.00–1.10 | 0.17 % | **1** |
| `end_cap` | default | ×1.21 | 0.38 % | 137 |
| `end_cap` | pbr | ×1.17 | 0.57 % | 95 |
| `end_cap` | normal | ×1.17 | 0.85 % | 119 |
| `end_cap` | depth | ×1.25 | 0.12 % | **1** |
| `motor_shield` | default | (noise) | 1.04 % | 128 |
| `motor_shield` | normal | (noise) | 2.11 % | 255 |
| `motor_shield` | depth | (noise) | 0.38 % | **1** |

0.1–2.1 % of the frame moves, which is the silhouette band and nothing else —
and the depth view is the proof of mechanism: there the change is *one level
everywhere it hits*, which is the sub-epsilon correction showing up as
exactly what it is. The normal view gains most (0.44–2.11 %), which follows:
a normal is sampled at the hit, so a hit that is short samples the gradient
off the surface.

Cost is 0 % on `starter` and 17–25 % on `end_cap`. It defaults **off** because
the default has to be the image the viewer has always drawn, not because it is
the worse image.

The clamp is what makes it safe to ship: every secant step is confined to the
bracket the march established, widened by one epsilon, so refinement can move
a hit by at most an epsilon and can never create, destroy or relocate one.
Silhouettes are identical with it on and off; only their sub-pixel coverage
changes.

### Bounds culling (a reserved slot in the parameter buffer, default on)

| scene | mode | culling off | pixels changed |
|---|---|---:|---:|
| `starter` | default | ×1.00 | **0.00 %** |
| `starter` | pbr | ×1.20 | **0.00 %** |
| `starter` | slice | ×1.13 | **0.00 %** |
| `end_cap` | default | **×2.18** | **0.00 %** |
| `end_cap` | pbr | **×2.10** | **0.00 %** |
| `end_cap` | slice | **×2.43** | **0.00 %** |
| `motor_shield` | default | ×1.70 | **0.00 %** |
| `motor_shield` | pbr | ×2.03 | **0.00 %** |
| `motor_shield` | slice | **×2.45** | **0.00 %** |

This is the table §13.4 could only assert from a benchmark build, now
measured as a live toggle on one shader: **2.0× to 2.45× the frame, and not
one pixel different.** A four-leaf scene has nothing to skip, which is why
`starter` is flat. The slice views gain most, as they did before — a slice
plane marches through empty space where nearly every leaf is skippable.

**Making it a toggle cost nothing.** The margin every skip test is compared
against became a traced argument rather than a folded constant, which by
§13.2's own lesson is exactly the kind of change that can cost 31×. It does
not: `end_cap` default was 2.9 ms with the constant margin and 2.8 ms with the
uniform one, `motor_shield` 41.7 against 41.1 — within noise on every scene.
The reason is that the threshold it is added to (`max(result + band, 0)`) is
already a runtime value, so nothing was folding there to lose.

The "off" position is `margin = +∞`, which makes every test false. That is a
stronger statement than a large finite margin: for a scene of unknown extent
only infinity guarantees *no* box test can pass. Verified against the flat
field at 2.4e-7 over 100 k points on both shipped scenes.

### What was refused, and why

- **Over-relaxation (ω > 1).** No lane to put it in, and no measurement. The
  march already runs *under*-relaxed at 0.9. Over-relaxation is fast and
  tunnels through thin walls, and a tunnelled wall looks like a modelling
  error rather than a render setting — the brief required the failure mode to
  be visible, and I have no way to make it so. Not exposed.
- **The surface epsilon** (`SURFACE_EPS`, 0.001) and **the far distance**
  (`MAX_TRACE_DISTANCE`, 100). Both are real knobs and both want a float lane;
  there is none (below). They are named constants now rather than literals
  buried in `trace`, which is the part of exposing them that was free.
- **A live frame-time readout beside the step slider.** It is the honest way
  to "show the effect", and it belongs in the renderer's draw loop, which
  another agent held for the duration. Proposed, not built.

### The widget contract: the struct is full

`Uniforms` is 7 × `vec4<f32>` = 112 bytes, and `cadjoint/viewer/widget.py`
allocates exactly that. **All 28 floats now carry meaning** — the six that
were spare in §12 were taken by the SDF views. So:

- The step budget needed no room: `path_settings.w` already held it.
- Refinement needed no room either: `display.z` is a bitfield with 5 of its
  bits used, and an f32 represents integers exactly to 2²⁴, so bits are the
  one thing still free. It took bit 32; 19 more are available.
- Culling could not have used the block at all — its tests are in the
  generated module — so it took a reserved slot in the parameter buffer,
  which is sized per scene and had room by construction.

**`path_settings.x/y/z` are read only by the path tracer** (sample index,
bounces, shadow rays) and are dead in the preview shader — but the two share
one buffer and both can be live in a single frame (the preview depth prepass
runs during path tracing), so they are **not** reusable. That is the last
apparent slack in the struct, and it is not slack.

**The proposal, if another float lane is ever needed**: grow `Uniforms` to
8 × `vec4` = 128 bytes and change `widget.py`'s allocation in the same commit.
The widget writes zeros to everything it does not understand, and every field
in the block is designed so 0 is its inert value, so an eighth vec4 costs the
widget one number and nothing else. What must not happen is the struct
growing while the widget's 112 stays — that faults the widget rather than
ignoring the new fields, which is why the two literals should be one named
constant when they next move.

### Reproducing §13.8

`shader-refine.mjs <label.json>...` in the scratch workspace renders every row
above off and on, times both and diffs the frames; `shader-compile.py` builds
its inputs through the real worker. The pixel columns are deterministic and
should reproduce exactly; the `motor_shield` timings should not be trusted to
better than ±40 %.

## 13.9 The capped section, the lighting rig, and what a tier actually buys

Status: shipped (2026-09-03). Three of the five things asked for in the render
settings, measured the same way §13.8 measured the march controls: every row
renders the setting off and on, times both, and diffs the frames.

**Method.** Apple Metal through Chromium, `scenes/starter.py` and
`scenes/end_cap.py` through the real compile worker, median of 9 frames after
2 warm-up. Screenshots at 2× in `research/design/light-chrome/`.

### A capped section, not a clipped one

`slice` and `∇f` draw the *field* on a card at the plane. That is a data view,
and it was never what "each object that we slice through gets a surface on the
slice plane instead of just being hollow there" was asking for. Clipping a
solid — refusing to draw in front of a plane — lets the ray carry on and hit
the **inside of the far shell**, which is exactly the hollow the complaint
names.

The capped version is one operation:

```wgsl
fn traced_field(p: vec3<f32>) -> f32 {
  let field = scene_field(p);
  if (!display_flag(DISPLAY_SECTION)) { return field; }
  return max(field, section_halfspace(p));
}
```

`max` is intersection, and intersecting the solid with a half-space *is* a
section. Three things fall out of it rather than having to be built:

- **The cap is a real surface** — the ray stops on the plane, because that is
  where the field reaches zero.
- **The cap has the right normal** — `sdf_normal` central-differences this
  same function, so on the cap it returns the plane's own normal. No special
  case anywhere.
- **The cap has the right material** — `material_base` is still evaluated at
  the hit position, which lies inside the solid that was cut, so the cut face
  arrives in the colour of the thing it cuts. In `section-capped-x.png` the
  copper slug and the two steel bushings are visible *in section*, in their
  own materials, inside the aluminium.

The march stays correct: both operands are proper distance fields, and the
`max` of two of those never overestimates the distance to their intersection,
which is the only property sphere tracing needs.

| scene | mode | frame off | frame on |
|---|---|---:|---:|
| `starter` | flat, hard shadows | 1.7 ms | 0.9 ms |
| `starter` | section on X | — | 0.9 ms |
| `starter` | section on Y | — | 0.9 ms |

**It is free, and slightly better than free**: less geometry survives the cut,
so fewer rays hit anything and the march returns sooner.

**Data and geometry compose rather than exclude.** The section cuts the solid;
the field card draws on the same plane and covers the cut where the card
reaches. They share one plane so a reader places it once, and the panel says
which is on top. Critically, the *data path* is untouched: `sdf_view_color`
and `sdf_gradient_magnitude` still read `scene_field`, not `traced_field`, so
a |∇f| view of a sectioned scene still reports the scene's own metric error
and not the plane's gradient. `test_the_section_leaves_the_data_path_on_the_true_field`
pins the routing function by function.

**Which half goes is a function of the camera, and that is deliberate.** The
design language forbids a *measurable* ground that moves with the eye. A
cutaway is not a scale: its purpose is to open the solid toward the reader,
and keeping the near half shows the outside of the far shell — the very
failure the mode exists to fix. So the half the camera is in is the half
removed, and orbiting through the plane swaps them, as it does in every
drafting package.

**The cut face is darkened to 0.72 and that is the weakest part of this.**
Drafting hatches a section because the cut is a fiction — not a surface the
part has — and a reader must be able to tell it from one that is. A hatch
cannot be drawn here: it is a screen-space pattern, and this shader is also
what the path tracer converges to and what the thumbnailer renders at other
sizes, so a fixed pitch would alias in one and vanish in the other. A flat
multiplier says the same thing at every scale, but it says it *quietly* —
in `section-capped-x.png` the large cut face reads as a plausible real face.
Recommend evaluating a hatch as its own piece of work, with the resolution
independence solved rather than assumed.

### Lighting: a key that no longer stands alone

One directional light left a whole hemisphere of normals on a flat ambient, so
orbiting to the far side of a part gave a dark unreadable slab. The rig now
is a key (shadow-casting, as before), two fills at ±120° about world up, and a
sky/ground hemisphere ambient in place of the constant. **Every lamp is
white**: the key used to be `vec3(1.0, 0.92, 0.82)`, which tinted every face of
every part and competed with both material colour and the viridis fields.

The second fill is pushed *below* the horizon. A textbook three-point rig
keeps every lamp above the subject, because a photographer's subject is not
inspected from underneath. A part is.

Model pixels only — the background is excluded, and a rig is judged by its
darkest face, not its average. "Dark" is luma < 60 of 255.

| orbit | mean luma before → after | dark fraction before → after |
|---|---|---|
| front-right (key side) | 69.4 → **136.7** | 17.9 % → **0.3 %** |
| back-left (away from key) | 69.7 → **140.5** | 16.3 % → **0.3 %** |
| back-low | 61.4 → **124.7** | 29.8 % → **0.6 %** |
| underside | 49.8 → **116.9** | 56.5 % → **0.1 %** |

The two worst views were the back-low and the underside, and they are now
within 12 % of the best. Nothing is washed out: the ground is 229 and the
model's mean sits at 117–141, well clear of it, with the silhouette contour
still closing the edge. Frame cost is unmeasurable — two dot products and a
mix, no extra marches, because the fills cast nothing (a fill that threw its
own shadow would put a second contradictory shadow under the part, and the
extra marches are the whole reason the key is the only one that gets them).

Before and after at all four orbits: `lighting-before-*.png`,
`lighting-after-*.png`.

### "The marching steps in ultra seem excessive" — they are not what costs

The step budget was measured against the pixel budget on `end_cap`, at the
three tiers' own resolutions:

| resolution | 64 steps | 96 | 192 | 384 |
|---|---:|---:|---:|---:|
| 715 × 447 (Draft, 319 k px) | 1.9 ms | 1.9 | 1.9 | 1.9 |
| 1200 × 750 (High, 900 k px) | 2.5 ms | 2.6 | 2.7 | 2.8 |
| 1600 × 1000 (Ultra, 1600 k px) | 4.1 ms | 3.9 | 3.9 | 3.9 |

**Read the rows: six times the step budget is inside the noise at every
resolution. Read the column: the tier is 1.9 → 2.7 → 3.9 ms, and that is the
whole of the difference between Draft and Ultra.** Ultra costs 2.05× Draft,
and 100 % of it is the pixel budget. The march cap contributes nothing
measurable to any of it.

So the budget was not lowered. Three things changed instead:

1. **The ladder is gone.** 64/96/192 became one number, 192, for every tier.
   A ladder that buys no time and costs image quality is all cost: Draft's 64
   steps punched holes in `end_cap`'s silhouette (§13.8: 0.02 % of pixels
   changed, worst level 188 — whole pixels lost) for zero saving.
   `DisplaySettings.marchSteps` still overrides it, which
   `scenes/motor_shield.py` needs since it is not converged at 192.
2. **"Quality" is now "Resolution"**, and prints what the tier actually
   contains — `1.60 MP · the whole of what a tier costs`, or the bounce and
   shadow-ray counts when path tracing is on.
3. **The march control says what it is.** "A cap, not a cost. Nearly every ray
   converges long before it, so raising it is close to free; lowering it drops
   the rays that needed the steps and erodes the silhouette."

The panel was misrepresenting the number, which is what made it look
excessive. The number was right.

### Panel organisation

The settings are now grouped by what they *are* rather than by when they were
added: **what is drawn** (shading, shadows, effects, annotations, the distance
field views and the section), **how well it is drawn** (Resolution, Marching),
and **Diagnostics — cost, not appearance** (bounds culling, which changes no
pixel on any shipped scene and 2.0–2.4× the frame). Every control that has a
measured cost prints it in its own subtitle. No new CSS classes and no new
colours: the group headings reuse `render-preset-heading`, the numeric rows
reuse `sim-slice` and `sim-legend-values`, the switches reuse `ToggleSwitch`.

### Not done: the per-object palette

Deliberately not attempted rather than asserted. A stable per-object colour
needs the *shader* to know which leaf owns a point, and to name that leaf by
its stable id. The material system already selects per-leaf through the CSG
tree, so the selection machinery exists — but the id it would have to hash is
in the construction payload, not in the SDF graph, which knows only a DFS
index. Hashing the DFS index is precisely the draw-order-derived colour the
brief rules out: it is not stable across the edits the mode exists to help
with. Doing it properly means carrying stable ids into `functionalize_scene`
and emitting a fourth entry point beside `material_base` / `material_optics`.
That is a codegen change of the same size as §12.8's, and it wants its own
measurement pass.

---

# 14. The constraint solve — 94 XLA programs for two Newton steps, and where they actually came from

Status: **implemented and measured** (2026-09-03). Changes `cadjoint/constraints/solve.py`
only. Same machine as §0; jax 0.8.2, CPU backend. Every number is a fresh
subprocess with a private, empty `CADJOINT_CACHE_DIR` unless the row says warm.
"XLA programs" counts calls to `jax._src.compiler.compile_or_get_cached`, which
is one per executable JAX asks the backend for, cache hit or miss.

## 14.1 The question

> *"Can we improve this by approximating the jacobian maybe? maybe in favor of
> more steps? we can also expose these constraints options maybe."*

§13.5 had found that the persistent cache's 482 entries for `starter` "are the
constraint solver's and scale with sketch content". The four candidate levers,
in the order they were expected to matter, were: roll the step loop into
`lax.scan`; exploit the Jacobian's block sparsity or go matrix-free; batch
identical constraints under `vmap`; approximate the Jacobian (Broyden / chord).

**The ordering was wrong, and so was the diagnosis of lever 1.** The cost was
not the Jacobian at all. It was that *the entire solve ran eagerly*: a Python
`for` loop calling `jax.jacobian` and `float(loss)` op by op, so JAX compiled
one program per primitive — `jit_pad`, `jit_subtract`, `jit_concatenate` — and
the 482 entries are that dust, not 482 Jacobians.

## 14.2 Baseline: what the solve owned

Executing the scene program (the constraint solve is the last statement of
every one), cold cache. The **no-op** column reruns the same scene with
`satisfy_constraints` replaced by a parameter read, which is the honest
attribution: everything else the scene does is unchanged.

| scene | DOF | constraints | wall | XLA programs | of which the solve | wall, no-op solve |
|---|---:|---:|---:|---:|---:|---:|
| two spheres, no sketch (control) | 2 | 0 | 0.014 s | 5 | 1 | 0.013 s |
| `scenes/starter.py` | 71 | 19 | 1.475 s | 155 | **94 (61 %)** | 0.484 s |
| `scenes/motor_shield.py` | 77 | 33 | 3.879 s | 411 | **92 (22 %)** | 2.774 s |

**The solve was 0.98 s of `starter`'s 1.48 s (66 %) and 1.09 s of
`motor_shield`'s 3.88 s (28 %).** Warm (populated cache, second subprocess) it
was still 0.32 s and 0.35 s — because the residue is Python-side eager
dispatch, which no disk cache touches.

Note the control: a scene with no sketch compiles 5 programs total. The whole
482-entry cloud is sketch content, exactly as §13.5 said, and 94 of `starter`'s
155 were the eight lines of `_newton_projection`.

## 14.3 Lever 1 — `lax.scan`. The premise was false; the fix was elsewhere

The expectation was that a Python `for _ in range(steps)` with
`jax.jacobian` *inside* it compiles one Jacobian program per step. It does
not. Eager dispatch caches per **primitive op**, so every step after the first
hits the same `jit_dot`, `jit_pad`, `jit_concatenate` entries:

| `steps` | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| solve XLA programs | 94 | 94 | 94 | 94 | 94 |
| solve wall | 0.996 s | 1.116 s | 1.162 s | 1.385 s | 1.804 s |

Flat in programs, linear in wall clock at ~50 ms of Python dispatch per step.

What *does* pay is putting the loop inside **one** `jax.jit`. Measured on the
real residual of each scene, `steps=2`:

| variant | `starter` programs | `starter` compile | `motor_shield` programs | `motor_shield` compile |
|---|---:|---:|---:|---:|
| eager (as shipped) | 85 | 0.725 s | 83 | 0.822 s |
| jit, loop unrolled in Python | 2 | 0.101 s | 2 | 0.163 s |
| **jit + `lax.scan`** | **1** | **0.088 s** | **1** | **0.139 s** |

And the scan is what makes the step count free, which is the trade the
question asked about. At `steps=16` on `starter` the unrolled jit costs
0.367 s to compile and the scanned one costs **0.081 s** — the same as at
`steps=2`, because `lax.scan` traces its body once whatever `length` is.

## 14.4 Lever 2 — matrix-free. Measured, and it loses

Conjugate gradient on `J Jᵀ λ = c` through `jvp`/`vjp` products, damped
(`λ = 1e-10 I`), 20 CG iterations, never forming `J`:

| problem | jit + scan (dense `lstsq`) | matrix-free CG |
|---|---:|---:|
| `starter` (J is 22 × 71) | 0.088 s | 0.122 s |
| `motor_shield` (37 × 77) | 0.139 s | 0.184 s |
| synthetic, 128 points / 171 constraints | 0.971 s | 2.012 s |

It loses at every size, and it loses *worse* as the problem grows. The reason
is the thing lever 2 was aimed at: **`J`'s density was never the cost.** A
22 × 71 dense least-squares is nothing. The cost is the *residual*, which is a
Python loop concatenating one traced sub-expression per constraint — and the
matrix-free body embeds a jvp and a vjp of that whole unrolled residual inside
20 CG iterations, so it compiles a multiple of what the dense route compiles.
Rejected, and it would also have had to answer for its own conditioning in the
rank-deficient case that §14.7 protects.

## 14.5 Lever 3 — batching identical constraints. The real *scaling* lever, not yet needed

`build_residual_fn` traces every constraint separately: sixteen horizontal
constraints are sixteen copies of `p1[1] - p2[1]` in the jaxpr. Holding the
solver fixed at jit + scan and varying only the residual — the shipped
per-constraint loop against a hand-written gathered/batched equivalent of the
same system:

| points | constraints | looped compile | batched compile | looped StableHLO | batched StableHLO |
|---:|---:|---:|---:|---:|---:|
| 8 | 11 | 0.065 s | 0.058 s | 50 kB | 51 kB |
| 32 | 43 | 0.179 s | 0.056 s | 171 kB | 52 kB |
| 128 | 171 | 0.782 s | 0.087 s | 666 kB | 54 kB |
| 512 | 683 | **5.391 s** | **0.066 s** | 2.68 MB | 63 kB |

**82× at 512 constraints, and the batched program is flat.** This is the lever
that scales, and it is the right next one — but it is worth ~0.12 s at the
sizes real scenes have today (19 and 33 constraints), against a change that
touches the `compute_residual` contract of all twelve constraint types. It is
written down here rather than built. The design that would do it generically:
group constraints by `(type, parameter arity, parameter shapes, and equality of
every non-`Parameter` dataclass field)`, gather each group's parameter slices
into stacked arrays, and `vmap` one group member's `compute_residual` over
them. The equal-constants clause is what makes it safe without per-type code,
and it is satisfied by exactly the types that dominate a sketch — Horizontal,
Vertical, Coincident, Parallel/Perpendicular, ParallelEdges/PerpendicularEdges
carry no constants at all. Distance, Fixed and Angle fall back to the loop
unless their targets happen to agree. **Costs**: residual ordering changes, so
geometry moves at solver tolerance rather than staying bit-identical.

## 14.6 Lever 4 — approximating the Jacobian. It buys nothing here

A chord/quasi-Newton step (factor `J` once at the initial point, reuse it for
every subsequent correction), scanned and jitted like the others:

| variant | `starter` programs | compile | `motor_shield` programs | compile |
|---|---:|---:|---:|---:|
| jit + scan, exact `J` per step | 1 | 0.088 s | 1 | 0.139 s |
| jit + scan, `J` frozen at step 0 | 1 | 0.077 s | 1 | 0.135 s |

**7 ms and 4 ms**, inside the noise, and it is the weakest lever for exactly
the reason predicted: an approximate Jacobian saves *evaluations*, and this
workload's cost is *compilation*, which the first Jacobian pays in full either
way. It also costs convergence — the chord method is linearly rather than
quadratically convergent, so a sketch far off-manifold needs more steps to
reach the same residual. Rejected: it trades a real property for a
measurement-noise gain.

**So the answer to "approximate the Jacobian in favour of more steps?" is: no,
and you no longer have to.** Steps became free by construction; the exact
Jacobian is what should be spent on them.

## 14.7 What shipped

`_newton_projection` and `_gradient_projection` are one `jax.jit` around a
`lax.scan` over their step body. The arithmetic is untouched: the same
residual, the same `jax.jacobian`, the same `jnp.linalg.lstsq(J Jᵀ, c)` —
which is load-bearing and is *why* the dense route was kept, since it is the
one that answers a rank-deficient `J Jᵀ` with a minimum-norm correction
instead of NaN. Two smaller things came with it: the loss history returns
through one `np.asarray` instead of iterating the device array (which was
compiling an `unstack` program of its own), and `project_to_manifold` calls
the scanned kernel directly so it does not sync the host every step inside an
optimizer loop.

Cold cache, fresh subprocess, whole scene program:

| scene | wall, before → after | XLA programs, before → after | the solve's own wall | the solve's own programs |
|---|---|---|---|---|
| two spheres (control) | 0.014 → 0.013 s | 5 → 5 | 0.002 → 0.002 s | 1 → 1 |
| `starter` | **1.475 → 0.770 s** | **155 → 76** | **0.978 → 0.278 s** | **94 → 15** |
| `motor_shield` | **3.879 → 3.065 s** | **411 → 333** | **1.087 → 0.321 s** | **92 → 14** |

Warm cache: `starter` 0.453 → 0.291 s and its solve 0.320 → 0.158 s;
`motor_shield` 1.156 → 0.952 s and its solve 0.347 → 0.177 s.

The solve is now **one** program. The 14–15 still attributed to it are
`pack_param_dict` / `unpack_param_vector` / `apply_parameters` dispatching
eagerly — `jit_concatenate`, `jit_dynamic_slice`, `jit_squeeze` — and those
same op-level entries are compiled by the rest of the scene anyway, so jitting
them would move cost rather than remove it. Measured directly: the scanned
kernel is 1 program and 0.073 s; packing is 9 and 0.026 s; unpacking is 5 and
0.023 s.

And the step count is now free:

| `steps` | 2 | 8 | 16 | 64 |
|---|---:|---:|---:|---:|
| solve XLA programs | 15 | 15 | 15 | 15 |
| solve compile | 0.122 s | 0.119 s | 0.121 s | 0.121 s |
| solve wall | 0.268 s | 0.265 s | 0.266 s | 0.268 s |

**`steps=64` and `steps=2` are the same number to three decimal places.**
Before the change, going from 1 step to 16 cost 0.81 s more.

### Geometry

**Bit-identical.** SHA-256 over the solved free-parameter arrays of both
scenes, before and after: `starter` `79340fde…`, `motor_shield` `86dedf5e…`,
unchanged. Nothing else was expected — the change is where the loop lives, not
what it computes — and the matrix-free route, which *would* have moved the
answer, was rejected on its own merits above.

### Tests

`tests/constraints/test_compiled_programs.py` pins the shape: a Newton solve
compiles exactly **one** program, at `steps` of 1, 2 and 32; an Adam solve
likewise at 4 and 64; a whole `satisfy_constraints` call stays at one program
with its eager ops warm; and — as a proxy that survives jax changing its log
format — the residual function is *traced* the same number of times at
`steps=2` and `steps=16`, which a Python loop cannot do. Counting uses the
public `jax.log_compiles()` context and a log handler, not a private hook.

`tests/constraints/test_redundant.py` gains structural rank deficiency next to
the geometric redundancy it already covered: a literally duplicated relation
(two identical rows in `J`) and a degenerate one (`Horizontal(p, p)` — a row
that is identically zero), each in float32 *and* float64, each asserted finite
*and* still enforced, and one asserting that stating a relation twice gives
the same sketch as stating it once. Swapping `lstsq` back to
`jnp.linalg.solve` fails 12 of the file's 14 tests.

## 14.8 Which options to expose — and which not to

The step count is the one that changed character. Before, `steps=8` cost 0.3 s
more than `steps=2` on `starter`; now it costs nothing, so
`satisfy_constraints(scene, steps=…)` — already in the scene language, already
in `satisfy_constraints`' signature — is a knob a user can turn freely, and
raising the default is a question of convergence rather than of budget. Both
shipped scenes reach machine-zero residual in 2 steps (`starter` 0.0,
`motor_shield` 5.5e-18), so nothing argues for moving the default today.

`method` (`"newton"` / `"adam"` / `"sgd"`) is already exposed and all three
paths are now scanned, so the Optax paths — the ones users are *expected* to
give 48 steps — got the same treatment.

**Nothing new should be exposed.** Specifically not:

- **A solver method for matrix-free or quasi-Newton.** Neither survived §14.4
  or §14.6. Exposing a method that is never the right choice is a knob whose
  effect has been measured to be negative.
- **A Newton tolerance.** The Newton path has no convergence loop — it is a
  fixed count of projections — and adding one would mean `lax.while_loop`,
  which reintroduces data-dependent iteration into a program the whole point
  of this section was to make one fixed shape. No measurement asks for it.
  `solve_constraints` (the Levenberg-Marquardt path) already has `tol` and
  `max_steps` and keeps them.
- **`lstsq`'s `rcond`.** It decides which singular values count as zero, i.e.
  how much redundancy is tolerated before the correction is truncated. That is
  a real dial with a real effect, and no scene has yet needed a value other
  than the default. Exposing it would be shipping a way to reintroduce §14.7's
  NaN.

For the patch vocabulary in `research/editing-operations.md`: `steps` is the
only constraint-solver control worth a GUI, it is an integer ≥ 1, and its
effect is monotone and now free. **No payload change is proposed here** — the
viewer already carries `satisfy_constraints`' diagnostics through
`capture_constraint_solves`, whose report (`method`, `iterations`, `losses`)
is unchanged in shape and length; a step slider would need only that existing
`losses` array, which is `steps + 1` long as before.

## 14.9 Reproducing §14

Scripts in the ephemeral scratch workspace, `solve-` prefixed:

| script | produces |
|---|---|
| `solve-count.py <scene>` | §14.2, §14.7 — per-scene wall/program split, `NOOP_CONSTRAINTS=1` for the attribution column |
| `solve-anatomy.py <scene>` | §14.1 — constraint census and the per-phase program count inside one solve |
| `solve-proto.py <scene>` | §14.3, §14.4, §14.6 — the five-variant shootout on the real residual (`STEPS`, `ONLY`, `CG` env) |
| `solve-scale.py` | §14.4 — the synthetic size sweep (`NPTS`, `STEPS`) |
| `solve-batch.py` | §14.5 — looped vs hand-batched residual under a fixed solver |
| `solve-geom.py <scene>` | §14.7 — the solved-parameter hash |
| `solve-two-spheres.py` | the no-sketch control scene |

Programs are counted by wrapping `jax._src.compiler.compile_or_get_cached`;
the tests use the public `jax.log_compiles()` instead, and the two agree.
