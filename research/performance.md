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

`cadjoint/meshing/native.py` already binds a rayon-parallel cdylib
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
`/private/tmp/claude-501/-Users-andrinrehnann-code-jaxcad/a114cfb9-aa54-4491-aa84-413fbdf84e92/scratchpad/perf/`
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
