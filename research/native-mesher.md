# Native (Rust) dual-contouring core

Rewrite of the numeric heart of `cadjoint/meshing` in Rust, integrated as a
cdylib behind a tesseract-jax endpoint. This document records the profile
that justified the split, the design, and the measured speedups.

## 1. Profile of the Python/JAX reference pipeline

Machine: Apple Silicon (darwin arm64), CPU jax 0.8.2, float32 pipeline.
`benchmarks/native_mesher_bench.py profile --resolutions 32 64 128`.
Steady-state = min over 3 repeats; "first" includes jit trace + compile.

### Stage breakdown (ms)

| scene | res | edges | cells | sample 1st | sample | edges | incidence | hermite 1st | hermite | qef smooth 1st | qef smooth | qef sharp | faces | extract total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sphere | 32 | 2886 | 2888 | 12.7 | 10.0 | 0.3 | 1.5 | 370.3 | 28.9 | 554.4 | 1.4 | 7.7 | 1.2 | 51.2 |
| sphere | 64 | 11406 | 11408 | 13.8 | 10.7 | 1.7 | 6.4 | 369.8 | 28.4 | 443.5 | 3.1 | 29.9 | 4.7 | 86.1 |
| sphere | 128 | 45678 | 45680 | 26.5 | 16.7 | 12.5 | 25.7 | 397.0 | 29.9 | 449.1 | 9.1 | 118.5 | 20.2 | 228.5 |
| box | 32 | 1726 | 1728 | 19.5 | 15.7 | 0.2 | 0.9 | 590.5 | 37.4 | 455.3 | 1.4 | 3.8 | 0.7 | 62.8 |
| box | 64 | 6782 | 6784 | 20.5 | 17.9 | 1.6 | 3.4 | 604.3 | 37.6 | 441.6 | 3.2 | 14.4 | 2.8 | 81.2 |
| box | 128 | 26862 | 26864 | 30.9 | 24.2 | 12.7 | 14.5 | 676.7 | 42.3 | 524.8 | 5.6 | 56.8 | 11.2 | 169.0 |
| bracket | 32 | 3760 | 3752 | 121.1 | 116.9 | 0.3 | 2.0 | 2178.3 | 479.4 | 471.1 | 1.9 | 8.5 | 1.5 | 605.9 |
| bracket | 64 | 15179 | 15164 | 118.4 | 113.1 | 1.7 | 8.3 | 2135.6 | 477.2 | 497.2 | 3.8 | 34.2 | 6.2 | 647.6 |
| bracket | 128 | 61009 | 60980 | 141.2 | 122.8 | 12.7 | 33.6 | 2294.0 | 520.6 | 540.0 | 11.8 | 136.6 | 26.7 | 850.4 |

### Frozen-topology gradient (`edge_hermite_data` + `qef_vertices` under `jax.grad`, jitted)

| scene | res | first (trace+compile) | steady |
|---|---:|---:|---:|
| sphere | 128 | 218.4 | 13.6 |
| box | 128 | 288.5 | 8.8 |
| bracket | 128 | 2499.5 | 35.0 |

### Reading of the profile

- **JAX trace/compile dominates first-touch cost**, as expected: the hermite
  stage costs 0.4–2.3 s on first call and the frozen gradient 0.2–2.5 s to
  trace/compile — versus tens of ms to execute. Nothing native can fix that;
  it is the price of tracing the user's SDF and it amortizes across an
  optimization loop.
- **The SDF-evaluating stages must stay in JAX** (the SDF is a JAX function):
  `sample_grid` and `edge_hermite_data`. For an expensive CSG field
  (bracket) they dominate steady state too (123 + 521 of 850 ms at res 128).
- **The movable numeric stages grow fast and are worth moving**: at res 128
  the host-side discrete + linear-algebra stages (crossing detection,
  manifold incidence, sharp QEF SVD, dual faces) sum to
  177 ms / 228 ms = 78 % (sphere), 95 ms / 169 ms = 56 % (box),
  210 ms / 850 ms = 25 % (bracket) of one full extraction. The sharp QEF
  (batched 12x3 SVDs in NumPy) is the single largest movable stage
  (57–137 ms at res 128), followed by manifold incidence (15–34 ms),
  crossing detection (a dense 4·(n+1)^3 sweep; ~13 ms at 128) and the dual
  face build (11–27 ms).

## 2. Design: what moved to Rust, what stayed in JAX

Array-in/array-out split; the exactness contract of the reference pipeline
(frozen discrete topology, exact continuous gradients) is untouched.

**Stays in JAX (gradient contract owner):**

- `sample_grid` — batched SDF evaluation on the lattice (the SDF is JAX).
- `edge_hermite_data` — bisection on `stop_gradient` values + one
  differentiable Newton correction against the true SDF; carries the exact
  implicit-function-theorem derivatives, verbatim from the reference.
- Unit normalization of Hermite gradients, the averaged vertex normals, and
  the final per-cell clamp of the smooth QEF vertices (`jnp.clip`), so their
  subgradient semantics stay bit-compatible with the reference.
- Octree pruning (`lipschitz=`): the octree loop interleaves SDF
  evaluations, so detection stays on the existing
  `adaptive.sparse_crossing_edges`; everything after it goes native.

**Moved to Rust (`native/`, cdylib `cadjoint_native_mesher`, rayon-parallel):**

- `dc_find_crossing_edges` — dense lattice sweep for sign changes; ordered
  exactly like `find_crossing_edges` (axis-major, then row-major index).
- `dc_manifold_cell_incidence` — candidate (cell, edge, inside-corner)
  generation, inside-corner connected components (min-label, identical to
  the Python fixed point), stable grouping by (cell key, component); rows
  and slot order bit-identical to `manifold_cell_incidence`.
- `dc_sharp_qef` — rank-escalating truncated-SVD QEF (Ju et al. style)
  via Jacobi eigendecomposition of the 3x3 Gram matrix; forward-only,
  matching `sharp_qef_vertices` semantics (masked slots, mass point over
  valid slots, per-rank clamped candidates, smallest-residual winner).
- `dc_qef_tikhonov` + `dc_qef_tikhonov_vjp` — the smooth, differentiable
  QEF: mass point, `A = sum(n n^T) + lambda*count*I`, 3x3 solve; and its
  hand-derived VJP w.r.t. the Hermite points and unit normals
  (linear-solve differentiation: `u = A^{-1} vbar`, `bbar = u`,
  `Abar = -u x^T`, chained through the gather/mask/mass-point structure).
- `dc_dual_faces` — quad emission per interior crossing edge with the
  frozen `start_inside` winding, (cell, edge)-keyed row lookup, and
  shorter-diagonal triangulation; identical to `dual_faces`.

**Integration (tesseract-jax):** the differentiable smooth QEF is packaged
as a tesseract (`native/tesseract_api.py`) with typed `apply`,
`abstract_eval`, and `vector_jacobian_product` endpoints wrapping the
cdylib via ctypes, mirroring `cadjoint/fem/tesseracts/*`. It is loaded
locally with `Tesseract.from_tesseract_api` and composed into JAX autodiff
with `tesseract_jax.apply_tesseract`, so `jax.grad` through
`qef_vertices_native` dispatches to the Rust VJP. The discrete stages are
concrete host-side calls (plain ctypes), exactly as the reference's NumPy
stages are concrete host-side calls.

`cadjoint/meshing/native.py` exposes `extract_mesh_native(...)` with the
same signature and result type as `extract_mesh`, plus
`qef_vertices_native(...)` as the differentiable drop-in for
`qef_vertices` in frozen-topology optimization loops. When the cdylib is
missing, importing helpers raise an actionable error naming the
`cargo build --release` command.

## 3. Results

Same machine and scenes as section 1; steady-state execute time (min over
5 repeats, ms), `benchmarks/native_mesher_bench.py compare`.

### Stage-by-stage, native vs Python

| scene | res | edges | edges py→nat | incidence py→nat | sharp QEF py→nat | faces py→nat | extract_mesh py→nat |
|---|---:|---:|---:|---:|---:|---:|---:|
| sphere | 32 | 2886 | 0.25 → 0.13 (1.9x) | 1.48 → 0.55 (2.7x) | 7.55 → 0.43 (18x) | 1.19 → 0.29 (4.1x) | 51.6 → 41.8 (1.2x) |
| sphere | 64 | 11406 | 1.71 → 0.92 (1.9x) | 6.11 → 1.40 (4.4x) | 29.5 → 0.67 (44x) | 4.82 → 0.52 (9.3x) | 89.0 → 48.4 (1.8x) |
| sphere | 128 | 45678 | 12.3 → 2.02 (6.1x) | 26.0 → 3.96 (6.6x) | 117.8 → 1.21 (97x) | 20.1 → 1.20 (17x) | 228 → 60.9 (3.7x) |
| box | 32 | 1726 | 0.24 → 0.11 (2.2x) | 0.85 → 0.51 (1.7x) | 3.73 → 0.34 (11x) | 0.69 → 0.24 (2.9x) | 62.8 → 58.1 (1.1x) |
| box | 64 | 6782 | 1.63 → 0.91 (1.8x) | 3.45 → 0.91 (3.8x) | 14.2 → 0.39 (36x) | 2.75 → 0.42 (6.5x) | 80.7 → 61.5 (1.3x) |
| box | 128 | 26862 | 12.4 → 1.71 (7.3x) | 14.6 → 2.35 (6.2x) | 56.1 → 0.62 (90x) | 11.3 → 0.67 (17x) | 166 → 76.5 (2.2x) |
| bracket | 32 | 3760 | 0.25 → 0.13 (1.9x) | 1.85 → 0.77 (2.4x) | 8.79 → 0.38 (23x) | 1.51 → 0.32 (4.7x) | 632 → 599 (1.1x) |
| bracket | 64 | 15179 | 1.70 → 0.83 (2.0x) | 8.15 → 1.55 (5.3x) | 34.6 → 0.63 (55x) | 6.55 → 0.63 (10x) | 688 → 634 (1.1x) |
| bracket | 128 | 61009 | 12.7 → 1.84 (6.9x) | 33.6 → 5.17 (6.5x) | 137.7 → 1.29 (107x) | 27.2 → 1.30 (21x) | 866 → 676 (1.3x) |

Reading: every movable stage is faster at every size (the batched sharp
QEF by up to ~100x — NumPy's per-cell 12x3 LAPACK SVDs against
rayon-parallel Jacobi Gram eigensolves; detection/incidence/faces by
6–21x at res 128). End-to-end extraction speedup is bounded by the JAX
stages that must stay: 3.7x on the cheap sphere at 128, 2.2x on the box,
1.3x on the bracket whose CSG field makes `edge_hermite_data` (≈ 520 ms)
the dominant cost. The remaining Python-side steady-state cost at res 128
is ≥ 90 % SDF sampling + Hermite refinement.

### Frozen-topology gradient step (eager, x64, `grad` mode)

| scene | res | py first | py steady | native first | native steady |
|---|---:|---:|---:|---:|---:|
| sphere | 32 | 1626 | 54.5 | 577 | 55.3 |
| sphere | 64 | 1689 | 58.9 | 59.4 | 57.9 |
| sphere | 128 | 1639 | 75.2 | 69.0 | 66.5 |
| bracket | 32 | 4354 | 1013 | 1025 | 1015 |
| bracket | 64 | 4524 | 1057 | 1060 | 1060 |
| bracket | 128 | 4989 | 1122 | 1178 | 1165 |

The tesseract-backed QEF is opaque to tracing, so the reference's
QEF-stage trace/compile (the historic wall-clock killer: 1.6–5.0 s on
every new topology/shape) disappears — first-gradient latency drops up to
28x (sphere 64: 1689 → 59 ms; the sphere-32 native first call includes
the one-time tesseract load). Steady-state eager gradients are equal or
slightly better (the SDF autodiff dominates). The native path also
composes under `jax.jit` via tesseract-jax: jitted steady state is
identical to the jitted reference (4.2 ms at sphere 64) with gradients
matching to 1e-13 relative.

## 4. Validation

`tests/meshing/test_native_mesher.py` — 20 tests, all passing; the module
skips cleanly (reason names the cargo command) when the cdylib is absent,
and `extract_mesh_native` raises an ImportError-style message with the
build command. Coverage:

- (a) Topology: edges / incidence / faces bit-identical per stage on box,
  box∪sphere, cylinder, bracket at res 16; full `extract_mesh` parity at
  res 16 and 32 (faces, quads, cells, winding).
- (b) Vertex positions and normals within 1e-6 of the reference (measured
  ≤ 2e-15 in f64 for the sharp QEF, ≤ 1e-15 for the Tikhonov QEF).
- (c) VJP: gradients of a vertex objective w.r.t. Hermite points and
  gradients match reference JAX autodiff (measured ≤ 4e-14 relative,
  asserted ≤ 1e-6).
- (d) End-to-end: design-parameter gradient through `edge_hermite_data` +
  tesseract QEF equals the reference chain (rel 1e-9) and central finite
  differences (rel 5e-4).
- Octree-pruned (`lipschitz=`) path identical to dense; empty-surface and
  inconsistent-inside error parity; x64 requirement surfaced clearly.

## 5. Caveats

- **Tied quad diagonals**: the shorter-diagonal triangulation compares
  squared diagonals of float32-cast vertices. On exactly/near symmetric
  quads (gap below the f32 quantization noise, common on the bracket's
  coplanar regions) a one-ulp vertex difference between LAPACK SVD and the
  native Jacobi solver flips the choice — 7 of 11 250 quads on bracket/32,
  both triangulations valid and watertight. The parity test pins faces bit
  for bit except on such provably-tied quads.
- `qef_vertices_native` requires jax x64 (float64 tesseract schema) and
  raises a clear error otherwise; `extract_mesh_native` itself has no x64
  or tesseract dependency (concrete forward runs over ctypes).
- With `sharp=False` at default f32 precision, reference vertices carry
  f32 solve error (~1e-4 worst case for damped cells) while the native
  solve is f64 throughout; parity to 1e-6 holds under x64 (asserted), and
  the native result is the more accurate one at f32.
- The octree (`lipschitz`) detection stays on the Python `adaptive` module
  (it interleaves SDF evaluations); only the post-detection stages go
  native on that path.
- Rank selection in the sharp QEF near the `rcond` threshold and near
  repeated singular values is float-sensitive in both implementations;
  observed forward differences stay ≤ 2e-15 on all test scenes.
- Bench numbers are single-machine (Apple Silicon, CPU jax); rayon uses
  all cores, so speedups scale with core count for large grids.
