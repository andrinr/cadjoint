# Native (Rust) dual-contouring core — retired

**Status: removed.** The Rust core existed on this branch from its port
until commit `d42d800`; `native/` (a rayon-parallel cdylib plus a
tesseract wrapping its QEF), `cadjoint/meshing/native.py`,
`tests/meshing/test_native_mesher.py` and
`benchmarks/native_mesher_bench.py` are gone, and the reference
Python/JAX/NumPy pipeline in `cadjoint/meshing` is now the only pipeline.
This note keeps what it measured and why it was not worth keeping.

## What it was

An array-in/array-out split of `cadjoint/meshing`. The SDF-evaluating
stages stayed in JAX — `sample_grid`, and `edge_hermite_data`'s bisection
plus differentiable Newton correction, which owns the exactness contract.
The discrete and small-dense-linear-algebra stages moved to Rust behind
ctypes: crossing detection, manifold cell incidence, the rank-escalating
sharp QEF, the Tikhonov QEF with a hand-derived linear-solve VJP, and the
dual face build. `cadjoint/meshing/native.py` exposed them as drop-ins
(`extract_mesh_native`, `qef_vertices_native`, and one function per
stage) with bit-identical discrete output and float64-accurate continuous
output; the differentiable Tikhonov QEF crossed into Rust through a
tesseract so `jax.grad` reached the hand-derived VJP.

It worked, and it was correct: 20 tests asserted per-stage bit-identity
on box, box∪sphere, cylinder and bracket, vertex/normal parity to 1e-6
(measured ≤ 2e-15 in f64), VJP agreement with JAX autodiff to 4e-14, and
a design-parameter gradient matching the reference chain to 1e-9 and
central differences to 5e-4.

## What it measured

Per-stage, on the 65³ lattice of `starter@d42d800` (1 506 crossing edges,
1 514 cells) — the numbers that decided its fate:

| stage | NumPy reference | Rust |
|---|---:|---:|
| `find_crossing_edges` | 1.70 ms | 1.15 ms |
| `manifold_cell_incidence` | 0.79 ms | 0.39 ms |
| `sharp_qef_vertices` | 4.18 ms | 0.37 ms |
| `dual_faces` | 0.68 ms | 0.06 ms |
| **whole discrete pipeline** | **7.36 ms** | **1.97 ms** |

The isolated speedups are real and large — up to 11× on the batched QEF
solves at this size, and 18–107× at resolutions 32–128 (the old
`benchmarks/native_mesher_bench.py compare` tables). They are also
irrelevant at the scale the application runs at.

## Why it was retired

**The whole thing saved ~5 ms.** In the same measurement the two stages
that *cannot* leave JAX cost `sample_grid` 334 ms and `edge_hermite_data`
1 601 ms, and one warm `mesh` request for the starter scene is 3.9–5.7 s.
`research/performance.md` §6.4 reached the same conclusion from the other
direction: a `tottime` profile of the warm `mesh` path shows no cadjoint
function in the top 30 — the time is JAX re-tracing and re-dispatching the
scene in eager mode, and XLA compiling shape-specific programs. The Rust
core was 0.13 % of a request.

Measured end to end after deletion, `_mesh_edge_payload` on the starter
scene (which drove the native path when the cdylib was built) went from
4 209 ms to 3 885 ms best-of-5 — i.e. the difference is inside the noise,
and the overlay's output is unchanged (3 012 wire and 453 sharp
segments either way).

**What it cost.** A Rust toolchain in the build path, a 1.39 GB container
image whose build compiles the cdylib, a second implementation of five
algorithms to keep bit-identical to the first, a `CADJOINT_NATIVE_MESHER`
environment variable, an `ImportError`-with-build-hint code path on every
entry point, a branch in `_edge_overlay.py` selecting a backend, a
per-stage parity suite, and a `qef_native` plugin registered under a `qef`
kind — a plugin boundary around a 2 ms in-process kernel, which is the
wrong shape for the abstraction. None of that is a fair trade for 5 ms.

## What is kept from it

Nothing in code. The reference pipeline was never a fallback: it is the
implementation the Rust core was written to match, and every parity test
was written as reference-first. Two lessons survive:

- **Port only what the profile says dominates a request**, not what
  dominates a micro-benchmark. Every stage moved here was genuinely
  25–78 % of an *extraction* at resolution 128; none was a measurable
  fraction of a *request*.
- **The JAX boundary is the real cost**, and it is where the remaining
  levers are — `jax.jit` on the traceable blocks and a worker process
  that outlives one request (`research/performance.md` §6.1, §6.5),
  measured at 100–1000× on exactly the stages Rust could not touch.
