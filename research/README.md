# cadjoint research notes

Design records and living reference for the shipped system, plus what remains
open. Historical plan documents whose content is fully integrated were removed;
their full text is in git history (PR #19).

## Design records (shipped systems)

- [Differentiable meshing pipeline](./differentiable-meshing-pipeline.md) —
  design principles and final architecture of `cadjoint/meshing`, with
  module/test pointers.
- [Native (Rust) dual-contouring core](./native-mesher.md) — the profile that
  justified the Rust split, the JAX/Rust boundary design, measured speedups,
  and caveats.
- [FEM integration](./fem-integration.md) — the `SolverBackend` ABI, adjoint
  mechanics, and the ccx 2.23 sensitivity correction.
- [Tet vs hex meshing](./tet-vs-hex.md) — the mesher-Tesseract validation
  matrix and the surface-interpolation VJP contract.
- [End-to-end optimization](./end-to-end-optimization.md) — measured run record
  of the bracket showcase (`examples/fem_bracket_optimization.py`), box-bound
  rationale, and a known CalculiX deck fragility.
- [WebGPU SDF path tracing](./path-tracing.md) — design and deliberate
  boundaries of the browser path-tracing mode.

## Living roadmap material

- [Simulator ecosystem survey](./simulator-ecosystem.md) — 31 candidates
  ranked for the next solver integrations (CalculiX shipped; jwave and
  JAX-Fluids are next).
- [Constraint system](./constraints.md) — final state and the remaining
  constraint vocabulary (tangent, midpoint, concentric, …).

## Screenshots

`app/` and `edge-view/` hold the viewer screenshots embedded in PR #19
(feature-edge QA, simulate-panel parity, the four editing modes).
