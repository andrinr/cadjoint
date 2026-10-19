# cadjoint research notes

Design records and living reference for the shipped system, plus what remains
open. Historical plan documents whose content is fully integrated were removed;
their full text is in git history (PR #19).

## Design records (shipped systems)

- [Differentiable meshing pipeline](./differentiable-meshing-pipeline.md) —
  design principles and final architecture of `cadjoint/meshing`, with
  module/test pointers.
- [FEM integration](./fem-integration.md) — the `SolverBackend` ABI, adjoint
  mechanics, and the ccx 2.23 sensitivity correction.
- [Tet vs hex meshing](./tet-vs-hex.md) — the mesher-Tesseract validation
  matrix and the surface-interpolation VJP contract.
- [End-to-end optimization](./end-to-end-optimization.md) — measured run record
  of the bracket showcase (`examples/fem_bracket_optimization.py`), box-bound
  rationale, and a known CalculiX deck fragility.
- [WebGPU SDF path tracing](./path-tracing.md) — design and deliberate
  boundaries of the browser path-tracing mode.
- [Design language](./design-language.md) — the settled visual and
  interaction specification of the playground.
- [Editing operations](./editing-operations.md) — the spec of record for the
  28 source-rewriting patch operations.
- [Performance](./performance.md) — where the playground's seconds go, and
  the measured record of every speed-up taken or declined.
- [Flow solver](./flow-solver.md) — the differentiable flow solver and its
  conjugate heat-transfer coupling.
- [Complex scene](./complex-scene.md) and [complex scene 2](./complex-scene-2.md)
  — what modelling two real parts broke, and what was fixed.

## Living roadmap material

- [Simulator ecosystem survey](./simulator-ecosystem.md) — 31 candidates
  ranked for the next solver integrations (CalculiX shipped; jwave and
  JAX-Fluids are next).
- [Constraint system](./constraints.md) — final state and the remaining
  constraint vocabulary (tangent, midpoint, concentric, …).

## Experiments

- [`cutfem/`](./cutfem/README.md) — a cut-cell finite-element solve on an
  implicit model, differentiated in θ.
- [`reverse/`](./reverse/README.md) — reverse-engineering a scene from a STEP
  file and measuring how close it is.
- [`zeroset_test/`](./zeroset_test/README.md) — end-to-end harness for
  `zero-set.proto`.

## Screenshots

`app/` and `edge-view/` hold the viewer screenshots embedded in PR #19
(feature-edge QA, simulate-panel parity, the editing modes); `refactor/` the
before/after panels of the panel refactor, `complex-scene/` the end-cap
part, and `design/` the design-language captures.
