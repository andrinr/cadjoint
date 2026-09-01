# Differentiable meshing pipeline — design ledger

> Trimmed to the final architecture. Full build history (stage-by-stage logs,
> the adversarial-review fixes, the STEP/OCCT bug hunt) is in git/PR #19.

Started 2026-08-15 on `feature/differentiable-meshing-pipeline`. The previous
spike (archived on `feature/adaptive-sdf-meshing`) attached a MeshSDF-style
implicit gradient *post-hoc* to vertices coming out of a black-box NumPy
extractor. That bridge only recovers normal-direction motion and never
differentiates the construction that actually places vertices. This pipeline
replaces it: build meshing bottom-up from the inputs in pure JAX, and
differentiate each real construction step.

## Principles

- **Discrete topology, continuous motion.** Which grid edges cross the
  surface, which cells are active, and how cells connect are discrete choices,
  frozen per extraction. Everything continuous — crossing positions, normals,
  QEF vertices — carries exact JAX derivatives with respect to design
  parameters. Optimization loops re-extract between steps when topology may
  change.
- **Implicit differentiation, not unrolling.** Iterative solvers run on
  `stop_gradient` values; a final differentiable Newton correction
  `t = t0 - f(x(t0)) / (df/dt)` re-attaches the gradient. At a converged root
  its derivative is exactly the implicit-function-theorem value
  `dt/dθ = -(∂f/∂θ) / (∂f/∂t)`, so gradients are exact regardless of how the
  root was found.
- **Black-box fields.** Stages sample only the field callable and
  `jax.grad` of it, so primitives, CSG, transforms, and user-written fields
  all take the same path. Structure-aware extras (CSG branch tracking,
  per-primitive patch fields) are optional layers, never requirements.
- **min/max is where sharpness lives.** Hard CSG enters the field through
  `jnp.minimum`/`maximum`. Root finding must therefore be kink-robust
  (bisection brackets by sign and cannot be fooled by piecewise-smooth
  fields); Hermite normals at seams are one-sided subgradients, which is
  exactly what dual contouring needs to reconstruct a crease; and the active
  branch switching *on the surface* is an exact, threshold-free crease signal.

## Final architecture (module → tests)

| Stage | Module | Tests |
|---|---|---|
| Hermite edge detection (bisection + secant + IFT Newton polish, frozen `start_inside`) | `cadjoint/meshing/edge_detection.py` | `tests/meshing/test_edge_detection.py` |
| Sharp-feature classification (normal-spread SVD + exact min/max seam cells) | `cadjoint/meshing/features.py` | `tests/meshing/test_features.py` |
| Dual contouring (Tikhonov QEF gradient path; rank-revealing sharp forward path; deterministic winding) | `cadjoint/meshing/dual_contouring.py` | `tests/meshing/test_dual_contouring.py`, `test_manifold_cells.py` |
| Octree-pruned detection (bit-identical to dense; `lipschitz` is the caller's contract) | `cadjoint/meshing/adaptive.py` | `tests/meshing/test_adaptive.py` |
| Per-primitive patch fields (exact feature-edge signatures for known trees) | `cadjoint/meshing/patch_fields.py` | `tests/meshing/test_patch_fields.py` |
| Simplification (half-edge collapse under QEF + SDF error bound, features pinned bitwise) | `cadjoint/meshing/simplify.py` | `tests/meshing/test_simplify.py` |
| Export (planar-patch merge, OBJ n-gons, binary STL, STEP AP214 validated against OCCT) | `cadjoint/meshing/export.py` | `tests/meshing/test_export.py`, `tests/meshing/test_step_kernel.py` |
| Native (Rust) kernels behind a Tesseract | `cadjoint/meshing/native.py`, `native/` | `tests/meshing/test_native_mesher.py` — see `research/native-mesher.md` |
| CSG stress scenes + viewer edge view | — | `tests/meshing/test_scenes.py`, `tests/viewer/test_edge_artifacts.py` |

Known structural limitation (strict xfail in `test_scenes.py`): when a CSG seam
grazes a lattice plane, uniform DC can emit nonmanifold edges — one QEF vertex
per cell cannot represent two sheets crossing one cell face; wants manifold
DC / cell disambiguation. Multi-size octree leaves (2:1 balancing, transition
stitching) also remain open.

## Benchmark policy

Every stage tracks four dimensions from day one, in `benchmarks/`:

- **Geometric fidelity** — analytic-shape error (roots, corners, Hausdorff
  distance) with pytest-enforced tolerances.
- **Gradient correctness** — autodiff against analytic implicit derivatives
  and central finite differences, including near creases.
- **Mesh quality/topology** — manifoldness, watertightness, triangle quality.
- **Performance** — wall-clock and scaling vs resolution as a runnable
  script; timing lives in benchmarks, not in CI-gating tests.
