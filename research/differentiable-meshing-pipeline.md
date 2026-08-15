# Differentiable meshing pipeline

Started 2026-08-15 on `feature/differentiable-meshing-pipeline`.

The previous spike (archived on `feature/adaptive-sdf-meshing`) attached a
MeshSDF-style implicit gradient *post-hoc* to vertices coming out of a
black-box NumPy extractor. That bridge only recovers normal-direction motion
and never differentiates the construction that actually places vertices. This
pipeline replaces it: build meshing bottom-up from the inputs in pure JAX, and
differentiate each real construction step. Every stage lands with tests and
benchmarks before the next stage starts.

## Principles

- **Discrete topology, continuous motion.** Which grid edges cross the
  surface, which cells are active, and how cells connect are discrete choices,
  frozen per extraction. Everything continuous — crossing positions, normals,
  later QEF vertices — carries exact JAX derivatives with respect to design
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
  all take the same path. Structure-aware extras (CSG branch tracking) are
  optional layers, never requirements.
- **min/max is where sharpness lives.** Hard CSG enters the field through
  `jnp.minimum`/`maximum`. Root finding must therefore be kink-robust
  (bisection brackets by sign and cannot be fooled by piecewise-smooth
  fields); Hermite normals at seams are one-sided subgradients, which is
  exactly what dual contouring needs to reconstruct a crease; and the active
  branch switching *on the surface* is an exact, threshold-free crease signal.

## Stages

1. **Edge detection — grid crossings (done).**
   `jaxcad.meshing.edge_detection`: sample a lattice, find sign-changing
   edges host-side, refine each crossing with vectorized bisection + secant +
   differentiable Newton polish, and evaluate spatial gradients at the roots
   (Hermite data). Robustness rules an adversarial review forced in: the
   detection-time inside/outside orientation of each edge start is frozen
   (`start_inside`) so lattice vertices exactly on the surface — constant in
   CAD, where faces sit on round coordinates — cannot flip the bracket under
   float32 re-evaluation; edge endpoints are derived in float64 with the same
   expressions detection used, then rounded once, so refinement evaluates the
   field at bit-identical coordinates; and a Newton step is accepted only if
   it does not worsen the frozen residual, so tangency-degenerate edges
   (grids far from the origin, where crossings can be float cancellation
   noise) fall back to the bisection point instead of jumping a full edge.
   Measured on the unit sphere at resolution 26: max residual 6e-8, autodiff
   `dt/dr` matches the analytic implicit derivative to 4e-7 across all 1830
   edges; a sphere at (1000, 2000, -500) stays below 6e-5.
2. **Edge detection — sharp features (done).**
   `jaxcad.meshing.features`: per-cell classification into face / crease /
   corner via singular values of the incident unit normals (`σ2/σ1` ≈ tangent
   of half the normal fan angle), plus exact hard-CSG seam cells from
   min/max branch changes. Measures are differentiable; labels are frozen
   like the edge set.
3. **Mesh generation from Hermite data (next).** One QEF-placed vertex per
   active cell, dual quads around crossing edges, feature-aware placement
   using the stage-2 classification, orientation and triangulation. The QEF
   solve is differentiable linear algebra on stage-1 outputs, so vertex
   positions inherit exact parameter gradients by construction.
4. **Mesh quality diagnostics and benchmarks.** Watertightness, manifold
   edge incidence, component/Euler signature, Hausdorff/SDF error, triangle
   aspect histograms; sharp-feature placement error against analytic corners.
5. **Adaptivity.** Conservative cell pruning (Lipschitz bounds), octree with
   2:1 balancing, manifold-preserving clustering.
6. **Viewer integration.** Meshes in the compile payload, surface handles,
   drag solve — after the pipeline itself is trustworthy.

## Benchmark policy

Every stage tracks four dimensions from day one, in `benchmarks/`:

- **Geometric fidelity** — analytic-shape error (roots, later corners and
  Hausdorff distance) with pytest-enforced tolerances.
- **Gradient correctness** — autodiff against analytic implicit derivatives
  and central finite differences, including near creases.
- **Mesh quality/topology** — from stage 3 on: manifoldness, watertightness,
  triangle quality.
- **Performance** — wall-clock and scaling vs resolution as a runnable
  script; timing lives in benchmarks, not in CI-gating tests.
