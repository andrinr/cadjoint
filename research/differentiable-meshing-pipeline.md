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
3. **Mesh generation from Hermite data (done).**
   `jaxcad.meshing.dual_contouring`: one vertex per active cell, with two
   placements sharing one contract. The differentiable path is a
   Tikhonov-regularized QEF — batched linear algebra with no
   SVD/eigendecomposition in the gradient path, so planar cells cannot NaN
   it. The concrete forward path (`sharp_qef_vertices`, default in
   `extract_mesh`) solves the unregularized QEF with a rank-revealing
   pseudo-inverse: planar cells project onto their face, crease cells land
   exactly on the crease, corners land exactly on the corner (box corner
   error 0.0; cylinder rim vertices sit on the cap plane to the last bit).
   The uniform-grid wiggle along feature curves was precisely the per-cell
   Tikhonov bias, which varies with how the grid slices the surface.
   Connectivity is deterministic, wound by the frozen `start_inside`
   orientation. Measured: sphere/box/CSG-union meshes are watertight
   manifolds (Euler 2), and the corner vertex's Jacobian with respect to
   the box half-extents is the identity to 0.2% — full tangential motion,
   which a normal-only backward pass structurally cannot produce.
4. **Mesh quality diagnostics and benchmarks (partial).** Watertightness,
   Euler characteristic, signed volume, corner error, and triangle minimum
   angles are tested and benchmarked; Hausdorff sampling and
   self-intersection checks remain open.
5. **Adaptivity (partial).** `jaxcad.meshing.adaptive` descends an octree
   over the cell lattice, discarding blocks where
   `|f(center) - level| > half_diagonal × L`, then evaluates only the
   surviving cells' corners. The octree adapts the *search*, not the mesh:
   leaves stay at uniform depth, so the edge set is bit-identical to dense
   detection (tested as exact equality) and every downstream stage works
   unchanged. Evaluations drop to 6–15% of the dense lattice at
   resolutions 64–96; wall-clock at viewer scales is dominated by the
   per-call JAX trace/compile of the scene, so the viewer keeps the dense
   default (a wrong user-supplied Lipschitz bound would punch holes),
   while `extract_mesh(..., lipschitz=...)` is the path for high-res
   export and controlled fields. Multi-size leaf cells — 2:1 balancing,
   transition stitching, manifold-preserving clustering — remain open.
6. **Viewer integration (working).** The compile payload carries a `wire`
   layer (the mesh's native quad edges) and a `sharp` layer that is *not*
   mesh edges: feature curves cross grid cells diagonally, so mesh edges
   trace a staircase around them. Instead, feature cells — normal-spread
   creases/corners plus exact `min`/`max` CSG seam cells — are linked to
   their lattice neighbors (`feature_cell_links`), and since feature-aware
   placement puts their vertices exactly on the feature curve, the links
   are chords of the true curve. Three rules keep the chains honest: cells
   are grouped by feature identity (owning operand for creases, operand
   pair for seams) so nearby distinct curves never cross-link into
   X-lattices; links that shortcut around a corner cell are dropped; and
   seam vertices are Newton-projected onto the exact intersection curve
   `f_a = f_b = 0` of their two owning operands, with a transversality
   guard for tangent/coincident surfaces. Two display switches ("Feature
   edges", "Mesh wireframe") draw the layers with a reduced depth nudge so
   coincident construction lines win. Primitives are themselves min/max
   compositions internally (a box is a max over axis distances, an extrude
   is max(profile, axial)), so per-primitive patch fields can replace the
   spread heuristic for known trees next; that needs local-frame plumbing
   through transforms. Surface handles and drag solve come later.

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

## CAD-system increment (2026-08-16)

Constraints, sketch editing, and construction operators, built against one
shared contract:

- **Constraints**: horizontal, vertical, coincident, equal-length, and
  point-on-line join the existing fixed/distance/angle/parallel/
  perpendicular set — all plain differentiable residuals through the same
  Levenberg–Marquardt (`solve_constraints`, exact-DOF) and projection
  (`satisfy_constraints`, under-constrained-tolerant) paths.
- **Sketch editing**: the viewer payload now carries every constraint kind
  with a stable per-profile `index`; new patch operations
  `delete_constraint`, `set_constraint_value`, and `add_revolution` extend
  the source-surgery layer, and the sketch panel gains removable constraint
  chips, click-to-edit distance values, vertex-pair and edge-pair constraint
  tools, and a Revolve operator button.
- **Construction**: `loft(profile_a, profile_b, height)` interpolates two
  equal-count vertex loops along the sketch normal (per-slice-exact polygon
  distance, documented as a bound in 3D); `extrude` gains `draft` and
  `twist` (twist documented non-1-Lipschitz); `jaxcad.sdf.operations` adds
  shell, offset, mirror, and linear/polar patterns. Everything traces to
  WGSL for the viewer and differentiates with respect to profile vertices.

## STEP validation against a real CAD kernel (2026-08-16)

`save_step` was "best-effort, not yet validated"; it is now validated
against OCCT. `cadquery-ocp` 7.9.3.1.1 (the first candidate tried)
installed cleanly on this macOS arm64 / CPython 3.14 venv via uv, imported,
and read a trivial hand-written AP214 file, so build123d and pythonocc-core
were never needed. It lives behind a dev-only `stepcheck` extra with
`python_version >= '3.10' and python_version < '3.15'` markers (the wheel
range), and `tests/meshing/test_step_kernel.py` importorskips `OCP`.

Round-trip results (read → `TransferRoots` → `BRepCheck_Analyzer` →
`BRepGProp` volume), per exported mesh:

- **Box and sphere passed unmodified**: one `BRepCheck`-valid closed
  `SOLID`, face counts exactly matching `merge_planar_faces` (6 for the
  box), kernel volume equal to the mesh's signed volume.
- **One real bug, found only by the union mesh**: sharp QEF placement can
  land two *adjacent* cells' vertices on the same crease point (observed
  distance ~3e-17), so the exported loops contained zero-length edges. OCCT
  cannot build a `LINE` from a zero-magnitude `VECTOR`
  (`Make Geom_Curve (3D) failed` → `wire not done`), silently dropped the
  four affected faces' wires, and demoted the `MANIFOLD_SOLID_BREP` to a
  compound of five open shells with garbage volume. Fix in `export.py`:
  `_weld_degenerate_edges` union-finds the endpoints of any loop edge
  shorter than the file's own declared `UNCERTAINTY` (1e-7), collapses
  consecutive duplicates, and drops loops left with fewer than three
  vertices. The union now reads back as one valid closed solid; the four
  dropped faces are exactly the triangles that collapsed at the two welds,
  and volume agrees to well under 1%.
- **Units**: the file declares `SI_UNIT($,.METRE.)`; OCCT defaults to a
  millimetre working unit, scaling volumes by `1000**3` on import. Not a
  file bug — the tests set `xstep.cascade.unit` to `M` (Interface statics
  initialize only after a reader is constructed) and compare volumes
  directly; `save_step`'s docstring now documents the metre convention.
