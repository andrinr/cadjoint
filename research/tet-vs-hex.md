# Tet vs hex for the FEM path — and a mesher-in-a-Tesseract with an interpolation VJP

Status: measured prototype (2026-09-01). Answers two user questions with working code
and numbers, on the bracket scene (`scenes/bracket.py` geometry via the parameterized
SDF mirror in `examples/fem_bracket_optimization.py`):

1. *"can't we go dual contouring and then to hex mesh? should we use another solver
   that supports tet meshes for this?"* — Routes 1 (DC → tet → tet FEM) and 2
   (DC-informed hex placement), both measured.
2. *"package the mesher into another tesseract and use an interpolation on the surface
   as the vjp map"* — Route 3, the headline: the **whole black-box mesher as a
   Tesseract** whose VJP w.r.t. the sampled implicit field is the
   implicit-function-theorem map built from trilinear interpolation weights at the
   frozen boundary vertices. Works, composes with the unmodified `elastic_jaxfem`
   solver tesseract inside one `jax.grad`, and the VJP is *exact* against autodiff
   of the underlying projection (mechanical check 0 rel err, end-to-end 1.4e-11).

Prototype code: `cadjoint/fem/tetmesh.py` (tet route),
`cadjoint/fem/tesseracts/mesher/tesseract_api.py` (mesher tesseract),
`tests/fem/test_tetmesh.py` (regression + demo). Nothing in the production
mesher/solver/backends was modified.

## TL;DR recommendation

- **Keep HEX8 voxelize+snap as the default FEM path.** At matched DOF it is the most
  accurate per DOF and per second of the three (see the compliance table), its
  element quality is far better (min scaled Jacobian 0.03–0.35 vs tet radius-ratio
  minima of 0.001–0.007), and its solve conditioning is what the existing iterative
  solvers were tuned on.
- **TET4 must not become a default: it is 20–45% overstiff at practical resolutions**
  (volumetric/bending locking) — its one virtue is geometric fidelity (boundary
  deviation 8–20x better than hex at matched cell size, since its boundary *is* the
  DC surface). TET10 fixes the stiffness (lands within ~2% of the hex reference) but
  costs 5–70x hex wall-clock at matched accuracy here and inherits sliver
  conditioning.
- **Where the tet route wins outright: gradient fidelity for features hexes
  under-resolve.** d(compliance)/d(rib_height) on the bracket: hex reports −0.020
  (res 48) / −0.0099 (res 30) while TET10 reports −0.128 and TET4 −0.076 — the
  staircased hex boundary simply does not see the inclined rib moving. If a design
  parameter moves geometry that the hex lattice aliases, the DC→tet path produces a
  usable gradient where the hex path produces a near-zero one.
- **Productionize Route 3's contract, not a solver switch:** the mesher-tesseract +
  interpolation-VJP makes *any* mesher differentiable w.r.t. the field samples
  without touching its internals — measured exact against autodiff of the projection.
  The honest caveat (measured, below): feeding the mesher *lattice samples* instead
  of the true SDF changes the gradient of crease-dominated parameters (a sign flip on
  `web_thickness` at 32x24x19), because the trilinear interpolant smears creases.
  The fix is a denser sampling lattice or a hybrid apply that also receives exact
  crease information — not a different VJP: the VJP is provably exact for the
  interpolant mesher it wraps.

## Install matrix (macOS arm64, CPython 3.14.5)

| package | result |
|---|---|
| `tetgen` 0.8.4 (PyPI wheel) | **installs and works** (`uv pip install tetgen`) |
| `wildmeshing` (fTetWild) 0.4.1 | **no cp314 wheel** (cp39–cp313 only) — unavailable here |
| `gmsh` 4.15.2 | already in the venv (fem extra); not needed since tetgen worked |

TetGen is driven through PLC mode with `-Y` (`nobisect`) + quality bounds
(`minratio 1.5`, `mindihedral 10`), so the DC surface triangulation is preserved
*verbatim*: the first `num_surface` output nodes are bit-identical to the input DC
vertices and every added Steiner vertex is strictly interior (asserted in
`surface_to_tet_mesh`, regression-tested).

## Route 1: DC surface → TetGen → jax-fem TET4/TET10

### What had to be true (and is, measured)

- **Project, then mesh.** DC/QEF vertices are *near* but not *on* the zero set
  (mean |sdf| 1.1e-3, max 2.0e-2 at 32x24x19). Newton-projecting them after
  tetrahedralization collapses boundary slivers (min tet volume 2.7e-22 → singular
  stiffness). `sdf_to_tet_mesh` therefore projects the DC vertices onto the zero set
  first and hands TetGen the projected surface, keeping the raw DC positions as the
  frozen Newton restart (`base_points`). Bonus: projection removed the coarse-grid
  self-intersections that made raw sharp DC surfaces un-meshable at 26x19x15.
- **Exact fixed point.** `recompute_tet_points(sdf, mesh)` at the nominal design
  reproduces `mesh.points` to max |dev| = 0.0 — the same frozen-topology contract as
  `hexmesh.recompute_points`.
- **jax-fem supports TET4 and TET10** (confirmed by running; `basis.py` element
  table, meshio `tetra`/`tetra10` ordering, `re_order` handled internally).
  `tet10_from_tet4` promotes meshes with shared midside nodes; midsides are
  (traced) corner midpoints, so gradients flow through them.
- **Node-membership face selection has a hole on tets** (found the hard way): at
  52x39x30, 212 *interior* faces had all three corners on the loaded surface patch —
  jax-fem's location-function rule selects them once per adjacent cell and
  double-loads faces that are not on the boundary. `tet_elastic_solve(...,
  traction_faces=...)` prunes the selection to exactly the requested boundary
  triangles and rebuilds the dependent assembly structures
  (`_restrict_traction_faces`). The hex path was checked clean on all meshes used
  (predicate faces == membership faces).
- **`Nodes` selections work on tet meshes unchanged** via duck typing
  (`num_points` / `points` / `all_boundary_faces` / `grid`) — `selection.py` was not
  touched.

### Mesh quality, geometric fidelity, accuracy, wall-clock (bracket, nominal design)

Load: bolt-ball clamps + traction on the outer web wall above z = 1.0, selected by a
mesh-independent center+normal rule on both mesh families. W = load work
= ∫ t·u dA (= classical compliance); Wn = W/area² normalizes away residual load-patch
area differences (areas differ ≤ 15% across meshes; exact wall patch ≈ 0.348).
Quality: hex = min scaled Jacobian; tet = min normalized radius ratio (3 r_in/r_circ;
regular tet = 1). dev = mean |sdf| at area-uniform boundary samples.

| mesh | res | cells | DOF | quality min | dev mean | W | **Wn** | t_mesh | t_solve |
|---|---|---|---|---|---|---|---|---|---|
| HEX8  | 24x17x13 | 742    | 4.3k  | 0.352 | 1.2e-2 | 0.1143 | 1.074 | 3.6 s | 3.2 s |
| TET4  | 26x19x15 | 5 143  | 4.5k  | 0.001 | 8.0e-4 | 0.0934 | 0.600 | 4.1 s | 3.1 s |
| TET10 | 26x19x15 | 5 143  | 28.3k | —     | 8.0e-4 | 0.1788 | 1.148 | —     | 32.9 s |
| HEX8  | 30x21x16 | 1 714  | 8.2k  | 0.028 | 5.3e-3 | 0.0996 | 1.053 | 4.2 s | 3.9 s |
| TET4  | 32x24x19 | 7 556  | 6.8k  | 0.001 | 6.3e-4 | 0.0777 | 0.586 | 5.1 s | 4.5 s |
| TET10 | 32x24x19 | 7 556  | 42.0k | —     | 6.3e-4 | 0.1442 | 1.088 | —     | 129 s |
| HEX8  | 48x34x26 | 7 460  | 30.1k | 0.071 | 2.2e-3 | 0.1598 | 1.123 | 5.0 s | 10.2 s |
| TET4  | 52x39x30 | 27 136 | 21.3k | 0.007 | 2.5e-4 | 0.1075 | 0.896 | 5.2 s | 8.8 s |
| TET10 | 52x39x30 | 27 136 | 139.5k| —     | 2.5e-4 | 0.1358 | 1.132 | —     | 654 s |
| HEX8 ref | 66x48x36 | 19 446 | 73.1k | — | — | 0.1284 | 1.112 | 4.4 s | 33.1 s |

Readings:

- Converged Wn clusters at **≈ 1.11–1.13** (HEX8 ref 1.112, TET10@52 1.132). HEX8 is
  within 5% of it already at 8.2k DOF; **TET4 is 20–47% low everywhere tested**
  (classic constant-strain locking on bending-dominated load paths) and is still
  10% low with 3.3x the DOF of the converged hex. TET10 is accurate but pays
  8–60x hex wall-clock (jax-fem assembles TET10 with 4-point quadrature over 27k
  cells; plus sliver conditioning slows the iterative solves).
- **Geometric fidelity inverts the picture**: tet boundary deviation is 8–20x
  smaller than hex at matched cell size (the tet boundary *is* the projected DC
  surface; hex has a snapped staircase). This is exactly why the rib gradient
  (below) is wrong on hexes.
- Element quality: hexes are excellent almost everywhere (mean scaled Jacobian
  0.95+); tets inherit DC's skinny surface triangles through `-Y` and carry sliver
  minima of 0.001–0.007 (1st percentile 0.14–0.27) — the cost of preserving the DC
  surface exactly. These slivers made jax-fem's default BiCGStab diverge more than
  once during this study; a direct/multigrid linear solver is a prerequisite for a
  production tet path.

### Gradient validation (frozen topology, adjoint vs central FD, eps 1e-3)

Objective: W (load work) at the nominal design; `recompute_tet_points` (Newton
re-projection of the frozen DC vertices, interior Steiner frozen) vs
`hexmesh.recompute_points`. rel = |adjoint − FD| / |FD|.

TET4 @ 32x24x19 (2 256 nodes / 7 556 tets, 1 932 surface):

| parameter | adjoint | central FD | rel |
|---|---|---|---|
| web_thickness   | −0.30176 | −0.29900 | 9.2e-3 |
| rib_height      | −0.06403 | −0.07426 | 1.4e-1 |
| plate_thickness | −0.51227 | −0.51237 | **2.0e-4** |

HEX8 @ 30x21x16 (same objective):

| parameter | adjoint | central FD | rel |
|---|---|---|---|
| web_thickness   | −0.28732 | −0.28461 | 9.5e-3 |
| rib_height      | −0.00990 | −0.00990 | 3.0e-5 |
| plate_thickness | −0.64001 | −0.64002 | 2.5e-5 |

- The adjoint machinery itself is exact: FD spot checks of dW/d(points) at
  individual nodes agree to 4–5 digits (interior −1.6614e-4 vs −1.6613e-4, boundary
  −2.2391e-3 vs −2.2391e-3).
- The elevated web/rib rel on the tet path is **subgradient structure, not error**:
  sharp DC placement puts many vertices exactly on SDF creases, where the Newton
  re-projection is subdifferentiable — AD picks a one-sided branch, central FD
  averages the branches. The eps study shows FD *approaching the adjoint* as eps
  grows for web (−0.2818 @ 3e-4 → −0.3035 @ 3e-3 vs adjoint −0.3018) and a
  persistent ~15% branch gap for rib. On the crease-free sphere the same chain
  matches FD inside 5e-2 (regression test).
- **The rib row is the pro-tet headline**: the hex discrete objective is nearly
  blind to `rib_height` (−0.0099 at res 30, −0.0205 at res 48) while the tet path
  reports −0.064 (adjoint) / −0.074 (FD) and TET10 −0.128. The hex staircase
  aliases the inclined rib; the DC-conforming tet boundary tracks it. High-res
  adjoints (load-work objective): TET10@32 = (−0.499, −0.128, −0.989),
  TET4@52 = (−0.354, −0.076, −0.741), HEX8@48 = (−0.464, −0.020, −1.013) — the
  discretizations agree on web/plate scales and disagree 5x on the rib, with the
  geometry-conforming meshes on one side.

### Interior (Steiner) node sensitivity — the frozen-interior justification, measured

Shape-derivative argument: the discrete sensitivity to interior node motion is a
mesh-motion term that vanishes in the continuous limit (Hadamard: only normal
boundary motion changes the shape). Measured on TET4 @ 32x24x19 via the full adjoint
dW/d(points):

| set | nodes | RMS |grad| | max |grad| |
|---|---|---|---|
| boundary (DC) | 1 932 | 5.89e-3 | 5.03e-2 |
| interior (Steiner) | 324 | 3.85e-4 | 1.91e-3 |

Interior sensitivity is 15x (RMS) / 26x (max) below boundary — and FD confirms those
small interior entries are real discrete values, not noise. Freezing interior nodes
biases the gradient by well under the crease-branch effects above. One differentiable
Jacobi-Laplacian pass propagating boundary deltas inward (`smooth_passes=2`) changed
the adjoint by ~1% and did not change FD agreement — worth having for large design
steps, irrelevant at validation scale.

## Route 3 (headline): the mesher as a Tesseract with a surface-interpolation VJP

`cadjoint/fem/tesseracts/mesher/tesseract_api.py`. `apply` = lattice samples f_i →
trilinear interpolant f(x) = Σ w_i(x) f_i → mesher → mesh (frozen topology; two
meshers behind one `element` switch: DC+TetGen → TET4, voxelize+snap → HEX8 — the
VJP is mesher-agnostic by construction). Every movable boundary vertex v satisfies
f(v) = 0, so the implicit function theorem gives dv/df_i = −w_i(v) g/|g|², and the
VJP rows *are* the interpolation weights at the frozen vertex positions. Interior
vertices contribute zero (measured justification above). Topology is promised via
shape-carrying template inputs (`point_ids`, `cell_template`) after one concrete
discovery `apply`.

### Exactness of the VJP (measured, sphere 14³/12³ lattice)

- Mechanical check vs a pure-JAX reference of the IFT map: **max rel err 0.0**.
- End-to-end: traced d(mean boundary radius)/d(radius) = 1.00624 (the 0.6% is the
  trilinear interpolant's own discretization of the sphere, not VJP error).
- The decisive check (bracket, sharp DC, 32x24x19, full FEM cotangent): the IFT VJP
  vs **full jax.grad through the actual Newton projection onto the interpolant** —
  gradients agree to **1.4e-11 relative** on all three parameters, and the pulled-back
  field-cotangent maps agree per-sample. The "interpolation weights as VJP map" idea
  is not an approximation of the wrapped mesher's derivative; for zero-set-conforming
  meshers it *is* the derivative (the dropped tangential/interior parts are gauge).

### The honest finding: sampling the field is the approximation, not the VJP

Same bracket, same objective, same 1 932 DC surface vertices:

| gradient path | web_thickness | rib_height | plate_thickness |
|---|---|---|---|
| mesher on **true SDF**, frozen reprojection (Route 1) | −0.302 | −0.064 | −0.512 |
| mesher on **lattice samples** (tesseract, IFT VJP)     | **+0.300** | −0.078 | −0.557 |
| same, via autodiff of the projection (control)         | +0.300 | −0.078 | −0.557 |
| central FD with full remeshing (truth, noisy)          | −0.821/−0.034 (eps 1e-3/3e-3) | +1.24/+0.33 | — |

The tesseract gradient equals its own mesher's true frozen-topology gradient
(control row) — but that mesher sees only the trilinear interpolant, which smears
every crease across a cell, and the `web_thickness` response of the bracket is
crease-dominated (the loaded wall is bounded by creases). Result: a genuine **sign
flip** relative to the true-SDF mesher at this lattice, while full-remesh FD is too
noisy to arbitrate at these eps (topology changes at |dθ| = 1e-3 — the frozen
promise check trips, which is the designed behavior). Plate thickness (large smooth
faces) agrees within 9% across all paths. v1 verdict: the interpolation-VJP
*contract* is right and exact; the lattice-sampling *interface* needs either a finer
lattice near creases or exact crease data in the schema before it can replace the
in-process Route 1 recompute for crease-dominated parameters.

Robustness note: DC on the interpolant is more fragile than DC on the SDF (the
interpolant is C0 with gradient jumps at every cell face). Of the probed configs,
sharp @ 32x24x19 meshed; smooth @ 32x24x19 and both @ 39x29x23 self-intersected in
TetGen. fTetWild would likely absorb these (it welcomes dirty input) — blocked only
by the missing cp314 wheel. The HEX8 mesher mode has no such fragility.

### Two-tesseract chain (mesher ∘ elastic_jaxfem, unmodified) — works

`tests/fem/test_tetmesh.py::TestTwoTesseractChain` doubles as the runnable demo
(`pytest -s`). CAD params → SDF lattice samples → mesher tesseract (HEX8 mode —
the packaged elastic tesseract's schema is HEX8; the tet mode is the same tesseract
one input flag away) → **unmodified** `elastic_jaxfem` tesseract → compliance
(`sum u²`) + smoothed-mass objective → one `jax.grad`. Bracket @ 30x21x16
(N = 2 678 nodes, 1 664 hexes, 1 534 snapped):

- One `jax.value_and_grad` through both tesseracts: **20 s**;
  grad = (+12.43, −5.75, −350.37) at the nominal design.
- Central FD of the same frozen-topology objective:
  **plate_thickness −350.37 vs −348.85, rel 4.4e-3** (the clean, crease-light
  parameter). rib −5.75 vs −4.31 (rel 0.34) and web +12.43 vs +41.0 at eps 3e-4
  (sign agrees; both parameters are kink/topology-noise dominated at this
  resolution — eps 1e-3 already flips voxelization topology, which the frozen
  promise check catches by design). The positive web component is physical for
  this objective: the loaded outer-wall area grows with the web (same sign as the
  hex-path bracket-demo table in `fem-integration.md`).
- **5 projected gradient-descent steps, one gradient each**:
  J = 21.678 → 18.497 → 15.901 → 15.198 → 13.980 (**monotone, −36%**), with the
  frozen-topology promise breaking at every step and the driver *refreezing
  automatically* (re-discover + re-resolve BCs, N drifting 2678 → 2725) — the
  remesh-when-invalid loop the frozen-topology doctrine prescribes, falling out of
  the tesseract contract for free.

## Route 2: DC/QEF-informed placement for the hex mesher — measured, not worth it

Measured on the bracket (hex 24x17x13 & 30x21x16 vs DC at matched cell size):

- Newton-snapped hex boundary vertices already sit on the zero set to **machine
  precision** (mean |sdf| 1.3e-17); DC's QEF vertices are the *less* accurate ones
  pointwise (mean 1.1e-3). Vertex placement is not the hex path's problem.
- Moving every snapped hex vertex to its closest point on the DC surface (proxy for
  feature-aware/QEF placement; mean move 6.0e-4, max 1.4e-2) changes sampled surface
  deviation by < 1% (0.00528 → 0.00526 mean at res 30) and slightly *improves* the
  min scaled Jacobian (0.028 → 0.083) with zero inversions — i.e. even a free,
  perfect QEF placement pass moves the needle marginally.
- The hex path's real geometric error (dev mean 5.3e-3 vs DC's 6.3e-4) lives in the
  **staircase faces between snapped vertices**, which no vertex placement can fix —
  only conforming topology (the tet route) or cut/dual cells can. Verdict: skip
  Route 2; its theoretical ceiling is already measured to be negligible.

## Solver support matrix

| solver | TET4 | TET10 | notes |
|---|---|---|---|
| jax-fem (in-process) | **confirmed by running** (`ele_type="TET4"`) | **confirmed by running** (`ele_type="TET10"`, meshio tetra10 order) | default BiCGStab fragile on sliver tets; direct solver advisable |
| CalculiX (ccx 2.23) | C3D4 deck: straightforward (same node order as meshio `tetra`) | C3D10: straightforward (ccx's canonical element; midside order matches meshio) | deck writer change only (`*ELEMENT, TYPE=C3D4/C3D10` + 3/6-node face loads); the STRAINENERGY DFDN correction's `d(detJ)` term needs re-deriving for tet shape functions — not implemented, no blocker identified |

## Wall-clock summary (M-series CPU, warm)

- TetGen itself is negligible: 0.03–0.16 s for 5k–27k tets (DC extraction dominates
  mesh build at 2–5 s for both routes).
- Adjoint gradient (3 params, forward+backward): TET4@32 12 s, HEX8@30 13 s,
  TET4@52 21 s, HEX8@48 26 s, TET10@32 207 s.
- Route 3 tesseract: discovery apply ~3 s; traced forward+VJP ~16 s (re-runs the
  mesher once in apply and once in the VJP — by design, statelessness over speed).

## Files

- `cadjoint/fem/tetmesh.py` — TetMesh, `sdf_to_tet_mesh` (project-then-mesh),
  `recompute_tet_points` (+ optional Laplacian passes), `tet10_from_tet4`,
  `tet_elastic_solve` (TET4/TET10, exact `traction_faces` targeting), quality
  metrics, load-work helpers (tri3/tri6/quad Gauss).
- `cadjoint/fem/tesseracts/mesher/tesseract_api.py` — the mesher tesseract
  (TET4/HEX8 modes, interpolation VJP, frozen-topology templates).
- `tests/fem/test_tetmesh.py` — 26 tests (mesh contract, quality metrics, selection
  duck typing, recompute fixed point + gradients, TET10 promotion, TET4/TET10
  solves, sphere adjoint-vs-FD, mesher tesseract, and the two-tesseract chain demo);
  all skip cleanly without tetgen / jax_fem / tesseract deps.

## Production verdict (2026-09-01): TET10 shipped behind `SimMesh(method=...)`

TET10 is now a first-class meshing method (user decision: "tet10 sounds a lot
better"), productionized as `SimMesh(method="hex"|"tet4"|"tet10")` with hex
remaining the fast default. What shipped, out of prototype status:

- `SimMesh.method` (validated at construction, in `describe()`/`inspect()`),
  `build()` routing hex → `sdf_to_hex_mesh`, tet4/tet10 → `sdf_to_tet_mesh`
  (+ `tet10_mesh` promotion), with a **sharp → Tikhonov DC fallback** when
  TetGen rejects the sharp surface. `resolution` stays the sampling lattice:
  elements for hex, the DC extraction grid for tets (TetGen decides tet counts).
- Studies route both physics to tet solves: `tet_elastic_solve` promoted, and a
  new `tet_thermal_solve` mirroring the lifted Dirichlet formulation, both with
  exact boundary-face targeting (`_restrict_surface_faces` now covers heat-flux
  patches too). TET10 BC sets are completed with midside nodes
  (`tet10_complete_nodes` / `tet10_face_midsides`). `SimulationResult` is
  method-agnostic (tet von Mises at centroids — for TET10 the corner shape
  gradients vanish there, midside-only formula; meshio `tetra`/`tetra10` VTK).
- Frozen-topology parity: `recompute_tet_points` handles TET10 (corner surface
  re-projection, midsides rebuilt as traced corner midpoints; exact fixed point
  at the nominal design, regression-tested).

### Grid caveat found during productionization

The bracket's filleted union (`smooth_min`) dips to **z ≈ −0.063**, below the
demo grid floor (−0.06). Voxelization never noticed; DC extraction returns an
*open* surface there and TetGen rejects it. All tet bracket work below uses the
deepened box `bounds=(-1.3, -0.95, -0.16)`, `size=(2.6, 1.9, 1.52)`. Sharp DC
also still self-intersects at unlucky resolutions (24x18x14 fails both modes at
this box; 28x21x17 needs the smooth fallback) — the fallback absorbs some of
this, the rest surfaces as a clear TetGen error naming the remedy.

### Thermal-on-tet validation (bar, Galerkin exactness)

The projected tet boundary is the exact box, so solutions inside the FE space
must be reproduced to solver tolerance — and are (`tests/fem/test_study.py::TestTetStudies`):

- linear conduction profile: max |T − (1−x)/2| < 1e-6 on TET4 **and** TET10;
- heat flux with exact face targeting: max err < 1e-6 (T = (q/k)(x+1));
- volumetric-source parabola (quadratic, in the TET10 space): max err < 1e-5.
- Elasticity: TET10 cantilever lands in the Euler-beam window (0.7–1.3) at
  15x6x6 where TET4 on the same mesh is visibly stiffer (the locking exhibit).

### The DOF-at-matched-accuracy benchmark (`benchmarks/tet_vs_hex_bench.py`)

Same bracket compliance problem, both families on the deepened box, load patch
selected by the mesh-independent center+normal wall rule (areas 0.29–0.38 vs
exact ≈ 0.348; Wn = W/area² as before). Reference: hex @ 66x48x39, 72.7k DOF,
Wn = 1.1056 (finest TET10 gives 1.1522 — 4.2% family spread, so the reference
itself carries a few-% uncertainty; the 3% band is relative to the hex ref).

| mesh | res | cells | DOF | Wn | vs ref | t_mesh | t_solve |
|---|---|---|---|---|---|---|---|
| HEX8  | 24x18x14 | 964    | 5.0k  | 1.066 | −3.6% | 3.5 s | 3.6 s |
| HEX8  | 30x22x18 | 1 846  | 8.8k  | 1.086 | **−1.7%** | 3.6 s | 4.0 s |
| HEX8  | 36x26x21 | 3 512  | 15.1k | 1.097 | −0.8% | 4.1 s | 5.4 s |
| HEX8  | 42x31x25 | 4 700  | 20.2k | 1.109 | +0.3% | 3.8 s | 7.0 s |
| HEX8  | 48x35x28 | 7 702  | 31.0k | 1.127 | +1.9% | 4.6 s | 9.0 s |
| HEX8 ref | 66x48x39 | 19 306 | 72.7k | 1.106 | — | 4.7 s | 24.6 s |
| TET10 | 22x16x13 | 4 629  | 25.3k | 1.139 | **+3.0%** | 4.9 s | 26.2 s |
| TET10 | 24x18x14 | 4 645  | 26.4k | 1.136 | +2.7% | 5.3 s | 26.8 s |
| TET10 | 26x19x16 | 5 750  | 32.0k | 1.196 | +8.2% | 4.9 s | 31.7 s |
| TET10 | 28x21x17 (smooth) | 7 037 | 39.1k | 1.173 | +6.1% | 6.8 s | 241.5 s |
| TET10 | 30x22x18 | 7 946  | 44.0k | 1.071 | −3.1% | 5.5 s | 52.3 s |
| TET10 | 32x23x19 | 10 176 | 53.8k | 1.149 | +4.0% | 5.1 s | 120.7 s |
| TET10 | 34x25x20 (smooth) | 11 873 | 62.7k | 1.152 | +4.2% | 6.9 s | 119.6 s |

Matched-accuracy picks (coarsest within 3% of the reference) and the
`d(W)/d(rib_height)` adjoint there (3-parameter value_and_grad wall-clock):

| | res | DOF | Wn | mesh+solve | grad (web, rib, plate) | t_grad |
|---|---|---|---|---|---|---|
| HEX8  | 30x22x18 | 8.8k  | 1.086 (−1.7%) | **7.6 s** | (−0.287, **−0.0086**, −0.503) | 13.9 s |
| TET10 | 22x16x13 | 25.3k | 1.139 (+3.0%) | **31.0 s** | (−0.408, **−0.0990**, −0.881) | 50.8 s |

**The headline correction this benchmark exists for: at matched accuracy TET10
costs ~4x hex wall-clock (31.0 s vs 7.6 s; 2.9x DOF; gradient 3.7x) —
materially better than the 8–60x of the matched-lattice tables.** The earlier
factor compared meshes whose accuracy differed by design; TET10 reaches the 3%
band at its coarsest meshable lattice, where hex needs a similar-DOF lattice of
its own. Honest caveats: the TET10 Wn ladder oscillates ±4% (its load patch
rides the DC surface), so its band-edge pick is less settled than hex's
monotone ladder; and the smooth-fallback meshes pay heavy solver-conditioning
penalties (241 s at 39k DOF — the "direct/multigrid solver" prerequisite
stands).

**The rib exhibit survives at matched accuracy** — and is the reason to pay the
4x: hex at its matched-accuracy size still reports d(W)/d(rib_height) = −0.0086
where TET10 reports −0.0990 (11.5x). Matching hex's *objective value* does not
fix its *gradient blindness* to features the lattice staircases over; an
optimizer driving `rib_height` needs the conforming mesh.

### Named-mesh tet10 gradient (the study-path regression, `TestTet10NamedMeshGradient`)

Bracket @ 22x16x13 TET10 (25.3k DOF), `SimMesh(method="tet10")` +
`ElasticStudy(mesh=...)` + `recompute_tet_points` + `solve(points=...)`,
objective mean |u|, adjoint vs central FD (eps 1e-3):

| parameter | adjoint | central FD | rel | asserted |
|---|---|---|---|---|
| web_thickness   | −0.034840 | −0.034283 | 1.6e-2 | sign + magnitude window |
| rib_height      | −0.018134 | −0.022867 | 2.1e-1 | sign + magnitude window |
| plate_thickness | −0.442785 | −0.438715 | **9.3e-3** | ≤ 1e-2 rel |

The web/rib gaps are the measured crease-subgradient structure from the
gradient-validation section above (sharp DC vertices sit on SDF kinks; AD takes
a one-sided branch, FD averages) — directionally exact, hence the sign/window
assertions rather than tight rel bounds.

## TET10 two-Tesseract chain + the gradient-path seam (2026-09-01)

The packaged SOLVER tesseract schemas are now element-agnostic, completing
the TET10 two-tesseract chain and opening the playground door:

- `elastic_jaxfem` / `thermal_jaxfem`: `cells` is `(T, K)` with K = 4/8/10
  picking TET4/HEX8/TET10 (meshio order); tet modes reuse
  `tet_elastic_solve` / `tet_thermal_solve` verbatim (same spsolve options,
  same adjoint contract — `points` differentiable). Optional exact-face
  targeting crosses the boundary as `traction_faces`/`flux_faces` +
  prefix-offset arrays (empty = pure node membership, the HEX8 behavior).
- `thermal_jaxfem` additionally carries heat-flux (Neumann) patches:
  `flux_nodes`/`flux_offsets`/`flux_values`, mirroring the direct backend's
  surface-map path; `TesseractBackend.thermal` forwards HeatFlux BCs now
  (the old NotImplementedError is gone).
- `elastic_calculix` accepts the two new face fields for schema parity
  (HEX8-only; non-empty raises).

Solver-stage parity (tests/fem/test_tesseract_tet.py, bar mesh 13x7x6):
TET4 and TET10, elastic and thermal, apply == direct solve at max diff
< 1e-9 (measured 0.0 for the starter chain below); traced gradients through
the tesseract boundary equal the direct adjoint < 1e-9; the thermal
design-gradient through `recompute_tet_points` + tesseract equals the
direct route to 1e-9 relative (both -71.734 on the bar half-height, where
central FD reads -60.42 — the known crease-subgradient gap, not a boundary
artifact).

### Starter heat sink on the chain (the crease-heavy validation)

Lattice caveat (measured): the mesher tesseract meshes the *trilinear
interpolant*, which self-intersects at the starter's declared 18x13x11
resolution (both sharp modes). 24x18x15 is the coarsest lattice where
sharp DC meshes the sink: 11 390 TET10 nodes / 6 459 tets / 1 384 surface
vertices. The direct path meshes the true SDF at the declared resolution —
the chain needs a finer lattice for the same scene.

Measured on the frozen chain (mesher TET10 + flux-capable thermal
tesseract, objective max temperature, conductivity 2.0, die flux 6.0):

- stage-2 parity vs `tet_thermal_solve` on the same mesh: max |dT| = 0.0;
- physics: T in [0, 1.139], slug bottom mean 0.990 vs held fin field 0.0;
- d(max T)/d(fin_depth) through both tesseracts: **-0.2498** (deeper fins
  cool the die — the sign the physics demands), via one `jax.grad` through
  mesher-VJP + solver adjoint.

The direct-path comparator (`recompute_tet_points` on the same frozen
mesh, smooth_passes=2 + direct TET10 solve) and the 4-step descent were
still factorizing at freeze time (SuperLU on the 11.4k-DOF TET10 system
dominates; the direct path re-solves per FD/descent step) — the harness
tests `tests/fem/test_starter_chain.py::TestStarterChain` assert the
sign-consistency and descent when they complete. Hex seam smoke (bar
22x5x5, Dirichlet+flux): tesseract path J=1.000000 grad_norm=3.637 vs
direct J=1.000000 grad_norm=8.119 — same forward value, both descend, the
norm gap is the box-crease gauge difference.

### The seam: `Optimization(..., gradient_path=)`

`cadjoint/optimize.py` now takes `gradient_path="direct"|"tesseract"`
(study form only, **default unchanged: "direct"**). "tesseract" swaps the
per-step derivative chain at the marked seam for
`cadjoint/fem/tesseracts/chain.py::freeze_study_chain`: lattice samples ->
mesher tesseract (frozen-topology templates, sharp->Tikhonov fallback,
interpolation VJP) -> solver tesseract adjoint -> metric (mean/max both
physics; compliance reuses the traction-work helper). Refreeze failures
degrade exactly like the direct path (keep previous topology mid-run,
clear error at step 0); the final reported result is always evaluated on
the direct path. Default-flip recommendation: NOT yet — the interpolant
needs a finer lattice than the declared meshes (18x13x11 starter fails to
mesh), the VJP drops tangential crease motion, and the direct-vs-chain
magnitude comparison on the starter is still running; revisit with those
numbers in hand.

## The narrow cut: only TetGen is a black box (`tetfill`, 2026-09-01)

User's design objection, verbatim: *"i dont understand why the whole meshing pipeline
is in the meshing tesseract? i think its only the tet meshing that needs this the rest
should already natively be differentiable"* — **correct, and the measurements below
say the wide cut was costing real accuracy.**

### The argument for cutting at TetGen

Everything in `cadjoint/meshing` is already differentiable *by construction*: crossing
edges are frozen per extraction, the root on each edge is bisected on
`stop_gradient` values and then Newton-corrected on the **true SDF**
(`edge_hermite_data`), and QEF placement is a `jnp.linalg.solve` of a 3x3 system
(`qef_vertices`). The Newton projection `sdf_to_tet_mesh` applies before handing the
surface to TetGen is the same `project_points` the hex path uses. None of it needs a
hand-written VJP. Only TetGen is a compiled black box.

The `mesher` tesseract nevertheless swallows the whole pipeline, and pays for it
twice: (1) the field crosses the boundary as *lattice samples*, so its DC runs on the
**trilinear interpolant**, which smears every crease across a cell — the measured
cause of the `web_thickness` sign flip above and of the interpolant DC's TetGen
self-intersections; (2) its VJP can only be the implicit-function-theorem map on that
interpolant, i.e. exact for a mesher that is not the one the direct path uses.

`cadjoint/fem/tesseracts/tetfill/` cuts at the seam instead. Inputs: a watertight
surface (`points` + `triangles`) and TetGen's options. Outputs: `nodes`, `cells`, a
`(P, 2)` `parents` table and a `steiner_mask`. Because TetGen runs with `-Y`
(`nobisect`), the input vertices survive **verbatim** (asserted bit-for-bit in the
forward, not to a tolerance — the whole VJP rests on it), so the boundary map is a
*gather* and its VJP is the gather's transpose:

- preserved vertex `i < V`: cotangent passes straight through (its `parents` row is
  `(i, i)`, and every parent carries weight 0.5, so `0.5 + 0.5 = 1`);
- Steiner node: no parents, cotangent dropped — justified by the measured interior
  sensitivity above (15x RMS / 26x max below the boundary on the bracket TET4);
- TET10 midside: `parents` lists its two corners, so the cotangent splits
  half-and-half (`m = (a + b)/2` is exactly linear — the same step the `mesher`
  tesseract takes).

### Frozen fill: why the traced call does not re-run TetGen

Measured, and the reason for a second mode: **TetGen's quality-driven Steiner
insertion is not continuous in the input surface.** On the box bar below a design
perturbation of **1e-4** already changes the Steiner count (222 -> 232 nodes), which
breaks the frozen-topology promise and makes both descent and finite differences
impossible. So `interior_points` (empty = run TetGen, non-empty = re-evaluate the
frozen fill with the interior held and `cell_template` carrying the connectivity
verbatim) is a first-class input, and the chain pins it by default. This is not a
weakening: holding the interior is exactly what `recompute_tet_points` does on the
direct path, and exactly what the VJP already asserts by dropping Steiner cotangents —
so in frozen mode the forward *is* the gather its derivative transposes, and the two
are consistent by construction rather than to a tolerance. Verified: the frozen fill
reproduces the TetGen fill node-for-node, cell-for-cell, parents-for-parents.

### VJP exactness (mechanical, sphere 10^3 lattice, 224 surface vertices)

VJP vs `jax.vjp` of the equivalent JAX gather (`concat(points, frozen_interior)`, plus
corner-mean midsides for TET10), random cotangent:

| element | mesh | max rel err |
|---|---|---|
| TET4  | 309 nodes / 1 132 tets | **0.0** (bit-exact) |
| TET10 | 1 971 nodes / 1 132 tets | **1.25e-16** |

A cotangent supported only on Steiner nodes pulls back to exactly 0.

### The chain: `freeze_study_chain_dc` / `gradient_path="tesseract-dc"`

`CAD params -> JAX dual contouring on the true SDF (frozen edges + incidence +
triangulation, differentiable Hermite roots, differentiable QEF, Newton projection)
-> tetfill tesseract -> solver tesseract -> metric`, one `jax.grad`. Frozen topology
per extraction exactly like the interpolant chain. One deliberate restriction:
the traced surface uses the **Tikhonov** QEF, not `sharp_qef_vertices` — singular-value
truncation has no usable derivative, and mixing placements would break the fixed point
(the frozen mesh's boundary must equal the traced surface at the nominal design; it
does, to `max |dev| = 0.0`).

### Bar exhibit: where frozen-base re-projection differentiates the wrong shape

Box bar, `SimMesh(method="tet4")` @ 14x7x7 (222 nodes / 591 tets / 210 surface), heat
flux 2.0 in at `-x`, held 0 at `+x`, `k = 1`. The exact solution `T = (q/k)(L - x)`
gives mean `T = q L / k = 2 L`, and is **independent of the cross-section**.

| quantity | tesseract-dc | direct (same frozen mesh) | truth |
|---|---|---|---|
| forward `J` | 1.585170 | 1.585170 (parity **0.0**) | 1.6 (exact) |
| `dJ/d(half_length)` | **+1.999469** (central FD **+1.999469**) | — | +2 |
| `dJ/d(half_thickness)` | **-0.0085** (FD -0.0085 at eps 1e-3 *and* 3e-3) | **-8.61** | 0 |

The last row is the finding. `recompute_tet_points` moves each frozen boundary vertex
along the SDF gradient, which cannot move an end-cap vertex *tangentially*: as the bar
thickens, the caps do not widen, the meshed shape stops being a box, and the direct
path reports a derivative three orders of magnitude too large for a quantity that is
physically insensitive. Re-extracting DC from the true SDF has no such gauge: every
vertex is re-derived, so the chain tracks the real shape. Tangential motion is gauge
*for the objective* only when the shape's tangential extent does not change — which is
exactly the assumption the DC chain does not need to make.

### Starter heat sink at its DECLARED resolution (the crease-heavy headline)

`scenes/starter.py` verbatim (18x13x11, `method="tet10"`, k = 2.0, die flux 6.0),
objective max temperature, parameter `fin_depth`:

| gradient path | mesh (nodes / cells / surface) | `J = max T` | `d(max T)/d(fin_depth)` |
|---|---|---|---|
| `direct` (own sharp-DC mesh, true SDF) | 5 963 / 3 136 / 860 | 1.145255 | **-0.157944** |
| **`tesseract-dc`** (JAX DC + tetfill) | 5 722 / 2 953 / 860 | 1.147747 | **-0.165900** |
| `tesseract` (interpolant mesher) | 6 412 / 3 499 / 860 | **0.968120** | **-0.090242** |
| `direct`, on the DC chain's own frozen mesh | 5 722 / 2 953 / 860 | 1.147747 | -0.135649 |

- **Meshing:** the DC chain meshes the sink at the declared 18x13x11 in 7.5 s. Honest
  correction to the earlier section: the interpolant chain *also* meshes at 18x13x11
  today (it did not when that note was written); it is at **24x18x15** — the lattice
  `tests/fem/test_starter_chain.py` pins — that the interpolant DC now self-intersects
  and TetGen refuses (that suite currently fails at HEAD, independent of this work).
  So the claim to make is not "the interpolant chain cannot mesh the starter" but the
  sharper one: **the interpolant chain's meshability is resolution-lottery, and its
  *physics* is wrong even when it meshes** — its `max T` is 16% below both true-SDF
  meshers, because the trilinear interpolant rounds the fin comb off.
- **Gradient:** `tesseract-dc` lands within **5%** of the direct path's adjoint on the
  same scene; the interpolant chain is **43% low**.
- **Stage-2 parity:** the packaged thermal tesseract equals `study.solve` on the
  chain's frozen mesh at max `|dT| = 0.0`.
- **The chain's adjoint is its own FD, to 6 digits** — on the crease-heavy fin comb,
  where the direct path's adjoint-vs-FD gap is 15-20% (the crease-subgradient
  structure measured above):

  | central FD eps | 1e-3 | 3e-3 | 1e-2 |
  |---|---|---|---|
  | `d(max T)/d(fin_depth)` | -0.165900 | -0.165902 | -0.165887 |

  (adjoint -0.165900). Pinning the interior is what buys this: the design step no
  longer perturbs a discrete Steiner set, so the whole map is smooth.
- **Descent** (plain gradient steps, lr 0.05, no refreeze):
  `J = 1.147747 -> 1.146387 -> 1.145090 -> 1.143849 -> 1.142663` (monotone,
  `fin_depth` 1.200 -> 1.232 — deeper fins cool the die). 17.2 s for the first
  `value_and_grad` (JIT), **6.1 s** each thereafter.

### Bracket (elastic, TET10 @ 22x16x13, deepened box, compliance)

8 744 nodes / 4 873 tets / 1 108 surface vertices; freeze 5.8 s.

- Stage-2 parity vs `study.solve` on the same frozen mesh: max `|du| = 0.0`.
- `J = 0.06532372`.

| parameter | tesseract-dc adjoint | central FD (eps 1e-3) | rel | direct, same mesh |
|---|---|---|---|---|
| `plate_thickness` | -0.430398 | -0.430021 | **8.8e-4** | -0.430191 |
| `web_thickness`   | -0.253404 | -0.237647 | 6.6e-2 | -0.219726 |

`plate_thickness` (large smooth faces) is exact to 0.09% against FD and 0.05% against
the direct adjoint — an order tighter than the direct path's own 9.3e-3 in the
named-mesh table above. `web_thickness` keeps the known crease structure (the loaded
wall is bounded by creases) but all three numbers now agree in sign and to ~15%,
against the interpolant chain's outright **sign flip** on the same parameter.

### Recommendation on the default gradient path

1. **Ship `gradient_path="direct"` as the default, unchanged.** It needs no extra
   dependency, and it is the only path that uses **sharp** DC placement — vertices
   landing exactly on creases and corners. `sharp_qef_vertices` has no usable
   derivative (singular-value truncation), so the DC chain must use the Tikhonov QEF;
   the measured cost is small here (starter `max T` 1.1477 vs 1.1453, 0.2%), but it is
   a real geometric-fidelity regression and the one blocker to flipping the default.
   **Differentiable sharp placement is the work item that would unblock it.**
2. **Make `"tesseract-dc"` the recommended Tesseract path, and stop recommending
   `"tesseract"` for tet meshes.** On every axis measured on the same scene,
   tesseract-dc dominates the interpolant chain: forward physics (0.2% vs 16% from the
   direct path), gradient (5% vs 43%), meshability at the declared resolution, and
   adjoint-vs-FD self-consistency (6 digits vs a chain whose FD probes change
   topology). The interpolant chain keeps exactly one advantage — it is
   mesher-agnostic, so it remains the right wrapper for a mesher that does *not*
   preserve its input vertices (the `-Y` contract is what makes the pass-through VJP
   possible at all), and it is the only one with a HEX8 mode.
3. **Where tesseract-dc should be preferred over `direct` today:** any objective whose
   parameter changes the shape's *tangential* extent (the bar exhibit: direct is 1000x
   off there), and any workflow that needs the mesher to run out of process.

### Files

- `cadjoint/fem/tesseracts/tetfill/` — `tesseract_api.py` (TetGen forward + frozen
  fill, gather VJP), `tesseract_config.yaml`, `tesseract_requirements.txt`.
- `cadjoint/fem/tesseracts/chain.py` — `freeze_study_chain_dc` / `FrozenDCChain` /
  `_freeze_dc_surface` (the frozen-topology JAX DC map), alongside the unchanged
  `freeze_study_chain`.
- `cadjoint/optimize.py` — `GRADIENT_PATHS = ("direct", "tesseract", "tesseract-dc")`
  at the marked gradient-path seam.
- `tests/fem/test_tetfill.py` — 20 tests (forward contract, `parents` table, exact VJP
  vs a JAX gather, frozen fill vs TetGen, chain fixed point, solver parity, adjoint vs
  FD with an analytic target, the seam and an end-to-end `Optimization` run); all skip
  cleanly without tetgen / jax_fem / tesseract deps, 17 s total.

## The frozen crossing BRACKETS expire: why the starter died at step 5 (2026-09-01)

The declared starter optimization (`scenes/starter.py`, `gradient_path="tesseract-dc"`,
12 steps) descended for five steps and then hard-failed inside the thermal solver
tesseract:

```
step  0  objective 1.615602  |grad| 1.3867e+00   24.6 s
step  1  objective 1.613115  |grad| 1.3989e+00    8.1 s
step  2  objective 1.611339  |grad| 1.4029e+00    7.1 s
step  3  objective 1.610146  |grad| 1.4106e+00    7.0 s
step  4  objective 1.609709  |grad| 1.4205e+00    7.1 s
step  5  RuntimeError: Direct linear solve failed on the tet system
         (best relative residual 1.16e+00) -- tetmesh.py:140
```

Note the descent itself: **-0.0059 over five steps**, and a gradient norm *rising*
every step. Both were already symptoms.

### The hypothesis that measurement killed

The obvious suspect is the frozen interior: the tetfill tesseract pins the Steiner
nodes while the boundary follows the design, so the tets straddling that gap should
degrade until the FEM system is ill-conditioned — and the direct path does not suffer
this because `recompute_tet_points(..., smooth_passes=2)` propagates boundary motion
inward. It is a good hypothesis. It is wrong, and one table settles it. Frozen fill at
the step-0 design (5 691 nodes / 2 926 tets / 860 surface), re-evaluated at each step's
parameters with N Jacobi-Laplacian interior passes:

| step | passes 0 | passes 2 | passes 5 | passes 10 | passes 20 |
|---|---|---|---|---|---|
| 0 | 0.1294 | 0.1294 | 0.1294 | 0.1294 | 0.1294 |
| 1 | 0.0603 | 0.0745 | 0.0748 | 0.0748 | 0.0748 |
| 2 | **7.30e-7** | 7.30e-7 | 7.30e-7 | 7.30e-7 | 7.30e-7 |
| 3 | **1.07e-6** | 1.07e-6 | 1.07e-6 | 1.07e-6 | 1.07e-6 |
| 4 | **7.03e-7** | 7.03e-7 | 7.03e-7 | 7.03e-7 | 7.03e-7 |
| 5 | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 |

(min element quality, `cbrt|vol| / longest edge`.) Smoothing is a **null operation**
from step 2 on — bit-identical numbers at 0 and 20 passes. The reason is in the next
column: of the 204 degenerate tets at step 2, **204 are boundary-only** (all four
corners are DC surface vertices, none touches a Steiner node), so no interior operator
can reach them. The mesh is not being pulled apart from the inside. Its *boundary* is
collapsing on itself.

### The actual cause

`_freeze_dc_surface` froze the crossing-edge set and then re-ran `edge_hermite_data` on
it every traced call. That root search is bracketed by the **frozen sign pattern at the
lattice vertices**, which is discrete topology, and it expires:

| step | frozen brackets still valid | lost | min \|vol\| | min quality | inverted | degenerate (<1e-3) |
|---|---|---|---|---|---|---|
| 0 | 858/858 | 0 | 6.17e-6 | 0.1294 | 0 | 0 |
| 1 | 858/858 | 0 | 3.34e-6 | 0.0603 | 2 | 0 |
| 2 | 690/858 | **168** | 1.56e-21 | 7.30e-7 | 105 | 204 |
| 3 | 690/858 | 168 | 2.73e-21 | 1.07e-6 | 104 | 204 |
| 4 | 690/858 | 168 | 8.69e-22 | 7.03e-7 | 117 | 204 |
| 5 | 570/858 | **288** | 0.0 | 0.0 | 133 | 298 |

168 brackets — **20% of them** — die in a single optimizer step, and the collapse is
immediate and total, not gradual. Why so abrupt: the sink's grid is 18x13x11 over
z ∈ [-0.3, 1.1], so lattice planes sit at z = -0.3 + 0.12727k, and the fin tips at
z = 0.85 sit **0.0045** from the plane at 0.84545. The comb's faces are axis-aligned,
so one small design step sweeps a whole plane of lattice vertices through the surface
at once.

`edge_hermite_data` documents its own behaviour on an expired bracket: bisection
collapses toward an endpoint. The cell is then fitted to a "surface sample" that is not
on the surface; `qef_vertices`' final cell clamp pins neighbouring cells' vertices onto
their shared face; the vertices coincide; the boundary tets built on them have exactly
zero volume. The solver sees an unsolvable system and is the first thing to complain,
several stages downstream of the actual fault.

The control confirms it end to end. The **direct** path's map, on the *same frozen
mesh* and the same parameters, never wavers:

| step | chain (old map) motion max / mean | direct motion max / mean | direct min quality | direct inverted |
|---|---|---|---|---|
| 1 | 0.1490 / 0.0035 | 0.0059 / 0.0028 | 0.1299 | 0 |
| 3 | 0.1350 / 0.0159 | 0.0179 / 0.0084 | 0.1299 | 0 |
| 5 | 0.1483 / 0.0229 | 0.0299 / 0.0140 | 0.1299 | 0 |

The real geometry moves by 0.03 at most. The old DC map amplified that into 0.148 of
vertex motion — garbage in, garbage out.

### Four candidate vertex maps, measured

All evaluated on the same frozen topology at the step-5 parameters:

| map | motion max / mean | min \|vol\| | min quality | inverted | degenerate | verdict |
|---|---|---|---|---|---|---|
| **A** re-solve frozen brackets (was) | 0.148 / 0.023 | 0.0 | 0.0 | 133 | 298 | the bug |
| **B** anchored crossings -> QEF | 0.148 / 0.020 | 1.6e-10 | 5.19e-3 | 2 | **0** | **shipped** |
| **C** re-project frozen vertices | 0.030 / 0.014 | 6.24e-6 | 0.1299 | 0 | 0 | ruled out |
| **D** A with `t` clamped to `[0,1]` | 0.148 / 0.023 | 7.7e-25 | 9.4e-8 | 115 | 252 | no help |

- **D** shows the unclamped Newton `t` is *not* the proximate cause: keeping the sample
  on its edge still leaves it off the surface, and the QEF is just as poisoned.
- **C** is the tempting one — it is the healthiest column in the table — and it is
  exactly what the **bar exhibit** above forbids. Re-projecting the final vertex moves
  it only along the SDF gradient, so `dJ/d(half_thickness)` goes from -0.0085 (truth 0)
  to **-8.61**: three orders of magnitude wrong on a quantity the physics is insensitive
  to. Tangential shape tracking is the whole reason this chain exists.
- **B** keeps it. The frozen crossings are located once, by the bracketed root search,
  at the freeze design; from then on they are *anchors* that ride the traced zero set
  under the same clamped Newton projection the pipeline already uses. No bracket is
  consulted, so nothing can expire. Every sample stays a genuine surface point with a
  genuine normal, so the QEF is still a **plane fit** — and a plane fit whose samples
  each move along their own local normal reproduces tangential vertex motion, which is
  precisely what per-vertex re-projection throws away.

One subtlety cost a measurement: `edge_hermite_data` re-evaluates a *degenerate*
gradient a fraction of an edge toward the inside endpoint (landing bit-exactly on a
polygon wall with a dead subgradient is common, not exotic). Without mirroring that,
map B lost those planes and moved the boundary by 0.0856 **at the freeze design** —
i.e. it stopped being a fixed point. With it, the fixed point is exact.

### The interior relaxation, kept on its own merits

Interior smoothing is not the fix, but it is now applied anyway — 2 Jacobi-Laplacian
sweeps, boundary pinned, shared with the direct path via
`tetmesh.smooth_interior_delta`. It is applied **in JAX in `chain.py`, on the
tesseract's returned nodes**, not inside the tesseract:

- the tetfill forward stays a pure gather and its VJP the exact transpose of one (no
  smoothing operator to transpose by hand);
- the relaxed nodes depend on the tesseract's output *only through the preserved
  boundary block*, so the cotangent reaching the tesseract is supported exactly where
  its pass-through is exact. The Steiner-cotangent drop stops being an approximation:
  the solver's interior-node sensitivity is now **transported onto the boundary**
  through the relaxation's transpose instead of being discarded;
- fixed point preserved exactly, TET10 midsides included (`max |dev| = 0.0` for both
  TET4 and TET10).

Measured worth: it recovers min quality 0.0344 -> 0.0624 at step 1 (where boundary
motion is concentrated on a few vertices) and is a null operation at steps 2-5, for the
same reason as before — those slivers are boundary-only.

### Frozen fill quality after the fix (same freeze, same parameters)

| step | min \|vol\| | min quality | inverted | quality < 0.01 |
|---|---|---|---|---|
| 0 | 6.31e-6 | 0.1304 | 0 | 0 |
| 1 | 3.68e-6 | 0.0624 | 2 | 0 |
| 2 | 2.69e-6 | 0.0632 | 4 | 0 |
| 3 | 9.06e-8 | 0.0368 | 3 | 0 |
| 4 | 8.40e-8 | 0.0366 | 1 | 0 |
| 5 | 1.55e-10 | 0.0052 | 2 | 52 |

### The declared 12-step run, completing

`scenes/starter.py`'s own `cool-sink` optimization, verbatim (`remesh_every` 6, so one
refreeze at step 6), 170.4 s total:

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J | 1.617200 | 1.614708 | 1.608122 | 1.600740 | 1.593071 | 1.585311 | 1.581312 | 1.574670 | 1.567995 | 1.561397 | 1.554886 | 1.548351 | **1.533429** |

Monotone, no solver failures, `fin_depth` 1.2000 -> 1.1524. 29.7 s for step 0 (JIT),
10.2-11.5 s thereafter, 27.7 s at the step-6 refreeze. Against the broken run's
-0.0059 over five steps this is **-0.0838 over twelve** — the old descent was not just
fragile, it was reading gradients off a mesh that had already collapsed.

### Re-validation: nothing regressed, several things improved

**Mechanical VJP** (unchanged code, re-checked): vs `jax.vjp` of the equivalent JAX
gather, TET4 max rel err **0.0**, TET10 **1.74e-16**; primal deviation 0.0 for both.

**Bar exhibit** — the property that ruled out map C, reproduced to every printed digit:

| quantity | before | after | truth |
|---|---|---|---|
| `J` | 1.585170 | 1.585170 | 1.6 |
| `dJ/d(half_length)` | +1.999469 (FD +1.999469) | +1.994235 (FD +1.994235) | +2 |
| `dJ/d(half_thickness)` | -0.008474 (FD -0.008474) | **-0.008473** (FD -0.008473) | 0 |
| direct, same mesh | -8.609769 | -8.609769 | 0 |

`half_length` moves by 0.26% and lands exactly on the direct path's +1.994235: that is
the interior relaxation, which makes interior nodes contribute to the *nodal* mean
exactly as they do on the direct path. `half_thickness` — the row the exhibit is about
— is unchanged, still 1000x better than re-projection.

**Starter gradient** (18x13x11, tet10, `d(max T)/d(fin_depth)`):

| | J | adjoint | FD 1e-3 | FD 3e-3 | FD 1e-2 |
|---|---|---|---|---|---|
| before | 1.147747 | -0.165900 | -0.165900 | -0.165902 | -0.165887 |
| after | 1.146109 | **-0.161859** | -0.161860 | -0.161862 | -0.161891 |

Adjoint-vs-FD self-consistency holds at 6 digits (rel 1.5e-6). Both numbers moved
*toward* the direct path's reference (`J` 1.145255, adjoint -0.157944): forward physics
0.22% -> **0.07%** off, gradient 5.0% -> **2.5%** off. Fixed point `max |dev| = 0.0`,
stage-2 parity `max |dT| = 0.0`.

**Bracket** (elastic TET10 @ 22x16x13, compliance; 8 642 nodes / 4 789 tets / 1 108
surface, freeze 6.8 s, `J = 0.06532303`, parity `max |du| = 0.0`):

| parameter | adjoint | FD 1e-3 | rel | rel before | direct, same mesh |
|---|---|---|---|---|---|
| `plate_thickness` | -0.425155 | -0.425171 | **3.7e-5** | 8.8e-4 | -0.429545 |
| `web_thickness` | -0.237656 | -0.240457 | **1.2e-2** | 6.6e-2 | -0.219700 |

Adjoint-vs-FD agreement is **24x tighter** on `plate_thickness` and **5x tighter** on
`web_thickness` — the expiring brackets were perturbing the FD probes themselves.

### On the alternatives, plainly

- **Interior smoothing alone is not merely insufficient, it is a null operation** for
  this failure (the table at the top). It ships anyway, for the reasons above, but no
  one should believe it fixed the starter.
- **Refreezing more often** would have worked and is the wrong trade. The measured
  validity horizon of the frozen brackets on the starter is **one step** (168 lost
  between step 1 and step 2), so it would mean `remesh_every=1` — a full TetGen
  re-extraction every step (27.7 s vs 10.4 s measured here, i.e. ~3x the wall clock),
  to paper over a map that should not have depended on the expiring data in the first
  place. With the anchored map the frozen topology survives its declared 6 steps with
  room to spare.
- **Quality-aware refill** was not needed: after the fix the worst element over the
  whole refreeze interval is quality 0.0052 with zero inverted tets, and the solver
  never falls past its first PETSc LU attempt.

### Files (delta)

- `cadjoint/fem/tesseracts/chain.py` — `_freeze_dc_surface` anchors the frozen
  crossings and projects them onto the traced zero set (mirroring `edge_hermite_data`'s
  degenerate-gradient fallback); new `_interior_relaxation` composes the shared
  Laplacian relaxation onto the tetfill tesseract's returned nodes.
- `cadjoint/fem/tetmesh.py` — `smooth_interior_delta` factored out of
  `recompute_tet_points` so the direct path and the DC chain relax through one
  operator; the `smooth_passes <= 0` branch is left as a concatenation so the direct
  path stays bit-identical.
- `cadjoint/fem/tesseracts/tetfill/` — **unchanged**. The black box stayed a black box.
- `tests/fem/test_tetfill.py` — 26 tests (was 20): `TestExpiredBrackets` freezes a bar
  whose faces sit 0.0014 above a lattice plane, checks that a 0.005 design step expires
  **304 of 336** brackets, and asserts the fill stays non-degenerate (min quality
  1.19e-2, zero inverted) and solvable — the old map produced **347 exactly-zero-volume
  tets** on that same step; plus a fixed-point/interior-follows check on the relaxation.
