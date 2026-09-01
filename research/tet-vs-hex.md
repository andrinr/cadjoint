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
