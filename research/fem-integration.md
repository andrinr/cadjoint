# FEM integration: SDF -> hex mesh -> jax-fem, end-to-end differentiable

Status: **working end to end** (2026-08-16). Design parameter -> SDF -> frozen-topology
hex mesh -> FEM solve -> objective -> `jax.grad`, validated against central finite
differences. This note records what runs today, the honest limits, and the follow-ups.

## What works today

| Piece | Status | Where |
|---|---|---|
| HEX8 volumetric mesher from any cadjoint SDF | working, no solver dependency | `cadjoint/fem/hexmesh.py` |
| Thermal (Poisson) solve on the hex mesh | working | `cadjoint/fem/simulate.py` |
| Linear elasticity + per-cell von Mises | working | `cadjoint/fem/simulate.py` |
| VTK export for ParaView | working (meshio) | `ThermalResult/ElasticResult.vtk_export` |
| Adjoint gradients w.r.t. node coordinates | working, FD-validated | `cadjoint/fem/backends.py` |
| End-to-end design gradient | working, FD-validated to ~1e-4 off kinks | `tests/fem/test_end_to_end.py` |
| Tesseract plugin ABI (local, no Docker) | working, gradients bit-identical to direct | `cadjoint/fem/tesseracts/thermal_jaxfem/` |
| Elastic solve packaged as a tesseract | working, gradients bit-identical to direct | `cadjoint/fem/tesseracts/elastic_jaxfem/` |
| Differentiable Dirichlet *values* (thermal, direct backend) | working via lifted solve, FD-validated | `JaxFemBackend.thermal` + `tests/fem/test_dirichlet_gradient.py` |
| Programmatic node selection (`Nodes` + boolean algebra) | working, serializable | `cadjoint/fem/selection.py` |
| Heat-flux (Neumann) BCs on the direct backend | working, analytic-validated | `JaxFemBackend.thermal` surface maps |

### Versions

- `jax-fem 0.0.12` on `jax 0.8.2` / Python 3.14 (macOS arm64). jax-fem ships an
  **empty `install_requires`**, so the `fem` extra in `pyproject.toml` lists its real
  imports explicitly: `fenics-basix 0.11.0`, `meshio 5.3.5`, `gmsh 4.15.2`,
  `petsc4py 3.25.4` (built from source in ~1m24s via the `petsc` PyPI package),
  `pyfiglet`, `scipy`. All coexist with the repo's pinned jax — no version conflicts.
- `tesseract-core 1.11.0` (with the `[runtime]` extra) + `tesseract-jax 0.4.1`.

### Timings (2.0 x 0.3 x 0.3 bar, 22x5x5 grid, 320 cells / ~500 nodes, M-series CPU)

- Forward thermal solve: ~0.75 s cold (jit warmup), ~0.08 s warm.
- Elastic cantilever solve (~1000 DOF): ~1.1 s cold.
- Forward + adjoint VJP roundtrip, warm: **direct 0.26 s, tesseract 0.39 s** —
  the tesseract boundary costs ~0.14 s/call (JSON+base64 payload encode/decode and
  endpoint dispatch; each call also rebuilds the jax-fem problem). Acceptable for
  interop, wrong default for inner-loop optimization — hence direct-in-process is
  the default backend.

## The gradient path: adjoint, not traced (report the truth)

`jax.grad` **cannot trace through** jax-fem's forward solver: assembly converts to
PETSc CSR (`TracerArrayConversionError` on `float64[1,8]` cell Jacobians — verified
empirically). The working path is jax-fem's `ad_wrapper`, a `jax.custom_vjp` whose
backward solves the adjoint system (`implicit_vjp`: one transposed linear solve +
`jax.vjp` through the residual). Consequences:

- Composes with eager `jax.grad`/`jax.vjp`; must **not** sit under `jax.jit`.
- Differentiable w.r.t. anything `Problem.set_params` consumes. Crucially,
  `Problem.initialize_geometric_quantities(fes_points=...)` recomputes all shape
  data through `jax.numpy`, so **nodal coordinates are a valid parameter**:
  coordinate gradients validated against FD (0.05% on a Poisson bar node).
- Dirichlet *values* are baked into DOF elimination at problem construction
  (`fe.vals_list`, outside `set_params`' parameter path — verified in
  `jax_fem/solver.py`: `implicit_vjp`'s `constraint_fn` only re-runs
  `set_params`, while `apply_bc_vec` reads the frozen `vals_list`), so they
  are **not** differentiable through jax-fem directly. The direct thermal
  backend therefore uses the standard **lift**: solve `T = u0 + g` with
  homogeneous Dirichlet conditions on `u0`, where `g` is the nodal field
  interpolating the prescribed boundary values (nonzero only at the
  Dirichlet nodes — any discrete lift with the right boundary trace is
  exact). The extra weak-form flux `k grad(g)` enters as an internal
  variable (`g`'s quad-point gradient from `fes[0].shape_grads`, itself
  recomputed differentiably), so `d(objective)/d(dirichlet value)` flows
  through the adjoint. FD-validated on the thermal bar (hot-end temperature
  as the design variable; observed agreement ~1e-8 relative, asserted at
  rtol 5e-2 in `tests/fem/test_dirichlet_gradient.py`). Elastic clamps are
  identically zero, so no elastic lift is needed yet; the tesseract thermal
  schema still treats Dirichlet values as static inputs.

### The end-to-end chain (frozen-topology doctrine)

1. Extract the mesh **once** at the nominal design: connectivity, the snapped-vertex
   set, and BC node sets are frozen (`HexMesh.base_points`, `.snap_mask`).
2. Per candidate parameter, `recompute_points(sdf, mesh)` re-runs only the Newton
   projection of the snapped vertices through the *traced* SDF (pure JAX, fixed
   iteration count); interior lattice points stay put.
3. `thermal_solve(..., points=...)` / `elastic_solve(..., points=...)` feeds the
   traced points into the solver's adjoint via `set_params`.

Measured on the cantilever with half-height h as the design parameter
(`J = sum(u^2)`): adjoint `dJ/dh = -172.694` vs central FD `-172.715` (**rel err
1.2e-4**) at h=0.16. At exactly h=0.15 the y/z half-extents tie and box-edge
vertices sit on a subgradient kink of the SDF's `max()`; AD returns the
subgradient while central FD averages the one-sided slopes (~5% apart). Real
lesson: symmetric nominal designs put snapped vertices on SDF kinks; evaluate
gradients off-tie or smooth the primitive.

One more autodiff trap fixed on the way: the projection's displacement clamp used
`jnp.linalg.norm`, whose gradient at zero displacement (vertex already on the
surface) is 0/0; the NaN leaks through `minimum()` into every cotangent. Guarded
via `sqrt(maximum(|d|^2, eps))` (`hexmesh.project_points`).

## Mesher notes (`cadjoint/fem/hexmesh.py`)

Voxelize (keep cells with `sdf(center) < 0`), share vertices through the lattice,
then Newton-project boundary vertices onto the zero set along the field gradient
(mirrors the viewer's `_project_to_seam`, single field). Total displacement is
clamped to half the cell diagonal — for a true distance field every boundary-face
vertex is within that bound of the surface (Lipschitz-1 argument), so snapping
reaches the zero set (sphere test: |sdf| < 1e-3 at all snapped vertices).
Inversion guard: a vertex's snap is reverted if any incident hex corner-tet
determinant would go non-positive (all-positive asserted in tests). Known
artifact: a vertex already on the surface but off a feature edge (e.g. the corner
ring where a bar's end face meets its sides) has `sdf = 0` and cannot improve, so
end-ring cells distort slightly (~3% local error on the misaligned thermal bar —
tested and tolerated explicitly). Cells use VTK/meshio HEX8 ordering, which
jax-fem consumes directly (it reorders to basix internally).

## BC selection design: programmatic node selections

**Now: first-class vertex selection** (`cadjoint/fem/selection.py`), replacing the
earlier `FaceSelector` face-group orientation ("the current face groups are not
so good to work with" — the face-group ids depended on the mesher's
gradient-axis tagging, which is opaque for anything non-box-like and useless
for sub-patches).

- `Nodes` is the factory namespace: `Nodes.box(min_corner, max_corner)`,
  `Nodes.sphere(center, radius)`, `Nodes.halfspace(point, normal)` (selects
  `dot(x - point, normal) >= 0`), `Nodes.side("+x", tol=None)` (axis-extreme
  plane; default tol resolves per mesh to half the smallest cell spacing), and
  `Nodes.predicate(fn)` as the code-only escape hatch (`fn` is vectorized:
  `(N, 3)` positions in, `(N,)` boolean mask out).
- Selections compose with **boolean algebra**: `&`, `|`, `~`.
- **Boundary restriction is implicit**: selections always resolve to boundary
  (surface) nodes, because boundary conditions only ever act on the surface.
  `~selection` therefore means "the rest of the surface", never the mesh
  interior.
- Evaluation is lazy: `selection.mask(mesh)` / `selection.resolve(mesh)` run
  against a concrete `HexMesh`, so one selection works across resolutions and
  remeshes.
- **Serialization contract**: `describe()` emits `{"kind": ..., numeric
  params}` with plain floats/lists; `selection_from_description(payload)`
  round-trips everything except predicates (`{"kind": "predicate", "name":
  fn_name}`, flagged via `selection.serializable == False`). Combinators
  describe as `{"kind": "and"|"or", "operands": [...]}` and `{"kind": "not",
  "operand": ...}`. Constructor signatures are deliberately literal-friendly
  (floats and 3-lists) so the viewer's next wave can write selections into
  scene source as patch operations (`Nodes.box([0, 0, 0], [1, 1, 1])`).

**BC semantics** (`cadjoint/fem/study.py`, `simulate.py`):

- Node-valued conditions — `Dirichlet(nodes, value)`, `Fixed(nodes)` — apply
  to the selected node set directly.
- Area-integrated conditions — `HeatFlux(nodes, flux)`, `Traction(nodes,
  vector)` — act on the boundary faces **spanned** by the selection: a quad
  carries the load iff all four of its corners are selected
  (`hexmesh.faces_from_nodes`). This matches jax-fem's own face-selection
  rule, so the spanned-face set and the solver's applied set coincide by
  construction; a selection spanning no complete face is an error raised
  before solving.
- Node sets still cross the backend boundary as plain int arrays (the
  Tesseract ABI is unchanged); inside jax-fem they become 2-arg location
  functions (`isin(index, set)`).

**Legacy face path**: the mesher's gradient-axis face groups
(`HexMesh.boundary_faces`), `select_faces(mesh, predicate)` and predicate
patches in `thermal_solve`/`elastic_solve` all still work — the viewer's
`/api/simulate` preview path is built on the group catalog and stays
untouched. New code and studies use node selections exclusively.

Later: **visual picking in the viewer** writes `Nodes.*` expressions into the
scene program via patch ops; a picked patch serializes through `describe()`
and lands in source as literal constructor calls.

## Plugin architecture (why Tesseract)

Direct in-process jax-fem is the **default backend and performance baseline**
(native JAX composition, no serialization). The Tesseract schema is the
**interop ABI**, not a mandatory wrapper:

- `cadjoint/fem/backends.py`: `SolverBackend` protocol —
  `thermal/elastic(points, cells, bcs, *, materials..., base_points)` returning
  JAX arrays with a VJP w.r.t. `points` (adjoint at minimum). `register_backend()`
  adds third-party solvers.
- `cadjoint/fem/tesseracts/thermal_jaxfem/tesseract_api.py`: the reference plugin.
  Typed pydantic `InputSchema`/`OutputSchema` (arrays only), opaque `apply`,
  `abstract_eval` for shape inference, and a hand-written
  `vector_jacobian_product` backed by the adjoint. `tesseract_jax.apply_tesseract`
  composes it into `jax.grad`; gradients agree with the direct path to 0 ULP in
  our test.
- **Local execution verdict: no Docker needed.** `Tesseract.from_tesseract_api`
  imports the module in-process (requires `tesseract-core[runtime]`; the bare
  package is missing runtime deps like mlflow). Containerized serving
  (`tesseract build` / `from_image`) is the distribution story, unverified here
  (no Docker on this machine).
- A third-party (even non-JAX) solver ships one `tesseract_api.py` with the same
  schema and endpoints — its `vector_jacobian_product` can wrap any adjoint
  (hand-derived, FEniCS, code-generated) — and plugs in via
  `TesseractBackend(api_path=...)` or `register_backend`.
- `cadjoint/fem/tesseracts/elastic_jaxfem/tesseract_api.py`: the elastic solve
  packaged the same way. Variable patch counts cross the fixed-rank schema as
  a union `fixed_nodes` set (clamps are all-zero, patch identity irrelevant)
  plus `traction_nodes`/`traction_offsets` (prefix offsets) with one traction
  vector per patch. Differentiable input: `points` only, matching the direct
  backend (materials and tractions are baked into weak-form closures).
  `TesseractBackend` loads both tesseracts lazily and routes by kind;
  gradients agree with the direct path to < 1e-9 in the tests. Differentiable
  material fields per-cell remain natural schema extensions.

## Visualization

Now: `result.vtk_export(path)` writes HEX8 + point/cell data (`temperature`,
`displacement`, `von_mises`) via meshio for ParaView.

Follow-up (needs viewer work, out of scope here): in-viewer heatmaps require a
WebGPU triangle-mesh pipeline — the current viewer raymarches SDFs and has no
indexed-mesh path. Spec: upload boundary quads (split to triangles) + per-vertex
scalars, colormap in fragment shader, plus a slicing mode that clips hexes
against a plane and shades the cut faces from the volumetric field. The mesher
already provides watertight boundary faces with normals; per-node fields come
straight from the results.

## End-to-end optimization example status

The gradient chain is proven (`tests/fem/test_end_to_end.py`): compliance of a
cantilever w.r.t. its half-height through mesh + solve, adjoint vs FD. A full
optimization loop (optax over multiple parameters, re-extracting topology when
the design drifts past the frozen mesh's validity) is the natural next step; the
missing piece is a trust-region rule for when to re-freeze topology.

## Test inventory

`tests/fem/`: solver-dependent files `pytest.importorskip` on `jax_fem` /
`tesseract_core` / `tesseract_jax`, so the suite skips (not fails) without
them. `test_hexmesh.py` (mesher only), `test_selection.py` (node-selection
layer: primitives, algebra, describe round-trip, `faces_from_nodes` — no
solver), `test_simulate.py` (incl. selection-vs-predicate parity and the
analytic heat-flux profile), `test_study.py`, `test_tesseract_backend.py`
(elastic parity + bit-identical gradients + a generous overhead-timing
bound), `test_dirichlet_gradient.py` (lifted Dirichlet values vs FD, through
node selections), `test_end_to_end.py`, plus the bracket demo and
render-payload files.

## Bracket demo: a realistic part through the whole chain

`scenes/bracket.py` is a playground scene of an L-bracket built from the
construction API: a base plate and a tapered vertical web extruded from
parameter-backed `PolygonProfile` sketches (`plate_thickness`,
`web_thickness` as named Scalars), a triangular gusset rib whose tip is tied
to a `rib_height` dimension by a `DistanceConstraint`, and two bolt holes
subtracted at constraint-pinned positions (`FixedConstraint` +
`DistanceConstraint` on the spacing). The scene compiles in the playground
and its SDF meshes watertight with Euler characteristic -2 — genus 2, i.e.
exactly the two bolt holes.

`examples/fem_bracket_optimization.py` mirrors the same geometry as a pure
parameterized SDF and runs the frozen-topology chain end to end: HEX8 mesh at
the nominal design, bolt regions clamped, a prying traction on the web tip's
outer face (chosen normal-restricted so the applied force does not scale with
the thickness being optimized), objective = total squared displacement + a
smoothed-volume mass penalty, and a few diagonally preconditioned gradient
steps on (web thickness, rib height) via `recompute_points` + the solver
adjoint. At the demo resolution the descent is monotone (~1% objective drop
in 4 steps, ~30 s total): the optimizer thins the slightly oversized web to
save mass while growing the gusset for stiffness. `tests/fem/test_bracket_demo.py`
guards all of it — scene compile, watertightness, one-step descent, and
adjoint-vs-FD agreement at the nominal (both parameters, rtol 5e-2; observed
agreement is ~1e-5, the tolerance is headroom).

Caveat worth carrying forward: with the web only one element thick, the raw
compliance sensitivity to web thickness is discretization-dominated (fully
integrated HEX8 locks in bending), so quantitative sizing of thin walls needs
either finer through-thickness resolution or an incompatible-modes element.
The gradient machinery itself is exact for the discrete model either way.

## Code parity: studies are scene-program citizens

Simulation follows the same doctrine as sketches and constraints: **code is the
source of truth, visual features are a layer on top**. `cadjoint/fem/study.py`
makes studies declarative and first-class:

- `ThermalStudy` / `ElasticStudy` are plain validated dataclasses declared
  directly in user programs and scripts. `.solve(sdf)` runs
  `sdf_to_hex_mesh` + the existing solvers and returns the usual result
  objects, so optimizers consume studies with no extra plumbing (a frozen
  mesh can be passed in for the frozen-topology gradient loop).
- Boundary conditions consume `Nodes` selections (see the selection section
  above): `Dirichlet(nodes, value)`, `HeatFlux(nodes, flux)`,
  `Fixed(nodes)`, `Traction(nodes, vector)`. Non-selection arguments are
  rejected at construction with a pointer to the `Nodes` factory.
- `.describe()` emits the JSON-ready payload the viewer needs (name, kind,
  resolution, domain, material, serialized BCs, each BC embedding its
  selection's description) — `json.dumps`-able, asserted in tests.
- `capture_studies()` mirrors `capture_constraint_solves` in
  `cadjoint/constraints/solve.py` (ContextVar + context manager): constructing
  a study inside the context registers it, so the compile worker can exec a
  user program and collect its declared studies in order. Verified against
  exec'd source, including nested-context isolation.
- `HeatFlux` **solves on the direct backend now**: with faces derivable from
  node selections, the flux enters jax-fem's weak form as a surface
  integral (`get_surface_maps` returning `-q`, so `k grad(T) . n = q`;
  positive flux heats the body). It composes with the lifted Dirichlet
  formulation unchanged (the surface term does not involve the lift).
  Validated against the analytic linear profile `T(x) = (q/k)(x + 1)` on
  the bar to 1e-6. The thermal tesseract schema does not carry fluxes yet;
  `TesseractBackend.thermal` raises `NotImplementedError` for them.
- `scenes/bracket.py` now declares a `bracket-pry` `ElasticStudy` in the
  scene program itself (bolt-ball clamps, spanned-face traction on the
  outer web wall), mirroring the optimization example's selections.

## Precision hygiene: x64 is scoped to the solve

jax-fem requires float64 and `jax_fem.solver` even flips `jax_enable_x64` at
module import. Flipping it globally poisons every float32 computation later
in the same process — the concrete symptom was
`tests/meshing/test_adaptive.py::test_identical_crossing_edges[union]`
failing when it ran after `tests/viewer/test_simulate.py` (the viewer's
forward solves flipped x64 via the old `_require_jax_fem`, and the sparse
octree pruning's dense-vs-sparse equivalence is x64-sensitive for the union
case). Fix: every backend call now wraps itself in `_x64_scope()`
(`backends.py`), enabling x64 for its own duration and restoring the
caller's setting afterwards — which also undoes the import-time flip, since
`jax_fem.solver` is first imported inside the scope. Forward-only callers
(the viewer preview) are therefore leak-free. Callers that differentiate
*through* a solve run the backward pass after the scope has exited and must
enable x64 process-wide themselves — `tests/fem/conftest.py` does it as a
package fixture and the optimization example does it at import.

Next wave (viewer side, not in this change): the Simulate panel becomes a
patch layer — `add_study` / `add_study_bc` patch ops edit the study
declarations in the program text, and the worker ships `describe()` payloads
to the frontend. `/api/simulate`'s ad-hoc BC payload then survives only as a
preview path for not-yet-committed panel state; committed studies always
live in code. Study defaults reuse the worker's simulate-domain convention
(bounds `(-3,-3,-3)`, size `(6,6,6)`), overridable per study.

Study tests: `test_study.py` (24) — construction/validation, JSON round-trip,
selector-vs-predicate equivalence, exec capture, and bar-scene solves
reproducing the direct thermal/elastic results through the study path.

## CalculiX: a Fortran solver with a native adjoint behind the Tesseract ABI

Status: **working end to end** (2026-08-16). Same mesh, same BCs, same
`jax.grad` call — but the elastic solve and its shape adjoint run in a
CalculiX 2.23 subprocess (`ccx`, Fortran 77/90 + C, GPL-2 held at the
process boundary: decks in, result files out, no linking).

### Getting the binary (what actually worked)

No Homebrew formula exists (`brew search calculix` is empty) and no conda
manager was installed, so: standalone micromamba + conda-forge, which ships
a native arm64 `ccx` 2.23 for macOS:

```sh
mkdir -p ~/.local/share/cadjoint && cd ~/.local/share/cadjoint
curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj bin/micromamba
./bin/micromamba create -y -p ./ccx-env -c conda-forge calculix
./ccx-env/bin/ccx -v   # "This is Version 2.23"
export CADJOINT_CCX=~/.local/share/cadjoint/ccx-env/bin/ccx
```

`cadjoint.fem.calculix.find_ccx` resolves the binary via `CADJOINT_CCX` / `CCX`
env vars or `PATH`; every live test skips cleanly when none is found. The
env above lives under `~/.local/share/cadjoint` on this machine (a /tmp
install evaporated once already — keep it somewhere durable).

### Integration shape

- `cadjoint/fem/calculix.py` — deck writer (`write_elastic_deck`: C3D8 from
  HexMesh, corner order is byte-identical to VTK so cells serialize 1:1;
  `*NSET`+`*BOUNDARY` clamps; tractions as `*CLOAD` **consistent nodal
  forces** from 2x2 Gauss on the bilinear boundary quads, exactly matching
  jax-fem's surface integration), `.dat`/`.frd` parsers, the sensitivity
  correction (below), a `CalculixBackend`, and `strain_energy_solve` for
  study-style patch arguments.
- `cadjoint/fem/tesseracts/elastic_calculix/tesseract_api.py` — mirrors the
  `elastic_jaxfem` schema, adds a `strain_energy` output; `apply` = write
  deck, run `ccx`, parse; `vector_jacobian_product` = the `*SENSITIVITY`
  adjoint run + normal-projection chain rule.
- `backend="calculix"` is registered in `cadjoint/fem/backends.py` (lazy
  import), so `elastic_solve(..., backend="calculix")` and
  `ElasticStudy.solve(..., backend="calculix")` run forward through ccx;
  `examples/fem_bracket_optimization.py --backend calculix` runs the whole
  design-parameter gradient loop through the ccx adjoint.

### The ccx 2.23 STRAINENERGY sensitivity misses a term (found, fixed, FD-proven)

The `*SENSITIVITY` step (design response STRAINENERGY, `*DESIGN VARIABLES,
TYPE=COORDINATE`) writes per-design-node normal-projected sensitivities to
the `.frd` as `DFDN` (raw) and `DFDNFIL` (mass-matrix-smoothed — solve of
`M s = g` over the design surface, per `filterbackwardmain.c`; a *field*,
not the discrete gradient). Validating raw `DFDN` against central FD of
`E = f·u/2` showed the MASS response is **exact** (matches the analytic
volume derivative to 6 digits) but STRAINENERGY was off by a
*non-constant* factor (1.4–2.0x across nodes, 4.5x on a degenerate
single-element cube) — so neither raw nor rescaled values are usable as-is.

Reading the 2.23 sources pinned it: `objective_shapeener_dx.f` computes the
frozen-displacement partial by accumulating only the stress-times-
strain-increment `sigma . d(eps)` over the perturbed volume and never adds
the Jacobian-variation term `w * d(detJ)` of the energy integrand. The true
fixed-load shape derivative is therefore

```
dE/ds_i = DFDN_i + sum_{e∋i} sum_q w_q detJ_q (grad N_i(q) . n_i)
```

with `w` the strain-energy density of the (unperturbed) solution and `n_i`
the outward node normal ccx itself writes to the `NORM` frd block. The
correction is pure post-processing (`energy_volume_gradient`, closed form
`d(detJ)/dx_ia = detJ * dN_i/dx_a`) computed from ccx's own displacement
output — no extra solver runs. The identity was confirmed to 6 significant
digits on the cube and to rel. 2e-4 per node (faces, edges) against central
FD with ccx forward solves on a 4x2x2 bar.

### Numbers (bar mesh from the existing tests: 180 cells / 336 nodes)

- **Forward parity** ccx vs jax-fem, identical discrete system (fully
  integrated trilinear HEX8 both sides): max deviation **1.5e-7 relative**
  to the peak displacement. The floor is ccx's 6-significant-digit text
  output, not the discretizations — C3D8 and jax-fem's HEX8 are the same
  element.
- **Adjoint vs FD** (corrected sensitivities, ccx forward solves for FD):
  agreement to **rel. 2e-4** at every checked design node.
- **Three gradient paths on one mesh** (ccx `*SENSITIVITY` adjoint vs
  jax-fem `ad_wrapper` adjoint vs FD, strain energy objective, 228 design
  nodes): max deviation **4.5e-5 of the gradient scale**; per-node relative
  error ≤ 3.2e-3 on entries above 1% of scale — again set by the 5–6 digit
  `.frd`/`.dat` text precision plus DFDN/correction cancellation.

### VJP coverage (honest contract)

Differentiability through the ccx tesseract is **objective-valued**:

- Supported: cotangents on the `strain_energy` output (compliance = 2E for
  fixed loads), w.r.t. `points`. `jax.grad` of the bracket objective
  (compliance + smoothed mass) flows through `recompute_points` into the
  SDF design parameters — demonstrated by the example's `--backend
  calculix` flag (objective decreases, gradients finite and stable).
- Rejected with `NotImplementedError`: nonzero cotangents on the raw
  `displacement` field — ccx exposes adjoints for its built-in design
  responses only, not for arbitrary functionals. Displacement-valued
  objectives (e.g. the default example's `sum(u^2)`) stay on the jax-fem
  backends. (ccx's ALL-DISP response could add the `sqrt(sum u^2)`
  functional later; not wired up.)
- Geometry contract: the gradient is nonzero only at boundary (design)
  nodes and only along ccx's outward node normals — tangential and
  interior components are zero. That is exact in the continuum limit
  (in-surface/interior mesh motion does not change the shape) and is
  precisely the component the Newton-snapped mesher consumes, but it is
  not the full discrete `dE/d(points)`. At traction-loaded nodes the VJP
  holds the consistent nodal loads fixed (neglects the load-area
  derivative); the gradient-path comparison therefore masks clamped and
  loaded patches.

### Caveats

- ccx prints results with 5–6 significant digits (`.dat` `%13.6E`, `.frd`
  `%12.5E`); everything downstream inherits that floor. Fine for
  optimization, unsuitable for tight bitwise cross-checks.
- Each objective evaluation is a fresh ccx process (~50 ms small meshes,
  plus one extra factorization for the sensitivity solve); the deck writer
  is O(mesh) Python string work. Wrong default for inner loops — the
  in-process jax-fem backend stays the default; `backend="calculix"` is
  the interop/demo path.
- Thermal is not wired (ccx does heat transfer; out of scope here —
  `CalculixBackend.thermal` raises with a pointer to the jax-fem
  backends).
- The DFDN correction is validated against ccx **2.23** behavior. If a
  future ccx fixes `objective_shapeener_dx.f`, the FD-backed live tests
  (`tests/fem/test_calculix.py::TestLiveAdjoint`) will catch the double
  count immediately.

Tests: `tests/fem/test_calculix.py` (20) — deck-writer golden file,
parser fixtures (`.dat` displacements/stresses, `.frd` DISP/NORM/SENENER
with run-together fixed columns), correction-term FD identity (no binary
needed), and live: cube-vs-theory, forward parity, von Mises parity,
sensitivity-vs-FD, three-path gradient agreement, tesseract roundtrip +
displacement-cotangent rejection. All live tests skip without a binary;
full `tests/fem` = 123 green with one.

## First-class meshes and results: SimMesh, SimulationResult (2026-09)

The user directives behind this pass: more control over meshing, meshes and
results inspectable as stored intermediate objects, object-level selection
of what participates in a simulation, vertex/area BC selection — all "as
long as this is fully differentiable through meshing."

### SimMesh: meshing intent as a scene-program citizen

`cadjoint/fem/simmesh.py` adds `SimMesh(name, resolution, domain=None,
bounds=None, size=None, padding=0.1)` — a mutable dataclass declared in the
scene program and captured by `capture_sim_meshes()` (a ContextVar registry
parallel to `capture_studies()`, same nesting/isolation semantics).  Design
decisions:

- **One meshing path.** Studies take `mesh=<SimMesh or declared name>`
  (name resolution happens at construction against the active capture
  context).  A study without one wraps its own
  resolution/bounds/size/domain into an *anonymous* SimMesh (capture
  suppressed), so implicit and explicit meshing are literally the same
  code.  Passing `mesh=` **and** resolution/bounds/size/domain on the
  study is a hard error — meshing intent lives in one place.
- **Domain selection.** `domain=` (on SimMesh, or on an implicit-mesh
  study) is any SDF/callable; when set, `build()` meshes it instead of the
  scene SDF passed to `solve()`.  `describe()` records
  `{"name": getattr(domain, "name", None), "type": type name}` — the
  bracket scene names its Difference (`bracket.name = "bracket"`) so the
  viewer can patch by name.
- **Auto bounds.** With bounds/size omitted, `grid()` scans the default
  volume (-3..3, 33^3 lattice) for inside samples and pads the tight box
  by `padding` plus one scan spacing.  Explicit bounds skip the scan.
- **Caching.** `build(sdf)` caches the HexMesh on the instance, keyed on
  (resolution, bounds, size, padding) equality plus *identity* of the
  meshed field object; `rebuild=True` forces re-extraction (needed after
  in-place parameter mutation, which the cache cannot see).  The cache is
  what lets several studies share one extraction and what serves the
  frozen-topology mesh to a traced `solve(points=...)`.
- **Inspection.** `quality()` returns per-element arrays; `inspect()` a
  JSON summary (counts, point bounds, grid, min/mean/max per metric).
  Metrics live in `hexmesh.py`: `scaled_jacobians(points, cells)` — per
  corner det(e1,e2,e3)/(|e1||e2||e3|) over the corner tets already used by
  the inversion guard, element value = min over its 8 corners (cube = 1,
  inverted < 0) — and `aspect_ratios(points, cells)` — max/min of the 12
  edge lengths.  Both are vectorized numpy, O(C).

### SimulationResult: solves you can look at twice

`cadjoint/fem/result.py`: `study.solve()` now returns
`SimulationResult(name, kind, field, solution, sim_mesh)` wrapping the
low-level `ThermalResult`/`ElasticResult` (which stay the raw
`thermal_solve`/`elastic_solve` API — backends untouched).  Delegating
properties keep the viewer's existing accesses (`result.temperature`,
`result.von_mises()`, `result.mesh`) working unchanged.  The instance is
stored on the study as `last_result` for re-inspection without re-solving.

Traced vs concrete is explicit: `temperature`/`displacement` and the
objective helpers `mean()`/`max()` (temperature, resp. guarded
displacement magnitude `sqrt(|u|^2 + 1e-30)` — the guard keeps gradients
finite at exactly-clamped nodes) stay JAX arrays and differentiate through
a traced solve; `nodal_scalar()` (display field: temperature or
cell-to-node von Mises), `describe()` (counts, range, per-field
min/mean/max) and `to_vtk()` (reuses the meshio writer) are concrete-only.
Von Mises stays the numpy post-process it was — putting it in the
objective would silently drop the geometry term of its derivative.

### Differentiability through the named-mesh path (FD-proven)

`tests/fem/test_bracket_demo.py::TestNamedMeshGradient` (successor of the
example-driven optimization test, whose Adam smoke moved next to the
example): SimMesh(24x17x13, domain=nominal bracket SDF) -> build (742
cells / 1422 nodes, 836 snapped; min scaled Jacobian 0.352) ->
`ElasticStudy(mesh=...)` with the bolt-clamp / web-tip-load selections ->
per theta `recompute_points` -> `solve(points=...)` -> `result.mean()`.
Adjoint vs central FD (eps 1e-3) at the nominal design:

| parameter        | adjoint      | central FD   | rel. diff |
|------------------|--------------|--------------|-----------|
| web_thickness    |  0.05098049  |  0.05097652  | 7.8e-5    |
| rib_height       | -0.00182567  | -0.00182565  | 9.4e-6    |
| plate_thickness  | -0.39855593  | -0.39856666  | 2.7e-5    |

(The positive web component is real: the objective is mean |u| and the
loaded outer wall moves with the web thickness; only plate thickness has
an invariantly negative sign, which the test asserts.)

### Reference scene

`scenes/bracket.py` now unions the bracket with a rendered mounting slab
(`mount`, plugs the bolt holes from below — scene surface genus 0, bracket
domain still genus 2, both watertight) and declares
`bracket_mesh = SimMesh(name="bracket-mesh", domain=bracket, ...)` +
`pry_study = ElasticStudy(..., mesh=bracket_mesh)` — the viewer wave's
starting point for mesh cards, domain badges, and result inspection.

Study constructor note for the viewer wave: `resolution` is now optional
(`None` when mesh-backed) and material parameters are keyword-only
(`dataclasses.KW_ONLY`); every existing call site already used keywords.
`describe()` gained `"mesh"` (SimMesh name or null) and `"domain"`
(name/type dict or null); `bounds`/`size` may now be null (auto bounds).

Tests: `tests/fem` = 162 green with a live ccx binary (26 new in
`test_simmesh.py`, 12 new study/result tests, bracket demo reworked);
`tests/viewer` = 200, `tests/test_playground.py` = 61 — both untouched.

## Solver-tesseract schema deltas: TET4/TET10 cells + thermal fluxes (2026-09-01)

Schema changes to the packaged solver tesseracts (element type inferred
from `cells.shape[1]`: 4/8/10; `(None, None)` in the schema):

- `elastic_jaxfem`: + `traction_faces (None,3) Int32`,
  `traction_face_offsets (None,) Int32` (exact tet face targeting; empty =
  node membership). HEX8 path byte-identical to before.
- `thermal_jaxfem`: + `flux_nodes/flux_offsets (None,) Int32`,
  `flux_values (None,) Float64`, `flux_faces (None,3) Int32`,
  `flux_face_offsets (None,) Int32`. Differentiable inputs unchanged
  (points, conductivity, source); flux values static like the direct
  backend's closures.
- `elastic_calculix`: + the two traction-face fields for parity with the
  base-class input dict (HEX8-only, must be empty).
- `TesseractBackend` passes the new fields (empty face arrays on hex);
  heat-flux BCs now route through `backend="tesseract"` — parity with the
  direct backend < 1e-9 (`tests/fem/test_tesseract_backend.py`).
- New `cadjoint/fem/tesseracts/chain.py`: `freeze_study_chain(study,
  sim_mesh, field)` — the frozen mesher+solver chain behind
  `Optimization(gradient_path="tesseract")` (see research/tet-vs-hex.md
  for the measurements and the default recommendation).

Tesseract schemas reject extra input keys (pydantic extra=forbid via
tesseract-core), so every schema fed by `TesseractBackend`'s input dict
must carry the union of its fields — that is why `elastic_calculix` grew
the (rejected-if-set) face fields.

## Container conformance: the five Tesseracts, actually built (2026-09-01)

Until now the packaged Tesseracts only ever ran in-process via
`Tesseract.from_tesseract_api`. This section records what happened when
all five were built into real Docker images with `tesseract build` and
exercised through a served HTTP client, on macOS 15 / Apple Silicon,
Docker 29.7.2, `linux/arm64`, tesseract-core 1.11.0.

### Conformance against the installed SDK

Everything was checked against the *installed* contract, not against
memory: `tesseract_core.sdk.api_parse.TesseractConfig` /
`TesseractBuildConfig` (the pydantic models behind `tesseract_config.yaml`,
`extra="forbid"` on both), `sdk/templates/Dockerfile.base` (where each
`build_config` field actually lands), and `sdk/engine.py`
(`parse_requirements`, `_stage_local_dependency`, `prepare_build_context`).

Confirmed mechanics worth writing down:

- **Local dependencies.** Requirement lines starting with `.`, `/` or
  `file://` are split out of `tesseract_requirements.txt`, `copytree`'d into
  `<context>/local_requirements/<name>` and rewritten to
  `./local_requirements/<name>`; extras (`[fem]`) are preserved. The same
  staging runs for the `pip:` sub-list of a conda
  `tesseract_environment.yaml`. So `../../../..` from a package directory is
  the supported way to install cadjoint itself. Staging the 1.9 GB repo root
  takes ~9 s on APFS (`copytree` clones); the generated `.dockerignore`
  drops `.venv`, `.git` and `__pycache__` before the context is sent to the
  daemon, but nothing else (`node_modules`, `native/target` do travel).
- **System dependencies.** `build_config.extra_packages` is apt-only and is
  rendered *twice* — once in the build stage, once in the run stage — which
  is what makes an apt-installed runtime library survive the multi-stage
  `COPY`.
- **Custom build steps.** `build_config.custom_build_steps` are injected
  verbatim into the run stage, after `package_data` and before the
  `tesseract-runtime check`, while the image is still `USER root`. The build
  context is shared with the build stage, so `COPY
  ["__tesseract_source__/…", …]` reaches the package's own source files.
- **Mutual exclusions.** `python_version` is rejected together with the
  conda provider and with `inherit_base_image_packages`. Versions must match
  `^\d+\.\d+\.\d+[a-zA-Z-0-9]*$`.

The inherited configs were schema-valid but two of the three mechanisms
they relied on did not survive contact with a real build (below).

### Why three packages had to move from pip to conda

`thermal_jaxfem` and `elastic_jaxfem` declared `../../../..[fem]` +
`python_version: "3.12"`. That build fails on Linux, and not by accident:

- **gmsh** publishes wheels for `manylinux_2_24_x86_64`, `macosx_*` and
  `win_amd64` only — no aarch64 wheel exists, so the resolver refuses the
  `[fem]` extra outright on this machine.
- **petsc4py** publishes *no* wheels on PyPI at all, for any platform or
  version (checked 3.23.7 … 3.25.5: zero `.whl` files). Its sdist runs
  PETSc's `configure`, which is why the first build died with
  `RuntimeError: 256` out of `setup.py`. Retargeting to `linux/amd64` fixes
  gmsh but not this.

Both are hard imports on the used path (`jax_fem.solver` does `from petsc4py
import PETSc`; `jax_fem.generate_mesh` does `import gmsh`), so neither can be
dropped. conda-forge carries both for `linux-aarch64`, so those two packages
now declare `build_config.requirements.provider: conda` with
`base_image: condaforge/miniforge3:latest` and a
`tesseract_environment.yaml`; jax-fem itself (not on conda-forge, empty
`install_requires`) and cadjoint stay in the `pip:` sub-list, deliberately
*without* the `[fem]` extra so the extra cannot drag gmsh back in from PyPI.

`elastic_calculix` moved for the same structural reason with a better
outcome: it declared `extra_packages: [calculix-ccx]`, and Debian's package
is **ccx 2.20** — older than the 2.23 whose `STRAINENERGY`/DFDN behaviour
the adjoint correction in `cadjoint/fem/calculix.py` is written against.
conda-forge ships `calculix` **2.23** for linux-aarch64, so the package now
installs that and pins `CADJOINT_CCX=/python-env/bin/ccx`.

### Build results

| Package | Provider | Build | Image | Fix forced by the build |
| --- | --- | --- | --- | --- |
| `mesher` | pip | 39 s | 1.36 GB | none — inherited config built as written |
| `qef_native (retired 2026-09-02, see native-mesher.md)` | pip | 74 s | 1.39 GB | base image bookworm → trixie; `ca-certificates` |
| `elastic_calculix` | conda | 87 s | 2.57 GB | apt `calculix-ccx` 2.20 → conda-forge `calculix` 2.23 |
| `thermal_jaxfem` | conda | 181 s | 5.51 GB | pip `[fem]` → conda provider (gmsh/petsc4py) |
| `elastic_jaxfem` | conda | 194 s | 5.51 GB | same |

The native package needed two corrections beyond the inherited config.
Debian bookworm's cargo is 0.66 (rustc 1.63) and cannot read this crate's
**Cargo.lock format v4** (needs cargo ≥ 1.78), so the base image moved to
`debian:trixie-slim` (cargo 1.85) rather than dropping the lockfile — the
build now runs `cargo build --release --locked`. The slim image also has no
`ca-certificates`, so cargo could not reach `index.crates.io`
(`[77] Problem with the SSL CA cert`). The toolchain is installed, used and
`apt-get purge`d inside one `RUN`, which is safe because a Rust cdylib links
libstd statically.

### Container round trips (served image vs. in-process, same inputs)

Every number below is a served `Tesseract.from_image(...)` HTTP call
compared against `Tesseract.from_tesseract_api(...)` in the host process.

| Tesseract | Case | Forward | VJP |
| --- | --- | --- | --- |
| `mesher` | HEX8, 9³ lattice, 117 pts / 56 cells | points max abs diff **1.11e-16**; cells and surface mask identical | `field_values` bar max abs diff **6.66e-16** (max abs value 3.31) |
| `thermal_jaxfem` | TET4 bar, 42 pts / 72 cells, Dirichlet + flux faces | temperature max abs diff **2.78e-16** (max T 0.551) | `points` **4.88e-15**, `conductivity` **6.66e-16**, `source` **3.33e-16** |
| `elastic_jaxfem` | TET4 bar, 42 pts / 72 cells, exact face traction | displacement max abs diff **1.16e-16** (max u 1.30e-2) | `points` **6.94e-16** (max abs 6.71e-2) |
| `qef_native` | 24 cells / 96 Hermite edges, λ=0.05 | vertices **0.0** (bit-identical) | `points_bar` **0.0**, `normals_bar` **0.0** |

`elastic_calculix` has no in-process reference here (the host has no ccx
binary), so it was validated three ways instead, all inside the container:
`ccx -v` reports **2.23** at `/python-env/bin/ccx`; on a 22×5×5 hex bar the
served `apply` gives `strain_energy = 1.508852250e-2`, `max|u| = 3.353357e-1`
in 0.38 s, and the served `vector_jacobian_product` (0.03 s) matches central
finite differences taken through the *container's own* `apply` to **6.5e-4**
relative on the three largest gradient rows — the ~6-significant-digit limit
of ccx's text output, consistent with the 2e-4·scale bound already recorded
for the in-process path. Cross-solver, the container's ccx displacement
matches in-process jax-fem on the same mesh to **1.2e-7** relative.

Serve latency is ~1 s for every image; the first `apply` carries a
one-off JAX/import cost (mesher 1.8 s, jax-fem images 3–3.6 s) and later
calls run at in-process speed.

### Two limitations of the served boundary (tesseract-core 1.11)

1. **Zero-size arrays cannot cross HTTP.** `runtime/array_encoding.py`'s
   `get_array_model` maps a polymorphic (`None`) dimension to `PositiveInt`,
   so an encoded array with a `0` in its shape fails the `EncodedArrayModel`
   branch, falls through to `python_to_array`, and is reported as
   `array_non_numeric`. Reproduced minimally: `Array[(None,), Int32]` accepts
   a base64-encoded `(3,)` array and rejects a `(0,)` one, even though
   `Base64ArrayData` and `_load_base64_arraydict` both handle the empty
   buffer fine. Consequences: the mesher's discovery mode (empty `point_ids`
   / `cell_template`) is in-process only, and any "this feature is unused"
   convention built on an empty array needs a non-empty spelling. For that
   reason `elastic_calculix` now rejects only a *populated* face-patch set
   (`offsets[-1] > offsets[0]`) instead of any non-empty `traction_face_offsets`
   array, so a served caller can say "no face targeting" with `[0, 0]`. The
   upstream fix is `NonNegativeInt` for polymorphic dimensions.
2. **TetGen topology is not portable.** The same field and the same TetGen
   0.8.4 produce **182 points / 673 cells** on macOS-arm64 and **185 points**
   in the Linux-aarch64 container — Steiner insertion differs with the
   compiler/libm, and the mesher's frozen-topology promise correctly rejects
   the mismatch. A frozen topology must therefore be discovered on the same
   platform that will execute it. HEX8 mode (voxelize + Newton-snap, pure
   numpy/JAX) is bit-reproducible across both, which is why the container
   parity test uses it.

### Housekeeping

`native/run_<uuid>/logs/` directories are per-invocation artifacts dropped by
`tesseract-runtime` next to the Tesseract it executes, not build output; two
stale ones were deleted and `/run_*/` added to `native/.gitignore`.

## The projection's floored denominator was a 1e12-per-iteration adjoint amplifier (2026-09-01)

`project_points` (`cadjoint/fem/hexmesh.py`) is the one Newton projection the whole
mesh half of the gradient chain runs through: `sdf_to_hex_mesh`'s boundary snap,
`recompute_points`, `sdf_to_tet_mesh`'s pre-tetrahedralization projection, and
`recompute_tet_points` all call it. Its step is `-f(x) grad f / |grad f|^2`, and the
denominator was guarded as `jnp.maximum(squared, 1e-12)`.

**The defect.** A floor is not a guard. Where `squared` falls under `1e-12` the
denominator stops being a function of `x` and becomes the *constant* `1e-12`, so the
step's Jacobian collapses to `value * Hessian / 1e-12` — a 1e12 amplification per
iteration, compounding over all eight. The forward pass stays perfectly finite and
smooth (the step itself is `value * grad / 1e-12`, and where `grad` is a bit-exact
zero it is exactly zero), so nothing downstream of the forward value ever hinted at
it. Reverse mode, meanwhile, was returning numbers around 1e+68.

**Where the dead gradients come from.** The same degeneracy `edge_detection._refine`
already documents: a vertex lands bit-exactly on a wall of a polygon-derived SDF,
where `ExtrudedPolygon.sdf`'s `jnp.maximum(d2, dz)` ties and JAX splits the
subgradient evenly between two branches that cancel. On the starter heat sink at its
declared 18x13x11 grid, 17 of the 860 DC surface vertices start with `|grad| ==
0.0` exactly — nine on the fin-tip corner line (`z = 0.85`, a profile vertex of the
comb) and eight on the slug-bottom rim (`z = -0.18`) — and 110 to 131 more join them
after the first Newton iteration lands them on those same walls.

**Why every shipped test missed it.** At the freeze design (`fin_depth = 1.2`) those
nodes carry `value == 0.0` exactly, so `value * Hessian / 1e-12` is zero and the
measured maximum per-node amplification is the healthy 0.503 that
`tests/fem/test_starter_tet.py` records. Step off the freeze design by any amount and
the residual becomes ~1.2e-8 — small in the forward pass, 1.2e+4 after the floor.

**The fix.** The repo's double-`where` idiom, applied so the *gradient* is also zero
when the guard trips:

```python
usable = jax.lax.stop_gradient(squared) > _MIN_GRADIENT_SQUARED   # 1e-8
step = jnp.where(usable, value[:, None] * gradient / jnp.where(usable, squared, 1.0), 0.0)
```

The inner `where` keeps the suppressed division finite so no NaN can reach the
cotangent; the outer one zeroes value and derivative together. The threshold is on
`|grad|^2`, not on the step: a signed *distance* field has `|grad| = 1` almost
everywhere, so `|grad| < 1e-4` never means "a shallow field", it means the
linearization is float noise. That is 1e4 more margin than the old floor had, and the
measured field has a clean gap — the starter's surface nodes are either exactly 0 or
above 0.195.

**Before / after**, starter heat sink, `fin_depth`, objective `sum(points**2)` through
`recompute_tet_points(..., smooth_passes=2)` on the frozen 860-vertex DC surface:

| `fin_depth` | adjoint (floored) | adjoint (guarded) | central FD (1e-5) |
|---|---|---|---|
| 1.10   | -3.867e+68 | +1.186e+02 | +118.574 |
| 1.15   | -3.532e+68 | +1.239e+02 | +123.851 |
| 1.1873 | +2.648e+69 | +1.278e+02 | +127.787 |
| 1.19   | +5.438e+69 | +1.281e+02 | +128.072 |
| 1.21   | +4.604e+01 | +4.591e+01 | +45.913  |
| 1.25   | +4.750e+01 | +4.737e+01 | +47.375  |

Per-node boundary Jacobian `|dx/d fin_depth|` over the 860 surface nodes: max
**4.2e+68 -> 0.500** (median 0.0 either way; 9-10 nodes exceeded 1e3 before, none
after). Note the old code was *already correct* above the freeze design (1.21, 1.25)
— the blow-up is one-sided, which is another reason it hid.

**The forward pass improved too.** The floored step was flinging 14-35 of the 860
vertices off the zero set (it multiplied a ~1e-8 residual by a ~4e4 factor and let
the displacement clamp absorb the rest). Max `|sdf|` residual over the projected
surface: **1.62e-2 -> 9.0e-5**; mean **4.14e-5 -> 4.25e-7**. The sphere hex test's
snapped-vertex residual is 1.1e-16, unchanged.

**`tetmesh` shares the code, not a second copy.** `cadjoint/fem/tetmesh.py` imports
`project_points` from `hexmesh` and calls it from both `sdf_to_tet_mesh` and
`recompute_tet_points`, so the one fix covers the tet path — which is in fact the
path the defect was measured on. `smooth_interior_delta`'s
`jnp.maximum(degrees, 1.0)` is not the same idiom: `degrees` is a frozen numpy
adjacency count, never traced. The other `np.maximum(..., 1e-30)` guards in `tetmesh`
and `hexmesh` are in numpy quality metrics that nothing differentiates.

**Not fixed, because it is not a defect:** `project_points`' displacement clamp still
floors at `jnp.maximum(squared_displacement, 1e-24)`. That one is benign — below the
floor, `scale = jnp.minimum(1.0, max_step / 1e-12)` selects the constant `1.0`
branch, so no frozen denominator ever reaches the output.

**Regression fence:** `tests/fem/test_projection.py`. A fast half builds a wedge whose
crease has a bit-exact dead subgradient (a minimal model of the polygon tie) and
pins the guard's contract: the point does not move, the adjoint is exactly zero
(**4.0e+04** on the old code), and no NaN reaches the cotangent. A starter half
sweeps `fin_depth` over the measured designs and asserts the per-node Jacobian stays
under 1e3, the projected vertices stay within 1e-3 of the zero set (the old code
misses by 16x), and the adjoint matches central differences off the freeze-design
kink.

**Re-validation of everything downstream of this path** (`pytest tests/fem -q`:
**291 passed, 14 skipped** — the 282/14 baseline plus the nine new projection tests,
with no existing test edited; ruff clean). Every shipped gradient check still passes
inside its own fence, but the numbers moved slightly, because the forward projection
genuinely changed at the 14-35 nodes it used to fling off the surface — and because a
different projected DC surface makes TetGen insert a different number of Steiner
points (985 -> 990 on the repro grid). Recorded here since the docstrings in
`test_starter_chain.py` and `test_starter_tet.py` still quote the pre-fix values:

| measurement | pre-fix | post-fix |
|---|---|---|
| DC chain `J` / adjoint / FD(1e-3) | 1.15330727 / -0.166356900 / -0.166357126 | 1.147698874 / **-0.161124633** / -0.161124858 |
| direct-path adjoint (same mesh) | -0.134186049 | **-0.130809776** |
| DC/direct ratio (fenced 1.0-1.6) | 1.240 | **1.232** |
| starter tet @ 1.25, adjoint vs FD(1e-3) | -0.0890 (3.6e-7 rel) | **-0.053691748** vs -0.053691768 (3.7e-7 rel) |
| starter tet @ the freeze kink: adjoint / bwd / fwd / central | -0.0890 / - / - / -0.0730 | **-0.091623843** / -0.091633779 / -0.058621438 / -0.075127609 |
| hex chain `J` / adjoint / FD(1e-4) | 0.988622627 / -0.072205495 / -0.070080773 | **identical** |
| bracket bar chain, adjoint vs FD | +1.994235 / +1.994235 | **identical** |
| bracket plate, adjoint vs FD(1e-3) | -350.3737 / -348.8485 | **identical** |

The hex chain and the bracket demo are bit-identical, which is the expected control:
their start points are lattice vertices where `|grad|` never drops below 0.707, so the
guard never trips. Only the DC/tet path, whose start points are surface vertices
sitting *on* the polygon walls, was ever affected. Adjoint-vs-FD agreement is
unchanged in quality (1.4e-6 on the DC chain, 3.7e-7 on the tet path), and both
descents stay monotone (DC 1.14769887 -> 1.14641564 -> 1.14518502).

Unrelated and still open: the frozen-topology objective is genuinely **kinked at the
freeze design** and the adjoint correctly returns the left derivative there. That is
the extrusion's cap-branch switch, not this defect — it is present identically before
and after the fix (measured left slope 128.6, right slope 45.8 on `sum(points**2)`)
and is already pinned by its own test in `test_starter_tet.py`.
