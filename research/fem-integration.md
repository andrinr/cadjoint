# FEM integration: SDF -> hex mesh -> jax-fem, end-to-end differentiable

Status: **working end to end** (2026-08-16). Design parameter -> SDF -> frozen-topology
hex mesh -> FEM solve -> objective -> `jax.grad`, validated against central finite
differences. This note records what runs today, the honest limits, and the follow-ups.

## What works today

| Piece | Status | Where |
|---|---|---|
| HEX8 volumetric mesher from any jaxcad SDF | working, no solver dependency | `jaxcad/fem/hexmesh.py` |
| Thermal (Poisson) solve on the hex mesh | working | `jaxcad/fem/simulate.py` |
| Linear elasticity + per-cell von Mises | working | `jaxcad/fem/simulate.py` |
| VTK export for ParaView | working (meshio) | `ThermalResult/ElasticResult.vtk_export` |
| Adjoint gradients w.r.t. node coordinates | working, FD-validated | `jaxcad/fem/backends.py` |
| End-to-end design gradient | working, FD-validated to ~1e-4 off kinks | `tests/fem/test_end_to_end.py` |
| Tesseract plugin ABI (local, no Docker) | working, gradients bit-identical to direct | `jaxcad/fem/tesseracts/thermal_jaxfem/` |
| Elastic solve packaged as a tesseract | working, gradients bit-identical to direct | `jaxcad/fem/tesseracts/elastic_jaxfem/` |
| Differentiable Dirichlet *values* (thermal, direct backend) | working via lifted solve, FD-validated | `JaxFemBackend.thermal` + `tests/fem/test_dirichlet_gradient.py` |

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

## Mesher notes (`jaxcad/fem/hexmesh.py`)

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

## BC selection design

Now: **predicates over boundary faces**. `sdf_to_hex_mesh` tags every boundary
quad with center + outward normal and groups by dominant SDF-gradient axis
(`"+x"`, `"-z"`, ...). `select_faces(mesh, predicate)` takes
`predicate(center)` or `predicate(center, normal)`; `thermal_solve`/`elastic_solve`
accept `(predicate, value)` pairs and resolve them to node index sets — which
cross any backend boundary as plain arrays. Inside jax-fem the sets become
2-arg location functions (`isin(index, set)`), avoiding coordinate matching.
Caveat: jax-fem applies a surface load to a face when *all* its vertices are in
the set, so a non-target face entirely inside a patch's vertex set would also be
selected (harmless for planar patches; keep patches face-aligned).

Later: **visual face picking in the viewer**. The boundary groups are exactly the
planar patches the viewer's hit-testing already understands; a picked patch
serializes to the same node-index-set ABI, so the solver layer needs no change.

## Plugin architecture (why Tesseract)

Direct in-process jax-fem is the **default backend and performance baseline**
(native JAX composition, no serialization). The Tesseract schema is the
**interop ABI**, not a mandatory wrapper:

- `jaxcad/fem/backends.py`: `SolverBackend` protocol —
  `thermal/elastic(points, cells, bcs, *, materials..., base_points)` returning
  JAX arrays with a VJP w.r.t. `points` (adjoint at minimum). `register_backend()`
  adds third-party solvers.
- `jaxcad/fem/tesseracts/thermal_jaxfem/tesseract_api.py`: the reference plugin.
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
- `jaxcad/fem/tesseracts/elastic_jaxfem/tesseract_api.py`: the elastic solve
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

`tests/fem/`: 44 tests, all passing with the extras installed; solver-dependent
files `pytest.importorskip` on `jax_fem` / `tesseract_core` / `tesseract_jax`, so
the suite skips (not fails) without them. `test_hexmesh.py` (11, mesher only),
`test_simulate.py` (10), `test_tesseract_backend.py` (5, incl. elastic parity +
bit-identical gradients + a generous overhead-timing bound),
`test_dirichlet_gradient.py` (3, lifted Dirichlet values vs FD),
`test_end_to_end.py` (2), plus the bracket demo and render-payload files.

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
source of truth, visual features are a layer on top**. `jaxcad/fem/study.py`
makes studies declarative and first-class:

- `ThermalStudy` / `ElasticStudy` are plain validated dataclasses declared
  directly in user programs and scripts. `.solve(sdf)` runs
  `sdf_to_hex_mesh` + the existing solvers and returns the usual result
  objects, so optimizers consume studies with no extra plumbing (a frozen
  mesh can be passed in for the frozen-topology gradient loop).
- `FaceSelector` makes boundary selection serializable: `side("+x")`
  (dominant-gradient-axis groups the mesher already computes),
  `box(center, extent)` (face-center containment), and `where(predicate)` as
  the code-only escape hatch. Side/box selectors round-trip through JSON;
  selectors resolve against `HexMesh.boundary_faces`/`select_faces` and are
  proven equivalent to hand-written predicates in tests.
- `.describe()` emits the JSON-ready payload the viewer needs (name, kind,
  resolution, domain, material, serialized BCs) — `json.dumps`-able,
  asserted in tests.
- `capture_studies()` mirrors `capture_constraint_solves` in
  `jaxcad/constraints/solve.py` (ContextVar + context manager): constructing
  a study inside the context registers it, so the compile worker can exec a
  user program and collect its declared studies in order. Verified against
  exec'd source, including nested-context isolation.
- `HeatFlux` is declarable and serializable now but raises
  `NotImplementedError` at solve time until the thermal backend grows a
  Neumann term — the schema leads, the solver follows.

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
