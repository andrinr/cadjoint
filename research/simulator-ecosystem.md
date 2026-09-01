# Simulator ecosystem survey: what plugs into the differentiable CAD chain next

Status: **survey, no code** (2026-08-16). Five domain sweeps (JAX-native CFD/misc,
mesh-based CFD adjoints, wave/EM/acoustics, multibody dynamics, legacy/Fortran FEA)
merged, deduplicated, and cross-ranked. Every maintenance/license claim below was
verified against the GitHub/PyPI APIs by the researchers in Aug 2026, and every
differentiability claim was checked against shipped code, not papers. Companion to
`research/fem-integration.md`, which defines the ABI all of this plugs into
(`cadjoint/fem/backends.py` `SolverBackend`, reference tesseracts under
`cadjoint/fem/tesseracts/`).

## How candidates were ranked

Composite of four axes, in this order of weight:

1. **Differentiability quality** — jax-native trace > discrete adjoint >
   continuous adjoint > FD-only. "Adjoint" had to be verified in shipped code;
   several famous codes failed this bar (see rejects).
2. **Demo value** — does the demo showcase *cadjoint's* geometry pipeline
   (constraints → SDF → mesh), and is it visually/narratively strong?
3. **Integration effort** — S < 1 week, M 1–3 weeks, L > 3 weeks, counting the
   geometry adapter, not just `pip install`.
4. **Usefulness for engineering CAD** — physics an engineer designing parts
   actually cares about.

### Packaging lanes (referenced throughout)

- **Lane A — direct fast lane**: pure-JAX solver imported in-process, composes
  with `jax.grad` natively, like `JaxFemBackend`. No serialization overhead.
- **Lane B — Tesseract subprocess/in-process API**: `tesseract_api.py` with
  typed schemas + hand-written `vector_jacobian_product`, loaded via
  `Tesseract.from_tesseract_api` (no Docker) — the proven ~0.14 s/roundtrip
  boundary. Right for JAX-version conflicts, non-JAX AD systems, and GPL
  isolation.
- **Lane C — containerized Tesseract**: `tesseract build` image around a heavy
  or unbuildable-locally stack (OpenFOAM, SU2-AD, PETSc, GPU-only). The
  distribution story; also the only sane route for some codes.

## Ranked table — all 31 candidates

| # | Candidate | Physics | Gradient | Geometry fit | Lane | Effort | Score |
|---|---|---|---|---|---|---|---|
| 1 | **CalculiX (ccx)** | structural FEA (Abaqus-like), thermal, frequency | discrete adjoint, native (`*SENSITIVITY`) | HEX8 `.inp` = cadjoint hex mesh 1:1 | B (subprocess) | M | 9.0 |
| 2 | **JAX-Fluids 2.0** | compressible NS, two-phase, levelset immersed solids | jax-native | SDF **is** the native levelset input | A | M | 9.0 |
| 3 | **jwave** | acoustics/ultrasound (time-domain + Helmholtz) | jax-native | SDF → material grid, ~10-line adapter | A (B if jax pins clash) | S | 8.5 |
| 4 | **MuJoCo MJX (JAX backend)** | rigid multibody + contact | jax-native ("mostly supported") | MJCF primitives + differentiable inertia from SDF | A | M | 8.0 |
| 5 | **Diffrax + in-repo mechanism layer** | ODE dynamics: linkages, lumped thermal, control | jax-native, reference adjoints | none needed — consumes cadjoint sketches directly | A | S | 8.0 |
| 6 | **XLB (Autodesk)** | lattice Boltzmann, incompressible-regime flows | jax-native (JAX backend only) | SDF → smoothed voxel mask | A | M | 7.5 |
| 7 | **fdtdx** | 3D Maxwell FDTD | jax-native, constant-memory backward | SDF → permittivity grid | A | S/M | 7.5 |
| 8 | **Firedrake + pyadjoint** | any UFL PDE: Helmholtz acoustics, elastodynamics | discrete adjoint w.r.t. **mesh coordinates** | consumes the HEX8 mesh natively — drop-in for the points-VJP contract | B/C | M | 7.5 |
| 9 | **MODFLOW 6 + MF6-ADJ** | groundwater flow (Darcy), USGS Fortran | discrete adjoint (non-intrusive, via libmf6) | SDF lattice **is** the model grid, zero meshing | B (pip, in-process) | M | 7.5 |
| 10 | **SU2** | compressible/incompressible RANS, external aero | discrete adjoint (CoDiPack, whole solver) | needs exterior **fluid** mesh; surface-node VJP | C | L | 7.0 |
| 11 | DAFoam | OpenFOAM RANS, internal flow, CHT | discrete adjoint (JFNK, <0.1% error) | fluid mesh via snappyHexMesh; surface-node VJP | C (official Docker) | L | 6.5 |
| 12 | Meep + meep.adjoint | broadband 3D EM FDTD | discrete adjoint (2 runs, density) | SDF → MaterialGrid density | C (conda/GPL-2) | M | 6.5 |
| 13 | PhiFlow | incompressible NS (no RANS) | jax-native | first-class SDF obstacles | A | S | 6.0 |
| 14 | Newton (NVIDIA Warp) | GPU rigid+soft body, XPBD contact | discrete adjoint (`wp.Tape`) | wp.Mesh from dual contouring; native SDF volumes | B/C (needs NVIDIA GPU) | L | 6.0 |
| 15 | EMopt | 2D FDFD / 3D CW-FDTD photonics | discrete adjoint w.r.t. **polygon vertices** | sketch `PolygonProfile` maps 1:1 | B/C (PETSc build) | M/L | 6.0 |
| 16 | ADflow (MDO Lab) | compressible RANS, Fortran + Tapenade | discrete adjoint (battle-tested) | structured multiblock CGNS — worst geometry fit | C | L | 6.0 |
| 17 | rfx | RF FDTD with ports / S-parameters | jax-native | CSG primitives or SDF raster | A | S/M | 5.5 |
| 18 | ceviche | 2D FDFD EM | discrete adjoint (HIPS autograd) | sketch cross-section → epsilon raster | A/B | S | 5.5 |
| 19 | JaxSim | reduced-coordinate robot dynamics | jax-native | URDF export; ground-contact only | A | M | 5.5 |
| 20 | OpenFOAM adjointOptimisationFoam | steady RANS + SA | continuous adjoint (mesh-consistent only asymptotically) | fluid mesh; per-node sensitivity maps | C | L | 5.5 |
| 21 | JAX-SPH | free-surface SPH (sloshing, dam break) | jax-native | SDF interior seeding + surface particles | A | M/L | 5.0 |
| 22 | Tidy3D adjoint | cloud GPU 3D EM FDTD | discrete adjoint (PolySlab vertices) | sketch-extrude → PolySlab, best CAD-EM match | paid SaaS | S/M | 5.0 |
| 23 | Elmer FEM | multiphysics (EM, thermal, CFD) | adjoint only in ElmerIce (glaciology params); FD otherwise | HEX8 via ElmerGrid | B/C | M/L | 5.0 |
| 24 | Nimble Physics | LCP hard-contact multibody | discrete adjoint (analytic, RSS'21) | primitive proxies + inertias | C (torch, Linux) | L | 4.5 |
| 25 | OpenSees / OpenSeesPy | nonlinear structural frames | forward DDM (partial coverage, sizing not shape) | script-built frames, not CAD solids | A/B | S | 4.0 |
| 26 | FEAP / FEAPpv | classic solid mechanics FEA (1976 lineage) | none — FD/SPSA only | trivial brick decks | B | M | 4.0 |
| 27 | FENIAX | nonlinear aeroelasticity | jax-native | needs Nastran condensed models, not cadjoint meshes | B (GPL-3) | L | 3.5 |
| 28 | Drake | industrial multibody | forward AutoDiffXd; **throws under contact** | URDF/SDF-format | A/B | M | 3.5 |
| 29 | Code_Aster | industrial nonlinear FEA (EDF) | none — sensitivity feature **removed** | MED hexes fine | C only | L | 3.0 |
| 30 | Dojo.jl | hard-contact multibody, IFT gradients | discrete adjoint (real, in code) | primitives only; **stale since 2024-09** | C (Julia) | L | 3.0 |
| 31 | Nek5000 / NekRS | spectral-element NS | none — "adjoint" is stability analysis, not design | curvilinear hex `.re2` (ironically matching) | C | L | 2.5 |

## Cross-cutting findings

### The SDF is the unlock (and the pitch line)

Every grid-based solver (JAX-Fluids, XLB, jwave, fdtdx, PhiFlow, Meep, ceviche,
rfx, MODFLOW) takes geometry as a field on a regular grid. Shape gradients
through a **binary** voxel mask are identically zero — the universal recipe is a
*smoothed SDF indicator* (Brinkman penalization for LBM, smoothed-heaviside
sound speed for jwave, permittivity for EM, native levelset for JAX-Fluids,
sigmoid conductivity for MODFLOW). cadjoint's traced SDF produces exactly that
smooth field for free; an STL-based CAD pipeline cannot. This is the pitch line
for all the Lane-A demos.

### Two VJP boundary shapes, both already proven in-repo

1. **Surface-node coordinates** (the frozen-topology doctrine): SU2
   (`GetMeshDisp_Sensitivity`), DAFoam (`dF/dX`), ADflow (`evalFunctionsSens`),
   adjointOptimisationFoam (sensitivity maps), Firedrake (mesh-coordinate
   Control), and CalculiX (per-design-node normal sensitivities) all natively
   emit d(objective)/d(surface nodes). cadjoint's dual-contour vertices and
   Newton-snapped hex boundary nodes are already differentiable w.r.t. SDF
   parameters, so the chain closes exactly as it does for jax-fem.
2. **Material/density fields**: the SDF-sampled grid crosses the boundary as a
   differentiable array; the solver returns dJ/d(field). Easier, but the design
   stays a density soup inside the solver; shape-native paths (1) keep the
   design manufacturable CAD at every step and demo the system's identity
   better.

### The fluid-domain gap

cadjoint meshes the **solid**; every mesh-based CFD code needs the complementary
**fluid** domain. The clean pattern mirroring the FEM doctrine: gmsh (already a
dependency) meshes the flow domain once around the dual-contoured surface at the
nominal design, topology frozen, the solver's own mesh deformation propagates
surface motion inward, and d(objective)/d(surface nodes) is the VJP boundary.
This is why SU2/DAFoam/ADflow are all L-effort despite turnkey adjoints — and
why the levelset/voxel candidates that skip fluid meshing entirely rank higher.

### Contact gradients need the FD-validation discipline

Every multibody candidate has verified gradient fragility at contact: MJX NaN
gradients in some configs (mujoco#2237), CG-solver reverse-mode issues
(mujoco#1182, prefer `solver=newton` with fixed iterations), literature showing
different contact formulations give different — sometimes wrong — gradients at
impact. The `tests/fem` adjoint-vs-central-FD discipline must carry over: FD-
validate every rollout gradient at the nominal design before trusting descent.

### Tesseract-core gradient facts (verified in installed 1.11.0 source)

- Gradient endpoints are **not** autogenerated from `apply`; only endpoints the
  author defines in `tesseract_api.py` register.
- `runtime/experimental/finite_differences.py` ships
  `finite_difference_{jacobian,jvp,vjp}` helpers ("central", "forward",
  "stochastic" = SPSA at O(√n) samples) designed to be a one-line endpoint body
  — the honest fallback for adjoint-less legacy codes (FEAP, Code_Aster,
  general Elmer), first-class but opt-in and marked experimental.
- `tesseract-runtime check-gradients` validates any gradient endpoint against
  FD at sampled indices — belongs in CI for every wrapped solver.
- `tesseract init --recipe jax` autogenerates all endpoints, but only for
  JAX-traceable `apply` functions — irrelevant for legacy codes.
- Pasteur's tesseract-core docs ship a JAX-CFD Navier-Stokes tesseract example
  worth cribbing for schema design.

### License containment

GPL codes (CalculiX GPL-2, DAFoam/OpenFOAM/Code_Aster GPL-3, Meep GPL-2) are
contained cleanly at the Tesseract process/container boundary — input decks in,
result files out, no linking. LGPL (jwave, Firedrake, SU2) is fine even as a
pip dependency. JAX-Fluids is MIT (verified in LICENSE — older sources wrongly
say GPLv3; it was relicensed). MODFLOW 6 + mf6adj (public domain + CC0) is the
most permissive stack surveyed.

---

## Top candidates in detail

### 1. CalculiX — the Fortran code with a real adjoint (score 9.0)

**What it is.** Abaqus-like structural FEA in Fortran 77/90 + C (Guido Dhondt,
MTU Aero Engines lineage, ~1998; v2.23 released late 2025, active forum,
conda-forge binaries). GPL-2.

**Gradient mechanism.** Native, built-in, and the reason it tops this list: a
`*SENSITIVITY` procedure step after `*STATIC` (or `*FREQUENCY, STORAGE=YES`)
computes adjoint-based sensitivities of built-in objectives (`*OBJECTIVE`:
MASS, STRAIN ENERGY/compliance, EIGENFREQUENCY, ALL-DISP) w.r.t.
`*DESIGN VARIABLES, TYPE=COORDINATE` — a surface node set. One adjoint-cost
pass yields the gradient w.r.t. all design nodes, written to `.frd`,
**projected on the local surface normal** (in-surface motion doesn't change
geometry). That projection composes *exactly* with cadjoint's mesher: the
Newton-snapped boundary vertices move along the SDF gradient (the surface
normal) by construction, so the normal-projected sensitivity is precisely the
derivative the chain rule needs. Restrictions: true 3D elements only (C3D8
fine); objectives limited to the built-in list, so the backend contract is
**objective-valued**, not field-valued — which is exactly what the bracket
demo's compliance + mass objective wants.

**Geometry coupling.** Byte-for-byte: `sdf_to_hex_mesh` HEX8 points/cells →
meshio (already a `cadjoint[fem]` dependency) writes Abaqus `.inp` C3D8 decks
directly. `Nodes` selections serialize to `*NSET` + `*BOUNDARY`/`*CLOAD`;
design-variable node set = the mesher's `snap_mask` surface nodes.

**Packaging plan.** Lane B: conda-forge `calculix` binary + a subprocess
tesseract. Schema mirrors `elastic_jaxfem` (points Differentiable, cells,
fixed/traction node sets) + an objective spec; `apply` = write `.inp`, run
`ccx`, parse `.frd`/`.dat`; `vector_jacobian_product` = the `*SENSITIVITY`
run, then map per-design-node normal sensitivities to d(objective)/d(points)
via the stored node normals, times the cotangent. No f2py, no Docker locally;
`tesseract build` container for distribution. GPL-2 stays behind the process
boundary.

**Demo storyline.** Rerun the existing L-bracket study (`scenes/bracket.py`,
`examples/fem_bracket_optimization.py`) with a 1990s Fortran Abaqus-clone as
the solver — same mesh, same BCs, same objective — and put **three independent
gradient paths on one screen**: ccx adjoint vs the in-repo jax-fem adjoint vs
`tesseract-runtime check-gradients` FD. "A battle-tested Fortran turbine-blade
FEA code is now one `jax.grad` call away from parametric CAD."

**Effort.** M — deck writer + `.frd` parser + normal-projection chain rule are
straightforward; robust output parsing and two-step orchestration need care.
Every cadjoint-side piece already exists.

Sources: [ccx *SENSITIVITY docs](https://www.feacluster.com/CalculiX/ccx_2.18/doc/ccx/node187.html),
[design variables](https://www.feacluster.com/CalculiX/ccx_2.18/doc/ccx/node23.html),
[ccx 2.23 manual](https://www.dhondt.de/ccx_2.23.pdf),
[dhondt.de](http://www.dhondt.de/new_calc.htm),
[conda-forge package](https://anaconda.org/conda-forge/calculix),
[ccx-shape](https://github.com/fandaL/ccx-shape).

### 2. JAX-Fluids 2.0 — the jax-native CFD flagship (score 9.0)

**What it is.** Compressible Navier-Stokes (single/two-phase), high-order
finite volume (WENO/TENO), **sharp-interface levelset immersed solid
boundaries**, 1D/2D/3D Cartesian, scales to 512 A100s. Pure JAX, MIT
(verified — relicensed from GPLv3), TU Munich, pushed 2026-08, CPC 2025 paper.

**Gradient mechanism.** The entire solver is `jax.numpy`, so `jax.grad`/`vjp`
trace through the full time loop (checkpointing for memory); end-to-end
gradient optimization validated in both the 2022 and 2.0 papers, including AD
through distributed simulations. Verified caveat: the levelset module exists,
but **no shape-optimization example ships in `examples/`** — the
differentiable driver around a solid-boundary case is ours to write, and
levelset reinitialization steps may need care in the backward pass.

**Geometry coupling.** The strongest of any candidate: cadjoint's SDF evaluated
on the sim grid **is** the native levelset input, tracing intact. No meshing,
no conversion, no smoothing layer.

**Packaging plan.** Lane A: pip install, in-process, composes with `jax.grad`
directly. Tesseract only for version isolation.

**Demo storyline.** Supersonic/transonic nose-cone or fairing drag
minimization: parametric revolved cadjoint profile (radius, length, bluntness as
constrained Scalars) → SDF → levelset grid → 2D-then-3D Mach 2 solve →
pressure drag integrated over the immersed boundary → `jax.grad` back to CAD
parameters, optax loop, schlieren-style density renders per optimizer step.

**Effort.** M — geometry coupling is trivial; the work is the drag functional
on the levelset boundary, the differentiable driver, and
backprop-through-time memory (checkpointing).

Sources: [repo](https://github.com/tumaer/JAXFLUIDS),
[JAX-Fluids 2.0, CPC 2025](https://www.sciencedirect.com/science/article/pii/S0010465524003564),
[docs](https://jax-fluids.readthedocs.io/en/latest/).

### 3. jwave — the smallest diff, a new physics domain (score 8.5)

**What it is.** Acoustics/ultrasound in pure JAX (on jaxdf): time-domain
k-space/FD wave propagation and time-harmonic Helmholtz in heterogeneous
media, PML boundaries. LGPL-3, UCL Biomedical Ultrasound Group, SoftwareX
2023, pushed 2026-03.

**Gradient mechanism.** Proven-in-docs, not paper-ware: the FWI tutorial
computes `jax.grad` of the simulation w.r.t. the sound-speed map with
checkpointed backprop and runs Adam on it; the Helmholtz path documents
"optimizing through GMRES" (implicit differentiation of the linear solve).

**Geometry coupling.** SDF → smoothed-heaviside sound-speed/density map on
`Domain(N, dx)` — a ~10-line adapter, no meshing.

**Packaging plan.** Lane A if its JAX pin coexists with the repo's jax 0.8.x;
if not, Lane B with its own venv — which is precisely the isolation the
Tesseract ABI was built for, and a good second proof of that story.

**Demo storyline.** Ultrasound focusing lens: a constraint-driven cadjoint lens
profile (arc radius, thickness, aperture) revolved into an SDF in front of a
flat transducer; rasterize to a sound-speed map; maximize focal pressure at a
target point via time-harmonic Helmholtz (one linear solve per iteration —
fast and robust). Variant: Helmholtz-resonator/speaker-port tuning of an
enclosure to a target resonance.

**Effort.** S — the smallest integration on this list. S/M if version
isolation is needed.

Sources: [repo](https://github.com/ucl-bug/jwave),
[FWI tutorial](https://ucl-bug.github.io/jwave/notebooks/time_varying/FWI.html),
[SoftwareX paper](https://www.sciencedirect.com/science/article/pii/S2352711023000341),
[jaxdf](https://github.com/ucl-bug/jaxdf).

### 4. MuJoCo MJX — contact-rich mechanism design (score 8.0)

**What it is.** DeepMind's rigid multibody dynamics with frictional contact,
reimplemented in JAX (`mjx.step`). Apache-2.0, pip `mujoco-mjx`, weekly
commits. Strategic fact: **MJX-JAX is the differentiable backend**; the newer
MJX-Warp/MuJoCo-Warp backends are fast but explicitly non-differentiable ("no
immediate plans").

**Gradient mechanism.** `mjx.Model` is a pytree of arrays — `geom_size`,
`geom_pos`, `body_mass`, `body_inertia`, actuator params are valid
differentiation targets through an unrolled `lax.scan` rollout. Verified
caveats: docs say "mostly supported"; prefer `solver=newton` with fixed
iterations (CG reverse-mode issues, mujoco#1182); open NaN-gradient issue in
some contact configs (mujoco#2237); stiff-contact gradients can flip sign at
coarse timesteps. Keep contacts soft-ish, FD-validate per scene. SDF geoms
are **not** supported in MJX-JAX; meshes limited to small convex
decompositions.

**Geometry coupling.** Two channels: (a) capsule/box collision proxies whose
sizes are traced functions of sketch parameters; (b) the robust one — cadjoint's
SDF gives **exact differentiable mass/inertia/CoM by voxel integration** (pure
JAX) feeding `body_mass`/`body_inertia`/`body_ipos`, solid even where contact
gradients are fragile.

**Packaging plan.** Lane A (pip, pure JAX, runs on CPU macOS). Tesseract only
if jax pins conflict.

**Demo storyline.** Parametric two-jaw gripper designed in cadjoint (jaw
profile, pivot positions, link lengths as constrained sketch parameters);
MJX rolls out a close-and-shake grasp; objective = object retention + minimum
actuation force; `jax.grad` flows from the rollout through
`geom_size`/inertias back to sketch parameters, optimized with optax.

**Effort.** M — MJCF generation from cadjoint assemblies plus contact-gradient
hygiene.

Sources: [MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html),
[mujoco#2237](https://github.com/google-deepmind/mujoco/issues/2237),
[mujoco#1182](https://github.com/google-deepmind/mujoco/issues/1182),
[DiffMJX](https://arxiv.org/html/2506.14186).

### 5. Diffrax + an in-repo mechanism layer — the cheapest full-chain demo (score 8.0)

**What it is.** Not a domain solver: the reference implementation of ODE/SDE
adjoints in JAX (Kidger ecosystem, Apache-2.0, the best-maintained project
surveyed). Pairs with cadjoint's **own constraint solver** to make a mechanism
simulation layer with zero third-party solver risk.

**Gradient mechanism.** Two exact paths, both already idiomatic in this
codebase: (1) kinematics = cadjoint constraint solve per crank angle,
differentiated via the implicit function theorem (`jax.custom_vjp` around the
solve — the same pattern as jax-fem's `ad_wrapper`); (2) dynamics via
`diffrax.diffeqsolve` with `RecursiveCheckpointAdjoint` (discretise-then-
optimise, exact discrete gradients; the docs explicitly recommend it over the
approximate continuous `BacksolveAdjoint`).

**Geometry coupling.** None needed — consumes cadjoint sketches directly; link
inertias, if dynamic, come from differentiable SDF volume integrals.

**Packaging plan.** Lane A, `pip install diffrax`, zero packaging risk.

**Demo storyline.** The classic that sells differentiable CAD: a four-bar (or
Jansen walking) linkage sketched in the playground with pin constraints; the
user draws a target curve; the optimizer adjusts link-length Scalars so the
coupler point traces it, **mechanism animating live in the viewer as gradients
update the sketch**. The constraint solver is the star. Extends to
minimum-motor-torque-over-cycle via diffrax dynamics.

**Effort.** S — constraint solver, viewer, and parameter plumbing all exist;
new code is a time-sweep loop + IFT wrapper + demo scene.

Sources: [repo](https://github.com/patrick-kidger/diffrax),
[adjoint docs](https://docs.kidger.site/diffrax/api/adjoints/).

### 6. XLB — a second physics for the bracket (score 7.5)

**What it is.** Autodesk Research lattice Boltzmann (weakly compressible NS),
2D/3D, LES, immersed boundaries, giga-lattice-updates/sec. Apache-2.0, pip
`xlb`, CPC 2024 paper. Caution: recent momentum (multi-GPU, grid refinement)
is on the **non-differentiable** Warp/Neon backends; the JAX backend is the
differentiable one.

**Gradient mechanism.** `jax.grad` through unrolled LBM stepping on the JAX
backend — real examples ship in-repo (`examples/cfd/differentiable_lbm.py`,
`examples/out_of_core/autodiff_lbm.py`). Verified skeptically: "adjoint-based
shape and topology optimization" is **roadmap, not shipped**. The critical
integration fact: shape gradients through a binary voxel mask are identically
zero — the shape-gradient path requires a smoothed indicator
(Brinkman/porosity penalization or partially-saturated bounce-back) built from
the SDF. That smoothing layer is our contribution, and cadjoint provides the
smooth field for free.

**Geometry coupling.** SDF < 0 voxelization plugs into its boundary-mask
machinery directly (it ships a trimesh STL voxelizer we can skip); the
smoothed-SDF variant makes it differentiable w.r.t. CAD parameters.

**Packaging plan.** Lane A, pip with extras.

**Demo storyline.** Differentiable wind tunnel: drop the **existing bracket
part** into a 3D LBM channel, soft-mask it from the SDF, gradient-descend the
sketch parameters to cut drag while a constraint holds frontal area — the same
part that already runs the jax-fem elastic study gets a second physics. The
killer multiphysics story. (Internal variant: manifold/heat-sink channel
pressure drop at Re ~ 1e3.)

**Effort.** M — solver is pip-and-go; the work is doing the smoothed
SDF-to-mask coupling right (non-degenerate gradients, FD-validated) plus
rollout memory strategy.

Sources: [repo](https://github.com/Autodesk/XLB),
[CPC paper](https://www.sciencedirect.com/science/article/abs/pii/S0010465524001103),
[releases](https://github.com/Autodesk/XLB/releases).

### 7. fdtdx — electromagnetics nobody else offers (score 7.5)

**What it is.** Full 3D Maxwell FDTD (Yee grid) in pure JAX,
photonics-oriented but scale-invariant. MIT, very active (pushed 2026-08,
JOSS paper).

**Gradient mechanism.** Reverse-mode AD with the package's raison d'être: it
exploits **time-reversibility of Maxwell's equations** to re-integrate fields
backward instead of storing the time history — constant-memory gradients for
millions of voxel parameters. Inverse-design examples ship in the repo.

**Geometry coupling.** SDF → smoothed permittivity in a voxel design region —
the same adapter pattern as jwave/XLB. Nanophotonics-tuned defaults mean
unit/scale mapping needs attention for macroscopic parts.

**Packaging plan.** Lane A, pip.

**Demo storyline.** EM-transparent CAD: optimize a sensor radome wall profile
or a dielectric waveguide splitter/collimating lens so transmission at the
target port/frequency is maximized — CAD parameters → SDF → permittivity →
gradient. Distinctive because no other candidate gives cadjoint EM.

**Effort.** S/M — AD machinery is turnkey; its object/config system (sources,
detectors, PML) and scale mapping take a few days.

Sources: [repo](https://github.com/ymahlau/fdtdx),
[memory-efficient AD paper](https://arxiv.org/abs/2412.12360),
[docs](https://fdtdx.readthedocs.io/en/latest/).

### 8. Firedrake + pyadjoint — the hex mesh's natural FEM home (score 7.5)

**What it is.** UFL-based FEM over PETSc with taped discrete adjoints
(pyadjoint); Helmholtz acoustics, elastodynamics, any UFL-expressible PDE on
unstructured meshes **including hexahedra**. LGPL-3, very active, Fireshape
adds a dedicated shape-optimization layer.

**Gradient mechanism.** pyadjoint tapes the forward solve and solves the
discrete adjoint; crucially it shape-differentiates w.r.t. **mesh
coordinates** via UFL's shape-derivative support (the official
shape_optimization demo uses `mesh.coordinates` as the Control). That is
bit-for-bit cadjoint's backend contract: VJP w.r.t. points on a
fixed-connectivity mesh.

**Geometry coupling.** The only candidate besides CalculiX that consumes
`sdf_to_hex_mesh` output natively: write points+cells to `.msh` (meshio), and
the coordinate cotangent comes straight back through the tesseract VJP into
`recompute_points`.

**Packaging plan.** Lane B/C: pip has binary PETSc wheels since 2025 but the
stack is heavy — cleanest as a subprocess/container tesseract with its own
env (schema: points, cells, BC node sets, frequency → pressure field +
dJ/d(points)), mirroring `thermal_jaxfem` exactly.

**Demo storyline.** Speaker horn vibroacoustics: cadjoint horn flare (throat
radius, flare exponent, length as constrained parameters) → HEX8 mesh of the
air volume → Helmholtz at 1–5 kHz with a piston BC → maximize on-axis SPL or
flatten frequency response; the adjoint returns dJ/d(node coords), chained
through the frozen-topology snap. A genuinely engineering-CAD demo no EM
candidate can tell.

**Effort.** M — env isolation, `.msh` round-trip, coordinate-cotangent
extraction from pyadjoint.

Sources: [shape optimization demo](https://www.firedrakeproject.org/firedrake/demos/shape_optimization.py.html),
[adjoint docs](https://www.firedrakeproject.org/adjoint.html),
[Fireshape](https://fireshape.readthedocs.io/en/latest/).

### 9. MODFLOW 6 + MF6-ADJ — the other legacy-Fortran story (score 7.5)

**What it is.** USGS groundwater flow in Fortran 2008, direct lineage of 1984
MODFLOW; MF6-ADJ (INTERA, peer-reviewed in *Groundwater* 2025) is a
**non-intrusive discrete adjoint** that drives MODFLOW 6 through its official
API library (`libmf6` via modflowapi/xmipy), harvests matrices at runtime, and
solves the adjoint backward — zero core-code changes. Public domain + CC0, the
most permissive stack surveyed; both actively maintained; mf6adj 1.2.0 on
PyPI.

**Gradient mechanism.** One backward run yields sensitivities of performance
measures (heads, boundary fluxes, composites) w.r.t. per-cell hydraulic
conductivity, storage, recharge, well rates — validated against analytics and
FD at 100–10000x speedup. Geometry enters as a **material field**: SDF at cell
centers → smooth sigmoid → per-cell K (pure JAX) → adjoint dJ/dK crosses the
tesseract as the VJP. Caveat: a density coupling, not a boundary-fitted shape
derivative; gradient quality depends on sigmoid bandwidth vs cell size.

**Geometry coupling.** Best format-fit of any candidate: the SDF lattice
**is** the DIS model grid; the hex mesher isn't even needed.

**Packaging plan.** Easiest surveyed: pip `mf6adj` + `modflowapi`, binaries
via flopy's get-modflow. Pure-Python in-process tesseract (Lane B); no
compiler, no container required. The non-intrusive-adjoint-via-API-library
pattern is worth remembering for other BMI/XMI codes.

**Demo storyline.** Slurry cutoff wall / seepage barrier as a cadjoint
parametric solid (arc length, depth, thickness) → per-cell low-K field in a
site-scale model with a pumped excavation → minimize inflow + bentonite-volume
penalty → mf6adj adjoint bends and deepens the wall. "The USGS Fortran code
that models half the world's aquifers, inside a differentiable CAD loop — and
the geometry needs no mesh at all."

**Effort.** M — install is S-like, but the MODFLOW input stack (flopy,
packages, units) and mapping mf6adj's HDF5 output into the VJP are real work.

Sources: [mf6adj](https://github.com/INTERA-Inc/mf6adj),
[Groundwater 2025 paper](https://ngwa.onlinelibrary.wiley.com/doi/10.1111/gwat.70025),
[INTERA announcement](https://www.intera.com/modflow-6-and-adjoint-sensitivity/).

### 10. SU2 — the credibility-grade RANS adjoint (score 7.0)

**What it is.** Industry-credible compressible/incompressible RANS (SA, SST),
external aero flagship, C++ with SWIG Python wrappers. LGPL-2.1, very active
(v8.5).

**Gradient mechanism.** The strongest adjoint mechanism surveyed: reverse-mode
AD of the **entire solver** via CoDiPack operator overloading, formulated as a
fixed-point adjoint — numerically exact discrete gradients including the
turbulence model. Verified programmatic path:
`pysu2ad.CDiscAdjSinglezoneDriver` (`MATH_PROBLEM=DISCRETE_ADJOINT`,
`OBJECTIVE_FUNCTION=DRAG`), then `GetMeshDisp_Sensitivity()` returns
dJ/d(mesh coordinates) — literally the VJP payload the ABI wants.

**Geometry coupling.** The impedance mismatch: SU2 needs the **fluid** domain.
gmsh meshes the exterior once around the dual-contoured surface, topology
frozen; SU2_DEF propagates surface motion inward; surface node coordinates are
the differentiable boundary.

**Packaging plan.** Lane C: no PyPI wheels for the adjoint module (pysu2 404s
on PyPI, verified) — source build via meson (`-Denable-autodiff=true
-Denable-pywrapper=true`) or a containerized tesseract built once.

**Demo storyline.** Transonic drag minimization of a cadjoint-parameterized 2D
profile (PolygonProfile with constraint-driven thickness/camber): apply =
deform mesh + primal solve returning C_D; VJP = pysu2ad adjoint returning
dC_D/d(surface nodes), chained through the differentiable dual-contour
vertices. The classic NACA0012 case with CAD as the design space.

**Effort.** L — AD build, fluid-domain meshing pipeline, and the surface-node
chain rule are each real work; highest-credibility aero gradient as payoff.
Second-act material, after the levelset/voxel CFD demos land.

Sources: [repo](https://github.com/su2code/SU2),
[python wrapper adjoint tutorial](https://su2code.github.io/tutorials/Adjoint_FSI_Python/),
[AD build docs](https://su2code.github.io/docs/Python-Wrapper-Build/).

---

## Notable mid-table candidates (short form)

- **DAFoam (6.5)**: turnkey OpenFOAM discrete adjoint (JFNK, <0.1% derivative
  error), official Docker images — the cleanest containerized-tesseract
  candidate; U-bend cooling-duct pressure-drop demo mirrors its own flagship
  tutorial. L effort (OpenFOAM stack + snappyHexMesh pipeline); GPL-3
  contained at the container boundary. Largely interchangeable with SU2 —
  pick SU2 for external transonic aero, DAFoam for internal flow.
- **Meep (6.5)**: built-in adjoint (exactly two timestepping runs), verified
  in-tree tests; conda/GPL-2 forces Lane C — which makes it the strongest
  "legacy C++ behind the tesseract ABI" proof before attempting Fortran.
  Radiation-pattern shaping of a dielectric lens antenna via the near2far
  adjoint.
- **PhiFlow (6.0)**: first-class SDF obstacles, `jax.grad` through the
  pressure solve, S effort — the best effort-to-demo ratio for a weekend 2D
  duct/wake demo, but no RANS, so position it as the toy tier below
  JAX-Fluids/XLB.
- **Newton / NVIDIA Warp (6.0)**: `wp.Tape` kernel-level discrete adjoints,
  the only candidate that natively samples volumetric SDFs (NanoVDB) with
  gradients — the natural long-term home for SDF-native contact — but
  differentiable solvers effectively need an NVIDIA GPU, forcing the
  container/remote leg of the Tesseract story. L effort; flagship potential
  later.
- **EMopt (6.0)**: the only candidate whose native gradient is a
  **boundary-shape derivative w.r.t. polygon vertices** — conceptually the
  cleanest CAD coupling in EM (sketch `PolygonProfile` maps 1:1, designs stay
  manufacturable). Single-author, PETSc source build; M/L.
- **ceviche (5.5)**: dormant since 2023 but MIT, tiny, vendorable; the
  fastest possible EM proof (days) and a clean proof that a **non-JAX AD
  system** (HIPS autograd) plugs into the ABI via a VJP endpoint.
- **rfx (5.5)**: the only ports/S-parameters candidate — the patch-antenna
  S11 demo engineers instantly recognize — but a 6-star single-maintainer
  project; FD-validate its gradients before building anything on it.

## Rejected / parked (with reasons)

| Candidate | Reason |
|---|---|
| Nek5000/NekRS | "Adjoint" is linearized-operator stability analysis (nekStab), not design gradients; FD-only for shape. Forward-validation role only. |
| Code_Aster | Historical SENSIBILITE feature **removed**; FD-only; notoriously heavy build. Dominated by CalculiX on every axis that matters here. |
| FEAP/FEAPpv | The real ~50-year-old Fortran artifact, but a gradient vacuum (no adjoint ever); only honest as a demo of tesseract's experimental FD/SPSA endpoints. Full FEAP source is fee/by-arrangement — a distribution problem. |
| Elmer FEM | Real adjoint code exists but only in ElmerIce, inverting glaciology parameter fields; not transferable to CAD shape objectives without writing Fortran adjoints. |
| Drake | SAP contact solver **throws** under AutoDiffXd with any contact present; forward-mode only, ~100x slowdown. |
| Dojo.jl | The best contact-gradient formulation on paper (IFT through interior-point), but stale since 2024-09 — research spike, not a shipping integration. |
| FENIAX | jax-native but consumes Nastran condensed models; cadjoint reduced to a parameter dashboard. GPL-3. |
| Brax | **Deprecated as a physics engine** — brax 0.13+ maintains only `brax/training`; repo directs physics users to MJX. Do not build on it. |
| Genesis | Rigid-body differentiability still "coming soon" (only MPM/Tool solvers differentiable as of 2026). |
| google/jax-cfd | Alive but no immersed-geometry story in core (periodic/rectangular domains); strictly dominated by XLB/JAX-Fluids for CAD shape work. (Its tesseract example in Pasteur's docs is worth cribbing for schema design.) |
| tofea | GitHub-archived 2025, autograd 2D toy. |
| jax-am | GPL-3, stale; its FEM core was spun out into jax-fem, which cadjoint already integrates. |
| sandialabs/optimism | Non-standard license, largely 2D, duplicates jax-fem coverage. |
| Stride/Devito | Real adjoints but AGPL-3, medium-property gradients only; jwave dominates the physics JAX-natively under LGPL. |
| SPECFEM3D | Legacy Fortran, natively eats HEX meshes, has adjoint kernels — but only w.r.t. material fields (rho, vp, vs), no shape derivative, and geophysics scaling makes a part-level demo contrived. |
| femwell | No adjoint anywhere in the code (scipy/SLEPc eigensolves); FD-only. |
| spins-b/angler | Unmaintained since ~2020. |
| fmmax/meent (RCWA) | Differentiable but restricted to periodic layered stacks, not free-form CAD shapes. |
| Tidy3D | Technically the most CAD-aligned 3D EM path (PolySlab vertices from sketches), but every gradient step is a paid cloud job with credentials — not a self-contained demo. Parked, not rejected. |
| Nimble | Analytic LCP contact gradients are real, but single-maintainer biomechanics pivot, wheel/platform friction, torch boundary. Parked. |
| OpenSees | DDM is forward-mode sizing sensitivities with patchy element coverage — demos parametric sizing, not CAD shape. Off-thesis. |
| JAX-SPH | Genuinely differentiable and the only free-surface option, but mildly stale and M/L bespoke particle-seeding glue. Watch list. |
| adjointOptimisationFoam | Subsumed by DAFoam (same ecosystem, discrete adjoint, Python API vs case-dict driving). |
| ADflow | Real battle-tested Fortran adjoint, but structured multiblock CGNS + pyHyp extrusion is the worst geometry fit surveyed; CalculiX carries the Fortran story with none of that pain. |
| DiFVM/JAX-FVM | 2026 arXiv differentiable unstructured FVM — immature/unreleased. Watch list. |

## Recommended next integrations

Ordering logic: (1) is the explicitly wanted Fortran/legacy story and lands the
strongest validation demo on geometry we already have; (2) is the smallest diff
and opens a new physics domain; (3) is the visual flagship. The diffrax linkage
demo is the cheap fourth whenever a weekend slot opens — it needs no external
solver at all.

### 1. CalculiX tesseract (Fortran/legacy, M)

The only living Fortran code with a native adjoint whose input matches
`sdf_to_hex_mesh` byte-for-byte. Concrete first steps:

1. `conda install -c conda-forge calculix`; hand-write a minimal C3D8 `.inp`
   for the existing thermal bar mesh via meshio; run `ccx` and parse `.dat` to
   confirm displacement parity with jax-fem on the identical mesh.
2. Add a `*SENSITIVITY` step (`*OBJECTIVE STRAIN ENERGY`,
   `*DESIGN VARIABLES, TYPE=COORDINATE` on the `snap_mask` node set); parse
   the `.frd` normal sensitivities and check dJ/d(normal offset) against
   central FD on a single node.
3. Write `cadjoint/fem/tesseracts/elastic_calculix/tesseract_api.py` cloning the
   `elastic_jaxfem` schema; `vector_jacobian_product` = sensitivity run +
   normal-projection chain rule (stored node normals × cotangent).
4. Wire `tesseract-runtime check-gradients` into CI; then rerun
   `examples/fem_bracket_optimization.py` with `backend="calculix"` and put
   the three gradient paths (ccx adjoint / jax-fem adjoint / FD) on one plot.

### 2. jwave ultrasound lens (S)

Smallest integration, new physics, proven-in-docs gradients. Concrete first
steps:

1. `pip install jwave` in a scratch venv; confirm whether its JAX pin coexists
   with jax 0.8.2 — that decides Lane A vs a Lane-B venv tesseract.
2. Write the ~10-line SDF → smoothed-heaviside sound-speed adapter
   (`Domain(N, dx)` + `Medium`); FD-validate d(pressure at focus)/d(one CAD
   parameter) through the time-harmonic Helmholtz solve.
3. Build the demo scene: revolved constraint-driven lens profile in front of a
   piston source, optax on (arc radius, thickness, aperture), focal-pressure
   objective; render the pressure field per step.

### 3. JAX-Fluids fairing (M, the flagship)

The SDF is the native levelset input — the strongest geometric fit surveyed.
Concrete first steps:

1. Run a shipped forward levelset example (2D immersed solid); swap the
   levelset for a cadjoint SDF evaluated on the sim grid and confirm the forward
   solve is unchanged.
2. Write the differentiable driver (their feed-forward-style API) around that
   case — this does not ship in `examples/` and is the main technical risk;
   watch levelset reinitialization in the backward pass, use checkpointing.
3. Implement the pressure-drag functional on the immersed boundary;
   FD-validate d(drag)/d(one profile parameter) on a coarse 2D grid before
   scaling up; then the Mach-2 nose-cone optimization with per-step density
   renders.

If a second legacy slot opens after CalculiX: **MODFLOW 6 + MF6-ADJ** is the
lowest-friction packaging surveyed (pip + CC0 + zero meshing) and the better
"1984 lineage" story, at the price of a density rather than shape coupling.
