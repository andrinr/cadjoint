# Tesseract Hackathon 2026 — cadjoint

**Track 01 — Inverse design & shape optimization** · solo entry by Andrin Rehmann

**Repo:** https://github.com/andrinr/cadjoint · **License:** Apache 2.0

## One sentence

cadjoint is a code-first, differentiable CAD system whose whole chain —
sketch constraints → signed-distance geometry → meshing → finite-element
solve → engineering objective — is one function `jax.grad` differentiates,
with every non-JAX stage crossing its boundary as a Tesseract.

## The composition (criterion 1)

Four differentiation strategies, three languages, one differentiable function:

| Stage | Component | Language | Gradient mechanism |
| --- | --- | --- | --- |
| CAD front-end | constraints → SDF | Python/JAX | autodiff |
| Meshing kernels | `native/tesseract_api.py` | **Rust** (rayon) | hand-derived linear-solve VJP (~4e-14 vs autodiff) |
| Meshing, whole-pipeline | `cadjoint/fem/tesseracts/mesher` *(experimental)* | mesher-agnostic | **surface-interpolation VJP** (below) |
| FEM solve | `tesseracts/{thermal,elastic}_jaxfem` | Python (jax-fem/PETSc) | implicit adjoint (solver is untraceable by JAX) |
| FEM solve, alternate | `tesseracts/elastic_calculix` | **Fortran** (CalculiX 2.23, subprocess) | native `*SENSITIVITY` discrete adjoint **+ our correction** |
| Objective + optimizer | compliance/mass, optax Adam | Python/JAX | autodiff |

Every boundary here is one that ordinarily kills gradients: a GPL-2 Fortran
code speaking input decks, a Rust cdylib, a PETSc-backed solver JAX cannot
trace, and a discrete meshing algorithm with no derivative at all.

## Gradients doing the work (criterion 2)

`examples/fem_bracket_optimization.py` optimizes a parametric mounting
bracket end to end: three named CAD parameters, box-projected optax Adam,
frozen mesh topology per phase with periodic re-extraction.

- Objective (compliance + mass): **17.44 → 4.69 (−73%)** in 30 steps;
  compliance −81%. Convergence figure: `examples/output/fem_bracket_convergence.png`.
- Adjoint vs central finite differences at the optimum path:
  rel. err. **2.3e-5 / 2.0e-3 / 3.3e-7** per parameter.
- `--backend calculix` swaps the Fortran adjoint into the same `jax.grad`
  call; three independent gradient paths (ccx adjoint / jax-fem adjoint / FD)
  agree to 4.5e-5 of gradient scale over 228 design nodes.

## Why this needs Tesseract (criterion 3)

- **CalculiX cannot be traced, linked, or relicensed.** It is GPL-2 Fortran
  driven by text decks. The Tesseract boundary is what lets its native
  discrete adjoint compose with JAX autodiff while staying a subprocess.
  Getting there required source-diving ccx 2.23: its shipped strain-energy
  sensitivity omits the Jacobian-variation term (a non-constant 1.4–4.5×
  error vs FD). We derived and apply the closed-form correction
  `dE/dsᵢ = DFDNᵢ + Σ_q w_q·detJ_q·(∇Nᵢ·nᵢ)`, bringing the adjoint to 2e-4
  of finite differences (`cadjoint/fem/calculix.py`, `research/fem-integration.md`).
- **The mesher VJP is a contract, not a computation.** A mesher's boundary
  vertex `v` lies on the zero set of the trilinearly interpolated SDF samples
  `f_interp(x) = Σᵢ wᵢ(x)·fᵢ`, so the implicit function theorem gives
  `∂v/∂fᵢ = −wᵢ(v)·∇f/|∇f|²` — the interpolation weights at the frozen
  vertex locations *are* the rows of the VJP. Defined purely at the Tesseract
  boundary from inputs and outputs, this makes **any** black-box mesher
  (TetGen, fTetWild, gmsh, our own) differentiable without touching its
  internals, carrying exactly the Hadamard-meaningful normal motion.
- **Performance without leaving the contract.** The Rust kernels (crossing
  sweep, manifold incidence, batched QEF) are 6–107× faster than the Python
  reference with bit-identical topology; the QEF Tesseract's VJP composes
  under both `jax.grad` and `jax.jit`.
- **Solvers are plugins.** One `SolverBackend` ABI routes
  `jaxfem | tesseract | calculix`; `research/simulator-ecosystem.md` ranks 31
  further candidates (jwave, JAX-Fluids, MJX, …) for the same slot.

## Reproduce

```bash
git clone https://github.com/andrinr/cadjoint && cd cadjoint
uv venv && uv sync --extra fem --extra tesseract
uv run python examples/fem_bracket_optimization.py --smoke   # <1 min, asserts descent + FD agreement
uv run python examples/fem_bracket_optimization.py           # full 30-step run, ~10 min CPU
# Fortran adjoint in the loop (needs a ccx binary, see README):
CADJOINT_CCX=... uv run python examples/fem_bracket_optimization.py --smoke --backend calculix
# Rust kernels (optional): cargo build --release --manifest-path native/Cargo.toml
uv run pytest tests/fem tests/meshing -q                     # gradient + parity suites
uv run cadjoint-viewer --open                                # the CAD app itself
```

## Provenance (what was built when)

cadjoint's geometry foundation predates the hackathon: the repo started
2026-01-19, and `main` as of 2026-07-31 contained the SDF primitive kernel,
the sketch/constraint system, and the WebGPU viewer shell — the pre-existing
library this entry composes with, in the same sense every entry composes with
jax-fem, CalculiX, or tesseract-core itself.

**Everything this submission claims was written during the hackathon window
(Aug 3–31 AoE), verifiable commit-by-commit in git history on this branch:**
the differentiable meshing pipeline (edge detection → dual contouring →
exports), the FEM layer and all five Tesseracts, the CalculiX integration
including the ccx 2.23 sensitivity correction, vertex-selection boundary
conditions, the Rust kernel port, the surface-interpolation mesher VJP and
its two-Tesseract validation, and the end-to-end bracket optimization. The
whole repo is licensed Apache 2.0.

## More

The interactive playground (WebGPU viewer with four editing modes) declares
simulation studies and meshes in the scene program itself and edits them by
patching source — meshes are inspectable with a scaled-Jacobian quality
heatmap before solving, results render with slicing, and boundary conditions
are composable vertex selections (`Nodes.box(...) & ~Nodes.sphere(...)`).
Design notes: `research/fem-integration.md`, `research/native-mesher.md`,
`research/simulator-ecosystem.md`, `research/end-to-end-optimization.md`,
`README.md` § "Tesseracts".
