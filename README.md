# cadjoint

Differentiable code-first CAD: sketches, constraints, SDF geometry, meshing, and
FEM simulation composed into one function JAX can differentiate end to end.

> [!WARNING]
> The API is not stable. Expect breaking changes.

[![The cadjoint playground in Model mode: scene.py on the left, the parametric heat sink rendered live on the right, and the declared cool-sink optimization in the side panel](examples/assets/playground-model.png)](https://andrinr.github.io/cadjoint/docs/viewer.html)

---

## Features

- **SDF primitives** — sphere, box, capsule, cylinder, torus, and more
- **Boolean ops** — union, intersection, subtraction with smooth blending
- **Transforms** — translate, rotate, scale, mirror, repeat
- **Forward raymarcher** — early-exit sphere tracing, reconstructed silhouettes, GGX materials, soft shadows, reflections, refraction, and anti-aliasing
- **Shader backend** — compile 3D SDFs through StableHLO to WGSL
- **WebGPU viewers** — interactively inspect SDFs or progressively path-trace
  materials in the browser playground
- **Sketch construction** — 2D profiles on work planes, extruded or revolved into
  solids that share their parameters, so constraints and gradients act on both
- **Construction primitives** — boxes, spheres, and cylinders with editable
  placement, mirroring the SDF primitives they generate
- **Editable in the browser** — construction geometry renders as depth-tested
  overlays you can click, drag, place, and transform with a gizmo; every edit
  rewrites the Python source that produced it
- **Constraint system** — geometric constraints (distance, angle, coincident) with Riemannian gradient descent and Newton projection onto the constraint manifold
- **JAX-native** — every scene is a pure function; `jit`, `grad`, and `vmap` work out of the box

---

## Install

Clone the repo and sync with
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/andrinr/cadjoint
cd cadjoint
uv venv                     # create .venv (--python 3.12 pins a version)
uv sync                     # CPU JAX — macOS, Linux, and Windows
# uv sync --extra cuda      # Linux + NVIDIA GPU
uv run pre-commit install   # optional: lint and format on commit
```

`uv sync` installs cadjoint into `.venv` in editable mode — creating the
environment first if you skipped `uv venv`. Run commands through it with
`uv run <cmd>`, or activate it once per shell:

```bash
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

The default sync pulls plain `jax`/`jaxlib`, so it works on Apple Silicon and any
CPU-only machine. CUDA wheels are Linux-only, so GPU support is opt-in through
the `cuda` extra.

Optional extras — repeat the flag, one `--extra` per name:

```bash
uv sync --extra viewer --extra fem --extra docs
```

| Extra | Pulls in |
| --- | --- |
| `cuda` | GPU JAX (Linux + NVIDIA only) |
| `viewer` | Jupyter widget (`anywidget`) |
| `fem` | jax-fem finite-element stack (basix, meshio, petsc4py) |
| `tesseract` | tesseract-core + tesseract-jax solver plugin runtime |
| `stepcheck` | OCCT kernel validation of STEP exports (dev) |
| `docs` | Quarto API reference (`quartodoc`) |

Avoid `--all-extras` on macOS — it includes `cuda`, which has no macOS wheels.

## Interactive browser playground

Start a local server for the split-pane Python editor and live WebGPU preview:

```bash
uv run cadjoint-viewer --open   # serves http://127.0.0.1:8765/ and opens your browser
```

Equivalent invocations and options (drop `uv run` inside an activated `.venv`):

```bash
uv run python -m cadjoint.viewer.playground   # same server, no browser launch
uv run cadjoint-viewer --port 9000            # pick a different port
uv run cadjoint-viewer --help                 # list all flags
```

Then open <http://127.0.0.1:8765/> if you did not pass `--open`. No extra
dependencies are needed — the server is stdlib-only — but the preview needs a
WebGPU-capable browser (recent Chrome, Edge, or Safari). Stop the server with
`Ctrl+C`.

Edit the example on the left and run it with `Ctrl+Enter` (or `Cmd+Enter`). The
program must assign its final SDF to `scene`. Use **Path trace** for progressive
multi-bounce lighting, GGX reflections, and glass transport; camera and scene
changes reset accumulation automatically. The server only listens on localhost
and compiles each edit in a timed child process, but the editor still executes
Python on your machine—only run code you trust.

### Modelling in the viewer

Construction geometry — sketch profiles and primitives — is drawn over the
rendered solid as a depth-tested wireframe (an edge behind the model is hidden
by it) and can be edited directly:

| Action | Result |
| --- | --- |
| Click a vertex handle | Selects it and highlights the exact literal in the code |
| Drag a handle | Rewrites that vertex's coordinates and rebuilds the solid |
| **Polygon**, then click edges | Inserts a vertex per click until Esc |
| Select a handle, press Delete | Removes that vertex |
| **Box** / **Sphere** / **Cylinder**, then click | Writes a `Solid.*` call and adds it to the scene |
| Click a solid's outline | Selects it and shows the move/rotate gizmo |
| Drag a gizmo arrow or ring | Rewrites `position=` or `rotation=` on that solid |
| Drag empty space / Shift-drag / scroll | Orbit / pan / zoom |

Every edit is applied to the Python source, which stays the single source of
truth — there is no hidden scene state to drift out of sync. Geometry whose
literals cannot be rewritten (built in a loop, or from a variable) still
renders, but is read-only in the viewer.

Solids created this way come from the construction layer, so they are ordinary
parametric geometry as well as viewer objects:

```python
from cadjoint.construction import Solid
from cadjoint.sdf.boolean import Union

scene = Union(
    Solid.box(size=[1, 1, 0.5], position=[0, 0, 0], rotation=[0, 0, 0.4]),
    Solid.sphere(radius=0.6, position=[1.5, 0, 0]),
)
```

`size`, `position`, and `rotation` become named free parameters shared with the
SDF the factory returns, so constraints and `jax.grad` reach them exactly as
they do for sketch vertices. `size` is half-extents and `rotation` is intrinsic
X, Y, Z angles in radians, matching the underlying primitives.

![Simulate mode with the sink-mesh SimMesh generated into 2969 tet10 elements, shaded by element quality with the edges and quality histogram shown](examples/assets/playground-mesh.png)

*Simulate → Meshes: the declared `SimMesh` discretized into tet10 elements,
shaded by element quality with a quality histogram beside it.*

![The sink-conduction thermal study solved in the playground, sliced through X so the hot die interface and the temperature gradient into the fins are visible](examples/assets/playground-field.png)

*Simulate → Studies: the solved temperature field on the same mesh, clipped by
the slice plane so the heat flux entering at the die interface is visible.*

### Developing the playground UI

The UI is a Solid + TypeScript app in `frontend/`, built into
`cadjoint/viewer/static` and committed, so installing cadjoint needs no Node
toolchain. To work on it:

```bash
cd frontend
npm install
npm run dev        # Vite on :5173, proxying the API to the Python server
npm run build      # refresh cadjoint/viewer/static (commit the result)
npm test           # projection and picking unit tests
npm run e2e        # Playwright, drives the real server end to end
```

Run `uv run cadjoint-viewer` alongside `npm run dev` so the dev server has an API
to proxy to.

## Shader compilation and live viewer

Compile an SDF to a standalone shader function:

```python
from cadjoint.backends import compile_sdf_to_wgsl, compile_scene_to_wgsl
from cadjoint.sdf.primitives import Sphere

sphere = Sphere(radius=1.0)
wgsl = compile_sdf_to_wgsl(sphere)
wgsl_scene = compile_scene_to_wgsl(sphere)
```

`compile_scene_to_wgsl` emits `sdf`, `material_base` (RGB + roughness), and
`material_optics` (metallic + opacity + IOR + reflectivity) from the same scene
snapshot, ready to embed in a WebGPU renderer.

The Jupyter viewer is an optional dependency:

```bash
uv sync --extra viewer
```

```python
from cadjoint.viewer import SDFViewer

SDFViewer(sphere)
```

See the [rendering notebook](examples/rendering.ipynb) for composition and
hot-reload examples.

For a split-pane Python editor and live WebGPU preview in the browser, see
[Interactive browser playground](#interactive-browser-playground) above.

The [WebGPU viewer guide](https://andrinr.github.io/cadjoint/docs/viewer.html) includes
a live interactive scene and covers the local playground, camera controls,
generated shader inspection, and the Jupyter widget.


## Forward rendering

The image renderer groups scene data and quality controls explicitly:

```python
from cadjoint.render import Camera, RenderSettings, Scene, render_scene
from cadjoint.sdf.primitives import Sphere

scene = Scene(
    Sphere(1.0),
    camera=Camera(position=(0, 1.5, 5), target=(0, 0, 0)),
)
image = render_scene(scene, RenderSettings.balanced((240, 320)))
```

Use `RenderSettings.draft()`, `.balanced()`, or `.high_quality()` to choose an
explicit performance/fidelity trade-off. See the [forward renderer guide](https://andrinr.github.io/cadjoint/docs/rendering.html)
for mode and quality comparisons.

## End-to-end optimization

The pipeline is differentiable from the first named dimension to the last
solver residual, and `examples/fem_bracket_optimization.py` walks the whole
chain on the parametric L-bracket from `scenes/bracket.py`:

**named CAD parameters** (`web_thickness`, `rib_height`, `plate_thickness`)
→ **bracket SDF** → **HEX8 mesh** (frozen topology, node positions recomputed
differentiably per candidate) → **linear elastic solve** (jax-fem adjoint, or
CalculiX's native `*SENSITIVITY` via `--backend calculix`) → **compliance +
mass objective** → **optax Adam** with box-bound projection.

Meshing splits into a discrete half (which cells are inside, how they connect)
and a continuous half (where the nodes sit). The discrete half cannot be
differentiated, so topology stays frozen while gradients flow through the node
positions, and every few steps the mesh is re-extracted at the current design —
the small objective jumps at those steps are the discretization being refreshed.

```bash
uv pip install optax                              # optimizer (one-off)
uv run python examples/fem_bracket_optimization.py            # ~10 min, CPU
uv run python examples/fem_bracket_optimization.py --smoke    # 2 cheap steps
```

Requires the `fem` extra (`uv sync --extra fem`). The run validates the adjoint
gradient against finite differences, descends for 30 steps, prints a summary
table, and writes convergence CSV + figure and before/after VTK files to
`examples/output/`.

The same loop runs interactively in the playground:

![The cool-sink optimization finished in the playground: convergence sparkline, trajectory scrubber, objective 1.618 to 1.601 over 4 steps, and the before/after parameter table](examples/assets/playground-optimize.png)

*Simulate → Optimize: four Adam steps of `cool-sink`, minimizing peak
temperature. The panel shows the convergence sparkline, a scrubber that replays
the geometry along the trajectory, and each parameter's before → after value;
the optimizer writes the new `fin_depth` straight back into `scene.py` on the
left.*

## Tesseracts: one differentiable function across four AD strategies

Real engineering pipelines die at tool boundaries — a Fortran solver here, a
Rust kernel there, a mesher nobody can differentiate — and the gradients die
with them. cadjoint crosses those boundaries with
[Tesseract](https://github.com/pasteurlabs/tesseract-core): every non-JAX
component is packaged as a Tesseract exposing typed `apply` and
`vector_jacobian_product` endpoints, and
[tesseract-jax](https://github.com/pasteurlabs/tesseract-jax) lifts each one
into a JAX primitive. The result is a single function from CAD parameters to
engineering objective that `jax.grad` differentiates end to end, even though
no two stages agree on how to compute a derivative.

### The Tesseracts in this repo

| Tesseract | Wraps | Boundary crossed | How its VJP works |
| --- | --- | --- | --- |
| `cadjoint/fem/tesseracts/thermal_jaxfem` | jax-fem Poisson solve | AD strategy: JAX cannot trace the solver (PETSc assembly) | Implicit adjoint — one transposed linear solve per cotangent |
| `cadjoint/fem/tesseracts/elastic_jaxfem` | jax-fem linear elasticity + von Mises | same | same; gradients bit-identical to the in-process path |
| `cadjoint/fem/tesseracts/elastic_calculix` | **CalculiX 2.23, Fortran**, over a subprocess (text decks in, result files out) | Language, licence (GPL-2 isolated), and AD strategy | The solver's native `*SENSITIVITY` discrete adjoint, plus a correction we derived for a missing Jacobian-variation term in ccx 2.23 — validated to 2e-4 of finite differences |
| `native/tesseract_api.py` | **Rust** dual-contouring kernels (batched QEF solves, rayon-parallel, 90–107× faster than the reference) | Language and memory model (cdylib over ctypes) | Hand-derived linear-solve VJP, matches JAX autodiff to ~4e-14 |
| `cadjoint/fem/tesseracts/tetfill` | **TetGen** — the tet fill alone, with dual contouring left differentiable in JAX upstream | A discrete meshing algorithm, cut at the narrowest place it can be cut | **Pass-through gather**: TetGen's `-Y` preserves the input vertices bit-for-bit, so the VJP is the exact transpose of a gather (0.0 relative error for TET4, 1.25e-16 for TET10); Steiner nodes take zero cotangent, and interior relaxation carries their sensitivity onto the boundary |
| `cadjoint/fem/tesseracts/mesher` | The **whole black-box mesher** — dual-contoured surface into a tetrahedral volume mesher whose internals nobody differentiates | The boundary everyone gives up on: a discrete, non-differentiable meshing algorithm | **Surface-interpolation VJP**: a boundary vertex lies on the zero set of the trilinearly interpolated SDF samples, so the implicit function theorem gives `∂v/∂fᵢ = −wᵢ(v)·∇f/|∇f|²` — the interpolation weights at the frozen vertex locations *are* the VJP rows. Any mesher becomes differentiable without touching its internals; only the Hadamard-meaningful normal motion is carried |

### The composition

```
CAD parameters θ  ──►  constraints ──► SDF        (JAX autodiff)
        │                              │
        │                              ▼
        │                  dual-contoured surface   (JAX autodiff:
        │                              │             Newton on the true SDF)
        │                              ▼
        │              ┌─ tetfill Tesseract ─────┐ (TetGen; frozen topology,
        │              │  surface → TET4/TET10   │  exact pass-through VJP)
        │              └───────────┬─────────────┘
        │                          ▼
        │              ┌─ solver Tesseract ──────┐ (jax-fem adjoint, or
        │              │  thermal / elastic FEM  │  CalculiX Fortran adjoint)
        │              └───────────┬─────────────┘
        │                          ▼
        └──────────  ∂J/∂θ  ◄──  objective J      (JAX autodiff)
```

**This is what the playground runs.** The starter scene's `cool-sink`
optimization declares `gradient_path="tesseract-dc"`, so pressing Run in the
browser drives exactly this chain — roughly 8 s per step on the heat sink,
streamed live, with the optimized parameters written back into the source.
`gradient_path="direct"` runs the same objective fully in-process for
comparison.

A second, more general boundary also ships: the `mesher` Tesseract wraps the
*whole* pipeline (samples → surface → mesh) and derives its VJP from the
interpolated field alone, so it differentiates meshers that do not preserve
their input vertices — at the cost of the interpolant smearing sharp features.
`tetfill` is sharper wherever TetGen's vertex-preserving mode applies;
`mesher` is the one to reach for when the mesher is a true black box (and it
carries the HEX8 path). Numbers for both: `research/tet-vs-hex.md`.

`examples/fem_bracket_optimization.py` runs this loop for real: optax Adam over
named CAD parameters, objective down 73% in 30 steps, adjoint checked against
finite differences at every boundary (2e-5 … 3e-7 per parameter), and
`--backend calculix` swaps a 1990s Fortran code into the same `jax.grad` call
without changing a line of the objective.

### Why Tesseract is load-bearing here

- **CalculiX cannot be traced, linked, or relicensed** — it is GPL-2 Fortran
  speaking input decks. The Tesseract boundary is what lets its native adjoint
  compose with JAX autodiff while staying a subprocess.
- **The mesher VJP is a contract, not a computation** — the surface-
  interpolation map is defined at the Tesseract boundary from the *inputs and
  outputs alone*, which is what makes swapping TetGen, fTetWild, or gmsh
  behind it a zero-cost experiment.
- **Solvers are plugins.** `SolverBackend` routes `backend="jaxfem" | "tesseract"
  | "calculix"` through one ABI; the survey in
  `research/simulator-ecosystem.md` ranks 31 further candidates (jwave,
  JAX-Fluids, MJX, …) that drop into the same slot.
- **It stays fast.** Tesseracts here run in-process via
  `Tesseract.from_tesseract_api` — no Docker, ~0.14 s per apply/VJP roundtrip —
  with containerization available when a component needs isolation.

### Building and serving them as containers

Each of the six directories above is a complete Tesseract package —
`tesseract_api.py` plus `tesseract_config.yaml` plus a requirements file — so
`tesseract build` turns it into a Docker image with no extra glue:

```bash
uv sync --extra tesseract          # installs the tesseract-core SDK
tesseract build native                                  # ~1 min
tesseract build cadjoint/fem/tesseracts/mesher          # ~1 min
tesseract build cadjoint/fem/tesseracts/elastic_calculix # ~1.5 min
tesseract build cadjoint/fem/tesseracts/thermal_jaxfem  # ~3 min
tesseract build cadjoint/fem/tesseracts/elastic_jaxfem  # ~3 min
```

Then serve one and call it like any other Tesseract — the same client code
that runs it in-process runs it in a container:

```python
from tesseract_core import Tesseract

with Tesseract.from_image("cadjoint_qef_native:latest") as t:
    vertices = t.apply(inputs)["vertices"]
    grads = t.vector_jacobian_product(
        inputs, vjp_inputs=["points", "normals"],
        vjp_outputs=["vertices"], cotangent_vector={"vertices": cotangent},
    )
```

`tesseract serve <image>` plus `Tesseract.from_url(...)` works identically for
a long-lived server. What each image contains, and what it costs:

| Image | Requirements provider | The non-obvious payload | Size |
| --- | --- | --- | --- |
| `cadjoint_mesher` | pip | cadjoint + TetGen + SciPy on a uv-installed CPython 3.12 | 1.36 GB |
| `cadjoint_qef_native` | pip | the Rust cdylib, `cargo build --release --locked` inside the image (toolchain installed and purged in one layer), pinned via `CADJOINT_NATIVE_MESHER` | 1.39 GB |
| `cadjoint_elastic_calculix` | conda | **the ccx 2.23 Fortran binary** from conda-forge at `/python-env/bin/ccx`, pinned via `CADJOINT_CCX` | 2.57 GB |
| `cadjoint_thermal_jaxfem` | conda | the full jax-fem stack: PETSc/petsc4py 3.25.5, gmsh, fenics-basix, meshio | 5.51 GB |
| `cadjoint_elastic_jaxfem` | conda | same | 5.51 GB |

Two packages use pip (`tesseract_requirements.txt`), three use the SDK's conda
provider (`tesseract_environment.yaml`) — because petsc4py publishes *no* PyPI
wheels at all and gmsh publishes manylinux wheels for x86-64 only, so the
jax-fem stack is simply not pip-installable on Linux, and ccx is a Fortran
binary that no Python requirement can supply. In every case cadjoint itself
is installed as a local path dependency (`../../../..`), which `tesseract
build` stages into the build context automatically.

Two behaviours to know when calling a *served* image rather than an
in-process one (both are properties of tesseract-core 1.11, not of cadjoint):

- **Zero-size arrays cannot cross the HTTP boundary.** Polymorphic array
  dimensions validate as `PositiveInt`, so an empty `(0, …)` input is
  rejected. The mesher's discovery mode (empty `point_ids` / `cell_template`)
  therefore has to run in-process; pass the frozen topology it returns to the
  served image.
- **TetGen topology is platform-dependent.** The same field meshes to 182
  points on macOS/arm64 and 185 in the Linux container, so a frozen-topology
  promise made on the host does not transfer to the container for TET4/TET10.
  HEX8 (voxelize + Newton-snap) is deterministic across both.

`tests/fem/test_tesseract_packaging.py` validates every package against the
installed SDK schema without Docker, and round-trips the built images against
the in-process path when Docker is present.

Design notes: `research/fem-integration.md` (solver ABI, adjoint mechanics,
the ccx sensitivity correction, container conformance and measured numbers),
`research/native-mesher.md` (Rust core and its VJP), `research/tet-vs-hex.md`
(the mesher-Tesseract validation matrix).

## Tesseract Hackathon 2026 entry

cadjoint is an entry in the [Tesseract Hackathon 2026](https://si-tesseract.discourse.group)
(**Track 01 — Inverse design & shape optimization**). The entry state is
frozen on the branch
[`tesseract-hackathon-2026`](https://github.com/andrinr/cadjoint/tree/tesseract-hackathon-2026),
which is kept permanently; development continues on `main`.

**The headline workflow** is the two-Tesseract differentiable chain built on
a surface-interpolation VJP:

```
CAD params θ ──► SDF lattice samples ──► ┌ mesher Tesseract ┐ ──► ┌ solver Tesseract ┐ ──► J
   (JAX)              (JAX)              │  black-box mesh  │      │  FEM adjoint     │
                                         └──────────────────┘      └──────────────────┘
                     ∂J/∂θ  ◄──  one jax.grad call through both boundaries
```

The mesher Tesseract wraps a *non-differentiable* meshing pipeline (dual-
contoured surface → TetGen / hex voxelization). Its VJP never looks inside
the mesher: a boundary vertex `v` lies on the zero set of the trilinearly
interpolated lattice samples, so the implicit function theorem gives
`∂v/∂fᵢ = −wᵢ(v)·∇f/|∇f|²` — **the interpolation weights at the frozen
vertex locations are the VJP rows**, making any mesher differentiable from
its inputs and outputs alone. Composed with the unmodified jax-fem solver
Tesseract, one `jax.grad` call crosses both boundaries: the VJP matches
autodiff of the same map to 1.4e-11, finite differences confirm the smooth
parameters, and five gradient steps drop the bracket objective 36%, with
the Tesseract's frozen-topology promise detecting when a step demands
re-meshing. Validation matrix: `research/tet-vs-hex.md`; runnable demo:
`tests/fem/test_tetmesh.py::TestTwoTesseractChain` (with `-s`).

Provenance: the geometry foundation (SDF kernel, constraints, viewer shell)
predates the hackathon; every contribution above — the meshing pipeline,
all Tesseracts, the CalculiX adjoint correction, the mesher VJP, and the
end-to-end optimization — was written during the hackathon window
(Aug 3–31, 2026), verifiable commit-by-commit on the frozen branch. Whole
repo under Apache 2.0.

## Tests

```bash
uv run pytest tests/
```

## Docs

Requires [Quarto](https://quarto.org/docs/get-started/) and the `docs` extras:

```bash
uv sync --extra docs
uv run quartodoc build   # generate API reference from docstrings
quarto preview           # serve locally at localhost:4321
```

---

Inspired by [Fidget](https://www.mattkeeter.com/projects/fidget/) and [Inigo Quilez's distance functions](https://iquilezles.org/articles/distfunctions/).

## License

[Apache License 2.0](LICENSE).
