# cadjoint

[![PyPI](https://img.shields.io/pypi/v/cadjoint)](https://pypi.org/project/cadjoint/)
[![Python](https://img.shields.io/pypi/pyversions/cadjoint)](https://pypi.org/project/cadjoint/)
[![License](https://img.shields.io/pypi/l/cadjoint)](https://github.com/andrinr/cadjoint/blob/main/LICENSE)

Differentiable code-first CAD: sketches, constraints, SDF geometry, meshing and
FEM simulation composed into one function JAX can differentiate end to end.

> [!WARNING]
> The API is not stable. Expect breaking changes.

[![The bracket in the playground's Sketch mode. The camera turns around the part, then the free handle at the tip of the gusset rib is dragged up the vertical web; the rib and its smooth-union blend follow at frame rate, and the rib_tip literal in the editor is rewritten on release.](https://raw.githubusercontent.com/andrinr/cadjoint/main/docs/assets/motion/bracket-orbit-drag.webp)](https://andrinr.github.io/cadjoint/docs/viewer.html)

*Dragging the `rib_tip` handle updates the shader's parameter buffer each frame and rewrites the `Vector2` literal in `scene.py` on release.*

## Pipeline

```
sketch vertices ─► constraints ─► SDF ─► mesh ─► FEM solve ─► objective
       └─────────────────────────── ∂J/∂θ ◄───────────────────────────┘
```

One Python program declares every stage, and JAX differentiates through each:

- A sketch profile is a list of named `Vector2` parameters.
- Constraints are residuals on those parameters, solved by Newton projection
  onto the constraint manifold.
- Extruding or revolving the profile gives an SDF that shares the parameter
  objects.
- Dual contouring extracts a surface, TetGen or Gmsh fills it, and jax-fem or
  CalculiX solves on it.
- A study and an `Optimization` are declared in the scene beside the geometry.
  Each optimization step is a mesh, a solve and an adjoint, projected back onto
  the constraints.
- No boundary representation is stored; every surface is extracted from the
  SDF at a chosen resolution.

[![The end cap's cap-conduction thermal study solved from the Simulate desk: the job chip counts the seconds while the cut-cell mesh is assembled and solved, then the temperature field appears on the part with its legend and the camera turns around it.](https://raw.githubusercontent.com/andrinr/cadjoint/main/docs/assets/motion/end-cap-solve.webp)](https://github.com/andrinr/cadjoint/blob/main/scenes/end_cap.py)

*The end cap's `ThermalStudy` is meshed and solved with one click.*

[![The heat sink's cool-sink optimization run from the Optimize panel: eight Adam steps stream in with the objective trace, then the scrubber replays the path while the fin comb changes proportions.](https://raw.githubusercontent.com/andrinr/cadjoint/main/docs/assets/motion/heat-sink-optimize.webp)](https://github.com/andrinr/cadjoint/blob/main/scenes/starter.py)

*The heat sink starts overbuilt; its declared `Optimization` slims it by descending on the gradient of peak temperature with respect to the sketch.*

[![The heat sink in Sketch mode: the armed sketch tool shows a ghost plane labelled new sketch on the floor; hovering the fin comb shows its plane, frame and normal, and dragging the gizmo's Z arrow lifts the comb while the editor's SketchPlane origin is rewritten.](https://raw.githubusercontent.com/andrinr/cadjoint/main/docs/assets/motion/sketch-planes.webp)](https://github.com/andrinr/cadjoint/blob/main/scenes/starter.py)

*In Sketch mode, dragging a literal work plane by its gizmo rewrites its `SketchPlane(origin=...)`; a plane taken from a face (`SketchPlane.on(body.cap("+"))`) is instead an expression over the parent feature's parameters, so it is differentiable.*

## Features

- **Geometry**: SDF primitives, smooth booleans, transforms, patterns;
  sketches on work planes extruded, revolved or lofted into solids that share
  their parameters; reference planes from faces, offsets and midplanes
- **Constraints**: distance, angle, coincident, horizontal, vertical,
  parallel, perpendicular, equal-length, point-on-line, fixed
- **Materials**: a sourced catalogue of density, conductivity, elastic
  constants and yield strength in SI, sampled per element by the solver
- **Meshing**: dual contouring with sharp-feature vertices; TET4/TET10 through
  TetGen, second-order tets through Gmsh, a voxelize-and-snap HEX8 path, or cut
  cells with no volume mesh; OBJ, STL and faceted STEP export
- **Simulation**: thermal and linear-elastic studies with programmatic node
  selections, solved by jax-fem or CalculiX 2.23; a lattice-Boltzmann flow
  prototype
- **Optimization**: declared in the scene, descended with optax, every step
  projected back onto the constraints
- **Plugins**: every non-JAX component (meshers, solvers) behind one
  `apply`/`vjp` interface, implemented as a
  [Tesseract](https://github.com/pasteurlabs/tesseract-core) and run
  in-process, in a container or remotely
- **Rendering**: a forward raymarcher in JAX, and a StableHLO → WGSL shader
  compiler for the browser
- **Playground**: a WebGPU app in which every edit rewrites the Python source
  that produced it

## Install

```bash
pip install cadjoint         # pin an exact version: the API is not stable
```

Everything beyond the geometry core is an extra (`pip install "cadjoint[fem]"`,
or `uv sync --extra fem` in a checkout, one `--extra` per name):

| Extra | Pulls in |
| --- | --- |
| `cuda` | GPU JAX (Linux + NVIDIA only; no macOS wheels, so avoid `--all-extras` on a Mac) |
| `fem` | jax-fem finite-element stack (basix, meshio, petsc4py, gmsh); compiles PETSc |
| `tesseract` | tesseract-core + tesseract-jax, the plugin runtime |
| `gmsh` | Gmsh in-process for the tet10 mesher (GPL, hence its own extra) |
| `viewer` | the playground server (`pydantic`), the Jupyter widget (`anywidget`) and the process monitor (`psutil`) |
| `editor` | playground lint, completion and signature help (`ruff`, `jedi`) |
| `stepcheck` | OCCT validation of STEP exports (dev) |
| `docs` | Quarto API reference (`quartodoc`) |

Installing from a checkout, the platforms CI verifies, and the toolchain the
`fem` extra needs are in
[Getting started](https://andrinr.github.io/cadjoint/docs/getting-started.html#install).

## Playground

```bash
uvx --from "cadjoint[viewer,editor]" cadjoint-viewer --open      # no checkout
uv run --extra viewer --extra editor cadjoint-viewer --open      # from a checkout
```

The editor holds `scene.py` and the viewport shows the compiled field. Vertex
handles, gizmos, material swatches, face picks, constraint chips, solver runs
and the properties window all write back into the source.

[![The properties window: clicking the fin comb's extrude row in the object tree shows its profile, its depth (the named parameter fin_depth) and its material. Typing a new depth deepens the comb, and choosing copper from the material dropdown recolours it.](https://raw.githubusercontent.com/andrinr/cadjoint/main/docs/assets/motion/properties-window.webp)](https://github.com/andrinr/cadjoint/blob/main/scenes/starter.py)

*The properties window edits the arguments of the selected call: here the comb's depth and material.*

The server listens only on localhost but executes the scene's Python on your
machine; only run code you trust. The
[playground guide](https://andrinr.github.io/cadjoint/docs/viewer.html) covers
every window, the scenes directory and browser support.

## Python example

```python
from cadjoint.construction import PolygonProfile, SketchPlane, extrude
from cadjoint.constraints import DistanceConstraint, satisfy_constraints
from cadjoint.geometry import Scalar, Vector2
from cadjoint.meshing import GridSpec, extract_mesh

a = Vector2([0.0, 0.0], free=True, name="a")
b = Vector2([1.0, 0.0], free=True, name="b")
c = Vector2([1.0, 1.0], free=True, name="c")
d = Vector2([0.0, 1.0], free=True, name="d")
DistanceConstraint(a, b, 1.2)

depth = Scalar(0.5, free=True, name="depth")
block = extrude(PolygonProfile([a, b, c, d], plane=SketchPlane()), depth=depth)
satisfy_constraints(block)          # projects a and b onto the constraint, in place

grid = GridSpec.from_bounds((-0.5, -0.5, -0.5), (2.0, 2.0, 1.5), 32)
mesh = extract_mesh(block, grid)
```

`functionalize(block)` returns a pure function of the parameter values, so
`jax.grad` of anything downstream reaches `a`, `b`, `c`, `d` and `depth`.
[Getting started](https://andrinr.github.io/cadjoint/docs/getting-started.html)
continues to meshing, a thermal study and an optimization;
`examples/fem_bracket_optimization.py` runs the full chain from the command
line, with the adjoint checked against finite differences at every boundary.

## Documentation

- [Getting started](https://andrinr.github.io/cadjoint/docs/getting-started.html)
- [Differentiable meshing](https://andrinr.github.io/cadjoint/docs/meshing.html),
  [simulation](https://andrinr.github.io/cadjoint/docs/simulation.html),
  [flow solver](https://andrinr.github.io/cadjoint/docs/flow.html),
  [optimization](https://andrinr.github.io/cadjoint/docs/optimization.html),
  [materials](https://andrinr.github.io/cadjoint/docs/materials.html)
- [Plugins](https://andrinr.github.io/cadjoint/docs/plugins.html): the Tesseract
  contract, the shipped packages, containers and remote transports
- [Rendering](https://andrinr.github.io/cadjoint/docs/rendering.html) and the
  [playground](https://andrinr.github.io/cadjoint/docs/viewer.html)
- [Research notes](https://github.com/andrinr/cadjoint/blob/main/research/README.md):
  design records for the meshing pipeline, FEM integration, the flow solver,
  the constraint system, and the measured
  [performance profile](https://github.com/andrinr/cadjoint/blob/main/research/performance.md)

## Developing

```bash
uv run pytest tests -q --ignore=tests/fem   # fast gate
uv run pytest tests/fem -q                  # needs the fem extra; CalculiX via CADJOINT_CCX
uv run pre-commit install                   # ruff on commit
```

The playground UI is a Solid + TypeScript app in `frontend/`, built into
`cadjoint/viewer/static` and committed, so installing cadjoint needs no Node
toolchain; its workflow is in
[developing the UI](https://andrinr.github.io/cadjoint/docs/viewer.html#developing-the-ui).
Docs need [Quarto](https://quarto.org/docs/get-started/) and the `docs` extra:
`uv run quartodoc build`, then `quarto preview`.
`benchmarks/jax_compile_profile.py` splits a worker request into trace, lower,
XLA compile and cache time per program;
[`research/performance.md`](https://github.com/andrinr/cadjoint/blob/main/research/performance.md)
records what each optimization bought.

## History

cadjoint began as an entry in the
[Tesseract Hackathon 2026](https://si-tesseract.discourse.group) (Track 01,
inverse design and shape optimization); that state is frozen on the
[`tesseract-hackathon-2026`](https://github.com/andrinr/cadjoint/tree/tesseract-hackathon-2026)
branch. Inspired by [Fidget](https://www.mattkeeter.com/projects/fidget/) and
[Inigo Quilez's distance functions](https://iquilezles.org/articles/distfunctions/).

## License

[Apache License 2.0](https://github.com/andrinr/cadjoint/blob/main/LICENSE).
