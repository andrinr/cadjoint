# jaxCAD

Differentiable SDF primitives, transformations, and constraint system built with JAX.

> [!WARNING]
> The API is not stable. Expect breaking changes.

![primitives](examples/assets/primitives.png)

---

## Features

- **SDF primitives** — sphere, box, capsule, cylinder, torus, and more
- **Boolean ops** — union, intersection, subtraction with smooth blending
- **Transforms** — translate, rotate, scale, mirror, repeat
- **Raymarcher** — sphere-tracing renderer with materials, lighting, refraction, and anti-aliasing
- **Differentiable rendering** — gradients flow through the full render pipeline via JAX
- **Shader backends** — compile 3D SDFs through StableHLO to GLSL or WGSL
- **Notebook viewer** — interactively inspect SDFs with WebGPU in Jupyter
- **Constraint system** — geometric constraints (distance, angle, coincident) with Riemannian gradient descent and Newton projection onto the constraint manifold
- **JAX-native** — every scene is a pure function; `jit`, `grad`, and `vmap` work out of the box

![primitives](examples/assets/constrained_optim.png)
---

## Shader compilation and live viewer

Compile an SDF to a standalone shader function:

```python
from jaxcad.backends import GLSLBackend, WGSLBackend
from jaxcad.sdf.primitives import Sphere

sphere = Sphere(radius=1.0)
glsl = GLSLBackend().compile_sdf(sphere)
wgsl = WGSLBackend().compile_sdf(sphere)
```

The Jupyter viewer is an optional dependency:

```bash
pip install -e ".[viewer]"
```

```python
from jaxcad.viewer import SDFViewer

SDFViewer(sphere)
```

See the [WebGPU viewer notebook](examples/webgpu_viewer.ipynb) for composition and
hot-reload examples.

For offscreen OpenGL rendering, install the `glsl` extra instead.

## Development install

Clone this repo and
```bash
cd jaxcad
uv sync
pre-commit install
```

## Tests

```bash
pytest tests/
```

## Docs

Requires [Quarto](https://quarto.org/docs/get-started/) and the `docs` extras:

```bash
pip install -e ".[docs]"
quartodoc build   # generate API reference from docstrings
quarto preview    # serve locally at localhost:4321
```

---

Inspired by [Fidget](https://www.mattkeeter.com/projects/fidget/) and [Inigo Quilez's distance functions](https://iquilezles.org/articles/distfunctions/).

---

![primitives](examples/assets/thingy.png)

## License

[Elastic License 2.0](LICENSE) — free for personal, research, and internal business use. Offering jaxcad as a hosted or managed service requires a commercial license. Contact [andrin.rehmann@gmail.com](mailto:andrin.rehmann@gmail.com) for commercial enquiries.
