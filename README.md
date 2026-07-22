# jaxCAD

Differentiable SDF primitives, transformations, and constraint system built with JAX.

> [!WARNING]
> The API is not stable. Expect breaking changes.

![primitives](examples/assets/primitives.png)

---

## Features

- **SDF primitives** — sphere, box, capsule, cylinder, torus, and more
- **Boolean ops** — union, intersection, subtraction with smooth blending
- **Transforms** — translate, rotate, scale, and twist, with fluent method chaining
- **Raymarcher** — sphere-tracing renderer with materials, lighting, refraction, and anti-aliasing
- **Differentiable rendering** — gradients flow through the full render pipeline via JAX
- **Constraint system** — distance, angle, parallel, and perpendicular constraints with solver and manifold-projection utilities
- **JAX-native** — every scene is a pure function; `jit`, `grad`, and `vmap` work out of the box

![primitives](examples/assets/constrained_optim.png)
---

## Development install

Clone this repo and
```bash
cd jaxcad
uv sync
uv run pre-commit install
```

The default install works on CPU-only systems. For NVIDIA CUDA 12 support, use:

```bash
uv sync --extra cuda12
```

Matplotlib display helpers, marching cubes, and environment-map loading use the
optional rendering dependencies:

```bash
uv sync --extra render
```

## Tests

```bash
uv run pytest tests/
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
