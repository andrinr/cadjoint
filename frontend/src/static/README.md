# Public demo

`static.html` is cadjoint's playground without the Python server: the shipped
scenes, drawn, edited and exported in the browser, built to plain files that
any static host (GitHub Pages) can serve.

It reuses the playground's renderer, viewport (`ViewerPane`), object tree,
render panel and design language. There is no code editor and no Simulate or
Optimize: those need Python.

## What runs where

| Feature | How the demo does it |
| --- | --- |
| Render | WGSL built in the browser from the scene graph (`../scene`), the module the Python worker would send. WebGPU only. |
| Parameters | Sliders and fields per free parameter. An edit writes the shader's parameter buffer at once, then installs the shaders rebuilt at the new values (`parameters.ts`). The module text does not change, so that is a buffer upload, not a recompile. |
| Construction overlay | The compile payload's construction data, exported at build time. Handles bound to a free parameter follow it, and dragging one edits it. |
| Constraint-driven parameters | Read-only, held at the shipped value, drawn as held squares. Moving them needs the constraint solver, which is Python. |
| Wireframe, feature edges | Dual contouring on jax-js (`jaxjs.ts`), 64 cells on the longest side. Feature edges are a dihedral-angle crease test on that mesh (`edges.ts`), not the Python overlay's lattice classifier, which also projects CSG seams onto both operands and is not ported. |
| STL export | The same dual contouring at 64 to 160 cells, of the current values. |
| Compute device | jax-js on WebGPU, else WebAssembly. Without WebGPU the viewport cannot draw; parameters and export still work on WebAssembly. |

## The bundle

`bundle/` holds one `NAME.demo.json` per scene in `scenes/` and the public
tier's capabilities (`public-tier.json`). The format is
`cadjoint/viewer/schema/demo.py` (`DemoScene`), emitted to TypeScript with the
rest of the API contract. It is generated, and committed so the demo builds
without Python:

```bash
npm run bundle:static     # python -m cadjoint.viewer.demo, from the repository root
```

`tests/viewer/test_demo_bundle.py` fails when the committed bundle is not the
build of the committed scenes.

## Run it locally

```bash
npm run dev:static        # http://localhost:5174/static.html, live reload
npm run build:static      # plain files in dist-static/ (index.html)
npx vite preview --config vite.static.config.ts   # serves dist-static on http://localhost:4318
```

`?scene=NAME` opens a scene. `window.__cadjointDemo()` reports the scene, the
values on screen, and the renderer's upload and pipeline counters.

Tests: `npx vitest run test/demoParameters.test.ts test/demoBundle.test.ts`,
and `npm run e2e:static`, which builds the demo, serves it on port 4319 and
drives it in Chromium: load a scene, move a slider, check the render changed
without a pipeline rebuild, export an STL.

## Deploy

The docs workflow (`.github/workflows/docs.yml`) builds the demo and copies
it to `demo/` at the repository root (git-ignored), which `_quarto.yml` lists
as a resource, so the docs site serves it at
https://andrinr.github.io/cadjoint/demo/ and the home page and navbar link to
it. The docs deploy on every push to main publishes it.

To check the site with the demo locally:

```bash
npm run build:static && rm -rf ../demo && cp -r dist-static ../demo
cd .. && quartodoc build && quarto render    # _site/, with _site/demo/
```

The build keeps its relative base (`./`), so it works under `/cadjoint/demo/`
or any other path. An absolute `--base` breaks the fonts: Quarto rewrites
site-absolute `url(/…)` in the CSS it copies.
