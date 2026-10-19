# Static viewer

`static.html` draws and exports the bundled scenes with no Python server.

- Scenes are `cadjoint-scene` v2 graphs in `scenes/`, written by
  `python -m cadjoint.backends.jaxjs scene scenes/NAME.py frontend/src/static/scenes/NAME.scene.json --version 2`.
- The WGSL is built in the browser from the graph (`../scene`). It is the
  module the Python worker would send for the same scene.
- Export STL samples the field and runs dual contouring on jax-js (WebGPU
  when available, else Wasm), through the runtime in `web-spike/src`.

```bash
npm run dev:static      # http://localhost:5174/static.html
npm run build:static    # plain files in dist-static/
```

`?scene=NAME` selects the scene to open. `window.__cadjointStatic` exposes the
graphs and `sceneShaders` for tests and benchmarks.
