import { defineConfig } from "vitest/config";

// Unit tests cover the pure geometry modules (projection, picking), so no
// Solid/JSX plugin is needed here — keeping this separate from vite.config.ts
// also avoids Vite version skew between the app and the test runner.
export default defineConfig({
  // Solid's browser build, not the server one Node would otherwise resolve:
  // on the server build `createEffect` never runs, so the reactive wiring in
  // `shell/compileCycle.ts` (the lazy mesh-edge fetch) could not be tested.
  resolve: { conditions: ["browser"] },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
