import { defineConfig } from "vitest/config";

// Unit tests cover the pure geometry modules (projection, picking), so no
// Solid/JSX plugin is needed here — keeping this separate from vite.config.ts
// also avoids Vite version skew between the app and the test runner.
export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
