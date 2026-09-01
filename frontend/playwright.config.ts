import { defineConfig } from "@playwright/test";

const PORT = Number(process.env.CADJOINT_E2E_PORT ?? 8799);

/**
 * End-to-end tests run against the real playground server serving the built
 * frontend, so they exercise the same static assets and API that ship.
 *
 * They deliberately avoid depending on WebGPU: headless browsers frequently
 * lack it, and every flow under test (selection, code parity, patching) is
 * driven by CPU-side geometry and the JSON API.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? "line" : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1400, height: 900 },
  },
  webServer: {
    command: `uv run python -m cadjoint.viewer.playground --port ${PORT}`,
    cwd: "..",
    url: `http://127.0.0.1:${PORT}/api/session`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
