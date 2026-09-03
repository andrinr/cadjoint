import { cpSync, mkdirSync, mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { defineConfig } from "@playwright/test";

const PORT = Number(process.env.CADJOINT_E2E_PORT ?? 8799);

/**
 * A throwaway scenes directory, seeded with copies of the shipped scenes.
 *
 * The server writes saved scenes to `./scenes` under its working directory,
 * which for these tests is the repository — so a run that exercised Save As
 * left a file in the checkout, and a run that exercised Save overwrote one.
 * `CADJOINT_SCENES_DIR` points the server somewhere disposable instead, and
 * the copies mean the scene browser still has three real programs to list,
 * summarise and draw.
 */
function scenesWorkspace(): string {
  const root = join(mkdtempSync(join(tmpdir(), "cadjoint-e2e-")), "scenes");
  mkdirSync(root, { recursive: true });
  const shipped = resolve(import.meta.dirname, "..", "scenes");
  for (const name of readdirSync(shipped)) {
    if (name.endsWith(".py")) cpSync(join(shipped, name), join(root, name));
  }
  return root;
}

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
    env: { CADJOINT_SCENES_DIR: scenesWorkspace() },
    url: `http://127.0.0.1:${PORT}/api/session`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
