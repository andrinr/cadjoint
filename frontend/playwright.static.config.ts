import { defineConfig } from "@playwright/test";

const PORT = Number(process.env.CADJOINT_DEMO_PORT ?? 4319);

/**
 * End-to-end tests of the public demo, against its production build served
 * as plain files: no Python server anywhere, which is the point.
 *
 * The render half needs WebGPU, so the browser is started with it on; where
 * the build of Chromium has no adapter, those checks skip themselves and the
 * rest (parameters, STL export on WebAssembly) still runs.
 */
export default defineConfig({
  testDir: "./e2e/static",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  workers: 1,
  reporter: process.env.CI ? "line" : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1440, height: 900 },
    acceptDownloads: true,
    launchOptions: {
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,WebGPU",
        "--use-angle=metal",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
        "--enable-gpu",
      ],
    },
  },
  webServer: {
    command: `npm run build:static && npx vite preview --config vite.static.config.ts --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
