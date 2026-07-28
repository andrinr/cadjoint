import { chromium, expect, test } from "@playwright/test";

/**
 * Renders the playground on a real GPU and screenshots it.
 *
 * Skipped automatically when the browser exposes no WebGPU adapter (common in
 * CI); the rest of the suite is written to pass without one.
 */
test("renders the scene with its construction overlay", async ({}, testInfo) => {
  const browser = await chromium.launch({
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,WebGPU",
      "--use-angle=metal",
      "--use-gl=angle",
      "--ignore-gpu-blocklist",
      "--enable-gpu",
    ],
  });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  const adapter = await page.evaluate(async () => {
    if (!navigator.gpu) return null;
    const found = await navigator.gpu.requestAdapter();
    return found ? "available" : null;
  });
  if (!adapter) {
    await browser.close();
    test.skip(true, "No WebGPU adapter in this browser build");
    return;
  }

  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  await page.goto(`http://127.0.0.1:${process.env.JAXCAD_E2E_PORT ?? 8799}/`);
  await expect(page.getByTestId("status")).toContainText("preview", { timeout: 90_000 });

  const shot = await page.getByTestId("viewer-canvas").screenshot();
  await testInfo.attach("viewer", { body: shot, contentType: "image/png" });
  await page.screenshot({ path: testInfo.outputPath("full.png"), fullPage: false });

  expect(errors.join("\n")).not.toContain("WGSL");
  await browser.close();
});
