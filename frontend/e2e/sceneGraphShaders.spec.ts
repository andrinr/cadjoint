import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { openViewer, recompile, SCENES, stats } from "./helpers/app";

/**
 * Shaders built in the browser from the compile payload's scene graph
 * (`?shaders=browser`) draw what the server's shaders draw.
 *
 * Each scene is drawn once per shader source, and the two canvases must be
 * the same image. The browser's shaders must be the ones installed (on a
 * hash mismatch the page falls back to the server's and warns), and a
 * free-parameter edit on them must stay a buffer upload.
 */

const COMPILE_TIMEOUT = 120_000;

const origin = (page: Page) => page.evaluate(() => document.documentElement.dataset.shaderOrigin);

async function draw(query: string, source: string) {
  const opened = await openViewer({ width: 1200, height: 800, timeout: COMPILE_TIMEOUT, query });
  if (!opened) return null;
  const warnings: string[] = [];
  opened.page.on("console", (message) => {
    if (message.type() === "warning") warnings.push(message.text());
  });
  await recompile(opened.page, source, COMPILE_TIMEOUT);
  await opened.page.waitForTimeout(1500);
  const image = await opened.page.getByTestId("viewer-canvas").screenshot();
  const result = { image, origin: await origin(opened.page), warnings };
  await opened.browser.close();
  return result;
}

for (const scene of ["duct_sink", "bracket", "end_cap"]) {
  test(`${scene}: the browser-built shaders draw the server's image`, async ({}, testInfo) => {
    test.setTimeout(300_000);
    const source = readFileSync(resolve(SCENES, `${scene}.py`), "utf8");
    const server = await draw("", source);
    test.skip(!server, "No WebGPU adapter in this browser build");
    const browser = (await draw("?shaders=browser", source))!;

    expect(server!.origin).toBe("server");
    expect(browser.origin).toBe("browser");
    expect(browser.warnings.filter((w) => w.includes("browser-built shaders differ"))).toEqual([]);
    await testInfo.attach("server", { body: server!.image, contentType: "image/png" });
    await testInfo.attach("browser", { body: browser.image, contentType: "image/png" });
    expect(browser.image.equals(server!.image), "the two shader sources draw the same image").toBe(true);
  });
}

test("a parameter edit on browser-built shaders uploads a buffer and rebuilds no pipelines", async () => {
  test.setTimeout(180_000);
  const opened = await openViewer({ width: 1200, height: 800, timeout: COMPILE_TIMEOUT, query: "?shaders=browser" });
  test.skip(!opened, "No WebGPU adapter in this browser build");
  const { browser, page } = opened!;
  const base = [
    "from cadjoint.sdf import Sphere, Box, Union",
    "from cadjoint.geometry.parameters import Scalar",
    "",
    "radius = Scalar(0.60, free=True, name='radius')",
    "scene = Union((Sphere(radius=radius), Box(size=[0.9, 0.5, 0.2])), smoothness=0.05)",
    "",
  ].join("\n");
  await recompile(page, base, COMPILE_TIMEOUT);
  expect(await origin(page)).toBe("browser");
  const first = (await stats(page))!;
  expect(first.hasParameterBuffer).toBe(true);

  await recompile(page, base.replace("Scalar(0.60,", "Scalar(0.42,"), COMPILE_TIMEOUT);
  const moved = (await stats(page))!;
  expect(moved.pipelineBuilds, "a parameter edit builds no pipeline").toBe(first.pipelineBuilds);
  expect(moved.parameterUploads).toBeGreaterThan(first.parameterUploads);
  await browser.close();
});
