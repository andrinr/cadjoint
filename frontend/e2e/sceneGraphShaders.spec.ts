import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium, expect, test, type Page } from "@playwright/test";

/**
 * Shaders built in the browser from the compile payload's scene graph
 * (`?shaders=browser`) draw what the server's shaders draw.
 *
 * Each shipped scene is drawn twice, once per shader source, and the two
 * canvases must be the same image. The browser source must also be the one
 * actually installed (it falls back to the server's, with a warning, when
 * its hash differs), and a free-parameter edit must stay a buffer upload.
 */

const PORT = process.env.CADJOINT_E2E_PORT ?? 8799;
const SCENES = ["duct_sink", "bracket", "end_cap"];

const launch = () =>
  chromium.launch({
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,WebGPU",
      "--use-angle=metal",
      "--use-gl=angle",
      "--ignore-gpu-blocklist",
      "--enable-gpu",
    ],
  });

async function waitForCompile(page: Page) {
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, { timeout: 120_000 });
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0, { timeout: 120_000 });
}

async function recompile(page: Page, source: string) {
  await page.evaluate((text) => {
    type EditorLike = {
      view?: { state: { doc: { length: number } }; dispatch: (spec: unknown) => void };
    };
    const content = document.querySelector("[data-testid=editor] .cm-content") as
      | (HTMLElement & { cmView?: EditorLike; cmTile?: EditorLike })
      | null;
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    if (!view) throw new Error("no CodeMirror view");
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
  }, source);
  await page.getByTestId("run").click();
  await waitForCompile(page);
}

/** Open the page with one shader source, draw `source`, and return the canvas image. */
async function draw(page: Page, query: string, source: string, warnings: string[]): Promise<Buffer | null> {
  await page.goto(`http://127.0.0.1:${PORT}/${query}`);
  const adapter = await page.evaluate(async () =>
    navigator.gpu && (await navigator.gpu.requestAdapter()) ? "available" : null,
  );
  if (!adapter) return null;
  await waitForCompile(page);
  page.on("console", (message) => {
    if (message.type() === "warning") warnings.push(message.text());
  });
  await recompile(page, source);
  await page.waitForTimeout(1500);
  return page.getByTestId("viewer-canvas").screenshot();
}

for (const scene of SCENES) {
  test(`${scene}: the browser-built shaders draw the server's image`, async ({}, testInfo) => {
    test.setTimeout(300_000);
    const source = readFileSync(resolve(import.meta.dirname, "..", "..", "scenes", `${scene}.py`), "utf8");
    const browser = await launch();
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
    const warnings: string[] = [];

    const server = await draw(page, "", source, warnings);
    if (!server) {
      await browser.close();
      test.skip(true, "No WebGPU adapter in this browser build");
      return;
    }
    expect(await page.evaluate(() => document.documentElement.dataset.shaderOrigin)).toBe("server");

    const built = await draw(page, "?shaders=browser", source, warnings);
    expect(await page.evaluate(() => document.documentElement.dataset.shaderOrigin)).toBe("browser");
    expect(warnings.filter((w) => w.includes("browser-built shaders differ"))).toEqual([]);

    await testInfo.attach("server", { body: server, contentType: "image/png" });
    await testInfo.attach("browser", { body: built!, contentType: "image/png" });
    expect(built!.equals(server), "the two shader sources draw the same image").toBe(true);
    await browser.close();
  });
}

test("a parameter edit on browser-built shaders uploads a buffer and rebuilds no pipelines", async () => {
  test.setTimeout(180_000);
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  await page.goto(`http://127.0.0.1:${PORT}/?shaders=browser`);
  const adapter = await page.evaluate(async () =>
    navigator.gpu && (await navigator.gpu.requestAdapter()) ? "available" : null,
  );
  if (!adapter) {
    await browser.close();
    test.skip(true, "No WebGPU adapter in this browser build");
    return;
  }
  await waitForCompile(page);
  const base = [
    "from cadjoint.sdf import Sphere, Box, Union",
    "from cadjoint.geometry.parameters import Scalar",
    "",
    "radius = Scalar(0.60, free=True, name='radius')",
    "scene = Union((Sphere(radius=radius), Box(size=[0.9, 0.5, 0.2])), smoothness=0.05)",
    "",
  ].join("\n");
  await recompile(page, base);
  expect(await page.evaluate(() => document.documentElement.dataset.shaderOrigin)).toBe("browser");
  const first = await page.evaluate(() => window.__cadjointShaders?.() ?? null);
  expect(first!.hasParameterBuffer).toBe(true);

  await recompile(page, base.replace("Scalar(0.60,", "Scalar(0.42,"));
  const moved = await page.evaluate(() => window.__cadjointShaders?.() ?? null);
  expect(moved!.pipelineBuilds, "a parameter edit builds no pipeline").toBe(first!.pipelineBuilds);
  expect(moved!.parameterUploads).toBeGreaterThan(first!.parameterUploads);
  await browser.close();
});
