import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";

/**
 * The public demo, end to end on its static build: load a scene, move a
 * parameter, see the render change without a shader rebuild, export an STL.
 */

type DemoState = {
  scene: string | null;
  values: Record<string, number[]>;
  uploads: number;
  pipelineBuilds: number;
  device: string | null;
};

const demo = (page: Page) => page.evaluate(() => window.__cadjointDemo!() as DemoState);

const hasWebGpu = (page: Page) =>
  page.evaluate(async () => (await navigator.gpu?.requestAdapter()) != null);

async function open(page: Page, scene: string) {
  await page.goto(`/?scene=${scene}`);
  await expect.poll(async () => (await demo(page)).scene).toBe(scene);
  await expect(page.getByTestId("status")).toContainText("free parameters");
}

test("a slider moves the solid through the parameter buffer, not a new shader", async ({ page }, testInfo) => {
  await open(page, "bracket");
  test.skip(!(await hasWebGpu(page)), "No WebGPU adapter in this browser build");
  await expect.poll(async () => (await demo(page)).pipelineBuilds).toBeGreaterThan(0);
  await page.waitForTimeout(1500);
  const canvas = page.getByTestId("viewer-canvas");
  const before = await canvas.screenshot();
  const { pipelineBuilds, uploads } = await demo(page);

  await page.getByTestId("slider-plate_thickness-0").fill("0.36");
  await expect.poll(async () => (await demo(page)).values.plate_thickness[0]).toBeCloseTo(0.36, 6);
  await expect(page.getByTestId("field-plate_thickness-0")).toHaveValue("0.360");
  await page.waitForTimeout(1500);
  const after = await canvas.screenshot();
  await testInfo.attach("before", { body: before, contentType: "image/png" });
  await testInfo.attach("after", { body: after, contentType: "image/png" });

  const state = await demo(page);
  expect(state.pipelineBuilds, "a parameter edit rebuilds no pipeline").toBe(pipelineBuilds);
  expect(state.uploads).toBeGreaterThan(uploads);
  expect(after.equals(before), "the render changed").toBe(false);
  await expect(page.getByTestId("parameters-reset")).toHaveText("Reset · 1");
});

test("a constraint-driven parameter is read-only and says what holds it", async ({ page }) => {
  await open(page, "bracket");
  const row = page.getByTestId("parameter-rib_tip");
  await expect(row.getByText("constraint-driven")).toBeVisible();
  await expect(row).toContainText("DistanceConstraint(rib_web_corner, rib_tip)");
  await expect(page.getByTestId("slider-rib_tip-0")).toBeDisabled();
  await expect(page.getByTestId("field-rib_tip-1")).toBeDisabled();
  await expect(page.getByTestId("slider-plate_bl-0")).toBeEnabled();
});

test("the STL export meshes the edited values in the browser", async ({ page }) => {
  await open(page, "bracket");
  await page.getByTestId("slider-plate_thickness-0").fill("0.3");
  await expect.poll(async () => (await demo(page)).values.plate_thickness[0]).toBeCloseTo(0.3, 6);

  const downloading = page.waitForEvent("download", { timeout: 120_000 });
  await page.getByTestId("export-stl").click();
  const download = await downloading;
  expect(download.suggestedFilename()).toBe("bracket-edited.stl");
  const stl = readFileSync((await download.path())!);
  const triangles = stl.readUInt32LE(80);
  expect(triangles).toBeGreaterThan(1000);
  expect(stl.length).toBe(84 + 50 * triangles);
  await expect(page.getByTestId("status")).toContainText("bracket-edited.stl");
  expect(["webgpu", "wasm"]).toContain((await demo(page)).device);
});
