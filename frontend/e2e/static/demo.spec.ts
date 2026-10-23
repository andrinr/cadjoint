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

test("the scene picker is a closed dropdown that opens, switches the scene and closes", async ({ page }) => {
  await open(page, "bracket");
  const field = page.getByTestId("scene-picker");
  const list = page.getByRole("listbox");
  await expect(field).toHaveAttribute("aria-expanded", "false");
  await expect(list).toHaveCount(0);

  await field.click();
  await expect(field).toHaveAttribute("aria-expanded", "true");
  await expect(list).toBeVisible();
  // Drawn as a menu below the field, not as a bulleted list in the flow.
  const fieldBox = (await field.boundingBox())!;
  const listBox = (await list.boundingBox())!;
  expect(listBox.y).toBeGreaterThanOrEqual(fieldBox.y + fieldBox.height);
  expect(await list.evaluate((el) => [getComputedStyle(el).position, getComputedStyle(el).listStyleType])).toEqual([
    "absolute",
    "none",
  ]);

  await page.getByTestId("scene-picker-option-end_cap").click();
  await expect(list).toHaveCount(0);
  await expect(field).toHaveAttribute("aria-expanded", "false");
  await expect.poll(async () => (await demo(page)).scene).toBe("end_cap");
  await expect(field).toHaveText("end cap");
  await expect(page.getByTestId("status")).toContainText("free parameters");

  // Escape and a click elsewhere both close it without switching.
  await field.click();
  await expect(list).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(list).toHaveCount(0);
  await field.click();
  await page.getByTestId("parameter-panel").click({ position: { x: 10, y: 200 } });
  await expect(list).toHaveCount(0);
  expect((await demo(page)).scene).toBe("end_cap");
});

/** Where a named sketch handle is: sweep pointer moves and read the hint. */
async function findHandle(page: Page, name: string): Promise<{ x: number; y: number; hint: string } | null> {
  return page.evaluate((wanted) => {
    const canvas = document.querySelector("[data-testid=viewer-canvas]")!;
    const r = canvas.getBoundingClientRect();
    for (let y = r.top; y < r.bottom; y += 4)
      for (let x = r.left; x < r.right; x += 4) {
        canvas.dispatchEvent(new PointerEvent("pointermove", { clientX: x, clientY: y, bubbles: true }));
        const hint = document.querySelector("[data-testid=viewer-hint]")?.textContent ?? "";
        if (hint.startsWith(`${wanted} ·`)) return { x, y, hint };
      }
    return null;
  }, name);
}

test("a slider follows the pointer and the render changes mid-drag", async ({ page }) => {
  await open(page, "bracket");
  test.skip(!(await hasWebGpu(page)), "No WebGPU adapter in this browser build");
  await expect.poll(async () => (await demo(page)).pipelineBuilds).toBeGreaterThan(0);
  await page.waitForTimeout(1000);
  const canvas = page.getByTestId("viewer-canvas");
  const before = await canvas.screenshot();
  const slider = page.getByTestId("slider-plate_thickness-0");
  const element = await slider.elementHandle();
  const box = (await slider.boundingBox())!;
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width / 2, y);
  await page.mouse.down();
  const seen: number[] = [];
  for (let i = 1; i <= 12; i++) {
    await page.mouse.move(box.x + box.width / 2 + i * 8, y);
    seen.push((await demo(page)).values.plate_thickness[0]);
  }
  await page.waitForTimeout(500);
  const during = await canvas.screenshot();
  // Still the same slider: the panel did not remount it under the pointer.
  expect(await page.evaluate((el) => el?.isConnected, element)).toBe(true);
  await page.mouse.up();
  expect(new Set(seen).size, "the value moved on every step, not once").toBeGreaterThan(8);
  expect(during.equals(before), "the render changed while the slider was held").toBe(false);
});

test("dragging a free handle moves its parameter while the pointer is down", async ({ page }) => {
  await open(page, "bracket");
  test.skip(!(await hasWebGpu(page)), "No WebGPU adapter in this browser build");
  await expect.poll(async () => (await demo(page)).pipelineBuilds).toBeGreaterThan(0);
  const handle = (await findHandle(page, "plate_tr"))!;
  expect(handle.hint).toContain("free parameter");
  const start = (await demo(page)).values.plate_tr;
  await page.mouse.move(handle.x, handle.y);
  await page.mouse.down();
  const seen: string[] = [];
  for (let i = 1; i <= 10; i++) {
    await page.mouse.move(handle.x + i * 6, handle.y + i * 2);
    seen.push((await demo(page)).values.plate_tr.join(","));
  }
  await page.mouse.up();
  expect(new Set(seen).size, "the parameter followed the drag").toBeGreaterThan(5);
  const end = (await demo(page)).values.plate_tr;
  expect(end).not.toEqual(start);
  await expect(page.getByTestId("field-plate_tr-0")).toHaveValue(end[0].toFixed(2));

  // A constraint-held handle says why it stays.
  const held = (await findHandle(page, "rib_tip"))!;
  expect(held.hint).toContain("held by DistanceConstraint");
});

test("the viewport's controls keep their size, at desktop and narrow widths", async ({ page }) => {
  for (const width of [1440, 900]) {
    await page.setViewportSize({ width, height: 900 });
    await open(page, "bracket");
    for (const bar of ["view-controls", "demo-tool-rail"]) {
      const sizes = await page
        .getByTestId(bar)
        .locator("button")
        .evaluateAll((buttons) => buttons.map((b) => [b.getBoundingClientRect().width, b.getBoundingClientRect().height]));
      expect(sizes.length).toBeGreaterThan(2);
      for (const [w, h] of sizes) expect([w, h], `${bar} at ${width}px`).toEqual([32, 32]);
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth), "no sideways scroll").toBe(width);
  }
});
