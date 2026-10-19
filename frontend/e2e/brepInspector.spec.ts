import { mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium, expect, test, type Browser, type Page } from "@playwright/test";

/**
 * The B-rep inspector: the private tier's derived B-rep in place of the solid.
 *
 * Turns the inspector on from the render popover, waits for `/api/brep`, and
 * checks what the card says against what the server sent: the counts, the
 * path, the legend's kinds (a legend switch takes a kind out of the drawing),
 * and the readout of a face under the pointer. Without the private tier the
 * switch must end up off and say why — the inspector never errors for a
 * missing install.
 *
 * With `CADJOINT_BREP_SCREENS=<dir>` it also photographs the inspector on the
 * shipped scenes (plus `CADJOINT_BREP_EXTRA`, comma-separated scene paths),
 * the editor parked so the viewport has the width:
 *
 *   CADJOINT_BREP_SCREENS=/tmp/brep npx playwright test e2e/brepInspector.spec.ts
 *
 * WebGPU is needed for the pictures and for the card's readout of a face
 * (picking runs on the CPU, but only against a view the renderer set up).
 */

const PORT = process.env.CADJOINT_E2E_PORT ?? 8799;
const SCENES = resolve(import.meta.dirname, "..", "..", "scenes");
const SHOTS = process.env.CADJOINT_BREP_SCREENS ?? "";
const EXTRA = (process.env.CADJOINT_BREP_EXTRA ?? "").split(",").filter(Boolean);

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

async function openViewer(width = 1400, height = 900): Promise<{ browser: Browser; page: Page } | null> {
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
  const page = await browser.newPage({ viewport: { width, height } });
  await page.goto(`http://127.0.0.1:${PORT}/`);
  const adapter = await page.evaluate(async () =>
    navigator.gpu && (await navigator.gpu.requestAdapter()) ? "available" : null,
  );
  if (!adapter) {
    await browser.close();
    return null;
  }
  await waitForCompile(page);
  const dismiss = page.getByRole("button", { name: "Dismiss" });
  if (await dismiss.isVisible().catch(() => false)) await dismiss.click();
  return { browser, page };
}

/** Flip a render-popover switch and close the popover. */
async function flip(page: Page, toggle: string) {
  await page.getByTestId("display-options").click();
  await page.getByTestId("render-customize").click();
  await page.getByTestId(toggle).click();
  await page.getByTestId("display-options").click();
}

/** Wait for the inspector's request to settle; returns the status word. */
async function settled(page: Page): Promise<string> {
  const status = page.getByTestId("brep-status");
  await expect(status).not.toHaveText(/Deriving|Waiting/, { timeout: 200_000 });
  return (await status.textContent()) ?? "";
}

/** Sweep the canvas until the card reads out a face; null if none is found. */
async function hoverFace(page: Page): Promise<string | null> {
  const box = (await page.getByTestId("viewer-canvas").boundingBox())!;
  for (let fy = 0.35; fy <= 0.75; fy += 0.05) {
    for (let fx = 0.35; fx <= 0.75; fx += 0.05) {
      await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
      await page.waitForTimeout(40);
      const readout = page.getByTestId("brep-readout");
      if ((await readout.count()) && /·/.test((await readout.textContent()) ?? "")) {
        const row = await page.locator(".brep-card dt", { hasText: /^Face$/i }).count();
        if (row) return readout.textContent();
      }
    }
  }
  return null;
}

test("the inspector draws the derived B-rep and reads it out", async () => {
  test.setTimeout(400_000);
  const opened = await openViewer();
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page } = opened;
  try {
    await flip(page, "toggle-showBrep");
    await expect(page.getByTestId("brep-inspector")).toBeVisible();
    const status = await settled(page);
    if (/Not installed/.test(status)) {
      // Public tier only: the mode says so and stays off.
      await expect(page.getByTestId("brep-message")).toContainText(/diff-brep/i);
      await page.getByTestId("display-options").click();
      await page.getByTestId("render-customize").click();
      await expect(page.getByTestId("toggle-showBrep")).not.toBeChecked();
      await expect(page.getByTestId("toggle-showBrep")).toBeDisabled();
      return;
    }
    expect(status).toBe("Derived");

    // The card's counts.
    const faces = Number((await page.getByTestId("brep-faces").textContent())!.split(" ")[0]);
    const edges = Number(await page.getByTestId("brep-edges").textContent());
    const vertices = Number(await page.getByTestId("brep-vertices").textContent());
    expect(faces).toBeGreaterThan(10);
    expect(edges).toBeGreaterThan(10);
    expect(vertices).toBeGreaterThan(0);
    await expect(page.getByTestId("brep-path")).toContainText(/Exact|Fallback/);

    // Every legend row is a kind the server named, with a count; the counts
    // add up to the edge total.
    const counts = await page
      .locator(".brep-legend-row:not(.brep-off-surface) small")
      .allTextContents();
    expect(counts.reduce((sum, text) => sum + Number(text), 0)).toBe(edges);
    await expect(page.getByTestId("brep-kind-traced")).toBeChecked();
    await page.getByTestId("brep-kind-traced").click();
    await expect(page.getByTestId("brep-kind-traced")).not.toBeChecked();
    await page.getByTestId("brep-kind-traced").click();

    // The off-surface switch is a view, not a filter: it changes nothing in
    // the card's counts.
    await page.getByTestId("brep-off-surface").click();
    await expect(page.getByTestId("brep-off-surface")).toBeChecked();
    await page.getByTestId("brep-off-surface").click();

    // A face under the pointer is named with its surface type and patch.
    const readout = await hoverFace(page);
    expect(readout, "no face found under a sweep of the viewport centre").not.toBeNull();
    expect(readout!).toMatch(/#\d+ · (plane|cylinder|sphere|cone|torus|blend|faceted)/);

    // Off again: the card goes, and the solid is back.
    await flip(page, "toggle-showBrep");
    await expect(page.getByTestId("brep-inspector")).toHaveCount(0);
  } finally {
    await browser.close();
  }
});

/** Bring the part up to fill the frame: a few wheel steps over the canvas centre. */
async function frame(page: Page) {
  const box = (await page.getByTestId("viewer-canvas").boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  for (let step = 0; step < 2; step++) {
    await page.mouse.wheel(0, -240);
    await page.waitForTimeout(120);
  }
}

/** Park the editor so the viewport takes the width, for the pictures. */
async function parkEditor(page: Page) {
  const group = page.locator(".dv-groupview:has([data-testid=window-tab-editor])").first();
  if (!(await group.isVisible().catch(() => false))) return;
  await page.getByTestId("window-tab-editor").click();
  await group.getByTestId("window-minimise").click();
  await page.waitForTimeout(400);
}

test("the inspector on the shipped scenes, photographed", async () => {
  test.skip(!SHOTS, "set CADJOINT_BREP_SCREENS to the output directory");
  test.setTimeout(1_800_000);
  mkdirSync(SHOTS, { recursive: true });
  const opened = await openViewer(1600, 1000);
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page } = opened;
  page.setDefaultTimeout(30_000);
  const scenes = [
    ...["starter", "end_cap", "cylinder_block"].map((name) => resolve(SCENES, `${name}.py`)),
    ...EXTRA.map((path) => resolve(path)),
  ];
  try {
    await flip(page, "toggle-construction-overlay");
    let first = true;
    for (const path of scenes) {
      const name = path.split("/").pop()!.replace(/\.py$/, "");
      await page.getByTestId("window-restore-editor").click().catch(() => {});
      await recompile(page, readFileSync(path, "utf-8"));
      await parkEditor(page);
      if (first) {
        // Once: the camera carries over from scene to scene.
        await flip(page, "toggle-showBrep");
        await frame(page);
        first = false;
      }
      const status = await settled(page);
      console.log(`${name}: ${status} — ${await page.getByTestId("brep-inspector").innerText()}`);
      await page.mouse.move(5, 5);
      await page.waitForTimeout(1500);
      const pane = page.locator(".viewer-pane");
      await pane.screenshot({ path: `${SHOTS}/${name}.png` });
      if (status === "Derived" && (await hoverFace(page))) {
        await page.waitForTimeout(300);
        await pane.screenshot({ path: `${SHOTS}/${name}-hover.png` });
        await page.mouse.move(5, 5);
      }
      // The solid at the same view, for comparison; back on is instant (the
      // answer is kept for the program it describes).
      await flip(page, "toggle-showBrep");
      await page.waitForTimeout(1500);
      await pane.screenshot({ path: `${SHOTS}/${name}-solid.png` });
      await flip(page, "toggle-showBrep");
      await expect(page.getByTestId("brep-status")).toHaveText(status);
      await page.waitForTimeout(800);
      if (status === "Derived" && Number(await page.getByTestId("brep-off-surface-count").textContent()) > 0) {
        await page.getByTestId("brep-off-surface").click();
        await page.waitForTimeout(800);
        await pane.screenshot({ path: `${SHOTS}/${name}-off-surface.png` });
        await page.getByTestId("brep-off-surface").click();
      }
    }
  } finally {
    await browser.close();
  }
});
