import { chromium, expect, test, type Browser, type Page } from "@playwright/test";

/**
 * The docs' screenshots, taken from the running app.
 *
 * `docs/assets/screens/` is "the app as shipped, in screenshots"
 * (research/design-language.md §14). This spec re-takes every one of them so
 * a UI change is reflected in the docs by re-running it, not by hand:
 *
 *   CADJOINT_DOCS_SCREENS=docs/assets/screens npx playwright test e2e/docsScreens.spec.ts
 *
 * Skipped without the variable, so it never runs as a test. Each shot is the
 * playground on the starter at 1400×900 — the size the docs pages assume —
 * with WebGPU on, since a screenshot of the viewport without the solid in it
 * is a screenshot of nothing.
 */

const PORT = process.env.CADJOINT_E2E_PORT ?? 8799;
const SHOTS = process.env.CADJOINT_DOCS_SCREENS ?? "";
/** Comma-separated attempt names to run; empty runs them all. */
const ONLY = (process.env.CADJOINT_DOCS_ONLY ?? "").split(",").filter(Boolean);

async function waitForCompile(page: Page) {
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, { timeout: 120_000 });
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0, { timeout: 120_000 });
}

async function openViewer(): Promise<{ browser: Browser; page: Page } | null> {
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

async function shot(page: Page, name: string, settle = 1200) {
  await page.waitForTimeout(settle);
  await page.screenshot({ path: `${SHOTS}/${name}.png` });
  console.log(`shot ${name}`);
}

/** Open the render popover's customize editor and flip one switch, then close it. */
async function flip(page: Page, toggle: string) {
  await page.getByTestId("display-options").click();
  await page.getByTestId("render-customize").click();
  await page.getByTestId(toggle).click();
  await page.getByTestId("display-options").click();
}

/** One screenshot's steps; a control that is not there within its timeout skips the shot, logged. */
async function attempt(name: string, steps: () => Promise<void>) {
  if (ONLY.length && !ONLY.includes(name)) return;
  try {
    await steps();
  } catch (error) {
    console.log(`skipped ${name}: ${error instanceof Error ? error.message.split("\n")[0] : String(error)}`);
  }
}

/**
 * Park windows in the tray for the length of `steps`, then restore them.
 *
 * At 1400 the editor column takes a third of the width, and a shot of a
 * solved field or a replayed optimization is a shot of the viewport: the
 * editor goes to the tray while it is taken. The dock hands the freed width
 * to the right-hand column, so the seam beside the viewport is dragged back
 * out to `column` pixels from the right edge, and returned afterwards.
 */
async function parked(page: Page, ids: string[], steps: () => Promise<void>, column = 420) {
  const restored: string[] = [];
  for (const id of ids) {
    const group = page.locator(`.dv-groupview:has([data-testid=window-tab-${id}])`).first();
    if (!(await group.isVisible().catch(() => false))) continue;
    await page.getByTestId(`window-tab-${id}`).click();
    await group.getByTestId("window-minimise").click();
    await expect(page.getByTestId(`window-tab-${id}`)).toHaveCount(0);
    restored.push(id);
  }
  await page.waitForTimeout(400);
  const viewport = page.locator(".dv-groupview:has([data-testid=window-tab-viewport])").first();
  const box = (await viewport.boundingBox())!;
  const seam = { x: box.x + box.width, y: box.y + box.height / 2 };
  const drag = async (from: number, to: number) => {
    await page.mouse.move(from, seam.y);
    await page.mouse.down();
    await page.mouse.move(to, seam.y, { steps: 8 });
    await page.mouse.up();
  };
  const target = page.viewportSize()!.width - column;
  const moved = seam.x < target - 4;
  if (moved) await drag(seam.x, target);
  try {
    await steps();
  } finally {
    if (moved) await drag(target, seam.x).catch(() => {});
    for (const id of restored) await page.getByTestId(`window-restore-${id}`).click().catch(() => {});
  }
}

test("the docs screenshots", async () => {
  test.skip(!SHOTS, "set CADJOINT_DOCS_SCREENS to the output directory");
  test.setTimeout(1_800_000);
  const opened = await openViewer();
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page } = opened;
  page.setDefaultTimeout(20_000);

  await attempt("model-desk", async () => {
    await page.getByTestId("editmode-model").click();
    await page.waitForTimeout(500);
    // ── Model desk: the starter as it opens ───────────────────────────────
    await shot(page, "model-desk", 2000);
  });

  await attempt("construction-overlay-off", async () => {
    await page.getByTestId("editmode-model").click();
    await page.waitForTimeout(500);
    // Construction overlay off: solid, field and floor only.
    await flip(page, "toggle-construction-overlay");
    await shot(page, "construction-overlay-off");
    await flip(page, "toggle-construction-overlay");
  });

  await attempt("feature-edges", async () => {
    await page.getByTestId("editmode-model").click();
    await page.waitForTimeout(500);
    // Feature edges, from the lattice or the derived B-rep.
    await flip(page, "toggle-showMeshEdges");
    await page.waitForTimeout(6000);
    await shot(page, "feature-edges");
    await flip(page, "toggle-showMeshEdges");
  });

  await attempt("sdf-slice", async () => {
    await page.getByTestId("editmode-model").click();
    await page.waitForTimeout(500);
    // The distance-field slice view.
    await page.getByTestId("display-options").click();
    const sdf = page.getByTestId("render-sdf");
    await sdf.getByText(/slice/i).first().click();
    await page.getByTestId("display-options").click();
    await shot(page, "sdf-slice");
    await page.getByTestId("display-options").click();
    await sdf.getByText(/solid/i).first().click();
    await page.getByTestId("display-options").click();
  });

  await attempt("sketch-constraints", async () => {
    await page.getByTestId("editmode-sketch").click();
    await page.waitForTimeout(500);
    // Sketch mode with the fin comb selected: its plane, handles and constraints.
    await page.getByTestId("editmode-sketch").click();
    await page.getByTestId("window-tab-objects").click().catch(() => {});
    await page.locator("[data-testid^=tree-row-]").filter({ hasText: /fin comb/i }).first().click();
    await shot(page, "sketch-constraints");
    await page.getByTestId("editmode-model").click();
  });

  /** Open a window from the Window menu and wait for its panel. */
  async function openWindow(page: Page, id: string, panel: string) {
    await page.keyboard.press("Escape");
    await page.getByTestId("menu-window").click({ force: true });
    await page.getByTestId(`menu-window-${id}`).click();
    await page.getByTestId(panel).waitFor({ state: "visible" });
    await page.waitForTimeout(600);
  }

  await attempt("scenes", async () => {
    await page.getByTestId("editmode-model").click();
    await page.waitForTimeout(500);
    await openWindow(page, "scenes", "scenes-panel");
    await shot(page, "scenes");
  });

  await attempt("processes", async () => {
    await page.getByTestId("editmode-model").click();
    await page.waitForTimeout(500);
    await openWindow(page, "processes", "processes-panel");
    await shot(page, "processes");
  });

  await attempt("export-dialog", async () => {
    await page.getByTestId("editmode-model").click();
    await page.waitForTimeout(500);
    await page.keyboard.press("Escape");
    await page.getByTestId("menu-file").click({ force: true });
    await page.getByTestId("menu-file-export").click();
    await expect(page.getByTestId("export-dialog")).toBeVisible();
    await shot(page, "export-dialog");
    await page.keyboard.press("Escape");
  });


  await attempt("simulate-studies", async () => {
    await page.getByTestId("editmode-simulate").click();
    await page.waitForTimeout(500);
    // ── Simulate desk ────────────────────────────────────────────────────
    await page.getByTestId("editmode-simulate").click();
    await page.getByTestId("simulate-study-sink-conduction").waitFor({ timeout: 60_000 });
    await shot(page, "simulate-studies");
  });

  await attempt("simulate-meshes", async () => {
    await page.getByTestId("editmode-simulate").click();
    await page.waitForTimeout(500);
    // The declared mesh inspected: quality heatmap and histogram. Meshes is
    // tabbed behind Studies in the Simulate desk; its tab raises it.
    await page.getByTestId("window-tab-meshes").click();
    await page.getByTestId("mesh-sink-mesh").waitFor({ state: "visible", timeout: 60_000 });
    const inspect = page.getByTestId("mesh-inspect-sink-mesh");
    await expect(inspect).toBeEnabled({ timeout: 120_000 });
    await inspect.click();
    // Inspecting the starter's tet10 mesh is a full extraction; the button
    // reads "Regenerate" once the mesh and its stats are in.
    await expect(inspect).toHaveText(/Regenerate/, { timeout: 900_000 });
    await parked(page, ["editor"], () => shot(page, "simulate-meshes", 2500));
  });

  await attempt("simulate-results", async () => {
    await page.getByTestId("editmode-simulate").click();
    await page.waitForTimeout(500);
    // The study solved, clipped by the slice plane.
    await page.getByTestId("window-tab-studies").click();
    await page.getByTestId("simulate-run-sink-conduction").click();
    await expect(page.getByTestId("simulate-legend")).toContainText("temperature", { timeout: 600_000 });
    await page.waitForTimeout(1500);
    await page.getByTestId("simulate-slice-enabled").check();
    await page.getByTestId("simulate-slice-fraction").fill("0.45").catch(() => {});
    await parked(page, ["editor"], () => shot(page, "simulate-results", 2500));
    await page.getByTestId("simulate-slice-enabled").uncheck();
  });

  await attempt("windows-float-stack", async () => {
    await page.getByTestId("editmode-simulate").click();
    await page.waitForTimeout(500);
    // Windows floated, tabbed and stacked over the Simulate desk: the
    // Studies/Meshes group lifted out of the right-hand column.
    await page.getByTestId("window-tab-studies").click();
    const group = page.locator(".dv-groupview:has([data-testid=window-tab-studies])").first();
    await group.getByTestId("window-float").click();
    await expect(page.locator(".dv-groupview-floating")).toHaveCount(1);
    await shot(page, "windows-float-stack");
    await page.locator(".dv-groupview-floating").getByTestId("window-float").click();
    await expect(page.locator(".dv-groupview-floating")).toHaveCount(0);
  });

  await attempt("optimize-replay", async () => {
    await page.getByTestId("editmode-model").click();
    await page.waitForTimeout(500);
    // ── The optimization, run to its end and replayed ────────────────────
    await page.getByTestId("window-tab-optimize").click({ force: true });
    await page.getByTestId("optimize-run-cool-sink").waitFor({ state: "visible" });
    for (const [field, value] of [
      ["optimize-steps-cool-sink", "8"],
      ["optimize-lr-cool-sink", "0.02"],
    ] as const) {
      const input = page.getByTestId(field).first();
      await input.fill(value);
      await input.press("Enter");
      await waitForCompile(page);
    }
    await page.getByTestId("optimize-run-cool-sink").click();
    await page.getByTestId("optimize-result-cool-sink").waitFor({ timeout: 1_800_000 });
    // The end of a run replays its last design: wait for that compile too.
    await waitForCompile(page);
    // Objects and Properties go to the tray too, so the Optimize card has
    // the column's height and its trajectory player is in view.
    await parked(page, ["editor", "objects", "properties"], async () => {
      await page.getByTestId("optimize-result-cool-sink").scrollIntoViewIfNeeded();
      await shot(page, "optimize-replay", 3000);
    });
  });

  await browser.close();
});
