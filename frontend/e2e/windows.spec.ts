import { expect, test, type Locator, type Page } from "@playwright/test";

/**
 * The window system: move, tab, park, close, float, remember.
 *
 * These tests drive the dock the way a user does — by dragging tabs and
 * clicking chrome — rather than through the manager API, except where a
 * control does not exist yet (floating has no menu item). The one thing they
 * check that a user cannot see is the WebGPU canvas: a docking system that
 * re-parents it loses the GPU context, and the only way to know that has not
 * happened is to watch the element itself.
 */

async function waitForDock(page: Page) {
  await expect(page.getByTestId("window-tab-viewport")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("run")).toBeEnabled({ timeout: 60_000 });
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await waitForDock(page);
  const dismiss = page.getByRole("button", { name: "Dismiss" });
  if (await dismiss.isVisible().catch(() => false)) await dismiss.click();
});

/** Drag a tab onto a point, the way dockview's pointer strategy expects. */
async function dragTabTo(page: Page, tab: Locator, to: { x: number; y: number }) {
  const from = (await tab.boundingBox())!;
  const start = { x: from.x + from.width / 2, y: from.y + from.height / 2 };
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  for (let step = 1; step <= 12; step++) {
    await page.mouse.move(
      start.x + (to.x - start.x) * (step / 12),
      start.y + (to.y - start.y) * (step / 12),
      { steps: 2 },
    );
  }
  // The drop overlay animates in; releasing on the same frame as the last
  // move sometimes lands before the target has resolved.
  await page.waitForTimeout(300);
  await page.mouse.up();
}

const groupCount = (page: Page) => page.locator(".dv-groupview").count();

/** The editor's whole document, read from CodeMirror rather than the DOM. */
async function editorText(page: Page): Promise<string> {
  return page.evaluate(() => {
    type DocView = { view?: { state: { doc: { toString(): string } } } };
    const content = document.querySelector("[data-testid=editor] .cm-content") as
      | (HTMLElement & { cmView?: DocView; cmTile?: DocView })
      | null;
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    if (view) return view.state.doc.toString();
    return content?.innerText ?? "";
  });
}

test("the model desk opens docked, tabbed, and with a viewport that cannot be closed", async ({
  page,
}) => {
  for (const id of ["viewport", "editor", "objects", "materials", "optimize"]) {
    await expect(page.getByTestId(`window-tab-${id}`)).toBeVisible();
  }
  for (const id of ["studies", "meshes", "results", "sketch"]) {
    await expect(page.getByTestId(`window-tab-${id}`)).toHaveCount(0);
  }

  // Materials and Optimize share one tab strip out of the box.
  const stacked = page.locator(".dv-groupview:has([data-testid=window-tab-optimize]) .dv-tab");
  await expect(stacked).toHaveCount(2);

  // The viewport is furniture: no close control, and no minimise either.
  await expect(page.getByTestId("window-close-viewport")).toHaveCount(0);
  await expect(page.getByTestId("window-close-objects")).toHaveCount(1);
  const viewportGroup = page.locator(".dv-groupview:has([data-testid=window-tab-viewport])");
  await expect(viewportGroup.getByTestId("window-minimise")).toBeHidden();
});

test("a window can be dragged into a new split", async ({ page }) => {
  const viewport = (await page.locator("[data-window=viewport]").boundingBox())!;
  const objectsBefore = (await page.locator("[data-window=objects]").boundingBox())!;
  // It starts in the right-hand column, beside the viewport.
  expect(objectsBefore.x).toBeGreaterThan(viewport.x + viewport.width - 1);

  await dragTabTo(page, page.getByTestId("window-tab-objects"), {
    x: viewport.x + viewport.width / 2,
    y: viewport.y + viewport.height * 0.9,
  });

  // It landed below the viewport, which is what the drop target promised, and
  // the viewport gave up the height for it.
  await expect
    .poll(async () => {
      const objects = await page.locator("[data-window=objects]").boundingBox();
      const shrunk = await page.locator("[data-window=viewport]").boundingBox();
      return objects && shrunk ? objects.y > shrunk.y + shrunk.height / 2 : false;
    })
    .toBe(true);
  const shrunk = (await page.locator("[data-window=viewport]").boundingBox())!;
  expect(shrunk.height).toBeLessThan(viewport.height);
});

test("a window can be dragged onto another's tab strip to stack them", async ({ page }) => {
  const strip = (await page.getByTestId("window-tab-materials").boundingBox())!;
  await dragTabTo(page, page.getByTestId("window-tab-objects"), {
    x: strip.x + strip.width + 40,
    y: strip.y + strip.height / 2,
  });

  const stacked = page.locator(".dv-groupview:has([data-testid=window-tab-objects]) .dv-tab");
  await expect(stacked).toHaveCount(3);
  // A background tab keeps its Solid tree alive: switching back finds the
  // panel already populated, with no re-mount flash.
  await page.getByTestId("window-tab-materials").click();
  await expect(page.getByTestId("material-aluminum")).toBeVisible();
  await page.getByTestId("window-tab-objects").click();
  await expect(page.getByTestId("object-tree-panel")).toBeVisible();
});

test("a window minimises to the tray and comes back", async ({ page }) => {
  // The tray is never empty now — Processes is parked in every desk — so
  // what is asserted is which windows are in it, not whether it exists.
  await expect(page.getByTestId("window-restore-objects")).toHaveCount(0);

  const group = page.locator(".dv-groupview:has([data-testid=window-tab-objects])");
  await group.getByTestId("window-minimise").click();

  await expect(page.getByTestId("window-tab-objects")).toHaveCount(0);
  await expect(page.getByTestId("window-tray")).toBeVisible();
  await expect(page.getByTestId("window-restore-objects")).toHaveText("Objects");

  await page.getByTestId("window-restore-objects").click();
  await expect(page.getByTestId("window-tab-objects")).toBeVisible();
  await expect(page.getByTestId("window-restore-objects")).toHaveCount(0);
});

test("a window closes from its tab and the Window menu brings it back", async ({ page }) => {
  await page.getByTestId("window-close-materials").click();
  await expect(page.getByTestId("window-tab-materials")).toHaveCount(0);
  await expect(page.getByTestId("material-panel")).toHaveCount(0);

  // The menu bar's Window items route through the same manager the dock owns.
  await page.getByTestId("menu-window").click();
  await page.getByTestId("menu-window-materials").click();
  await expect(page.getByTestId("window-tab-materials")).toBeVisible();
  await expect(page.getByTestId("material-panel")).toBeVisible();
});

test("a group floats out of the grid and docks back", async ({ page }) => {
  const group = page.locator(".dv-groupview:has([data-testid=window-tab-objects])");
  await group.getByTestId("window-float").click();

  await expect(page.locator(".dv-groupview-floating")).toHaveCount(1);
  await expect(page.getByTestId("object-tree-panel")).toBeVisible();

  await page.locator(".dv-groupview-floating").getByTestId("window-float").click();
  await expect(page.locator(".dv-groupview-floating")).toHaveCount(0);
  await expect(page.getByTestId("object-tree-panel")).toBeVisible();
});

test("each mode restores its own desk", async ({ page }) => {
  await expect(page.getByTestId("window-tab-materials")).toBeVisible();

  await page.getByTestId("editmode-simulate").click();
  // Simulate is a desk, not a window: it arranges four ordinary windows.
  for (const id of ["studies", "meshes", "optimize", "results"]) {
    await expect(page.getByTestId(`window-tab-${id}`)).toBeVisible();
  }
  await expect(page.getByTestId("mode-simulate")).toBeVisible();
  await expect(page.getByTestId("window-tab-materials")).toHaveCount(0);

  // A change made in Simulate belongs to Simulate.
  await page.getByTestId("window-close-editor").click();
  await expect(page.getByTestId("window-tab-editor")).toHaveCount(0);

  await page.getByTestId("editmode-model").click();
  await expect(page.getByTestId("window-tab-materials")).toBeVisible();
  await expect(page.getByTestId("window-tab-editor")).toBeVisible();
  await expect(page.getByTestId("mode-simulate")).toHaveCount(0);

  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("window-tab-editor")).toHaveCount(0);
});

test("the arrangement survives a reload, and Reset Layout undoes it", async ({ page }) => {
  const strip = (await page.getByTestId("window-tab-materials").boundingBox())!;
  await dragTabTo(page, page.getByTestId("window-tab-objects"), {
    x: strip.x + strip.width + 40,
    y: strip.y + strip.height / 2,
  });
  await expect(
    page.locator(".dv-groupview:has([data-testid=window-tab-objects]) .dv-tab"),
  ).toHaveCount(3);
  const groups = await groupCount(page);

  // The record is written on a settle timer; give it one.
  await page.waitForTimeout(600);
  await page.reload();
  await waitForDock(page);

  await expect(
    page.locator(".dv-groupview:has([data-testid=window-tab-objects]) .dv-tab"),
  ).toHaveCount(3);
  expect(await groupCount(page)).toBe(groups);

  await page.evaluate(() => window.__cadjointWindows!.resetLayout());
  await expect(
    page.locator(".dv-groupview:has([data-testid=window-tab-objects]) .dv-tab"),
  ).toHaveCount(1);
});

test("a parked window is still parked after a reload", async ({ page }) => {
  await page
    .locator(".dv-groupview:has([data-testid=window-tab-objects])")
    .getByTestId("window-minimise")
    .click();
  await expect(page.getByTestId("window-restore-objects")).toBeVisible();

  await page.waitForTimeout(600);
  await page.reload();
  await waitForDock(page);

  await expect(page.getByTestId("window-restore-objects")).toBeVisible();
  await expect(page.getByTestId("window-tab-objects")).toHaveCount(0);
  await page.evaluate(() => window.__cadjointWindows!.resetLayout());
  await expect(page.getByTestId("window-restore-objects")).toHaveCount(0);
  // Reset restores the default desk, which parks Processes again.
  await expect(page.getByTestId("window-restore-processes")).toBeVisible();
});

test("the viewer canvas survives everything the dock does to it", async ({ page }) => {
  // The classic docking failure: a canvas that gets destroyed and rebuilt as
  // it is dragged around loses its GPU context, and the viewport goes black.
  //
  // What is asserted: the canvas element is created once and never replaced,
  // never leaves the document, and keeps the same pane and panel body around
  // it. Laying out a new desk does move that panel body between the library's
  // own wrapper divs — the element survives, and `Renderer.reconfigure()`
  // re-attaches the swap chain afterwards — so the wrapper above the panel
  // body is the one ancestor allowed to change.
  await page.evaluate(() => {
    const record = { detached: 0, recreated: 0, reparented: 0, chain: "" };
    (window as unknown as { __canvasWatch: typeof record }).__canvasWatch = record;
    let node: Element | null = null;
    // Identity, not class names: dockview toggles classes on the ancestors
    // (active group, floating) all the time, and that is not a re-parent.
    let chain: Element[] = [];
    /** The canvas up to and including its `.win-body` panel container. */
    const chainOf = (element: Element) => {
      const parts: Element[] = [];
      for (let cursor: Element | null = element; cursor; cursor = cursor.parentElement) {
        parts.push(cursor);
        if (cursor.classList.contains("win-body")) break;
      }
      return parts;
    };
    const tick = () => {
      const canvas = document.querySelector("[data-testid=viewer-canvas]");
      if (!canvas) record.detached += 1;
      else {
        if (node && node !== canvas) record.recreated += 1;
        node = canvas;
        if (!document.contains(canvas)) record.detached += 1;
        const next = chainOf(canvas);
        if (
          chain.length > 0 &&
          (next.length !== chain.length || next.some((element, i) => element !== chain[i]))
        ) {
          record.reparented += 1;
        }
        chain = next;
        record.chain = next.map((element) => element.className || element.tagName).join(" < ");
      }
      requestAnimationFrame(tick);
    };
    tick();
  });

  const viewport = (await page.locator("[data-window=viewport]").boundingBox())!;
  const sizeBefore = await page
    .getByTestId("viewer-canvas")
    .evaluate((node: HTMLCanvasElement) => node.width);

  // Split, stack, park, float, and change modes twice.
  await dragTabTo(page, page.getByTestId("window-tab-objects"), {
    x: viewport.x + viewport.width / 2,
    y: viewport.y + viewport.height * 0.9,
  });
  const strip = (await page.getByTestId("window-tab-materials").boundingBox())!;
  await dragTabTo(page, page.getByTestId("window-tab-objects"), {
    x: strip.x + strip.width + 40,
    y: strip.y + strip.height / 2,
  });
  await page
    .locator(".dv-groupview:has([data-testid=window-tab-viewport])")
    .getByTestId("window-float")
    .click();
  await page.waitForTimeout(400);
  await page
    .locator(".dv-groupview-floating")
    .getByTestId("window-float")
    .click();
  await page.getByTestId("editmode-simulate").click();
  await page.waitForTimeout(600);
  await page.getByTestId("editmode-model").click();
  await page.waitForTimeout(600);

  const watch = await page.evaluate(
    () => (window as unknown as { __canvasWatch: Record<string, number | string> }).__canvasWatch,
  );
  expect(watch.detached).toBe(0);
  expect(watch.recreated).toBe(0);
  expect(watch.reparented).toBe(0);
  expect(watch.chain).toContain("viewer-canvas");

  // The renderer was still told about every size change along the way.
  const sizeAfter = await page
    .getByTestId("viewer-canvas")
    .evaluate((node: HTMLCanvasElement) => node.width);
  expect(sizeAfter).toBeGreaterThan(0);
  expect(sizeAfter).not.toBe(sizeBefore);
  await expect(page.getByTestId("viewer-canvas")).toBeVisible();
});

test("the Window menu floats a window, docks it again, and resets the layout", async ({
  page,
}) => {
  const float = () => page.getByTestId("menu-window-float-objects");
  await page.getByTestId("menu-window").click();
  await expect(float()).toHaveAttribute("aria-checked", "false");
  await float().click();
  await expect(page.locator(".dv-groupview-floating")).toHaveCount(1);

  // The same item is the way back: it is a checkbox, not a one-way door.
  await page.getByTestId("menu-window").click();
  await expect(float()).toHaveAttribute("aria-checked", "true");
  await float().click();
  await expect(page.locator(".dv-groupview-floating")).toHaveCount(0);

  // Reset Layout throws the mode's arrangement away and rebuilds its default.
  await page.getByTestId("window-close-editor").click();
  await expect(page.getByTestId("window-tab-editor")).toHaveCount(0);
  await page.getByTestId("menu-window").click();
  await page.getByTestId("menu-window-reset").click();
  await expect(page.getByTestId("window-tab-editor")).toBeVisible();
});

/**
 * The process monitor, and the persistence it exists to make possible.
 *
 * Before the job registry a result lived only in the panel that asked for
 * it: leaving Simulate mode disposed that panel's Solid root and a nine-
 * second solve went with it. These tests are the proof that it no longer
 * does — and that the proof is not a cache trick, because the number of
 * `/api/simulate` requests is counted, not assumed.
 */

/** Count POSTs to one endpoint for the life of the page. */
function countPosts(page: Page, path: string): () => number {
  let seen = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().includes(path)) seen += 1;
  });
  return () => seen;
}

test("the process monitor is parked in every desk and reports the server", async ({
  page,
}) => {
  // Parked, not docked: a monitor that took a column of the desk before it
  // was asked for would be in the way of the work it is monitoring.
  await expect(page.getByTestId("window-restore-processes")).toHaveText("Processes");
  await expect(page.getByTestId("window-tab-processes")).toHaveCount(0);

  await page.getByTestId("window-restore-processes").click();
  const panel = page.getByTestId("processes-panel");
  await expect(panel).toBeVisible();

  // Three numbered zones, in the order the sheet reads them.
  await expect(page.getByTestId("processes-running")).toContainText("Running");
  await expect(page.getByTestId("processes-history")).toContainText("History");
  await expect(page.getByTestId("processes-load")).toContainText("Load");

  // Loading the playground compiles the starter, so the history is not a
  // hypothetical: it holds that compile, with what it cost.
  const compiles = page.locator('[data-testid^="processes-open-"]', { hasText: "compile" });
  await expect(compiles.first()).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("processes-totals")).toContainText("uptime");
  await expect(page.getByTestId("processes-budget")).toContainText("/50 jobs");

  // Expanding a row is what fetches its samples; nothing else does.
  const first = compiles.first();
  const jobId = await first.getAttribute("data-testid");
  const id = jobId!.replace("processes-open-", "");
  await page.getByTestId(`processes-expand-${id}`).click();
  await expect(page.getByTestId(`processes-detail-${id}`)).toContainText("pid");

  // The Window menu closes it and opens it again — the tray is not the only
  // way in, because a user who closed it needs a way back.
  await page.getByTestId("menu-window").click();
  await page.getByTestId("menu-window-processes").click();
  await expect(panel).toHaveCount(0);
  // A visibility checkbox leaves the menu open, like the panel items above
  // it, so the way back is the same item rather than a second trip.
  await expect(page.getByTestId("menu-window-processes")).toHaveAttribute(
    "aria-checked",
    "false",
  );
  await page.getByTestId("menu-window-processes").click();
  await expect(panel).toBeVisible();
  await page.keyboard.press("Escape");

  // And it is parked in the Simulate desk too, not only in Model's.
  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("simulate-panel")).toBeVisible();
  await expect(page.getByTestId("window-restore-processes")).toBeVisible();
});

test("a solved field survives a mode switch and is fetched back by job id", async ({
  page,
}) => {
  // A cold cache meshes and solves a design the server has never seen.
  test.setTimeout(300_000);
  const solves = countPosts(page, "/api/simulate");

  await page.getByTestId("editmode-simulate").click();
  await page.getByTestId("simulate-run-sink-conduction").click();
  await expect(page.getByTestId("simulate-legend")).toBeVisible({ timeout: 240_000 });
  await expect(page.getByTestId("simulate-legend")).toContainText("temperature");
  expect(solves()).toBe(1);

  // Leave Simulate. The panel's Solid root is disposed with the desk — this
  // is exactly the moment the result used to be lost.
  await page.getByTestId("editmode-model").click();
  await expect(page.getByTestId("simulate-panel")).toHaveCount(0);

  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("simulate-legend")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("simulate-legend")).toContainText("temperature");
  await expect(page.getByTestId("simulate-result-summary")).toContainText("nodes");
  // The proof that it was fetched rather than re-solved: the study was
  // posted once, and only once, for the whole round trip.
  expect(solves()).toBe(1);

  // The result is not marked stale, because the program has not changed.
  await expect(page.getByTestId("simulate-stale")).toHaveCount(0);

  // Now change the program. The result still describes the old text, so it
  // stays on screen and says so — throwing it away would be worse.
  await page.getByTestId("sim-tab-meshes").click();
  await page.getByTestId("mesh-arg-sink-mesh-padding").fill("0.03");
  await page.getByTestId("mesh-arg-sink-mesh-padding").blur();
  await page.getByTestId("sim-tab-results").click();
  await expect(page.getByTestId("simulate-stale")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("simulate-legend")).toContainText("temperature");
  expect(solves()).toBe(1);
});

test("a running optimization is cancelled from its own button and lands in the monitor", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await page.getByTestId("window-tab-optimize").click();
  await page.getByTestId("optimize-run-cool-sink").click();

  // The Run button becomes the Cancel button as soon as the stream has
  // named the job — which is the first thing the stream says.
  const cancel = page.getByTestId("optimize-cancel-cool-sink");
  await expect(cancel).toBeVisible();
  await expect(cancel).toHaveText("Cancel", { timeout: 60_000 });
  await cancel.click();

  // The request that started the work ends by itself, and a cancellation is
  // a decision rather than a failure: no error notice anywhere.
  await expect(page.getByTestId("optimize-run-cool-sink")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("optimize-error")).toHaveCount(0);
  await expect(page.getByTestId("optimize-result-cool-sink")).toHaveCount(0);

  // And the monitor records what happened to it.
  await page.getByTestId("window-restore-processes").click();
  await expect(page.getByTestId("processes-panel")).toBeVisible();
  await expect(
    page.locator('[data-testid^="processes-status-"]', { hasText: "cancelled" }).first(),
  ).toBeVisible({ timeout: 20_000 });
});

test("an optimization trajectory outlives a mode switch and a reload", async ({ page }) => {
  // One real study-backed run: the first step pays for the FEM freeze.
  test.setTimeout(600_000);
  await page.getByTestId("window-tab-optimize").click();
  await page.getByTestId("optimize-steps-cool-sink").fill("2");
  await page.getByTestId("optimize-steps-cool-sink").blur();
  await expect(page.getByTestId("optimize-steps-cool-sink")).toHaveValue("2");

  await page.getByTestId("optimize-run-cool-sink").click();
  await expect(page.getByTestId("optimize-result-cool-sink")).toBeVisible({
    timeout: 420_000,
  });
  const scrub = page.getByTestId("optimize-scrub");
  await expect(scrub).toBeVisible();
  const max = await scrub.getAttribute("max");

  // A mode switch tears the Optimize window down and builds the Simulate
  // desk; coming back must find the same run under the same cursor.
  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("simulate-panel")).toBeVisible();
  await page.getByTestId("editmode-model").click();
  await page.getByTestId("window-tab-optimize").click();
  await expect(page.getByTestId("optimize-scrub")).toHaveAttribute("max", max!);

  // The scrubber still drives the replay: dragging it to the first frame
  // ghost-compiles that step's parameters.
  await page.getByTestId("optimize-scrub").fill("0");
  await expect(page.getByTestId("optimize-frame-label")).toContainText("step");

  // And a reload — which loses every signal in the page — restores the run
  // from the server's job store, marked stale because reloading also puts
  // the starter program back in the editor.
  await page.reload();
  await waitForDock(page);
  await page.getByTestId("window-tab-optimize").click();
  await expect(page.getByTestId("optimize-scrub")).toHaveAttribute("max", max!, {
    timeout: 60_000,
  });
  await expect(page.getByTestId("optimize-stale-cool-sink")).toBeVisible();
});

/**
 * The Simulate desk.
 *
 * Simulation used to be one window with a tab strip inside it, which meant
 * four views the dock could not move, split, float or park independently.
 * They are ordinary windows now, and "Simulate" is what arranges them.
 */
test("the Simulate desk arranges four windows that move like any other", async ({
  page,
}) => {
  await page.getByTestId("editmode-simulate").click();

  // Setup above, outcomes below: Studies over Results, with Meshes tabbed
  // behind the first and Optimize behind the second.
  const studiesGroup = page.locator(
    ".dv-groupview:has([data-testid=window-tab-studies]) .dv-tab",
  );
  const resultsGroup = page.locator(
    ".dv-groupview:has([data-testid=window-tab-results]) .dv-tab",
  );
  await expect(studiesGroup).toHaveCount(2);
  await expect(resultsGroup).toHaveCount(2);
  await expect(page.getByTestId("simulate-panel")).toBeVisible();
  await expect(page.getByTestId("results-panel")).toBeVisible();

  // The old tab strip's ids live on the dock's tabs, so the control that
  // used to switch a tab is the control that raises the window.
  await page.getByTestId("sim-tab-meshes").click();
  await expect(page.getByTestId("meshes-panel")).toBeVisible();
  await expect(page.getByTestId("simulate-panel")).toHaveCount(0);
  await page.getByTestId("sim-tab-studies").click();
  await expect(page.getByTestId("simulate-panel")).toBeVisible();

  // And they are windows in every sense: close one and the Window menu has
  // it back, which a tab inside a panel never offered.
  await page.getByTestId("window-close-results").click();
  await expect(page.getByTestId("results-panel")).toHaveCount(0);
  await page.getByTestId("menu-window").click();
  await page.getByTestId("menu-window-results").click();
  await expect(page.getByTestId("results-panel")).toBeVisible();
  await page.keyboard.press("Escape");

  // Optimize is one window with two homes rather than two copies of a card
  // list: the Model desk tabs it behind Materials, this desk behind Results.
  await page.getByTestId("sim-tab-optimize").click();
  await expect(page.getByTestId("optimize-panel")).toBeVisible();
  await page.getByTestId("editmode-model").click();
  await page.getByTestId("window-tab-optimize").click();
  await expect(page.getByTestId("optimize-panel")).toBeVisible();
});

test("a running solve is cancelled from its own button and from the monitor", async ({
  page,
}) => {
  test.setTimeout(300_000);

  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("simulate-study-sink-conduction")).toBeVisible();

  // Solve becomes Cancel in place, once the registry has named the job.
  await page.getByTestId("simulate-run-sink-conduction").click();
  const cancel = page.getByTestId("simulate-cancel-sink-conduction");
  await expect(cancel).toBeVisible();
  await expect(cancel).toHaveText("Cancel", { timeout: 60_000 });
  await cancel.click();

  // A cancellation is a decision, not a failure: the request ends by itself,
  // the button comes back, and nothing anywhere reads as an error.
  await expect(page.getByTestId("simulate-run-sink-conduction")).toBeVisible({
    timeout: 120_000,
  });
  await expect(page.getByTestId("simulate-error")).toHaveCount(0);
  await expect(page.getByTestId("results-error")).toHaveCount(0);

  // The same kill from the other end: the monitor's own stop control.
  await page.getByTestId("window-restore-processes").click();
  await expect(page.getByTestId("processes-panel")).toBeVisible();
  await page.getByTestId("simulate-run-sink-conduction").click();

  const running = page.locator('[data-testid^="processes-cancel-"]');
  await expect(running.first()).toBeVisible({ timeout: 60_000 });
  await running.first().click();

  await expect(page.getByTestId("simulate-run-sink-conduction")).toBeVisible({
    timeout: 120_000,
  });
  await expect(page.getByTestId("simulate-error")).toHaveCount(0);
  await expect(
    page.locator('[data-testid^="processes-status-"]', { hasText: "cancelled" }).first(),
  ).toBeVisible({ timeout: 20_000 });
});

/**
 * The scene browser.
 *
 * What it must prove is that it describes the shipped scenes without running
 * them: the counts and the docstring line come from the server's `ast` pass,
 * and the picture — which is the one thing that does cost a compile — is
 * queued rather than blocking the panel it sits in.
 */
test("the scene browser describes what is saved and opens one", async ({ page }) => {
  test.setTimeout(180_000);

  // File → Open… raises the window; it is parked in every desk.
  await page.getByTestId("menu-file").click();
  await page.getByTestId("menu-file-open").click();
  await expect(page.getByTestId("scenes-panel")).toBeVisible();

  for (const name of ["starter.py", "bracket.py", "end_cap.py"]) {
    await expect(page.getByTestId(`scene-${name}`)).toBeVisible({ timeout: 30_000 });
  }

  // Read, not run: the summary is the docstring's first line and the counts
  // are of declarations, and the end-cap declares no optimization.
  await expect(page.getByTestId("scene-summary-end_cap.py")).toContainText(
    "Gearbox output end-cap",
  );
  const counts = page.getByTestId("scene-counts-bracket.py");
  await expect(counts).toContainText("free");
  await expect(counts).toContainText("studies");
  await expect(page.getByTestId("scene-materials-bracket.py")).toContainText("steel");
  await expect(page.getByTestId("scene-meta-bracket.py")).toContainText("kB");

  // One picture at a time: the frames exist immediately, at the thumbnail's
  // aspect, and leave the queue one after another. A headless browser has no
  // WebGPU, so "drawn" is not what is asserted — "not stuck in the queue" is.
  const frame = page.getByTestId("scene-thumb-starter.py");
  await expect(frame).toBeVisible();
  await expect
    .poll(async () => frame.getAttribute("data-state"), { timeout: 120_000 })
    .not.toBe("queued");

  // Opening one goes through the same path the menu uses.
  await page.getByTestId("scene-open-bracket.py").click();
  await expect(page.getByTestId("menu-scene-name")).toContainText("bracket.py");

  // The browser opens *over* the editor rather than as a fourth column, so
  // the editor's tab is one click away — and that is where the program the
  // browser just loaded is.
  await page.getByTestId("window-tab-editor").click();
  await expect
    .poll(async () => (await editorText(page)).includes("Parametric L-bracket"), {
      timeout: 60_000,
    })
    .toBe(true);
});

/**
 * A solve that fails has to say so where it was asked for.
 *
 * The failure this is written against: switching a mesh to TET10 made the
 * solver refuse the surface, and the only sign in the UI was the previous
 * result still on screen with a "stale" chip — which reads as "still
 * thinking", not as "that run failed". The solver's own words went to the
 * server log.
 */
test("a failed solve shows the solver's message in the Results window", async ({
  page,
}) => {
  test.setTimeout(300_000);

  const editorHas = (needle: string) =>
    expect
      .poll(async () => (await editorText(page)).includes(needle), { timeout: 60_000 })
      .toBe(true);

  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("simulate-study-sink-conduction")).toBeVisible();

  // A boundary condition that selects nothing: a millimetre sphere fifty
  // metres away from the part. The declaration is valid, the solve is not.
  await page.getByTestId("simulate-add-bc-sink-conduction").click();
  await page.getByTestId("simulate-builder-selection").selectOption("sphere");
  for (const component of [0, 1, 2]) {
    const field = page.getByTestId(`simulate-builder-center-${component}`);
    await field.fill("50");
    await field.blur();
  }
  await page.getByTestId("simulate-builder-radius").fill("0.001");
  await page.getByTestId("simulate-builder-radius").blur();
  await page.getByTestId("simulate-builder-add").click();
  // The starter already declares a `Nodes.sphere(...)`, so wait for *this*
  // one: the fifty-metre centre is what makes the selection empty.
  await editorHas("Nodes.sphere([50.0, 50.0, 50.0]");

  await page.getByTestId("simulate-run-sink-conduction").click();

  // The whole message, in the window that would have shown the field.
  const failure = page.getByTestId("results-failure");
  await expect(failure).toBeVisible({ timeout: 240_000 });
  await expect(failure).toContainText("matched no boundary nodes");
  // And the job it came from, with the way to its row in the monitor.
  await expect(failure).toContainText("job ");

  // The button that started the work is idle again, not stuck on "solving".
  await expect(page.getByTestId("simulate-run-sink-conduction")).toBeVisible();
  await expect(page.getByTestId("simulate-run-sink-conduction")).toBeEnabled();

  // "Show in Processes" opens the monitor on that job's row.
  await page.getByTestId("results-error-job").click();
  await expect(page.getByTestId("processes-panel")).toBeVisible();
  await expect(page.locator('[data-testid^="processes-detail-"]').first()).toBeVisible({
    timeout: 20_000,
  });
});
