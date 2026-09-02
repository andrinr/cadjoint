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

test("the model desk opens docked, tabbed, and with a viewport that cannot be closed", async ({
  page,
}) => {
  for (const id of ["viewport", "editor", "objects", "materials", "optimize"]) {
    await expect(page.getByTestId(`window-tab-${id}`)).toBeVisible();
  }
  await expect(page.getByTestId("window-tab-simulate")).toHaveCount(0);
  await expect(page.getByTestId("window-tab-sketch")).toHaveCount(0);

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
  await expect(page.getByTestId("window-tray")).toHaveCount(0);

  const group = page.locator(".dv-groupview:has([data-testid=window-tab-objects])");
  await group.getByTestId("window-minimise").click();

  await expect(page.getByTestId("window-tab-objects")).toHaveCount(0);
  await expect(page.getByTestId("window-tray")).toBeVisible();
  await expect(page.getByTestId("window-restore-objects")).toHaveText("Objects");

  await page.getByTestId("window-restore-objects").click();
  await expect(page.getByTestId("window-tab-objects")).toBeVisible();
  await expect(page.getByTestId("window-tray")).toHaveCount(0);
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
  await expect(page.getByTestId("window-tab-simulate")).toBeVisible();
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
  await expect(page.getByTestId("window-tray")).toHaveCount(0);
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
