import { expect, test, type Page } from "@playwright/test";
import { recompile, waitForCompile } from "./helpers/app";

/**
 * Two pieces of "the app should just do the obvious thing".
 *
 * **Focus.** Selecting something in the viewport reveals it in the code.
 * Every node publishes the span of the statement that declares it, so a
 * primitive reveals its declaration rather than its *position literal* —
 * three numbers, not the object — and a sketch, whose profile publishes no
 * argument spans, is revealed at all.
 *
 * **Escape.** A ladder, one rung per press, rather than one flat clear of the
 * pending pick, the loft, the probe, the selection, the tool *and* the mode.
 *
 * Both are CPU-side — selection, payload spans, DOM — so neither needs a GPU.
 */

/** How long a compile of this spec's scenes may take. */
const COMPILE_TIMEOUT = 90_000;

/** The text CodeMirror is currently highlighting, or "". */
async function highlighted(page: Page): Promise<string> {
  return page.evaluate(
    () =>
      [...document.querySelectorAll(".cm-vertex-highlight")]
        .map((node) => node.textContent ?? "")
        .join(""),
  );
}

test("selecting an object reveals the statement that declares it", async ({ page }) => {
  await page.goto("/");
  await waitForCompile(page, COMPILE_TIMEOUT);

  // ── a sketch: no argument spans, only its statement ──────────────────
  await page.getByTestId("tree-row-profile_0").click();
  await expect
    .poll(() => highlighted(page), { timeout: 10_000 })
    .toContain("comb_profile = PolygonProfile(");

  // The mark is the declaration's first line, not all twenty-two of it: the
  // reveal lands on the statement without painting a block over the file.
  expect((await highlighted(page)).split("\n").length).toBe(1);

  // ── a primitive: its declaration, not its position literal ───────────
  await page.getByTestId("tree-row-cylinder_2").click();
  await expect
    .poll(() => highlighted(page), { timeout: 10_000 })
    .toContain("bush_a = Solid.cylinder(");

  // ── and from the viewport, which is where a user actually clicks ─────
  await page.getByTestId("editmode-model").click();
  await page.keyboard.press("o");
  const hit = await page.evaluate(() => {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    for (let y = rect.top + 6; y < rect.bottom - 6; y += 5) {
      for (let x = rect.left + 6; x < rect.right - 6; x += 5) {
        canvas.dispatchEvent(
          new PointerEvent("pointermove", { clientX: x, clientY: y, bubbles: true, pointerId: 1 }),
        );
        if (canvas.style.cursor === "pointer") return { x, y };
      }
    }
    return null;
  });
  expect(hit, "something in the viewport is pickable").not.toBeNull();
  await page.mouse.click(hit!.x, hit!.y);
  // Whatever was picked, the editor lands on a *declaration* — `name = Call(`
  // — rather than on a bare literal.
  await expect.poll(() => highlighted(page), { timeout: 10_000 }).toMatch(/^\w+ = \w/);
});

test("Escape cancels one thing per press, starting with the pending command", async ({
  page,
}) => {
  await page.goto("/");
  await waitForCompile(page, COMPILE_TIMEOUT);

  // The starter's sketches are both already consumed by an operator, and the
  // rail disables Loft for those — so this needs two un-operated profiles.
  await recompile(
    page,
    [
      "from cadjoint.construction import PolygonProfile, SketchPlane, Solid",
      "",
      "lower = PolygonProfile([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], name='lower')",
      "upper = PolygonProfile(",
      "    [[0.0, 0.0], [0.6, 0.0], [0.6, 0.6]],",
      "    plane=SketchPlane(origin=[0.0, 0.0, 1.0], normal=[0.0, 0.0, 1.0]),",
      "    name='upper',",
      ")",
      "scene = Solid.box(size=[0.4, 0.4, 0.4], position=[0.0, 0.0, 0.0], name='blk')",
      "",
    ].join("\n"),
    COMPILE_TIMEOUT,
  );

  const patches: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/api/patch")) patches.push(request.url());
  });

  // Arm a loft: select a sketch, open the rail's Shape flyout (only its
  // parent entry is on the rail at rest), then Loft.
  await page.getByTestId("tree-row-profile_0").click();
  await expect
    .poll(() => highlighted(page), { timeout: 10_000 })
    .toContain("lower = PolygonProfile(");
  await page.getByTestId("tool-group-modify").click();
  await page.getByTestId("modify-loft").click();
  await expect(page.getByTestId("viewer-hint")).toContainText("Loft: click the second sketch");

  // ── rung 1: the pending command, and nothing below it ────────────────
  await page.locator("canvas").first().hover();
  await page.keyboard.press("Escape");
  // The hint shows only while a step is in progress (or something is
  // hovered), so with the loft cancelled it is either gone or says something
  // else.
  await expect(page.getByTestId("viewer-hint").filter({ hasText: "Loft:" })).toHaveCount(0);
  // The selection survives — cancelling a command is not a reset. The editor
  // is still showing the sketch's declaration, which is how we can see it.
  expect(await highlighted(page)).toContain("lower = PolygonProfile(");
  expect(patches, "cancelling a loft sends no patch").toHaveLength(0);

  // ── rung 2: the selection ────────────────────────────────────────────
  await page.keyboard.press("Escape");
  await expect.poll(() => highlighted(page), { timeout: 10_000 }).toBe("");

  // ── rung 3: the mode, as the mode switcher shows it ──────────────────
  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("editmode-simulate")).toHaveClass(/active/);
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("editmode-model")).toHaveClass(/active/);
  await expect(page.getByTestId("editmode-simulate")).not.toHaveClass(/active/);

  // ── and a press with nothing pending changes nothing ─────────────────
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("editmode-model")).toHaveClass(/active/);
  expect(patches, "no rung of the ladder edits the program").toHaveLength(0);
});

/**
 * A click on a handle must not leave the viewport dragging it.
 *
 * Selecting a sketch point auto-enters sketch mode; the dock rearranges its
 * panels for that mode and re-parents the canvas mid-gesture, so the
 * `pointerup` is delivered to a detached node and the gesture never finishes:
 * a plain click sends `pointerdown` and no `pointerup`.
 *
 * Unhandled, the viewport stays stuck mid-drag — cursor `grabbing`, every
 * subsequent mouse move calling `setDrag` — so the point silently follows the
 * pointer with no button held, and the next press commits it to the source.
 * So: move the mouse after a click, and nothing may move with it.
 */
test("clicking a sketch handle does not leave the point following the mouse", async ({
  page,
}) => {
  await page.goto("/");
  await waitForCompile(page, COMPILE_TIMEOUT);

  await page.getByTestId("mode-vertex").click();
  const handle = await page.evaluate(() => {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    for (let y = rect.top + 6; y < rect.bottom - 6; y += 5) {
      for (let x = rect.left + 6; x < rect.right - 6; x += 5) {
        canvas.dispatchEvent(
          new PointerEvent("pointermove", { clientX: x, clientY: y, bubbles: true, pointerId: 1 }),
        );
        if (canvas.style.cursor === "grab") return { x, y };
      }
    }
    return null;
  });
  expect(handle, "a sketch handle is on screen").not.toBeNull();

  const patches: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/api/patch")) patches.push(request.url());
  });

  // A plain click: press and release, no movement between them.
  await page.mouse.click(handle!.x, handle!.y);
  await expect(page.getByTestId("selection-chip")).toBeVisible();

  // Now merely move the mouse. Nothing is held, so nothing may be dragged.
  for (let step = 1; step <= 15; step += 1) {
    await page.mouse.move(handle!.x + step * 6, handle!.y - step * 4);
  }
  const cursor = await page.evaluate(
    () => (document.querySelector("canvas") as HTMLCanvasElement | null)?.style.cursor,
  );
  expect(cursor, "the viewport is not still dragging").not.toBe("grabbing");
  expect(patches, "and a phantom drag committed nothing").toHaveLength(0);
});
