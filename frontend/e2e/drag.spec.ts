import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";
import {
  dragFrom,
  findHandle,
  openViewer,
  recompile,
  SCENES,
  setEditorText,
  stats,
  waitForCompile,
} from "./helpers/app";

/**
 * What a handle drag actually costs, and whether the viewport says so.
 *
 * `shader.spec.ts` proves the *mechanism*: sixty synthetic parameter writes
 * rebuild nothing. This file proves the mechanism is wired to the pointer,
 * on the shipped `scenes/starter.py`, through real mouse events — and proves
 * the other half, which is that the viewport's mark cannot lie about it.
 *
 * The starter is the right scene for it because it contains both cases side
 * by side:
 *
 * - the fin comb's sixteen points are declared `free=True` and named, so
 *   each one owns a slot in the `@group(3)` buffer. Dragging one is a
 *   `writeBuffer` per pointer move and, on release, a compile whose shader
 *   source is byte-identical — no pipeline at either end.
 * - the slug section's four points are pinned `Vector2`s. They are constants
 *   in the generated WGSL, so dragging one moves the wireframe alone and the
 *   release really does build a new module. That is the designed fallback,
 *   and it is pinned here so it stays deliberate.
 *
 * The viewport draws the first kind filled and the second hollow, and the
 * hint bar names the parameter behind the handle under the pointer. The test
 * *finds* its handles by that readout, so a mark that disagreed with the
 * path would fail the assertion it was used to set up.
 */

/** How long a compile of this spec's scenes may take. */
const COMPILE_TIMEOUT = 120_000;

/** The canvas as a string, for "did the image change" without pixel maths. */
async function frame(page: Page): Promise<number> {
  return page.evaluate(async () => {
    await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    return canvas ? canvas.toDataURL().length : 0;
  });
}

test("a free handle drags live; a fixed one falls back to a recompile", async ({}, testInfo) => {
  test.setTimeout(240_000);
  const opened = await openViewer({ width: 1200, height: 800, timeout: COMPILE_TIMEOUT });
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page, errors } = opened;

  await recompile(page, readFileSync(resolve(SCENES, "starter.py"), "utf8"), COMPILE_TIMEOUT);
  const loaded = await stats(page);
  expect(loaded!.hasParameterBuffer, "the starter's shader reads a parameter buffer").toBe(true);

  // Vertex selection: handles are what this test drags, not whole objects.
  // The key handler lives on `window` and ignores keys typed into the
  // editor, so the press has to land with focus outside it — which it is,
  // on the Run button the recompile just clicked.
  await page.keyboard.press("v");

  // ── the coverage the mark is claiming ─────────────────────────────────
  const bindings = await page.evaluate(() => window.__cadjointBindings?.() ?? []);
  const counted = bindings.reduce<Record<string, number>>((totals, binding) => {
    totals[binding.state] = (totals[binding.state] ?? 0) + 1;
    return totals;
  }, {});
  expect(counted.free, "the starter's free handles").toBeGreaterThan(0);
  expect(counted.fixed, "and its literal ones").toBeGreaterThan(0);

  // ── the fast path ─────────────────────────────────────────────────────
  const live = await findHandle(page, "free parameter");
  expect(live, "a filled handle is somewhere in the viewport").not.toBeNull();
  const beforeLive = await stats(page);
  const imageBefore = await frame(page);

  await dragFrom(page, live!.x, live!.y, 0, -40, 20);
  const during = await stats(page);
  expect(during!.pipelineBuilds, "a live drag builds no pipeline").toBe(
    beforeLive!.pipelineBuilds,
  );
  expect(during!.misses, "and compiles no shader module").toBe(beforeLive!.misses);
  expect(during!.parameterUploads, "it writes the buffer instead, per move").toBeGreaterThan(
    beforeLive!.parameterUploads + 4,
  );
  const imageDuring = await frame(page);
  expect(imageDuring, "and the solid actually moved").not.toBe(imageBefore);

  // The release patches the source; the compile it triggers emits the same
  // shader with different numbers, so it takes the values-only path too.
  await page.mouse.up();
  await waitForCompile(page, COMPILE_TIMEOUT);
  const afterLive = await stats(page);
  expect(afterLive!.pipelineBuilds, "committing a free parameter builds no pipeline").toBe(
    beforeLive!.pipelineBuilds,
  );

  // ── the fallback ──────────────────────────────────────────────────────
  const pinned = await findHandle(page, "fixed value");
  expect(pinned, "a hollow handle is somewhere in the viewport").not.toBeNull();
  const beforePinned = await stats(page);

  await dragFrom(page, pinned!.x, pinned!.y, 0, -30, 12);
  const pinnedDuring = await stats(page);
  expect(pinnedDuring!.parameterUploads, "a literal is not written to the buffer").toBe(
    beforePinned!.parameterUploads,
  );
  expect(pinnedDuring!.pipelineBuilds, "and nothing is built mid-drag either").toBe(
    beforePinned!.pipelineBuilds,
  );

  await page.mouse.up();
  await waitForCompile(page, COMPILE_TIMEOUT);
  const afterPinned = await stats(page);
  // A literal's cost is the recompile itself: the release patches the source
  // and the program is executed again. The direct shader form gives literal
  // vertices slots too (its profile loop reads one contiguous run), so the
  // module that comes back is byte-identical and, as for a free value, no
  // pipeline is rebuilt — the buffer is written with the new numbers.
  expect(afterPinned!.pipelineBuilds, "committing a literal rebuilds no pipeline").toBe(
    beforePinned!.pipelineBuilds,
  );
  expect(afterPinned!.parameterUploads, "it writes the buffer on the recompile").toBeGreaterThan(
    pinnedDuring!.parameterUploads,
  );

  expect(errors.join("\n")).not.toContain("WGSL");
  testInfo.attach("drag", {
    body: JSON.stringify(
      { counted, live, pinned, beforeLive, during, afterLive, beforePinned, afterPinned },
      null,
      1,
    ),
    contentType: "application/json",
  });
  await browser.close();
});

/**
 * Switching scenes must not draw through a destroyed buffer.
 *
 * Installing the new program and its parameter buffer at the *top* of
 * `setShaders` would destroy the buffer that the cached
 * `previewParameterGroup` and `pathParameterGroup` still hold, while modules
 * and pipelines take up to hundreds of milliseconds to build (745 ms on
 * `scenes/motor_shield.py`). Any frame drawn in that window would fail
 * validation with "Buffer with 'SDF parameters' label has been destroyed",
 * which on a big scene is every frame the user causes by
 * touching the mouse while they wait.
 *
 * So the test needs three things at once, and the assertion is worthless
 * without all three: a scene whose buffer is a *different size* from the one
 * installed (an equal size takes an early return and would hide the fault), a shader
 * big enough that its pipelines take real time to build, and frames actually
 * being drawn while that happens. `scenes/end_cap.py` supplies the first two
 * — 11 free parameters against the first scene's one, 1.7 MB of WGSL — and
 * the wheel loop below supplies the third.
 */
test("switching to a differently sized scene draws no destroyed buffer", async ({}, testInfo) => {
  test.setTimeout(240_000);
  const opened = await openViewer({ width: 1200, height: 800, timeout: COMPILE_TIMEOUT });
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page, errors } = opened;

  const small = [
    "from cadjoint.sdf import Sphere, Box, Union",
    "from cadjoint.geometry.parameters import Scalar",
    "",
    "radius = Scalar(0.60, free=True, name='radius')",
    "scene = Union((Sphere(radius=radius), Box(size=[0.9, 0.5, 0.2])), smoothness=0.05)",
    "",
  ].join("\n");
  await recompile(page, small, COMPILE_TIMEOUT);
  expect((await stats(page))!.hasParameterBuffer).toBe(true);

  // The switch, with the viewport busy. The wheel loop redraws every frame:
  // `onWheel` zooms and invalidates, so each tick is a real submitted frame
  // through whatever bind groups the renderer holds at that instant.
  await setEditorText(page, readFileSync(resolve(SCENES, "end_cap.py"), "utf8"));

  const running = page.getByTestId("run").click();
  await page.evaluate(() => {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    if (!canvas) return null;
    const until = performance.now() + 20_000;
    return new Promise((resolve) => {
      let sign = 1;
      const tick = () => {
        sign = -sign;
        canvas.dispatchEvent(
          new WheelEvent("wheel", { deltaY: sign, bubbles: true, cancelable: true }),
        );
        if (performance.now() < until) requestAnimationFrame(tick);
        else resolve(null);
      };
      requestAnimationFrame(tick);
    });
  });
  await running;
  await waitForCompile(page, COMPILE_TIMEOUT);

  // Two witnesses: the renderer forwards every uncaptured GPU error to the
  // viewport's error banner, and Chromium logs it to the console as well.
  const banner = await page.locator(".viewer-error").allTextContents();
  expect(banner.join("\n")).not.toContain("destroyed");
  expect(banner.join("\n")).not.toContain("WebGPU validation error");
  // Chromium words it differently between versions ("used in submit while
  // destroyed", "has been destroyed"); the noun is what both agree on.
  expect(errors.join("\n")).not.toContain("SDF parameters");

  const after = await stats(page);
  expect(after!.hasParameterBuffer, "the new scene installed its own buffer").toBe(true);
  expect(after!.pipelineBuilds, "and it really did rebuild").toBeGreaterThan(0);

  testInfo.attach("switch", {
    body: JSON.stringify({ after, banner, errors }, null, 1),
    contentType: "application/json",
  });
  await browser.close();
});

/**
 * The gizmo takes the same fast path, on the argument it is actually moving.
 *
 * A primitive's placement is the other half of the drag surface, and it is
 * the more interesting half: one node's three gizmo modes bind differently.
 * `head_a`'s position, radius and height are free parameters with slots;
 * its rotation is three free angle parameters the SDF never built, because
 * an identity rotation is not emitted at all — so a rotate drag on the same
 * object must fall back while a translate drag does not. This pins the
 * translate case end to end; the classification of all four arguments is
 * asserted beside it, from the same table the viewport reads.
 */
test("a gizmo translate writes the buffer instead of rebuilding", async ({}, testInfo) => {
  test.setTimeout(240_000);
  const opened = await openViewer({ width: 1200, height: 800, timeout: COMPILE_TIMEOUT });
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page, errors } = opened;

  await recompile(page, readFileSync(resolve(SCENES, "starter.py"), "utf8"), COMPILE_TIMEOUT);

  // `head_a`: a screw head whose position, radius and height are all free.
  await page.getByTestId("tree-row-cylinder_6").click();
  await page.waitForTimeout(500);

  const arguments_ = await page.evaluate(() =>
    (window.__cadjointBindings?.() ?? []).filter((row) => row.nodeId === "cylinder_6"),
  );
  const state = Object.fromEntries(arguments_.map((row) => [row.handle, row.state]));
  expect(state["gizmo position"], "its position is a named free parameter").toBe("free");
  expect(state["gizmo rotation"], "its identity rotation reaches no shader slot").toBe("unbound");

  // The gizmo's arrows are the only thing that puts the canvas in "grab".
  const arrow = await page.evaluate(() => {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    for (let y = rect.top + 4; y < rect.bottom - 4; y += 4) {
      for (let x = rect.left + 4; x < rect.right - 4; x += 4) {
        canvas.dispatchEvent(
          new PointerEvent("pointermove", { clientX: x, clientY: y, bubbles: true, pointerId: 1 }),
        );
        if (canvas.style.cursor === "grab") return { x, y };
      }
    }
    return null;
  });
  expect(arrow, "the selected solid shows a transform gizmo").not.toBeNull();

  const before = await stats(page);
  await dragFrom(page, arrow!.x, arrow!.y, 30, -15, 15);
  const during = await stats(page);
  await page.mouse.up();
  await waitForCompile(page, COMPILE_TIMEOUT);

  expect(during!.pipelineBuilds, "a gizmo drag builds no pipeline").toBe(before!.pipelineBuilds);
  expect(during!.misses, "and compiles no module").toBe(before!.misses);
  expect(during!.parameterUploads, "it writes the position slot per move").toBeGreaterThan(
    before!.parameterUploads + 4,
  );

  expect(errors.join("\n")).not.toContain("WGSL");
  testInfo.attach("gizmo", {
    body: JSON.stringify({ state, arrow, before, during }, null, 1),
    contentType: "application/json",
  });
  await browser.close();
});
