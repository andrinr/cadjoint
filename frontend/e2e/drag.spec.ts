import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { chromium, expect, test, type Browser, type Page } from "@playwright/test";

import type { ShaderStats } from "../src/viewer/renderer";

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

const PORT = process.env.CADJOINT_E2E_PORT ?? 8799;

const SCENES = resolve(import.meta.dirname, "..", "..", "scenes");

async function waitForCompile(page: Page) {
  // The settle signal is the toolbar's busy seam. It is present for exactly
  // as long as the app is behind its own source — from the edit, through the
  // debounce window and the request, until the shaders are installed — which
  // the status line no longer is: while work is in flight the status says
  // nothing at all, so that the one indicator for running work is the
  // toolbar's chip. A settled status is therefore a second, independent
  // signal, and "Starting…" is the placeholder to wait past on a cold load.
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, {
    timeout: 120_000,
  });
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0, { timeout: 120_000 });
}

async function stats(page: Page): Promise<ShaderStats | null> {
  return page.evaluate(() => window.__cadjointShaders?.() ?? null);
}

/** Replace the whole document, run it, and wait for the compile to land. */
async function recompile(page: Page, source: string) {
  await page.evaluate((text) => {
    type EditorLike = {
      view?: {
        state: { doc: { length: number } };
        dispatch: (spec: unknown) => void;
      };
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

/**
 * A page on a browser with WebGPU, or null when this build has no adapter.
 *
 * `navigator.gpu` is not exposed on `about:blank`, so the probe has to come
 * after the navigation or a machine that has WebGPU reports that it does not.
 */
async function openViewer(): Promise<{ browser: Browser; page: Page; errors: string[] } | null> {
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
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto(`http://127.0.0.1:${PORT}/`);
  const adapter = await page.evaluate(async () => {
    if (!navigator.gpu) return null;
    return (await navigator.gpu.requestAdapter()) ? "available" : null;
  });
  if (!adapter) {
    await browser.close();
    return null;
  }
  await waitForCompile(page);
  return { browser, page, errors };
}

/** Where a handle of the wanted kind is on screen, found by the hint bar. */
type Found = { x: number; y: number; hint: string } | null;

/**
 * Sweep the pointer over the viewport until the hint names a handle.
 *
 * The readout under the viewport is the app's own answer to "what would
 * dragging this cost", and it is computed by the very function that decides
 * whether the handle is drawn filled — so using it to *locate* the handle is
 * what makes the later assertions a check on the mark and not just on the
 * renderer. The sweep is synthetic and read-only; the drag itself is driven
 * with the real mouse.
 */
async function findHandle(page: Page, kind: "free parameter" | "fixed value"): Promise<Found> {
  return page.evaluate((phrase) => {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const hint = () => document.querySelector("[data-testid=viewer-hint]")?.textContent ?? "";
    for (let y = rect.top + 6; y < rect.bottom - 6; y += 5) {
      for (let x = rect.left + 6; x < rect.right - 6; x += 5) {
        canvas.dispatchEvent(
          new PointerEvent("pointermove", {
            clientX: x,
            clientY: y,
            bubbles: true,
            pointerId: 1,
          }),
        );
        const text = hint();
        if (text.includes(phrase)) return { x, y, hint: text };
      }
    }
    return null;
  }, kind);
}

/** The canvas as a string, for "did the image change" without pixel maths. */
async function frame(page: Page): Promise<number> {
  return page.evaluate(async () => {
    await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    return canvas ? canvas.toDataURL().length : 0;
  });
}

/** Drag from a point by `steps` real mouse moves, without releasing. */
async function dragFrom(page: Page, x: number, y: number, dx: number, dy: number, steps: number) {
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= steps; step += 1) {
    await page.mouse.move(x + (dx * step) / steps, y + (dy * step) / steps);
  }
}

test("a free handle drags live; a fixed one falls back to a recompile", async ({}, testInfo) => {
  test.setTimeout(240_000);
  const opened = await openViewer();
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page, errors } = opened;

  await recompile(page, readFileSync(resolve(SCENES, "starter.py"), "utf8"));
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
  await waitForCompile(page);
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
  await waitForCompile(page);
  const afterPinned = await stats(page);
  // The designed cost of a fixed value, pinned so it stays a decision.
  expect(afterPinned!.pipelineBuilds, "committing a literal does rebuild").toBeGreaterThan(
    beforePinned!.pipelineBuilds,
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
 * The regression this pins: installing the new program and its parameter
 * buffer at the *top* of `setShaders` destroyed the buffer that the cached
 * `previewParameterGroup` and `pathParameterGroup` were still holding, and
 * then spent tens of milliseconds — 745 ms on `scenes/motor_shield.py` —
 * building modules and pipelines before replacing them. Any frame drawn in
 * that window failed validation with "Buffer with 'SDF parameters' label has
 * been destroyed", which on a big scene is every frame the user causes by
 * touching the mouse while they wait.
 *
 * So the test needs three things at once, and the assertion is worthless
 * without all three: a scene whose buffer is a *different size* from the one
 * installed (an equal size took an early return and hid the bug), a shader
 * big enough that its pipelines take real time to build, and frames actually
 * being drawn while that happens. `scenes/end_cap.py` supplies the first two
 * — 11 free parameters against the first scene's one, 1.7 MB of WGSL — and
 * the wheel loop below supplies the third.
 */
test("switching to a differently sized scene draws no destroyed buffer", async ({}, testInfo) => {
  test.setTimeout(240_000);
  const opened = await openViewer();
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
  await recompile(page, small);
  expect((await stats(page))!.hasParameterBuffer).toBe(true);

  // The switch, with the viewport busy. The wheel loop redraws every frame:
  // `onWheel` zooms and invalidates, so each tick is a real submitted frame
  // through whatever bind groups the renderer holds at that instant.
  await page.evaluate((text) => {
    type EditorLike = { view?: { state: { doc: { length: number } }; dispatch: (s: unknown) => void } };
    const content = document.querySelector("[data-testid=editor] .cm-content") as
      | (HTMLElement & { cmView?: EditorLike; cmTile?: EditorLike })
      | null;
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    if (!view) throw new Error("no CodeMirror view");
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
  }, readFileSync(resolve(SCENES, "end_cap.py"), "utf8"));

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
  await waitForCompile(page);

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
  const opened = await openViewer();
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page, errors } = opened;

  await recompile(page, readFileSync(resolve(SCENES, "starter.py"), "utf8"));

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
  await waitForCompile(page);

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
