import { chromium, expect, test, type Page } from "@playwright/test";

import type { ShaderStats } from "../src/viewer/renderer";

/**
 * A parameter edit must not rebuild the scene's pipelines.
 *
 * This is the whole point of the uniform form, and it is a *negative*
 * claim — nothing was compiled — so it can only be checked against a
 * counter. `window.__cadjointShaders()` publishes the renderer's, and the
 * test drives a real edit through the real editor, server and GPU:
 *
 * 1. Compile the scene. Pipelines get built; the parameter buffer exists.
 * 2. Change one numeric literal and recompile. The worker emits the same
 *    shader source with a different buffer, so the renderer takes the
 *    values-only path: `parameterUploads` moves and `pipelineBuilds` does
 *    not.
 * 3. Change the *topology* and recompile. Now the source really is
 *    different, and pipelines are expected to be built.
 *
 * Step 3 matters as much as step 2: without it a renderer that had simply
 * stopped installing anything would pass.
 */

const PORT = process.env.CADJOINT_E2E_PORT ?? 8799;

async function waitForCompile(page: Page) {
  await expect(page.getByTestId("status")).not.toContainText("compiling", { timeout: 90_000 });
  await expect(page.getByTestId("run")).toBeEnabled({ timeout: 90_000 });
}

async function stats(page: Page): Promise<ShaderStats | null> {
  return page.evaluate(() => window.__cadjointShaders?.() ?? null);
}

/** Replace the whole document and press Run. */
async function recompile(page: Page, source: string) {
  await page.evaluate((text) => {
    type EditorLike = {
      view?: {
        state: { doc: { length: number } };
        dispatch: (spec: unknown) => void;
        focus: () => void;
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

test("a parameter edit uploads a buffer and rebuilds no pipelines", async ({}, testInfo) => {
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

  // Navigate before asking for an adapter: `navigator.gpu` is not exposed on
  // `about:blank`, so probing first reports "no WebGPU" on a machine that
  // has it and silently skips the test.
  await page.goto(`http://127.0.0.1:${PORT}/`);
  const adapter = await page.evaluate(async () => {
    if (!navigator.gpu) return null;
    return (await navigator.gpu.requestAdapter()) ? "available" : null;
  });
  if (!adapter) {
    await browser.close();
    test.skip(true, "No WebGPU adapter in this browser build");
    return;
  }
  await waitForCompile(page);

  // A scene small enough to recompile three times inside the timeout, and
  // parametric enough that one literal is a parameter edit.
  //
  // The radius has to be a *declared free parameter*, not a bare float. Only
  // the free parameters live in the uniform buffer — a bare float is a fixed
  // one, folded into the source as a constant, and editing it is a different
  // module by design. See `compile_scene_with_uniforms`: buffering the fixed
  // ones too costs 31x the frame time, because it is exactly the constants
  // that let the GPU's compiler fold the scene away.
  const base = [
    "from cadjoint.sdf import Sphere, Box, Union",
    "from cadjoint.geometry.parameters import Scalar",
    "",
    "radius = Scalar(0.60, free=True, name='radius')",
    "ball = Sphere(radius=radius)",
    "slab = Box(size=[0.9, 0.5, 0.2])",
    "scene = Union((ball, slab), smoothness=0.05)",
    "",
  ].join("\n");
  await recompile(page, base);

  const first = await stats(page);
  expect(first, "the app publishes its shader counters").not.toBeNull();
  expect(first!.hasParameterBuffer, "the scene shader reads a parameter buffer").toBe(true);
  expect(first!.pipelineBuilds).toBeGreaterThan(0);

  // ── the claim ────────────────────────────────────────────────────────
  await recompile(page, base.replace("Scalar(0.60,", "Scalar(0.42,"));
  const moved = await stats(page);
  expect(moved!.pipelineBuilds, "a parameter edit builds no pipeline").toBe(
    first!.pipelineBuilds,
  );
  expect(moved!.parameterUploads, "it uploads the buffer instead").toBeGreaterThan(
    first!.parameterUploads,
  );

  // ── the control ──────────────────────────────────────────────────────
  // A different tree is a different shader, and must still be installed.
  const topology = base.replace(
    "scene = Union((ball, slab), smoothness=0.05)",
    "scene = Union((ball, slab, Box(size=[0.2, 0.2, 1.4])), smoothness=0.05)",
  );
  await recompile(page, topology);
  const rebuilt = await stats(page);
  expect(rebuilt!.pipelineBuilds, "a topology edit does build pipelines").toBeGreaterThan(
    moved!.pipelineBuilds,
  );

  // ── the module cache ─────────────────────────────────────────────────
  // Going back to a source this session already compiled is an undo, and it
  // must not recompile the module. The pipelines are rebuilt — they were
  // replaced by the topology edit — but `misses` must not move.
  await recompile(page, base);
  const undone = await stats(page);
  expect(undone!.pipelineBuilds, "the earlier shader is installed again").toBeGreaterThan(
    rebuilt!.pipelineBuilds,
  );
  expect(undone!.misses, "an undo compiles no new module").toBe(rebuilt!.misses);
  expect(undone!.hits, "it is served from the module cache").toBeGreaterThan(
    rebuilt!.hits,
  );

  expect(errors.join("\n")).not.toContain("WGSL");
  testInfo.attach("stats", {
    body: JSON.stringify(
      {
        first,
        moved,
        rebuilt,
        undone,
        hitRate: undone!.hits / Math.max(1, undone!.hits + undone!.misses),
      },
      null,
      1,
    ),
    contentType: "application/json",
  });
  await browser.close();
});

/**
 * A drag at frame rate rebuilds nothing.
 *
 * The test above proves a *committed* parameter edit takes the values-only
 * path through the compile cycle. This one proves the path underneath it:
 * while a handle is actually moving, the viewer answers every pointer move
 * with a buffer write and a redraw, and never touches a shader module or a
 * pipeline. Sixty of them stand in for a second of dragging.
 *
 * The image is read back before and after, because a counter alone would be
 * satisfied by a renderer that ignored the overrides entirely.
 */
test("dragging a parameter at frame rate rebuilds no pipeline", async ({}, testInfo) => {
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

  await page.goto(`http://127.0.0.1:${PORT}/`);
  const adapter = await page.evaluate(async () => {
    if (!navigator.gpu) return null;
    return (await navigator.gpu.requestAdapter()) ? "available" : null;
  });
  if (!adapter) {
    await browser.close();
    test.skip(true, "No WebGPU adapter in this browser build");
    return;
  }
  await waitForCompile(page);

  await recompile(
    page,
    [
      "from cadjoint.sdf import Sphere, Box, Union",
      "from cadjoint.geometry.parameters import Scalar",
      "",
      "radius = Scalar(0.60, free=True, name='radius')",
      "scene = Union((Sphere(radius=radius), Box(size=[0.9, 0.5, 0.2])), smoothness=0.05)",
      "",
    ].join("\n"),
  );

  const before = await stats(page);
  expect(before!.hasParameterBuffer, "the scene shader reads a parameter buffer").toBe(true);

  // ── the drag ─────────────────────────────────────────────────────────
  const dragged = await page.evaluate(async () => {
    const set = window.__cadjointSetParameters;
    if (!set) return { accepted: false, frames: 0 };
    let frames = 0;
    for (let i = 0; i < 60; i += 1) {
      const radius = 0.6 - (0.35 * i) / 59;
      if (!set({ radius: [radius] })) return { accepted: false, frames };
      frames += 1;
      await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    }
    return { accepted: true, frames };
  });
  expect(dragged.accepted, "the renderer accepted the parameter overrides").toBe(true);
  expect(dragged.frames).toBe(60);

  const after = await stats(page);
  expect(after!.pipelineBuilds, "a drag builds no pipeline").toBe(before!.pipelineBuilds);
  expect(after!.misses, "a drag compiles no shader module").toBe(before!.misses);
  expect(after!.parameterUploads, "one upload per dragged frame").toBe(
    before!.parameterUploads + 60,
  );

  // ── and the overrides actually reached the image ─────────────────────
  const changed = await page.evaluate(async () => {
    const set = window.__cadjointSetParameters!;
    const shot = async () => {
      await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
      await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
      const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
      return canvas ? canvas.toDataURL().length : 0;
    };
    set({ radius: [0.25] });
    const small = await shot();
    set({ radius: [1.4] });
    const large = await shot();
    set(null);
    return { small, large };
  });
  expect(changed.small, "the canvas read back").toBeGreaterThan(0);
  expect(changed.large).not.toBe(changed.small);

  expect(errors.join("\n")).not.toContain("WGSL");
  testInfo.attach("drag", {
    body: JSON.stringify({ before, after, dragged, changed }, null, 1),
    contentType: "application/json",
  });
  await browser.close();
});
