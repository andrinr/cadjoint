import { resolve } from "node:path";

import { chromium, expect, type Browser, type Page } from "@playwright/test";

import type { ShaderStats } from "../../src/viewer/renderer";

/** The playground server's port; `playwright.config.ts` starts it there. */
export const PORT = process.env.CADJOINT_E2E_PORT ?? 8799;

/** The repository's shipped scenes. */
export const SCENES = resolve(import.meta.dirname, "..", "..", "..", "scenes");

/** Chromium flags that expose a hardware WebGPU adapter to a test browser. */
export const WEBGPU_ARGS = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan,WebGPU",
  "--use-angle=metal",
  "--use-gl=angle",
  "--ignore-gpu-blocklist",
  "--enable-gpu",
];

/**
 * Wait until the app has caught up with its own source.
 *
 * The settle signal is the toolbar's busy seam. It is present for exactly as
 * long as the app is behind its own source — from the edit, through the
 * debounce window and the request, until the shaders are installed — which
 * the status line is not: while work is in flight the status says nothing at
 * all, so that the one indicator for running work is the toolbar's chip. A
 * settled status is therefore a second, independent signal, and "Starting…"
 * is the placeholder to wait past on a cold load.
 */
export async function waitForCompile(page: Page, timeout = 60_000) {
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, { timeout });
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0, { timeout });
}

/** Replace the whole document without compiling it. */
export async function setEditorText(page: Page, source: string) {
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
}

/** Replace the whole document, run it, and wait for the compile to land. */
export async function recompile(page: Page, source: string, timeout?: number) {
  await setEditorText(page, source);
  await page.getByTestId("run").click();
  await waitForCompile(page, timeout);
}

/** The full document, read off the CodeMirror view rather than the DOM. */
export async function editorText(page: Page): Promise<string> {
  return page.evaluate(() => {
    type DocView = { view?: { state: { doc: { toString(): string } } } };
    const content = document.querySelector("[data-testid=editor] .cm-content") as
      | (HTMLElement & { cmView?: DocView; cmTile?: DocView })
      | null;
    // CodeMirror hangs its doc view off the content element (`cmView` in
    // older builds, `cmTile` in current ones); either path reaches the full
    // document. innerText is the last resort and truncates tall programs.
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    if (view) return view.state.doc.toString();
    return content?.innerText ?? "";
  });
}

/** Close the banner a machine without WebGPU shows over the canvas, if any. */
export async function dismissBanner(page: Page) {
  const dismiss = page.getByRole("button", { name: "Dismiss" });
  if (await dismiss.isVisible().catch(() => false)) await dismiss.click();
}

/**
 * Launch a WebGPU-capable Chromium on the playground and wait for the first
 * compile.
 *
 * @param options.dismiss Also close a banner left over the canvas.
 * @returns Null, with the browser closed, when no adapter is available.
 *   `errors` collects page errors and console errors from before the load.
 */
export async function openViewer(
  options: { width?: number; height?: number; timeout?: number; dismiss?: boolean } = {},
): Promise<{ browser: Browser; page: Page; errors: string[] } | null> {
  const { width = 1400, height = 900, timeout, dismiss = false } = options;
  const browser = await chromium.launch({ args: WEBGPU_ARGS });
  const page = await browser.newPage({ viewport: { width, height } });
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
  await waitForCompile(page, timeout);
  if (dismiss) await dismissBanner(page);
  return { browser, page, errors };
}

/** The renderer's shader counters, as the app publishes them. */
export async function stats(page: Page): Promise<ShaderStats | null> {
  return page.evaluate(() => window.__cadjointShaders?.() ?? null);
}

/** Where a handle was found, and the hint it showed there. */
export type Found = { x: number; y: number; hint: string } | null;

/**
 * Sweep the pointer over the viewport until the hint names `phrase`.
 *
 * The readout under the viewport is the app's own answer to "what would
 * dragging this cost", and it is computed by the very function that decides
 * whether the handle is drawn filled — so using it to *locate* the handle is
 * what makes later assertions a check on the mark and not just on the
 * renderer. The sweep is synthetic and read-only; a drag is driven with the
 * real mouse (`dragFrom`).
 */
export async function findHandle(page: Page, phrase: string): Promise<Found> {
  return page.evaluate((phrase) => {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const hint = () => document.querySelector("[data-testid=viewer-hint]")?.textContent ?? "";
    for (let y = rect.top + 6; y < rect.bottom - 6; y += 5) {
      for (let x = rect.left + 6; x < rect.right - 6; x += 5) {
        canvas.dispatchEvent(
          new PointerEvent("pointermove", { clientX: x, clientY: y, bubbles: true, pointerId: 1 }),
        );
        const text = hint();
        if (text.includes(phrase)) return { x, y, hint: text };
      }
    }
    return null;
  }, phrase);
}

/**
 * Press at (x, y) and move by (dx, dy) in `steps` real mouse moves, leaving
 * the button down. `pauseMs` waits after each move.
 */
export async function dragFrom(
  page: Page,
  x: number,
  y: number,
  dx: number,
  dy: number,
  steps: number,
  pauseMs = 0,
) {
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= steps; step += 1) {
    await page.mouse.move(x + (dx * step) / steps, y + (dy * step) / steps);
    if (pauseMs) await page.waitForTimeout(pauseMs);
  }
}

/** Toggle one display switch in the render popover's customise panel. */
export async function flip(page: Page, toggle: string) {
  await page.getByTestId("display-options").click();
  await page.getByTestId("render-customize").click();
  await page.getByTestId(toggle).click();
  await page.getByTestId("display-options").click();
}
