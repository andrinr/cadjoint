import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium, expect, test, type Browser, type Page } from "@playwright/test";

/**
 * The feature-edge overlay follows a drag at fixed topology.
 *
 * A values-only edit keeps the surfaces the overlay's points lie on, so the
 * server re-solves the last extraction (`/api/mesh_refresh`) rather than
 * extracting again (`/api/mesh`): during the drag for the live picture, and
 * once more with the certificate when the release recompiles. This drives a
 * real drag with the overlay on and counts which of the two the app asked for.
 */

const PORT = process.env.CADJOINT_E2E_PORT ?? 8799;
const SCENES = resolve(import.meta.dirname, "..", "..", "scenes");

async function waitForCompile(page: Page) {
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, {
    timeout: 120_000,
  });
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

type Found = { x: number; y: number; hint: string } | null;

async function findHandle(page: Page, phrase: string): Promise<Found> {
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

async function dragFrom(page: Page, x: number, y: number, dx: number, dy: number, steps: number) {
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= steps; step += 1) {
    await page.mouse.move(x + (dx * step) / steps, y + (dy * step) / steps);
    await page.waitForTimeout(30);
  }
}

test("a drag moves the feature-edge overlay by refresh, not extraction", async ({}, testInfo) => {
  test.setTimeout(300_000);
  const opened = await openViewer();
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page, errors } = opened;

  const refreshes: { ok: boolean; stale: number; certify: boolean; ms: number }[] = [];
  const started = new Map<string, number>();
  let extractions = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/api/mesh_refresh")) started.set(request.url() + refreshes.length, Date.now());
  });
  const traffic: string[] = [];
  page.on("response", async (response) => {
    const url = response.url();
    if (url.endsWith("/patch") || url.endsWith("/compile")) {
      try {
        const body = (await response.json()) as { ok?: boolean; error?: string; shader_hash?: string };
        traffic.push(`${url.split("/").pop()} ok=${body.ok} ${body.error ?? ""} ${body.shader_hash?.slice(0, 8) ?? ""}`);
      } catch {
        traffic.push(`${url.split("/").pop()} unreadable`);
      }
    }
    if (url.endsWith("/api/mesh")) extractions += 1;
    if (!url.endsWith("/api/mesh_refresh")) return;
    const sent = JSON.parse(response.request().postData() ?? "{}") as { certify?: boolean };
    const at = started.get(url + refreshes.length) ?? Date.now();
    try {
      const body = (await response.json()) as { ok: boolean; stale?: number };
      refreshes.push({ ok: body.ok, stale: body.stale ?? -1, certify: Boolean(sent.certify), ms: Date.now() - at });
    } catch {
      refreshes.push({ ok: false, stale: -1, certify: Boolean(sent.certify), ms: Date.now() - at });
    }
  });

  await recompile(page, readFileSync(resolve(SCENES, "starter.py"), "utf8"));

  // Feature edges on: the eye opens the render popover, Customize shows the
  // switches, and the popover closes again so the canvas is free to drag.
  await page.getByTestId("display-options").click();
  await page.getByTestId("render-customize").click();
  await page.getByTestId("toggle-showMeshEdges").click();
  await page.getByTestId("display-options").click();
  await expect.poll(() => extractions, { timeout: 120_000 }).toBe(1);
  // the overlay's points are on screen once the answer is drawn; the server
  // builds the refreshable overlay in the background meanwhile
  await page.waitForTimeout(3000);

  await page.keyboard.press("v");
  const live = await findHandle(page, "free parameter");
  expect(live, "a filled handle is somewhere in the viewport").not.toBeNull();

  await dragFrom(page, live!.x, live!.y, 0, -40, 20);
  await expect.poll(() => refreshes.length, { timeout: 60_000 }).toBeGreaterThan(0);
  const duringDrag = refreshes.length;
  expect(refreshes.every((r) => r.ok), `live refreshes answered ok: ${JSON.stringify(refreshes)}`).toBe(true);
  expect(refreshes.every((r) => !r.certify), "a live refresh skips the certificate").toBe(true);

  // The release recompiles: a values-only edit, so the overlay is refreshed
  // with the certificate rather than extracted again.
  await page.mouse.up();
  await waitForCompile(page);
  await page.waitForTimeout(5000);
  console.log(`after release: refreshes=${refreshes.length} extractions=${extractions} ${JSON.stringify(refreshes.slice(-3))}`);
  console.log(`traffic: ${traffic.join(" | ")}`);
  console.log(`errors: ${JSON.stringify(errors)}`);
  await expect.poll(() => refreshes.length, { timeout: 30_000 }).toBeGreaterThan(duringDrag);
  await expect.poll(() => refreshes.filter((r) => r.certify).length, { timeout: 60_000 }).toBeGreaterThan(0);
  await page.waitForTimeout(1500);
  const certified = refreshes.filter((r) => r.certify);
  expect(certified.every((r) => r.ok && r.stale >= 0 && r.stale <= 0.005), JSON.stringify(certified)).toBe(true);
  expect(extractions, "no second extraction: the certificate held").toBe(1);
  expect(errors).toEqual([]);

  testInfo.annotations.push({
    type: "refresh",
    description: `${duringDrag} live refreshes during the drag, ${certified.length} certified after; ms=${refreshes.map((r) => r.ms).join(",")}`,
  });
  console.log(`refresh: live=${duringDrag} certified=${certified.length} stale=${certified.map((r) => r.stale).join(",")} ms=${refreshes.map((r) => r.ms).join(",")}`);
  await browser.close();
});
