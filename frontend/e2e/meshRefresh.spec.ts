import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { dragFrom, findHandle, openViewer, recompile, SCENES, waitForCompile } from "./helpers/app";

/**
 * The feature-edge overlay follows a drag at fixed topology.
 *
 * A values-only edit keeps the surfaces the overlay's points lie on, so the
 * server re-solves the last extraction (`/api/mesh_refresh`) rather than
 * extracting again (`/api/mesh`): during the drag for the live picture, and
 * once more with the certificate when the release recompiles. This drives a
 * real drag with the overlay on and counts which of the two the app asked for.
 */

/** How long a compile of this spec's scenes may take. */
const COMPILE_TIMEOUT = 120_000;

test("a drag moves the feature-edge overlay by refresh, not extraction", async ({}, testInfo) => {
  test.setTimeout(300_000);
  const opened = await openViewer({ width: 1200, height: 800, timeout: COMPILE_TIMEOUT });
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

  await recompile(page, readFileSync(resolve(SCENES, "starter.py"), "utf8"), COMPILE_TIMEOUT);

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

  await dragFrom(page, live!.x, live!.y, 0, -40, 20, 30);
  await expect.poll(() => refreshes.length, { timeout: 60_000 }).toBeGreaterThan(0);
  const duringDrag = refreshes.length;
  expect(refreshes.every((r) => r.ok), `live refreshes answered ok: ${JSON.stringify(refreshes)}`).toBe(true);
  expect(refreshes.every((r) => !r.certify), "a live refresh skips the certificate").toBe(true);

  // The release recompiles: a values-only edit, so the overlay is refreshed
  // with the certificate rather than extracted again.
  await page.mouse.up();
  await waitForCompile(page, COMPILE_TIMEOUT);
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
