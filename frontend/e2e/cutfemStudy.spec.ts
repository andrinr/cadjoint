import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium, expect, test, type Browser, type Page } from "@playwright/test";

/**
 * A thermal study on cut cells runs from the Studies window like any other.
 *
 * The starter's own heat study with its mesh switched to `method="cutfem"`:
 * no volume mesh is built, the study solves on the lattice's cut cells, and
 * the temperature is drawn on the dual-contour surface — the legend, the
 * result summary and the quality toggle all behave, with nothing to grade.
 */

const PORT = process.env.CADJOINT_E2E_PORT ?? 8799;
const SCENES = resolve(import.meta.dirname, "..", "..", "scenes");
const SHOTS = process.env.CADJOINT_E2E_SHOTS ?? "";

async function waitForCompile(page: Page) {
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, {
    timeout: 120_000,
  });
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0, { timeout: 120_000 });
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
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
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

test("the starter's heat study solves on cut cells from the Studies window", async ({}, testInfo) => {
  test.setTimeout(400_000);
  const opened = await openViewer();
  test.skip(opened === null, "No WebGPU adapter in this browser build");
  if (!opened) return;
  const { browser, page, errors } = opened;

  page.on("response", async (response) => {
    const url = response.url();
    if (!/\/api\/(simulate|mesh_inspect)$/.test(url)) return;
    const sent = (JSON.parse(response.request().postData() ?? "{}") as { source?: string }).source ?? "";
    try {
      const body = (await response.json()) as {
        ok?: boolean;
        result?: { nodes?: number };
        mesh_info?: { method?: string };
        info?: { method?: string };
        field?: string | null;
      };
      console.log(
        `${url.split("/").pop()}: sent cutfem=${sent.includes('method="cutfem"')} tet10=${sent.includes('method="tet10"')} ` +
          `length=${sent.length} studies=${(sent.match(/name="sink-conduction"/g) ?? []).length} ok=${body.ok} ` +
          `nodes=${body.result?.nodes} method=${body.mesh_info?.method ?? body.info?.method} field=${body.field}`,
      );
    } catch {
      console.log(`${url.split("/").pop()}: unreadable`);
    }
  });
  const source = readFileSync(resolve(SCENES, "starter.py"), "utf8");
  expect(source).toContain('method="tet10"');
  // Typed through the editor, so the app's own source follows: a dispatch
  // straight into CodeMirror compiles but leaves the Studies window on the
  // program it had.
  const editor = page.locator("[data-testid=editor] .cm-content");
  await editor.click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText(source.replaceAll('method="tet10"', 'method="cutfem"'));
  const compiled = page.waitForResponse((r) => r.url().endsWith("/compile"), { timeout: 180_000 });
  await page.getByTestId("run").click();
  const compileBody = JSON.parse((await compiled).request().postData() ?? "{}") as { source?: string };
  expect(compileBody.source ?? "").toContain('method="cutfem"');
  await waitForCompile(page);

  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("simulate-study-sink-conduction")).toBeVisible();
  await page.getByTestId("simulate-run-sink-conduction").click();
  await expect(page.getByTestId("simulate-legend")).toBeVisible({ timeout: 240_000 });
  await expect(page.getByTestId("simulate-legend")).toContainText("temperature");
  const summary = page.getByTestId("simulate-result-summary");
  await expect(summary).toBeVisible();
  await expect(summary).toContainText("nodes");
  const summaryText = (await summary.textContent()) ?? "";
  const legendText = (await page.getByTestId("simulate-legend").textContent()) ?? "";
  console.log(`legend: ${legendText.replace(/\s+/g, " ").trim()}`);
  console.log(`summary: ${summaryText.replace(/\s+/g, " ").trim()}`);
  await page.waitForTimeout(1500);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/cutfem-study.png` });

  // Quality view: cut cells have no elements to grade, so the toggle must
  // neither fail nor claim a jacobian; it comes back to temperature.
  await page.getByTestId("simulate-quality-toggle").check();
  await page.waitForTimeout(4000);
  await expect(page.getByTestId("results-failure")).toHaveCount(0);
  const qualityLegend = (await page.getByTestId("simulate-legend").textContent()) ?? "";
  console.log(`quality legend: ${qualityLegend.replace(/\s+/g, " ").trim()}`);
  expect(qualityLegend).toContain("no elements to grade");
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/cutfem-quality.png` });
  await page.getByTestId("simulate-quality-toggle").uncheck();
  await expect(page.getByTestId("simulate-legend")).toContainText("temperature");

  expect(errors.filter((e) => !e.includes("favicon"))).toEqual([]);
  testInfo.annotations.push({ type: "cutfem", description: summaryText });
  await browser.close();
});
