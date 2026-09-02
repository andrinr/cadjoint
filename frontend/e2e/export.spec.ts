import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";

/**
 * File → Export…, end to end: the starter scene leaves the browser as a file.
 *
 * Everything under test is CPU-side — the dialog, the request, the worker's
 * extraction, the download — so no WebGPU is needed. What the unit tests
 * cannot see is checked here: that the menu opens the dialog, that the
 * download the browser receives is the STL the server wrote (a binary STL is
 * self-describing: 84 bytes of header plus fifty per triangle), and that the
 * run was registered as an `export` job the monitor can list.
 */

async function waitForCompile(page: Page) {
  await expect(page.getByTestId("status")).not.toContainText("compiling", { timeout: 60_000 });
  await expect(page.getByTestId("run")).toBeEnabled({ timeout: 60_000 });
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await waitForCompile(page);
  const dismiss = page.getByRole("button", { name: "Dismiss" });
  if (await dismiss.isVisible().catch(() => false)) await dismiss.click();
});

test("File → Export… downloads the starter scene as a binary STL", async ({ page }) => {
  // A cold compilation cache pays XLA for the extraction once; the export is
  // a tracked job with its own budget, so give the test room for that.
  test.setTimeout(300_000);

  await page.getByTestId("menu-file").click();
  await page.getByTestId("menu-file-export").click();
  const dialog = page.getByTestId("export-dialog");
  await expect(dialog).toBeVisible();

  // The form opens on the top-level scene at the overlay's resolution, and
  // each format says what it will write.
  await expect(page.getByTestId("export-name")).toHaveValue("scene");
  await expect(page.getByTestId("export-resolution")).toHaveValue("64");
  await page.getByTestId("export-format").selectOption("step");
  await expect(page.getByTestId("export-note")).toContainText("B-rep");
  await expect(page.getByTestId("export-option-analytic")).toBeChecked();
  await page.getByTestId("export-format").selectOption("stl");
  await expect(page.getByTestId("export-option-binary")).toBeChecked();

  // Coarse on purpose: the point is the file, not its fidelity.
  await page.getByTestId("export-resolution").fill("16");
  await page.getByTestId("export-resolution").dispatchEvent("change");

  const downloaded = page.waitForEvent("download", { timeout: 240_000 });
  await page.getByTestId("export-confirm").click();
  await expect(page.getByTestId("export-confirm")).toBeDisabled();
  const download = await downloaded;

  // Unsaved buffer: the file keeps the server's own name.
  expect(download.suggestedFilename()).toBe("scene.stl");
  const bytes = readFileSync((await download.path())!);
  expect(bytes.subarray(0, 26).toString("ascii")).toBe("cadjoint binary STL export");
  const triangles = bytes.readUInt32LE(80);
  expect(triangles).toBeGreaterThan(100);
  expect(bytes.length).toBe(84 + 50 * triangles);

  // The dialog is done with, and the status line says what was written.
  await expect(dialog).toHaveCount(0);
  await expect(page.getByTestId("status")).toContainText("Exported scene.stl");

  // The run was a job: the registry lists it, done, under its format.
  const session = (await (await page.request.get("/api/session")).json()) as { token: string };
  const jobs = (await (
    await page.request.get("/api/jobs", { headers: { "X-Cadjoint-Token": session.token } })
  ).json()) as { jobs: { kind: string; status: string; fields: Record<string, unknown> }[] };
  const exportJob = jobs.jobs.find((job) => job.kind === "export");
  expect(exportJob).toBeDefined();
  expect(exportJob!.status).toBe("done");
  expect(exportJob!.fields).toMatchObject({ format: "stl", resolution: 16 });
});

test("an object the program does not bind is refused with the names it does", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await page.getByTestId("menu-file").click();
  await page.getByTestId("menu-file-export").click();
  await page.getByTestId("export-name").fill("no_such_part");
  await page.getByTestId("export-resolution").fill("8");
  await page.getByTestId("export-resolution").dispatchEvent("change");
  await page.getByTestId("export-confirm").click();

  const error = page.getByTestId("export-error");
  await expect(error).toBeVisible({ timeout: 150_000 });
  await expect(error).toContainText("no SDF object named 'no_such_part'");
  await expect(error).toContainText("'scene'");
  // A failure keeps the dialog open, ready to be corrected.
  await expect(page.getByTestId("export-dialog")).toBeVisible();
  await expect(page.getByTestId("export-confirm")).toBeEnabled();
});
