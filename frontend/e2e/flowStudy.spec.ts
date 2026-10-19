/**
 * A flow study, declared and authored in the panel like the other two.
 *
 * `scenes/duct_sink.py` is the one shipped scene whose study is a
 * `FlowStudy`. Two things are checked here that a unit test cannot: that a real payload
 * from a real compile renders without throwing — three of a flow study's
 * conditions carry no node selection, which a mesh-study row expects — and
 * that adding one writes a program the server can run.
 */

import { expect, test, type Page } from "@playwright/test";
import { dismissBanner, waitForCompile } from "./helpers/app";

/** How long a compile of this spec's scenes may take. */
const COMPILE_TIMEOUT = 120_000;

async function openStudies(page: Page) {
  // The Window menu item toggles, and dock arrangements survive a reload, so
  // clicking it blindly closes the panel a previous test left open.
  if (await page.getByTestId("simulate-add-study").isVisible().catch(() => false)) return;
  await page.getByTestId("menu-window").click();
  await page.getByTestId("menu-window-studies").click();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("simulate-add-study")).toBeVisible({ timeout: 30_000 });
}

async function start(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto("/");
  await waitForCompile(page, COMPILE_TIMEOUT);
  await dismissBanner(page);
  return errors;
}

test("the duct scene's conditions read in the Studies window", async ({ page }) => {
  const errors = await start(page);
  await page.getByTestId("menu-file").click();
  await page.getByTestId("menu-file-open").click();
  await page.getByTestId("scene-open-duct_sink.py").click();
  await openStudies(page);

  // Waiting on the opened scene's own payload: the status has been settled
  // since the first compile, so it is not the signal that this one arrived.
  await expect(page.getByText("duct-cooling").first()).toBeVisible({ timeout: 120_000 });

  const rows = await page.getByTestId(/^simulate-bcs-/).first().innerText();
  // The three that place nothing say where they act; the one that does
  // describes its region the ordinary way.
  expect(rows).toContain("lattice inlet face");
  expect(rows).toContain("lattice outlet face");
  expect(rows).toContain("duct walls");
  expect(rows).toContain("box [-0.14, -0.18, -0.42] → [0.14, 0.18, -0.18]");
  expect(errors, errors.join("\n")).toEqual([]);
});

test("adding a flow study writes a runnable declaration", async ({ page }) => {
  const errors = await start(page);
  // The server's session carries whatever scene ran last, so say which one
  // this starts from rather than inheriting the previous test's.
  await page.getByTestId("menu-file").click();
  await page.getByTestId("menu-file-open").click();
  await page.getByTestId("scene-open-starter.py").click();
  await openStudies(page);
  await expect(page.getByText("sink-conduction").first()).toBeVisible({ timeout: 120_000 });
  // One "+ Study" opens the kinds, as Simulate › New study does.
  await page.getByTestId("simulate-add-study").click();
  await page.getByTestId("simulate-add-flow").click();
  await expect(page.getByText("study1").first()).toBeVisible({ timeout: 120_000 });

  // That it compiled at all is the claim: a flow study refuses to construct
  // without an inlet, so a declaration written with `bcs=[]` — which is what
  // the mesh kinds get — would have raised on this very compile. The card is
  // rendered from the payload of the program the patch produced.
  const card = page.getByTestId(/^simulate-bcs-study1$/);
  await expect(card).toContainText("Inlet");
  await expect(card).toContainText("Outlet");
  await expect(card).toContainText("Duct walls");

  await expect(page.getByTestId("simulate-mesh-sink-conduction")).toBeVisible();
  expect(errors, errors.join("\n")).toEqual([]);
});
