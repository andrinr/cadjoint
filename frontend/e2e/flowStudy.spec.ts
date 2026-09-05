/**
 * A scene declaring a flow study opens, compiles, and reads in the panel.
 *
 * `scenes/duct_sink.py` is the one shipped scene whose study is a
 * `FlowStudy`, and it is the case the Studies window historically had no
 * rendering for. What the panel may do with it is narrower than for a mesh
 * study — the GUI cannot author one yet, so the card reports it and points
 * at the code — but "narrower" has to mean fewer controls, not a thrown
 * error, and that is what this test holds.
 */

import { expect, test, type Page } from "@playwright/test";

async function waitForCompile(page: Page) {
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, { timeout: 120_000 });
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0, { timeout: 120_000 });
}

test("the duct scene's flow study reads in the Studies window", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(String(error)));

  await page.goto("/");
  await waitForCompile(page);
  const dismiss = page.getByRole("button", { name: "Dismiss" });
  if (await dismiss.isVisible().catch(() => false)) await dismiss.click();

  await page.getByTestId("menu-file").click();
  await page.getByTestId("menu-file-open").click();
  await page.getByTestId("scene-open-duct_sink.py").click();
  await waitForCompile(page);

  await page.getByTestId("menu-window").click();
  await page.getByTestId("menu-window-studies").click();
  await page.keyboard.press("Escape");

  // The card names the study, and says where it can be edited.
  await expect(page.getByText("duct-cooling").first()).toBeVisible();
  await expect(page.getByText(/edit it there/).first()).toBeVisible();

  // Rendering it must not throw: three of a flow study's conditions carry no
  // node selection at all, which every mesh-study row assumes it has.
  expect(errors, errors.join("\n")).toEqual([]);
});
