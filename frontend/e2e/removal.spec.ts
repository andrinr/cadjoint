import { expect, test } from "@playwright/test";
import { dismissBanner } from "./helpers/app";

/**
 * F7: a removal is not confirmed, and the undo for it is in the top bar
 * straight after, on one line beside the status. Undo brings back exactly
 * what was removed and the notice goes with it.
 */

test("removing a boundary condition offers its undo, which restores it", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("run")).toBeEnabled({ timeout: 60_000 });
  await dismissBanner(page);

  await page.getByTestId("editmode-simulate").click();
  const removeButtons = page.locator('button[title="Remove this boundary condition"]');
  await expect(removeButtons.first()).toBeVisible({ timeout: 30_000 });
  const before = await removeButtons.count();

  await removeButtons.first().click();
  const notice = page.getByTestId("removal");
  await expect(notice).toBeVisible({ timeout: 30_000 });
  await expect(notice).toContainText("removed a boundary condition");
  await expect(removeButtons).toHaveCount(before - 1);

  // One line: the notice never wraps, and the status beside it never runs
  // under it.
  const box = (await notice.boundingBox())!;
  expect(box.height).toBeLessThanOrEqual(24);
  const status = (await page.getByTestId("status").boundingBox())!;
  expect(status.x + status.width).toBeLessThanOrEqual(box.x);

  await page.getByTestId("removal-undo").click();
  await expect(notice).toBeHidden({ timeout: 30_000 });
  await expect(removeButtons).toHaveCount(before);
});
