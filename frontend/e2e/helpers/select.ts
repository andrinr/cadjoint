import { expect, type Locator, type Page } from "@playwright/test";

/**
 * The design-language Select (src/components/ui/Select.tsx) is a button field
 * that opens a listbox, not a native `<select>`: `selectOption` and
 * `toHaveValue` do not apply. The field carries its value in `data-value`,
 * and each option is `${testId}-option-${value}`.
 */

/** Open the Select with `testId` and pick the option whose value is `value`. */
export async function pickOption(page: Page, testId: string, value: string) {
  await page.getByTestId(testId).click();
  await page.getByTestId(`${testId}-option-${value}`).click();
  await expectValue(page.getByTestId(testId), value);
  // A pick closes the list (a wrapping <label> once re-opened it).
  await expect(page.getByTestId(testId)).toHaveAttribute("aria-expanded", "false");
}

/** Assert that a Select field currently holds `value`. */
export async function expectValue(field: Locator, value: string, options?: { timeout?: number }) {
  await expect(field).toHaveAttribute("data-value", value, options);
}
