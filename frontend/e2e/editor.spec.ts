import { expect, test, type Page } from "@playwright/test";

/**
 * The editor's three analysis features, end to end against the real ruff and
 * jedi endpoints.
 *
 * These are the parts of the editor a unit test cannot reach: whether the
 * diagnostics the adapter produces actually reach the gutter, whether a fix
 * button applies, and whether the popups appear at all. The coordinate
 * arithmetic under them is asserted in `test/editorIntelligence.test.ts`;
 * what is checked here is that the wiring exists.
 */

async function waitForCompile(page: Page) {
  // The settle signal is the toolbar's busy seam. It is present for exactly
  // as long as the app is behind its own source — from the edit, through the
  // debounce window and the request, until the shaders are installed — which
  // the status line no longer is: while work is in flight the status says
  // nothing at all, so that the one indicator for running work is the
  // toolbar's chip. A settled status is therefore a second, independent
  // signal, and "Starting…" is the placeholder to wait past on a cold load.
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, {
    timeout: 60_000,
  });
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0, { timeout: 60_000 });
}

/** The full document, read off the CodeMirror view rather than the DOM. */
async function editorText(page: Page): Promise<string> {
  return page.evaluate(() => {
    type DocView = { view?: { state: { doc: { toString(): string } } } };
    const content = document.querySelector("[data-testid=editor] .cm-content") as
      | (HTMLElement & { cmView?: DocView; cmTile?: DocView })
      | null;
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    return view ? view.state.doc.toString() : (content?.innerText ?? "");
  });
}

/** Put the caret at the end of the document, so typing appends. */
async function caretToEnd(page: Page) {
  await page.locator("[data-testid=editor] .cm-content").click();
  await page.evaluate(() => {
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
    const end = view.state.doc.length;
    view.dispatch({ selection: { anchor: end, head: end } });
    view.focus();
  });
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await waitForCompile(page);
  const dismiss = page.getByRole("button", { name: "Dismiss" });
  if (await dismiss.isVisible().catch(() => false)) await dismiss.click();
});

test("lint marks an undefined name and applies a ruff fix", async ({ page }) => {
  await caretToEnd(page);
  // An undefined name (F821, an error) and an unused import (F401, a warning
  // that ruff can fix) — one keystroke stream, two severities, one fix.
  await page.keyboard.type("\nimport os\nundefined_name\n");

  // The squiggle and the gutter marker are the two halves of "the editor
  // noticed"; both have to be there, and the error one has to be the error one.
  await expect(page.locator(".cm-lintRange-error").first()).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".cm-lint-marker-error").first()).toBeVisible();
  await expect(page.locator(".cm-lint-marker-warning").first()).toBeVisible();

  // The unused import carries a safe fix; the lint panel lists it as a button.
  // (The panel is CodeMirror's own and orders by document position, so the
  // fix is found by its name rather than by being first.)
  await page.keyboard.press(process.platform === "darwin" ? "Meta+Shift+m" : "Control+Shift+m");
  const action = page.locator(".cm-panel-lint .cm-diagnosticAction", {
    hasText: "Remove unused import",
  });
  await expect(action).toBeVisible({ timeout: 15_000 });
  await action.click();

  await expect
    .poll(async () => (await editorText(page)).includes("import os"), { timeout: 20_000 })
    .toBe(false);
  // …and the undefined name it did not claim to fix is still there.
  expect(await editorText(page)).toContain("undefined_name");
});

test("completion offers the library's own names, with documentation", async ({ page }) => {
  await caretToEnd(page);
  await page.keyboard.type("\nSketchPl");

  const popup = page.locator(".cm-tooltip-autocomplete");
  await expect(popup).toBeVisible({ timeout: 30_000 });
  await expect(popup).toContainText("SketchPlane");
  // The head of the list pays for a docstring, shown in its own panel.
  await expect(page.locator(".cm-completionInfo-doc").first()).toBeVisible({ timeout: 15_000 });
});

test("signature help names the argument being typed", async ({ page }) => {
  await caretToEnd(page);
  await page.keyboard.type("\nSketchPlane(");

  const tooltip = page.locator(".cm-signature-tooltip");
  await expect(tooltip).toBeVisible({ timeout: 30_000 });
  await expect(tooltip).toContainText("SketchPlane(");
  // The active parameter is the one in bold, and there is exactly one.
  await expect(tooltip.locator(".cm-signature-param-active")).toHaveCount(1);
});
