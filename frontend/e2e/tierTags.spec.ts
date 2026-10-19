import { mkdirSync } from "node:fs";
import { chromium, expect, test, type Browser, type Page } from "@playwright/test";
import { pickOption } from "./helpers/select";

/**
 * The tier, said at every feature it limits.
 *
 * Reads `/api/capabilities` from the page and checks each tagged feature
 * against it, so the same spec holds on either tier:
 *
 *   private: the tags say "Exact … · private tier" and every control runs;
 *   public (`CADJOINT_PUBLIC_TIER=1` on the server, or no diff-brep): the
 *   B-rep inspector is disabled with the server's sentence, feature edges
 *   are tagged approximate, STEP is "STEP · faceted" with the analytic
 *   option off and the reason in place of the note, and a Gmsh mesh is
 *   tagged frozen geometry.
 *
 * With `CADJOINT_TIER_SCREENS=<dir>` it photographs each place, file names
 * prefixed with the tier:
 *
 *   CADJOINT_E2E_PORT=6812 CADJOINT_TIER_SCREENS=/tmp/tier npx playwright test e2e/tierTags.spec.ts
 */

const PORT = process.env.CADJOINT_E2E_PORT ?? 8799;
const SHOTS = process.env.CADJOINT_TIER_SCREENS ?? "";

type Kinds = Record<string, { available: boolean; label: string; message?: string | null }>;

async function waitForCompile(page: Page) {
  await expect(page.getByTestId("status")).not.toHaveText(/^(|Starting…)$/, { timeout: 120_000 });
  await expect(page.getByTestId("toolbar-busy")).toHaveCount(0, { timeout: 120_000 });
}

async function open(): Promise<{ browser: Browser; page: Page }> {
  const browser = await chromium.launch({
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-angle=metal", "--enable-gpu"],
  });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  await page.goto(`http://127.0.0.1:${PORT}/`);
  await waitForCompile(page);
  const dismiss = page.getByRole("button", { name: "Dismiss" });
  if (await dismiss.isVisible().catch(() => false)) await dismiss.click();
  return { browser, page };
}

async function shot(page: Page, tier: string, name: string) {
  if (!SHOTS) return;
  mkdirSync(SHOTS, { recursive: true });
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${SHOTS}/${tier}-${name}.png` });
}

async function recompile(page: Page, edit: (text: string) => string) {
  await page.evaluate((body) => {
    type EditorLike = {
      view?: { state: { doc: { length: number; toString(): string } }; dispatch: (spec: unknown) => void };
    };
    const content = document.querySelector("[data-testid=editor] .cm-content") as
      | (HTMLElement & { cmView?: EditorLike; cmTile?: EditorLike })
      | null;
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    if (!view) throw new Error("no CodeMirror view");
    // eslint-disable-next-line no-new-func
    const next = new Function("text", body)(view.state.doc.toString()) as string;
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: next } });
  }, `return (${edit.toString()})(text);`);
  await page.getByTestId("run").click();
  await waitForCompile(page);
}

test("every tier-limited feature says which tier it runs on", async () => {
  test.setTimeout(300_000);
  const { browser, page } = await open();
  try {
    const kinds = (await page.evaluate(async () => (await fetch("/api/capabilities")).json()))
      .kinds as Kinds;
    const tier = kinds.brep.available ? "private" : "public";

    // ── the render popover: B-rep inspector and feature edges ───────────
    await page.getByTestId("display-options").click();
    await page.getByTestId("render-customize").click();
    const inspector = page.getByTestId("toggle-showBrep");
    await expect(page.getByTestId("tier-brep")).toHaveText(kinds.brep.label);
    await expect(page.getByTestId("tier-feature_edges")).toHaveText(kinds.feature_edges.label);
    if (kinds.brep.available) {
      await expect(inspector).toBeEnabled();
      await expect(page.getByTestId("tier-reason-brep")).toHaveCount(0);
    } else {
      await expect(inspector).toBeDisabled();
      await expect(page.getByTestId("tier-reason-brep")).toHaveText(kinds.brep.message!);
      await expect(page.getByTestId("tier-brep")).toHaveAttribute("data-available", "false");
    }
    await inspector.scrollIntoViewIfNeeded();
    await shot(page, tier, "render-popover");
    await page.getByTestId("display-options").click();

    // ── File › Export…: STEP at the point of choice ────────────────────
    await page.getByTestId("menu-file").click();
    await page.getByTestId("menu-file-export").click();
    await expect(page.getByTestId("export-dialog")).toBeVisible();
    await pickOption(page, "export-format", "step");
    await expect(page.getByTestId("export-tier")).toHaveText(kinds.step_export.label);
    const analytic = page.getByTestId("export-option-analytic");
    if (kinds.step_export.available) {
      await expect(analytic).toBeEnabled();
      await expect(page.getByTestId("export-tier-reason")).toHaveCount(0);
    } else {
      await expect(analytic).toBeDisabled();
      await expect(analytic).not.toBeChecked();
      await expect(page.getByTestId("export-tier-reason")).toHaveText(kinds.step_export.message!);
    }
    await shot(page, tier, "export-step");
    // The choice itself is labelled: the picker's STEP entry.
    await page.getByTestId("export-format").click();
    await expect(page.getByTestId("export-format-option-step")).toHaveText(
      kinds.step_export.available ? /STEP$/ : /STEP · faceted$/,
    );
    await shot(page, tier, "export-picker");
    await page.getByTestId("export-format-option-step").click();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("export-dialog")).toHaveCount(0);

    // ── a Gmsh mesh in the Meshes window ────────────────────────────────
    // A second mesh of the sink, filled by Gmsh. Its own declaration: the
    // starter's optimization runs over `sink-mesh`, and over a frozen Gmsh
    // mesh it would (rightly) refuse at compile.
    await recompile(page, (text: string) =>
      text +
      '\ngmsh_mesh = SimMesh(name="gmsh-mesh", resolution=(18, 13, 11), bounds=(-1.05, -0.8, -0.3),' +
      ' size=(2.1, 1.6, 1.4), method="tet10", mesher="gmsh", domain=thermal_body)\n',
    );
    await page.getByTestId("editmode-simulate").click();
    await page.getByTestId("window-tab-meshes").click();
    const tag = page.getByTestId("mesh-tier-tag-gmsh-mesh");
    await expect(tag).toHaveText(kinds.node_map.label, { timeout: 30_000 });
    if (!kinds.node_map.available) {
      await expect(page.getByTestId("mesh-tier-gmsh-mesh")).toContainText(kinds.node_map.message!);
    }
    // A TetGen mesh carries no tag: it follows the design in either tier.
    await expect(page.getByTestId("mesh-tier-sink-mesh")).toHaveCount(0);
    await page.getByTestId("mesh-gmsh-mesh").scrollIntoViewIfNeeded();
    await shot(page, tier, "meshes-gmsh");
  } finally {
    await browser.close();
  }
});
