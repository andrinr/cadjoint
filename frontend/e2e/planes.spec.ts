import { expect, test, type Page } from "@playwright/test";
import { dismissBanner, editorText, recompile, waitForCompile } from "./helpers/app";
import { canvasMetrics, gizmoTip, projectToCss } from "./helpers/projection";
import { expectValue, pickOption } from "./helpers/select";

/**
 * Sketch planes as viewport elements, and the properties window.
 *
 * Three things the source must answer to: a scene with a sketch draws a
 * plane for it (a frame the GPU paints, and a name tag the DOM paints — the
 * tag is what these tests read, so they hold on a browser with no WebGPU);
 * taking hold of a plane and dragging the gizmo rewrites the plane's origin
 * literal and nothing else; and the properties window shows the selection's
 * written arguments and writes an edit back through `/api/patch`.
 *
 * The projection comes from `helpers/projection.ts`, the independent
 * reimplementation of `src/viewer/math` that `playground.spec.ts` uses too:
 * a drift in the app's projection must fail the test, not be shared by it.
 */

/**
 * Turn on the "all sketch planes" switch in the render popover.
 *
 * Planes are otherwise drawn only while sketching, and only for the sketch
 * that is selected or under the pointer; the tests that look at every plane
 * at once ask for them explicitly, the way a user would.
 */
async function showAllPlanes(page: Page) {
  await page.getByTestId("display-options").click();
  await page.getByTestId("render-customize").click();
  await page.getByTestId("toggle-showAllPlanes").click();
  await page.getByTestId("display-options").click();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await waitForCompile(page);
  await dismissBanner(page);
});

test("every sketch in the scene draws its plane, named", async ({ page }) => {
  // Not by default: a dozen sketches would be a dozen quads. The switch
  // shows them all; the policy itself is the test below this one.
  await expect(page.getByTestId("plane-label")).toHaveCount(0);
  await showAllPlanes(page);
  // The starter has two sketches; each gets a frame and a name tag.
  const labels = page.getByTestId("plane-label");
  await expect(labels).toHaveCount(2);
  await expect(labels.nth(0)).toHaveText("fin comb");
  await expect(labels.nth(1)).toHaveText("slug section");
  await expect(page.getByTestId("plane-label-preview")).toHaveCount(0);

  // Clicking the tag takes hold of the plane: the sketch is the selection,
  // and the properties window leads with the plane's own numbers.
  await labels.nth(0).click();
  await expect(page.getByTestId("selection-chip")).toHaveText("fin comb");
  await expect(page.getByTestId("gizmo-translate")).toHaveClass(/active/);
  const sections = page.locator(".properties-section");
  await expect(sections.first()).toHaveAttribute("data-element", "plane:comb_profile");
  await expect(page.getByTestId("prop-plane-normal-1")).toHaveValue("1");
});

test("the sketch tool previews the plane a click will create", async ({ page }) => {
  const child = page.getByTestId("tool-sketch");
  for (let attempt = 0; attempt < 2 && !(await child.isVisible()); attempt++) {
    await page.getByTestId("tool-group-create").click();
  }
  await child.click();
  const metrics = await canvasMetrics(page);
  await page.mouse.move(metrics.left + metrics.clientWidth * 0.3, metrics.top + metrics.clientHeight * 0.4);
  await expect(page.getByTestId("plane-label-preview")).toHaveText("new sketch");
  await expect(page.getByTestId("viewer-hint")).toContainText("the ghost shows where it lands");

  // Escape drops the tool, and the ghost with it.
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("plane-label-preview")).toHaveCount(0);
});

const FLAT_SKETCH = [
  "from cadjoint.construction import PolygonProfile, SketchPlane, extrude",
  "",
  "pad = PolygonProfile(",
  "    [[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]],",
  "    plane=SketchPlane(origin=[0.3, 0.2, 0.4], normal=[0.0, 0.0, 1.0]),",
  '    name="pad",',
  ")",
  "scene = extrude(pad, depth=0.3)",
  "",
].join("\n");

test("planes are drawn while sketching, for the active sketch only", async ({ page }) => {
  await recompile(page, FLAT_SKETCH);
  const labels = page.getByTestId("plane-label");
  await expect(labels).toHaveCount(0);
  await page.getByTestId("editmode-sketch").click();
  await expect(labels).toHaveCount(0);

  // Selecting the sketch — its outline, not its (invisible) plane — reveals its plane.
  const metrics = await canvasMetrics(page);
  const edge = projectToCss([0.3, -0.2, 0.4], metrics);
  await page.mouse.click(metrics.left + edge.x, metrics.top + edge.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("pad");
  await expect(labels).toHaveText(["pad"]);

  // Leaving sketch mode hides it again, selection or not.
  await page.getByTestId("editmode-model").click();
  await expect(labels).toHaveCount(0);
});

test("dragging a plane's origin rewrites the plane literal and nothing else", async ({ page }) => {
  await recompile(page, FLAT_SKETCH);
  await showAllPlanes(page);
  await expect(page.getByTestId("plane-label")).toHaveText(["pad"]);

  // Take hold of the plane by its frame edge: the frame's -v edge runs
  // through origin + v * -(0.4 + 0.25), and its midpoint is on that edge.
  const metrics = await canvasMetrics(page);
  const edge = projectToCss([0.3, 0.2 - 0.65, 0.4], metrics);
  await page.mouse.move(metrics.left + edge.x, metrics.top + edge.y);
  await expect(page.getByTestId("viewer-hint")).toContainText("Sketch plane of pad");
  await page.mouse.click(metrics.left + edge.x, metrics.top + edge.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("pad");

  // The gizmo sits on the plane's origin, not the polygon's centre; drag its
  // Z arrow up.
  const from = gizmoTip([0.3, 0.2, 0.4], 2, metrics);
  const to = projectToCss([0.3, 0.2, 1.1], metrics);
  await page.mouse.move(metrics.left + from.x, metrics.top + from.y);
  await page.mouse.down();
  await page.mouse.move(metrics.left + to.x, metrics.top + to.y, { steps: 12 });
  await page.mouse.up();

  await expect
    .poll(
      async () => {
        const match = (await editorText(page)).match(/SketchPlane\(origin=\[([^\]]+)\]/);
        return match ? Number(match[1].split(",")[2]) : 0;
      },
      { timeout: 45_000 },
    )
    .toBeGreaterThan(0.6);
  const text = await editorText(page);
  // The polygon's points and the normal are untouched.
  expect(text).toContain("[[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]]");
  expect(text).toContain("normal=[0.0, 0.0, 1.0]");
  expect(text).toContain("depth=0.3");
});

test("the properties window edits a plane's numbers into the source", async ({ page }) => {
  await recompile(page, FLAT_SKETCH);
  await page.getByTestId("tree-row-profile_0").click();
  await expect(page.locator(".properties-section").first()).toHaveAttribute(
    "data-element",
    "assign:pad",
  );
  const originY = page.getByTestId("prop-plane-origin-1");
  await expect(originY).toHaveValue("0.2");
  await originY.fill("0.9");
  await originY.press("Enter");
  await expect
    .poll(async () => (await editorText(page)).includes("origin=[0.3, 0.9, 0.4]"), {
      timeout: 45_000,
    })
    .toBe(true);
});

test("the properties window shows an extrusion's depth and writes it back", async ({ page }) => {
  // Nothing selected: the window says what it is for.
  await expect(page.getByTestId("properties-empty")).toBeVisible();

  // The tree's operator row points the window at the extrusion.
  await page.locator("[data-testid^=tree-row-profile_0-op-extrude]").click();
  const section = page.getByTestId("properties-feature");
  await expect(section).toHaveAttribute("data-element", "assign:sink");
  await expect(section).toContainText("Extrude");
  await expect(section).toContainText("comb_profile");
  const depth = page.getByTestId("prop-feature-depth");
  await expect(depth).toHaveValue("1.2");
  // The depth is the named parameter fin_depth, and the window says so.
  await expect(section).toContainText("fin_depth");
  await expectValue(page.getByTestId("prop-feature-material"), "aluminum");

  await depth.fill("1.5");
  await depth.press("Enter");
  // set_value follows the name to the parameter's declaration.
  await expect
    .poll(async () => (await editorText(page)).includes('fin_depth = Scalar(1.5, free=True, name="fin_depth")'), {
      timeout: 45_000,
    })
    .toBe(true);
  await waitForCompile(page);
  await expect(page.getByTestId("prop-feature-depth")).toHaveValue("1.5");

  // A material choice goes through assign_material.
  await pickOption(page, "prop-feature-material", "copper");
  await expect
    .poll(async () => (await editorText(page)).includes("sink = extrude(comb_profile, depth=fin_depth, material=copper)"), {
      timeout: 45_000,
    })
    .toBe(true);
});

test("a boolean's blend radius is reachable from the tree and edits the source", async ({ page }) => {
  await page.getByTestId("tree-row-boolean:thermal_body").click();
  const section = page.getByTestId("properties-boolean");
  await expect(section).toContainText("Union");
  await expect(page.getByTestId("prop-boolean-operands")).toHaveText("sink, slug, bush_a, bush_b");
  const smoothness = page.getByTestId("prop-boolean-smoothness");
  await expect(smoothness).toHaveValue("0.03");
  await smoothness.fill("0.05");
  await smoothness.press("Enter");
  await expect
    .poll(async () => (await editorText(page)).includes("thermal_body = Union(sink, slug, bush_a, bush_b, smoothness=0.05)"), {
      timeout: 45_000,
    })
    .toBe(true);
});

test("a face-derived plane is shown but says it cannot be dragged", async ({ page }) => {
  const program = [
    "from cadjoint.construction import PolygonProfile, SketchPlane, extrude",
    "from cadjoint.sdf.boolean import Union",
    "",
    'base = PolygonProfile([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], name="base")',
    "body = extrude(base, depth=0.4)",
    'boss = PolygonProfile([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]], plane=SketchPlane.on(body.cap("+")), name="boss")',
    "scene = Union(body, extrude(boss, depth=0.2))",
    "",
  ].join("\n");
  await recompile(page, program);
  await showAllPlanes(page);
  const labels = page.getByTestId("plane-label");
  await expect(labels).toHaveCount(2);
  const derived = page.locator("[data-testid=plane-label][data-node=profile_1]");
  await expect(derived).toHaveClass(/derived/);
  await derived.click();
  await expect(page.getByTestId("properties-plane-derived")).toBeVisible();
  await expect(page.getByTestId("prop-plane-reference")).toHaveText('SketchPlane.on(body.cap("+"))');
  // No gizmo for a plane the source cannot move: the translate control is
  // still the mode, but the drag would have nothing to write.
  await expect(page.getByTestId("prop-plane-origin-0")).toHaveCount(0);
});
