import { expect, test, type Page } from "@playwright/test";

/**
 * End-to-end coverage of viewer ↔ code parity.
 *
 * The projection below is an independent reimplementation of `src/viewer/math`
 * (same formulas as the preview shader). Duplicating it here is deliberate: if
 * the app's projection drifts, these tests click the wrong pixel and fail,
 * instead of both sides agreeing on something wrong.
 */

const FOV_SCALE = 1.5;
/** Defaults from `Renderer`; the tests never move the camera. */
const CAMERA = { yaw: 0.75, pitch: 0.32, distance: 4.6, target: [0, 0, 0] as const };
const FRONT_CAMERA = { ...CAMERA, yaw: 0, pitch: 0 };

type Vec3 = [number, number, number];
type CameraSpec = typeof CAMERA;

const sub = (a: readonly number[], b: readonly number[]): Vec3 =>
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a: Vec3): Vec3 => {
  const n = Math.hypot(...a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};

function cameraPosition(camera: CameraSpec = CAMERA): Vec3 {
  const cp = Math.cos(camera.pitch);
  return [
    camera.target[0] + camera.distance * cp * Math.sin(camera.yaw),
    camera.target[1] + camera.distance * Math.sin(camera.pitch),
    camera.target[2] + camera.distance * cp * Math.cos(camera.yaw),
  ];
}

/** Project a world point to CSS pixels inside the canvas element. */
function projectToCss(
  world: Vec3,
  canvas: { width: number; height: number; clientWidth: number; clientHeight: number },
  camera: CameraSpec = CAMERA,
  projection: "perspective" | "orthographic" = "orthographic",
) {
  const position = cameraPosition(camera);
  const forward = norm(sub(camera.target, position));
  const right = norm(cross(forward, [0, 1, 0]));
  const up = cross(right, forward);
  const delta = sub(world, position);
  const viewDepth = dot(delta, forward);
  const aspect = canvas.width / canvas.height;
  const divisor =
    FOV_SCALE * (projection === "orthographic" ? camera.distance : viewDepth);
  const u = dot(delta, right) / divisor;
  const v = dot(delta, up) / divisor;
  const px = (u / aspect + 0.5) * canvas.width;
  const py = (0.5 - v) * canvas.height;
  // Framebuffer pixels back to CSS pixels for Playwright's mouse.
  return {
    x: (px * canvas.clientWidth) / canvas.width,
    y: (py * canvas.clientHeight) / canvas.height,
  };
}

async function canvasMetrics(page: Page) {
  return page.evaluate(() => {
    const canvas = document.querySelector<HTMLCanvasElement>("[data-testid=viewer-canvas]")!;
    const rect = canvas.getBoundingClientRect();
    return {
      width: canvas.width,
      height: canvas.height,
      clientWidth: canvas.clientWidth,
      clientHeight: canvas.clientHeight,
      left: rect.left,
      top: rect.top,
    };
  });
}

/** The example sketch's first vertex, in world space (plane is the world XY plane). */
const FIRST_VERTEX: Vec3 = [-1.1, -0.7, 0];
const ROOF_PEAK: Vec3 = [0, 1, 0];

async function editorText(page: Page): Promise<string> {
  return page.locator("[data-testid=editor] .cm-content").innerText();
}

async function waitForCompile(page: Page) {
  // "Run" is enabled before the first compile starts, so wait on the status
  // leaving the compiling state instead.
  await expect(page.getByTestId("status")).not.toContainText("compiling", { timeout: 60_000 });
  await expect(page.getByTestId("run")).toBeEnabled({ timeout: 60_000 });
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await waitForCompile(page);
  // A machine without WebGPU shows a dismissable banner over the canvas.
  const dismiss = page.getByRole("button", { name: "Dismiss" });
  if (await dismiss.isVisible().catch(() => false)) await dismiss.click();
});

test("serves the app and loads the starter sketch", async ({ page }) => {
  await expect(page).toHaveTitle(/JAXCAD/);
  expect(await editorText(page)).toContain("PolygonProfile(");
  expect(await editorText(page)).toContain("[-1.1, -0.7]");
  expect(await editorText(page)).toContain("ring = revolve(ring_profile");
  expect(await editorText(page)).toContain("jax.value_and_grad(clearance_metric)");
});

test("the starter exposes its live autodiff proof without an open panel", async ({
  page,
}) => {
  await expect(page.getByTestId("autodiff-toggle")).toContainText("AD live");
  await expect(page.getByTestId("autodiff-toggle")).toContainText("6");
  await expect(page.getByTestId("autodiff-panel")).toHaveCount(0);

  await page.getByTestId("autodiff-toggle").click();
  const panel = page.getByTestId("autodiff-panel");
  await expect(panel).toBeVisible();
  await expect(panel).toContainText("Profile -> Extrude -> SDF");
  await expect(panel).toContainText("two-probe clearance");
  await expect(panel).toContainText("eave_right.x");
  await expect(panel).toContainText("-0.700");
  await expect(panel).toContainText("body_depth");
  await expect(panel).toContainText("-0.500");

  await page.getByTestId("autodiff-toggle").click();
  await expect(panel).toHaveCount(0);
});

test("the revolved starter object exposes its profile and operator", async ({ page }) => {
  const metrics = await canvasMetrics(page);
  const sectionEdge = projectToCss([0.76, 1.49, 0.15], metrics);
  await page.mouse.click(
    metrics.left + sectionEdge.x,
    metrics.top + sectionEdge.y,
  );

  await expect(page.getByTestId("selection-chip")).toHaveText("revolve section");
  await expect(page.getByTestId("sketch-panel")).toContainText("revolve");
  await expect(page.getByTestId("gizmo-translate")).toHaveClass(/active/);
  await expect(page.getByTestId("rail-hint")).toHaveText("move");
});

test("sketch constraints are drawn over the viewport", async ({ page }) => {
  await expect(page.getByTestId("constraint-overlay")).toBeVisible();
  await expect(page.getByTestId("constraint-fixed-overlay")).toHaveCount(2);
  await expect(page.getByTestId("constraint-distance-overlay")).toHaveCount(3);
  await expect(
    page.locator('[data-testid="constraint-distance-overlay"][data-scope="object"]'),
  ).toContainText("3.8");
  await expect(page.getByTestId("constraint-overlay")).toContainText("2.2");
  await expect(page.getByTestId("constraint-overlay")).toContainText("1");

  await page.getByTestId("view-front").click({ force: true });
  await expect(page.getByTestId("constraint-fixed-overlay")).toHaveCount(2);
  await expect(page.getByTestId("constraint-distance-overlay")).toHaveCount(3);

  await page.getByTestId("display-options").click();
  await page.getByTestId("render-customize").click();
  await page.getByTestId("toggle-constraints").uncheck();
  await expect(page.getByTestId("constraint-overlay")).toHaveCount(0);
  await expect(page.getByTestId("toggle-fixed-constraints")).toBeDisabled();

  await page.getByTestId("toggle-constraints").check();
  await page.getByTestId("toggle-fixed-constraints").uncheck();
  await expect(page.getByTestId("constraint-fixed-overlay")).toHaveCount(0);
  await expect(page.getByTestId("constraint-distance-overlay")).toHaveCount(3);

  await page.getByTestId("toggle-constraint-values").uncheck();
  await expect(page.getByTestId("constraint-overlay")).not.toContainText("2.2");
  await page.getByTestId("toggle-distance-constraints").uncheck();
  await expect(page.getByTestId("constraint-distance-overlay")).toHaveCount(0);
});

test("compilation uses a non-blocking viewport progress indicator", async ({ page }) => {
  await page.getByTestId("run").click();
  await expect(page.getByTestId("viewer-compiling")).toBeVisible();
  await expect(page.getByTestId("viewer-compiling")).toHaveCSS(
    "pointer-events",
    "none",
  );
  expect(
    await page
      .getByTestId("viewer-canvas")
      .evaluate((canvas) => getComputedStyle(canvas).cursor),
  ).not.toBe("progress");
  await waitForCompile(page);
  await expect(page.getByTestId("viewer-compiling")).toHaveCount(0);
});

test("clicking a sketch handle selects it and highlights its source span", async ({ page }) => {
  await page.getByTestId("mode-vertex").click();
  const metrics = await canvasMetrics(page);
  const point = projectToCss(FIRST_VERTEX, metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);

  await expect(page.getByTestId("selection-chip")).toHaveText("vertex 0");
  // The editor marks exactly the literal that defines that vertex.
  await expect(page.locator(".cm-vertex-highlight")).toHaveText("[-1.1, -0.7]");
});

test("escape clears the selection", async ({ page }) => {
  await page.getByTestId("mode-vertex").click();
  const metrics = await canvasMetrics(page);
  const point = projectToCss(FIRST_VERTEX, metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);
  await expect(page.getByTestId("selection-chip")).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(page.getByTestId("selection-chip")).toHaveCount(0);
});

test("dragging a handle rewrites the vertex literal", async ({ page }) => {
  await page.getByTestId("mode-vertex").click();
  const metrics = await canvasMetrics(page);
  // Use an unconstrained point: the fixed base corner is intentionally driven
  // back to its target by the starter constraint system.
  const from = projectToCss(ROOF_PEAK, metrics);
  const to = projectToCss([0.2, 1.35, 0], metrics);

  await page.mouse.move(metrics.left + from.x, metrics.top + from.y);
  await page.mouse.down();
  await page.mouse.move(metrics.left + to.x, metrics.top + to.y, { steps: 12 });
  await page.mouse.up();

  await expect(page.locator("[data-testid=editor] .cm-content")).not.toContainText("[0.0, 1.0]");
  const text = await editorText(page);
  const match = text.match(
    /roof_peak = Vector2\(value=\[(-?[\d.]+), (-?[\d.]+)\]/,
  );
  expect(match).not.toBeNull();
  expect(Number(match![1])).toBeGreaterThan(0);
  expect(Number(match![2])).toBeGreaterThan(1);
  await waitForCompile(page);
});

/** Count `[x, y]` literals, allowing any numeric formatting. */
const NUMBER = String.raw`-?[\d.]+(?:e[-+]?\d+)?`;
const vertexLiteralCount = async (page: Page) =>
  ((await editorText(page)).match(new RegExp(`\\[${NUMBER}, ${NUMBER}\\]`, "g")) ?? []).length;

test("the polygon tool inserts a vertex and stays active", async ({ page }) => {
  const before = await vertexLiteralCount(page);

  await page.getByTestId("tool-polygon").click();
  const metrics = await canvasMetrics(page);
  // Midpoint of the sketch's bottom edge.
  const point = projectToCss([0, -0.7, 0], metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);

  // The patch and its recompile are async; poll rather than racing them.
  await expect.poll(() => vertexLiteralCount(page), { timeout: 45_000 }).toBe(before + 1);
  // Unlike the old one-shot button, the tool keeps going for the next click.
  await expect(page.getByTestId("tool-polygon")).toHaveClass(/active/);
  await waitForCompile(page);

  await page.keyboard.press("Escape");
  await expect(page.getByTestId("mode-object")).toHaveClass(/active/);
});

/** The projection toggle shows one icon per mode. */
const projectionMode = async (page: Page) =>
  page.getByTestId("projection-toggle").getAttribute("title");

test("the view cube snaps to a face and switches projection", async ({ page }) => {
  expect(await projectionMode(page)).toContain("Orthographic");

  await page.getByTestId("view-front").click({ force: true });
  expect(await projectionMode(page)).toContain("Orthographic");

  // Iso returns to a perspective camera.
  await page.getByTestId("view-iso").click();
  expect(await projectionMode(page)).toContain("Perspective");
});

test("dragging the view cube orbits the camera", async ({ page }) => {
  const cube = page.locator(".view-cube .cube");
  const before = await cube.evaluate((node) => getComputedStyle(node).transform);

  const stage = await page.locator(".cube-stage").boundingBox();
  const centre = { x: stage!.x + stage!.width / 2, y: stage!.y + stage!.height / 2 };
  await page.mouse.move(centre.x, centre.y);
  await page.mouse.down();
  await page.mouse.move(centre.x - 40, centre.y + 10, { steps: 8 });
  await page.mouse.up();

  await expect
    .poll(() => cube.evaluate((node) => getComputedStyle(node).transform))
    .not.toBe(before);
  // A drag must not be mistaken for a face click, which would snap to a view.
  expect(await projectionMode(page)).toContain("Orthographic");
});

test("the view cube tracks the camera", async ({ page }) => {
  const cube = page.locator(".view-cube .cube");
  const before = await cube.evaluate((node) => getComputedStyle(node).transform);

  // Orbit by dragging empty space well clear of the sketch and the overlays.
  const metrics = await canvasMetrics(page);
  await page.mouse.move(metrics.left + metrics.clientWidth * 0.8, metrics.top + metrics.clientHeight * 0.85);
  await page.mouse.down();
  await page.mouse.move(
    metrics.left + metrics.clientWidth * 0.55,
    metrics.top + metrics.clientHeight * 0.85,
    { steps: 10 },
  );
  await page.mouse.up();

  await expect
    .poll(() => cube.evaluate((node) => getComputedStyle(node).transform))
    .not.toBe(before);
});

test("the projection toggle works on its own", async ({ page }) => {
  await page.getByTestId("projection-toggle").click();
  expect(await projectionMode(page)).toContain("Perspective");
  await page.getByTestId("projection-toggle").click();
  expect(await projectionMode(page)).toContain("Orthographic");
});

test("object and gizmo picking use the orthographic camera", async ({ page }) => {
  const program = [
    "from jaxcad.construction import Solid",
    'block = Solid.box(size=[0.5, 0.5, 0.5], position=[0, 0, 1.5], name="block")',
    "scene = block",
    "",
  ].join("\n");
  const editor = page.locator("[data-testid=editor] .cm-content");
  await editor.click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText(program);
  await page.getByTestId("run").click();
  await waitForCompile(page);

  await page.getByTestId("view-front").click({ force: true });
  expect(await projectionMode(page)).toContain("Orthographic");

  const metrics = await canvasMetrics(page);
  const outline = projectToCss([0.5, 0, 1.5], metrics, FRONT_CAMERA, "orthographic");
  await page.mouse.click(metrics.left + outline.x, metrics.top + outline.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("block");

  const handle = gizmoTip([0, 0, 1.5], 0, metrics, FRONT_CAMERA, "orthographic");
  const destination = projectToCss([1.2, 0, 1.5], metrics, FRONT_CAMERA, "orthographic");
  await page.mouse.move(metrics.left + handle.x, metrics.top + handle.y);
  await page.mouse.down();
  await page.mouse.move(metrics.left + destination.x, metrics.top + destination.y, {
    steps: 12,
  });
  await page.mouse.up();

  await expect
    .poll(() => editorText(page), { timeout: 45_000 })
    .toMatch(/position=\[(?:0\.)?[1-9][\d.]*, 0, 1.5\]/);
  await waitForCompile(page);
});

test("a solid can be deleted from the viewer", async ({ page }) => {
  const metrics = await canvasMetrics(page);
  // The constraint system projects the starter sphere to this runtime position.
  const outline = projectToCss([1.8726, -0.32795, 0.82205], metrics);
  await page.mouse.click(metrics.left + outline.x, metrics.top + outline.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("glass");

  await page.getByTestId("delete-selection").click();
  await expect
    .poll(async () => (await editorText(page)).includes("glass = Solid.sphere"), {
      timeout: 45_000,
    })
    .toBe(false);
  await waitForCompile(page);
  await expect(page.getByTestId("status")).not.toContainText("failed");
});

test("selection mode decides what a click picks", async ({ page }) => {
  const metrics = await canvasMetrics(page);
  const point = projectToCss(FIRST_VERTEX, metrics);

  // Object mode is the default, so a sketch vertex is not picked as a vertex.
  await expect(page.getByTestId("mode-object")).toHaveClass(/active/);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);
  await expect(page.locator(".cm-vertex-highlight")).toHaveCount(0);

  await page.getByTestId("mode-vertex").click();
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("vertex 0");
  await expect(page.locator(".cm-vertex-highlight")).toHaveText("[-1.1, -0.7]");

  // Switching back to object selection promotes the point to its polygon and
  // reveals the polygon's translation gizmo without requiring another click.
  await page.getByTestId("mode-object").click();
  await expect(page.getByTestId("selection-chip")).toHaveText("house");
  await expect(page.getByTestId("gizmo-translate")).toHaveClass(/active/);
});

test("render presets activate, edit, and persist without bloating the closed UI", async ({
  page,
}) => {
  const before = await editorText(page);
  await expect(page.locator(".render-preset-card")).toHaveCount(0);
  await page.getByTestId("display-options").click();

  await expect(page.locator(".render-preset-card")).toHaveCount(3);
  await expect(page.getByTestId("render-preset-editor")).toHaveCount(0);
  await expect(page.getByTestId("render-preset-xray")).toHaveClass(/active/);
  const compactPanel = await page.locator(".display-options .popover").boundingBox();
  expect(compactPanel!.width).toBeLessThanOrEqual(310);
  expect(compactPanel!.height).toBeLessThan(260);

  await page.getByTestId("render-preset-studio").click();
  await expect(page.getByTestId("render-preset-studio")).toHaveClass(/active/);
  expect(await projectionMode(page)).toContain("Perspective");

  await page.getByTestId("render-customize").click();
  await expect(page.getByTestId("render-preset-editor")).toBeVisible();
  await expect(page.getByTestId("shadows-soft")).toHaveClass(/active/);
  await expect(page.getByTestId("shading-full")).toHaveClass(/active/);
  await expect(page.getByTestId("toggle-path-tracing")).toBeChecked();
  await expect(page.getByTestId("toggle-xray")).not.toBeChecked();
  await expect(page.getByTestId("render-preset-save")).toBeDisabled();

  await page.getByTestId("shadows-off").click();
  await expect(page.getByTestId("shadows-off")).toHaveClass(/active/);
  await page.getByTestId("quality-draft").click();
  await expect(page.getByTestId("quality-draft")).toHaveClass(/active/);
  await page.getByTestId("toggle-xray").check();
  await expect(page.getByTestId("render-preset-save")).toBeEnabled();
  await page.getByTestId("render-preset-save").click();
  await expect(page.getByTestId("render-preset-studio")).toHaveClass(/active/);

  await page.reload();
  await waitForCompile(page);
  const dismiss = page.getByRole("button", { name: "Dismiss" });
  if (await dismiss.isVisible().catch(() => false)) await dismiss.click();
  await page.getByTestId("display-options").click();
  await expect(page.getByTestId("render-preset-studio")).toHaveClass(/active/);
  await page.getByTestId("render-customize").click();
  await expect(page.getByTestId("shadows-off")).toHaveClass(/active/);
  await expect(page.getByTestId("quality-draft")).toHaveClass(/active/);
  await expect(page.getByTestId("toggle-path-tracing")).toBeChecked();
  await expect(page.getByTestId("toggle-xray")).toBeChecked();

  await page.getByTestId("render-preset-reset").click();
  await expect(page.getByTestId("shadows-soft")).toHaveClass(/active/);
  await expect(page.getByTestId("quality-high")).toHaveClass(/active/);
  await expect(page.getByTestId("toggle-path-tracing")).toBeChecked();
  await expect(page.getByTestId("toggle-xray")).not.toBeChecked();

  // Render settings are viewer state, never edits to the program.
  expect(await editorText(page)).toBe(before);
});

test("the source-code pane stays above floating panels", async ({ page }) => {
  await page.getByTestId("display-options").click();
  const popover = page.locator(".display-options .popover");
  await expect(popover).toBeVisible();

  const editorZ = await page
    .locator(".editor-pane")
    .evaluate((node) => Number.parseInt(getComputedStyle(node).zIndex, 10));
  const popoverZ = await popover.evaluate((node) =>
    Number.parseInt(getComputedStyle(node).zIndex, 10),
  );
  expect(editorZ).toBeGreaterThan(popoverZ);
});

test("editing the code updates the sketch the viewer reports", async ({ page }) => {
  await page.getByTestId("mode-vertex").click();
  const program = [
    "from jaxcad.construction import PolygonProfile, extrude",
    'profile = PolygonProfile([[-2.2, -0.9], [1.0, -0.9], [0.0, 1.2]], name="t")',
    "scene = extrude(profile, depth=0.5)",
    "",
  ].join("\n");

  await page.locator("[data-testid=editor] .cm-content").click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText(program);
  await expect(page.locator("[data-testid=editor] .cm-content")).toContainText("[-2.2, -0.9]");

  await page.getByTestId("run").click();
  await waitForCompile(page);

  // The viewer now knows about the sketch the edited code describes.
  const metrics = await canvasMetrics(page);
  const point = projectToCss([-2.2, -0.9, 0], metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("vertex 0");
  await expect(page.locator(".cm-vertex-highlight")).toHaveText("[-2.2, -0.9]");
});

/** Where the gizmo's X arrow tip lands for a primitive at `origin`. */
function gizmoTip(
  origin: Vec3,
  axis: 0 | 1 | 2,
  canvas: Parameters<typeof projectToCss>[1],
  camera: CameraSpec = CAMERA,
  projection: "perspective" | "orthographic" = "orthographic",
) {
  const position = cameraPosition(camera);
  const size = Math.max(0.15, 0.18 * Math.hypot(...sub(origin, position)));
  const unit: Vec3 = [0, 0, 0];
  unit[axis] = size * 0.6;
  return projectToCss(
    [origin[0] + unit[0], origin[1] + unit[1], origin[2] + unit[2]],
    canvas,
    camera,
    projection,
  );
}

const sphereCount = async (page: Page) =>
  ((await editorText(page)).match(/Solid\.sphere/g) ?? []).length;

test("placing a primitive writes a Solid call into the source", async ({ page }) => {
  // The starter program already has one sphere, so count rather than presence.
  const before = await sphereCount(page);

  await page.getByTestId("tool-sphere").click();
  const metrics = await canvasMetrics(page);
  // Somewhere on the world XY plane, clear of the existing sketch.
  const point = projectToCss([2.4, 1.4, 0], metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);

  await expect.poll(() => sphereCount(page), { timeout: 45_000 }).toBe(before + 1);
  // Solid is already imported by the starter program, so no duplicate appears.
  expect(await editorText(page)).toMatch(/from jaxcad\.construction import .*\bSolid\b/);

  // CodeMirror only renders the lines in view, so scroll to the scene
  // assignment before checking the new solid was wired into it rather than
  // left as an orphan statement.
  await page.locator("[data-testid=editor] .cm-scroller").evaluate((node) => {
    node.scrollTop = node.scrollHeight;
  });
  await expect(page.locator("[data-testid=editor] .cm-content")).toContainText(
    /scene = Union\(|sphere1,/,
  );
  await expect(page.locator("[data-testid=editor] .cm-content")).toContainText("sphere1");
  await waitForCompile(page);
  await expect(page.getByTestId("status")).not.toContainText("failed");

  // The tool reverts to picking, with the new solid already selected so the
  // gizmo is usable straight away.
  await expect(page.getByTestId("mode-object")).toHaveClass(/active/);
  await expect(page.getByTestId("selection-chip")).toBeVisible();
});

test("the material browser creates, edits, and drag-assigns materials", async ({ page }) => {
  await expect(page.getByTestId("material-panel")).toHaveCount(0);
  const materialButton = await page.getByTestId("material-open").boundingBox();
  const projectionButton = await page.getByTestId("projection-toggle").boundingBox();
  expect(materialButton).not.toBeNull();
  expect(projectionButton).not.toBeNull();
  expect(materialButton!.y).toBeGreaterThan(
    projectionButton!.y + projectionButton!.height + 6,
  );
  await page.getByTestId("material-open").click();
  await expect(page.getByTestId("material-panel")).toBeVisible();
  await expect(page.getByTestId("material-clay")).toBeVisible();
  await expect(page.getByTestId("material-brass")).toBeVisible();

  await page.getByTestId("material-add").click();
  await expect
    .poll(() => editorText(page), { timeout: 45_000 })
    .toContain("material1 = Material(");
  await waitForCompile(page);

  await page.getByTestId("material-material1").click();
  await page.getByTestId("material-roughness").evaluate((input: HTMLInputElement) => {
    input.value = "0.65";
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect
    .poll(() => editorText(page), { timeout: 45_000 })
    .toContain("roughness=0.65");
  await waitForCompile(page);

  const metrics = await canvasMetrics(page);
  const metal = projectToCss([-1.9, -0.65, 0], metrics);
  await page.getByTestId("material-glass_material").dragTo(
    page.getByTestId("viewer-canvas"),
    { targetPosition: { x: metal.x, y: metal.y } },
  );
  await expect
    .poll(() => editorText(page), { timeout: 45_000 })
    .toMatch(/metal = Solid\.cylinder\([\s\S]*?material=glass_material/);
  await waitForCompile(page);

  await page.getByTestId("material-close").click();
  await expect(page.getByTestId("material-panel")).toHaveCount(0);
  await expect(page.getByTestId("material-open")).toBeVisible();
});

test("a placed primitive can be selected and moved along an axis", async ({ page }) => {
  await page.getByTestId("tool-box").click();
  const metrics = await canvasMetrics(page);
  // Kept near the middle of the view so the gizmo arrows stay on the canvas.
  const drop = projectToCss([1.6, 0, 0], metrics);
  await page.mouse.click(metrics.left + drop.x, metrics.top + drop.y);
  await expect
    .poll(async () => (await editorText(page)).includes("Solid.box"), { timeout: 45_000 })
    .toBe(true);
  await waitForCompile(page);

  // Read back where it actually landed, then click its wireframe to select it.
  const placed = (await editorText(page)).match(/Solid\.box\([^)]*position=\[([^\]]+)\]/);
  expect(placed).not.toBeNull();
  const origin = placed![1].split(",").map(Number) as Vec3;

  // Re-read the canvas: the panes resize as the program grows, and projecting
  // with stale metrics aims at the wrong pixel.
  const afterPlacing = await canvasMetrics(page);
  // Midpoint of a top edge: clear of the sketch's own vertex handles, which
  // would otherwise win the pick.
  const edge = projectToCss([origin[0], origin[1] + 0.5, origin[2] + 0.5], afterPlacing);
  await page.mouse.click(afterPlacing.left + edge.x, afterPlacing.top + edge.y);
  await expect(page.getByTestId("selection-chip")).toBeVisible();
  await expect(page.getByTestId("gizmo-translate")).toBeEnabled();
  // A whole-object selection offers the move/rotate gizmo.
  await expect(page.getByTestId("gizmo-translate")).toBeVisible();

  // Drag the Y arrow upward; the vertical axis stays comfortably in frame.
  const view = await canvasMetrics(page);
  const from = gizmoTip(origin, 1, view);
  const to = projectToCss([origin[0], origin[1] + 1.0, origin[2]], view);
  await page.mouse.move(view.left + from.x, view.top + from.y);
  await page.mouse.down();
  await page.mouse.move(view.left + to.x, view.top + to.y, { steps: 12 });
  await page.mouse.up();

  await expect
    .poll(
      async () => {
        const match = (await editorText(page)).match(/Solid\.box\([^)]*position=\[([^\]]+)\]/);
        return match ? Number(match[1].split(",")[1]) : origin[1];
      },
      { timeout: 45_000 },
    )
    .toBeGreaterThan(origin[1] + 0.2);
});

test("a sketch can be moved by its plane", async ({ page }) => {
  // Selecting the sketch as an object targets its plane, which is where a
  // profile's placement actually lives.
  const metrics = await canvasMetrics(page);
  const edge = projectToCss([0, -0.7, 0], metrics);
  await page.mouse.click(metrics.left + edge.x, metrics.top + edge.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("house");
  await expect(page.getByTestId("gizmo-translate")).toBeEnabled();
  await expect(page.getByTestId("gizmo-translate")).toHaveClass(/active/);

  // The arrows are centered on the polygon, not a potentially remote plane origin.
  const view = await canvasMetrics(page);
  const from = gizmoTip([0, 0.04, 0], 1, view);
  const to = projectToCss([0, 1.0, 0], view);
  await page.mouse.move(view.left + from.x, view.top + from.y);
  await page.mouse.down();
  await page.mouse.move(view.left + to.x, view.top + to.y, { steps: 12 });
  await page.mouse.up();

  // The plane's origin is what gets rewritten, not the vertices.
  await expect
    .poll(
      async () => {
        const match = (await editorText(page)).match(/SketchPlane\(origin=\[([^\]]+)\]/);
        return match ? Number(match[1].split(",")[1]) : 0;
      },
      { timeout: 45_000 },
    )
    .toBeGreaterThan(0.2);
  expect(await editorText(page)).toContain("[-1.1, -0.7]");
});

test("a default polygon can move without losing parameter-backed points", async ({ page }) => {
  const program = [
    "from jaxcad.construction import PolygonProfile, extrude",
    "from jaxcad.geometry import Vector2",
    "p0 = Vector2(value=[-1.0, -0.6], free=True, name='p0')",
    "p1 = Vector2(value=[1.0, -0.6], free=True, name='p1')",
    "p2 = Vector2(value=[0.0, 0.9], free=True, name='p2')",
    "points = [p0, p1, p2]",
    "profile = PolygonProfile(points, name='parameterized')",
    "scene = extrude(profile, depth=0.5)",
    "",
  ].join("\n");
  const editor = page.locator("[data-testid=editor] .cm-content");
  await editor.click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText(program);
  await page.getByTestId("run").click();
  await waitForCompile(page);

  const metrics = await canvasMetrics(page);
  const edge = projectToCss([0, -0.6, 0], metrics);
  await page.mouse.click(metrics.left + edge.x, metrics.top + edge.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("parameterized");
  await expect(page.getByTestId("gizmo-translate")).toBeEnabled();

  const view = await canvasMetrics(page);
  const from = gizmoTip([0, -0.1, 0], 1, view);
  const to = projectToCss([0, 0.9, 0], view);
  await page.mouse.move(view.left + from.x, view.top + from.y);
  await page.mouse.down();
  await page.mouse.move(view.left + to.x, view.top + to.y, { steps: 12 });
  await page.mouse.up();

  await expect
    .poll(async () => (await editorText(page)).includes("SketchPlane(origin="), {
      timeout: 45_000,
    })
    .toBe(true);
  const text = await editorText(page);
  expect(text).toContain("PolygonProfile(points");
  expect(text).toContain("p0 = Vector2(value=[-1.0, -0.6]");
});

test("a primitive can be scaled along an axis", async ({ page }) => {
  await page.getByTestId("tool-box").click();
  let metrics = await canvasMetrics(page);
  const drop = projectToCss([1.5, 0.6, 0], metrics);
  await page.mouse.click(metrics.left + drop.x, metrics.top + drop.y);
  await waitForCompile(page);
  await expect(page.getByTestId("gizmo-scale")).toBeEnabled();
  await page.getByTestId("gizmo-scale").click();

  const placed = (await editorText(page)).match(
    /Solid\.box\(size=\[([^\]]+)\][^)]*position=\[([^\]]+)\]/,
  );
  expect(placed).not.toBeNull();
  const before = placed![1].split(",").map(Number);
  const origin = placed![2].split(",").map(Number) as Vec3;
  metrics = await canvasMetrics(page);
  const from = gizmoTip(origin, 1, metrics);
  const to = projectToCss([origin[0], origin[1] + 1.1, origin[2]], metrics);
  await page.mouse.move(metrics.left + from.x, metrics.top + from.y);
  await page.mouse.down();
  await page.mouse.move(metrics.left + to.x, metrics.top + to.y, { steps: 12 });
  await page.mouse.up();

  await expect
    .poll(
      async () => {
        const match = (await editorText(page)).match(/Solid\.box\(size=\[([^\]]+)\]/);
        return match ? Number(match[1].split(",")[1]) : before[1];
      },
      { timeout: 45_000 },
    )
    .toBeGreaterThan(before[1]);
});

test("sketch constraints and extrusion are represented in UI and code", async ({ page }) => {
  await page.getByTestId("tool-sketch").click();
  let metrics = await canvasMetrics(page);
  const origin: Vec3 = [1.0, 1.4, 0];
  const drop = projectToCss(origin, metrics);
  await page.mouse.click(metrics.left + drop.x, metrics.top + drop.y);
  await waitForCompile(page);
  await expect(page.getByTestId("sketch-panel")).toBeVisible();
  await expect(page.getByTestId("selection-chip")).toContainText("sketch");

  await page.getByTestId("mode-vertex").click();
  metrics = await canvasMetrics(page);
  const first = projectToCss([origin[0] - 0.6, origin[1] - 0.6, 0], metrics);
  const second = projectToCss([origin[0] + 0.6, origin[1] - 0.6, 0], metrics);
  await page.mouse.click(metrics.left + first.x, metrics.top + first.y);
  await page.getByTestId("constraint-fix").click();
  await waitForCompile(page);
  await expect(page.getByTestId("sketch-panel")).toContainText("fix · P1");

  await page.getByTestId("constraint-distance").click();
  await expect(page.getByTestId("status")).toContainText("choose the second point");
  metrics = await canvasMetrics(page);
  await page.mouse.click(metrics.left + second.x, metrics.top + second.y);
  await expect
    .poll(async () => (await editorText(page)).includes("DistanceConstraint("), {
      timeout: 45_000,
    })
    .toBe(true);
  await expect(page.getByTestId("sketch-panel")).toContainText("distance · P1–P2");
  await waitForCompile(page);

  // A relational constraint uses the same two-click flow: select the first
  // point, arm the tool (which adopts the selection), then pick the second.
  metrics = await canvasMetrics(page);
  await page.mouse.click(metrics.left + first.x, metrics.top + first.y);
  await page.getByTestId("constraint-horizontal").click();
  await page.mouse.click(metrics.left + second.x, metrics.top + second.y);
  await expect(page.getByTestId("sketch-panel")).toContainText("horiz · P1–P2", {
    timeout: 45_000,
  });
  await waitForCompile(page);

  await expect(page.getByTestId("solver-panel")).toHaveCount(0);
  await page.getByTestId("solver-toggle").click();
  await expect(page.getByTestId("solver-panel")).toBeVisible();
  await page.getByTestId("solver-method").selectOption("adam");
  await page.getByTestId("solver-iterations").fill("24");
  await expect(page.getByTestId("solver-loss-chart")).toHaveCount(0);
  await page.getByTestId("constraint-solve").click();
  await expect
    .poll(
      async () =>
        (await editorText(page)).includes(
          "satisfy_constraints(sketch1, method='adam', steps=24)",
        ),
      { timeout: 45_000 },
    )
    .toBe(true);
  await waitForCompile(page);
  await expect(page.getByTestId("solver-loss-chart")).toBeVisible();
  await expect(page.getByTestId("solver-panel")).toContainText(
    "adam · 24 iterations",
  );
  await expect(
    page.getByTestId("solver-loss-chart").locator("polyline"),
  ).not.toHaveAttribute("points", "");

  await page.getByTestId("sketch-extrude").click();
  await expect
    .poll(async () => (await editorText(page)).includes("_body = extrude("), {
      timeout: 45_000,
    })
    .toBe(true);
  await waitForCompile(page);

  const text = await editorText(page);
  expect(text).toContain("FixedConstraint(");
  expect(text).toContain("DistanceConstraint(");
  expect(text).toContain("satisfy_constraints(sketch1, method='adam', steps=24)");
  expect(text).toContain("_body = extrude(");
  await expect(page.getByTestId("sketch-extrude")).toBeDisabled();
  // An operator now exists, so the sketch cannot also be revolved.
  await expect(page.getByTestId("sketch-revolve")).toBeDisabled();

  // The distance chip (serialized index 1: fix, distance, horizontal) edits
  // its numeric target in place.
  await page.getByTestId("constraint-label-1").click();
  await page.getByTestId("constraint-value-1").fill("1.2345");
  await page.getByTestId("constraint-value-1").press("Enter");
  await expect.poll(() => editorText(page), { timeout: 45_000 }).toContain("1.2345");
  await waitForCompile(page);

  // Chips delete their constraint by serialized index.
  await page.getByTestId("constraint-delete-0").click();
  await expect
    .poll(async () => (await editorText(page)).includes("FixedConstraint("), {
      timeout: 45_000,
    })
    .toBe(false);
  await waitForCompile(page);
  await expect(page.getByTestId("status")).not.toContainText("failed");
});

test("relational constraint chips render for API-added kinds", async ({ page }) => {
  const program = [
    "from jaxcad.construction import PolygonProfile, extrude",
    'profile = PolygonProfile([[-1.0, -0.8], [1.0, -0.8], [1.0, 0.8], [-1.0, 0.8]], name="quad")',
    "scene = extrude(profile, depth=0.4)",
    "",
  ].join("\n");
  // Drive /patch directly — the same API the app uses — to cover the kinds the
  // starter program never adds, then load the accumulated source into the app.
  const patched = await page.evaluate(async (source) => {
    const session = (await (await fetch("/api/session")).json()) as { token: string };
    let text = source;
    const operations = [
      { op: "add_constraint", line: 2, kind: "horizontal", indices: [0, 1] },
      { op: "add_constraint", line: 2, kind: "vertical", indices: [1, 2] },
      { op: "add_constraint", line: 2, kind: "coincident", indices: [0, 3] },
      { op: "add_constraint", line: 2, kind: "parallel", indices: [0, 1, 2, 3] },
      { op: "add_constraint", line: 2, kind: "perpendicular", indices: [0, 1, 1, 2] },
    ];
    for (const operation of operations) {
      const response = await fetch("/patch", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Jaxcad-Token": session.token,
        },
        body: JSON.stringify({ source: text, ...operation }),
      });
      const result = (await response.json()) as {
        ok: boolean;
        source?: string;
        error?: string;
      };
      if (!result.ok || !result.source) {
        return { error: result.error ?? "patch failed" };
      }
      text = result.source;
    }
    return { source: text };
  }, program);
  expect(patched.error).toBeUndefined();

  const editor = page.locator("[data-testid=editor] .cm-content");
  await editor.click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText(patched.source!);
  await page.getByTestId("run").click();
  await waitForCompile(page);

  await page.getByTestId("mode-vertex").click();
  const metrics = await canvasMetrics(page);
  const point = projectToCss([-1.0, -0.8, 0], metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);
  const panel = page.getByTestId("sketch-panel");
  await expect(panel).toBeVisible();
  await expect(panel).toContainText("horiz · P1–P2");
  await expect(panel).toContainText("vert · P2–P3");
  await expect(panel).toContainText("coinc · P1–P4");
  await expect(panel).toContainText("∥ · P1P2–P3P4");
  await expect(panel).toContainText("⊥ · P1P2–P2P3");
  // Every chip carries a delete affordance keyed by serialized index.
  await expect(page.getByTestId("constraint-delete-0")).toBeVisible();
  // This sketch already drives an extrusion, so Revolve is unavailable.
  await expect(page.getByTestId("sketch-revolve")).toBeDisabled();
});

test("a standalone sketch can be revolved from the panel", async ({ page }) => {
  await page.getByTestId("tool-sketch").click();
  const metrics = await canvasMetrics(page);
  const drop = projectToCss([1.0, 1.4, 0], metrics);
  await page.mouse.click(metrics.left + drop.x, metrics.top + drop.y);
  await waitForCompile(page);
  await expect(page.getByTestId("sketch-panel")).toBeVisible();

  await expect(page.getByTestId("sketch-revolve")).toBeEnabled();
  await page.getByTestId("sketch-revolve").click();
  await expect
    .poll(async () => (await editorText(page)).includes("_body = revolve("), {
      timeout: 45_000,
    })
    .toBe(true);
  await waitForCompile(page);
  await expect(page.getByTestId("status")).not.toContainText("failed");
  await expect(page.getByTestId("sketch-revolve")).toBeDisabled();
  await expect(page.getByTestId("sketch-panel")).toContainText("revolve");
});

test("path tracing yields to interactive dragging and resumes afterwards", async ({ page }) => {
  await expect(page.getByTestId("path-trace")).toHaveCount(0);
  await page.getByTestId("display-options").click();
  await page.getByTestId("render-customize").click();
  await page.getByTestId("toggle-path-tracing").check();
  await expect(page.getByTestId("toggle-path-tracing")).toBeChecked();
  await page.getByTestId("display-options").click();
  await page.getByTestId("mode-vertex").click();

  const metrics = await canvasMetrics(page);
  const from = projectToCss(ROOF_PEAK, metrics);
  const to = projectToCss([0.25, 1.25, 0], metrics);
  await page.mouse.move(metrics.left + from.x, metrics.top + from.y);
  await page.mouse.down();
  await page.mouse.move(metrics.left + to.x, metrics.top + to.y, { steps: 10 });
  await page.mouse.up();

  await expect
    .poll(async () => !(await editorText(page)).includes("[0.0, 1.0]"), {
      timeout: 45_000,
    })
    .toBe(true);
  await waitForCompile(page);
  await page.getByTestId("display-options").click();
  await expect(page.getByTestId("toggle-path-tracing")).toBeChecked();
});

test("hovering an object highlights it before the click", async ({ page }) => {
  const metrics = await canvasMetrics(page);
  const away = { x: metrics.clientWidth * 0.5, y: metrics.clientHeight * 0.08 };
  await page.mouse.move(metrics.left + away.x, metrics.top + away.y);
  await expect(page.locator("[data-testid=viewer-canvas]")).toHaveCSS("cursor", "default");

  // Moving onto the sketch outline shows it is pickable.
  const edge = projectToCss([0, -0.7, 0], metrics);
  await page.mouse.move(metrics.left + edge.x, metrics.top + edge.y);
  await expect(page.locator("[data-testid=viewer-canvas]")).toHaveCSS("cursor", "pointer");
});
