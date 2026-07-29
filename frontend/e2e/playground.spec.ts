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

type Vec3 = [number, number, number];

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

function cameraPosition(): Vec3 {
  const cp = Math.cos(CAMERA.pitch);
  return [
    CAMERA.target[0] + CAMERA.distance * cp * Math.sin(CAMERA.yaw),
    CAMERA.target[1] + CAMERA.distance * Math.sin(CAMERA.pitch),
    CAMERA.target[2] + CAMERA.distance * cp * Math.cos(CAMERA.yaw),
  ];
}

/** Project a world point to CSS pixels inside the canvas element. */
function projectToCss(
  world: Vec3,
  canvas: { width: number; height: number; clientWidth: number; clientHeight: number },
) {
  const position = cameraPosition();
  const forward = norm(sub(CAMERA.target, position));
  const right = norm(cross(forward, [0, 1, 0]));
  const up = cross(right, forward);
  const delta = sub(world, position);
  const viewDepth = dot(delta, forward);
  const aspect = canvas.width / canvas.height;
  const u = dot(delta, right) / (FOV_SCALE * viewDepth);
  const v = dot(delta, up) / (FOV_SCALE * viewDepth);
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
  const from = projectToCss(FIRST_VERTEX, metrics);
  const to = projectToCss([-1.1, -1.4, 0], metrics);

  await page.mouse.move(metrics.left + from.x, metrics.top + from.y);
  await page.mouse.down();
  await page.mouse.move(metrics.left + to.x, metrics.top + to.y, { steps: 12 });
  await page.mouse.up();

  await expect(page.locator("[data-testid=editor] .cm-content")).not.toContainText("[-1.1, -0.7]");
  const text = await editorText(page);
  const match = text.match(/\[-1\.1, (-?[\d.]+)\]/);
  expect(match).not.toBeNull();
  // Dragged downward in the sketch plane, so y decreased.
  expect(Number(match![1])).toBeLessThan(-0.7);
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

test("the view cube switches the projection to orthographic", async ({ page }) => {
  await expect(page.getByTestId("projection-toggle")).toHaveText("PERSP");

  await page.getByTestId("view-menu").click();
  await page.getByTestId("view-menu-front").click();
  await expect(page.getByTestId("projection-toggle")).toHaveText("ORTHO");

  // Iso returns to a perspective camera.
  await page.getByTestId("view-iso").click();
  await expect(page.getByTestId("projection-toggle")).toHaveText("PERSP");
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
  await expect(page.getByTestId("projection-toggle")).toHaveText("ORTHO");
  await page.getByTestId("projection-toggle").click();
  await expect(page.getByTestId("projection-toggle")).toHaveText("PERSP");
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
});

test("render settings group shading, shadows, and quality", async ({ page }) => {
  const before = await editorText(page);
  await page.getByTestId("display-options").click();

  // Modelling defaults: flat shading with crisp shadows.
  await expect(page.getByTestId("shadows-hard")).toHaveClass(/active/);
  await expect(page.getByTestId("shading-flat")).toHaveClass(/active/);

  await page.getByTestId("shadows-soft").click();
  await expect(page.getByTestId("shadows-soft")).toHaveClass(/active/);
  await page.getByTestId("shadows-off").click();
  await expect(page.getByTestId("shadows-off")).toHaveClass(/active/);

  await page.getByTestId("shading-full").click();
  await expect(page.getByTestId("shading-full")).toHaveClass(/active/);
  await page.getByTestId("quality-draft").click();
  await expect(page.getByTestId("quality-draft")).toHaveClass(/active/);
  await page.getByTestId("toggle-xray").check();

  // Render settings are viewer state, never edits to the program.
  expect(await editorText(page)).toBe(before);
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
function gizmoTip(origin: Vec3, axis: 0 | 1 | 2, canvas: Parameters<typeof projectToCss>[1]) {
  const position = cameraPosition();
  const size = Math.max(0.15, 0.18 * Math.hypot(...sub(origin, position)));
  const unit: Vec3 = [0, 0, 0];
  unit[axis] = size * 0.6;
  return projectToCss([origin[0] + unit[0], origin[1] + unit[1], origin[2] + unit[2]], canvas);
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
