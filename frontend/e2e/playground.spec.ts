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
  const metrics = await canvasMetrics(page);
  const point = projectToCss(FIRST_VERTEX, metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);

  await expect(page.getByTestId("selection-chip")).toHaveText("vertex 0");
  // The editor marks exactly the literal that defines that vertex.
  await expect(page.locator(".cm-vertex-highlight")).toHaveText("[-1.1, -0.7]");
});

test("escape clears the selection", async ({ page }) => {
  const metrics = await canvasMetrics(page);
  const point = projectToCss(FIRST_VERTEX, metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);
  await expect(page.getByTestId("selection-chip")).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(page.getByTestId("selection-chip")).toHaveCount(0);
});

test("dragging a handle rewrites the vertex literal", async ({ page }) => {
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

test("add-vertex mode inserts a vertex into the source", async ({ page }) => {
  const before = await vertexLiteralCount(page);

  await page.getByTestId("tool-add").click();
  const metrics = await canvasMetrics(page);
  // Midpoint of the sketch's bottom edge.
  const point = projectToCss([0, -0.7, 0], metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);

  // The patch and its recompile are async; poll rather than racing them.
  await expect.poll(() => vertexLiteralCount(page), { timeout: 45_000 }).toBe(before + 1);
  // The tool returns to select so the next click does not add another vertex.
  await expect(page.getByTestId("tool-select")).toHaveClass(/active/);
  await waitForCompile(page);
});

test("editing the code updates the sketch the viewer reports", async ({ page }) => {
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
