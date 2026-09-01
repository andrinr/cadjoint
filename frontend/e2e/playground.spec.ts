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

/**
 * The heat-sink starter sketches on the XZ plane (normal +Y): profile
 * u = −world-X, v = +world-Z, so a vertex [u, v] sits at world [−u, 0, v].
 */
/** base_l, uv [-0.9, 0.0] — the sketch's first vertex. */
const FIRST_VERTEX: Vec3 = [0.9, 0, 0];
/** fin2_tip_l (vertex index 9), uv [-0.15, 0.85] — free and unconstrained. */
const FIN_TIP: Vec3 = [0.15, 0, 0.85];
/** Midpoint of the fin comb's base edge, for outline picks. */
const BASE_EDGE: Vec3 = [0.5, 0, 0];

/**
 * The full editor document. CodeMirror renders only the visible lines, so
 * `.cm-content` innerText truncates tall programs (the starter is ~160
 * lines); read the document from the editor state instead.
 */
async function editorText(page: Page): Promise<string> {
  return page.evaluate(() => {
    type DocView = { view?: { state: { doc: { toString(): string } } } };
    const content = document.querySelector("[data-testid=editor] .cm-content") as
      | (HTMLElement & { cmView?: DocView; cmTile?: DocView })
      | null;
    // CodeMirror hangs its doc view off the content element (`cmView` in
    // older builds, `cmTile` in current ones); either path reaches the full
    // document. innerText is the last resort and truncates tall programs.
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    if (view) return view.state.doc.toString();
    return content?.innerText ?? "";
  });
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

/**
 * Click a rail tool that lives inside a grouped flyout.
 *
 * The rail's clusters expand on click of their parent icon; children keep
 * their stable testids (`tool-box`, `gizmo-scale`, …) but are only visible
 * while the flyout is open.
 */
async function railTool(page: Page, group: string, testid: string) {
  const child = page.getByTestId(testid);
  for (let attempt = 0; attempt < 2 && !(await child.isVisible()); attempt++) {
    await page.getByTestId(`tool-group-${group}`).click();
  }
  await child.click();
}

test("serves the app and loads the starter sketch", async ({ page }) => {
  await expect(page).toHaveTitle(/CADJOINT/);
  expect(await editorText(page)).toContain("PolygonProfile(");
  expect(await editorText(page)).toContain("[-0.9, 0.0]");
  expect(await editorText(page)).toContain("slug = revolve(slug_profile");
});

test("the optimize panel lists the starter optimization and edits it as source", async ({
  page,
}) => {
  // A real study-backed run plus its auto-replay outlives the default budget.
  test.setTimeout(300_000);
  // Model mode owns the panel; the starter declares one study-backed
  // optimization whose objective label reads metric(study).
  await expect(page.getByTestId("optimize-panel")).toBeVisible();
  const card = page.getByTestId("optimize-cool-sink");
  await expect(card).toBeVisible();
  await expect(card).toContainText("minimize");
  await expect(card).toContainText("max(sink-conduction)");
  await expect(page.getByTestId("optimize-run-cool-sink")).toBeEnabled();

  // Steps and learning rate edit the declaration's literals in place.
  // Three steps keep the FEM-in-the-loop run short enough for CI.
  await page.getByTestId("optimize-steps-cool-sink").fill("3");
  await page.getByTestId("optimize-steps-cool-sink").blur();
  await expect.poll(() => editorText(page), { timeout: 45_000 }).toContain("steps=3");
  await waitForCompile(page);

  await page.getByTestId("optimize-lr-cool-sink").fill("0.02");
  await page.getByTestId("optimize-lr-cool-sink").blur();
  await expect
    .poll(() => editorText(page), { timeout: 45_000 })
    .toContain("learning_rate=0.02");
  await waitForCompile(page);

  // A real run: the starter objective is a smoothed-volume evaluation (a few
  // seconds even at the declared step count), and the optimizer writes the
  // final parameter literals back into the program like a patch. While it
  // runs the card shows a progress block (a live bar when the server
  // streams NDJSON, an indeterminate sweep otherwise).
  await page.getByTestId("optimize-run-cool-sink").click();
  await expect(page.getByTestId("optimize-progress-cool-sink")).toBeVisible();
  await expect(page.getByTestId("optimize-progress-step-cool-sink")).toContainText(
    "step",
  );
  await expect(page.getByTestId("optimize-result-cool-sink")).toBeVisible({
    timeout: 240_000,
  });
  await expect(page.getByTestId("optimize-history-cool-sink")).toBeVisible();
  await expect(page.getByTestId("optimize-player")).toBeVisible();
  await expect(page.getByTestId("optimize-player-hint")).toContainText("scrub");

  // The finished run auto-plays its trajectory once (ghost-compiling
  // intermediate literals through the editor). Catch it in flight — the
  // button flips to Pause once adoption lands — then wait until the player
  // rests on the final frame before asserting on the adopted source. (The
  // catch tolerates the replay finishing between polls or a degenerate
  // single-frame trajectory.)
  await page
    .waitForFunction(
      () =>
        document
          .querySelector("[data-testid=optimize-play]")
          ?.textContent?.includes("Pause") ?? false,
      { timeout: 20_000 },
    )
    .catch(() => {});
  await expect
    .poll(
      async () => {
        const scrub = page.getByTestId("optimize-scrub");
        const [value, max, label] = await Promise.all([
          scrub.inputValue(),
          scrub.getAttribute("max"),
          page.getByTestId("optimize-play").textContent(),
        ]);
        return value === max && !(label ?? "").includes("Pause");
      },
      { timeout: 90_000 },
    )
    .toBe(true);
  // The initial fin_depth literal is gone: the optimized value was adopted.
  await expect
    .poll(async () => (await editorText(page)).includes("fin_depth = Scalar(1.2,"), {
      timeout: 45_000,
    })
    .toBe(false);
  await waitForCompile(page);
  await expect(page.getByTestId("status")).not.toContainText("failed");

  // A study-backed run publishes the optimized design's solved field: the
  // Simulate panel opens on Results showing optimization and simulation
  // together.
  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("simulate-legend")).toBeVisible();
  await expect(page.getByTestId("simulate-legend")).toContainText("cool-sink");
});

test("the revolved starter object exposes its profile and operator", async ({ page }) => {
  const metrics = await canvasMetrics(page);
  const sectionEdge = projectToCss([-0.26, 0, -0.07], metrics);
  await page.mouse.click(
    metrics.left + sectionEdge.x,
    metrics.top + sectionEdge.y,
  );

  await expect(page.getByTestId("selection-chip")).toHaveText("slug section");
  await expect(page.getByTestId("sketch-panel")).toContainText("revolve");
  await expect(page.getByTestId("gizmo-translate")).toHaveClass(/active/);
  await expect(page.getByTestId("rail-hint")).toHaveText("move");
});

test("sketch constraints are drawn over the viewport", async ({ page }) => {
  // The starter's constraint catalog is still growing, so never hardcode
  // counts or dimension values here: derive the expected overlay population
  // from a fresh /compile of the session program — the same ground truth the
  // viewer renders from. Overlay rules mirrored from ViewerPane: every fixed
  // sketch constraint or object relation draws a marker; every *numeric*
  // distance draws a dimension with label Number(value.toPrecision(4));
  // relational kinds (horizontal, parallel, …) are panel chips, not overlays.
  const truth = await page.evaluate(async () => {
    const session = await (await fetch("/api/session")).json();
    const compiled = await (
      await fetch("/compile", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Cadjoint-Token": session.token,
        },
        body: JSON.stringify({ source: session.example }),
      })
    ).json();
    let fixed = 0;
    const sketchLabels: string[] = [];
    for (const node of compiled.construction ?? []) {
      for (const constraint of node.constraints ?? []) {
        if (constraint.kind === "fixed") fixed += 1;
        else if (constraint.kind === "distance" && typeof constraint.value === "number") {
          sketchLabels.push(Number(constraint.value.toPrecision(4)).toString());
        }
      }
    }
    const objectLabels: string[] = [];
    for (const relation of compiled.relations ?? []) {
      if (relation.kind === "fixed") fixed += 1;
      else if (relation.kind === "distance" && typeof relation.value === "number") {
        objectLabels.push(Number(relation.value.toPrecision(4)).toString());
      }
    }
    return { fixed, sketchLabels, objectLabels };
  });
  const distanceCount = truth.sketchLabels.length + truth.objectLabels.length;
  // The starter must keep exercising both overlay kinds for this test to
  // mean anything; if this fails the scene lost its constraint showcase.
  expect(truth.fixed).toBeGreaterThan(0);
  expect(distanceCount).toBeGreaterThan(0);

  await expect(page.getByTestId("constraint-overlay")).toBeVisible();
  await expect(page.getByTestId("constraint-fixed-overlay")).toHaveCount(truth.fixed);
  await expect(page.getByTestId("constraint-distance-overlay")).toHaveCount(distanceCount);
  for (const label of truth.objectLabels) {
    await expect(
      page.locator('[data-testid="constraint-distance-overlay"][data-scope="object"]'),
    ).toContainText(label);
  }
  for (const label of truth.sketchLabels) {
    await expect(page.getByTestId("constraint-overlay")).toContainText(label);
  }

  // Top view: the heat sink lies in the XZ plane, so both dimensions stay
  // extended on screen (front view would collapse the fin-height one). The
  // top face is occluded on the 3D cube at the iso camera, so a forced
  // pointer click would land on the front face; dispatch directly instead.
  await page.getByTestId("view-top").dispatchEvent("click");
  await expect(page.getByTestId("constraint-fixed-overlay")).toHaveCount(truth.fixed);
  await expect(page.getByTestId("constraint-distance-overlay")).toHaveCount(distanceCount);

  await page.getByTestId("display-options").click();
  await page.getByTestId("render-customize").click();
  await page.getByTestId("toggle-constraints").uncheck();
  await expect(page.getByTestId("constraint-overlay")).toHaveCount(0);
  await expect(page.getByTestId("toggle-fixed-constraints")).toBeDisabled();

  await page.getByTestId("toggle-constraints").check();
  await page.getByTestId("toggle-fixed-constraints").uncheck();
  await expect(page.getByTestId("constraint-fixed-overlay")).toHaveCount(0);
  await expect(page.getByTestId("constraint-distance-overlay")).toHaveCount(distanceCount);

  await page.getByTestId("toggle-constraint-values").uncheck();
  await expect(page.getByTestId("constraint-overlay")).not.toContainText(
    truth.sketchLabels[0] ?? truth.objectLabels[0],
  );
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
  await expect(page.locator(".cm-vertex-highlight")).toHaveText("[-0.9, 0.0]");
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
  const from = projectToCss(FIN_TIP, metrics);
  const to = projectToCss([0.25, 0, 1.1], metrics);

  await page.mouse.move(metrics.left + from.x, metrics.top + from.y);
  await page.mouse.down();
  await page.mouse.move(metrics.left + to.x, metrics.top + to.y, { steps: 12 });
  await page.mouse.up();

  await expect
    .poll(async () => (await editorText(page)).includes("[-0.15, 0.85]"), {
      timeout: 45_000,
    })
    .toBe(false);
  const text = await editorText(page);
  const match = text.match(
    /fin2_tip_l = Vector2\(value=\[(-?[\d.]+), (-?[\d.]+)\]/,
  );
  expect(match).not.toBeNull();
  // World [0.25, 0, 1.1] maps to sketch uv ≈ [-0.25, 1.1] (u = −X, v = +Z).
  expect(Number(match![1])).toBeLessThan(-0.1);
  expect(Number(match![2])).toBeGreaterThan(1);
  await waitForCompile(page);
});

/** Count `[x, y]` literals, allowing any numeric formatting. */
const NUMBER = String.raw`-?[\d.]+(?:e[-+]?\d+)?`;
const vertexLiteralCount = async (page: Page) =>
  ((await editorText(page)).match(new RegExp(`\\[${NUMBER}, ${NUMBER}\\]`, "g")) ?? []).length;

test("the polygon tool inserts a vertex and stays active", async ({ page }) => {
  const before = await vertexLiteralCount(page);

  await railTool(page, "create", "tool-polygon");
  const metrics = await canvasMetrics(page);
  // A point on the fin comb's base edge.
  const point = projectToCss(BASE_EDGE, metrics);
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

  // Orbit by dragging empty space well clear of the sketch, the dock (which
  // overlays the right side), and the hint bar along the bottom.
  const metrics = await canvasMetrics(page);
  await page.mouse.move(metrics.left + metrics.clientWidth * 0.42, metrics.top + metrics.clientHeight * 0.82);
  await page.mouse.down();
  await page.mouse.move(
    metrics.left + metrics.clientWidth * 0.18,
    metrics.top + metrics.clientHeight * 0.82,
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
    "from cadjoint.construction import Solid",
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
  // A point on the rim of the bush_b bushing cylinder at [-0.78, 0, 0.1].
  const outline = projectToCss([-0.71, 0, 0.16], metrics);
  await page.mouse.click(metrics.left + outline.x, metrics.top + outline.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("bush_b");

  await page.getByTestId("delete-selection").click();
  await expect
    .poll(async () => (await editorText(page)).includes("bush_b = Solid.cylinder"), {
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
  await expect(page.locator(".cm-vertex-highlight")).toHaveText("[-0.9, 0.0]");

  // Switching back to object selection promotes the point to its polygon and
  // reveals the polygon's translation gizmo without requiring another click.
  await page.getByTestId("mode-object").click();
  await expect(page.getByTestId("selection-chip")).toHaveText("fin comb");
  await expect(page.getByTestId("gizmo-translate")).toHaveClass(/active/);
});

test("render presets activate, edit, and persist without bloating the closed UI", async ({
  page,
}) => {
  const before = await editorText(page);
  // Closed, the settings cost nothing: no panel, no cards in the DOM.
  await expect(page.getByTestId("render-panel")).toHaveCount(0);
  await expect(page.locator(".render-preset-card")).toHaveCount(0);
  // The eye icon opens the render-settings popover (from any mode).
  await page.getByTestId("display-options").click();
  await expect(page.getByTestId("render-popover")).toBeVisible();

  await expect(page.getByTestId("render-panel")).toBeVisible();
  await expect(page.locator(".render-preset-card")).toHaveCount(3);
  await expect(page.getByTestId("render-preset-editor")).toHaveCount(0);
  await expect(page.getByTestId("render-preset-xray")).toHaveClass(/active/);
  const compactPanel = await page.getByTestId("render-panel").boundingBox();
  expect(compactPanel!.width).toBeLessThanOrEqual(310);
  expect(compactPanel!.height).toBeLessThan(300);

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
  // The popover is transient, but the preset state persists: reopen it.
  await expect(page.getByTestId("render-popover")).toHaveCount(0);
  await page.getByTestId("display-options").click();
  await expect(page.getByTestId("render-panel")).toBeVisible();
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
  await expect(page.getByTestId("render-panel")).toBeVisible();

  const editorZ = await page
    .locator(".editor-pane")
    .evaluate((node) => Number.parseInt(getComputedStyle(node).zIndex, 10));
  const dockZ = await page
    .locator(".dock")
    .evaluate((node) => Number.parseInt(getComputedStyle(node).zIndex, 10));
  expect(editorZ).toBeGreaterThan(dockZ);
});

test("editing the code updates the sketch the viewer reports", async ({ page }) => {
  await page.getByTestId("mode-vertex").click();
  const program = [
    "from cadjoint.construction import PolygonProfile, extrude",
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
  // The heat-sink starter has no spheres; count anyway to stay robust.
  const before = await sphereCount(page);

  await railTool(page, "create", "tool-sphere");
  const metrics = await canvasMetrics(page);
  // Somewhere on the world XY plane, clear of the sink and of the dock
  // panels that overlay the right side of the viewport.
  const point = projectToCss([0, 2.4, 0], metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);

  await expect.poll(() => sphereCount(page), { timeout: 45_000 }).toBe(before + 1);
  // Solid is already imported by the starter program, so no duplicate appears.
  expect(await editorText(page)).toMatch(/from cadjoint\.construction import .*\bSolid\b/);

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
  await expect(page.getByTestId("material-aluminum")).toBeVisible();
  await expect(page.getByTestId("material-copper")).toBeVisible();

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
  const bushing = projectToCss([-0.78, 0, 0.1], metrics);
  await page.getByTestId("material-copper").dragTo(
    page.getByTestId("viewer-canvas"),
    { targetPosition: { x: bushing.x, y: bushing.y } },
  );
  await expect
    .poll(() => editorText(page), { timeout: 45_000 })
    .toMatch(/bush_b = Solid\.cylinder\([\s\S]*?material=copper/);
  await waitForCompile(page);

  await page.getByTestId("material-close").click();
  await expect(page.getByTestId("material-panel")).toHaveCount(0);
  await expect(page.getByTestId("material-open")).toBeVisible();
});

test("a placed primitive can be selected and moved along an axis", async ({ page }) => {
  await railTool(page, "create", "tool-box");
  const metrics = await canvasMetrics(page);
  // Kept left of centre: the gizmo arrows stay on the canvas and the drop
  // click cannot land on the dock overlaying the right side.
  const drop = projectToCss([-1.4, 0, 0], metrics);
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
  // A whole-object selection offers the transform cluster on the rail.
  await expect(page.getByTestId("tool-group-transform")).toBeVisible();

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
  const edge = projectToCss(BASE_EDGE, metrics);
  await page.mouse.click(metrics.left + edge.x, metrics.top + edge.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("fin comb");
  await expect(page.getByTestId("gizmo-translate")).toBeEnabled();
  await expect(page.getByTestId("gizmo-translate")).toHaveClass(/active/);

  // The arrows are centered on the polygon, not a potentially remote plane
  // origin; the fin comb's centroid sits near [0, 0, 0.41].
  const view = await canvasMetrics(page);
  const from = gizmoTip([0, 0, 0.41], 1, view);
  const to = projectToCss([0, 1.0, 0.41], view);
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
  expect(await editorText(page)).toContain("[-0.9, 0.0]");
});

test("a default polygon can move without losing parameter-backed points", async ({ page }) => {
  const program = [
    "from cadjoint.construction import PolygonProfile, extrude",
    "from cadjoint.geometry import Vector2",
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
  await railTool(page, "create", "tool-box");
  let metrics = await canvasMetrics(page);
  const drop = projectToCss([-1.4, 0.9, 0], metrics);
  await page.mouse.click(metrics.left + drop.x, metrics.top + drop.y);
  await waitForCompile(page);
  await expect(page.getByTestId("gizmo-scale")).toBeEnabled();
  await railTool(page, "transform", "gizmo-scale");

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
  await railTool(page, "create", "tool-sketch");
  let metrics = await canvasMetrics(page);
  // Placed left of center: the right side of the viewport hosts the dock
  // (object tree / sketch / materials), and the vertex clicks below must land
  // on the canvas, not on the dock's panels.
  const origin: Vec3 = [0.0, 1.6, 0];
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
    .poll(async () => (await editorText(page)).includes("FixedConstraint(sketch1"), {
      timeout: 45_000,
    })
    .toBe(false);
  await waitForCompile(page);
  await expect(page.getByTestId("status")).not.toContainText("failed");
});

test("relational constraint chips render for API-added kinds", async ({ page }) => {
  const program = [
    "from cadjoint.construction import PolygonProfile, extrude",
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
          "X-Cadjoint-Token": session.token,
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
  await railTool(page, "create", "tool-sketch");
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
  const from = projectToCss(FIN_TIP, metrics);
  const to = projectToCss([0.25, 0, 1.1], metrics);
  await page.mouse.move(metrics.left + from.x, metrics.top + from.y);
  await page.mouse.down();
  await page.mouse.move(metrics.left + to.x, metrics.top + to.y, { steps: 10 });
  await page.mouse.up();

  await expect
    .poll(async () => !(await editorText(page)).includes("[-0.15, 0.85]"), {
      timeout: 45_000,
    })
    .toBe(true);
  await waitForCompile(page);
  // Reopening the popover mounts a fresh panel with Customize collapsed.
  await page.getByTestId("display-options").click();
  await page.getByTestId("render-customize").click();
  await expect(page.getByTestId("toggle-path-tracing")).toBeChecked();
});

test("editing modes scope the tool rail and Escape returns to model", async ({ page }) => {
  // Model mode is the default: the create cluster is offered, no simulate slot.
  await expect(page.getByTestId("editmode-model")).toHaveClass(/active/);
  await expect(page.getByTestId("hint-mode")).toHaveText("model");
  await expect(page.getByTestId("tool-group-create")).toBeVisible();
  await expect(page.getByTestId("mode-simulate")).toHaveCount(0);

  // Simulate mode swaps the tool clusters for the simulation slot.
  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("mode-simulate")).toBeVisible();
  await expect(page.getByTestId("hint-mode")).toHaveText("simulate");
  await expect(page.getByTestId("tool-group-create")).toHaveCount(0);

  // Render is not a mode: the eye-icon popover opens from Simulate too, and
  // Escape closes just the popover — the mode and selection stay put.
  await page.getByTestId("editmode-simulate").click();
  await page.getByTestId("display-options").click();
  await expect(page.getByTestId("render-popover")).toBeVisible();
  await expect(page.getByTestId("render-panel")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("render-popover")).toHaveCount(0);
  await expect(page.getByTestId("editmode-simulate")).toHaveClass(/active/);

  // The next Escape backs out to model mode.
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("editmode-model")).toHaveClass(/active/);
  await expect(page.getByTestId("mode-simulate")).toHaveCount(0);

  // Selecting a sketch auto-enters sketch mode, where constraint tools live.
  const metrics = await canvasMetrics(page);
  const edge = projectToCss(BASE_EDGE, metrics);
  await page.mouse.click(metrics.left + edge.x, metrics.top + edge.y);
  await expect(page.getByTestId("selection-chip")).toHaveText("fin comb");
  await expect(page.getByTestId("editmode-sketch")).toHaveClass(/active/);
  await expect(page.getByTestId("tool-group-annotate")).toBeVisible();
});

test("hovering an object highlights it before the click", async ({ page }) => {
  const metrics = await canvasMetrics(page);
  const away = { x: metrics.clientWidth * 0.5, y: metrics.clientHeight * 0.08 };
  await page.mouse.move(metrics.left + away.x, metrics.top + away.y);
  await expect(page.locator("[data-testid=viewer-canvas]")).toHaveCSS("cursor", "default");

  // Moving onto the sketch outline shows it is pickable.
  const edge = projectToCss(BASE_EDGE, metrics);
  await page.mouse.move(metrics.left + edge.x, metrics.top + edge.y);
  await expect(page.locator("[data-testid=viewer-canvas]")).toHaveCSS("cursor", "pointer");
});

test("studies are declared, edited, and deleted through source patches", async ({ page }) => {
  // Patch → recompile is a round trip; poll the editor rather than racing it.
  const editorHas = (needle: string, negate = false) =>
    expect
      .poll(async () => (await editorText(page)).includes(needle), { timeout: 45_000 })
      .toBe(!negate);

  // The starter declares the heat-sink conduction study up front.
  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("simulate-panel")).toBeVisible();
  await expect(page.getByTestId("simulate-study-sink-conduction")).toBeVisible();

  // cool-sink optimizes this study, so retire the optimization first —
  // deleting the study out from under it would leave a broken program.
  await page.getByTestId("sim-tab-optimize").click();
  await page.getByTestId("optimize-delete-cool-sink").click();
  await editorHas("Optimization(", true);
  await page.getByTestId("sim-tab-studies").click();

  // Deleting it reaches the empty state, which offers to add one again.
  await page.getByTestId("simulate-delete-sink-conduction").click();
  await editorHas("ThermalStudy(", true);
  await expect(page.getByTestId("simulate-empty")).toBeVisible();

  // Adding a study writes a ThermalStudy declaration into the program.
  await page.getByTestId("simulate-add-thermal").click();
  await editorHas("ThermalStudy(");
  const card = page.locator("[data-testid^=simulate-study-]").first();
  await expect(card).toBeVisible();
  const name = (await card.getAttribute("data-testid"))!.replace("simulate-study-", "");

  // A boundary condition built from a box vertex selection lands as source.
  await page.getByTestId(`simulate-add-bc-${name}`).click();
  await page.getByTestId("simulate-builder-selection").selectOption("box");
  await page.getByTestId("simulate-builder-value").fill("250");
  await page.getByTestId("simulate-builder-add").click();
  await editorHas("Nodes.box(");
  await editorHas("value=250.0");

  // Editing the BC value patches the literal in place.
  await page.getByTestId(`simulate-bc-value-${name}-0`).fill("325");
  await page.getByTestId(`simulate-bc-value-${name}-0`).blur();
  await editorHas("value=325.0");

  // Deleting the BC and the study removes both declarations again.
  await page.getByTestId(`simulate-bc-delete-${name}-0`).click();
  await editorHas("Nodes.box(", true);
  await page.getByTestId(`simulate-delete-${name}`).click();
  await editorHas("ThermalStudy(", true);
  await expect(page.getByTestId("simulate-empty")).toBeVisible();
});

test("meshes are declared, inspected, and deleted through source patches", async ({
  page,
}) => {
  const editorHas = (needle: string, negate = false) =>
    expect
      .poll(async () => (await editorText(page)).includes(needle), { timeout: 45_000 })
      .toBe(!negate);

  // The starter declares its mesh (sink-mesh) up front: the tab lists the
  // card rather than the empty state, and the study references it.
  await page.getByTestId("editmode-simulate").click();
  await page.getByTestId("sim-tab-meshes").click();
  await expect(page.getByTestId("mesh-empty")).toHaveCount(0);
  await expect(page.getByTestId("mesh-sink-mesh")).toBeVisible();

  // Generate builds the declared mesh and reports quality stats plus the
  // histogram.
  await page.getByTestId("mesh-inspect-sink-mesh").click();
  await expect(page.getByTestId("mesh-stats")).toBeVisible({ timeout: 90_000 });
  await expect(page.getByTestId("mesh-stats")).toContainText("nodes");
  await expect(page.getByTestId("mesh-stats")).toContainText("jacobian");
  await expect(page.getByTestId("mesh-histogram")).toBeVisible();

  // The inspected mesh owns the viewport with element edges on; "Scene"
  // hands it back to the raymarched SDF without dropping the loaded state.
  await expect(page.getByTestId("simulate-viewport")).toBeVisible();
  await expect(page.getByTestId("simulate-edges")).toBeChecked();
  await page.getByTestId("simulate-viewport-scene").click();
  await expect(page.getByTestId("simulate-viewport-scene")).toHaveClass(/active/);
  await page.getByTestId("simulate-viewport-mesh").click();
  await expect(page.getByTestId("simulate-viewport-mesh")).toHaveClass(/active/);

  // "+ Mesh" writes a second SimMesh declaration into the program.
  await page.getByTestId("mesh-add").click();
  const card = page.getByTestId("mesh-mesh1");
  await expect(card).toBeVisible();

  // Editing the padding rewrites the new declaration's literal.
  await page.getByTestId("mesh-arg-mesh1-padding").fill("0.2");
  await page.getByTestId("mesh-arg-mesh1-padding").blur();
  await editorHas("padding=0.2");

  // Deleting it removes only that declaration — sink-mesh stays (the study
  // references it, so it is not deletable without editing the study first).
  await page.getByTestId("mesh-delete-mesh1").click();
  await editorHas("padding=0.2", true);
  await expect(page.getByTestId("mesh-mesh1")).toHaveCount(0);
  await expect(page.getByTestId("mesh-sink-mesh")).toBeVisible();
});

test("solving the declared study reports the result and swaps to quality view", async ({
  page,
}) => {
  await page.getByTestId("editmode-simulate").click();
  await expect(page.getByTestId("simulate-study-sink-conduction")).toBeVisible();

  await page.getByTestId("simulate-run-sink-conduction").click();
  await expect(page.getByTestId("simulate-legend")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId("simulate-legend")).toContainText("temperature");
  const summary = page.getByTestId("simulate-result-summary");
  await expect(summary).toBeVisible();
  await expect(summary).toContainText("nodes");
  await expect(summary).toContainText("temperature");

  // Solved fields read clean: element edges default off (one toggle away).
  await expect(page.getByTestId("simulate-edges")).not.toBeChecked();

  // The quality toggle re-inspects the mesh and swaps the displayed scalars.
  await page.getByTestId("simulate-quality-toggle").check();
  await expect(page.getByTestId("simulate-legend")).toContainText("scaled jacobian", {
    timeout: 90_000,
  });
  await page.getByTestId("simulate-quality-toggle").uncheck();
  await expect(page.getByTestId("simulate-legend")).toContainText("temperature");
});

test("clicking the inspected mesh proposes a sphere BC selection", async ({ page }) => {
  const editorHas = (needle: string, negate = false) =>
    expect
      .poll(async () => (await editorText(page)).includes(needle), { timeout: 45_000 })
      .toBe(!negate);

  await page.getByTestId("editmode-simulate").click();
  await page.getByTestId("sim-tab-meshes").click();
  await page.getByTestId("mesh-add").click();
  await editorHas("SimMesh(");
  await page.getByTestId("mesh-inspect-mesh1").click();
  await expect(page.getByTestId("mesh-stats")).toBeVisible({ timeout: 90_000 });

  // Arm viewport picking from the study's add-BC builder (Studies tab).
  await page.getByTestId("sim-tab-studies").click();
  await page.getByTestId("simulate-add-bc-sink-conduction").click();
  await expect(page.getByTestId("simulate-builder")).toBeVisible();
  await page.getByTestId("simulate-builder-pick").click();
  await expect(page.getByTestId("viewer-hint")).toContainText("Pick BC");

  // Click a point on the sink's front face; the nearest mesh vertex becomes
  // the centre of a proposed Nodes.sphere sized from the cell spacing.
  const metrics = await canvasMetrics(page);
  const point = projectToCss([0.5, -0.6, 0.1], metrics);
  await page.mouse.click(metrics.left + point.x, metrics.top + point.y);

  await expect(page.getByTestId("simulate-builder-selection")).toHaveValue("sphere");
  const radius = await page.getByTestId("simulate-builder-radius").inputValue();
  expect(Number(radius)).toBeGreaterThan(0);

  // Confirming emits the ordinary add_study_bc patch with that selection.
  await page.getByTestId("simulate-builder-add").click();
  await editorHas("Nodes.sphere(");
  await waitForCompile(page);
  await expect(page.getByTestId("status")).not.toContainText("failed");
});
