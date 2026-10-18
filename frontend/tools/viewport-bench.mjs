// Viewport frame-time benchmark for the playground.
//
// Loads each shipped scene into a running playground, then reads the timings
// the renderer publishes on `window`: `__cadjointBench` (preview frames while
// the camera turns, at rest and as during a drag) and `__cadjointBenchCompile`
// (a cold compile to the first frame, and to the path tracer). Writes a JSON
// summary and one viewport screenshot per scene, for before/after comparison.
//
//   node tools/viewport-bench.mjs --port 6811 --out /tmp/bench [--scenes a,b] [--dpr 2]
//
// The server is `python -m cadjoint.viewer.playground --port <port>`.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, value, index, all) => {
    if (value.startsWith("--")) pairs.push([value.slice(2), all[index + 1]]);
    return pairs;
  }, []),
);
const port = Number(args.port ?? 6811);
const out = resolve(args.out ?? "viewport-bench");
const frames = Number(args.frames ?? 40);
const scenes = (args.scenes ?? "starter,end_cap,cylinder_block,motor_shield").split(",");
const width = Number(args.width ?? 1440);
const height = Number(args.height ?? 900);
const dpr = Number(args.dpr ?? 1);
const SCENES = resolve(import.meta.dirname, "..", "..", "scenes");

mkdirSync(out, { recursive: true });

async function settle(page) {
  await page.waitForFunction(
    () => {
      const status = document.querySelector("[data-testid=status]")?.textContent ?? "";
      return (
        status !== "" &&
        status !== "Starting…" &&
        document.querySelectorAll("[data-testid=toolbar-busy]").length === 0
      );
    },
    null,
    { timeout: 300_000 },
  );
}

async function load(page, source) {
  await page.evaluate((text) => {
    const content = document.querySelector("[data-testid=editor] .cm-content");
    const view = content?.cmView?.view ?? content?.cmTile?.view;
    if (!view) throw new Error("no CodeMirror view");
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
  }, source);
  await page.getByTestId("run").click();
  await page.waitForTimeout(500);
  await settle(page);
}

const browser = await chromium.launch({
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--use-angle=metal",
    "--use-gl=angle",
    "--ignore-gpu-blocklist",
    "--enable-gpu",
  ],
});
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: dpr });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});
await page.goto(`http://127.0.0.1:${port}/`);
await settle(page);

const results = {};
for (const scene of scenes) {
  const source = readFileSync(resolve(SCENES, `${scene}.py`), "utf8");
  const loadStarted = Date.now();
  await load(page, source);
  const loadMs = Date.now() - loadStarted;
  // Let the path tracer's compile finish so it does not share the GPU.
  await page.waitForTimeout(1500);
  await page.locator("canvas.viewer-canvas").first().screenshot({
    path: resolve(out, `${scene}.png`),
  });
  // And one frame as a gesture draws it, for checking the reduced preview.
  await page.evaluate(() => window.__cadjointGesture?.(true));
  await page.waitForTimeout(300);
  await page.locator("canvas.viewer-canvas").first().screenshot({
    path: resolve(out, `${scene}-gesture.png`),
  });
  await page.evaluate(() => window.__cadjointGesture?.(false));
  await page.waitForTimeout(300);
  const rest = await page.evaluate((n) => window.__cadjointBench(n, false), frames);
  const drag = await page.evaluate((n) => window.__cadjointBench(n, true), frames);
  // A handle drag: the first free parameter overridden with its own value,
  // so the image is unchanged but the renderer is in its live-drag state.
  const handle = await page.evaluate(async (n) => {
    const names = window.__cadjointBindings?.().map((b) => b.parameter).filter(Boolean) ?? [];
    const program = window.__cadjointProgram?.();
    const slot = program?.parameters.find((p) => p.free && names.includes(p.name))
      ?? program?.parameters.find((p) => p.free);
    if (!slot) return null;
    window.__cadjointSetParameters({ [slot.name]: slot.value });
    const timing = await window.__cadjointBench(n, true);
    window.__cadjointSetParameters(null);
    return { parameter: slot.name, medianMs: timing.medianMs };
  }, frames);
  const compile = await page.evaluate(() => window.__cadjointBenchCompile());
  results[scene] = {
    loadMs,
    rest: { medianMs: rest.medianMs, meanMs: rest.meanMs, size: [rest.width, rest.height] },
    drag: { medianMs: drag.medianMs, meanMs: drag.meanMs, size: [drag.width, drag.height] },
    handle,
    compile,
  };
  console.log(scene, JSON.stringify(results[scene]));
}
writeFileSync(resolve(out, "results.json"), JSON.stringify({ results, errors }, null, 2));
if (errors.length) console.log("errors:", errors.slice(0, 5));
await browser.close();
