/**
 * Live capture of the running playground, for the README banner hero.
 *
 *   node research/design/banner/capture.mjs --port 8792
 *
 * Starts `.venv/bin/python -m cadjoint.viewer.playground` on the given port,
 * drives it with playwright (compile -> Run -> Simulate -> Solve -> Results),
 * turns the construction overlay off, and writes the viewport region at 2x to
 *   research/design/banner/assets/hero-capture.png
 *
 * IMPORTANT: the server hands out whatever bundle is in cadjoint/viewer/static/.
 * Run `npm run build` in frontend/ first, or you will capture a stale UI.
 *
 * Flags: --port N (required-ish, default 8792) · --out PATH · --width/--height
 *        --keep (leave the browser open) · --timeout MS
 */
import { chromium } from "/Users/andrinrehnann/code/jaxcad/frontend/node_modules/@playwright/test/index.mjs";
import { spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../../..");
const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i === -1 ? d : process.argv[i + 1];
};
const flag = (k) => process.argv.includes(`--${k}`);

const PORT = +arg("port", 8792);
const OUT = path.resolve(here, arg("out", "assets/hero-capture.png"));
const W = +arg("width", 1500), H = +arg("height", 940);
const TIMEOUT = +arg("timeout", 180000);
const URL = `http://127.0.0.1:${PORT}/`;

const staticIndex = path.join(repo, "cadjoint/viewer/static/index.html");
if (!fs.existsSync(staticIndex)) {
  console.error("no built frontend at cadjoint/viewer/static/index.html — run `npm run build` in frontend/");
  process.exit(1);
}
console.log("bundle mtime:", fs.statSync(staticIndex).mtime.toISOString());

const server = spawn(path.join(repo, ".venv/bin/python"),
  ["-m", "cadjoint.viewer.playground", "--port", String(PORT)],
  { cwd: repo, stdio: ["ignore", "pipe", "pipe"] });
server.stdout.on("data", (b) => process.stdout.write("[server] " + b));
server.stderr.on("data", (b) => process.stderr.write("[server] " + b));
const stop = () => { try { server.kill("SIGTERM"); } catch {} };
process.on("exit", stop); process.on("SIGINT", () => { stop(); process.exit(1); });

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
for (let i = 0; i < 120; i++) {
  try { const r = await fetch(URL); if (r.ok) break; } catch {}
  await wait(500);
}
console.log("server up at", URL);

const browser = await chromium.launch({
  headless: !flag("keep"),
  args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal",
         "--ignore-gpu-blocklist", "--force-color-profile=srgb"],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
await page.goto(URL);

const testid = (id) => page.locator(`[data-testid=${id}]`);
const clickIfThere = async (loc, what) => {
  if (await loc.first().isVisible().catch(() => false)) {
    await loc.first().click(); console.log("clicked", what); return true;
  }
  return false;
};

await testid("viewer-canvas").waitFor({ timeout: TIMEOUT });
await clickIfThere(page.getByRole("button", { name: /dismiss|got it|close/i }), "first-run dialog");
await page.waitForFunction(
  () => !document.querySelector("[data-testid=viewer-compiling]")
     && !/compil/i.test(document.querySelector("[data-testid=status]")?.textContent ?? ""),
  null, { timeout: TIMEOUT });
console.log("scene compiled");

await clickIfThere(testid("run"), "Run");
await page.waitForTimeout(4000);

// "M cycles modes" — the keyboard is the stable contract; the switcher markup is not.
const mode = () => page.evaluate(() => document.querySelector("[data-mode]")?.getAttribute("data-mode"));
await testid("viewer-canvas").click({ position: { x: 8, y: 8 } });
for (let i = 0; i < 6 && (await mode()) !== "simulate"; i++) {
  await page.keyboard.press("m"); await page.waitForTimeout(500);
}
console.log("mode:", await mode());
await page.waitForTimeout(1200);

// Studies -> Solve. The study name comes from scenes/starter.py.
await clickIfThere(page.locator("[data-testid=sim-tabs] button", { hasText: /studies/i }), "Studies tab");
await page.waitForTimeout(600);
const solve = testid("simulate-run-sink-conduction");
if (await solve.first().isVisible().catch(() => false)) {
  await solve.first().click(); console.log("clicked Solve");
  await page.waitForFunction(
    () => !/solving|meshing/i.test(document.body.textContent ?? ""),
    null, { timeout: TIMEOUT });
} else {
  console.log("!! no simulate-run-sink-conduction button — capturing whatever is on screen");
}
await clickIfThere(page.locator("[data-testid=sim-tabs] button", { hasText: /results/i }), "Results tab");
await page.waitForTimeout(2500);

// overlays off: the construction/sketch overlay is chrome, not data.
await clickIfThere(testid("display-options"), "display options");
await page.waitForTimeout(400);
console.log("render popover controls:", (await page.locator("[data-testid=render-popover] label").allTextContents()).join(" | "));
for (const rx of [/construction/i, /overlay/i, /sketch/i, /gizmo/i]) {
  const box = page.locator("label", { hasText: rx }).locator("input[type=checkbox]");
  if (await box.first().isVisible().catch(() => false)
      && await box.first().isChecked().catch(() => false)) {
    await box.first().uncheck(); console.log("unchecked", rx);
  }
}
await page.keyboard.press("Escape");
await page.waitForTimeout(1200);

const canvas = testid("viewer-canvas");
await canvas.waitFor();
// frame the part: the graticule is calibrated, so zoom is a view setting, not data
const zoom = +arg("zoom", 5);
{
  const b = await canvas.boundingBox();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
  for (let i = 0; i < Math.abs(zoom); i++) {
    await page.mouse.wheel(0, zoom > 0 ? -260 : 260);
    await page.waitForTimeout(180);
  }
  await page.waitForTimeout(1500);
}
const box = await canvas.boundingBox();
fs.mkdirSync(path.dirname(OUT), { recursive: true });
await page.screenshot({ path: OUT, clip: box });
console.log("captured", OUT, `${Math.round(box.width * 2)}x${Math.round(box.height * 2)}`);

// a whole-window reference shot, so a stale or broken UI is obvious at a glance
const ref = OUT.replace(/\.png$/, "-context.png");
await page.screenshot({ path: ref });
console.log("context", ref);

if (!flag("keep")) { await browser.close(); stop(); process.exit(0); }
