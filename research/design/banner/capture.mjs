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
 * Make sure that bundle is the one you mean (`npm run build` in frontend/, or
 * the committed one), or you will capture a stale UI.
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

// The server warms its compilation cache for every shipped scene at launch, in
// worker subprocesses of its own. Driving it while those run puts two workers
// on one cache lock, and the lock timeout JAX then warns about lands in the
// editor's stderr pane — in the picture. So: wait until the job registry says
// nothing is running before touching the page.
{
  const token = (await (await fetch(URL + "api/session")).json()).token;
  for (let i = 0; i < 240; i++) {
    const jobs = await (await fetch(URL + "api/jobs", { headers: { "X-Cadjoint-Token": token } })).json().catch(() => null);
    const running = jobs?.totals?.running ?? 0;
    if (i % 6 === 0) console.log("warm-up: running jobs", running);
    if (jobs && running === 0 && i > 2) break;
    await wait(5000);
  }
}

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

// The mode strip's own button; "M cycles modes" stays the fallback. (The canvas
// is not clicked for focus any more: its top-left corner is under the tool rail.)
const mode = () => page.evaluate(() => document.querySelector("[data-mode]")?.getAttribute("data-mode"));
if (!(await clickIfThere(testid("editmode-simulate"), "Simulate mode"))) {
  await testid("viewer-canvas").click({ position: { x: 200, y: 8 } });
  for (let i = 0; i < 6 && (await mode()) !== "simulate"; i++) {
    await page.keyboard.press("m"); await page.waitForTimeout(500);
  }
}
console.log("mode:", await mode());
await page.waitForTimeout(1200);

// Studies -> Solve. The study name comes from scenes/starter.py. Studies,
// Results and the rest are dockview windows now; their tab labels carry the
// old sim-tab-* ids (frontend/src/windows/panels.ts), and the old tab strip
// is kept as a fallback for an older bundle.
const simTab = (name) => page.locator(`[data-testid=sim-tab-${name}], [data-testid=sim-tabs] button:has-text("${name}")`);
await clickIfThere(simTab("studies"), "Studies tab");
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
await clickIfThere(simTab("results"), "Results tab");
await page.waitForTimeout(2500);
// Results shares its column with Studies in the Simulate desk and is the short
// one; park the setup group in the tray so the view controls are on screen.
{
  const setup = page.locator(".dv-groupview:has([data-testid=window-tab-studies])");
  if (await clickIfThere(setup.getByTestId("window-minimise"), "park Studies/Meshes")) await page.waitForTimeout(800);
  await clickIfThere(simTab("results"), "Results tab");
}

// optional slice: a real inspection mode, and the only honest way to show the
// interior of a field whose hot region is an internal boundary.
const slice = arg("slice", "");
if (slice) {
  const en = testid("simulate-slice-enabled");
  await en.first().scrollIntoViewIfNeeded().catch(() => {});
  if (await en.first().isVisible().catch(() => false)) {
    const inp = (await en.first().evaluate((el) => el.tagName)) === "INPUT"
      ? en.first() : en.first().locator("input").first();
    if (!(await inp.isChecked().catch(() => false))) await inp.check();
    const fr = testid("simulate-slice-fraction");
    if (await fr.first().isVisible().catch(() => false)) await fr.first().fill(slice);
    await page.waitForTimeout(1500);
    console.log("slice on at", slice);
  } else console.log("!! simulate-slice-enabled not found");
}

// overlays off: the construction overlay is chrome, not data.
await clickIfThere(testid("display-options"), "display options");
await page.waitForTimeout(400);
await clickIfThere(testid("render-customize"), "customize disclosure");
await page.waitForTimeout(400);
const ov = testid("toggle-construction-overlay");
if (await ov.first().isVisible().catch(() => false)) {
  const input = (await ov.first().evaluate((el) => el.tagName)) === "INPUT"
    ? ov.first() : ov.first().locator("input").first();
  if (await input.isChecked().catch(() => true)) await input.uncheck();
  else console.log("construction overlay already off");
  console.log("construction overlay OFF");
} else {
  console.log("!! toggle-construction-overlay not found — overlay may still be on");
}
await page.keyboard.press("Escape");
await page.waitForTimeout(1200);

const canvas = testid("viewer-canvas");
await canvas.waitFor();
// frame the part: the graticule is calibrated, so zoom is a view setting, not data
const zoom = +arg("zoom", 5);
{
  const b = await canvas.boundingBox();
  // orbit: "dx,dy" in css px, dragged from the viewport centre
  const orbit = (arg("orbit", "0,0")).split(",").map(Number);
  if (orbit[0] || orbit[1]) {
    await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
    await page.mouse.down();
    for (let i = 1; i <= 12; i++) {
      await page.mouse.move(b.x + b.width / 2 + (orbit[0] * i) / 12,
                            b.y + b.height / 2 + (orbit[1] * i) / 12);
      await page.waitForTimeout(40);
    }
    await page.mouse.up();
    await page.waitForTimeout(900);
    console.log("orbited", orbit.join(","));
  }
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
