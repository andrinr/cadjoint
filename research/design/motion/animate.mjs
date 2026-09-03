/**
 * Live animation capture of the running playground, for the README.
 *
 *   node research/design/motion/animate.mjs                 # every clip
 *   node research/design/motion/animate.mjs --only sdf-sweep,solve-field
 *   node research/design/motion/animate.mjs --formats       # + gif/mp4/webm, for size evidence
 *
 * Sister to `capture.mjs`, which cuts the one still hero. This one starts the
 * same server, drives the same real app through playwright on the real scenes
 * in `scenes/`, and records each interaction with the DevTools screencast —
 * so what lands in `docs/assets/motion/` is the app, at frame rate, not a
 * reconstruction.
 *
 * Chain, per clip:
 *
 *   Page.startScreencast  ->  jpeg frames + wall timestamps
 *   resample by timestamp ->  a constant-rate sequence (a clip may be a
 *                             time-lapse: `speed` is wall seconds per
 *                             played second)
 *   ffmpeg crop+scale     ->  png sequence at the delivery size
 *   img2webp              ->  one animated WebP, the file the README uses
 *
 * Why animated WebP and not GIF or <video>: see `docs/assets/motion/README.md`,
 * which this script writes with the measured numbers every time it runs.
 *
 * IMPORTANT: the server hands out whatever bundle is in cadjoint/viewer/static/.
 * This script prints its mtime and refuses to run against a missing one. It
 * never builds; if the bundle is stale the animations are of a stale UI.
 *
 * Requires: ffmpeg and img2webp on PATH (`brew install ffmpeg webp`).
 *
 * Flags:
 *   --port N        server port (default 6700; an already-serving port is reused)
 *   --only a,b      capture just these clips
 *   --list          print the clip names and exit
 *   --formats       also encode gif / mp4 / webm / apng variants and print sizes
 *   --keep-frames   leave the raw frame dumps in the work directory
 *   --work DIR      scratch directory (default: os tmp)
 *   --headed        watch it happen
 */
import { chromium } from "/Users/andrinrehnann/code/jaxcad/frontend/node_modules/@playwright/test/index.mjs";
import { spawn, spawnSync } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../../..");
const OUT_DIR = path.join(repo, "docs/assets/motion");

const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i === -1 ? d : process.argv[i + 1];
};
const flag = (k) => process.argv.includes(`--${k}`);

const PORT = +arg("port", 6700);
const URL = `http://127.0.0.1:${PORT}/`;
const WORK = arg("work", path.join(os.tmpdir(), "cadjoint-motion"));
const HEADED = flag("headed");

/** The capture viewport. Every clip shares it, so crops are comparable. */
const VIEW = { width: 1440, height: 900 };
/** 2x surface, downsampled by the screencast to 1 css px per frame px. */
const DSF = 2;
/** Played frame rate. 16 is enough for UI motion and a fifth cheaper than 20. */
const FPS = 16;

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// ─────────────────────────────────────────────────────────── tools

function tool(name) {
  const which = spawnSync("which", [name], { encoding: "utf8" });
  if (which.status !== 0) {
    console.error(`missing ${name} — brew install ffmpeg webp`);
    process.exit(1);
  }
  return which.stdout.trim();
}
const FFMPEG = tool("ffmpeg");
const IMG2WEBP = tool("img2webp");

function run(cmd, args, quiet = true) {
  const r = spawnSync(cmd, args, { encoding: "utf8" });
  if (r.status !== 0) {
    console.error(`${path.basename(cmd)} failed:`, r.stderr?.slice(-2000));
    throw new Error(`${cmd} exited ${r.status}`);
  }
  if (!quiet) process.stdout.write(r.stderr ?? "");
  return r;
}
const bytes = (p) => fs.statSync(p).size;
const kb = (n) => `${(n / 1024).toFixed(0)} KB`;

// ─────────────────────────────────────────────────────────── recorder

/**
 * A DevTools screencast, dumped to disk frame by frame.
 *
 * `page.video()` would be simpler, but playwright's recorder is a fixed-rate
 * VP8 encode of the compositor: lossy before anything is cropped, and no
 * timestamps to resample a time-lapse against. The screencast hands over the
 * frames themselves, each with the wall clock it was composited at, which is
 * what makes `speed` possible at all.
 */
class Screencast {
  constructor(page, dir) {
    this.page = page;
    this.dir = dir;
    this.frames = [];
    this.n = 0;
  }
  async start() {
    fs.rmSync(this.dir, { recursive: true, force: true });
    fs.mkdirSync(this.dir, { recursive: true });
    this.cdp = await this.page.context().newCDPSession(this.page);
    this.cdp.on("Page.screencastFrame", ({ data, sessionId, metadata }) => {
      const file = path.join(this.dir, `raw-${String(this.n++).padStart(6, "0")}.jpg`);
      fs.writeFileSync(file, Buffer.from(data, "base64"));
      this.frames.push({ t: metadata.timestamp, file });
      this.cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
    });
    await this.cdp.send("Page.startScreencast", {
      format: "jpeg",
      quality: 95,
      maxWidth: VIEW.width,
      maxHeight: VIEW.height,
      everyNthFrame: 1,
    });
    this.t0 = Date.now() / 1000;
  }
  async stop() {
    await this.cdp.send("Page.stopScreencast").catch(() => {});
    await this.cdp.detach().catch(() => {});
    // Chrome sends a frame per composite, so a still page sends nothing at
    // all: the last frame's timestamp is when motion stopped, not when the
    // clip did. Ending the timeline on the wall clock instead keeps the beat
    // of stillness a clip needs before it loops.
    this.t1 = Date.now() / 1000;
    return this.frames;
  }
}

/**
 * Constant-rate output frames, chosen from the captured ones by wall clock.
 *
 * `speed` > 1 is a time-lapse: a 300-second optimisation played in six
 * seconds is still every frame the app drew, sampled — not a montage.
 * `trim` drops lead-in the interaction needed but the reader does not.
 */
function resample(frames, { speed = 1, fit = 0, trim = [0, 0], until = 0 } = {}) {
  if (frames.length === 0) throw new Error("no frames captured");
  const first = frames[0].t + trim[0];
  const end = Math.max(until, frames[frames.length - 1].t);
  const last = end - trim[1];
  const span = Math.max(last - first, 0.1);
  // `fit` is for a clip whose wall duration is not knowable in advance — a
  // solve, an optimisation. It picks the rate that lands the whole run in
  // that many played seconds.
  const rate = fit ? Math.max(span / fit, 1) : speed;
  const step = rate / FPS;
  const picked = [];
  let cursor = 0;
  for (let t = first; t <= last + 1e-6; t += step) {
    while (cursor + 1 < frames.length && frames[cursor + 1].t <= t) cursor++;
    picked.push(frames[cursor].file);
  }
  return { picked, wall: span, rate };
}

// ─────────────────────────────────────────────────────────── encoding

/** A crop rectangle in css pixels, snapped even for the scaler. */
const even = (n) => Math.max(2, Math.round(n / 2) * 2);
function clampRect(r) {
  const x = Math.max(0, Math.min(VIEW.width - 2, Math.round(r.x)));
  const y = Math.max(0, Math.min(VIEW.height - 2, Math.round(r.y)));
  return {
    x,
    y,
    w: even(Math.min(r.w, VIEW.width - x)),
    h: even(Math.min(r.h, VIEW.height - y)),
  };
}

/** The union of some elements' boxes, padded, as a crop. */
async function boxOf(page, selectors, pad = 0) {
  const boxes = [];
  for (const sel of [].concat(selectors)) {
    const b = await page.locator(sel).first().boundingBox().catch(() => null);
    if (b) boxes.push(b);
  }
  if (boxes.length === 0) return { x: 0, y: 0, w: VIEW.width, h: VIEW.height };
  const x0 = Math.min(...boxes.map((b) => b.x)) - pad;
  const y0 = Math.min(...boxes.map((b) => b.y)) - pad;
  const x1 = Math.max(...boxes.map((b) => b.x + b.width)) + pad;
  const y1 = Math.max(...boxes.map((b) => b.y + b.height)) + pad;
  return clampRect({ x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
}
const FULL = { x: 0, y: 0, w: VIEW.width, h: VIEW.height };

/** Crop + scale the picked frames into a numbered png sequence. */
function stage(picked, crop, outWidth, dir) {
  const seq = path.join(dir, "seq");
  const png = path.join(dir, "png");
  for (const d of [seq, png]) {
    fs.rmSync(d, { recursive: true, force: true });
    fs.mkdirSync(d, { recursive: true });
  }
  picked.forEach((src, i) => {
    fs.copyFileSync(src, path.join(seq, `${String(i).padStart(5, "0")}.jpg`));
  });
  run(FFMPEG, [
    "-y", "-loglevel", "error",
    "-framerate", String(FPS),
    "-i", path.join(seq, "%05d.jpg"),
    "-vf", `crop=${crop.w}:${crop.h}:${crop.x}:${crop.y},scale=${outWidth}:-2:flags=lanczos`,
    path.join(png, "%05d.png"),
  ]);
  const files = fs.readdirSync(png).filter((f) => f.endsWith(".png")).sort()
    .map((f) => path.join(png, f));
  if (files.length === 0) throw new Error("ffmpeg staged no frames");
  return { png, files };
}

/** The delivery encode: one animated WebP, lossy, inter-frame predicted. */
function encodeWebp(files, out, quality) {
  const d = Math.round(1000 / FPS);
  run(IMG2WEBP, [
    "-loop", "0", "-kmin", "9", "-kmax", "80", "-mixed", "-sharp_yuv",
    "-d", String(d), "-lossy", "-q", String(quality), "-m", "6",
    ...files, "-o", out,
  ]);
  return bytes(out);
}

/** The alternatives, encoded from the same staged frames, for the record. */
function encodeAlternatives(files, dir, stem, quality) {
  const png = path.dirname(files[0]);
  const src = ["-framerate", String(FPS), "-i", path.join(png, "%05d.png")];
  const results = {};
  const gif = path.join(dir, `${stem}.gif`);
  run(FFMPEG, ["-y", "-loglevel", "error", ...src,
    "-vf", "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
    "-loop", "0", gif]);
  results.gif = bytes(gif);
  const mp4 = path.join(dir, `${stem}.mp4`);
  run(FFMPEG, ["-y", "-loglevel", "error", ...src,
    "-c:v", "libx264", "-crf", "26", "-preset", "veryslow",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]);
  results.mp4 = bytes(mp4);
  const webm = path.join(dir, `${stem}.webm`);
  run(FFMPEG, ["-y", "-loglevel", "error", ...src,
    "-c:v", "libvpx-vp9", "-crf", "36", "-b:v", "0", "-row-mt", "1", webm]);
  results.webm = bytes(webm);
  const apng = path.join(dir, `${stem}.apng`);
  run(FFMPEG, ["-y", "-loglevel", "error", ...src,
    "-plays", "0", "-f", "apng", apng]);
  results.apng = bytes(apng);
  const lossless = path.join(dir, `${stem}-lossless.webp`);
  run(IMG2WEBP, ["-loop", "0", "-kmin", "9", "-kmax", "80",
    "-d", String(Math.round(1000 / FPS)), "-lossless", "-m", "6",
    ...files, "-o", lossless]);
  results["webp-lossless"] = bytes(lossless);
  void quality;
  return results;
}

// ─────────────────────────────────────────────────────────── app driving
//
// The helpers below are the ones `frontend/e2e/playground.spec.ts` uses to
// click the right pixel — the projection is reimplemented there rather than
// imported, and reimplemented again here, on purpose: if the app's camera
// drifts, these clips point at the wrong thing and it is visible in the
// output, instead of both sides agreeing on something wrong.

const FOV_SCALE = 1.5;
const CAMERA = { yaw: Math.PI / 4, pitch: Math.atan(1 / Math.SQRT2), distance: 4.6, target: [0, 0, 0] };
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };

function cameraPosition(c = CAMERA) {
  const cp = Math.cos(c.pitch);
  return [
    c.target[0] + c.distance * cp * Math.sin(c.yaw),
    c.target[1] - c.distance * cp * Math.cos(c.yaw),
    c.target[2] + c.distance * Math.sin(c.pitch),
  ];
}
function projectToCss(world, canvas, camera = CAMERA) {
  const position = cameraPosition(camera);
  const forward = norm(sub(camera.target, position));
  const reference = Math.abs(forward[2]) > 0.999 ? [0, 1, 0] : [0, 0, 1];
  const right = norm(cross(forward, reference));
  const up = cross(right, forward);
  const delta = sub(world, position);
  const aspect = canvas.width / canvas.height;
  const divisor = FOV_SCALE * camera.distance;
  const u = dot(delta, right) / divisor;
  const v = dot(delta, up) / divisor;
  const px = (u / aspect + 0.5) * canvas.width;
  const py = (0.5 - v) * canvas.height;
  return {
    x: (px * canvas.clientWidth) / canvas.width,
    y: (py * canvas.clientHeight) / canvas.height,
  };
}
const canvasMetrics = (page) =>
  page.evaluate(() => {
    const c = document.querySelector("[data-testid=viewer-canvas]");
    const r = c.getBoundingClientRect();
    return {
      width: c.width, height: c.height,
      clientWidth: c.clientWidth, clientHeight: c.clientHeight,
      left: r.left, top: r.top,
    };
  });
/** A world point, in page coordinates. */
async function at(page, world) {
  const m = await canvasMetrics(page);
  const p = projectToCss(world, m);
  return { x: m.left + p.x, y: m.top + p.y };
}

/**
 * Where a draggable handle actually is, asked of the app's own hint bar.
 *
 * The readout under the viewport names the parameter under the pointer, and
 * it is computed by the same function that decides whether the handle is
 * drawn filled — so sweeping for it (as `frontend/e2e/drag.spec.ts` does)
 * finds the handle wherever the constraint solver has just put it, which a
 * fixed projection cannot after a drag has moved it.
 */
async function findHandle(page, near, radius = 90, phrase = "free parameter") {
  return page.evaluate(([cx, cy, r, want]) => {
    const canvas = document.querySelector("[data-testid=viewer-canvas]");
    const rect = canvas.getBoundingClientRect();
    const hint = () => document.querySelector("[data-testid=viewer-hint]")?.textContent ?? "";
    for (let ring = 0; ring <= r; ring += 3) {
      for (let a = 0; a < 360; a += ring === 0 ? 360 : 12) {
        const x = cx + ring * Math.cos((a * Math.PI) / 180);
        const y = cy + ring * Math.sin((a * Math.PI) / 180);
        if (x < rect.left + 4 || x > rect.right - 4 || y < rect.top + 4 || y > rect.bottom - 4) continue;
        canvas.dispatchEvent(new PointerEvent("pointermove", {
          clientX: x, clientY: y, bubbles: true, pointerId: 1,
        }));
        if (hint().includes(want)) return { x, y, hint: hint() };
      }
    }
    return null;
  }, [near.x, near.y, radius, phrase]);
}

async function waitForCompile(page, timeout = 120_000) {
  await page.waitForFunction(
    () => {
      const s = document.querySelector("[data-testid=status]")?.textContent ?? "";
      return s !== "" && s !== "Starting…"
        && !document.querySelector("[data-testid=toolbar-busy]");
    },
    null, { timeout, polling: 100 },
  );
}
const tid = (page, id) => page.locator(`[data-testid="${id}"]`);
async function clickIf(loc, ms = 0) {
  if (await loc.first().isVisible().catch(() => false)) {
    await loc.first().click();
    if (ms) await wait(ms);
    return true;
  }
  return false;
}
async function railTool(page, group, id) {
  const child = tid(page, id);
  for (let i = 0; i < 2 && !(await child.isVisible().catch(() => false)); i++) {
    await tid(page, `tool-group-${group}`).click();
    await wait(200);
  }
  await child.click();
}
const editorText = (page) =>
  page.evaluate(() => {
    const c = document.querySelector("[data-testid=editor] .cm-content");
    const v = c?.cmView?.view ?? c?.cmTile?.view;
    return v ? v.state.doc.toString() : (c?.innerText ?? "");
  });

/**
 * A pointer drag with the cursor drawn.
 *
 * A screencast has no cursor in it — the compositor does not composite one —
 * so a clip of a drag would otherwise be a solid moving with nothing moving
 * it. `cursor()` injects a ring that follows the real pointer events this
 * script sends, so what the reader sees is the actual pointer path.
 */
async function installCursor(page) {
  await page.evaluate(() => {
    if (document.getElementById("__cursor")) return;
    const dot = document.createElement("div");
    dot.id = "__cursor";
    dot.style.cssText = [
      "position:fixed", "z-index:2147483647", "pointer-events:none",
      "width:18px", "height:18px", "margin:-9px 0 0 -9px", "border-radius:50%",
      "border:2px solid rgba(20,20,24,.82)",
      "box-shadow:0 0 0 1.5px rgba(255,255,255,.9), inset 0 0 0 1.5px rgba(255,255,255,.9)",
      "background:rgba(20,20,24,.10)",
      "transition:width .08s,height .08s,background .08s",
      "left:-100px", "top:-100px",
    ].join(";");
    document.documentElement.appendChild(dot);
    const move = (e) => {
      dot.style.left = `${e.clientX}px`;
      dot.style.top = `${e.clientY}px`;
    };
    for (const type of ["pointermove", "mousemove", "pointerdown", "mousedown"]) {
      window.addEventListener(type, move, true);
    }
    window.addEventListener("pointerdown", () => {
      dot.style.width = dot.style.height = "12px";
      dot.style.margin = "-6px 0 0 -6px";
      dot.style.background = "rgba(20,20,24,.30)";
    }, true);
    window.addEventListener("pointerup", () => {
      dot.style.width = dot.style.height = "18px";
      dot.style.margin = "-9px 0 0 -9px";
      dot.style.background = "rgba(20,20,24,.10)";
    }, true);
  });
}

/** Glide the pointer somewhere, so the reader can follow it. */
async function glide(page, to, steps = 14, pause = 16) {
  for (let i = 1; i <= steps; i++) {
    const from = glide.last ?? to;
    await page.mouse.move(
      from.x + (to.x - from.x) * (i / steps),
      from.y + (to.y - from.y) * (i / steps),
    );
    await wait(pause);
  }
  glide.last = to;
}
/** Glide to an element's centre and click it. */
async function point(page, locator, { steps = 14, pause = 16, settle = 260 } = {}) {
  const b = await locator.first().boundingBox();
  if (!b) throw new Error("no box for pointer target");
  await glide(page, { x: b.x + b.width / 2, y: b.y + b.height / 2 }, steps, pause);
  await wait(settle);
}
async function clickAt(page, locator, opts) {
  await point(page, locator, opts);
  await page.mouse.down();
  await wait(90);
  await page.mouse.up();
}
async function dragTo(page, from, to, steps = 26, pause = 22) {
  await glide(page, from);
  await wait(200);
  await page.mouse.down();
  await wait(140);
  for (let i = 1; i <= steps; i++) {
    await page.mouse.move(
      from.x + (to.x - from.x) * (i / steps),
      from.y + (to.y - from.y) * (i / steps),
    );
    await wait(pause);
  }
  glide.last = to;
  await wait(240);
  await page.mouse.up();
}

/** A fresh page on the starter, settled, with the pointer parked off-frame. */
async function freshPage(context) {
  const page = await context.newPage();
  page.on("console", (m) => { if (m.type() === "error") console.log("   [page]", m.text().slice(0, 160)); });
  await page.goto(URL);
  await tid(page, "viewer-canvas").waitFor({ timeout: 180_000 });
  await clickIf(page.getByRole("button", { name: /^dismiss$/i }), 300);
  await waitForCompile(page, 180_000);
  await installCursor(page);
  glide.last = { x: VIEW.width - 8, y: VIEW.height - 8 };
  await page.mouse.move(glide.last.x, glide.last.y);
  await wait(400);
  return page;
}

/** Wheel the camera in over the viewport's centre. */
async function zoom(page, ticks) {
  const b = await tid(page, "viewer-canvas").boundingBox();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
  for (let i = 0; i < Math.abs(ticks); i++) {
    await page.mouse.wheel(0, ticks > 0 ? -260 : 260);
    await wait(160);
  }
  await wait(900);
}

/**
 * Park the right-hand column in the tray.
 *
 * Not staging: it is the dock's own minimise control, and it is how you get
 * the viewport the width of the desk when what you are looking at is the
 * model rather than its tree. It also stops a popover anchored to the toolbar
 * from being cropped against a panel it happens to overlap.
 */
async function parkColumn(page, tabs) {
  for (const tab of tabs) {
    const group = page.locator(`.dv-groupview:has([data-testid=window-tab-${tab}])`);
    const minimise = group.getByTestId("window-minimise").first();
    if (await minimise.isVisible().catch(() => false)) {
      await minimise.click();
      await wait(500);
    }
  }
  await wait(600);
}

/** Turn the construction overlay off — chrome, not data — for the render clips. */
async function overlayOff(page) {
  await clickIf(tid(page, "display-options"), 350);
  await clickIf(tid(page, "render-customize"), 350);
  const ov = tid(page, "toggle-construction-overlay");
  if (await ov.first().isVisible().catch(() => false)) {
    const input = (await ov.first().evaluate((el) => el.tagName)) === "INPUT"
      ? ov.first() : ov.first().locator("input").first();
    if (await input.isChecked().catch(() => true)) await input.uncheck();
  }
  await page.keyboard.press("Escape");
  await wait(500);
}

// ─────────────────────────────────────────────────────────── the clips
//
// Each clip is: `setup` (not recorded — get the app into the state the clip
// starts from, however long that takes), `crop` (evaluated after setup), then
// `act` (recorded). `speed` is wall seconds per played second.

/** fin2_tip_l, uv [-0.15, 0.85]: free, named, and in the parameter buffer. */
const FIN_TIP = [0.15, 0, 0.85];
/** The comb's near cap (y = -0.6), between two fins. */
const CAP_POINT = [0.3, -0.6, 0.09];
/**
 * Where a new sketch goes: on the floor in front of the part, which projects
 * to the empty left of the viewport. `-Y` is toward the camera at the session's
 * default iso, so nothing in the clip happens behind the heat sink.
 */
const SKETCH_ORIGIN = [0.0, -2.2, 0];

/** The desk's left two columns — editor and viewport — without the tray. */
const CODE_AND_VIEW = { x: 0, y: 0, w: 1120, h: 876 };
/** The viewport column alone, plus whatever sits to its right. */
const VIEW_AND_PANEL = { x: 460, y: 0, w: 980, h: 876 };

const CLIPS = [
  {
    name: "parameter-drag",
    width: 1120,
    quality: 76,
    async setup(page) {
      await tid(page, "mode-vertex").click();
      await wait(400);
      await zoom(page, 3);
      // Select the handle before recording: selecting scrolls the editor to
      // the `Vector2` literal this drag rewrites, and that scroll is a jump,
      // not motion worth four seconds.
      const m = await canvasMetrics(page);
      const found = await findHandle(page,
        { x: m.left + m.clientWidth / 2, y: m.top + m.clientHeight / 2 }, 400, "fin2_tip_l");
      if (!found) throw new Error("no fin2_tip_l handle in the viewport");
      this.tip = { x: found.x, y: found.y };
      await page.mouse.move(this.tip.x, this.tip.y);
      await page.mouse.down(); await wait(90); await page.mouse.up();
      await wait(1200);
      glide.last = { x: this.tip.x + 130, y: this.tip.y + 90 };
      await page.mouse.move(glide.last.x, glide.last.y);
      await wait(600);
    },
    crop: () => CODE_AND_VIEW,
    async act(page) {
      const tip = this.tip;
      // One continuous drag, up and back. It ends where it began — so the
      // loop does not jump — and, because the pointer never comes up, every
      // frame between is a parameter-buffer write and not a recompile, which
      // is the claim the clip is making.
      await glide(page, tip, 14, 26);
      await wait(500);
      await page.mouse.down();
      await wait(200);
      const ease = (u) => (1 - Math.cos(u * Math.PI)) / 2;
      const sweep = async (from, to, steps) => {
        for (let i = 1; i <= steps; i++) {
          const u = ease(i / steps);
          await page.mouse.move(from.x + (to.x - from.x) * u, from.y + (to.y - from.y) * u);
          await wait(24);
        }
      };
      const top = { x: tip.x - 6, y: tip.y - 150 };
      await sweep(tip, top, 26);
      await wait(650);
      await sweep(top, tip, 26);
      await wait(350);
      // Stop on the release. The compile it triggers replaces the document,
      // which sends the editor back to line 1 — a jump, and one that would
      // make the loop restart somewhere the clip never was.
      await page.mouse.up();
      glide.last = tip;
      await wait(500);
    },
    async after(page) { await waitForCompile(page); },
  },

  {
    name: "face-sketch",
    width: 1440,
    quality: 74,
    async setup(page) {
      await railTool(page, "create", "tool-face");
      await wait(600);
    },
    crop: () => FULL,
    async act(page) {
      const cap = await at(page, CAP_POINT);
      await glide(page, { x: cap.x + 150, y: cap.y - 110 }, 12, 22);
      await glide(page, cap, 16, 26);
      await page.waitForFunction(
        () => /cap\('-'\)/.test(document.querySelector("[data-testid=viewer-hint]")?.textContent ?? ""),
        null, { timeout: 30_000 },
      ).catch(() => console.log("   !! hint never named the face"));
      await wait(1100);
      await page.mouse.down(); await wait(90); await page.mouse.up();
      await page.waitForFunction(
        () => {
          const c = document.querySelector("[data-testid=editor] .cm-content");
          const v = c?.cmView?.view ?? c?.cmTile?.view;
          return (v ? v.state.doc.toString() : "").includes("SketchPlane.on(sink.cap('-'))");
        }, null, { timeout: 60_000 },
      );
      await waitForCompile(page);
      await wait(1400);
    },
  },

  {
    name: "sketch-solve",
    width: 0,
    quality: 74,
    // Seven deliberate clicks is a lot of clip; a third faster than life still
    // reads as someone working rather than a montage.
    speed: 1.35,
    async setup(page) {
      await railTool(page, "create", "tool-sketch");
      const m = await canvasMetrics(page);
      const d = projectToCss(SKETCH_ORIGIN, m);
      await page.mouse.click(m.left + d.x, m.top + d.y);
      await waitForCompile(page);
      await tid(page, "sketch-panel").waitFor({ timeout: 60_000 });
      await tid(page, "mode-vertex").click();
      await wait(600);
      this.origin = SKETCH_ORIGIN;
    },
    crop: () => VIEW_AND_PANEL,
    async act(page) {
      const m = await canvasMetrics(page);
      const world = (dx, dy) => {
        const p = projectToCss([this.origin[0] + dx, this.origin[1] + dy, 0], m);
        return { x: m.left + p.x, y: m.top + p.y };
      };
      // ±0.6: the corners of the square the sketch tool drops, which is what
      // `frontend/e2e/playground.spec.ts` clicks. Anywhere else is empty plane.
      const first = world(-0.6, -0.6);
      const second = world(0.6, -0.6);
      const tap = async (at) => {
        await glide(page, at, 14, 22);
        await wait(220);
        await page.mouse.down(); await wait(80); await page.mouse.up();
      };

      await tap(first);
      await wait(360);
      await clickAt(page, tid(page, "constraint-fix"), { steps: 14 });
      await page.waitForFunction(
        () => /fix/.test(document.querySelector("[data-testid=sketch-panel]")?.textContent ?? ""),
        null, { timeout: 60_000 });
      await waitForCompile(page);
      await wait(700);

      await clickAt(page, tid(page, "constraint-distance"), { steps: 12 });
      await page.waitForFunction(
        () => /second point/i.test(document.querySelector("[data-testid=status]")?.textContent ?? ""),
        null, { timeout: 30_000 });
      await wait(300);
      await tap(second);
      await page.waitForFunction(
        () => /distance/.test(document.querySelector("[data-testid=sketch-panel]")?.textContent ?? ""),
        null, { timeout: 60_000 });
      await waitForCompile(page);
      await wait(800);

      // Retarget the dimension, then solve for it. Without this the sketch is
      // already satisfied, the solver reports zero iterations, and nothing on
      // screen moves — which is the opposite of what the clip is about.
      await clickAt(page, tid(page, "constraint-label-1"), { steps: 12 });
      await wait(400);
      await tid(page, "constraint-value-1").fill("0.7");
      await wait(300);
      await tid(page, "constraint-value-1").press("Enter");
      await waitForCompile(page);
      await wait(700);

      await clickAt(page, tid(page, "solver-toggle"), { steps: 12 });
      await tid(page, "solver-panel").waitFor({ timeout: 30_000 });
      await wait(500);
      await clickAt(page, tid(page, "constraint-solve"), { steps: 12 });
      await tid(page, "solver-loss-chart").waitFor({ timeout: 120_000 });
      await waitForCompile(page);
      await wait(1800);
    },
  },

  {
    name: "sdf-sweep",
    width: 0,
    quality: 78,
    async setup(page) {
      await parkColumn(page, ["objects", "materials"]);
      await zoom(page, 2);
      await clickIf(tid(page, "display-options"), 400);
      await tid(page, "render-sdf").waitFor({ timeout: 30_000 });
      await tid(page, "sdf-slice").click();
      await tid(page, "sdf-legend").waitFor({ timeout: 30_000 });
      await tid(page, "sdf-fraction").fill("0.08");
      await wait(1600);
    },
    crop: (page) => boxOf(page, ["[data-testid=viewer-canvas]", "[data-testid=render-popover]"], 6),
    async act(page) {
      // The plane is dragged, not typed: the readout over the viewport and the
      // coordinate in the panel are both functions of the slider, and the
      // point of the clip is that all three move together.
      const b = await tid(page, "sdf-fraction").first().boundingBox();
      const y = b.y + b.height / 2;
      const at = (f) => ({ x: b.x + 6 + (b.width - 12) * f, y });
      await glide(page, at(0.08), 14, 24);
      await wait(500);
      await page.mouse.down();
      await wait(200);
      const sweepTo = async (from, to, steps, pause) => {
        for (let i = 1; i <= steps; i++) {
          const u = (1 - Math.cos((i / steps) * Math.PI)) / 2;
          const f = from + (to - from) * u;
          await page.mouse.move(at(f).x, at(f).y);
          await wait(pause);
        }
      };
      await sweepTo(0.08, 0.94, 34, 46);
      await wait(600);
      await sweepTo(0.94, 0.08, 30, 40);
      await wait(300);
      await page.mouse.up();
      glide.last = at(0.08);
      await wait(700);
    },
  },

  {
    name: "solve-field",
    width: 1440,
    quality: 74,
    // No warm pass. The result the server has already computed is fetched
    // back into a brand-new session the moment Simulate opens, so the only
    // way to record a field arriving is to be the run that produces it —
    // which means this clip has to be the first solve the server is asked
    // for. `--only solve-field` on a server that has already solved will
    // open on a finished field; restart the server first.
    async setup(page) {
      await tid(page, "editmode-simulate").click();
      await tid(page, "simulate-run-sink-conduction").waitFor({ timeout: 60_000 });
      await wait(1400);
    },
    crop: () => FULL,
    async act(page) {
      await clickAt(page, tid(page, "simulate-run-sink-conduction"), { steps: 18, pause: 22 });
      await tid(page, "simulate-legend").waitFor({ timeout: 900_000 });
      await wait(3200);
    },
  },

  {
    name: "optimize-converge",
    width: 0,
    quality: 74,
    // A real gradient-descent run on a FEM objective: twelve steps, each a
    // mesh and a solve and an adjoint, six or seven minutes of wall clock.
    // `fit` plays all of it in ten seconds — every frame is one the app drew,
    // sampled, which is the only honest way to show a thing that slow.
    fit: 10,
    async setup(page) {
      await clickIf(tid(page, "window-tab-optimize"), 600);
      await parkColumn(page, ["objects"]);
      await tid(page, "optimize-run-cool-sink").waitFor({ timeout: 60_000 });
      await wait(1000);
    },
    crop: () => VIEW_AND_PANEL,
    async act(page) {
      await clickAt(page, tid(page, "optimize-run-cool-sink"), { steps: 18, pause: 22 });
      await tid(page, "optimize-result-cool-sink").waitFor({ timeout: 1_800_000 });
      await wait(20_000);
    },
  },

  {
    name: "orbit",
    width: 0,
    quality: 70,
    async setup(page) {
      await zoom(page, 3);
      await wait(600);
    },
    crop: (page) => boxOf(page, ["[data-testid=viewer-canvas]"], 4),
    async act(page) {
      // A closed rectangle in pointer space, dragged in one go. Left-drag is
      // orbit, and the mapping from pointer delta to yaw and pitch is linear,
      // so a path that returns to its start returns the camera to its start:
      // the loop closes on the frame it opened on. The lower two legs take
      // the camera under the floor, which is the half of the sphere the new
      // lighting made readable.
      const b = await tid(page, "viewer-canvas").boundingBox();
      const from = { x: b.x + 110, y: b.y + b.height * 0.42 };
      // Measured on this build: pointer +y raises the camera, at about a
      // quarter degree per pixel, and the pitch clamps at the poles — so the
      // legs go *down* first and stop short of -90, or the return leg would
      // not undo the outbound one and the loop would not close.
      const legs = [[300, 0], [0, -250], [300, 0], [0, 250], [-600, 0]];
      await glide(page, from, 14, 26);
      await wait(500);
      await page.mouse.down();
      await wait(180);
      let at = { ...from };
      for (const [dx, dy] of legs) {
        const steps = Math.max(10, Math.round(Math.hypot(dx, dy) / 11));
        const to = { x: at.x + dx, y: at.y + dy };
        for (let i = 1; i <= steps; i++) {
          const u = (1 - Math.cos((i / steps) * Math.PI)) / 2;
          await page.mouse.move(at.x + dx * u, at.y + dy * u);
          await wait(20);
        }
        at = to;
        await wait(260);
      }
      await page.mouse.up();
      glide.last = at;
      await wait(900);
    },
  },

  {
    name: "scene-motor-shield",
    width: 1440,
    quality: 74,
    // Opening it is a real 25-70 s compile (helical channel, bolt circle,
    // knurl, 41 free parameters); `fit` plays the whole open in ten seconds.
    fit: 10,
    crop: () => FULL,
    async act(page) {
      await clickAt(page, tid(page, "menu-file"), { steps: 14 });
      await wait(400);
      await clickAt(page, tid(page, "menu-file-open"), { steps: 10 });
      await tid(page, "scenes-panel").waitFor({ timeout: 60_000 });
      await wait(2200);
      await clickAt(page, tid(page, "scene-open-motor_shield.py"), { steps: 16 });
      await page.waitForFunction(
        () => /motor_shield/.test(document.querySelector("[data-testid=menu-scene-name]")?.textContent ?? ""),
        null, { timeout: 300_000 });
      await waitForCompile(page, 300_000);
      await wait(1200);
      await clickAt(page, tid(page, "window-tab-editor"), { steps: 12 });
      await wait(2500);
      // One slow turn, so the helix and the bolt circle are both seen.
      const b = await tid(page, "viewer-canvas").boundingBox();
      const from = { x: b.x + b.width * 0.3, y: b.y + b.height * 0.5 };
      await glide(page, from, 14, 26);
      await page.mouse.down();
      await wait(160);
      for (let i = 1; i <= 34; i++) {
        await page.mouse.move(from.x + (260 * i) / 34, from.y - (60 * i) / 34);
        await wait(26);
      }
      await page.mouse.up();
      glide.last = { x: from.x + 260, y: from.y - 60 };
      await wait(2500);
    },
  },

  {
    name: "scene-duct-sink",
    width: 1440,
    quality: 74,
    fit: 9,
    async setup(page) {
      await tid(page, "menu-file").click();
      await tid(page, "menu-file-open").click();
      await tid(page, "scenes-panel").waitFor({ timeout: 60_000 });
      await wait(1500);
      await tid(page, "scene-open-duct_sink.py").click();
      await page.waitForFunction(
        () => /duct_sink/.test(document.querySelector("[data-testid=menu-scene-name]")?.textContent ?? ""),
        null, { timeout: 300_000 });
      await waitForCompile(page, 300_000);
      await tid(page, "editmode-simulate").click();
      await tid(page, "simulate-run-duct-cooling").waitFor({ timeout: 120_000 });
      await wait(1800);
    },
    crop: () => FULL,
    async act(page) {
      await clickAt(page, tid(page, "simulate-run-duct-cooling"), { steps: 18, pause: 22 });
      await tid(page, "simulate-legend").waitFor({ timeout: 1_800_000 });
      await wait(9000);
    },
  },

  {
    name: "windows-dock",
    width: 1120,
    quality: 72,
    async setup(page) {
      await wait(400);
    },
    crop: () => FULL,
    async act(page) {
      const tab = tid(page, "window-tab-objects");
      const from = await tab.first().boundingBox();
      const start = { x: from.x + from.width / 2, y: from.y + from.height / 2 };
      const editor = await tid(page, "window-tab-editor").first().boundingBox();

      // 1. drag Objects onto the editor's tab strip: two windows, one strip.
      await glide(page, start, 14, 22);
      await wait(320);
      await page.mouse.down();
      await wait(160);
      const drop = { x: editor.x + editor.width / 2, y: editor.y + editor.height / 2 };
      for (let i = 1; i <= 20; i++) {
        await page.mouse.move(
          start.x + (drop.x - start.x) * (i / 20),
          start.y + (drop.y - start.y) * (i / 20),
        );
        await wait(28);
      }
      glide.last = drop;
      await wait(520);
      await page.mouse.up();
      await wait(1100);

      // 2. float it back out over the desk.
      await clickAt(page, page.locator(".dv-groupview:has([data-testid=window-tab-objects])")
        .getByTestId("window-float"), { steps: 14 });
      await wait(1400);

      // 3. park Materials in the tray, and bring it back.
      await clickAt(page, page.locator(".dv-groupview:has([data-testid=window-tab-materials])")
        .getByTestId("window-minimise").first(), { steps: 16 });
      await wait(1100);
      await clickAt(page, tid(page, "window-restore-materials"), { steps: 14 });
      await wait(1400);
    },
  },
];

// ─────────────────────────────────────────────────────────── main

if (flag("list")) {
  for (const c of CLIPS) console.log(c.name);
  process.exit(0);
}
const only = (arg("only", "") || "").split(",").filter(Boolean);
const chosen = only.length ? CLIPS.filter((c) => only.includes(c.name)) : CLIPS;
if (chosen.length === 0) { console.error("no clip matched --only"); process.exit(1); }

const staticIndex = path.join(repo, "cadjoint/viewer/static/index.html");
if (!fs.existsSync(staticIndex)) {
  console.error("no built frontend at cadjoint/viewer/static/index.html");
  process.exit(1);
}
console.log("bundle mtime:", fs.statSync(staticIndex).mtime.toISOString());

let server = null;
const alive = await fetch(URL).then((r) => r.ok).catch(() => false);
if (alive) {
  console.log("reusing the server already on", URL);
} else {
  server = spawn(path.join(repo, ".venv/bin/python"),
    ["-m", "cadjoint.viewer.playground", "--port", String(PORT)],
    { cwd: repo, stdio: ["ignore", "pipe", "pipe"] });
  server.stdout.on("data", (b) => process.stdout.write("[server] " + b));
  server.stderr.on("data", (b) => { const s = String(b); if (/error|Traceback/i.test(s)) process.stderr.write("[server] " + s); });
  for (let i = 0; i < 240; i++) {
    if (await fetch(URL).then((r) => r.ok).catch(() => false)) break;
    await wait(500);
  }
  console.log("server up at", URL);
}
const stop = () => { if (server) try { server.kill("SIGTERM"); } catch {} };
process.on("exit", stop);
process.on("SIGINT", () => { stop(); process.exit(1); });

// The server warms every shipped scene in worker subprocesses at launch.
// Driving it while those run puts two workers on one compile-cache lock, and
// the warning that produces lands in the editor's stderr pane — in the clip.
{
  const token = (await (await fetch(URL + "api/session")).json()).token;
  for (let i = 0; i < 240; i++) {
    const jobs = await (await fetch(URL + "api/jobs", { headers: { "X-Cadjoint-Token": token } }))
      .json().catch(() => null);
    const running = jobs?.totals?.running ?? 0;
    if (jobs && running === 0 && i > 1) break;
    if (i % 6 === 0) console.log("warm-up: running jobs", running);
    await wait(2000);
  }
}

const browser = await chromium.launch({
  headless: !HEADED,
  args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal",
         "--ignore-gpu-blocklist", "--force-color-profile=srgb",
         "--hide-scrollbars", "--force-prefers-reduced-motion"],
});
/**
 * One browser context per clip.
 *
 * Layouts, the last render preset and the id of the last solved job all live
 * in the page's own storage, so a shared context would let one clip open on
 * the state the previous one left — and, for `solve-field`, on a field that
 * is already there.
 */
const newContext = () => browser.newContext({ viewport: VIEW, deviceScaleFactor: DSF });

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.mkdirSync(WORK, { recursive: true });
const report = [];

for (const clip of chosen) {
  const t0 = Date.now();
  console.log(`\n── ${clip.name}`);
  const dir = path.join(WORK, clip.name);
  // A warm pass in a context of its own: it pays the server's cold cost — a
  // mesh, a factorisation, a JIT — and throws away every trace of having done
  // so, leaving only the server-side cache the recorded run benefits from.
  if (clip.warm) {
    const warmContext = await newContext();
    try {
      await clip.warm.call(clip, await freshPage(warmContext));
    } finally {
      await warmContext.close().catch(() => {});
    }
    console.log("   warmed");
  }
  const context = await newContext();
  const page = await freshPage(context);
  try {
    if (clip.setup) await clip.setup.call(clip, page);
    const crop = clampRect(await clip.crop(page));
    // width 0 means "deliver at capture scale" — never upscale a crop, the
    // extra bytes buy nothing a reader can see.
    const width = even(clip.width ? Math.min(clip.width, crop.w) : crop.w);
    console.log(`   crop ${crop.w}x${crop.h} +${crop.x}+${crop.y} -> ${width}px`);

    const cast = new Screencast(page, path.join(dir, "raw"));
    await cast.start();
    await wait(250);
    await clip.act.call(clip, page);
    await wait(250);
    const frames = await cast.stop();
    if (clip.after) await clip.after.call(clip, page);
    console.log(`   ${frames.length} frames captured`);

    const { picked, wall, rate } = resample(frames,
      { speed: clip.speed ?? 1, fit: clip.fit ?? 0, trim: clip.trim, until: cast.t1 });
    const { files } = stage(picked, crop, width, dir);
    const out = path.join(OUT_DIR, `${clip.name}.webp`);
    const size = encodeWebp(files, out, clip.quality ?? 74);
    const played = files.length / FPS;
    console.log(`   ${files.length} frames · ${played.toFixed(1)}s played`
      + (rate > 1.02 ? ` (${wall.toFixed(0)}s wall, ${rate.toFixed(1)}x)` : "")
      + ` · ${kb(size)}`);

    const row = { name: clip.name, frames: files.length, played, wall, size, crop, width };
    if (flag("formats")) {
      row.alternatives = encodeAlternatives(files, dir, clip.name, clip.quality ?? 74);
      for (const [k, v] of Object.entries(row.alternatives)) console.log(`   ${k.padEnd(14)} ${kb(v)}`);
    }
    report.push(row);
  } catch (error) {
    console.error(`   !! ${clip.name} failed:`, error.message);
    report.push({ name: clip.name, error: error.message });
  } finally {
    await context.close().catch(() => {});
    if (!flag("keep-frames")) fs.rmSync(path.join(dir, "raw"), { recursive: true, force: true });
    console.log(`   ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
}

await browser.close();
stop();

// ─────────────────────────────────────────────────────────── the record

const ok = report.filter((r) => !r.error);
const total = ok.reduce((s, r) => s + r.size, 0);
console.log(`\n${ok.length}/${report.length} clips · ${(total / 1024 / 1024).toFixed(2)} MB total`);
for (const r of report.filter((x) => x.error)) console.log(`  FAILED ${r.name}: ${r.error}`);

const lines = [
  "# README motion",
  "",
  "Regenerated by one command, against the real app:",
  "",
  "```",
  "node research/design/motion/animate.mjs",
  "```",
  "",
  `Captured at ${VIEW.width}x${VIEW.height} css px, device scale ${DSF}, played at ${FPS} fps.`,
  "Do not hand-edit anything in this directory.",
  "",
  "| clip | played | frames | delivered | bytes |",
  "| --- | ---: | ---: | ---: | ---: |",
  ...ok.map((r) => `| \`${r.name}.webp\` | ${r.played.toFixed(1)}s${r.wall && r.wall > r.played * 1.5 ? ` (${r.wall.toFixed(0)}s wall)` : ""} | ${r.frames} | ${r.width}px wide | ${kb(r.size)} |`),
  `| **total** | | | | **${(total / 1024 / 1024).toFixed(2)} MB** |`,
  "",
];
if (ok.some((r) => r.alternatives)) {
  const kinds = Object.keys(ok.find((r) => r.alternatives).alternatives);
  lines.push("## Format evidence", "",
    "The same staged frames, encoded every way. `webp` is the column the",
    "README ships.", "",
    `| clip | webp | ${kinds.join(" | ")} |`,
    `| --- | ---: | ${kinds.map(() => "---:").join(" | ")} |`,
    ...ok.filter((r) => r.alternatives).map((r) =>
      `| ${r.name} | ${kb(r.size)} | ${kinds.map((k) => kb(r.alternatives[k])).join(" | ")} |`),
    "");
}
fs.writeFileSync(path.join(OUT_DIR, "README.md"), lines.join("\n"));
console.log("wrote", path.join(OUT_DIR, "README.md"));
process.exit(report.some((r) => r.error) ? 1 : 0);
