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
 *   --table         rewrite docs/assets/motion/README.md from the files on disk, and exit
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
  /**
   * `everyNthFrame` thins the capture at the source. A clip of a slow run
   * that is later time-lapsed forty-fold needs a frame a second, not ninety,
   * and encoding ninety full-size JPEGs a second is enough load to starve
   * the very worker the clip is watching (the optimisation clip once ran at
   * a fifteenth of its normal speed under it).
   */
  async start({ everyNthFrame = 1 } = {}) {
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
      everyNthFrame,
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
      // The angular step shrinks with the ring so the samples stay about
      // three pixels apart along the arc; a fixed step walks past a handle
      // once the ring is wider than a few dozen pixels.
      const step = ring === 0 ? 360 : Math.max(0.5, (3 / ring) * (180 / Math.PI));
      for (let a = 0; a < 360; a += step) {
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

// ─────────────────────────────────────────────────────────── the clips
//
// Each clip is: `setup` (not recorded — get the app into the state the clip
// starts from, however long that takes), `crop` (evaluated after setup), then
// `act` (recorded). `speed` is wall seconds per played second.

/** The desk's left two columns — editor and viewport — without the tray. */
const CODE_AND_VIEW = { x: 0, y: 0, w: 1120, h: 876 };
/** The viewport column alone, plus whatever sits to its right. */
const VIEW_AND_PANEL = { x: 460, y: 0, w: 980, h: 876 };
const FULL = { x: 0, y: 0, w: VIEW.width, h: VIEW.height };

/**
 * Open a scene from the File menu, not recorded. The compile it triggers is a
 * real one (2-3 s on the direct shader path), so the clip starts warm.
 */
async function openScene(page, file) {
  await tid(page, "menu-file").click();
  await wait(300);
  await tid(page, "menu-file-open").click();
  await tid(page, "scenes-panel").waitFor({ timeout: 60_000 });
  await wait(800);
  await tid(page, `scene-open-${file}`).click();
  const stem = file.replace(/\.py$/, "");
  await page.waitForFunction(
    (s) => (document.querySelector("[data-testid=menu-scene-name]")?.textContent ?? "").includes(s),
    stem, { timeout: 300_000 });
  await waitForCompile(page, 300_000);
  await tid(page, "window-tab-editor").click();
  await wait(900);
}

/**
 * One turn of the camera and one pull on a named handle, on a scene opened
 * from `scenes/`. The two motions both return to where they started — the
 * orbit is a closed rectangle in pointer space and the drag comes back down
 * the same path — so the loop closes, and the pointer never lifts during the
 * drag, so every frame of it is a parameter-buffer write and not a recompile.
 * The clip stops before the release, whose recompile would otherwise send
 * the editor back to line 1 mid-loop.
 */
function orbitAndDrag({
  name, scene, handle, zoomTicks = 2, pull = { x: 40, y: -130 }, width = 1120,
  // The orbit, as pointer legs before the drag and after it, in degrees of
  // yaw (x) and pitch (y). The app publishes its camera, so the pixels per
  // degree are measured on the build being filmed rather than assumed: the
  // rate has changed before, and a turn that is not the angle it claims to
  // be puts the edit on the wrong side of the part. Before and after should
  // sum to a whole number of turns, so the loop closes.
  legs = [[110, 0], [0, -70], [-110, 0], [0, 70]],
  legsAfter = [],
}) {
  const ease = (u) => (1 - Math.cos(u * Math.PI)) / 2;
  const sweep = async (page, from, to, steps, pause = 24) => {
    for (let i = 1; i <= steps; i++) {
      const u = ease(i / steps);
      await page.mouse.move(from.x + (to.x - from.x) * u, from.y + (to.y - from.y) * u);
      await wait(pause);
    }
  };
  const locate = async (page) => {
    const m = await canvasMetrics(page);
    const found = await findHandle(page,
      { x: m.left + m.clientWidth / 2, y: m.top + m.clientHeight / 2 }, 460, handle);
    if (!found) throw new Error(`no ${handle} handle in the viewport of ${scene}`);
    return { x: found.x, y: found.y };
  };
  return {
    name,
    width,
    quality: 76,
    async setup(page) {
      await openScene(page, scene);
      await zoom(page, zoomTicks);
      await tid(page, "mode-vertex").click();
      await wait(400);
      // Take the orbit once now, off film, to learn where the handle will be
      // when the drag comes, then take it back: a search on film is seconds
      // of a still frame. Selecting the handle there also scrolls the editor
      // to the literal the drag rewrites, and that scroll is a jump.
      const orbit = async (path, pause, settle = 0) => {
        const b = await tid(page, "viewer-canvas").boundingBox();
        let at = { x: b.x + b.width * 0.22, y: b.y + b.height * 0.36 };
        await page.mouse.move(at.x, at.y);
        await page.mouse.down();
        await wait(120);
        for (const [dx, dy] of path) {
          const to = { x: at.x + dx, y: at.y + dy };
          await sweep(page, at, to, Math.max(10, Math.round(Math.hypot(dx, dy) / 11)), pause);
          at = to;
          if (settle) await wait(settle);
        }
        await page.mouse.up();
        await wait(300);
        return at;
      };
      // Pixels per degree, measured: a hundred pixels, and back.
      const camera = () => page.evaluate(() => window.__cadjointCamera());
      const c0 = await camera();
      await orbit([[100, 0]], 6);
      const c1 = await camera();
      await orbit([[-100, 0]], 6);
      const perDegree = 100 / Math.abs(((c1.yaw - c0.yaw) * 180) / Math.PI);
      const sign = Math.sign(c1.yaw - c0.yaw) || 1;
      // Yaw was read from a leftward-turning rate; a positive degree means
      // the same direction that a rightward pointer drag gives.
      this.px = ([yaw, pitch]) => [yaw * perDegree, -pitch * perDegree * sign * sign];
      this.orbit = orbit;
      if (legs.length) await orbit(legs.map(this.px), 8);
      const tip = await locate(page);
      this.tip = tip;
      await page.mouse.move(tip.x, tip.y);
      await page.mouse.down(); await wait(90); await page.mouse.up();
      await wait(1200);
      if (legs.length) await orbit(legs.map(this.px).map(([dx, dy]) => [-dx, -dy]).reverse(), 8);
      glide.last = { x: tip.x + 150, y: tip.y + 120 };
      await page.mouse.move(glide.last.x, glide.last.y);
      await wait(600);
    },
    crop: () => CODE_AND_VIEW,
    async act(page) {
      // The orbit, up to the drag.
      if (legs.length) {
        const b = await tid(page, "viewer-canvas").boundingBox();
        const from = { x: b.x + b.width * 0.22, y: b.y + b.height * 0.36 };
        await glide(page, from, 14, 26);
        await wait(400);
        await page.mouse.down();
        await wait(180);
        let at = { ...from };
        for (const [dx, dy] of legs.map(this.px)) {
          const to = { x: at.x + dx, y: at.y + dy };
          await sweep(page, at, to, Math.max(10, Math.round(Math.hypot(dx, dy) / 11)), 20);
          at = to;
          await wait(220);
        }
        await page.mouse.up();
        await wait(700);
      }
      // The drag: out along `pull` and back, pointer down throughout. The
      // orbit closed, so the handle is where setup found it; the short
      // re-sweep only absorbs a pixel of drift, and is not the seconds-long
      // search a full one would put on film.
      const found = await findHandle(page, this.tip, 40, handle);
      const tip = found ? { x: found.x, y: found.y } : await locate(page);
      await glide(page, tip, 14, 26);
      await wait(450);
      await page.mouse.down();
      await wait(200);
      const out = { x: tip.x + pull.x, y: tip.y + pull.y };
      await sweep(page, tip, out, 26);
      await wait(650);
      await sweep(page, out, tip, 26);
      await wait(350);
      glide.last = tip;
      if (!legsAfter.length) return;
      // The release is on film here: the orbit that follows is a drag on the
      // same canvas, so the handle has to be let go first. Its recompile is
      // the values-only path, and the editor keeps its place through it.
      await page.mouse.up();
      this.released = true;
      await wait(700);
      const c = await tid(page, "viewer-canvas").boundingBox();
      let here = { x: c.x + c.width * 0.22, y: c.y + c.height * 0.36 };
      await glide(page, here, 14, 26);
      await page.mouse.down();
      await wait(180);
      for (const [dx, dy] of legsAfter.map(this.px)) {
        const to = { x: here.x + dx, y: here.y + dy };
        await sweep(page, here, to, Math.max(10, Math.round(Math.hypot(dx, dy) / 11)), 20);
        here = to;
        await wait(220);
      }
      await page.mouse.up();
      glide.last = here;
      await wait(500);
    },
    // Without a turn home the release stays off film, so its recompile
    // cannot land inside the loop.
    async after(page) {
      if (!this.released) await page.mouse.up();
      await waitForCompile(page);
    },
  };
}

const CLIPS = [
  // The gusset's tip slid up the web, from the opening view where the rib
  // is nearest the camera: the smooth union re-blends the joint as it moves.
  // Then one full turn, so the loop closes on the frame it opened on.
  orbitAndDrag({ name: "bracket-orbit-drag", scene: "bracket.py", handle: "rib_tip",
                 legs: [], legsAfter: [[360, 0]], pull: { x: 30, y: -120 } }),
  {
    name: "end-cap-solve",
    width: 1440,
    quality: 74,
    // The end cap's declared thermal study, solved on film. The result the
    // server has computed once is fetched back into any later session the
    // moment Simulate opens, so the only way to record a field arriving is
    // to be the run that produces it: this clip must be the first solve the
    // server is asked for, and the recorder starts a fresh server per run.
    // `fit` plays the whole thing in ten seconds: the solve — mesh, assembly,
    // field — and then as long again with the field on a turning part, so
    // the result is on screen for half the clip and not the last frames.
    fit: 10,
    async setup(page) {
      await openScene(page, "end_cap.py");
      await zoom(page, 1);
      await tid(page, "editmode-simulate").click();
      await tid(page, "simulate-run-cap-conduction").waitFor({ timeout: 60_000 });
      await wait(1400);
    },
    crop: () => FULL,
    async act(page) {
      const t0 = Date.now();
      await clickAt(page, tid(page, "simulate-run-cap-conduction"), { steps: 18, pause: 22 });
      await tid(page, "simulate-legend").waitFor({ timeout: 900_000 });
      const solved = Date.now() - t0;
      await wait(1500);
      // One slow turn, for as long as the solve took.
      const b = await tid(page, "viewer-canvas").boundingBox();
      const from = { x: b.x + b.width * 0.35, y: b.y + b.height * 0.55 };
      await glide(page, from, 14, 26);
      await page.mouse.down();
      await wait(160);
      const steps = 60;
      for (let i = 1; i <= steps; i++) {
        await page.mouse.move(from.x + (300 * i) / steps, from.y - (70 * i) / steps);
        await wait(Math.max(20, solved / steps));
      }
      await page.mouse.up();
      glide.last = { x: from.x + 300, y: from.y - 70 };
      await wait(2000);
    },
  },
  {
    name: "heat-sink-optimize",
    width: 0,
    quality: 74,
    // A real gradient-descent run on a FEM objective, every step a mesh, a
    // solve and an adjoint, then the app's own replay of the trajectory.
    // `fit` plays all of it in twelve seconds — every frame is one the app
    // drew, sampled, which is the only honest way to show a thing that slow
    // — and the capture is thinned to one frame in six so the recorder does
    // not compete with the run for the machine.
    fit: 12,
    captureEvery: 6,
    async setup(page) {
      await clickIf(tid(page, "window-tab-optimize"), 600);
      await parkColumn(page, ["objects"]);
      await tid(page, "optimize-run-cool-sink").waitFor({ timeout: 60_000 });
      // The scene declares twelve steps at a cautious rate, which moves the
      // comb by a few percent: right for a default, invisible on film. The
      // panel's own fields raise the rate and shorten the run (each is a
      // patch to the declaration, like any edit). Eight steps at 0.02 take
      // the objective down about nine percent and complete; the tenth step
      // at that rate meshes a fin into sliver tets and the solve fails.
      for (const [field, value] of [["optimize-steps-cool-sink", "8"], ["optimize-lr-cool-sink", "0.02"]]) {
        const input = tid(page, field).first();
        await input.fill(value);
        await input.press("Enter");
        await waitForCompile(page);
        await wait(600);
      }
      await wait(1000);
    },
    crop: () => VIEW_AND_PANEL,
    async act(page) {
      await clickAt(page, tid(page, "optimize-run-cool-sink"), { steps: 18, pause: 22 });
      await tid(page, "optimize-result-cool-sink").waitFor({ timeout: 1_800_000 });
      // The replay: the app steps the viewport through the trajectory.
      await wait(24_000);
    },
  },
];
/**
 * The directory's README, from the files in it.
 *
 * Every clip on disk gets a row, whether or not this run recorded it, so a
 * `--only` run leaves a table that is still true of the directory. Frames
 * and canvas are read back out of each WebP; `wall` is known only for the
 * clips just recorded and is shown for those.
 */
function writeTable(report = []) {
  const byName = Object.fromEntries(report.filter((r) => !r.error).map((r) => [r.name, r]));
  const rows = fs.readdirSync(OUT_DIR).filter((f) => f.endsWith(".webp")).sort().map((file) => {
    const full = path.join(OUT_DIR, file);
    const info = spawnSync("webpmux", ["-info", full], { encoding: "utf8" }).stdout ?? "";
    const frames = +(info.match(/Number of frames:\s*(\d+)/)?.[1] ?? 0);
    const width = +(info.match(/Canvas size:\s*(\d+)/)?.[1] ?? 0);
    const size = bytes(full);
    const wall = byName[file.replace(/\.webp$/, "")]?.wall;
    return { file, frames, width, size, played: frames / FPS, wall };
  });
  const total = rows.reduce((s, r) => s + r.size, 0);
  const lines = [
    "# README motion",
    "",
    "Regenerated by one command, against the real app:",
    "",
    "```",
    "node research/design/motion/animate.mjs            # every clip",
    "node research/design/motion/animate.mjs --only a,b # some; the table covers all",
    "```",
    "",
    `Captured at ${VIEW.width}x${VIEW.height} css px, device scale ${DSF}, played at ${FPS} fps.`,
    "Do not hand-edit anything in this directory.",
    "",
    "| clip | played | frames | delivered | bytes |",
    "| --- | ---: | ---: | ---: | ---: |",
    ...rows.map((r) => `| \`${r.file}\` | ${r.played.toFixed(1)}s${r.wall && r.wall > r.played * 1.5 ? ` (${r.wall.toFixed(0)}s wall)` : ""} | ${r.frames} | ${r.width}px wide | ${kb(r.size)} |`),
    `| **total** | | | | **${(total / 1024 / 1024).toFixed(2)} MB** |`,
    "",
  ];
  const withAlternatives = report.filter((r) => r.alternatives);
  if (withAlternatives.length) {
    const kinds = Object.keys(withAlternatives[0].alternatives);
    lines.push("## Format evidence", "",
      `| clip | ${kinds.join(" | ")} |`, `| --- | ${kinds.map(() => "---:").join(" | ")} |`,
      ...withAlternatives.map((r) => `| \`${r.name}\` | ${kinds.map((k) => kb(r.alternatives[k])).join(" | ")} |`), "");
  }
  fs.writeFileSync(path.join(OUT_DIR, "README.md"), lines.join("\n"));
  console.log("wrote", path.join(OUT_DIR, "README.md"));
}

if (flag("list")) {
  for (const c of CLIPS) console.log(c.name);
  process.exit(0);
}
if (flag("table")) {
  writeTable();
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
    await cast.start({ everyNthFrame: clip.captureEvery ?? 1 });
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

writeTable(report);
process.exit(report.some((r) => r.error) ? 1 : 0);
