/**
 * Live animation capture of the running playground, for the README.
 *
 *   node research/design/motion/animate.mjs                 # every clip
 *   node research/design/motion/animate.mjs --only bracket-orbit-drag,properties-window
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
import { spawn, spawnSync } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../../..");

/**
 * The checkout that owns `.git`: a worktree has no `.venv` or `node_modules`
 * of its own unless someone made them, and the main checkout's serve fine as
 * long as Python is pointed at this tree's sources.
 */
const mainCheckout = (() => {
  const r = spawnSync("git", ["rev-parse", "--path-format=absolute", "--git-common-dir"],
    { cwd: repo, encoding: "utf8" });
  return r.status === 0 ? path.dirname(r.stdout.trim()) : repo;
})();
const firstExisting = (...candidates) => candidates.find((p) => fs.existsSync(p));

const playwrightEntry = firstExisting(
  ...[repo, mainCheckout].map((root) => path.join(root, "frontend/node_modules/@playwright/test/index.mjs")));
if (!playwrightEntry) {
  console.error("no playwright — run `npm ci` in frontend/");
  process.exit(1);
}
const { chromium } = await import(pathToFileURL(playwrightEntry).href);
const PYTHON = firstExisting(
  ...[repo, mainCheckout].map((root) => path.join(root, ".venv/bin/python")));
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
 * what makes a time-lapse possible at all.
 *
 * The timeline is cut into segments, each played at its own rate: a click is
 * filmed at 1x so the pointer reads as a hand, and the solve it starts can be
 * squeezed into a few seconds (`pace`).
 */
class Screencast {
  constructor(page, dir) {
    this.page = page;
    this.dir = dir;
    this.frames = [];
    this.n = 0;
    this.segments = [];
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
    await this.thin(everyNthFrame);
    this.t0 = Date.now() / 1000;
    this.segments = [{ t: this.t0, speed: 1 }];
  }
  /** Restart the stream at another source rate; the timeline is unbroken. */
  async thin(everyNthFrame) {
    if (this.streaming) await this.cdp.send("Page.stopScreencast").catch(() => {});
    await this.cdp.send("Page.startScreencast", {
      format: "jpeg",
      quality: 95,
      maxWidth: VIEW.width,
      maxHeight: VIEW.height,
      everyNthFrame,
    });
    this.streaming = true;
  }
  /**
   * From now on, play at `speed` (wall seconds per played second), or fit
   * everything until the next call into `fit` played seconds — for a stretch
   * whose wall length is not knowable in advance, a solve or a run.
   */
  pace({ speed = 1, fit = 0, every } = {}) {
    this.segments.push({ t: Date.now() / 1000, speed, fit });
    if (every) return this.thin(every);
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
 * Constant-rate output frames, chosen from the captured ones by wall clock,
 * segment by segment.
 *
 * A segment with `speed` > 1 is a time-lapse: a 300-second optimisation played
 * in six seconds is still every frame the app drew, sampled — not a montage.
 */
function resample(frames, segments, until) {
  if (frames.length === 0) throw new Error("no frames captured");
  const end = Math.max(until, frames[frames.length - 1].t);
  const picked = [];
  const report = [];
  let cursor = 0;
  segments.forEach((seg, i) => {
    const first = Math.max(seg.t, frames[0].t);
    const last = i + 1 < segments.length ? segments[i + 1].t : end;
    const span = Math.max(last - first, 0);
    if (span <= 0) return;
    const rate = seg.fit ? Math.max(span / seg.fit, 1) : seg.speed;
    const step = rate / FPS;
    for (let t = first; t < last - 1e-6; t += step) {
      while (cursor + 1 < frames.length && frames[cursor + 1].t <= t) cursor++;
      picked.push(frames[cursor].file);
    }
    report.push({ wall: span, rate });
  });
  const wall = report.reduce((s, r) => s + r.wall, 0);
  const rate = Math.max(...report.map((r) => r.rate));
  return { picked, wall, rate };
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


/**
 * Where the pointer shows a given cursor, nearest to `near` first, sweeping
 * rings like `findHandle`. The pane's cursor vocabulary is the app's own
 * answer to "what is under the pointer": `grab` is a handle or a gizmo arrow,
 * `pointer` selects, `crosshair` places. `pick` chooses among the hits.
 */
async function findCursor(page, near, radius, cursor, pick = (hits) => hits[0]) {
  const hits = await page.evaluate(([cx, cy, r, want]) => {
    const canvas = document.querySelector("[data-testid=viewer-canvas]");
    const rect = canvas.getBoundingClientRect();
    const found = [];
    for (let ring = 0; ring <= r; ring += 3) {
      const step = ring === 0 ? 360 : Math.max(0.5, (3 / ring) * (180 / Math.PI));
      for (let a = 0; a < 360; a += step) {
        const x = cx + ring * Math.cos((a * Math.PI) / 180);
        const y = cy + ring * Math.sin((a * Math.PI) / 180);
        if (x < rect.left + 4 || x > rect.right - 4 || y < rect.top + 4 || y > rect.bottom - 4) continue;
        canvas.dispatchEvent(new PointerEvent("pointermove", {
          clientX: x, clientY: y, bubbles: true, pointerId: 1,
        }));
        if (canvas.style.cursor === want) found.push({ x, y });
      }
    }
    return found;
  }, [near.x, near.y, radius, cursor]);
  return hits.length ? pick(hits) : null;
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
 * The pointer, drawn.
 *
 * A screencast has no cursor in it — the compositor does not composite one —
 * so a clip of a drag would otherwise be a solid moving with nothing moving
 * it. This injects an arrow that follows the real pointer events the script
 * sends, and marks a press: a ring bursts from the tip on the way down and a
 * dot stays under it while the button is held, so a click and a drag read
 * differently. Only trusted events move it: `findHandle`'s synthetic sweeps
 * are a search, not a gesture, and must not show.
 *
 * The same pass hides the perf readout. It is the viewport's own debugging
 * aid, it ticks twice a second, and in a clip it is motion that says nothing
 * about the interaction and costs bytes on every frame.
 */
async function installCursor(page) {
  await page.evaluate(() => {
    if (document.getElementById("__cursor")) return;
    const style = document.createElement("style");
    style.textContent = `
      [data-testid=perf-readout] { display: none !important; }
      #__cursor { position: fixed; z-index: 2147483647; pointer-events: none;
        left: -100px; top: -100px; width: 0; height: 0; }
      #__cursor svg { position: absolute; left: -3px; top: -2px;
        filter: drop-shadow(0 1px 1.5px rgba(0,0,0,.35));
        transform-origin: 3px 2px; transition: transform .09s ease-out; }
      #__cursor.down svg { transform: scale(.88); }
      #__cursor .held { position: absolute; left: -7px; top: -7px; width: 14px; height: 14px;
        border-radius: 50%; background: rgba(255,174,0,.55); border: 2px solid #ffae00;
        box-sizing: border-box; opacity: 0; transition: opacity .09s; }
      #__cursor.down .held { opacity: 1; }
      .__ripple { position: fixed; z-index: 2147483646; pointer-events: none;
        width: 44px; height: 44px; margin: -22px 0 0 -22px; border-radius: 50%;
        border: 2.5px solid #ffae00; box-sizing: border-box;
        animation: __ripple .48s cubic-bezier(.2,.7,.3,1) forwards; }
      @keyframes __ripple { from { transform: scale(.2); opacity: 1; }
                            to { transform: scale(1); opacity: 0; } }`;
    document.head.appendChild(style);
    const cursor = document.createElement("div");
    cursor.id = "__cursor";
    cursor.innerHTML = `<div class="held"></div>
      <svg width="24" height="28" viewBox="0 0 24 28">
        <path d="M3 2 L3 21 L8 16.6 L11.6 24.4 L15 22.9 L11.4 15.3 L18 15.3 Z"
          fill="#18181c" stroke="#fff" stroke-width="1.7" stroke-linejoin="round"/>
      </svg>`;
    document.documentElement.appendChild(cursor);
    const move = (e) => {
      if (!e.isTrusted) return;
      cursor.style.left = `${e.clientX}px`;
      cursor.style.top = `${e.clientY}px`;
    };
    for (const type of ["pointermove", "pointerdown", "pointerup"]) {
      window.addEventListener(type, move, true);
    }
    window.addEventListener("pointerdown", (e) => {
      if (!e.isTrusted) return;
      cursor.classList.add("down");
      const ring = document.createElement("div");
      ring.className = "__ripple";
      ring.style.left = `${e.clientX}px`;
      ring.style.top = `${e.clientY}px`;
      document.documentElement.appendChild(ring);
      setTimeout(() => ring.remove(), 600);
    }, true);
    window.addEventListener("pointerup", (e) => {
      if (e.isTrusted) cursor.classList.remove("down");
    }, true);
  });
}

/** Ease in and out: no motion starts or stops at full speed. */
const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

/**
 * Move the pointer somewhere along an eased straight line, at a pace a reader
 * can follow: longer moves take longer, but never more than about a second.
 * With the button held, this is a drag.
 */
async function glide(page, to, { ms } = {}) {
  const from = glide.last ?? to;
  const dist = Math.hypot(to.x - from.x, to.y - from.y);
  const duration = ms ?? Math.min(1000, Math.max(380, 300 + dist * 0.75));
  const frames = Math.max(2, Math.round(duration / 16));
  const t0 = Date.now();
  for (let i = 1; i <= frames; i++) {
    const u = ease(i / frames);
    await page.mouse.move(from.x + (to.x - from.x) * u, from.y + (to.y - from.y) * u);
    // Paced by the clock, not by a fixed sleep: a slow frame does not stretch
    // the move, and a fast one does not rush it.
    const due = t0 + (duration * i) / frames;
    const left = due - Date.now();
    if (left > 0) await wait(left);
  }
  glide.last = to;
}

/** The beat a reader needs before and after an action for it to register. */
const BEAT = 550;
const beat = (n = 1) => wait(BEAT * n);

/** Where the pointer rests between clips: just off the bottom-right corner. */
const PARK = { x: VIEW.width - 4, y: VIEW.height + 40 };

/** A fresh page on the starter, settled, with the pointer parked off-frame. */
async function freshPage(context) {
  const page = await context.newPage();
  page.on("console", (m) => { if (m.type() === "error") console.log("   [page]", m.text().slice(0, 160)); });
  await page.goto(URL);
  await tid(page, "viewer-canvas").waitFor({ timeout: 180_000 });
  await clickIf(page.getByRole("button", { name: /^dismiss$/i }), 300);
  await waitForCompile(page, 180_000);
  await installCursor(page);
  await park(page);
  await wait(400);
  return page;
}

/** Put the pointer back off-frame without film: setup ends here. */
async function park(page) {
  glide.last = PARK;
  await page.mouse.move(PARK.x, PARK.y);
}

/** Leave the frame the way the clip came in, so the loop closes. */
async function exit(page) {
  await glide(page, PARK, { ms: 800 });
  await beat(1.2);
}

const centre = (b) => ({ x: b.x + b.width / 2, y: b.y + b.height / 2 });

/** Glide to an element (or a point) and settle there. */
async function point(page, target, { ms, settle = 220 } = {}) {
  let to = target;
  if (typeof target.boundingBox === "function") {
    const b = await target.first().boundingBox();
    if (!b) throw new Error("no box for pointer target");
    to = centre(b);
  }
  await glide(page, to, { ms });
  await wait(settle);
  return to;
}

/** Glide, press, release: a click a reader can see happen. */
async function clickAt(page, target, opts) {
  await point(page, target, opts);
  await page.mouse.down();
  await wait(110);
  await page.mouse.up();
}

/** Press at `from`, drag along an eased line to `to`, and optionally let go. */
async function drag(page, from, to, { ms = 1100, release = true, hold = 250 } = {}) {
  await point(page, from, { settle: 250 });
  await page.mouse.down();
  await wait(hold);
  await glide(page, to, { ms });
  await wait(hold);
  if (release) await page.mouse.up();
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
// starts from, however long that takes, and find every screen point the clip
// will need, so no search is ever on film), `crop` (evaluated after setup),
// then `act(page, cast)` (recorded; `cast.pace` time-lapses a stretch).
//
// The rules every clip keeps, so a reader can follow it:
//   - one idea per clip;
//   - the pointer enters from off-frame, and leaves the same way, after the
//     state has been put back: the last frame is the first, and the loop
//     restarts without a jump;
//   - every move is eased, and every action has a beat of stillness before
//     and after it;
//   - the crop is the panels the idea happens in, below the menu bar.

/** The editor and the viewport. */
const CODE_AND_VIEW = { x: 0, y: 72, w: 1120, h: 804 };
/** The viewport and the dock column to its right. */
const VIEW_AND_PANEL = { x: 460, y: 72, w: 980, h: 804 };

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

/** The canvas centre, in page pixels. */
async function canvasCentre(page) {
  const m = await canvasMetrics(page);
  return { x: m.left + m.clientWidth / 2, y: m.top + m.clientHeight / 2 };
}

/**
 * A half turn of the camera, one pull on a named vertex handle from the far
 * side, and the other half turn, on a scene opened from `scenes/`.
 *
 * The handle is pulled along `pull` and let go, so the release rewrites the
 * literal in the editor; after a beat it is pulled back and let go again, so
 * the literal is restored, and the second half turn brings the camera to the
 * view the clip opened on: the loop closes.
 *
 * `unconstrain` names a constraint (by its chip's text in the sketch panel)
 * to delete off film first, through the panel's own delete control, so the
 * handle follows the pointer instead of sliding along what the constraint
 * leaves it.
 */
function orbitAndDrag({ name, scene, handle, zoomTicks = 2, pull, turn = 180, unconstrain, width = 1120, quality = 74 }) {
  return {
    name,
    width,
    quality,
    async setup(page) {
      await openScene(page, scene);
      await zoom(page, zoomTicks);
      await tid(page, "mode-vertex").click();
      await wait(400);
      const b = await tid(page, "viewer-canvas").boundingBox();
      // Orbits start on empty floor, top left of the part.
      this.orbitFrom = { x: b.x + b.width * 0.22, y: b.y + b.height * 0.3 };
      const orbit = async (dx) => {
        const from = this.orbitFrom;
        await page.mouse.move(from.x, from.y);
        await page.mouse.down();
        const steps = Math.max(10, Math.round(Math.abs(dx) / 8));
        for (let i = 1; i <= steps; i++) {
          await page.mouse.move(from.x + (dx * i) / steps, from.y);
          await wait(8);
        }
        await page.mouse.up();
        await wait(300);
      };
      // The turn is in degrees of yaw. The app publishes its camera, so the
      // pixels per degree are measured on the build being filmed rather than
      // assumed: the rate has changed before, and a turn that is not the
      // angle it claims to be shows the wrong side of the part.
      const camera = () => page.evaluate(() => window.__cadjointCamera());
      const c0 = await camera();
      await orbit(100);
      const c1 = await camera();
      await orbit(-100);
      this.turnPx = (turn * 100) / Math.abs(((c1.yaw - c0.yaw) * 180) / Math.PI);
      await orbit(this.turnPx);
      await wait(600);
      const locate = async () => {
        const found = await findHandle(page, await canvasCentre(page), 460, handle);
        if (!found) throw new Error(`no ${handle} handle in the viewport of ${scene}`);
        return { x: found.x, y: found.y };
      };
      this.tip = await locate();
      // Selecting the handle scrolls the editor to the literal the drag
      // rewrites, and brings up its sketch's constraints.
      await page.mouse.move(this.tip.x, this.tip.y);
      await page.mouse.down(); await wait(90); await page.mouse.up();
      await waitForCompile(page);
      await wait(1200);
      if (unconstrain) {
        const chip = page.locator("[data-testid^=constraint-chip-]").filter({ hasText: unconstrain }).first();
        await chip.waitFor({ timeout: 30_000 });
        await chip.locator("[data-testid^=constraint-delete-]").click();
        await waitForCompile(page);
        await wait(1200);
        this.tip = await locate();
      }
      // A rehearsal of the pull, off film: the handle lands where the solver
      // puts it, which is not always under the pointer, and the pull back on
      // film has to start where it really is. The return leaves the literal
      // where film will.
      const out = { x: this.tip.x + pull.x, y: this.tip.y + pull.y };
      await drag(page, this.tip, out, { ms: 500, hold: 120 });
      await waitForCompile(page);
      await wait(800);
      const landed = await findHandle(page, out, 220, handle);
      if (!landed) throw new Error(`lost ${handle} after the pull`);
      this.landed = { x: landed.x, y: landed.y };
      await drag(page, this.landed, this.tip, { ms: 500, hold: 120 });
      await waitForCompile(page);
      await wait(800);
      await orbit(-this.turnPx);
      await wait(1200);
      await park(page);
      await wait(600);
    },
    crop: () => CODE_AND_VIEW,
    async act(page) {
      await beat();
      const half = async () => {
        await drag(page, this.orbitFrom,
          { x: this.orbitFrom.x + this.turnPx, y: this.orbitFrom.y }, { ms: 1500, hold: 150 });
        await beat(0.6);
      };
      await half();
      const out = { x: this.tip.x + pull.x, y: this.tip.y + pull.y };
      await drag(page, this.tip, out, { ms: 1300 });
      await waitForCompile(page);
      await beat(2.2);
      await drag(page, this.landed, this.tip, { ms: 1300 });
      await waitForCompile(page);
      await beat(0.8);
      // The same direction again: a whole turn ends on the opening view.
      await half();
      await exit(page);
    },
  };
}

const CLIPS = [
  // A half turn to the far side; the gusset's tip pulled up the web, the
  // smooth union re-blending the joint as it moves and the literal rewritten
  // on release; pulled back; the other half turn. The rib's distance
  // constraint is deleted first, so the tip goes where the pointer goes.
  orbitAndDrag({ name: "bracket-orbit-drag", scene: "bracket.py", handle: "rib_tip",
                 unconstrain: /distance/i, pull: { x: 40, y: -110 }, width: 880, quality: 66 }),
  {
    name: "end-cap-solve",
    width: 0,
    quality: 72,
    // The end cap's declared thermal study, solved on film. The result the
    // server has computed once is fetched back into any later session the
    // moment Simulate opens, so the only way to record a field arriving is
    // to be the run that produces it: this clip must be the first solve the
    // server is asked for, and the recorder starts a fresh server per run.
    async setup(page) {
      await openScene(page, "end_cap.py");
      await zoom(page, 2);
      await tid(page, "editmode-simulate").click();
      await tid(page, "simulate-run-cap-conduction").waitFor({ timeout: 60_000 });
      await wait(1400);
      await park(page);
    },
    crop: () => VIEW_AND_PANEL,
    async act(page, cast) {
      await beat();
      await clickAt(page, tid(page, "simulate-run-cap-conduction"));
      await wait(500);
      // Mesh, assembly and solve, whatever they take, in three seconds.
      await cast.pace({ fit: 3, every: 4 });
      await tid(page, "simulate-legend").waitFor({ timeout: 900_000 });
      await cast.pace({ speed: 1, every: 1 });
      await wait(800);
      await beat(4);
      await exit(page);
    },
  },
  {
    name: "heat-sink-optimize",
    width: 0,
    quality: 72,
    // A real gradient-descent run on a FEM objective, every step a mesh, a
    // solve and an adjoint, then the app's own replay of the trajectory in
    // the viewport. Both are time-lapsed — every frame is one the app drew,
    // sampled, which is the only honest way to show a thing that slow — and
    // thinned at the source so the recorder does not compete with the run.
    async setup(page) {
      await zoom(page, 2);
      await clickIf(tid(page, "window-tab-optimize"), 600);
      // Only the optimizer in the column: the tree and an empty properties
      // window would take half its height and say nothing.
      await parkColumn(page, ["objects", "properties"]);
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
      await park(page);
    },
    crop: () => VIEW_AND_PANEL,
    async act(page, cast) {
      await beat();
      await clickAt(page, tid(page, "optimize-run-cool-sink"));
      await wait(500);
      await glide(page, PARK, { ms: 700 });
      await cast.pace({ fit: 4, every: 6 });
      await tid(page, "optimize-result-cool-sink").waitFor({ timeout: 1_800_000 });
      // The finished run replays itself once in the viewport, a ghost compile
      // per step, and stops on the final shape.
      await cast.pace({ fit: 4, every: 3 });
      await page.waitForFunction(
        () => /pause/i.test(document.querySelector("[data-testid=optimize-play]")?.textContent ?? ""),
        null, { timeout: 30_000 }).catch(() => {});
      await page.waitForFunction(
        () => /replay/i.test(document.querySelector("[data-testid=optimize-play]")?.textContent ?? ""),
        null, { timeout: 120_000 });
      await cast.pace({ speed: 1, every: 1 });
      await beat(4);
    },
  },
  {
    name: "sketch-planes",
    width: 0,
    quality: 74,
    // In Sketch mode, selecting the fin comb in the object tree brings up its
    // work plane with the move gizmo on its origin; the gizmo's Z arrow lifts
    // the plane, and the Properties window's ORIGIN follows the source.
    async setup(page) {
      await zoom(page, 2);
      await tid(page, "editmode-sketch").click();
      await wait(800);
      // Find the arrow once, off film, then put the selection back.
      await tid(page, "tree-row-profile_0").click();
      await wait(1200);
      this.arrow = await findCursor(page, await canvasCentre(page), 360, "grab",
        (hits) => hits.reduce((top, h) => (h.y < top.y ? h : top)));
      if (!this.arrow) throw new Error("no gizmo arrow on the fin comb's plane");
      await page.mouse.move(PARK.x, PARK.y);
      await page.keyboard.press("Escape");
      await wait(400);
      if (await page.locator(".object-tree-row.active").count()) await page.keyboard.press("Escape");
      await wait(800);
      await park(page);
    },
    crop: () => VIEW_AND_PANEL,
    async act(page) {
      await beat();
      await clickAt(page, tid(page, "tree-row-profile_0"));
      await beat(2);
      const up = { x: this.arrow.x, y: this.arrow.y - 90 };
      // Taken a few pixels down the shaft, not at the very tip.
      const grip = { x: this.arrow.x, y: this.arrow.y + 8 };
      await drag(page, grip, { x: up.x, y: up.y + 8 }, { ms: 1300 });
      await waitForCompile(page);
      await beat(2.4);
      await drag(page, { x: up.x, y: up.y + 8 }, grip, { ms: 1300 });
      await waitForCompile(page);
      await beat(1.2);
      await glide(page, PARK, { ms: 800 });
      await page.keyboard.press("Escape");
      await wait(300);
      if (await page.locator(".object-tree-row.active").count()) await page.keyboard.press("Escape");
      await beat(1.4);
    },
  },
  {
    name: "properties-window",
    width: 0,
    quality: 74,
    async setup(page) {
      await zoom(page, 1);
      // The materials library and the optimizer share a third panel in the
      // column; parked, the tree and the properties window get its height.
      await parkColumn(page, ["materials", "optimize"]);
      await park(page);
    },
    crop: () => VIEW_AND_PANEL,
    async act(page) {
      await beat();
      // The tree's operator row points the window at the extrusion.
      await clickAt(page, page.locator("[data-testid^=tree-row-profile_0-op-extrude]").first());
      await tid(page, "properties-feature").waitFor({ timeout: 30_000 });
      await beat(2);
      const type = async (value) => {
        const field = tid(page, "prop-feature-depth").first();
        await clickAt(page, field);
        await page.keyboard.press("ControlOrMeta+A");
        await wait(250);
        await page.keyboard.type(value, { delay: 160 });
        await wait(400);
        await page.keyboard.press("Enter");
        await waitForCompile(page);
      };
      await type("1.8");
      await beat(2.6);
      // And back, so the loop closes on the frame it opened on.
      await type("1.2");
      await beat(1.2);
      // The window's own way back: it follows the viewport selection again,
      // which is none, and the tree row lets go — the opening frame.
      await clickAt(page, page.locator(".properties-follow").first());
      await beat(0.6);
      await exit(page);
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
    // img2webp folds a run of identical frames into one longer frame, so the
    // played length is the sum of the durations (the 7th column), not a count.
    const played = [...info.matchAll(/^\s*\d+:(?:\s+\S+){5}\s+(\d+)/gm)]
      .reduce((sum, m) => sum + +m[1], 0) / 1000;
    return { file, frames, width, size, played, wall };
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
    "It needs `ffmpeg` and `img2webp` (`brew install ffmpeg webp`), Playwright in",
    "`frontend/node_modules`, and the committed bundle in `cadjoint/viewer/static/`,",
    "which it films as-is. From a git worktree it borrows the main checkout's `.venv`",
    "and `node_modules` and serves the worktree's own sources. It starts its own",
    "server unless one already answers on `--port` (default 6700); `end-cap-solve`",
    "must be the first solve that server sees, so record it on a fresh one.",
    "",
    "Each clip is one idea: the drawn pointer enters from off-frame, every move is",
    "eased, each press shows a ring, each action has a beat before and after, and",
    "the clip puts the state back before the pointer leaves, so the loop closes.",
    "Solves and optimisation runs are time-lapsed; clicks play at 1x.",
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
  if (!PYTHON) {
    console.error("no .venv/bin/python here or in", mainCheckout);
    process.exit(1);
  }
  // PYTHONPATH first: the venv's editable install points at the main
  // checkout, and a worktree must serve its own sources.
  server = spawn(PYTHON,
    ["-m", "cadjoint.viewer.playground", "--port", String(PORT)],
    { cwd: repo, stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONPATH: [repo, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) } });
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
    await clip.act.call(clip, page, cast);
    await wait(250);
    const frames = await cast.stop();
    if (clip.after) await clip.after.call(clip, page);
    console.log(`   ${frames.length} frames captured`);

    const { picked, wall, rate } = resample(frames, cast.segments, cast.t1);
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
