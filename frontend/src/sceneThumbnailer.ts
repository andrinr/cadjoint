/**
 * One picture per saved scene, drawn by the app's own renderer.
 *
 * A scene browser that showed only names would be a directory listing with a
 * nicer font. What makes it a browser is seeing the part — and the honest way
 * to see the part is to *draw* it, with the same raymarcher the viewport
 * uses, rather than to ship a screenshot somebody remembered to update.
 *
 * So this compiles the scene through the ordinary `/compile` endpoint and
 * hands the resulting shaders to a second {@link Renderer} bound to a
 * 320x200 canvas parked off-screen. That renderer is deliberately its own
 * instance: sharing the viewport's would mean swapping the shaders of the
 * model the user is working on, one scene at a time, in front of them.
 *
 * The framing is fixed and stated rather than inherited, because a thumbnail
 * is a *catalogue* picture: the ISO detent so every part is seen from the
 * same corner, overlays off so no sketch handles or constraint marks land in
 * it, and the floor grid on because a solid floating in white has no size.
 *
 * Everything degrades. A machine without WebGPU gets no pictures and says so
 * once; a scene that does not compile gets its first error line; a frame that
 * cannot be read back is simply absent. None of those stop the browser from
 * listing what is on disk.
 */

import * as api from "./api";
import { readThumbnail, writeThumbnail, THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH } from "./thumbnails";
import { DEFAULT_DISPLAY, QUALITY_PRESETS, Renderer } from "./viewer/renderer";

/** What one thumbnail request produced. */
export type ThumbnailOutcome =
  | { state: "ok"; dataUrl: string }
  /** The scene did not compile; `message` is the first line of why. */
  | { state: "failed"; message: string }
  /** No pictures are possible here at all (no WebGPU, no canvas). */
  | { state: "unavailable"; message: string };

export interface Thumbnailer {
  /** Draw (or recall) the picture of one scene, by its source hash. */
  render: (name: string, hash: string | null) => Promise<ThumbnailOutcome>;
  /** Drop the offscreen canvas and its GPU device. */
  dispose: () => void;
}

/** Two frames: one to configure and size the swap chain, one to draw into it. */
function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/**
 * Park a canvas where it is laid out but never seen.
 *
 * `display: none` would give it a zero client rect, and the renderer sizes
 * its swap chain from exactly that — so the canvas is positioned off the left
 * edge of the document instead, at the size the picture wants.
 */
function offscreenCanvas(): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = THUMBNAIL_WIDTH;
  canvas.height = THUMBNAIL_HEIGHT;
  canvas.style.position = "fixed";
  canvas.style.left = `${-2 * THUMBNAIL_WIDTH}px`;
  canvas.style.top = "0";
  canvas.style.width = `${THUMBNAIL_WIDTH}px`;
  canvas.style.height = `${THUMBNAIL_HEIGHT}px`;
  canvas.setAttribute("aria-hidden", "true");
  canvas.dataset.testid = "scene-thumbnailer";
  return canvas;
}

export function createThumbnailer(): Thumbnailer {
  let canvas: HTMLCanvasElement | undefined;
  let renderer: Renderer | undefined;
  let ready: Promise<string> | undefined;

  /** Bring up the offscreen renderer once; resolves to "" or the reason. */
  const start = (): Promise<string> => {
    if (ready) return ready;
    ready = (async () => {
      if (typeof document === "undefined" || typeof requestAnimationFrame === "undefined") {
        return "This browser cannot draw scene thumbnails.";
      }
      canvas = offscreenCanvas();
      document.body.appendChild(canvas);
      const created = new Renderer();
      await created.init(canvas);
      if (created.unavailableReason) return created.unavailableReason;
      created.display = {
        ...DEFAULT_DISPLAY,
        // A catalogue picture of the part, not of the work on it.
        showOverlays: false,
        showSketches: false,
        showGraticule: true,
      };
      created.quality = QUALITY_PRESETS.draft ?? created.quality;
      created.applyViewPreset("iso");
      renderer = created;
      return "";
    })();
    return ready;
  };

  const draw = async (source: string): Promise<ThumbnailOutcome> => {
    const compiled = await api.compile(source);
    if (!compiled.ok) {
      return { state: "failed", message: compiled.error ?? "This scene did not compile." };
    }
    const active = renderer;
    if (!active || !canvas) return { state: "unavailable", message: "No renderer." };
    await active.setShaders({
      preview: compiled.preview_shader,
      path: compiled.path_shader,
      present: compiled.present_shader,
      program: compiled.program ?? null,
    });
    // Nothing *about* the model is drawn, so the construction tree is only
    // handed over to keep the renderer's own bookkeeping consistent.
    active.setConstruction(compiled.construction ?? [], null, null);
    active.resize();
    active.invalidate();
    await nextFrame();
    await nextFrame();
    try {
      const dataUrl = canvas.toDataURL("image/png");
      // A canvas that drew nothing still encodes; a 1x1 or empty result is
      // not worth caching as though it were a picture.
      if (!dataUrl.startsWith("data:image/png") || dataUrl.length < 256) {
        return { state: "unavailable", message: "The frame could not be read back." };
      }
      return { state: "ok", dataUrl };
    } catch (error) {
      return {
        state: "unavailable",
        message: error instanceof Error ? error.message : String(error),
      };
    }
  };

  return {
    async render(name, hash) {
      const cached = await readThumbnail(hash);
      if (cached) return { state: "ok", dataUrl: cached };
      const reason = await start();
      if (reason) return { state: "unavailable", message: reason };
      const loaded = await api.loadScene(name);
      if (!loaded.ok || typeof loaded.source !== "string") {
        return { state: "failed", message: loaded.error ?? `Could not read ${name}.` };
      }
      const outcome = await draw(loaded.source);
      if (outcome.state === "ok" && hash) await writeThumbnail(hash, outcome.dataUrl);
      return outcome;
    },

    dispose() {
      renderer?.destroy();
      renderer = undefined;
      canvas?.remove();
      canvas = undefined;
      ready = undefined;
    },
  };
}
