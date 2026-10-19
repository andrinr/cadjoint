/**
 * The fixed numbers a viewer frame is drawn with: the paper ground, the
 * key light, the overlay's stroke metrics and the gesture-preview budget.
 */

import { VIEWPORT_BACKGROUND } from "../tokens";

/**
 * The viewport's paper ground, `#e6e6e9`, as the swapchain sees it.
 *
 * The colour target is `bgra8unorm` and the SDF shaders gamma-encode
 * themselves, so this is a *display-encoded* value: it is what a `loadOp:
 * "clear"` writes, and it is what the fullscreen preview pass has to land on
 * for a ray that hits nothing. It is `VIEWPORT_BACKGROUND` from tokens.ts,
 * which is where every overlay measures itself against it.
 */
export const BACKGROUND: GPUColorDict = {
  r: VIEWPORT_BACKGROUND[0],
  g: VIEWPORT_BACKGROUND[1],
  b: VIEWPORT_BACKGROUND[2],
  a: 1,
};

/**
 * The display value a linear radiance has to be to land on `display` after
 * `pow(aces_tone_map(x), 1 / 2.2)` — the Narkowicz fit `_webgpu.py` uses,
 * `x (2.51 x + 0.03) / (x (2.43 x + 0.59) + 0.14)`, inverted as the positive
 * root of its quadratic. Tone-map constants, not colours.
 */
function inverseAcesDisplay(display: number): number {
  const y = Math.pow(display, 2.2);
  const a = 2.51 - 2.43 * y;
  const b = 0.03 - 0.59 * y;
  const c = -0.14 * y;
  return (-b + Math.sqrt(b * b - 4 * a * c)) / (2 * a);
}

/**
 * The same paper, as *linear radiance* for `u.bg_color`.
 *
 * `environment_radiance` in `cadjoint/viewer/_webgpu.py` returns this
 * unchanged and the result goes through ACES then gamma 2.2, so handing it the
 * display value would land the background near `#c4c4c8` — visibly grey
 * against the chrome. These are the pre-images of the three channels above
 * under `pow(aces(x), 1/2.2)`, solved once at load rather than per fragment.
 */
export const BACKGROUND_RADIANCE: readonly [number, number, number] = [
  inverseAcesDisplay(VIEWPORT_BACKGROUND[0]),
  inverseAcesDisplay(VIEWPORT_BACKGROUND[1]),
  inverseAcesDisplay(VIEWPORT_BACKGROUND[2]),
];

/**
 * Key-light intensity for the PBR and path-traced modes.
 *
 * The part is lit against paper, so the key cannot run hot: a brighter key
 * puts every lit face in the tone map's shoulder, where it converges with the
 * background. 1.5 keeps the lit faces below the ground and lets the shading,
 * not the exposure, carry the form.
 */
export const KEY_LIGHT_INTENSITY = 1.5;

/** Fraction of the distance to the camera that construction overlays are pulled forward. */
export const DEPTH_NUDGE = 0.004;
export const LINE_WIDTH_PX = 2.4;
/**
 * The sheet keyline either side of a construction stroke (`fs_edge`). Mesh
 * edges are written with none: they are the mesh's, not the skeleton's.
 */
export const CASING_PX = 0.8;
/** Handle radius including its casing. */
export const HANDLE_RADIUS_PX = 7.5;

/**
 * The ground grid's metrics.
 *
 * The line width is in CSS pixels (the renderer scales it into framebuffer
 * pixels, which are fewer than CSS pixels under the quality budget's
 * resolution cap): 1px, because this is furniture and a 2px grid on paper
 * reads as a table. The fades are multiples of the orbit distance, so the
 * plane dissolves at the same *apparent* place however far out you are.
 */
export const GRATICULE = {
  lineWidth: 1,
  /** The two ground axes carry more weight than a grid line, and no more. */
  axisWidth: 1.6,
} as const;

/**
 * What one preview frame may cost while a gesture is in flight, in GPU ms.
 *
 * A frame predicted to cost more is raymarched at a reduced resolution for
 * the length of the gesture and stretched back over the framebuffer, so an
 * orbit or a drag stays interactive on a scene that is slow at full size.
 * At rest — and on every scene cheap enough to fit — nothing changes.
 */
export const GESTURE_FRAME_BUDGET_MS = 12;

/** The smallest linear scale a gesture preview drops to. */
export const MIN_GESTURE_SCALE = 0.4;

/** `promise`, or a rejection naming `label` once `ms` have passed. */
export function timeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms.`)), ms),
    ),
  ]);
}
