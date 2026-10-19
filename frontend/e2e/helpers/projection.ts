import type { Page } from "@playwright/test";

/**
 * World-to-pixel projection for the specs that click the viewport.
 *
 * An independent reimplementation of `src/viewer/math` (same formulas as the
 * preview shader), deliberately not importing it: if the app's projection
 * drifts, the specs click the wrong pixel and fail, instead of both sides
 * agreeing on something wrong.
 */

export const FOV_SCALE = 1.5;
/**
 * Defaults from `Renderer`; the tests never move the camera.
 *
 * The +X−Y+Z corner at the true isometric elevation, atan(1/√2) — the same
 * pair of numbers as `VIEW_PRESETS.iso`, which is what the session opens on.
 */
export const CAMERA = {
  yaw: Math.PI / 4,
  pitch: Math.atan(1 / Math.SQRT2),
  distance: 4.6,
  target: [0, 0, 0] as const,
};
export const FRONT_CAMERA = { ...CAMERA, yaw: 0, pitch: 0 };

export type Vec3 = [number, number, number];
export type CameraSpec = typeof CAMERA;

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

/** Z-up, matching `cameraPosition` in `src/viewer/math`: azimuth about +Z
 *  measured from −Y, so yaw 0 / pitch 0 is the Front view. */
export function cameraPosition(camera: CameraSpec = CAMERA): Vec3 {
  const cp = Math.cos(camera.pitch);
  return [
    camera.target[0] + camera.distance * cp * Math.sin(camera.yaw),
    camera.target[1] - camera.distance * cp * Math.cos(camera.yaw),
    camera.target[2] + camera.distance * Math.sin(camera.pitch),
  ];
}

export interface CanvasMetrics {
  width: number;
  height: number;
  clientWidth: number;
  clientHeight: number;
  left: number;
  top: number;
}

/** Project a world point to CSS pixels inside the canvas element. */
export function projectToCss(
  world: Vec3,
  canvas: Omit<CanvasMetrics, "left" | "top">,
  camera: CameraSpec = CAMERA,
  projection: "perspective" | "orthographic" = "orthographic",
) {
  const position = cameraPosition(camera);
  const forward = norm(sub(camera.target, position));
  // World up is +Z, with the +Y fallback the app uses at the poles.
  const reference: Vec3 = Math.abs(forward[2]) > 0.999 ? [0, 1, 0] : [0, 0, 1];
  const right = norm(cross(forward, reference));
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

/** The canvas's framebuffer and CSS sizes, and where it sits on the page. */
export async function canvasMetrics(page: Page): Promise<CanvasMetrics> {
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

/** Where the gizmo's arrow tip for an axis sits, the way `gizmoScale` sizes it. */
export function gizmoTip(
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
