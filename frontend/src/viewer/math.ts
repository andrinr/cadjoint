/**
 * Camera, projection, and ray math for the viewer.
 *
 * Everything here mirrors the ray construction inside the preview shader
 * (`jaxcad/viewer/_webgpu.py`, `trace_pixel`):
 *
 *   uv  = (frag.xy / res - 0.5) * (aspect, -1)
 *   dir = normalize(forward + 1.5 * (uv.x * right + uv.y * up))
 *
 * Overlay geometry is projected with the matrix built here and the SDF's depth
 * is written through the same matrix, so screen positions and depths agree
 * exactly between the raymarched image and the construction overlay.
 *
 * Pure functions with no GPU or DOM dependency — unit tested in `test/`.
 */

export type Vec3 = readonly [number, number, number];

/** Tangent of the half field of view baked into the preview shader. */
export const FOV_SCALE = 1.5;

export const DEPTH_NEAR = 0.05;
export const DEPTH_FAR = 200;

export type Projection = "perspective" | "orthographic";

/**
 * Everything needed to map between world space and this viewport's pixels.
 *
 * Shared by projection, ray casting, and picking so the three can never
 * disagree about which camera they are talking about.
 */
export interface View {
  position: Vec3;
  target: Vec3;
  /** Framebuffer size in pixels. */
  width: number;
  height: number;
  projection?: Projection;
  /** Viewport height in world units; only used when orthographic. */
  orthoHeight?: number;
}

const isOrthographic = (view: View): boolean => view.projection === "orthographic";

/**
 * Orthographic viewport height that frames the same scene as a perspective
 * camera at `distance` — the frustum height there. Using it keeps the framing
 * unchanged when the projection is toggled.
 */
export const orthoHeightFor = (distance: number): number => FOV_SCALE * distance;

/** Viewport height in world units for an orthographic view. */
function viewOrthoHeight(view: View): number {
  return view.orthoHeight ?? orthoHeightFor(length(subtract(view.target, view.position)));
}

export interface CameraState {
  /** Rotation about the world Y axis, radians. */
  yaw: number;
  /** Elevation above the XZ plane, radians, clamped by the controller. */
  pitch: number;
  /** Distance from the orbit target. */
  distance: number;
  /** Point the camera orbits and looks at. */
  target: Vec3;
}

export interface Basis {
  forward: Vec3;
  right: Vec3;
  up: Vec3;
}

export interface Projected {
  /** Framebuffer pixel coordinate (pixel centres at .5, matching `frag.xy`). */
  x: number;
  y: number;
  /** Distance along the camera forward axis. */
  viewDepth: number;
  /** False when the point is at or behind the camera plane. */
  visible: boolean;
}

export const subtract = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

export const length = (a: Vec3): number => Math.sqrt(dot(a, a));

export function normalize(a: Vec3): Vec3 {
  const n = length(a);
  return n < 1e-12 ? [0, 0, 0] : [a[0] / n, a[1] / n, a[2] / n];
}

/** World-space camera position for an orbit state. */
export function cameraPosition(camera: CameraState): Vec3 {
  const cp = Math.cos(camera.pitch);
  return [
    camera.target[0] + camera.distance * cp * Math.sin(camera.yaw),
    camera.target[1] + camera.distance * Math.sin(camera.pitch),
    camera.target[2] + camera.distance * cp * Math.cos(camera.yaw),
  ];
}

/**
 * Orthonormal camera frame, matching `camera_basis` in the shaders.
 *
 * World up is +Y, except when looking almost straight up or down — the Top and
 * Bottom presets do exactly that, and cross(forward, +Y) is degenerate there.
 */
export function cameraBasis(position: Vec3, target: Vec3): Basis {
  const forward = normalize(subtract(target, position));
  const reference: Vec3 = Math.abs(forward[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
  const right = normalize(cross(forward, reference));
  const up = cross(right, forward);
  return { forward, right, up };
}

/**
 * Column-major view-projection matrix for WGSL `mat4x4<f32>`.
 *
 * Maps world space to WebGPU clip space (z from 0 at the near plane to w at the
 * far plane) such that the resulting screen positions match the shader's rays.
 */
export function viewProjection(
  view: View,
  near = DEPTH_NEAR,
  far = DEPTH_FAR,
): Float32Array<ArrayBuffer> {
  const { position } = view;
  const { forward, right, up } = cameraBasis(position, view.target);
  const aspect = view.width / view.height;

  // Rows of the mathematical matrix; clip = M * (world, 1).
  let rows: number[][];
  if (isOrthographic(view)) {
    const height = viewOrthoHeight(view);
    const sx = 2 / (height * aspect);
    const sy = 2 / height;
    const sz = 1 / (far - near);
    rows = [
      [sx * right[0], sx * right[1], sx * right[2], -sx * dot(right, position)],
      [sy * up[0], sy * up[1], sy * up[2], -sy * dot(up, position)],
      [sz * forward[0], sz * forward[1], sz * forward[2], -sz * dot(forward, position) - sz * near],
      [0, 0, 0, 1],
    ];
  } else {
    const sx = 2 / (FOV_SCALE * aspect);
    const sy = 2 / FOV_SCALE;
    const a = far / (far - near);
    rows = [
      [sx * right[0], sx * right[1], sx * right[2], -sx * dot(right, position)],
      [sy * up[0], sy * up[1], sy * up[2], -sy * dot(up, position)],
      [a * forward[0], a * forward[1], a * forward[2], -a * dot(forward, position) - a * near],
      [forward[0], forward[1], forward[2], -dot(forward, position)],
    ];
  }

  const out = new Float32Array(16);
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      out[column * 4 + row] = rows[row][column];
    }
  }
  return out;
}

/** Project a world point to framebuffer pixel coordinates. */
export function projectPoint(world: Vec3, view: View): Projected {
  const { forward, right, up } = cameraBasis(view.position, view.target);
  const delta = subtract(world, view.position);
  const viewDepth = dot(delta, forward);
  if (viewDepth <= 1e-6) {
    return { x: NaN, y: NaN, viewDepth, visible: false };
  }
  const aspect = view.width / view.height;
  // Perspective divides by depth; orthographic maps a fixed world height, so
  // the divisor must not vary from point to point.
  const divisor = isOrthographic(view) ? viewOrthoHeight(view) : FOV_SCALE * viewDepth;
  const u = dot(delta, right) / divisor;
  const v = dot(delta, up) / divisor;
  return {
    x: (u / aspect + 0.5) * view.width,
    y: (0.5 - v) * view.height,
    viewDepth,
    visible: true,
  };
}

export interface Ray {
  origin: Vec3;
  direction: Vec3;
}

/** Camera ray through a framebuffer pixel coordinate. */
export function rayFromPixel(x: number, y: number, view: View): Ray {
  const { forward, right, up } = cameraBasis(view.position, view.target);
  const aspect = view.width / view.height;
  const u = (x / view.width - 0.5) * aspect;
  const v = -(y / view.height - 0.5);
  if (isOrthographic(view)) {
    const height = viewOrthoHeight(view);
    const offset = add(scale(right, u * height), scale(up, v * height));
    return { origin: add(view.position, offset), direction: forward };
  }
  const direction = normalize(
    add(forward, scale(add(scale(right, u), scale(up, v)), FOV_SCALE)),
  );
  return { origin: view.position, direction };
}

/**
 * Intersect a ray with an infinite plane.
 *
 * Returns null when the ray is parallel to the plane or would hit it behind the
 * ray origin — both mean "the click missed the sketch".
 */
export function intersectPlane(
  ray: Ray,
  planeOrigin: Vec3,
  planeNormal: Vec3,
): Vec3 | null {
  const denominator = dot(ray.direction, planeNormal);
  if (Math.abs(denominator) < 1e-6) return null;
  const t = dot(subtract(planeOrigin, ray.origin), planeNormal) / denominator;
  if (t <= 0) return null;
  return add(ray.origin, scale(ray.direction, t));
}

/** Convert a world point on a sketch plane to its (u, v) sketch coordinates. */
export function worldToPlane(
  world: Vec3,
  planeOrigin: Vec3,
  u: Vec3,
  v: Vec3,
): [number, number] {
  const delta = subtract(world, planeOrigin);
  return [dot(delta, u), dot(delta, v)];
}

/** Convert sketch coordinates back to a world point. */
export function planeToWorld(
  xy: readonly [number, number],
  planeOrigin: Vec3,
  u: Vec3,
  v: Vec3,
): Vec3 {
  return add(planeOrigin, add(scale(u, xy[0]), scale(v, xy[1])));
}
