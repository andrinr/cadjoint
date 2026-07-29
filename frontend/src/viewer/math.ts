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
 * Orthonormal camera frame, matching the shader's `safe_normalize` chain.
 *
 * World up is +Y; looking straight up or down is prevented by the pitch clamp
 * in the camera controller.
 */
export function cameraBasis(position: Vec3, target: Vec3): Basis {
  const forward = normalize(subtract(target, position));
  const right = normalize(cross(forward, [0, 1, 0]));
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
  position: Vec3,
  target: Vec3,
  aspect: number,
  near = DEPTH_NEAR,
  far = DEPTH_FAR,
): Float32Array<ArrayBuffer> {
  const { forward, right, up } = cameraBasis(position, target);
  const sx = 2 / (FOV_SCALE * aspect);
  const sy = 2 / FOV_SCALE;
  const a = far / (far - near);

  // Rows of the mathematical matrix; clip = M * (world, 1).
  const rows: number[][] = [
    [sx * right[0], sx * right[1], sx * right[2], -sx * dot(right, position)],
    [sy * up[0], sy * up[1], sy * up[2], -sy * dot(up, position)],
    [a * forward[0], a * forward[1], a * forward[2], -a * dot(forward, position) - a * near],
    [forward[0], forward[1], forward[2], -dot(forward, position)],
  ];

  const out = new Float32Array(16);
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      out[column * 4 + row] = rows[row][column];
    }
  }
  return out;
}

/** Project a world point to framebuffer pixel coordinates. */
export function projectPoint(
  world: Vec3,
  position: Vec3,
  target: Vec3,
  width: number,
  height: number,
): Projected {
  const { forward, right, up } = cameraBasis(position, target);
  const delta = subtract(world, position);
  const viewDepth = dot(delta, forward);
  if (viewDepth <= 1e-6) {
    return { x: NaN, y: NaN, viewDepth, visible: false };
  }
  const aspect = width / height;
  const u = dot(delta, right) / (FOV_SCALE * viewDepth);
  const v = dot(delta, up) / (FOV_SCALE * viewDepth);
  return {
    x: (u / aspect + 0.5) * width,
    y: (0.5 - v) * height,
    viewDepth,
    visible: true,
  };
}

export interface Ray {
  origin: Vec3;
  direction: Vec3;
}

/** Camera ray through a framebuffer pixel coordinate. */
export function rayFromPixel(
  x: number,
  y: number,
  position: Vec3,
  target: Vec3,
  width: number,
  height: number,
): Ray {
  const { forward, right, up } = cameraBasis(position, target);
  const aspect = width / height;
  const u = (x / width - 0.5) * aspect;
  const v = -(y / height - 0.5);
  const direction = normalize(
    add(forward, scale(add(scale(right, u), scale(up, v)), FOV_SCALE)),
  );
  return { origin: position, direction };
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
