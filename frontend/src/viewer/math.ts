/**
 * Camera, projection, and ray math for the viewer.
 *
 * Everything here mirrors the ray construction inside the preview shader
 * (`cadjoint/viewer/_webgpu.py`, `trace_pixel`):
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

/**
 * How far a primary ray travels before the marcher gives up, in world units.
 *
 * Mirrors `MAX_TRACE_DISTANCE` in `cadjoint/viewer/_webgpu.py`, and it is the
 * only hard bound the viewer has on where a fragment can be: past it a ray
 * reports a miss, so nothing the scene pass can draw is further than this from
 * the ray's origin. `DEPTH_FAR` is twice it — the perspective camera's own
 * margin over the same bound — and `orthoDepthRange` spends that margin
 * differently.
 */
export const MAX_TRACE_DISTANCE = 100;

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

/** Near and far clip planes, as distances along the view axis from the camera. */
export interface DepthRange {
  near: number;
  far: number;
}

/**
 * The depth slab an orthographic view brackets.
 *
 * A perspective near plane is a real object: the eye is a point, nothing can be
 * drawn behind it, and `DEPTH_NEAR` is only "close enough to the eye not to
 * matter". An orthographic camera has no eye. It sits at `distance` in front of
 * the orbit target because that is where the orbit put it, not because anything
 * is projected through it, and the half of the world on the camera's own side
 * of that plane is exactly as visible as the other half. Measuring the near
 * plane from the camera therefore throws that half away: at the default framing
 * everything more than 4.55 units in front of the target was clipped, and
 * zoomed in to `MIN_DISTANCE` the budget was 0.35 units — a third of the frame.
 *
 * So the slab is hung about the *orbit target* instead, half a slab either
 * side, and its half-depth is `MAX_TRACE_DISTANCE`: the distance at which a
 * primary ray gives up, and hence the furthest anything the scene pass can draw
 * ever is from the camera plane, in either direction. Nothing that could be on
 * screen falls outside it. The slab is `2 × MAX_TRACE_DISTANCE = DEPTH_FAR`
 * deep — exactly the perspective range, so the depth buffer is no coarser than
 * it already was, and being linear in distance rather than in 1/distance it is
 * uniformly finer than the perspective one it replaces.
 *
 * `near` comes out negative for every reachable orbit distance, which is legal
 * and ordinary in a parallel projection: it is the statement that the camera
 * plane is inside the scene rather than in front of it.
 */
export function orthoDepthRange(distance: number): DepthRange {
  return { near: distance - MAX_TRACE_DISTANCE, far: distance + MAX_TRACE_DISTANCE };
}

/** The clip planes a view is drawn with, in the projection it is drawn in. */
export function depthRange(view: View): DepthRange {
  if (!isOrthographic(view)) return { near: DEPTH_NEAR, far: DEPTH_FAR };
  return orthoDepthRange(length(subtract(view.target, view.position)));
}

export interface CameraState {
  /** Azimuth about the world Z axis, radians; 0 looks along +Y (Front). */
  yaw: number;
  /** Elevation above the XY ground plane, radians, clamped by the controller. */
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

/**
 * World-space camera position for an orbit state.
 *
 * The world is Z-up, because the library is: `SketchPlane`'s default normal is
 * +Z, so a sketch lies on the XY floor, and the starter's FEM boundary
 * conditions select its die side and fin field by `z`. The viewer used to
 * assume +Y and drew every part standing on its edge.
 *
 * Azimuth is measured about +Z from −Y, so yaw 0 / pitch 0 puts the camera at
 * −Y looking toward +Y — the Front view — and the preset angles keep the
 * meanings their names claim: yaw +π/2 is Right, pitch +π/2 is Top.
 */
export function cameraPosition(camera: CameraState): Vec3 {
  const cp = Math.cos(camera.pitch);
  return [
    camera.target[0] + camera.distance * cp * Math.sin(camera.yaw),
    camera.target[1] - camera.distance * cp * Math.cos(camera.yaw),
    camera.target[2] + camera.distance * Math.sin(camera.pitch),
  ];
}

/**
 * Orthonormal camera frame, matching `camera_basis` in the shaders.
 *
 * World up is +Z, except when looking almost straight up or down — the Top and
 * Bottom presets do exactly that, and cross(forward, +Z) is degenerate there.
 */
export function cameraBasis(position: Vec3, target: Vec3): Basis {
  const forward = normalize(subtract(target, position));
  const reference: Vec3 = Math.abs(forward[2]) > 0.999 ? [0, 1, 0] : [0, 0, 1];
  const right = normalize(cross(forward, reference));
  const up = cross(right, forward);
  return { forward, right, up };
}

/**
 * Column-major view-projection matrix for WGSL `mat4x4<f32>`.
 *
 * Maps world space to WebGPU clip space (z from 0 at the near plane to w at the
 * far plane) such that the resulting screen positions match the shader's rays.
 *
 * The clip planes default to `depthRange(view)`, which is the perspective pair
 * under perspective and a slab about the orbit target under orthographic — the
 * two projections need different answers and only the view knows which it is.
 */
export function viewProjection(
  view: View,
  nearPlane?: number,
  farPlane?: number,
): Float32Array<ArrayBuffer> {
  const range = depthRange(view);
  const near = nearPlane ?? range.near;
  const far = farPlane ?? range.far;
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
  // Behind the camera plane is off screen only when there is a camera *point*
  // to be behind. A parallel projection has none: the pixel a point lands in
  // does not depend on its depth at all, so the sign of that depth cannot make
  // it invisible, and rejecting it here is how hit testing stopped being able
  // to click the near half of an orthographic scene.
  if (!isOrthographic(view) && viewDepth <= 1e-6) {
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
  /**
   * Smallest ray parameter that is still on screen.
   *
   * Zero under perspective: the origin is the eye and nothing behind it is
   * drawn. Negative under orthographic, where the origin is a station on the
   * camera plane rather than an eye and the ray is visible on both sides of it
   * — as far back as the near plane, which is where this comes from. Absent
   * means zero, so a hand-built ray behaves the way one always did.
   */
  tMin?: number;
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
    // The origin stays on the camera plane — callers place things along the
    // ray from it — and the near plane travels with the ray instead, as the
    // parameter at which it enters the slab `viewProjection` brackets.
    return {
      origin: add(view.position, offset),
      direction: forward,
      tMin: depthRange(view).near,
    };
  }
  const direction = normalize(
    add(forward, scale(add(scale(right, u), scale(up, v)), FOV_SCALE)),
  );
  return { origin: view.position, direction, tMin: 0 };
}

/**
 * Intersect a ray with an infinite plane.
 *
 * Returns null when the ray is parallel to the plane or would hit it outside
 * the visible part of the ray — both mean "the click missed the sketch". The
 * visible part starts at `ray.tMin`, which is the origin under perspective and
 * the near plane, behind the origin, under orthographic.
 */
export function intersectPlane(
  ray: Ray,
  planeOrigin: Vec3,
  planeNormal: Vec3,
): Vec3 | null {
  const denominator = dot(ray.direction, planeNormal);
  if (Math.abs(denominator) < 1e-6) return null;
  const t = dot(subtract(planeOrigin, ray.origin), planeNormal) / denominator;
  if (t <= (ray.tMin ?? 0)) return null;
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
