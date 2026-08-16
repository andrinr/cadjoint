/**
 * Sketch-plane choice for the place-sketch flow.
 *
 * A new sketch lands on a quick-pick world plane (XY, YZ, ZX) or "on face":
 * the click is ray-cast against the solid under the cursor — the same hit
 * geometry the raymarched image shows — and the surface point plus its
 * normal become `SketchPlane(origin=hit, normal=n)` via two patches
 * (`add_sketch`, then `set_value planeNormal`).
 *
 * Pure geometry, unit tested in `test/sketchPlanes.test.ts`.
 */

import type { ConstructionNode } from "./types";
import { add, dot, scale, subtract, type Ray, type Vec3 } from "./viewer/math";
import { rotationMatrix } from "./viewer/gizmo";

export type SketchPlaneChoice = "xy" | "yz" | "zx" | "face";

export const SKETCH_PLANE_CHOICES: { key: SketchPlaneChoice; label: string; hint: string }[] = [
  { key: "xy", label: "XY", hint: "Place the sketch on the world XY plane" },
  { key: "yz", label: "YZ", hint: "Place the sketch on the world YZ plane" },
  { key: "zx", label: "ZX", hint: "Place the sketch on the world ZX plane" },
  { key: "face", label: "Face", hint: "Place the sketch on a solid's surface" },
];

export const DEFAULT_SKETCH_PLANE: SketchPlaneChoice = "xy";

/** World normal of a quick-pick plane; null for the face pick. */
export function quickPlaneNormal(choice: SketchPlaneChoice): Vec3 | null {
  switch (choice) {
    case "xy":
      return [0, 0, 1];
    case "yz":
      return [1, 0, 0];
    case "zx":
      return [0, 1, 0];
    default:
      return null;
  }
}

/** What one placement click should write into the program. */
export interface SketchPlaneEmission {
  origin: [number, number, number];
  /** Null when the default +Z plane needs no explicit normal patch. */
  normal: [number, number, number] | null;
}

/**
 * Emission for a quick-pick plane: intersect the pick ray with the chosen
 * world plane through the origin, falling back to a point in front of the
 * camera when the view is edge-on to that plane.
 *
 * The XY choice emits no normal — it is the `SketchPlane` default, and the
 * generated source stays as short as before this feature existed.
 */
export function quickPlaneEmission(
  choice: Exclude<SketchPlaneChoice, "face">,
  ray: Ray,
  fallbackDistance: number,
): SketchPlaneEmission {
  const normal = quickPlaneNormal(choice)!;
  const denominator = dot(ray.direction, normal);
  const hit =
    Math.abs(denominator) > 1e-6
      ? add(ray.origin, scale(ray.direction, -dot(ray.origin, normal) / denominator))
      : add(ray.origin, scale(ray.direction, Math.max(1, fallbackDistance)));
  return {
    origin: [hit[0], hit[1], hit[2]],
    normal: choice === "xy" ? null : [normal[0], normal[1], normal[2]],
  };
}

export interface SurfaceHit {
  nodeId: string;
  point: [number, number, number];
  /** Outward unit surface normal, oriented toward the viewer. */
  normal: [number, number, number];
  /** Ray parameter, for choosing the nearest of several solids. */
  t: number;
}

const applyMatrix = (matrix: Vec3[], point: Vec3): Vec3 => [
  dot(matrix[0], point),
  dot(matrix[1], point),
  dot(matrix[2], point),
];

/** Transpose-multiply: world direction into the primitive's local frame. */
const applyInverse = (matrix: Vec3[], point: Vec3): Vec3 => [
  matrix[0][0] * point[0] + matrix[1][0] * point[1] + matrix[2][0] * point[2],
  matrix[0][1] * point[0] + matrix[1][1] * point[1] + matrix[2][1] * point[2],
  matrix[0][2] * point[0] + matrix[1][2] * point[1] + matrix[2][2] * point[2],
];

interface LocalHit {
  t: number;
  normal: Vec3;
}

/** Ray–sphere intersection in the primitive's local frame. */
function hitSphere(origin: Vec3, direction: Vec3, radius: number): LocalHit | null {
  const b = dot(origin, direction);
  const c = dot(origin, origin) - radius * radius;
  const discriminant = b * b - c;
  if (discriminant < 0) return null;
  const root = Math.sqrt(discriminant);
  const t = -b - root > 1e-6 ? -b - root : -b + root;
  if (t <= 1e-6) return null;
  const point = add(origin, scale(direction, t));
  return { t, normal: scale(point, 1 / Math.max(radius, 1e-9)) };
}

/** Slab-test ray–box intersection; `size` holds the half extents. */
function hitBox(origin: Vec3, direction: Vec3, size: Vec3): LocalHit | null {
  let tNear = -Infinity;
  let tFar = Infinity;
  let nearAxis = 0;
  for (let axis = 0; axis < 3; axis++) {
    if (Math.abs(direction[axis]) < 1e-9) {
      if (Math.abs(origin[axis]) > size[axis]) return null;
      continue;
    }
    let t0 = (-size[axis] - origin[axis]) / direction[axis];
    let t1 = (size[axis] - origin[axis]) / direction[axis];
    if (t0 > t1) [t0, t1] = [t1, t0];
    if (t0 > tNear) {
      tNear = t0;
      nearAxis = axis;
    }
    tFar = Math.min(tFar, t1);
    if (tNear > tFar) return null;
  }
  if (tFar <= 1e-6) return null;
  const t = tNear > 1e-6 ? tNear : tFar;
  const point = add(origin, scale(direction, t));
  const normal: [number, number, number] = [0, 0, 0];
  normal[nearAxis] = Math.sign(point[nearAxis]) || 1;
  return { t, normal };
}

/** Ray–cylinder (Z axis, half-height `height`) with caps. */
function hitCylinder(
  origin: Vec3,
  direction: Vec3,
  radius: number,
  height: number,
): LocalHit | null {
  let best: LocalHit | null = null;
  const consider = (candidate: LocalHit | null) => {
    if (candidate && candidate.t > 1e-6 && (best === null || candidate.t < best.t)) {
      best = candidate;
    }
  };

  // Side wall: project onto the XY plane.
  const a = direction[0] * direction[0] + direction[1] * direction[1];
  if (a > 1e-12) {
    const b = origin[0] * direction[0] + origin[1] * direction[1];
    const c = origin[0] * origin[0] + origin[1] * origin[1] - radius * radius;
    const discriminant = b * b - a * c;
    if (discriminant >= 0) {
      const root = Math.sqrt(discriminant);
      for (const t of [(-b - root) / a, (-b + root) / a]) {
        const z = origin[2] + direction[2] * t;
        if (Math.abs(z) <= height) {
          const point = add(origin, scale(direction, t));
          consider({
            t,
            normal: [point[0] / radius, point[1] / radius, 0],
          });
        }
      }
    }
  }

  // Caps at z = ±height.
  if (Math.abs(direction[2]) > 1e-9) {
    for (const sign of [1, -1]) {
      const t = (sign * height - origin[2]) / direction[2];
      const point = add(origin, scale(direction, t));
      if (point[0] * point[0] + point[1] * point[1] <= radius * radius) {
        consider({ t, normal: [0, 0, sign] });
      }
    }
  }
  return best;
}

const asNumber = (value: number | number[] | undefined): number | null =>
  typeof value === "number" ? value : null;

/**
 * Nearest solid surface under a pick ray, with its outward normal.
 *
 * Only primitives with a placement are considered — profiles and derived
 * solids have no analytic surface here. The normal is flipped toward the
 * viewer, so a sketch placed on a face always opens facing the camera.
 */
export function pickSurfacePoint(
  nodes: readonly ConstructionNode[],
  ray: Ray,
): SurfaceHit | null {
  let best: SurfaceHit | null = null;
  for (const node of nodes) {
    const transform = node.transform;
    if (!transform || node.kind === "profile") continue;
    const matrix = rotationMatrix(transform.rotation);
    const origin = applyInverse(matrix, subtract(ray.origin, transform.position as Vec3));
    const direction = applyInverse(matrix, ray.direction);

    let local: LocalHit | null = null;
    if (node.kind === "sphere") {
      const radius = asNumber(transform.dimensions.radius);
      if (radius !== null) local = hitSphere(origin, direction, radius);
    } else if (node.kind === "box") {
      const size = transform.dimensions.size;
      if (Array.isArray(size) && size.length === 3) {
        local = hitBox(origin, direction, [size[0], size[1], size[2]]);
      }
    } else if (node.kind === "cylinder") {
      const radius = asNumber(transform.dimensions.radius);
      const height = asNumber(transform.dimensions.height);
      if (radius !== null && height !== null) {
        local = hitCylinder(origin, direction, radius, height);
      }
    }
    if (!local || (best !== null && local.t >= best.t)) continue;

    const world = add(ray.origin, scale(ray.direction, local.t));
    let normal = applyMatrix(matrix, local.normal);
    if (dot(normal, ray.direction) > 0) normal = scale(normal, -1);
    best = {
      nodeId: node.id,
      point: [world[0], world[1], world[2]],
      normal: [normal[0], normal[1], normal[2]],
      t: local.t,
    };
  }
  return best;
}
