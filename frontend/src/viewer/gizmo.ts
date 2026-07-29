/**
 * Translate/rotate gizmo geometry and drag math.
 *
 * The gizmo is drawn with the same overlay edge pipeline as construction
 * wireframes — three axis arrows for translation, three rings for rotation —
 * and all of its interaction is solved on the CPU here, so it can be unit
 * tested without a GPU.
 *
 * Axis order is X, Y, Z throughout, matching the rotation angles the Python
 * side applies intrinsically in that order.
 */

import type { ConstructionTransform } from "../types";
import {
  add,
  cross,
  dot,
  normalize,
  scale,
  subtract,
  type Ray,
  type Vec3,
  type View,
} from "./math";
import { projectPoint } from "./math";

export type AxisIndex = 0 | 1 | 2;

export const AXES: Vec3[] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

/** Axis colours: X red, Y green, Z blue, brightened when active. */
export const AXIS_COLORS: [number, number, number][] = [
  [1.0, 0.35, 0.35],
  [0.45, 0.95, 0.45],
  [0.42, 0.62, 1.0],
];

const RING_SEGMENTS = 40;

export type Edge = [Vec3, Vec3];

/** Rotation matrix rows for intrinsic X→Y→Z angles (Rz·Ry·Rx). */
export function rotationMatrix(rotation: readonly number[]): Vec3[] {
  const [rx, ry, rz] = rotation;
  const [cx, sx] = [Math.cos(rx), Math.sin(rx)];
  const [cy, sy] = [Math.cos(ry), Math.sin(ry)];
  const [cz, sz] = [Math.cos(rz), Math.sin(rz)];
  return [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx],
  ];
}

const applyMatrix = (matrix: Vec3[], point: Vec3): Vec3 => [
  dot(matrix[0], point),
  dot(matrix[1], point),
  dot(matrix[2], point),
];

/** Transpose-multiply: recover a local point from a world one. */
const applyInverse = (matrix: Vec3[], point: Vec3): Vec3 => [
  matrix[0][0] * point[0] + matrix[1][0] * point[1] + matrix[2][0] * point[2],
  matrix[0][1] * point[0] + matrix[1][1] * point[1] + matrix[2][1] * point[2],
  matrix[0][2] * point[0] + matrix[1][2] * point[1] + matrix[2][2] * point[2],
];

/**
 * Re-place a primitive's world edges under a new position and rotation.
 *
 * The payload only carries world-space edges, so they are first mapped back
 * into the primitive's own frame using the placement they were built with.
 */
export function placeEdges(
  edges: readonly (readonly number[][])[],
  transform: ConstructionTransform,
  position: Vec3,
  rotation: readonly number[],
): [number, number, number][][] {
  const original = rotationMatrix(transform.rotation);
  const next = rotationMatrix(rotation);
  const origin = transform.position as Vec3;

  const move = (point: readonly number[]): [number, number, number] => {
    const local = applyInverse(original, subtract(point as Vec3, origin));
    return add(applyMatrix(next, local), position) as [number, number, number];
  };
  return edges.map((edge) => [move(edge[0]), move(edge[1])]);
}

/** World-space size of the gizmo, so it stays usable at any zoom. */
export const gizmoScale = (view: View, origin: Vec3): number =>
  Math.max(0.15, 0.18 * Math.hypot(...subtract(origin, view.position)));

/** Two perpendicular axes spanning the plane normal to `axis`. */
function planeBasis(axis: Vec3): [Vec3, Vec3] {
  const reference: Vec3 = Math.abs(axis[1]) > 0.9 ? [0, 0, 1] : [0, 1, 0];
  const u = normalize(cross(axis, reference));
  return [u, cross(axis, u)];
}

/** Arrow shaft plus head for one translate axis. */
export function translateAxisEdges(origin: Vec3, axis: Vec3, size: number): Edge[] {
  const tip = add(origin, scale(axis, size));
  const [u, v] = planeBasis(axis);
  const back = add(origin, scale(axis, size * 0.82));
  const width = size * 0.07;
  return [
    [origin, tip],
    [tip, add(back, scale(u, width))],
    [tip, add(back, scale(u, -width))],
    [tip, add(back, scale(v, width))],
    [tip, add(back, scale(v, -width))],
  ];
}

/** Ring in the plane normal to one rotate axis. */
export function rotateAxisEdges(origin: Vec3, axis: Vec3, size: number): Edge[] {
  const [u, v] = planeBasis(axis);
  const radius = size * 0.78;
  const points: Vec3[] = [];
  for (let step = 0; step < RING_SEGMENTS; step++) {
    const angle = (2 * Math.PI * step) / RING_SEGMENTS;
    points.push(
      add(origin, add(scale(u, radius * Math.cos(angle)), scale(v, radius * Math.sin(angle)))),
    );
  }
  return points.map((point, index) => [point, points[(index + 1) % points.length]] as Edge);
}

/** Every edge of the gizmo, grouped by axis so hits can be attributed. */
export function gizmoEdges(
  origin: Vec3,
  size: number,
  mode: "translate" | "rotate",
): { axis: AxisIndex; edges: Edge[] }[] {
  return AXES.map((axis, index) => ({
    axis: index as AxisIndex,
    edges:
      mode === "translate"
        ? translateAxisEdges(origin, axis, size)
        : rotateAxisEdges(origin, axis, size),
  }));
}

function segmentDistance(px: number, py: number, a: Vec3, b: Vec3, view: View): number {
  const first = projectPoint(a, view);
  const second = projectPoint(b, view);
  if (!first.visible || !second.visible) return Infinity;
  const dx = second.x - first.x;
  const dy = second.y - first.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-12) return Math.hypot(px - first.x, py - first.y);
  const t = Math.max(
    0,
    Math.min(1, ((px - first.x) * dx + (py - first.y) * dy) / lengthSquared),
  );
  return Math.hypot(px - (first.x + t * dx), py - (first.y + t * dy));
}

/** Which gizmo axis the pointer is over, if any. */
export function pickGizmoAxis(
  origin: Vec3,
  size: number,
  mode: "translate" | "rotate",
  x: number,
  y: number,
  view: View,
  radius = 10,
): AxisIndex | null {
  let best: { axis: AxisIndex; distance: number } | null = null;
  for (const group of gizmoEdges(origin, size, mode)) {
    for (const [a, b] of group.edges) {
      const distance = segmentDistance(x, y, a, b, view);
      if (distance <= radius && (best === null || distance < best.distance)) {
        best = { axis: group.axis, distance };
      }
    }
  }
  return best?.axis ?? null;
}

/**
 * Parameter along an axis line closest to a ray.
 *
 * Used for axis-constrained dragging: the pointer ray rarely meets the axis, so
 * the nearest point on it is what the object should follow.
 */
export function closestPointOnAxis(ray: Ray, origin: Vec3, axis: Vec3): number {
  const w = subtract(origin, ray.origin);
  const a = dot(axis, axis);
  const b = dot(axis, ray.direction);
  const c = dot(ray.direction, ray.direction);
  const d = dot(axis, w);
  const e = dot(ray.direction, w);
  const denominator = a * c - b * b;
  // Parallel ray: fall back to the projection of the origin offset.
  if (Math.abs(denominator) < 1e-9) return d / (a || 1);
  return (b * e - c * d) / denominator;
}

/**
 * Angle of the pointer around an axis, in the plane through `origin`.
 *
 * Returns null when the ray is edge-on to that plane, where the angle would be
 * numerically meaningless.
 */
export function angleAroundAxis(ray: Ray, origin: Vec3, axis: Vec3): number | null {
  const denominator = dot(ray.direction, axis);
  if (Math.abs(denominator) < 1e-4) return null;
  const t = dot(subtract(origin, ray.origin), axis) / denominator;
  if (t <= 0) return null;
  const hit = add(ray.origin, scale(ray.direction, t));
  const [u, v] = planeBasis(axis);
  const offset = subtract(hit, origin);
  return Math.atan2(dot(offset, v), dot(offset, u));
}

/** Shortest signed difference between two angles. */
export function angleDelta(from: number, to: number): number {
  let delta = to - from;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;
  return delta;
}
