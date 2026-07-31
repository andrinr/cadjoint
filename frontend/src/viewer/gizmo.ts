/**
 * Translate/rotate gizmo geometry and drag math.
 *
 * Translation axes are rendered as filled screen-space arrows and rotation
 * axes as rings. Their interaction geometry and drag math live on the CPU
 * here, so both remain unit-testable without a GPU.
 *
 * Axis order is X, Y, Z throughout, matching the rotation angles the Python
 * side applies intrinsically in that order.
 */

import type { ConstructionKind, ConstructionTransform, GizmoMode } from "../types";
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

/** Crisp, slightly softened CAD axis colours: X red, Y green, Z blue. */
export const AXIS_COLORS: [number, number, number][] = [
  [0.96, 0.28, 0.32],
  [0.3, 0.8, 0.4],
  [0.28, 0.52, 0.98],
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
  dimensions: Record<string, number | number[]> = transform.dimensions,
): [number, number, number][][] {
  const original = rotationMatrix(transform.rotation);
  const next = rotationMatrix(rotation);
  const origin = transform.position as Vec3;
  const ratios = dimensionRatios(transform.dimensions, dimensions);

  const move = (point: readonly number[]): [number, number, number] => {
    const local = applyInverse(original, subtract(point as Vec3, origin));
    const resized: Vec3 = [
      local[0] * ratios[0],
      local[1] * ratios[1],
      local[2] * ratios[2],
    ];
    return add(applyMatrix(next, resized), position) as [number, number, number];
  };
  return edges.map((edge) => [move(edge[0]), move(edge[1])]);
}

function safeRatio(next: number, original: number): number {
  return Math.abs(original) < 1e-9 ? 1 : next / original;
}

/** Per-local-axis size ratio between two primitive dimension records. */
function dimensionRatios(
  original: Record<string, number | number[]>,
  next: Record<string, number | number[]>,
): Vec3 {
  if (Array.isArray(original.size) && Array.isArray(next.size)) {
    return [
      safeRatio(next.size[0], original.size[0]),
      safeRatio(next.size[1], original.size[1]),
      safeRatio(next.size[2], original.size[2]),
    ];
  }
  if (typeof original.radius === "number" && typeof next.radius === "number") {
    const radial = safeRatio(next.radius, original.radius);
    if (typeof original.height === "number" && typeof next.height === "number") {
      return [radial, radial, safeRatio(next.height, original.height)];
    }
    return [radial, radial, radial];
  }
  return [1, 1, 1];
}

/** Return primitive dimensions after scaling one supported axis. */
export function scaleDimensions(
  kind: ConstructionKind,
  dimensions: Record<string, number | number[]>,
  axis: AxisIndex,
  factor: number,
): Record<string, number | number[]> {
  const next = Object.fromEntries(
    Object.entries(dimensions).map(([key, value]) => [
      key,
      Array.isArray(value) ? [...value] : value,
    ]),
  );
  const safeFactor = Math.max(0.05, factor);
  if (kind === "box" && Array.isArray(next.size)) {
    next.size[axis] *= safeFactor;
  } else if (kind === "sphere" && typeof next.radius === "number") {
    next.radius *= safeFactor;
  } else if (kind === "cylinder") {
    const argument = axis === 2 ? "height" : "radius";
    if (typeof next[argument] === "number") {
      next[argument] *= safeFactor;
    }
  }
  return next;
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

/** Picking geometry for one translate arrow; rendering uses its first edge as the axis span. */
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
  mode: GizmoMode,
): { axis: AxisIndex; edges: Edge[] }[] {
  return AXES.map((axis, index) => ({
    axis: index as AxisIndex,
    edges:
      mode === "rotate"
        ? rotateAxisEdges(origin, axis, size)
        : translateAxisEdges(origin, axis, size),
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
  mode: GizmoMode,
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
