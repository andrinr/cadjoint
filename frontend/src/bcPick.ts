/**
 * Screen-space picking over the simulation surface.
 *
 * The render payload ships every boundary vertex, so projecting them with the
 * same camera math the renderer uses is enough for both interactions:
 *
 * - click → nearest projected vertex within a pixel radius (the probe chip,
 *   and the `Nodes.sphere` BC proposal sized from the mesh cell spacing);
 * - shift-drag rectangle → world-space AABB of the vertices whose projection
 *   falls inside the rect (the `Nodes.box` BC proposal).
 *
 * Limitations, accepted for simplicity: occluded (back-side) vertices can be
 * picked too — the nearest projected vertex is not necessarily the frontmost
 * surface — and slice-clipped fragments still participate.
 */

import { projectPoint, type Vec3, type View } from "./viewer/math";
import type { BcProposal } from "./types";
import type { GridSpacing } from "./selectionEval";

export const round3 = (value: number): number => Number(value.toFixed(3));

export interface VertexHit {
  index: number;
  world: [number, number, number];
  /** Projected framebuffer pixel position. */
  x: number;
  y: number;
  distance: number;
}

/** Nearest projected vertex to a framebuffer pixel, within `radius` px. */
export function nearestVertex(
  positions: readonly number[],
  x: number,
  y: number,
  view: View,
  radius = 14,
): VertexHit | null {
  let best: VertexHit | null = null;
  const count = Math.floor(positions.length / 3);
  for (let index = 0; index < count; index++) {
    const world: Vec3 = [
      positions[index * 3],
      positions[index * 3 + 1],
      positions[index * 3 + 2],
    ];
    const projected = projectPoint(world, view);
    if (!projected.visible) continue;
    const distance = Math.hypot(projected.x - x, projected.y - y);
    if (distance <= radius && (best === null || distance < best.distance)) {
      best = {
        index,
        world: [world[0], world[1], world[2]],
        x: projected.x,
        y: projected.y,
        distance,
      };
    }
  }
  return best;
}

/**
 * Sphere selection proposal around a picked surface point.
 *
 * The radius is twice the mean cell spacing so the sphere reliably grabs a
 * small patch of nodes at the current resolution; without grid info it falls
 * back to a fixed fraction of nothing better than 0.1.
 */
export function sphereProposal(
  center: readonly [number, number, number],
  grid: GridSpacing | null,
): Extract<BcProposal, { kind: "sphere" }> {
  const spacing =
    grid && grid.spacing.length > 0
      ? grid.spacing.reduce((sum, value) => sum + value, 0) / grid.spacing.length
      : 0.05;
  return {
    kind: "sphere",
    center: [round3(center[0]), round3(center[1]), round3(center[2])],
    radius: round3(2 * spacing),
  };
}

/** A screen rectangle in framebuffer pixels (any two opposite corners). */
export interface ScreenRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Box selection proposal: the world AABB of vertices projecting into `rect`.
 *
 * Returns null when the rectangle catches no vertex. The AABB is padded by a
 * hair after rounding so vertices sitting exactly on the bounds stay inside
 * the server's inclusive box test despite 3-decimal rounding.
 */
export function rectAabbProposal(
  positions: readonly number[],
  rect: ScreenRect,
  view: View,
): Extract<BcProposal, { kind: "box" }> | null {
  const left = Math.min(rect.x0, rect.x1);
  const right = Math.max(rect.x0, rect.x1);
  const top = Math.min(rect.y0, rect.y1);
  const bottom = Math.max(rect.y0, rect.y1);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  let caught = 0;
  const count = Math.floor(positions.length / 3);
  for (let index = 0; index < count; index++) {
    const world: Vec3 = [
      positions[index * 3],
      positions[index * 3 + 1],
      positions[index * 3 + 2],
    ];
    const projected = projectPoint(world, view);
    if (!projected.visible) continue;
    if (projected.x < left || projected.x > right) continue;
    if (projected.y < top || projected.y > bottom) continue;
    caught++;
    for (let axis = 0; axis < 3; axis++) {
      if (world[axis] < min[axis]) min[axis] = world[axis];
      if (world[axis] > max[axis]) max[axis] = world[axis];
    }
  }
  if (caught === 0) return null;
  const pad = 0.001;
  return {
    kind: "box",
    min: [round3(min[0] - pad), round3(min[1] - pad), round3(min[2] - pad)],
    max: [round3(max[0] + pad), round3(max[1] + pad), round3(max[2] + pad)],
  };
}
