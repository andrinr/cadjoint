/**
 * Screen-space picking against the construction tree.
 *
 * Pure geometry: profiles in, indices out. Kept free of GPU and DOM types so
 * the selection and insertion rules can be unit tested without a browser.
 */

import type { ConstructionNode } from "../types";
import { projectPoint, type View } from "./math";

/** Picking uses the same view descriptor as projection and ray casting. */
export type PickView = View;

export interface VertexHit {
  nodeId: string;
  vertexIndex: number;
  distance: number;
}

export interface EdgeHit {
  nodeId: string;
  /** Index the new vertex should take, i.e. after the edge's start vertex. */
  insertIndex: number;
  distance: number;
}

/** Squared distance from a point to a segment, plus where along it that lands. */
function pointSegmentDistance(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-12) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSquared));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/**
 * Nearest sketch vertex to a pixel, within `radius` pixels.
 *
 * Ties break toward the closer vertex; only editable profiles are considered
 * when `editableOnly` is set, since non-editable sketches cannot be modified.
 */
export function pickVertex(
  profiles: readonly ConstructionNode[],
  x: number,
  y: number,
  view: PickView,
  radius = 12,
  editableOnly = false,
): VertexHit | null {
  let best: VertexHit | null = null;
  for (const profile of profiles) {
    if (editableOnly && !profile.editable) continue;
    for (let index = 0; index < profile.vertices.length; index++) {
      const projected = projectPoint(profile.vertices[index].world, view);
      if (!projected.visible) continue;
      const distance = Math.hypot(projected.x - x, projected.y - y);
      if (distance <= radius && (best === null || distance < best.distance)) {
        best = { nodeId: profile.id, vertexIndex: index, distance };
      }
    }
  }
  return best;
}

/**
 * Nearest sketch edge to a pixel, reported as the insertion index for a new
 * vertex on that edge.
 *
 * The profile loop is closed, so the edge from the last vertex back to the
 * first yields an append (`insertIndex === vertices.length`).
 */
export function pickEdge(
  profiles: readonly ConstructionNode[],
  x: number,
  y: number,
  view: PickView,
  radius = 24,
): EdgeHit | null {
  let best: EdgeHit | null = null;
  for (const profile of profiles) {
    if (!profile.editable) continue;
    const count = profile.vertices.length;
    const screen = profile.vertices.map((vertex) =>
      projectPoint(vertex.world, view),
    );
    for (let index = 0; index < count; index++) {
      const start = screen[index];
      const end = screen[(index + 1) % count];
      if (!start.visible || !end.visible) continue;
      const distance = pointSegmentDistance(x, y, start.x, start.y, end.x, end.y);
      if (distance <= radius && (best === null || distance < best.distance)) {
        best = { nodeId: profile.id, insertIndex: index + 1, distance };
      }
    }
  }
  return best;
}

export interface NodeHit {
  nodeId: string;
  distance: number;
}

/**
 * Nearest construction wireframe to a pixel.
 *
 * Selecting a primitive means clicking its outline, since a primitive has no
 * vertex handles of its own. Sketch profiles only join in when `includeProfiles`
 * is set — in vertex mode their handles are the interactive part, and picking
 * the whole loop would fight with that.
 */
export function pickNode(
  nodes: readonly ConstructionNode[],
  x: number,
  y: number,
  view: PickView,
  radius = 10,
  includeProfiles = false,
): NodeHit | null {
  let best: NodeHit | null = null;
  for (const node of nodes) {
    if (!node.editable) continue;
    if (node.kind === "profile" && !includeProfiles) continue;
    for (const [start, end] of node.edges) {
      const a = projectPoint(start, view);
      const b = projectPoint(end, view);
      if (!a.visible || !b.visible) continue;
      const distance = pointSegmentDistance(x, y, a.x, a.y, b.x, b.y);
      if (distance <= radius && (best === null || distance < best.distance)) {
        best = { nodeId: node.id, distance };
      }
    }
  }
  return best;
}

/**
 * Where a new vertex should go when the click is not near any edge.
 *
 * Falls back to appending after the vertex whose screen position is closest,
 * which keeps the winding sensible for clicks inside or outside the profile.
 */
export function nearestInsertIndex(
  profile: ConstructionNode,
  x: number,
  y: number,
  view: PickView,
): number {
  let bestIndex = profile.vertices.length;
  let bestDistance = Infinity;
  for (let index = 0; index < profile.vertices.length; index++) {
    const projected = projectPoint(profile.vertices[index].world, view);
    if (!projected.visible) continue;
    const distance = Math.hypot(projected.x - x, projected.y - y);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index + 1;
    }
  }
  return bestIndex;
}
