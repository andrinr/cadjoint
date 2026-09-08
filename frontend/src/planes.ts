/**
 * Sketch planes as things the viewport can draw, pick, and label.
 *
 * A `SketchPlane` is infinite; what the viewport shows is a bounded frame
 * around the sketch that sits on it — a translucent quad with an origin
 * marker and a normal glyph — so a user can see where a sketch lives, where
 * a new one will land, and take hold of a plane to move it. Everything here
 * is pure geometry over the construction payload: the frame's extent follows
 * the sketch's own points, and a plane the source derived from a face is
 * marked so the viewport can say it cannot be dragged.
 *
 * Unit tested in `test/planes.test.ts`.
 */

import type { ConstructionNode } from "./types";
import { add, cross, dot, normalize, scale, type Vec3 } from "./viewer/math";

/** Half extent a frame never shrinks below, so an empty plane is still visible. */
export const PLANE_MIN_HALF = 0.6;
/** Breathing room between the sketch's points and the frame's edge. */
export const PLANE_MARGIN = 0.25;
/** Length of the normal glyph as a fraction of the frame's larger half extent. */
export const NORMAL_GLYPH = 0.55;
/** Size of the origin cross as a fraction of the frame's larger half extent. */
export const ORIGIN_MARK = 0.12;

export interface PlaneFrame {
  /** The sketch this plane belongs to; null for the placement preview. */
  nodeId: string | null;
  /** What the label says: the sketch's name, or the preview's promise. */
  label: string;
  origin: Vec3;
  u: Vec3;
  v: Vec3;
  normal: Vec3;
  /** Centre of the drawn quad in plane coordinates (the sketch's centre). */
  centerU: number;
  centerV: number;
  halfU: number;
  halfV: number;
  /** The quad's corners in world space, counter-clockwise about the normal. */
  corners: [Vec3, Vec3, Vec3, Vec3];
  /** True when the plane is a literal the viewport may move. */
  editable: boolean;
  /** True when the source derived the plane from other geometry. */
  derived: boolean;
}

/**
 * In-plane axes for a normal, the way `cadjoint.construction` chooses them.
 *
 * The default `+Z` normal yields the identity frame, and a `±Y` normal falls
 * back to `+Z` as the up reference — the same select the Python side writes,
 * so the preview of a plane about to be created matches the plane the
 * program then builds.
 */
export function planeAxes(normal: Vec3): { u: Vec3; v: Vec3 } {
  const n = normalize(normal);
  const up: Vec3 = Math.abs(dot(n, [0, 1, 0])) > 0.99 ? [0, 0, 1] : [0, 1, 0];
  const u = normalize(cross(up, n));
  const v = cross(n, u);
  return { u, v };
}

function corners(
  origin: Vec3,
  u: Vec3,
  v: Vec3,
  centerU: number,
  centerV: number,
  halfU: number,
  halfV: number,
): [Vec3, Vec3, Vec3, Vec3] {
  const at = (du: number, dv: number): Vec3 =>
    add(origin, add(scale(u, centerU + du), scale(v, centerV + dv)));
  return [at(-halfU, -halfV), at(halfU, -halfV), at(halfU, halfV), at(-halfU, halfV)];
}

/** The frame of a sketch's plane, or null for a node that has no plane. */
export function planeFrame(node: ConstructionNode): PlaneFrame | null {
  const plane = node.plane;
  if (!plane) return null;
  const origin = plane.origin as Vec3;
  const u = plane.u as Vec3;
  const v = plane.v as Vec3;
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (const vertex of node.vertices) {
    minU = Math.min(minU, vertex.uv[0]);
    maxU = Math.max(maxU, vertex.uv[0]);
    minV = Math.min(minV, vertex.uv[1]);
    maxV = Math.max(maxV, vertex.uv[1]);
  }
  const empty = node.vertices.length === 0;
  const centerU = empty ? 0 : (minU + maxU) / 2;
  const centerV = empty ? 0 : (minV + maxV) / 2;
  const halfU = empty ? PLANE_MIN_HALF : Math.max(PLANE_MIN_HALF, (maxU - minU) / 2 + PLANE_MARGIN);
  const halfV = empty ? PLANE_MIN_HALF : Math.max(PLANE_MIN_HALF, (maxV - minV) / 2 + PLANE_MARGIN);
  return {
    nodeId: node.id,
    label: node.name ?? node.id,
    origin,
    u,
    v,
    normal: plane.normal as Vec3,
    centerU,
    centerV,
    halfU,
    halfV,
    corners: corners(origin, u, v, centerU, centerV, halfU, halfV),
    // The payload publishes a transform exactly when the plane is a literal
    // the patch layer can rewrite: absent for a face-derived plane.
    editable: node.editable && node.transform !== null,
    derived: plane.reference !== null && plane.reference !== undefined,
  };
}

/**
 * The frame a new sketch would get at `origin` facing `normal`.
 *
 * Sized to the square `add_sketch` writes (±0.6, plus the margin) so the
 * ghost the pointer carries is the plane the click will produce.
 */
export function previewFrame(origin: Vec3, normal: Vec3): PlaneFrame {
  const { u, v } = planeAxes(normal);
  const half = 0.6 + PLANE_MARGIN;
  return {
    nodeId: null,
    label: "new sketch",
    origin,
    u,
    v,
    normal: normalize(normal),
    centerU: 0,
    centerV: 0,
    halfU: half,
    halfV: half,
    corners: corners(origin, u, v, 0, 0, half, half),
    editable: true,
    derived: false,
  };
}

/** Where the normal glyph ends. */
export function normalTip(frame: PlaneFrame): Vec3 {
  return add(frame.origin, scale(frame.normal, NORMAL_GLYPH * Math.max(frame.halfU, frame.halfV)));
}

/** The corner the label hangs off: the quad's +u, +v corner. */
export function labelAnchor(frame: PlaneFrame): Vec3 {
  return frame.corners[2];
}

/** Half the size of the origin cross, in world units. */
export function originMarkSize(frame: PlaneFrame): number {
  return ORIGIN_MARK * Math.max(frame.halfU, frame.halfV);
}

/** Every drawable plane of a construction tree. */
/** What decides which planes are drawn: see {@link visiblePlaneFrames}. */
export interface PlaneVisibility {
  /** The editing mode; planes belong to sketching. */
  sketching: boolean;
  /** The selected sketch (any part of it: a handle, an edge, the plane itself). */
  selectedNodeId: string | null;
  /** The sketch under the pointer, any part of it. */
  hoveredNodeId: string | null;
  /** The display switches: the construction overlay, sketch geometry, and "all planes". */
  showOverlays: boolean;
  showSketches: boolean;
  showAllPlanes: boolean;
}

/**
 * The planes to draw, pick and label right now.
 *
 * Nothing without the overlay or the sketch geometry; everything with the
 * "all planes" switch; otherwise only while sketching, and only the planes
 * of the sketch that is selected and the one under the pointer. Hovering a
 * sketch's outline or a handle is enough to reveal its plane, so a plane
 * that is not drawn can still be reached — through the sketch it carries.
 */
export function visiblePlaneFrames(
  frames: readonly PlaneFrame[],
  visibility: PlaneVisibility,
): PlaneFrame[] {
  if (!visibility.showOverlays || !visibility.showSketches) return [];
  if (visibility.showAllPlanes) return [...frames];
  if (!visibility.sketching) return [];
  const active = new Set([visibility.selectedNodeId, visibility.hoveredNodeId]);
  return frames.filter((frame) => active.has(frame.nodeId));
}

export function planeFrames(nodes: readonly ConstructionNode[]): PlaneFrame[] {
  const frames: PlaneFrame[] = [];
  for (const node of nodes) {
    const frame = planeFrame(node);
    if (frame) frames.push(frame);
  }
  return frames;
}
