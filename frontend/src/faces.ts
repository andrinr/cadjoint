/**
 * Picking an analytic face under the pointer, and writing it back as a plane.
 *
 * cadjoint is an implicit modeller: the only mesh it has is the render of the
 * surface, and a picture of the geometry must never be allowed to define a
 * plane. So a face pick is resolved against the *construction tree's* faces —
 * exact, parametric, and shipped with every compile — rather than against
 * anything the raymarcher produced.
 *
 * The resolution is two steps, and they answer different questions:
 *
 *  1. **Where is the surface under the cursor?** A ray cast that mirrors what
 *     the raymarched image shows: the analytic primitives (box, sphere,
 *     cylinder) plus the declared face polygons, nearest hit wins. That gives
 *     a point `p` and a surface normal `n̂`.
 *  2. **Which declared face is that?** Four rejections against every face —
 *     off its plane, outside its boundary, facing away, and then nearest
 *     plane with alignment breaking ties.
 *
 * Step 2 is separate from step 1 on purpose. A hit can land on a surface that
 * has no analytic face at all (the side of a revolve, a blended union), and
 * the honest answer there is not "the nearest face" — it is *no face*, which
 * is what makes the caller fall back to a tangent plane instead of planting a
 * sketch on a face the user never pointed at.
 *
 * Pure geometry, unit tested in `test/faces.test.ts`.
 */

import type { ConstructionFace, ConstructionNode, SketchPlaneReference } from "./types";
import { pickSurfacePoint, type SurfaceHit } from "./sketchPlanes";
import { add, dot, scale, subtract, type Ray, type Vec3 } from "./viewer/math";

/**
 * The preview raymarcher's surface tolerance.
 *
 * The shader stops marching at `abs(d) < 0.001` (`trace()` in
 * `cadjoint/viewer/_webgpu.py`), so the point it reports can sit that far off
 * the true surface. The frontend has no copy of that constant to import —
 * the shader arrives as generated text — so it is restated here, and any face
 * tolerance smaller than twice it would reject hits the image says are on the
 * face.
 */
export const RAYMARCH_SURFACE_EPSILON = 0.001;

/** How closely the surface normal must agree with the face's own. */
const ALIGNMENT_FLOOR = 0.5;

/** A face pick, with the numbers that chose it. */
export interface FacePick {
  face: ConstructionFace;
  /** Construction node the face belongs to. */
  nodeId: string;
  /** |(p − origin) · normal| — how far off the plane the hit landed. */
  planeDistance: number;
  /** n̂ · normal — how squarely the surface faces the same way. */
  alignment: number;
}

/** The slack one face allows a hit, never tighter than the raymarcher's own. */
export const faceTolerance = (face: ConstructionFace): number =>
  Math.max(face.tolerance, 2 * RAYMARCH_SURFACE_EPSILON);

/** A world point in the face's own in-plane coordinates. */
export function toFacePlane(face: ConstructionFace, point: Vec3): [number, number] {
  const delta = subtract(point, face.origin);
  return [dot(delta, face.xAxis), dot(delta, face.yAxis)];
}

/**
 * Signed distance from a 2D point to a closed polygon: negative inside.
 *
 * The same formulation as `polygon_sdf_2d` on the Python side — nearest
 * distance to any edge, signed by a crossing-number test — so the client and
 * the server agree about what "inside this face" means.
 */
export function polygonDistance(point: readonly [number, number], polygon: readonly [number, number][]): number {
  const count = polygon.length;
  if (count === 0) return Infinity;
  let squared = Infinity;
  let inside = false;
  for (let index = 0, previous = count - 1; index < count; previous = index++) {
    const a = polygon[index];
    const b = polygon[previous];
    const ex = b[0] - a[0];
    const ey = b[1] - a[1];
    const wx = point[0] - a[0];
    const wy = point[1] - a[1];
    const lengthSquared = ex * ex + ey * ey;
    const t = lengthSquared < 1e-20 ? 0 : Math.max(0, Math.min(1, (wx * ex + wy * ey) / lengthSquared));
    const dx = wx - ex * t;
    const dy = wy - ey * t;
    squared = Math.min(squared, dx * dx + dy * dy);
    // Crossing number, evaluated on the half-open edge so a vertex counts once.
    if (a[1] > point[1] !== b[1] > point[1]) {
      const crossing = a[0] + ((point[1] - a[1]) / (b[1] - a[1])) * ex;
      if (point[0] < crossing) inside = !inside;
    }
  }
  return inside ? -Math.sqrt(squared) : Math.sqrt(squared);
}

/** The face's boundary loop, projected into its own plane. */
export const facePolygon2d = (face: ConstructionFace): [number, number][] =>
  face.polygon.map((point) => toFacePlane(face, point));

/** Every face the compile shipped, tagged with the node that declared it. */
export function allFaces(
  nodes: readonly ConstructionNode[],
): { face: ConstructionFace; nodeId: string }[] {
  const entries: { face: ConstructionFace; nodeId: string }[] = [];
  for (const node of nodes) {
    for (const face of node.faces ?? []) entries.push({ face, nodeId: node.id });
  }
  return entries;
}

/**
 * The declared face a surface hit belongs to, or null when it belongs to none.
 *
 * Four rejections, in the order they are cheapest to evaluate:
 *
 *  - **off the plane** — `|(p − o) · n| > tol`;
 *  - **outside the boundary** — the in-plane point further than `tol` outside
 *    the polygon (inclusive, so a hit exactly on a shared edge belongs to
 *    both faces rather than to neither);
 *  - **facing away** — `n̂ · n < 0.5`, which is what stops the far cap of a
 *    thin extrusion from claiming a hit on the near one;
 *
 * and then the survivors are ranked by plane distance, alignment breaking a
 * tie — two coplanar faces meeting at an edge are separated by which one the
 * surface actually faces along.
 */
export function pickFace(
  nodes: readonly ConstructionNode[],
  point: Vec3,
  normal: Vec3,
): FacePick | null {
  let best: FacePick | null = null;
  for (const { face, nodeId } of allFaces(nodes)) {
    const tolerance = faceTolerance(face);
    const planeDistance = Math.abs(dot(subtract(point, face.origin), face.normal));
    if (planeDistance > tolerance) continue;
    if (polygonDistance(toFacePlane(face, point), facePolygon2d(face)) > tolerance) continue;
    const alignment = dot(normal, face.normal);
    if (alignment < ALIGNMENT_FLOOR) continue;
    if (
      best === null ||
      planeDistance < best.planeDistance ||
      (planeDistance === best.planeDistance && alignment > best.alignment)
    ) {
      best = { face, nodeId, planeDistance, alignment };
    }
  }
  return best;
}

/**
 * Where a ray meets a declared face, for the solids that have no analytic
 * primitive to cast against.
 *
 * An extrusion is an SDF, not a box: `pickSurfacePoint` cannot see it, and
 * without this a pointer over the starter's fin comb would resolve nothing at
 * all. Casting against the face polygons is the same surface the raymarcher
 * draws wherever the feature is exact, which is exactly where a face pick is
 * meaningful.
 */
export function pickFaceSurface(
  nodes: readonly ConstructionNode[],
  ray: Ray,
): SurfaceHit | null {
  let best: SurfaceHit | null = null;
  for (const { face, nodeId } of allFaces(nodes)) {
    const denominator = dot(ray.direction, face.normal);
    if (Math.abs(denominator) < 1e-9) continue;
    const t = dot(subtract(face.origin, ray.origin), face.normal) / denominator;
    if (t <= 1e-6 || (best !== null && t >= best.t)) continue;
    const point = add(ray.origin, scale(ray.direction, t));
    const tolerance = faceTolerance(face);
    if (polygonDistance(toFacePlane(face, point), facePolygon2d(face)) > tolerance) continue;
    // Orient toward the viewer, exactly like the primitive cast does, so the
    // alignment test downstream compares like with like.
    const normal = denominator > 0 ? scale(face.normal, -1) : (face.normal as Vec3);
    best = {
      nodeId,
      point: [point[0], point[1], point[2]],
      normal: [normal[0], normal[1], normal[2]],
      t,
    };
  }
  return best;
}

/**
 * The nearest surface under a pick ray: analytic primitives *and* faces.
 *
 * Both casts run because neither covers the scene alone — a box has no
 * declared face polygon problem, an extrusion has no primitive — and the
 * nearer hit wins, so a bushing standing in front of the comb correctly
 * shadows it.
 */
export function resolveSurfaceHit(
  nodes: readonly ConstructionNode[],
  ray: Ray,
): SurfaceHit | null {
  const primitive = pickSurfacePoint(nodes, ray);
  const face = pickFaceSurface(nodes, ray);
  if (!primitive) return face;
  if (!face) return primitive;
  return face.t < primitive.t ? face : primitive;
}

/**
 * The `/patch` reference that reproduces a picked face in source.
 *
 * Null when the face has no variable to name — a feature built inside a loop
 * has faces the viewer can draw but cannot write back, and the honest answer
 * is to highlight it and refuse the action rather than to guess a name.
 */
export function faceReference(face: ConstructionFace): SketchPlaneReference | null {
  const owner = face.owner;
  if (!face.usable || !owner || owner.variable === null) return null;
  const [argument] = face.reference.args;
  switch (face.reference.call) {
    case "cap":
      return String(argument) === "-"
        ? { kind: "cap", owner: owner.line, sign: "-" }
        : { kind: "cap", owner: owner.line, sign: "+" };
    case "side":
      return { kind: "side", owner: owner.line, edge: Number(argument) };
    case "face":
      return { kind: "face", owner: owner.line, key: String(argument) };
    default:
      return null;
  }
}

/**
 * The tangent-plane fallback for a hit on a surface with no analytic face.
 *
 * `SketchPlane.tangent(solid, near=…)` reads the plane off the SDF's own
 * gradient at the picked point, so a curved wall still accepts a sketch — it
 * just stops being a *face* reference and becomes a point one.
 */
export function tangentReference(
  nodes: readonly ConstructionNode[],
  hit: SurfaceHit,
): SketchPlaneReference | null {
  const node = nodes.find((item) => item.id === hit.nodeId);
  if (!node) return null;
  const owner = (node.faces ?? []).find((face) => face.owner?.variable)?.owner;
  const line = owner?.line ?? node.line;
  if (line === null || line === undefined) return null;
  return { kind: "tangent", owner: line, near: hit.point };
}

/** A short human name for a face, for the hint bar: `sink.cap('+')`. */
export function faceLabel(face: ConstructionFace): string {
  const variable = face.owner?.variable ?? face.owner?.kind ?? "solid";
  const [argument] = face.reference.args;
  const rendered = typeof argument === "number" ? String(argument) : `'${argument}'`;
  return `${variable}.${face.reference.call}(${rendered})`;
}

/**
 * What a face click aimed at, in terms that survive an edit.
 *
 * A reference names its owner by *source line*, and placing a new sketch
 * inserts lines — so a reference resolved before the insert can point at the
 * wrong statement after it. Face ids (`profile_0:cap-`) and node ids are
 * stable across rebuilds, so the click carries those and the reference is
 * derived again from the recompiled tree, at the moment it is sent.
 */
export interface FaceTarget {
  /** The face that was picked, or null when the pick fell back to a tangent. */
  faceId: string | null;
  nodeId: string;
  /** The surface point, which a tangent plane is read at. */
  near: [number, number, number];
}

/** The `/patch` reference a target resolves to against the current tree. */
export function referenceFor(
  nodes: readonly ConstructionNode[],
  target: FaceTarget,
): SketchPlaneReference | null {
  if (target.faceId !== null) {
    const found = allFaces(nodes).find(({ face }) => face.id === target.faceId);
    return found ? faceReference(found.face) : null;
  }
  return tangentReference(nodes, {
    nodeId: target.nodeId,
    point: target.near,
    normal: [0, 0, 1],
    t: 0,
  });
}
