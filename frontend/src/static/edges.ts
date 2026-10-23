/**
 * The mesh-edge overlay from a dual-contour mesh, built in the browser.
 *
 * `wire` is the quad wireframe: every mesh edge once, less the diagonal each
 * quad was split on (dual contouring emits one quad per crossing edge, as two
 * consecutive triangles). `sharp` is a crease classification by
 * dihedral angle: an edge whose two triangles' normals part by more than
 * `CREASE_DEGREES`. Dual contouring places its vertices on sharp features
 * (the QEF solve), so on this mesh a crease is a geometric edge and a
 * filleted blend is not one.
 *
 * It is not the Python overlay's lattice classifier
 * (`cadjoint/viewer/_lattice_edges.py`), which also finds the exact CSG
 * seams between operands and Newton-projects them onto both zero sets; that
 * needs every leaf's field and its Jacobian, and is not ported. Both are the
 * public tier's approximate edges.
 */
import type { MeshEdgePayload } from "../types";

/** Normals parting by more than this make an edge a crease. */
export const CREASE_DEGREES = 35;

type Vec3 = [number, number, number];

export function meshEdges(
  vertices: Float32Array,
  triangles: Uint32Array,
  resolution: number,
  creaseDegrees = CREASE_DEGREES,
): MeshEdgePayload {
  const point = (i: number): Vec3 => [vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]];
  const normals: Vec3[] = [];
  // Edge key → the triangles on it; a manifold edge has two.
  const faces = new Map<number, number[]>();
  const count = vertices.length / 3;
  for (let t = 0; t < triangles.length / 3; t++) {
    const [a, b, c] = [triangles[3 * t], triangles[3 * t + 1], triangles[3 * t + 2]];
    const [p, q, r] = [point(a), point(b), point(c)];
    const u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    const v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
    const n: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const length = Math.hypot(...n) || 1;
    normals.push([n[0] / length, n[1] / length, n[2] / length]);
    for (const [i, j] of [[a, b], [b, c], [c, a]]) {
      const key = Math.min(i, j) * count + Math.max(i, j);
      const list = faces.get(key);
      if (list) list.push(t);
      else faces.set(key, [t]);
    }
  }
  const limit = Math.cos((creaseDegrees * Math.PI) / 180);
  const wire: Vec3[][] = [];
  const sharp: Vec3[][] = [];
  for (const [key, list] of faces) {
    // A quad's own diagonal: the edge its two triangles, 2k and 2k + 1, share.
    if (list.length === 2 && list[0] % 2 === 0 && list[1] === list[0] + 1) continue;
    const segment = [point(Math.floor(key / count)), point(key % count)];
    wire.push(segment);
    if (list.length !== 2) continue;
    const [m, n] = [normals[list[0]], normals[list[1]]];
    if (m[0] * n[0] + m[1] * n[1] + m[2] * n[2] < limit) sharp.push(segment);
  }
  return { wire, sharp, resolution, edges: "lattice" };
}
