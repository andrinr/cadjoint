/**
 * cadjoint's dual contouring, ported: `extract_mesh` with `sharp=True`.
 *
 * Stage by stage, after the Python source it names:
 *   find_crossing_edges        cadjoint/meshing/edge_detection.py
 *   edge_hermite_data          cadjoint/meshing/edge_detection.py
 *   manifold_cell_incidence    cadjoint/meshing/features.py
 *   sharp_qef_vertices         cadjoint/meshing/dual_contouring.py
 *   dual_faces                 cadjoint/meshing/dual_contouring.py
 *
 * The field and its gradient run on jax-js (batched over every crossing
 * edge at once); the integer bookkeeping and the per-vertex 3x3 solves are
 * plain TypeScript.
 */
import { grad, jit, numpy as np } from "@jax-js/jax";
import type { Field, Pts } from "./sdf";
import { chunked } from "./chunk";

export type Lattice = {
  origin: [number, number, number];
  spacing: number;
  cells: [number, number, number]; // lattice has cells + 1 vertices per axis
  values: Float32Array; // row-major over the (cells + 1) lattice
};

export type Mesh = { vertices: Float32Array; triangles: Uint32Array; skippedBoundary: number };

type Edges = { axis: Int8Array; index: Int32Array; startInside: Uint8Array; count: number };

/** find_crossing_edges: every lattice edge whose ends straddle zero, by axis then index. */
function crossingEdges(l: Lattice): Edges {
  const [cx, cy, cz] = l.cells;
  const [vx, vy, vz] = [cx + 1, cy + 1, cz + 1];
  const at = (i: number, j: number, k: number) => l.values[(i * vy + j) * vz + k] < 0;
  const axis: number[] = [], index: number[] = [], startInside: number[] = [];
  const unit = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let a = 0; a < 3; a++) {
    const [ux, uy, uz] = unit[a];
    for (let i = 0; i < vx - ux; i++)
      for (let j = 0; j < vy - uy; j++)
        for (let k = 0; k < vz - uz; k++) {
          const s = at(i, j, k);
          if (s !== at(i + ux, j + uy, k + uz)) {
            axis.push(a); index.push(i, j, k); startInside.push(s ? 1 : 0);
          }
        }
  }
  return {
    axis: Int8Array.from(axis), index: Int32Array.from(index),
    startInside: Uint8Array.from(startInside), count: axis.length,
  };
}

/** edge_hermite_data: 16 bisections, one secant step, one guarded Newton step. */
function hermite(field: Field, l: Lattice, e: Edges, bisections = 16, newtonSteps = 1) {
  const n = e.count;
  const evalField = jit((p: Pts) => field(p));
  const evalGrad = jit(grad((p: Pts) => np.sum(field(p))));
  const sx = new Float64Array(n), sy = new Float64Array(n), sz = new Float64Array(n);
  const dx = new Float64Array(n), dy = new Float64Array(n), dz = new Float64Array(n);
  for (let m = 0; m < n; m++) {
    sx[m] = l.origin[0] + e.index[3 * m] * l.spacing;
    sy[m] = l.origin[1] + e.index[3 * m + 1] * l.spacing;
    sz[m] = l.origin[2] + e.index[3 * m + 2] * l.spacing;
    dx[m] = e.axis[m] === 0 ? l.spacing : 0;
    dy[m] = e.axis[m] === 1 ? l.spacing : 0;
    dz[m] = e.axis[m] === 2 ? l.spacing : 0;
  }
  const pointsAt = (t: Float64Array) => {
    const xs = new Float32Array(n), ys = new Float32Array(n), zs = new Float32Array(n);
    for (let m = 0; m < n; m++) { xs[m] = sx[m] + t[m] * dx[m]; ys[m] = sy[m] + t[m] * dy[m]; zs[m] = sz[m] + t[m] * dz[m]; }
    return [xs, ys, zs] as const;
  };
  const f = (t: Float64Array) =>
    chunked(...pointsAt(t), (p) => [evalField(p).dataSync() as Float32Array])[0];
  const g = (t: Float64Array) =>
    chunked(...pointsAt(t), (p) => {
      const r = evalGrad(p);
      return [r.x.dataSync(), r.y.dataSync(), r.z.dataSync()] as Float32Array[];
    });

  const lo = new Float64Array(n), hi = new Float64Array(n).fill(1);
  let fLo = Float64Array.from(f(lo)), fHi = Float64Array.from(f(hi));
  for (let it = 0; it < bisections; it++) {
    const mid = new Float64Array(n);
    for (let m = 0; m < n; m++) mid[m] = 0.5 * (lo[m] + hi[m]);
    const fMid = f(mid);
    for (let m = 0; m < n; m++) {
      // Bisect against the frozen detection-time orientation.
      const startSide = (fMid[m] < 0 || Object.is(fMid[m], -0)) === (e.startInside[m] === 1);
      if (startSide) { lo[m] = mid[m]; fLo[m] = fMid[m]; } else { hi[m] = mid[m]; fHi[m] = fMid[m]; }
    }
  }
  const t = new Float64Array(n);
  for (let m = 0; m < n; m++) {
    const den = fLo[m] - fHi[m];
    const frac = Math.abs(den) > 1e-30 ? fLo[m] / den : 0.5;
    t[m] = lo[m] + Math.min(1, Math.max(0, frac)) * (hi[m] - lo[m]);
  }
  for (let step = 0; step < newtonSteps; step++) {
    const residual = f(t);
    const [gx, gy, gz] = g(t);
    const proposed = new Float64Array(n);
    for (let m = 0; m < n; m++) {
      const slope = gx[m] * dx[m] + gy[m] * dy[m] + gz[m] * dz[m];
      const usable = Math.abs(slope) > 1e-12;
      const correction = Math.min(1, Math.max(-1, usable ? residual[m] / slope : 0));
      proposed[m] = t[m] - correction;
    }
    const fProposed = f(proposed);
    for (let m = 0; m < n; m++) if (Math.abs(fProposed[m]) <= Math.abs(residual[m])) t[m] = proposed[m];
  }
  const [px, py, pz] = pointsAt(t);
  let [gx, gy, gz] = g(t).map((a) => Float64Array.from(a));
  // Dead subgradient at the root: re-evaluate a little toward the inside end.
  const degenerate: number[] = [];
  for (let m = 0; m < n; m++) if (gx[m] ** 2 + gy[m] ** 2 + gz[m] ** 2 < 1e-12) degenerate.push(m);
  if (degenerate.length) {
    const nudged = Float64Array.from(t);
    for (const m of degenerate) nudged[m] += e.startInside[m] ? -1e-3 : 1e-3;
    const [fx, fy, fz] = g(nudged);
    for (const m of degenerate) { gx[m] = fx[m]; gy[m] = fy[m]; gz[m] = fz[m]; }
  }
  // unit_normals: safely normalized, zero stays zero.
  const normals = new Float64Array(3 * n), points = new Float64Array(3 * n);
  for (let m = 0; m < n; m++) {
    const len2 = gx[m] ** 2 + gy[m] ** 2 + gz[m] ** 2;
    const s = len2 > 1e-24 ? 1 / Math.sqrt(len2) : 0;
    normals[3 * m] = gx[m] * s; normals[3 * m + 1] = gy[m] * s; normals[3 * m + 2] = gz[m] * s;
    points[3 * m] = px[m]; points[3 * m + 1] = py[m]; points[3 * m + 2] = pz[m];
  }
  return { points, normals };
}

// features.py tables
const CELL_EDGE_OFFSETS = [
  [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]],
  [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]],
  [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
];
const CORNER_OFFSETS: number[][] = [];
for (const a of [0, 1]) for (const b of [0, 1]) for (const c of [0, 1]) CORNER_OFFSETS.push([a, b, c]);
const CORNER_EDGE_PAIRS: [number, number][] = [];
for (let corner = 0; corner < 8; corner++)
  for (const bit of [4, 2, 1]) if (!(corner & bit)) CORNER_EDGE_PAIRS.push([corner, corner | bit]);

type Incidence = { cells: Int32Array; rows: number[][]; rowOf: Map<string, number> };

/** manifold_cell_incidence: one row per (active cell, inside-corner component). */
function manifoldIncidence(l: Lattice, e: Edges): Incidence {
  const [cx, cy, cz] = l.cells;
  const [vy, vz] = [cy + 1, cz + 1];
  const inside = (i: number, j: number, k: number) => l.values[(i * vy + j) * vz + k] < 0;
  const cellKey = (i: number, j: number, k: number) => (i * cy + j) * cz + k;
  const candidates = new Map<number, { edge: number; corner: number }[]>();
  for (let m = 0; m < e.count; m++) {
    const a = e.axis[m];
    const s = [e.index[3 * m], e.index[3 * m + 1], e.index[3 * m + 2]];
    const inward = [0, 0, 0];
    if (!e.startInside[m]) inward[a] = 1;
    for (const off of CELL_EDGE_OFFSETS[a]) {
      const c = [s[0] - off[0], s[1] - off[1], s[2] - off[2]];
      if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= cx || c[1] >= cy || c[2] >= cz) continue;
      const corner = (s[0] + inward[0] - c[0]) * 4 + (s[1] + inward[1] - c[1]) * 2 + (s[2] + inward[2] - c[2]);
      const key = cellKey(c[0], c[1], c[2]);
      if (!candidates.has(key)) candidates.set(key, []);
      candidates.get(key)!.push({ edge: m, corner });
    }
  }
  const keys = [...candidates.keys()].sort((p, q) => p - q);
  const cells: number[] = [], rows: number[][] = [];
  const rowOf = new Map<string, number>();
  for (const key of keys) {
    const k = key % cz, j = Math.floor(key / cz) % cy, i = Math.floor(key / (cy * cz));
    const cornerInside = CORNER_OFFSETS.map(([a, b, c]) => inside(i + a, j + b, k + c));
    const labels = cornerInside.map((ins, c) => (ins ? c : 8));
    for (let changed = true; changed; ) {
      changed = false;
      for (const [a, b] of CORNER_EDGE_PAIRS)
        if (cornerInside[a] && cornerInside[b]) {
          const merged = Math.min(labels[a], labels[b]);
          if (labels[a] !== merged || labels[b] !== merged) { labels[a] = labels[b] = merged; changed = true; }
        }
    }
    const byComponent = new Map<number, number[]>();
    for (const { edge, corner } of candidates.get(key)!) {
      const component = labels[corner];
      if (component === 8) throw new Error("inside lattice inconsistent with the edge set");
      if (!byComponent.has(component)) byComponent.set(component, []);
      byComponent.get(component)!.push(edge);
    }
    for (const component of [...byComponent.keys()].sort((p, q) => p - q)) {
      const row = rows.length;
      cells.push(i, j, k);
      rows.push(byComponent.get(component)!);
      for (const edge of byComponent.get(component)!) rowOf.set(`${key}:${edge}`, row);
    }
  }
  return { cells: Int32Array.from(cells), rows, rowOf };
}

/** Eigen-decomposition of a symmetric 3x3 matrix (cyclic Jacobi), descending. */
function eigenSym3(A: number[][]): { values: number[]; vectors: number[][] } {
  const a = A.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 32; sweep++) {
    const off = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2;
    if (off < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((p, q) => a[q][q] - a[p][p]);
  return { values: order.map((i) => Math.max(a[i][i], 0)), vectors: order.map((i) => [v[0][i], v[1][i], v[2][i]]) };
}

/**
 * sharp_qef_vertices: a rank-revealing QEF solve per row. With N the
 * row's masked unit normals, N = U S V^T, so the truncated pseudo-inverse
 * solution is sum_j v_j (v_j . N^T b) / s_j^2 over the kept ranks, and the
 * s_j^2 are the eigenvalues of N^T N. Every rank limit is tried, clamped to
 * the cell, and the smallest QEF error wins.
 */
function sharpQefVertices(l: Lattice, inc: Incidence, h: { points: Float64Array; normals: Float64Array }, rcond = 5e-2) {
  const out = new Float32Array(inc.rows.length * 3);
  inc.rows.forEach((edges, r) => {
    const normals: number[][] = [], points: number[][] = [];
    const mass = [0, 0, 0];
    for (const m of edges) {
      const nrm = [h.normals[3 * m], h.normals[3 * m + 1], h.normals[3 * m + 2]];
      const usable = nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2 > 0.25;
      normals.push(usable ? nrm : [0, 0, 0]);
      const p = [h.points[3 * m], h.points[3 * m + 1], h.points[3 * m + 2]];
      points.push(p);
      for (let d = 0; d < 3; d++) mass[d] += p[d] / edges.length;
    }
    const AtA = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], Atb = [0, 0, 0];
    normals.forEach((nrm, e) => {
      const b = nrm[0] * (points[e][0] - mass[0]) + nrm[1] * (points[e][1] - mass[1]) + nrm[2] * (points[e][2] - mass[2]);
      for (let i = 0; i < 3; i++) { Atb[i] += nrm[i] * b; for (let j = 0; j < 3; j++) AtA[i][j] += nrm[i] * nrm[j]; }
    });
    const { values, vectors } = eigenSym3(AtA);
    const singular = values.map(Math.sqrt);
    const leading = Math.max(singular[0], 1e-30);
    const lo = [0, 1, 2].map((d) => l.origin[d] + inc.cells[3 * r + d] * l.spacing);
    let best: number[] | null = null, bestError = Infinity;
    for (const rank of [3, 2, 1]) {
      const candidate = mass.slice();
      for (let j = 0; j < rank; j++) {
        if (!(singular[j] > rcond * leading)) continue;
        const coeff = (vectors[j][0] * Atb[0] + vectors[j][1] * Atb[1] + vectors[j][2] * Atb[2]) / Math.max(values[j], 1e-60);
        for (let d = 0; d < 3; d++) candidate[d] += vectors[j][d] * coeff;
      }
      for (let d = 0; d < 3; d++) candidate[d] = Math.min(lo[d] + l.spacing, Math.max(lo[d], candidate[d]));
      let error = 0;
      normals.forEach((nrm, e) => {
        const res = nrm[0] * (candidate[0] - points[e][0]) + nrm[1] * (candidate[1] - points[e][1]) + nrm[2] * (candidate[2] - points[e][2]);
        error += res * res;
      });
      if (best === null || error < bestError) { best = candidate; bestError = error; }
    }
    out.set(best!, 3 * r);
  });
  return out;
}

/** dual_faces: one oriented quad per interior crossing edge, split on its shorter diagonal. */
function dualFaces(l: Lattice, e: Edges, inc: Incidence, vertices: Float32Array) {
  const [cx, cy, cz] = l.cells;
  const cellCounts = [cx, cy, cz];
  const cellKey = (c: number[]) => (c[0] * cy + c[1]) * cz + c[2];
  const NEIGHBORS = [[1, 1], [0, 1], [0, 0], [1, 0]];
  const tris: number[] = [];
  let skipped = 0;
  const dist2 = (a: number, b: number) =>
    (vertices[3 * a] - vertices[3 * b]) ** 2 + (vertices[3 * a + 1] - vertices[3 * b + 1]) ** 2 + (vertices[3 * a + 2] - vertices[3 * b + 2]) ** 2;
  for (let m = 0; m < e.count; m++) {
    const a = e.axis[m], u = (a + 1) % 3, v = (a + 2) % 3;
    const s = [e.index[3 * m], e.index[3 * m + 1], e.index[3 * m + 2]];
    if (!(s[u] > 0 && s[u] < cellCounts[u] && s[v] > 0 && s[v] < cellCounts[v])) { skipped++; continue; }
    const quad: number[] = [];
    for (const [du, dv] of NEIGHBORS) {
      const c = s.slice(); c[u] -= du; c[v] -= dv;
      const row = inc.rowOf.get(`${cellKey(c)}:${m}`);
      if (row === undefined) break;
      quad.push(row);
    }
    if (quad.length < 4) continue;
    if (!e.startInside[m]) quad.reverse();
    const [qa, qb, qc, qd] = quad;
    if (dist2(qa, qc) <= dist2(qb, qd)) tris.push(qa, qb, qc, qa, qc, qd);
    else tris.push(qa, qb, qd, qb, qc, qd);
  }
  return { triangles: Uint32Array.from(tris), skipped };
}

/** extract_mesh(sdf, grid, sharp=True) on a lattice already sampled. */
export function dualContour(field: Field, l: Lattice): Mesh {
  return dualContourStages(field, l).mesh;
}

/** The same, with every intermediate stage, for comparing against Python. */
export function dualContourStages(field: Field, l: Lattice) {
  const edges = crossingEdges(l);
  const incidence = manifoldIncidence(l, edges);
  const h = hermite(field, l, edges);
  const vertices = sharpQefVertices(l, incidence, h);
  const { triangles, skipped } = dualFaces(l, edges, incidence, vertices);
  return { edges, incidence, hermite: h, mesh: { vertices, triangles, skippedBoundary: skipped } };
}
