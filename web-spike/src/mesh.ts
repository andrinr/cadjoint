/**
 * Surface nets on a sampled grid, and the checks a slicer would make.
 *
 * Not cadjoint's dual contouring (no QEF, so no sharp corners): a
 * stand-in to show that a field sampled in the browser becomes a closed
 * mesh and an STL without a server.
 */

export type Grid = { n: [number, number, number]; origin: number[]; step: number; values: Float32Array };

export function surfaceNets(g: Grid): { vertices: Float32Array; triangles: Uint32Array } {
  const [nx, ny, nz] = g.n;
  const at = (i: number, j: number, k: number) => g.values[(i * ny + j) * nz + k];
  const cellId = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const verts: number[] = [];
  const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (let i = 0; i < nx - 1; i++)
    for (let j = 0; j < ny - 1; j++)
      for (let k = 0; k < nz - 1; k++) {
        const v = corners.map(([a, b, c]) => at(i + a, j + b, k + c));
        let sum = [0, 0, 0], count = 0;
        for (const [e0, e1] of edges) {
          if (v[e0] < 0 === v[e1] < 0) continue;
          const t = v[e0] / (v[e0] - v[e1]);
          for (let d = 0; d < 3; d++) sum[d] += corners[e0][d] + t * (corners[e1][d] - corners[e0][d]);
          count++;
        }
        if (!count) continue;
        cellId[(i * (ny - 1) + j) * (nz - 1) + k] = verts.length / 3;
        verts.push(
          g.origin[0] + (i + sum[0] / count) * g.step,
          g.origin[1] + (j + sum[1] / count) * g.step,
          g.origin[2] + (k + sum[2] / count) * g.step,
        );
      }
  const cell = (i: number, j: number, k: number) => cellId[(i * (ny - 1) + j) * (nz - 1) + k];
  const tris: number[] = [];
  const quad = (a: number, b: number, c: number, d: number, flip: boolean) => {
    if (flip) tris.push(a, c, b, a, d, c);
    else tris.push(a, b, c, a, c, d);
  };
  // One quad per sign-changing lattice edge, from the four cells around it.
  for (let i = 1; i < nx - 1; i++)
    for (let j = 1; j < ny - 1; j++)
      for (let k = 1; k < nz - 1; k++) {
        const inside = at(i, j, k) < 0;
        if (k + 1 < nz && inside !== at(i, j, k + 1) < 0)
          quad(cell(i - 1, j - 1, k), cell(i, j - 1, k), cell(i, j, k), cell(i - 1, j, k), !inside);
        if (j + 1 < ny && inside !== at(i, j + 1, k) < 0)
          quad(cell(i - 1, j, k - 1), cell(i - 1, j, k), cell(i, j, k), cell(i, j, k - 1), !inside);
        if (i + 1 < nx && inside !== at(i + 1, j, k) < 0)
          quad(cell(i, j - 1, k - 1), cell(i, j, k - 1), cell(i, j, k), cell(i, j - 1, k), !inside);
      }
  return { vertices: new Float32Array(verts), triangles: new Uint32Array(tris) };
}

/** Edges not shared by exactly two triangles (0 for a closed manifold surface). */
export function badEdges(triangles: Uint32Array): number {
  const counts = new Map<string, number>();
  for (let t = 0; t < triangles.length; t += 3)
    for (let e = 0; e < 3; e++) {
      const a = triangles[t + e], b = triangles[t + ((e + 1) % 3)];
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  let bad = 0;
  counts.forEach((c) => { if (c !== 2) bad++; });
  return bad;
}

/** Enclosed volume by the divergence theorem. */
export function volume(m: { vertices: Float32Array; triangles: Uint32Array }): number {
  const v = m.vertices, t = m.triangles;
  let total = 0;
  for (let i = 0; i < t.length; i += 3) {
    const a = t[i] * 3, b = t[i + 1] * 3, c = t[i + 2] * 3;
    total += v[a] * (v[b + 1] * v[c + 2] - v[b + 2] * v[c + 1])
      - v[a + 1] * (v[b] * v[c + 2] - v[b + 2] * v[c])
      + v[a + 2] * (v[b] * v[c + 1] - v[b + 1] * v[c]);
  }
  return total / 6;
}

/** Binary STL. */
export function stl(m: { vertices: Float32Array; triangles: Uint32Array }): Blob {
  const n = m.triangles.length / 3;
  const buf = new ArrayBuffer(84 + 50 * n);
  const view = new DataView(buf);
  view.setUint32(80, n, true);
  for (let f = 0; f < n; f++) {
    const o = 84 + 50 * f;
    for (let c = 0; c < 3; c++)
      for (let d = 0; d < 3; d++)
        view.setFloat32(o + 12 + c * 12 + d * 4, m.vertices[m.triangles[3 * f + c] * 3 + d], true);
  }
  return new Blob([buf], { type: "model/stl" });
}
