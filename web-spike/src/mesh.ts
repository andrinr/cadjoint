/**
 * The checks a slicer would make on a mesh, and binary STL output.
 * The mesh itself comes from src/dc.ts, cadjoint's dual contouring.
 */

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

/**
 * Bad edges as an STL reader sees them: vertices merged by position first,
 * triangles that collapse (zero-area slivers, cadjoint #52) dropped.
 */
export function badEdgesAfterWeld(m: { vertices: Float32Array; triangles: Uint32Array }): { bad: number; slivers: number } {
  const ids = new Map<string, number>();
  const remap = new Uint32Array(m.vertices.length / 3);
  for (let v = 0; v < remap.length; v++) {
    const key = `${m.vertices[3 * v]},${m.vertices[3 * v + 1]},${m.vertices[3 * v + 2]}`;
    if (!ids.has(key)) ids.set(key, ids.size);
    remap[v] = ids.get(key)!;
  }
  const kept: number[] = [];
  let slivers = 0;
  for (let t = 0; t < m.triangles.length; t += 3) {
    const a = remap[m.triangles[t]], b = remap[m.triangles[t + 1]], c = remap[m.triangles[t + 2]];
    if (a === b || b === c || a === c) { slivers++; continue; }
    kept.push(a, b, c);
  }
  return { bad: badEdges(Uint32Array.from(kept)), slivers };
}
