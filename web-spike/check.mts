/**
 * Headless check of the generated path on Wasm (Node): the same four checks
 * the page runs, for each exported scene. Usage: npx tsx check.mts [scene...]
 */
import { init, defaultDevice, numpy as np, grad } from "@jax-js/jax";
import { field, totalOverFree, type SceneNode } from "./src/runtime.ts";
import { dualContour } from "./src/dc.ts";
import { badEdges, badEdgesAfterWeld, volume } from "./src/mesh.ts";
import { readFileSync } from "node:fs";

function columns(flat: Float32Array) {
  const n = flat.length / 3;
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = flat[3 * i]; y[i] = flat[3 * i + 1]; z[i] = flat[3 * i + 2]; }
  return { x: np.array(x), y: np.array(y), z: np.array(z) };
}
function firstBlend(node: SceneNode): number | null {
  if (node.type === "union" && Number(node.params.smoothness) > 0) return Number(node.params.smoothness);
  for (const c of node.children) { const k = firstBlend(c); if (k !== null) return k; }
  return null;
}

await init("wasm");
defaultDevice("wasm");
for (const name of process.argv.slice(2).length ? process.argv.slice(2) : ["node", "clevis"]) {
  const scene = JSON.parse(readFileSync(`${name}.scene.json`, "utf8")) as SceneNode;
  const ref = JSON.parse(readFileSync(`${name}.reference.json`, "utf8"));
  const refMesh = JSON.parse(readFileSync(`${name}.mesh.json`, "utf8"));
  const r: Record<string, unknown> = {};
  const f = field(scene);
  const pts = Float32Array.from(ref.points);
  const values = f(columns(pts)).dataSync() as Float32Array;
  r.max_abs_error = Math.max(...ref.values.map((v: number, i: number) => Math.abs(values[i] - v)));
  const k0 = firstBlend(scene);
  if (ref.grad && k0 !== null) {
    const total = totalOverFree(scene, { type: "union", name: "smoothness" }, () => np.array(pts).reshape([pts.length / 3, 3]));
    const g = (grad(total)(np.array(k0)).dataSync() as Float32Array)[0];
    r.grad_blend = g;
    r.grad_python = ref.grad.d_sum;
    r.grad_rel_error = Math.abs(g - ref.grad.d_sum) / Math.abs(ref.grad.d_sum);
  }
  const [cx, cy, cz] = refMesh.cells, s = refMesh.spacing, o = refMesh.origin;
  const flat = new Float32Array((cx + 1) * (cy + 1) * (cz + 1) * 3);
  let q = 0;
  for (let i = 0; i <= cx; i++) for (let j = 0; j <= cy; j++) for (let k = 0; k <= cz; k++) {
    flat[q++] = o[0] + i * s; flat[q++] = o[1] + j * s; flat[q++] = o[2] + k * s;
  }
  let t = performance.now();
  const grid = f(columns(flat)).dataSync() as Float32Array;
  r.sample_ms = Math.round(performance.now() - t);
  t = performance.now();
  const mesh = dualContour(f, { origin: o, spacing: s, cells: [cx, cy, cz], values: grid });
  r.mesh_ms = Math.round(performance.now() - t);
  r.mesh = { vertices: mesh.vertices.length / 3, triangles: mesh.triangles.length / 3, bad_edges: badEdges(mesh.triangles),
             welded: badEdgesAfterWeld(mesh), volume: Math.round(volume(mesh) * 10) / 10 };
  r.python_mesh = { vertices: refMesh.vertices, triangles: refMesh.triangles, bad_edges: refMesh.bad_edges,
                    volume: Math.round(refMesh.volume * 10) / 10 };
  console.log(name, JSON.stringify(r, null, 1));
}
