/**
 * Headless check of the generated path (Node, Wasm by default): for each
 * exported scene, values, one jax.grad, and the export mesh, against the
 * Python references export_scene.py wrote.
 *
 * Usage: npx tsx check.mts [scene...] [--per-node] [--no-mesh] [--stl]
 */
import { init, defaultDevice, numpy as np, grad } from "@jax-js/jax";
import { field, fieldPerNode, sceneRoot, totalOverFree, totalOverFreePerNode, unsupported, type SceneGraph } from "./src/runtime.ts";
import { dualContour } from "./src/dc.ts";
import { sample } from "./src/chunk.ts";
import { badEdges, badEdgesAfterWeld, stl, volume } from "./src/mesh.ts";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

function columns(flat: Float32Array) {
  const n = flat.length / 3;
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = flat[3 * i]; y[i] = flat[3 * i + 1]; z[i] = flat[3 * i + 2]; }
  return { x: np.array(x), y: np.array(y), z: np.array(z) };
}

/** The lattice export_scene.py meshed, in the same f32 arithmetic GridSpec uses. */
export function lattice(origin: number[], spacing: number, cells: number[]) {
  const [cx, cy, cz] = cells;
  const flat = new Float32Array((cx + 1) * (cy + 1) * (cz + 1) * 3);
  let q = 0;
  for (let i = 0; i <= cx; i++) for (let j = 0; j <= cy; j++) for (let k = 0; k <= cz; k++) {
    flat[q++] = origin[0] + i * spacing; flat[q++] = origin[1] + j * spacing; flat[q++] = origin[2] + k * spacing;
  }
  return flat;
}

await init("wasm");
defaultDevice("wasm");
const perNode = process.argv.includes("--per-node");
const doMesh = !process.argv.includes("--no-mesh");
const names = process.argv.slice(2).filter((a) => !a.startsWith("--"));
console.log(perNode ? "mode: per-node kernels" : "mode: fused scene kernel");
for (const name of names.length ? names : ["node", "clevis"]) {
  const scene = sceneRoot(JSON.parse(readFileSync(`${name}.scene.json`, "utf8")) as SceneGraph);
  const missing = unsupported(scene);
  if (missing.length) { console.log(name, "unsupported node types:", [...new Set(missing)]); continue; }
  const ref = JSON.parse(readFileSync(`${name}.reference.json`, "utf8"));
  const refMesh = JSON.parse(readFileSync(`${name}.mesh.json`, "utf8"));
  const r: Record<string, unknown> = {};
  let t0 = performance.now();
  const f = perNode ? fieldPerNode(scene) : field(scene);
  const pts = Float32Array.from(ref.points);
  const values = f(columns(pts)).dataSync() as Float32Array;
  r.first_call_ms = Math.round(performance.now() - t0);
  const scale = Math.max(...ref.values.map((v: number) => Math.abs(v)));
  r.max_abs_error = Math.max(...ref.values.map((v: number, i: number) => Math.abs(values[i] - v)));
  r.max_rel_error = (r.max_abs_error as number) / scale;
  if (ref.grad) {
    const free = { type: ref.grad.type, name: ref.grad.name, index: ref.grad.index };
    const total = (perNode ? totalOverFreePerNode : totalOverFree)(scene, free, () => np.array(pts).reshape([pts.length / 3, 3]));
    t0 = performance.now();
    const g = (grad(total)(np.array(ref.grad.value)).dataSync() as Float32Array)[0];
    r.grad_ms = Math.round(performance.now() - t0);
    r.grad = `d/d ${ref.grad.type}[${ref.grad.index}].${ref.grad.name}: ${g} vs ${ref.grad.d_sum}`;
    r.grad_rel_error = Math.abs(g - ref.grad.d_sum) / Math.abs(ref.grad.d_sum);
  }
  if (doMesh) {
    const { origin: o, spacing: s, cells } = refMesh;
    let t = performance.now();
    const grid = sample(f, lattice(o, s, cells));
    r.sample_ms = Math.round(performance.now() - t);
    t = performance.now();
    const mesh = dualContour(f, { origin: o, spacing: s, cells, values: grid });
    r.mesh_ms = Math.round(performance.now() - t);
    r.mesh = { vertices: mesh.vertices.length / 3, triangles: mesh.triangles.length / 3, bad_edges: badEdges(mesh.triangles),
               welded: badEdgesAfterWeld(mesh), volume: Math.round(volume(mesh) * 10) / 10 };
    r.python_mesh = { vertices: refMesh.vertices, triangles: refMesh.triangles, bad_edges: refMesh.bad_edges,
                      volume: Math.round(refMesh.volume * 10) / 10 };
    r.cells = cells;
    // Vertex by vertex against extract_mesh (same order when the counts match).
    const bin = `${name}.mesh_vertices.bin`;
    if (existsSync(bin)) {
      const py = new Float32Array(readFileSync(bin).buffer.slice(0));
      if (py.length === mesh.vertices.length) {
        let worst = 0, off = 0;
        for (let v = 0; v < py.length / 3; v++) {
          const d = Math.hypot(py[3 * v] - mesh.vertices[3 * v], py[3 * v + 1] - mesh.vertices[3 * v + 1], py[3 * v + 2] - mesh.vertices[3 * v + 2]);
          worst = Math.max(worst, d);
          if (d > 1e-3 * s && off++ < 8)
            ((r.off_vertices ??= []) as number[][][]).push([Array.from(py.subarray(3 * v, 3 * v + 3)), Array.from(mesh.vertices.subarray(3 * v, 3 * v + 3))]);
        }
        r.vertex_max_diff_cells = worst / s;
        r.vertices_off_by_1e3_cell = off;
      }
    }
    if (process.argv.includes("--stl"))
      writeFileSync(`${name}.jaxjs.stl`, Buffer.from(await stl(mesh).arrayBuffer()));
  }
  console.log(name, JSON.stringify(r));
}
