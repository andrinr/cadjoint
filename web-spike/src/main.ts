/**
 * Generated-kernel path, checked against Python cadjoint for two scenes.
 *
 * Per scene (exported by export_scene.py) and per backend:
 * 1. field values at 2000 points vs cadjoint
 * 2. jax-js grad of sum(field) w.r.t. the first blended Union's
 *    smoothness vs Python jax.grad through cadjoint's functional form
 * 3. sample the same lattice cadjoint meshed
 * 4. dual contouring (src/dc.ts) vs cadjoint's extract_mesh on it
 *
 * The field is composed by src/runtime.ts from src/kernels.gen.ts, which
 * emit_kernels.py generates from cadjoint's own Python through StableHLO.
 * Results land in <pre id="out"> as JSON.
 */
import { defaultDevice, grad, init, numpy as np } from "@jax-js/jax";
import { field, fieldPerNode, totalOverFree, type SceneNode } from "./runtime";
import { spaceFrameNode } from "./sdf";
import { dualContour } from "./dc";
import { badEdges, badEdgesAfterWeld, stl, volume } from "./mesh";
import clevisScene from "../clevis.scene.json";
import clevisReference from "../clevis.reference.json";
import clevisMesh from "../clevis.mesh.json";
import nodeScene from "../node.scene.json";
import nodeReference from "../node.reference.json";
import nodeMesh from "../node.mesh.json";

const SCENES = {
  node: { scene: nodeScene as unknown as SceneNode, reference: nodeReference, mesh: nodeMesh },
  clevis: { scene: clevisScene as unknown as SceneNode, reference: clevisReference, mesh: clevisMesh },
};

const out = document.getElementById("out")!;
const results: Record<string, Record<string, unknown>> = {};
const show = () => (out.textContent = JSON.stringify(results, null, 2));
/** Let the page repaint between steps, so a slow step shows where it is. */
const started = performance.now();
const tick = (step: string, r: Record<string, unknown>) => {
  r.step = step;
  console.log(`[spike] ${Math.round(performance.now() - started)} ms: ${step}`, JSON.stringify(r));
  show();
  return new Promise((go) => setTimeout(go, 0));
};

function columns(flat: number[] | Float32Array) {
  const n = flat.length / 3;
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = flat[3 * i]; y[i] = flat[3 * i + 1]; z[i] = flat[3 * i + 2]; }
  return { x: np.array(x), y: np.array(y), z: np.array(z) };
}

/** The first Union with a positive smoothness, depth first: the one runtime.ts frees. */
function firstBlend(node: SceneNode): number | null {
  if (node.type === "union" && Number(node.params.smoothness) > 0) return Number(node.params.smoothness);
  for (const child of node.children) {
    const k = firstBlend(child);
    if (k !== null) return k;
  }
  return null;
}

async function check(name: keyof typeof SCENES, device: "wasm" | "webgpu") {
  defaultDevice(device);
  const { scene, reference, mesh: refMesh } = SCENES[name];
  const r: Record<string, unknown> = {};
  results[name][device] = r;
  console.log(`[spike] start ${name} ${device}`);
  const f = field(scene);

  // 1. values
  await tick("values", r);
  const values = f(columns(reference.points)).dataSync() as Float32Array;
  let maxErr = 0;
  reference.values.forEach((v: number, i: number) => (maxErr = Math.max(maxErr, Math.abs(values[i] - v))));
  r.max_abs_error_vs_python = maxErr;
  show();

  // 2. gradient w.r.t. the blend
  await tick("grad", r);
  const k0 = firstBlend(scene);
  if (reference.grad && k0 !== null) {
    const n = reference.points.length / 3;
    const total = totalOverFree(scene, { type: "union", name: "smoothness" }, () =>
      np.array(Float32Array.from(reference.points)).reshape([n, 3]));
    const g = (grad(total)(np.array(k0)).dataSync() as Float32Array)[0];
    r.grad_blend = g;
    r.grad_blend_python_jax = reference.grad.d_sum;
    r.grad_rel_error = Math.abs(g - reference.grad.d_sum) / Math.abs(reference.grad.d_sum);
    show();
  }

  // 3. the lattice cadjoint meshed
  await tick("sample", r);
  const [cx, cy, cz] = refMesh.cells;
  const s = refMesh.spacing, o = refMesh.origin;
  const flat = new Float32Array((cx + 1) * (cy + 1) * (cz + 1) * 3);
  let q = 0;
  for (let i = 0; i <= cx; i++) for (let j = 0; j <= cy; j++) for (let k = 0; k <= cz; k++) {
    flat[q++] = o[0] + i * s; flat[q++] = o[1] + j * s; flat[q++] = o[2] + k * s;
  }
  await f(columns(flat.subarray(0, 3 * 4096))).data(); // compile
  const t0 = performance.now();
  const grid = (await f(columns(flat)).data()) as Float32Array;
  r.lattice_points = flat.length / 3;
  r.sample_ms = Math.round(performance.now() - t0);
  show();

  // 4. mesh
  await tick("mesh", r);
  const t1 = performance.now();
  const mesh = dualContour(f, { origin: o as [number, number, number], spacing: s, cells: [cx, cy, cz], values: grid });
  r.mesh_ms = Math.round(performance.now() - t1);
  r.vertices = mesh.vertices.length / 3;
  r.triangles = mesh.triangles.length / 3;
  r.bad_edges = badEdges(mesh.triangles);
  r.as_stl_reader_sees_it = badEdgesAfterWeld(mesh);
  r.volume_mm3 = Math.round(volume(mesh) * 10) / 10;
  r.python_extract_mesh = {
    vertices: refMesh.vertices, triangles: refMesh.triangles,
    bad_edges: refMesh.bad_edges, volume_mm3: Math.round(refMesh.volume * 10) / 10,
  };
  const link = document.createElement("a");
  link.href = URL.createObjectURL(stl(mesh));
  link.download = `${name}_${device}.stl`;
  link.textContent = `download ${link.download}`;
  document.body.append(link, document.createElement("br"));
  show();
}

async function quick() {
  // ?quick: the smallest end-to-end step, for finding where the page stalls.
  const t = performance.now();
  await init("wasm");
  defaultDevice("wasm");
  const r: Record<string, unknown> = { init_ms: Math.round(performance.now() - t) };
  results.quick = r;
  show();
  const hand = location.search.includes("hand");
  r.path = hand ? "hand-written node (sdf.ts)" : "generated clevis";
  const f = hand ? spaceFrameNode() : field(clevisScene as unknown as SceneNode);
  const pts = hand ? nodeReference.points : clevisReference.points;
  const t1 = performance.now();
  const v = f(columns(pts)).dataSync() as Float32Array;
  r.values_ms = Math.round(performance.now() - t1);
  r.first = v[0];
  r.python_first = (hand ? nodeReference : clevisReference).values[0];
  r.visibility = document.visibilityState;
  (results as Record<string, unknown>).done = true;
  show();
}

async function split() {
  // ?split[=node][&webgpu][&pernode]: compile and run times for one scene,
  // fused (one jitted scene kernel) or per-node (one jitted kernel per type).
  const device = location.search.includes("webgpu") ? "webgpu" : "wasm";
  const mode = location.search.includes("pernode") ? "per-node" : "fused";
  await init(device);
  defaultDevice(device);
  const which = location.search.includes("node") && !location.search.includes("pernode") ? "node"
    : location.search.includes("split=node") ? "node" : "clevis";
  const r: Record<string, unknown> = { device, mode, scene: which };
  results.split = r;
  const scene = (which === "node" ? nodeScene : clevisScene) as unknown as SceneNode;
  const ref = which === "node" ? nodeReference : clevisReference;
  const f = mode === "per-node" ? fieldPerNode(scene) : field(scene);
  const log = () => { console.log("[spike] split", JSON.stringify(r)); show(); return new Promise((go) => setTimeout(go, 0)); };
  const time = async (key: string, pts: Float32Array) => {
    const t = performance.now();
    await f(columns(pts)).data();
    r[key] = Math.round(performance.now() - t);
    await log();
  };
  const small = Float32Array.from(ref.points.slice(0, 3 * 256));
  await time("first_call_256_ms", small);
  await time("second_call_256_ms", small);
  const big = new Float32Array(3 * 1_000_000).map((_, i) => ((i * 7919) % 1000) / 10 - 50);
  await time("first_call_1M_ms", big);
  await time("second_call_1M_ms", big);
  (results as Record<string, unknown>).done = true;
  show();
}

(async () => {
  if (location.search.includes("split")) return split();
  if (location.search.includes("quick")) return quick();
  const available = await init();
  for (const name of Object.keys(SCENES) as (keyof typeof SCENES)[]) {
    results[name] = {};
    for (const device of ["wasm", "webgpu"] as const) {
      if (!available.includes(device)) { results[name][device] = "unavailable"; continue; }
      try { await check(name, device); }
      catch (error) {
        results[name][device] = { error: String(error), stack: (error as Error).stack?.split("\n").slice(0, 5) };
        show();
      }
    }
  }
  (results as Record<string, unknown>).done = true;
  show();
})();
