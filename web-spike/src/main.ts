/**
 * The spike: can jax-js carry cadjoint's field to an export in the browser?
 *
 * 1. values at 2000 points against Python cadjoint (reference.json)
 * 2. grad of sum(field) w.r.t. the hub radius against a float64 central
 *    difference from Python
 * 3. time to sample a 160^3 grid, per backend
 * 4. surface nets on that grid -> watertight check, volume, STL download
 *
 * Results land in <pre id="out"> as JSON.
 */
import { defaultDevice, grad, init, jit, numpy as np } from "@jax-js/jax";
import { spaceFrameNode, type Pts } from "./sdf";
import { badEdges, stl, surfaceNets, volume } from "./mesh";
import reference from "../reference.json";

const out = document.getElementById("out")!;
const results: Record<string, unknown> = {};
const show = () => (out.textContent = JSON.stringify(results, null, 2));

function columns(flat: number[] | Float32Array): Pts {
  const n = flat.length / 3;
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = flat[3 * i]; y[i] = flat[3 * i + 1]; z[i] = flat[3 * i + 2]; }
  return { x: np.array(x), y: np.array(y), z: np.array(z) };
}

async function check(device: "wasm" | "webgpu") {
  defaultDevice(device);
  const r: Record<string, unknown> = {};
  results[device] = r;

  // 1. values
  const values = spaceFrameNode()(columns(reference.points)).dataSync() as Float32Array;
  let maxErr = 0;
  reference.values.forEach((v: number, i: number) => (maxErr = Math.max(maxErr, Math.abs(values[i] - v))));
  r.max_abs_error_vs_python = maxErr;
  show();

  // 2. gradient w.r.t. a parameter
  const total = (hub: np.Array) => np.sum(spaceFrameNode(hub)(columns(reference.points)));
  const g = (grad(total)(np.array(15)).dataSync() as Float32Array)[0];
  r.grad_hub = g;
  r.grad_hub_python_fd = reference.d_sum_d_hub_radius;
  r.grad_rel_error = Math.abs(g - reference.d_sum_d_hub_radius) / Math.abs(reference.d_sum_d_hub_radius);
  show();

  // 3. sample a 160^3 grid over [-42, 42]^3
  const n = 160, lo = -42, step = 84 / (n - 1);
  const flat = new Float32Array(n * n * n * 3);
  let q = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
    flat[q++] = lo + i * step; flat[q++] = lo + j * step; flat[q++] = lo + k * step;
  }
  const field = jit((p: Pts) => spaceFrameNode()(p));
  // First call compiles; time the second.
  await field(columns(flat.subarray(0, 3 * 4096))).data();
  const t0 = performance.now();
  const grid = (await field(columns(flat)).data()) as Float32Array;
  r.grid_points = n ** 3;
  r.grid_ms = Math.round(performance.now() - t0);
  show();

  // 4. mesh and export
  const t1 = performance.now();
  const mesh = surfaceNets({ n: [n, n, n], origin: [lo, lo, lo], step, values: grid });
  r.mesh_ms = Math.round(performance.now() - t1);
  r.triangles = mesh.triangles.length / 3;
  r.bad_edges = badEdges(mesh.triangles);
  r.volume_mm3 = Math.round(volume(mesh));
  const link = document.createElement("a");
  link.href = URL.createObjectURL(stl(mesh));
  link.download = `space_frame_node_${device}.stl`;
  link.textContent = `download ${link.download}`;
  document.body.append(link, document.createElement("br"));
  show();
}

(async () => {
  const available = await init();
  results.available_devices = available;
  show();
  for (const device of ["wasm", "webgpu"] as const) {
    if (!available.includes(device)) { results[device] = "unavailable"; show(); continue; }
    try { await check(device); }
    catch (error) { results[device] = { error: String(error), stack: (error as Error).stack?.split("\n").slice(0, 4) }; show(); }
  }
  results.done = true;
  show();
})();
