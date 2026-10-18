/**
 * The page: cadjoint scenes evaluated, differentiated, meshed and exported
 * in the browser on generated jax-js kernels, checked against Python.
 *
 * Modes (query string; add `&webgpu` for WebGPU, `&pernode` for per-node
 * composition, default fused on Wasm):
 *   ?export=NAME   values, one grad, the export lattice, dual contouring and
 *                  an STL download for one exported scene, each against the
 *                  Python reference export_scene.py wrote
 *   (none)         ?export for every scene, on Wasm and WebGPU
 *   ?split=NAME    compile + run times at 256 and 1M points
 *   ?kernels       compile + run time of every node type's kernels
 * Results land in <pre id="out"> as JSON.
 */
import { defaultDevice, grad, init, numpy as np } from "@jax-js/jax";
import { batchedKernel, field, fieldPerNode, sceneRoot, totalOverFree, totalOverFreePerNode, type SceneGraph } from "./runtime";
import { nodeTypes, polygonBuckets } from "./kernels.gen";
import { dualContour } from "./dc";
import { sample } from "./chunk";
import { badEdges, badEdgesAfterWeld, stl, volume } from "./mesh";

type Reference = { points: number[]; values: number[]; grad: { type: string; name: string; index: number; value: number; d_sum: number } | null };
type MeshReference = { origin: number[]; spacing: number; cells: number[]; vertices: number; triangles: number; volume: number; bad_edges: number };

const glob = <T>(files: Record<string, unknown>) =>
  Object.fromEntries(Object.entries(files).map(([path, mod]) => [path.split("/").pop()!.split(".")[0], (mod as { default: T }).default]));
const scenes = glob<SceneGraph>(import.meta.glob("../*.scene.json", { eager: true }));
const references = glob<Reference>(import.meta.glob("../*.reference.json", { eager: true }));
const meshes = glob<MeshReference>(import.meta.glob("../*.mesh.json", { eager: true }));

const query = new URLSearchParams(location.search);
const device = query.has("webgpu") ? "webgpu" : "wasm";
const perNode = query.has("pernode");
const out = document.getElementById("out")!;
const results: Record<string, unknown> = {};
const show = () => (out.textContent = JSON.stringify(results, null, 2));
const tick = () => { show(); return new Promise((go) => setTimeout(go, 0)); };

function lattice({ origin: o, spacing: s, cells: [cx, cy, cz] }: MeshReference) {
  const flat = new Float32Array((cx + 1) * (cy + 1) * (cz + 1) * 3);
  let q = 0;
  for (let i = 0; i <= cx; i++) for (let j = 0; j <= cy; j++) for (let k = 0; k <= cz; k++) {
    flat[q++] = o[0] + i * s; flat[q++] = o[1] + j * s; flat[q++] = o[2] + k * s;
  }
  return flat;
}

async function exportScene(name: string, dev: "wasm" | "webgpu", r: Record<string, unknown>) {
  defaultDevice(dev);
  const scene = sceneRoot(scenes[name]);
  const ref = references[name], refMesh = meshes[name];
  const f = perNode ? fieldPerNode(scene) : field(scene);

  let t = performance.now();
  const values = sample(f, Float32Array.from(ref.points));
  r.values_ms = Math.round(performance.now() - t);
  r.max_abs_error = ref.values.reduce((m, v, i) => Math.max(m, Math.abs(values[i] - v)), 0);
  await tick();

  if (ref.grad) {
    const g0 = ref.grad;
    const n = ref.points.length / 3;
    const total = (perNode ? totalOverFreePerNode : totalOverFree)(scene, g0, () => np.array(Float32Array.from(ref.points)).reshape([n, 3]));
    t = performance.now();
    const g = (await grad(total)(np.array(g0.value)).data())[0];
    r.grad_ms = Math.round(performance.now() - t);
    r.grad_rel_error = Math.abs(g - g0.d_sum) / Math.abs(g0.d_sum);
    await tick();
  }

  t = performance.now();
  const grid = sample(f, lattice(refMesh));
  r.sample_ms = Math.round(performance.now() - t);
  r.lattice_points = grid.length;
  await tick();

  t = performance.now();
  const mesh = dualContour(f, { origin: refMesh.origin as [number, number, number], spacing: refMesh.spacing, cells: refMesh.cells as [number, number, number], values: grid });
  r.mesh_ms = Math.round(performance.now() - t);
  r.mesh = { vertices: mesh.vertices.length / 3, triangles: mesh.triangles.length / 3, bad_edges: badEdges(mesh.triangles),
             welded: badEdgesAfterWeld(mesh), volume: volume(mesh) };
  r.python_mesh = { vertices: refMesh.vertices, triangles: refMesh.triangles, bad_edges: refMesh.bad_edges, volume: refMesh.volume };
  const link = document.createElement("a");
  link.href = URL.createObjectURL(stl(mesh));
  link.download = `${name}_${dev}.stl`;
  link.textContent = `download ${link.download}`;
  document.body.append(link, document.createElement("br"));
  await tick();
}

async function split(name: string) {
  const f = perNode ? fieldPerNode(sceneRoot(scenes[name])) : field(sceneRoot(scenes[name]));
  const r: Record<string, unknown> = { device, mode: perNode ? "per-node" : "fused", scene: name };
  results.split = r;
  // Through sample(), as an export evaluates: 1M points of a large fused
  // scene in one call exceed jax-js's 4 GiB Wasm heap.
  const time = async (key: string, pts: Float32Array) => {
    const t = performance.now();
    sample(f, pts);
    r[key] = Math.round(performance.now() - t);
    await tick();
  };
  const small = Float32Array.from(references[name].points.slice(0, 3 * 256));
  await time("first_call_256_ms", small);
  await time("second_call_256_ms", small);
  const { origin: o, spacing: s, cells } = meshes[name];
  const extent = cells.map((c) => c * s);
  const big = new Float32Array(3 * 1_000_000).map((_, i) => o[i % 3] + (((i * 7919) % 1000) / 1000) * extent[i % 3]);
  await time("first_call_1M_ms", big);
  await time("second_call_1M_ms", big);
}

async function kernelTimes() {
  // Every kernel batched over 256 points (or distances), logged before it
  // runs so a kernel that stalls is named.
  const r: Record<string, unknown> = { device };
  results.kernels = r;
  const n = 256;
  const example = (type: string, name: string, bucket: number) =>
    type === "revolved_polygon" && name === "offset" ? 0.1
    : name.startsWith("vertices") ? Array.from({ length: bucket }, (_, i) => [Math.cos(i), Math.sin(i)])
      : ["size", "offset", "axis", "scale", "origin", "normal", "direction"].includes(name) ? [0.3, 0.5, 0.8] : 0.5;
  for (const [type, spec] of Object.entries(nodeTypes)) {
    for (const role of Object.keys(spec.kernels)) {
      const bucket = role.startsWith("sdf_") ? Number(role.slice(4)) : polygonBuckets[0];
      const pts = () => np.array(Float32Array.from({ length: 3 * n }, (_, i) => Math.sin(i))).reshape([n, 3]);
      const d = () => np.array(Float32Array.from({ length: n }, (_, i) => Math.cos(i)));
      let args: np.Array[];
      if (spec.kind === "pattern") {
        const extra = type === "linear_pattern" ? [np.array([1, 0, 0]), np.array(0.5)] : [np.zeros([3]), np.array([0, 0, 1]), np.array(1.0)];
        args = role === "query" ? [pts(), ...extra] : [d(), d()];
      } else {
        const params = spec.params.map((p) => np.array(example(type, p, bucket)));
        args = role === "combine" ? [...(spec.nary ? [d(), d()] : [d()]), ...params] : [pts(), ...params];
      }
      const batched = spec.kind === "pattern" ? (role === "query" ? 1 : 2) : role === "combine" ? (spec.nary ? 2 : 1) : 1;
      console.log(`[spike] compiling ${type}.${role} on ${device}`);
      const t = performance.now();
      await batchedKernel(type, role, batched, args.length)(...args).data();
      r[`${type}.${role}`] = Math.round(performance.now() - t);
      await tick();
    }
  }
}

(async () => {
  try {
    if (query.has("export")) {
      await init(device);
      const name = query.get("export") || "clevis";
      const r: Record<string, unknown> = { device, mode: perNode ? "per-node" : "fused" };
      results[name] = r;
      await exportScene(name, device, r);
    } else if (query.has("split")) {
      await init(device);
      defaultDevice(device);
      await split(query.get("split") || "clevis");
    } else if (query.has("kernels")) {
      await init(device);
      defaultDevice(device);
      await kernelTimes();
    } else {
      const available = await init();
      for (const name of Object.keys(scenes)) {
        const row: Record<string, unknown> = {};
        results[name] = row;
        for (const dev of ["wasm", "webgpu"] as const) {
          if (!available.includes(dev)) { row[dev] = "unavailable"; continue; }
          const r: Record<string, unknown> = {};
          row[dev] = r;
          try { await exportScene(name, dev, r); }
          catch (error) { r.error = String(error); await tick(); }
        }
      }
    }
  } catch (error) {
    results.error = { message: String(error), stack: (error as Error).stack?.split("\n").slice(0, 6) };
  }
  results.done = true;
  show();
})();
