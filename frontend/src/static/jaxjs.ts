/**
 * The scene's field on jax-js, with no Python server: dual contouring for the
 * STL export and the wireframe overlay, through the runtime and mesher of the
 * jax-js backend (`web-spike/src`, through `tools/jaxjs-runtime.ts`).
 *
 * That runtime reads `cadjoint-scene` v1. A v2 graph is a v1 graph with
 * more fields (a shared subtree is written out in full at each use), so it
 * is handed over as v1.
 */
import { defaultDevice, init } from "@jax-js/jax";
import { badEdges, dualContour, field, sample, sceneRoot, stl, volume } from "cadjoint-jaxjs-runtime";
import type { Bounds } from "../scene/bounds.ts";
import type { SceneGraph } from "../scene/graph.ts";

/** Where jax-js runs: WebGPU when the browser has it, else WebAssembly. */
export type ComputeDevice = "webgpu" | "wasm";

export type MeshResult = {
  vertices: Float32Array;
  triangles: Uint32Array;
  device: ComputeDevice;
  cells: [number, number, number];
  badEdges: number;
  volume: number;
  sampleMs: number;
  meshMs: number;
};

let ready: Promise<ComputeDevice> | null = null;

/** Start jax-js once, on the best device it found; rejects when it found neither. */
export function computeDevice(): Promise<ComputeDevice> {
  ready ??= init().then((available) => {
    const chosen = (["webgpu", "wasm"] as const).find((d) => available.includes(d));
    if (!chosen) throw new Error(`jax-js found no WebGPU or WebAssembly device (it has: ${available.join(", ") || "none"})`);
    defaultDevice(chosen);
    return chosen;
  });
  return ready;
}

/** Dual contouring over `bounds`, grown by a cell, with `longest` cells on the longest side. */
export async function meshScene(graph: SceneGraph, bounds: Bounds, longest: number): Promise<MeshResult> {
  const device = await computeDevice();
  const f = field(sceneRoot({ ...graph, version: 1 }));
  const size = bounds.half.map((h) => 2 * h);
  const spacing = Math.max(...size) / (longest - 2);
  const origin = bounds.center.map((c, i) => c - bounds.half[i] - spacing) as [number, number, number];
  const cells = size.map((s) => Math.ceil(s / spacing) + 2) as [number, number, number];
  const flat = new Float32Array((cells[0] + 1) * (cells[1] + 1) * (cells[2] + 1) * 3);
  let q = 0;
  for (let i = 0; i <= cells[0]; i++)
    for (let j = 0; j <= cells[1]; j++)
      for (let k = 0; k <= cells[2]; k++) {
        flat[q++] = origin[0] + i * spacing;
        flat[q++] = origin[1] + j * spacing;
        flat[q++] = origin[2] + k * spacing;
      }
  let t = performance.now();
  const values = sample(f, flat);
  const sampleMs = performance.now() - t;
  t = performance.now();
  const mesh = dualContour(f, { origin, spacing, cells, values });
  const meshMs = performance.now() - t;
  return {
    vertices: mesh.vertices,
    triangles: mesh.triangles,
    device,
    cells,
    badEdges: badEdges(mesh.triangles),
    volume: volume(mesh),
    sampleMs,
    meshMs,
  };
}

/** A binary STL of the mesh. */
export const stlBlob = (mesh: Pick<MeshResult, "vertices" | "triangles">): Blob => stl(mesh);
