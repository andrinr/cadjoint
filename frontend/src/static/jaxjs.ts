/**
 * The scene's field on jax-js, with no Python server: values and the STL
 * export, through the runtime and mesher of the jax-js backend
 * (`web-spike/src`, through `tools/jaxjs-runtime.ts`).
 *
 * That runtime reads `cadjoint-scene` v1. A v2 graph is a v1 graph with
 * more fields (a shared subtree is written out in full at each use), so it
 * is handed over as v1.
 */
import { defaultDevice, init } from "@jax-js/jax";
import { badEdges, dualContour, field, sample, sceneRoot, stl, volume } from "cadjoint-jaxjs-runtime";
import type { Bounds } from "../scene/bounds.ts";
import type { SceneGraph } from "../scene/graph.ts";

export type ExportResult = {
  stl: Blob;
  device: string;
  cells: [number, number, number];
  vertices: number;
  triangles: number;
  badEdges: number;
  volume: number;
  sampleMs: number;
  meshMs: number;
};

let ready: Promise<string> | null = null;

/** jax-js on WebGPU when the browser has it, else Wasm. */
function device(): Promise<string> {
  ready ??= init().then((available) => {
    const chosen = available.includes("webgpu") ? "webgpu" : "wasm";
    defaultDevice(chosen);
    return chosen;
  });
  return ready;
}

/** Dual contouring over `bounds`, grown by a cell, with `longest` cells on the longest side. */
export async function exportMesh(graph: SceneGraph, bounds: Bounds, longest = 96): Promise<ExportResult> {
  const chosen = await device();
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
    stl: stl(mesh),
    device: chosen,
    cells,
    vertices: mesh.vertices.length / 3,
    triangles: mesh.triangles.length / 3,
    badEdges: badEdges(mesh.triangles),
    volume: volume(mesh),
    sampleMs,
    meshMs,
  };
}
