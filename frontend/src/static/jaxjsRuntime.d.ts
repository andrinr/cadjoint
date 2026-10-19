/** What the static viewer uses of the jax-js runtime (see tools/jaxjs-runtime.ts). */
declare module "cadjoint-jaxjs-runtime" {
  import type { numpy as np } from "@jax-js/jax";

  export type Pts = { x: np.Array; y: np.Array; z: np.Array };
  export type Field = (p: Pts) => np.Array;
  export type SceneNode = { type: string; params: Record<string, unknown>; children: SceneNode[] };
  export type Mesh = { vertices: Float32Array; triangles: Uint32Array; skippedBoundary: number };
  export type Lattice = {
    origin: [number, number, number];
    spacing: number;
    cells: [number, number, number];
    values: Float32Array;
  };

  export function sceneRoot(graph: { format: string; version: number; root: unknown }): SceneNode;
  export function field(scene: SceneNode): Field;
  export function sample(field: Field, flat: Float32Array): Float32Array;
  export function dualContour(field: Field, lattice: Lattice): Mesh;
  export function stl(mesh: { vertices: Float32Array; triangles: Uint32Array }): Blob;
  export function badEdges(triangles: Uint32Array): number;
  export function volume(mesh: { vertices: Float32Array; triangles: Uint32Array }): number;
}
