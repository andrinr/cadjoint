/**
 * Compose generated kernels (src/kernels.gen.ts) into a field, from a JSON
 * scene graph exported by export_scene.py.
 *
 * This is the only hand-written part of the field: four wiring patterns,
 * each mirroring a cadjoint node's `sdf(child_sdf, p, ...)`:
 *
 *   primitive   kernel(p, ...params)
 *   transform   child(kernel(p, ...params))      Translate, Rotate
 *   field       kernel(child(p), ...params)      Offset, Shell
 *   boolean     fold kernel(acc, child_i(p), k)  Union, Difference, Intersection
 *               left to right, as cadjoint's n-ary sdf loops
 *
 * The formulas themselves are never written here.
 */
import { jit, vmap, numpy as np } from "@jax-js/jax";
import { kernels } from "./kernels.gen";
import type { Field, Pts } from "./sdf";

export type SceneNode = {
  type: string;
  params: Record<string, number | number[]>;
  children: SceneNode[];
};

/** A parameter to differentiate: the first node matching `type` with a positive `name`. */
export type Free = { type: string; name: string };

type PerPoint = (p: np.Array, free: np.Array | null) => np.Array;

function compose(node: SceneNode, free: Free | null, claimed: { done: boolean }): PerPoint {
  const kernel = kernels[node.type];
  if (!kernel) throw new Error(`no generated kernel for ${node.type}`);
  const isFree =
    free !== null && !claimed.done && node.type === free.type && Number(node.params[free.name]) > 0;
  if (isFree) claimed.done = true;
  // A free parameter comes in as an array; every other one is a constant.
  const param = (name: string, f: np.Array | null) =>
    isFree && name === free!.name ? f!.ref : np.array(node.params[name] as number | number[]);
  const params = (f: np.Array | null) => kernel.params.map((name) => param(name, f));
  const children = node.children.map((child) => compose(child, free, claimed));

  switch (kernel.kind) {
    case "primitive":
      return (p, f) => kernel.fn(p, ...params(f));
    case "transform":
      return (p, f) => children[0](kernel.fn(p, ...params(f)), f);
    case "field":
      return (p, f) => kernel.fn(children[0](p, f), ...params(f));
    case "boolean":
      return (p, f) => {
        let acc = children[0](p.ref, f);
        children.slice(1).forEach((child, i) => {
          const last = i === children.length - 2;
          acc = kernel.fn(acc, child(last ? p : p.ref, f), ...params(f));
        });
        if (children.length === 1) p.dispose();
        return acc;
      };
  }
}

/** The scene as a per-point function (p[3], free | null) -> distance. */
export function perPoint(scene: SceneNode, free: Free | null = null): PerPoint {
  const fn = compose(scene, free, { done: false });
  // Every use above took .ref on the free parameter; release the original.
  return (p, f) => {
    const d = fn(p, f);
    f?.dispose();
    return d;
  };
}

/** A batched field over point columns, for dc.ts and sampling. */
export function field(scene: SceneNode): Field {
  const one = perPoint(scene);
  const batched = jit(vmap((p: np.Array) => one(p, null)));
  return ({ x, y, z }: Pts) => batched(np.stack([x, y, z], 1));
}

/** sum(field(points)) as a function of the free parameter, for grad. */
export function totalOverFree(scene: SceneNode, free: Free, points: () => np.Array) {
  const one = perPoint(scene, free);
  const batched = vmap((p: np.Array, f: np.Array) => one(p, f));
  return (k: np.Array) => {
    const pts = points();
    const n = pts.shape[0];
    return np.sum(batched(pts, np.broadcastTo(k, [n])));
  };
}

// ---------------------------------------------------------------------------
// Per-node mode: one jitted, batched kernel per node *type*, composed at the
// batch level. The fused mode above hands jit the whole scene as one
// program, which WebGPU compiles very slowly for a 40-node scene; here the
// GPU compiles at most one small kernel per type (shared by every node of
// that type) and the scene costs one dispatch per node.

type Batched = (P: np.Array, free: np.Array | null) => np.Array;

const batchedKernels = new Map<string, (...args: np.Array[]) => np.Array>();

/** jit(vmap(kernel)) with points (or distances) batched and params not. */
export function batchedKernel(type: string) {
  let fn = batchedKernels.get(type);
  if (!fn) {
    const kernel = kernels[type];
    const params = kernel.params.map(() => null);
    const axes = kernel.kind === "boolean" ? [0, 0, ...params] : [0, ...params];
    fn = jit(vmap(kernel.fn, axes)) as (...args: np.Array[]) => np.Array;
    batchedKernels.set(type, fn);
  }
  return fn;
}

function composeBatched(node: SceneNode, free: Free | null, claimed: { done: boolean }): Batched {
  const kernel = kernels[node.type];
  if (!kernel) throw new Error(`no generated kernel for ${node.type}`);
  const run = batchedKernel(node.type);
  const isFree =
    free !== null && !claimed.done && node.type === free.type && Number(node.params[free.name]) > 0;
  if (isFree) claimed.done = true;
  const params = (f: np.Array | null) =>
    kernel.params.map((name) =>
      isFree && name === free!.name ? f!.ref : np.array(node.params[name] as number | number[]),
    );
  const children = node.children.map((child) => composeBatched(child, free, claimed));

  switch (kernel.kind) {
    case "primitive":
      return (P, f) => run(P, ...params(f));
    case "transform":
      return (P, f) => children[0](run(P, ...params(f)), f);
    case "field":
      return (P, f) => run(children[0](P, f), ...params(f));
    case "boolean":
      return (P, f) => {
        let acc = children[0](P.ref, f);
        children.slice(1).forEach((child, i) => {
          const last = i === children.length - 2;
          acc = run(acc, child(last ? P : P.ref, f), ...params(f));
        });
        if (children.length === 1) P.dispose();
        return acc;
      };
  }
}

/** The per-node counterpart of `field`. */
export function fieldPerNode(scene: SceneNode): Field {
  const fn = composeBatched(scene, null, { done: false });
  return ({ x, y, z }: Pts) => fn(np.stack([x, y, z], 1), null);
}

/** The per-node counterpart of `totalOverFree`. */
export function totalOverFreePerNode(scene: SceneNode, free: Free, points: () => np.Array) {
  const fn = composeBatched(scene, free, { done: false });
  return (k: np.Array) => {
    const d = fn(points(), k.ref);
    k.dispose();
    return np.sum(d);
  };
}
