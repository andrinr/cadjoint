/**
 * Compose generated kernels (src/kernels.gen.ts) into a field, from a JSON
 * scene graph (`python -m cadjoint.backends.jaxjs scene`).
 *
 * This is the only hand-written part of the field, and it holds no
 * formulas, only how a node's kernels are wired:
 *
 *   primitive   sdf(p, ...params)
 *   operation   combine(child(query(p, ...params)), ...params)
 *               query is skipped when the child sees p itself (fields,
 *               booleans), combine when the node returns the child's value
 *               (transforms). n-ary nodes fold combine left over their
 *               children, as cadjoint's Booleans loop.
 *   pattern     fold combine over the kept instances: copy 0 is child(p),
 *               copy i is child(query(p, ...params, instance i)).
 *
 * Profiles are padded to the next kernel bucket by repeating their last
 * vertex (a zero-length edge, exact in cadjoint's polygon distance).
 *
 * Two compositions: fused (`field`: the scene as one per-point function,
 * jit(vmap) of the whole) and per-node (`fieldPerNode`: one jit(vmap)
 * kernel per node type and role, composed at batch level).
 */
import { jit, vmap, numpy as np } from "@jax-js/jax";
import { nodeTypes, polygonBuckets, sceneFormat, type Kernel } from "./kernels.gen";
import type { Field, Pts } from "./sdf";

export type SceneNode = {
  type: string;
  params: Record<string, number | number[] | number[][]>;
  children: SceneNode[];
};
export type SceneGraph = { format: string; version: number; root: SceneNode };

/** A parameter to differentiate: `name` of the `index`-th node of `type`, depth first. */
export type Free = { type: string; name: string; index: number };

/** The root of a scene graph, checking its format and version. */
export function sceneRoot(graph: SceneGraph): SceneNode {
  if (graph.format !== sceneFormat.format || graph.version !== sceneFormat.version)
    throw new Error(`scene is ${graph.format} v${graph.version}; this runtime reads ${sceneFormat.format} v${sceneFormat.version}`);
  return graph.root;
}

/** Every node type a scene uses, with the reason it cannot run if one is missing. */
export function unsupported(node: SceneNode, out: string[] = []): string[] {
  if (!nodeTypes[node.type]) out.push(node.type);
  node.children.forEach((c) => unsupported(c, out));
  return out;
}

type Fn = (p: np.Array, free: np.Array | null) => np.Array;
/** Call one kernel: per-point (fused) or batched (per-node). `batched` = how many leading args carry the batch. */
type Call = (type: string, role: string, batched: number, args: np.Array[]) => np.Array;

function bucketFor(n: number): number {
  const b = polygonBuckets.find((size) => size >= n);
  if (b === undefined) throw new Error(`profile with ${n} vertices; the largest kernel takes ${polygonBuckets.at(-1)}`);
  return b;
}

function padded(vertices: number[][], size: number): number[][] {
  const last = vertices[vertices.length - 1];
  return [...vertices, ...Array.from({ length: size - vertices.length }, () => last)];
}

function compose(node: SceneNode, call: Call, free: Free | null, seen: Map<string, number>): Fn {
  const spec = nodeTypes[node.type];
  if (!spec) throw new Error(`no generated kernels for ${node.type}`);
  const ordinal = seen.get(node.type) ?? 0;
  seen.set(node.type, ordinal + 1);
  const isFree = free !== null && node.type === free.type && ordinal === free.index;

  const vertices = node.params.vertices as number[][] | undefined;
  const bucket = vertices ? bucketFor(vertices.length) : 0;
  const value = (name: string) => {
    const v = node.params[name];
    return name.startsWith("vertices") ? padded(v as number[][], bucket) : v;
  };
  // A free parameter comes in as an array; every other one is a constant.
  const param = (name: string, f: np.Array | null) =>
    isFree && name === free!.name ? f!.ref : np.array(value(name) as number);
  const params = (f: np.Array | null) => spec.params.map((name) => param(name, f));
  const children = node.children.map((child) => compose(child, call, free, seen));
  const has = (role: string) => role in spec.kernels;

  if (spec.kind === "primitive") {
    const role = vertices ? `sdf_${bucket}` : "sdf";
    return (p, f) => call(node.type, role, 1, [p, ...params(f)]);
  }

  if (spec.kind === "pattern") {
    const count = Number(node.params.count), mask = Number(node.params.skip_mask);
    const kept = Array.from({ length: count }, (_, i) => i).filter((i) => !((mask >> i) & 1));
    const child = children[0];
    // The instance's own parameter, computed as cadjoint computes it:
    // spacing * i (LinearPattern) or -2π·i / count (PolarPattern), in f32.
    const instance = (i: number, f: np.Array | null) =>
      node.type === "linear_pattern"
        ? [param("direction", f), np.multiply(param("spacing", f), np.array(i))]
        : [param("origin", f), param("direction", f), np.divide(np.multiply(np.array(i), -2.0 * Math.PI), count)];
    return (p, f) => {
      let acc = child(kept.length > 1 ? p.ref : p, f);
      kept.slice(1).forEach((i, n) => {
        const last = n === kept.length - 2;
        const q = call(node.type, "query", 1, [last ? p : p.ref, ...instance(i, f)]);
        acc = call(node.type, "combine", 2, [acc, child(q, f)]);
      });
      return acc;
    };
  }

  if (spec.nary) {
    return (p, f) => {
      let acc = children[0](children.length > 1 ? p.ref : p, f);
      children.slice(1).forEach((child, i) => {
        const last = i === children.length - 2;
        acc = call(node.type, "combine", 2, [acc, child(last ? p : p.ref, f), ...params(f)]);
      });
      return acc;
    };
  }

  const child = children[0];
  return (p, f) => {
    const q = has("query") ? call(node.type, "query", 1, [p, ...params(f)]) : p;
    const d = child(q, f);
    return has("combine") ? call(node.type, "combine", 1, [d, ...params(f)]) : d;
  };
}

/** Releases the free parameter after a call: every use above took `.ref`. */
function owning(fn: Fn): Fn {
  return (p, f) => {
    const d = fn(p, f);
    f?.dispose();
    return d;
  };
}

const direct: Call = (type, role, _batched, args) => nodeTypes[type].kernels[role](...args);

const batchedKernels = new Map<string, Kernel>();
/** jit(vmap(kernel)) with the leading `batched` args batched and the parameters not. */
export function batchedKernel(type: string, role: string, batched: number, arity: number): Kernel {
  const key = `${type}:${role}`;
  let fn = batchedKernels.get(key);
  if (!fn) {
    const axes = Array.from({ length: arity }, (_, i) => (i < batched ? 0 : null));
    fn = jit(vmap(nodeTypes[type].kernels[role], axes)) as Kernel;
    batchedKernels.set(key, fn);
  }
  return fn;
}
const perNode: Call = (type, role, batched, args) => batchedKernel(type, role, batched, args.length)(...args);

/** The scene as a per-point function (p[3], free | null) -> distance. */
export function perPoint(scene: SceneNode, free: Free | null = null): Fn {
  return owning(compose(scene, direct, free, new Map()));
}

const stack = ({ x, y, z }: Pts) => np.stack([x, y, z], 1);

/** A batched field over point columns, the whole scene one jitted program. */
export function field(scene: SceneNode): Field {
  const one = perPoint(scene);
  const batched = jit(vmap((p: np.Array) => one(p, null)));
  return (pts: Pts) => batched(stack(pts));
}

/** The per-node counterpart of `field`: one dispatch per node. */
export function fieldPerNode(scene: SceneNode): Field {
  const fn = compose(scene, perNode, null, new Map());
  return (pts: Pts) => fn(stack(pts), null);
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

/** The per-node counterpart of `totalOverFree`. */
export function totalOverFreePerNode(scene: SceneNode, free: Free, points: () => np.Array) {
  const fn = owning(compose(scene, perNode, free, new Map()));
  return (k: np.Array) => np.sum(fn(points(), k));
}
