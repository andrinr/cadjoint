/**
 * The demo's design parameters, and everything that follows from their values.
 *
 * The playground's source of truth is the Python program: an edit is a patch,
 * and a compile brings back the shaders, the construction tree and the handle
 * positions. The demo has no Python, so its source of truth is one record of
 * parameter values, and this module derives from it what a compile would
 * have said:
 *
 * - the scene graph at those values (`graphAt`), from which the page builds
 *   its WGSL and its uniform contract, and on which jax-js meshes;
 * - the construction tree at those values (`constructionAt`), so each handle
 *   bound to a parameter sits where the parameter puts it;
 * - which handles the page may move (`handleLock`). Only a free parameter
 *   that no constraint holds: anything a constraint holds needs the solver,
 *   which is Python's, and stays at its shipped position, drawn as held.
 *
 * Everything here is pure: bundle data and values in, plain values out.
 */
import type {
  ConstructionNode,
  ConstructionTransform,
  DemoParameter,
  ParameterBinding,
  SceneGraph,
  SceneGraphNode,
} from "../types";
import { placeEdges } from "../viewer/gizmo";
import type { Bounds } from "../scene/bounds.ts";
import type { ShaderProgramPayload } from "../viewer/shaderProgram";

/** Parameter values by name, one array of components each. */
export type Values = Readonly<Record<string, readonly number[]>>;

/** Each parameter's value as the scene ships it. */
export function shippedValues(parameters: readonly DemoParameter[]): Record<string, number[]> {
  return Object.fromEntries(parameters.map((p) => [p.name, [...p.value]]));
}

/** Whether the page may move this parameter on its own. */
export const editable = (parameter: DemoParameter): boolean => parameter.drivenBy === null;

/** One component held inside the parameter's range. */
export function clampComponent(parameter: DemoParameter, index: number, value: number): number {
  return Math.min(parameter.upper[index], Math.max(parameter.lower[index], value));
}

/**
 * Only the values that may change, each held inside its range.
 *
 * What a handle drag or a field proposes goes through here before it is
 * adopted, so a driven parameter cannot be moved by any route and an
 * out-of-range number cannot reach the shader.
 */
export function acceptValues(
  parameters: readonly DemoParameter[],
  current: Values,
  proposed: Values,
): Record<string, number[]> {
  const next: Record<string, number[]> = Object.fromEntries(
    Object.entries(current).map(([name, value]) => [name, [...value]]),
  );
  for (const parameter of parameters) {
    const value = proposed[parameter.name];
    if (!value || !editable(parameter) || value.length !== parameter.components) continue;
    if (value.some((component) => !Number.isFinite(component))) continue;
    next[parameter.name] = value.map((component, i) => clampComponent(parameter, i, component));
  }
  return next;
}

/** The values that differ from the shipped ones, for a page URL or a reset. */
export function editedValues(parameters: readonly DemoParameter[], values: Values): Record<string, number[]> {
  const edited: Record<string, number[]> = {};
  for (const parameter of parameters) {
    const value = values[parameter.name];
    if (value && value.some((component, i) => component !== parameter.value[i])) edited[parameter.name] = [...value];
  }
  return edited;
}

function nodeAt(node: SceneGraphNode, values: Values): SceneGraphNode {
  const params = { ...node.params };
  for (const [key, declared] of Object.entries(node.free ?? {})) {
    if (Array.isArray(declared)) {
      // `vertices` / `vertices_b`: one declared name per vertex, null for a fixed one.
      const vertices = (params[key] as number[][]).map((vertex, i) => {
        const name = declared[i];
        const value = name ? values[name] : undefined;
        return value ? [...value] : vertex;
      });
      params[key] = vertices;
    } else {
      const value = values[declared];
      if (value) params[key] = value.length === 1 ? value[0] : [...value];
    }
  }
  return { ...node, params, children: node.children.map((child) => nodeAt(child, values)) };
}

/** The scene graph with every free parameter at `values`. */
export function graphAt(graph: SceneGraph, values: Values): SceneGraph {
  return { ...graph, root: nodeAt(graph.root, values) };
}

const byName = (parameters: readonly DemoParameter[]) => new Map(parameters.map((p) => [p.name, p]));

/** The binding's share of a value: the whole parameter, or one component of it. */
function bound(binding: ParameterBinding, values: Values): number[] | null {
  const value = values[binding.name];
  if (!value || value.length !== binding.components) return null;
  return [...value];
}

/**
 * One transform argument at `values`, or null when its bindings do not cover it.
 *
 * A position is one 3-component parameter; a rotation is three angle
 * scalars, one per component (`binding.index`); a dimension is a scalar or a
 * 3-vector.
 */
function argumentAt(
  bindings: readonly ParameterBinding[] | undefined,
  shipped: number | readonly number[],
  values: Values,
): number | number[] | null {
  if (!bindings?.length) return null;
  const width = typeof shipped === "number" ? 1 : shipped.length;
  const result = typeof shipped === "number" ? [shipped] : [...shipped];
  for (const binding of bindings) {
    const value = bound(binding, values);
    if (!value) return null;
    if (binding.index === null || binding.index === undefined) result.splice(0, value.length, ...value);
    else result[binding.index] = value[0];
  }
  return width === 1 ? result[0] : result.slice(0, width);
}

function transformAt(transform: ConstructionTransform, values: Values): ConstructionTransform | null {
  const bindings = transform.bindings ?? {};
  const position = argumentAt(bindings.position, transform.position, values) as number[] | null;
  const rotation = argumentAt(bindings.rotation, transform.rotation, values) as number[] | null;
  const dimensions = { ...transform.dimensions };
  let moved = position !== null || rotation !== null;
  for (const key of Object.keys(dimensions)) {
    const value = argumentAt(bindings[key], dimensions[key], values);
    if (value !== null) {
      dimensions[key] = value;
      moved = true;
    }
  }
  if (!moved) return null;
  return {
    ...transform,
    position: (position ?? transform.position) as ConstructionTransform["position"],
    rotation: (rotation ?? transform.rotation) as ConstructionTransform["rotation"],
    dimensions,
  };
}

/** A sketch-plane point in world coordinates. */
function planeWorld(node: ConstructionNode, uv: readonly number[]): [number, number, number] {
  const { origin, u, v } = node.plane!;
  return [0, 1, 2].map((i) => origin[i] + u[i] * uv[0] + v[i] * uv[1]) as [number, number, number];
}

/**
 * The construction tree with every handle at `values`.
 *
 * A sketch vertex bound to a parameter moves to it, and a polygon's outline
 * (one edge per vertex, closed) follows its vertices. A primitive whose
 * position, rotation or dimensions are bound is re-placed the way the
 * playground's gizmo preview re-places it. A vertex some constraint holds is
 * drawn as held (`free: false`) at its shipped position.
 */
export function constructionAt(
  nodes: readonly ConstructionNode[],
  parameters: readonly DemoParameter[],
  values: Values,
): ConstructionNode[] {
  const lookup = byName(parameters);
  return nodes.map((node) => {
    let changed = false;
    const vertices = node.vertices.map((vertex) => {
      const parameter = vertex.binding ? lookup.get(vertex.binding.name) : undefined;
      if (!parameter) return vertex;
      if (!editable(parameter)) {
        changed = true;
        return { ...vertex, free: false };
      }
      const uv = bound(vertex.binding!, values);
      if (!uv || !node.plane || uv.length !== 2) return vertex;
      changed = true;
      return { ...vertex, uv: uv as [number, number], world: planeWorld(node, uv) };
    });
    let edges = node.edges;
    if (changed && node.kind === "profile" && edges.length === vertices.length && vertices.length > 1) {
      edges = vertices.map((vertex, i) => [vertex.world, vertices[(i + 1) % vertices.length].world]);
    }
    let transform = node.transform;
    if (node.kind !== "profile" && node.transform) {
      const placed = transformAt(node.transform, values);
      if (placed) {
        edges = placeEdges(
          node.edges,
          node.transform,
          placed.position as [number, number, number],
          placed.rotation,
          placed.dimensions,
        );
        transform = placed;
        changed = true;
      }
    }
    return changed ? { ...node, vertices, edges, transform } : node;
  });
}

/** Why the bindings behind one handle cannot be moved here, or null when they can. */
function bindingsLock(
  bindings: readonly ParameterBinding[] | null | undefined,
  parameters: readonly DemoParameter[],
  program: ShaderProgramPayload | null,
): string | null {
  if (!bindings?.length) return "fixed value in the source: the demo moves free parameters only";
  const lookup = byName(parameters);
  for (const binding of bindings) {
    const parameter = lookup.get(binding.name);
    const slot = program?.parameters.find((s) => s.name === binding.name && s.components === binding.components);
    if (!parameter || !slot) return `${binding.name} is not in this scene's shader, so it has nothing to move`;
    if (!editable(parameter))
      return `held by ${parameter.drivenBy}: moving it needs the constraint solver, which runs in Python, not in this demo`;
  }
  return null;
}

/**
 * Why a handle cannot be dragged in the demo, or null when it can.
 *
 * @param handle A sketch vertex's index, or the gizmo argument a drag would
 *   write (`position`, `rotation`, `radius`, …).
 */
export function handleLock(
  node: ConstructionNode | undefined,
  handle: number | string,
  parameters: readonly DemoParameter[],
  program: ShaderProgramPayload | null,
): string | null {
  if (!node) return "nothing to move";
  if (typeof handle === "number") {
    const binding = node.vertices[handle]?.binding;
    return bindingsLock(binding ? [binding] : null, parameters, program);
  }
  return bindingsLock(node.transform?.bindings?.[handle], parameters, program);
}

/** The parameter values a handle edit writes, keyed by name. */
export function bindingValues(
  bindings: readonly ParameterBinding[] | null | undefined,
  value: readonly number[],
): Record<string, number[]> {
  const values: Record<string, number[]> = {};
  for (const binding of bindings ?? []) {
    values[binding.name] =
      binding.index === null || binding.index === undefined
        ? value.slice(0, binding.components)
        : [value[binding.index]];
  }
  return values;
}

/**
 * A box around the construction tree, grown by `margin` of its size on each
 * side: where to mesh and frame a scene whose graph gives no bounds (a node
 * the bounds pass cannot size, such as a smooth union with an unbounded
 * operand). Null for a tree with no geometry.
 */
export function constructionBounds(nodes: readonly ConstructionNode[], margin = 0.15): Bounds | null {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  const take = (p: readonly number[]) => p.forEach((c, i) => ((lo[i] = Math.min(lo[i], c)), (hi[i] = Math.max(hi[i], c))));
  for (const node of nodes) {
    node.edges.forEach((edge) => edge.forEach(take));
    node.vertices.forEach((vertex) => take(vertex.world));
  }
  if (!Number.isFinite(lo[0])) return null;
  const grow = margin * Math.max(...hi.map((h, i) => h - lo[i]));
  return {
    center: [0, 1, 2].map((i) => (lo[i] + hi[i]) / 2),
    half: [0, 1, 2].map((i) => (hi[i] - lo[i]) / 2 + grow),
    // World units: the construction tree is written in them.
    scale: 1,
  };
}
