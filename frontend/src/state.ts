/**
 * Application state.
 *
 * The Python source is the single source of truth: viewer edits go through
 * `/patch`, the patched text replaces the editor contents, and a recompile
 * regenerates the construction tree. Nothing about a sketch is stored outside
 * the program text except the transient drag override below.
 */

import { createSignal } from "solid-js";
import type {
  ConstructionNode,
  GizmoMode,
  Selection,
  SelectionMode,
  ToolMode,
} from "./types";
import { placeEdges } from "./viewer/gizmo";

export const [source, setSource] = createSignal("");
export const [nodes, setNodes] = createSignal<ConstructionNode[]>([]);
export const [selection, setSelection] = createSignal<Selection | null>(null);
export const [hover, setHover] = createSignal<Selection | null>(null);
export const [tool, setTool] = createSignal<ToolMode>("select");
export const [status, setStatus] = createSignal({ kind: "", text: "Starting…" });
export const [viewerError, setViewerError] = createSignal("");
const [dismissedError, setDismissedError] = createSignal("");

/**
 * Show a viewer error unless the user already dismissed that same message.
 *
 * Every recompile re-reports a persistent problem (no WebGPU adapter, say), and
 * a banner that keeps returning would sit over the viewport blocking clicks.
 */
export function reportViewerError(message: string): void {
  if (message && message !== dismissedError()) setViewerError(message);
}

export function dismissViewerError(): void {
  setDismissedError(viewerError());
  setViewerError("");
}
export const [consoleText, setConsoleText] = createSignal("");
export const [busy, setBusy] = createSignal(false);
export const [dirty, setDirty] = createSignal(false);

/** Vertex being dragged, with its live sketch-plane position. */
export interface DragState {
  nodeId: string;
  vertexIndex: number;
  xy: [number, number];
}

/** A gizmo drag in flight, with the placement it has produced so far. */
export interface GizmoDrag {
  nodeId: string;
  mode: GizmoMode;
  axis: 0 | 1 | 2;
  position: [number, number, number];
  rotation: [number, number, number];
}

export const [drag, setDrag] = createSignal<DragState | null>(null);
export const [gizmoDrag, setGizmoDrag] = createSignal<GizmoDrag | null>(null);
export const [gizmoMode, setGizmoMode] = createSignal<GizmoMode>("translate");
export const [selectionMode, setSelectionMode] = createSignal<SelectionMode>("object");

/** Camera angles mirrored into a signal so the ViewCube can track them. */
export const [cameraAngles, setCameraAngles] = createSignal({ yaw: 0.75, pitch: 0.32 });

/** Find a construction node by id. */
export function nodeById(id: string): ConstructionNode | undefined {
  return nodes().find((node) => node.id === id);
}

/** Sketch profiles only — the nodes that carry editable vertices. */
export function profiles(): ConstructionNode[] {
  return nodes().filter((node) => node.kind === "profile");
}

/**
 * Profiles as they should be drawn right now.
 *
 * While a drag is in flight the moved vertex is patched in locally so the
 * sketch follows the pointer; the solid only rebuilds once the drag ends and
 * the source is recompiled.
 */
export function displayProfiles(): ConstructionNode[] {
  const active = drag();
  const gizmo = gizmoDrag();
  const all = nodes();
  if (gizmo) {
    return all.map((node) =>
      node.id === gizmo.nodeId && node.transform
        ? withPlacement(node, gizmo.position, gizmo.rotation)
        : node,
    );
  }
  if (!active) return all;
  return all.map((profile) => {
    if (profile.id !== active.nodeId || !profile.plane) return profile;
    const { origin, u, v } = profile.plane;
    const [x, y] = active.xy;
    const world: [number, number, number] = [
      origin[0] + u[0] * x + v[0] * y,
      origin[1] + u[1] * x + v[1] * y,
      origin[2] + u[2] * x + v[2] * y,
    ];
    const vertices = profile.vertices.map((vertex, index) =>
      index === active.vertexIndex ? { ...vertex, uv: active.xy, world } : vertex,
    );
    return { ...profile, vertices };
  });
}


/**
 * A primitive re-placed at a new position/rotation, for live drag feedback.
 *
 * Its outline is recomputed on the client so the wireframe tracks the pointer;
 * the solid itself only catches up when the patched source recompiles.
 */
function withPlacement(
  node: ConstructionNode,
  position: [number, number, number],
  rotation: [number, number, number],
): ConstructionNode {
  if (!node.transform) return node;
  const move = (point: [number, number, number]): [number, number, number] => [
    point[0] + (position[0] - node.transform!.position[0]),
    point[1] + (position[1] - node.transform!.position[1]),
    point[2] + (position[2] - node.transform!.position[2]),
  ];
  return {
    ...node,
    edges: placeEdges(node.edges, node.transform, position, rotation),
    // A sketch's handles ride along with its plane.
    vertices: node.vertices.map((vertex) => ({ ...vertex, world: move(vertex.world) })),
    transform: { ...node.transform, position, rotation },
  };
}
