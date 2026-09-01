/**
 * Packing the construction overlay into GPU instance data.
 *
 * The overlay pipelines draw one instance per line segment and per handle, so
 * every frame the construction tree, the transform gizmo and the extracted
 * mesh edges have to be flattened into interleaved float arrays laid out to
 * match the vertex attributes in `overlay.wgsl`. That flattening is plain CPU
 * work over plain payloads — no device, no buffers — so it lives here, away
 * from the renderer's WebGPU state.
 *
 * The strides are the contract with the pipelines: change one and the
 * attribute offsets in `pipelines.ts` have to move with it.
 */

import type { ConstructionNode, GizmoMode, MeshEdgePayload, Selection } from "../types";
import { AXIS_COLORS, gizmoEdges, type AxisIndex } from "./gizmo";
import type { Vec3 } from "./math";

/** Bytes per instance: position pair + rgba (+ an active flag for gizmos). */
export const EDGE_STRIDE = 40;
export const HANDLE_STRIDE = 32;
export const GIZMO_STRIDE = 44;

export type Rgba = readonly [number, number, number, number];

/**
 * Construction-overlay ink, tuned for the paper viewport.
 *
 * Every value here is *darker* than the ground: on `#e6e6e9` an overlay reads
 * by weight, not by glow, so the whole set is the same hues the dark viewport
 * used, dropped to the lightest step that still clears ~3:1 against paper.
 * Measured against paper (`#e6e6e9`) and against the lightest facet the SDF
 * shading produces (`#c8c8cb`):
 *
 *   edge           #6a7f1a   3.62 / 2.70      edgeSelected   #915b16  4.54 / 3.39
 *   edgeHover      #809821   2.62 / 1.96      handle         #db3c1c  3.61 / 2.69
 *   handleSelected #17171b  14.35 / 10.71     handleHover    #d66b21  2.82 / 2.10
 *   meshSharp      #1f8696   3.43 / 2.56      meshWire       #5a5a60  5.50 / 4.10
 *
 * The hover tones sit deliberately below 3:1 — hover is a transient state
 * shown next to its resting colour, and the pair is separated by lightness
 * as well as by the shape under the pointer.
 */
export const COLORS: Record<string, Rgba> = {
  edge: [0.416, 0.498, 0.102, 0.95],
  edgeLocked: [0.482, 0.482, 0.502, 0.7],
  handle: [0.859, 0.235, 0.11, 1.0],
  handleSelected: [0.09, 0.09, 0.106, 1.0],
  handleHover: [0.839, 0.42, 0.129, 1.0],
  handleLocked: [0.549, 0.549, 0.569, 0.9],
  edgeSelected: [0.569, 0.357, 0.086, 1.0],
  edgeHover: [0.502, 0.596, 0.129, 1.0],
  meshWire: [0.353, 0.353, 0.376, 0.3],
  meshSharp: [0.122, 0.525, 0.588, 0.95],
};

/**
 * Flatten the construction tree into edge and handle instances.
 *
 * The payload ships a ready-made wireframe, so boxes, spheres, and sketches
 * all draw through one path; selection and hover only choose colours.
 */
export function packConstructionOverlay(
  profiles: readonly ConstructionNode[],
  selection: Selection | null,
  hover: Selection | null,
): { edges: number[]; handles: number[] } {
  const edges: number[] = [];
  const handles: number[] = [];

  for (const node of profiles) {
    const selected = selection?.nodeId === node.id;
    // Whole-object hover previews the pick before it is committed.
    const hovered = hover?.nodeId === node.id && hover.vertexIndex === null;
    const edgeColor = selected
      ? COLORS.edgeSelected
      : hovered
        ? COLORS.edgeHover
        : node.editable
          ? COLORS.edge
          : COLORS.edgeLocked;
    for (const [start, end] of node.edges) {
      edges.push(start[0], start[1], start[2], end[0], end[1], end[2], ...edgeColor);
    }
    for (let index = 0; index < node.vertices.length; index++) {
      const isSelected = selection?.nodeId === node.id && selection.vertexIndex === index;
      const isHovered = hover?.nodeId === node.id && hover.vertexIndex === index;
      const handleColor = !node.editable
        ? COLORS.handleLocked
        : isSelected
          ? COLORS.handleSelected
          : isHovered
            ? COLORS.handleHover
            : COLORS.handle;
      const position = node.vertices[index].world;
      handles.push(
        position[0],
        position[1],
        position[2],
        ...handleColor,
        isSelected || isHovered ? 1 : 0,
      );
    }
  }

  return { edges, handles };
}

/**
 * Flatten the transform gizmo into its own instance buffer.
 *
 * Transform controls have their own buffer and pass. Translation only needs
 * one instance per axis; rotation keeps its segmented rings. The hovered or
 * dragged axis is brightened and flagged so the shader can emphasise it.
 */
export function packGizmoInstances(
  origin: Vec3,
  size: number,
  mode: GizmoMode,
  activeAxis: AxisIndex | null,
): number[] {
  const gizmo: number[] = [];
  for (const group of gizmoEdges(origin, size, mode)) {
    const base = AXIS_COLORS[group.axis];
    const active = activeAxis === group.axis;
    const color: Rgba = active
      ? [
          base[0] + (1 - base[0]) * 0.38,
          base[1] + (1 - base[1]) * 0.38,
          base[2] + (1 - base[2]) * 0.38,
          1,
        ]
      : [base[0], base[1], base[2], 0.98];
    const visibleEdges = mode === "rotate" ? group.edges : group.edges.slice(0, 1);
    for (const [start, end] of visibleEdges) {
      gizmo.push(
        start[0],
        start[1],
        start[2],
        end[0],
        end[1],
        end[2],
        ...color,
        active ? 1 : 0,
      );
    }
  }
  return gizmo;
}

/**
 * Flatten the extracted mesh edges.
 *
 * They share the sketch edge pipeline in one buffer: wire first, sharp
 * second, so the two display switches can draw either instance range
 * independently.
 */
export function packMeshEdgeInstances(payload: MeshEdgePayload | null): number[] {
  const segments: number[] = [];
  if (!payload) return segments;
  for (const [start, end] of payload.wire) {
    segments.push(start[0], start[1], start[2], end[0], end[1], end[2], ...COLORS.meshWire);
  }
  for (const [start, end] of payload.sharp) {
    segments.push(start[0], start[1], start[2], end[0], end[1], end[2], ...COLORS.meshSharp);
  }
  return segments;
}
