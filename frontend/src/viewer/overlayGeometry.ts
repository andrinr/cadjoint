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

export const COLORS: Record<string, Rgba> = {
  edge: [0.851, 1.0, 0.341, 0.95],
  edgeLocked: [0.58, 0.6, 0.56, 0.7],
  handle: [1.0, 0.506, 0.404, 1.0],
  handleSelected: [0.98, 0.99, 0.94, 1.0],
  handleHover: [1.0, 0.72, 0.4, 1.0],
  handleLocked: [0.62, 0.64, 0.6, 0.9],
  edgeSelected: [1.0, 0.95, 0.6, 1.0],
  edgeHover: [0.95, 1.0, 0.72, 1.0],
  meshWire: [0.5, 0.56, 0.62, 0.22],
  meshSharp: [0.35, 0.85, 1.0, 0.95],
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
