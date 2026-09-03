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

import type {
  ConstructionFace,
  ConstructionNode,
  GizmoMode,
  MeshEdgePayload,
  Selection,
} from "../types";
import { vertexState } from "./dragBinding";
import type { ShaderProgramPayload } from "./shaderProgram";
import { AXIS_COLORS, gizmoEdges, type AxisIndex } from "./gizmo";
import type { Vec3 } from "./math";
import { triangulate } from "./triangulate";

/** Bytes per instance: position pair + rgba (+ an active flag for gizmos). */
export const EDGE_STRIDE = 40;
/** Handle: centre + rgba + emphasis + fill. */
export const HANDLE_STRIDE = 36;
export const GIZMO_STRIDE = 44;
/** Bytes per *vertex* of the face highlight: position + rgba. */
export const FACE_STRIDE = 28;

export type Rgba = readonly [number, number, number, number];

/**
 * Construction-overlay ink, tuned for the paper viewport.
 *
 * Every value here is *darker* than the ground: on `#e6e6e9` an overlay reads
 * by weight, not by glow, so each tone is the lightest step of its hue that
 * still clears ~3:1 against paper. Measured against paper (`#e6e6e9`) and
 * against the lightest facet the SDF shading produces (`#c8c8cb`):
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
  // The face under the pointer, in --viewport-ink (#18161a) at two strengths.
  // Achromatic on purpose and not negotiable: inside the viewport rectangle
  // the field ramp owns colour, so an annotation that picked up a hue would
  // be readable as a value. What separates the highlight from everything else
  // is that it is a *surface* — nothing else in the overlay is filled.
  faceOutline: [0.094, 0.086, 0.102, 0.9],
  faceFill: [0.094, 0.086, 0.102, 0.12],
  // A face the source cannot name still highlights, at half the weight, so
  // "I see it, and I cannot write it" is one look rather than silence.
  faceOutlineLocked: [0.094, 0.086, 0.102, 0.45],
  faceFillLocked: [0.094, 0.086, 0.102, 0.05],
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
  program: ShaderProgramPayload | null = null,
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
      // Filled means the drag is live: the point is a free design parameter,
      // so moving it is a write into the shader's uniform buffer. A ring is a
      // point the shader holds as a constant, and dragging it costs a
      // recompile — the same disc, hollowed, because it is the same handle in
      // a weaker state rather than a different kind of thing.
      const live = vertexState(node.vertices[index], program) === "free";
      handles.push(
        position[0],
        position[1],
        position[2],
        ...handleColor,
        isSelected || isHovered ? 1 : 0,
        live ? 1 : 0,
      );
    }
  }

  return { edges, handles };
}

/**
 * Flatten a highlighted face into a filled surface and a hairline outline.
 *
 * The fill is ear-clipped in the face's own 2D frame and mapped straight back
 * to the boundary's world points, so a concave face — the starter's fin comb
 * is one — fills its notches correctly instead of printing a fan across them.
 * The outline reuses the edge instance format, so the hairline is drawn by
 * the same pipeline, at the same pixel width, as every other overlay line.
 */
export function packFaceHighlight(face: ConstructionFace | null): {
  fill: number[];
  outline: number[];
} {
  const fill: number[] = [];
  const outline: number[] = [];
  if (!face || face.polygon.length < 3) return { fill, outline };
  const fillColor = face.usable ? COLORS.faceFill : COLORS.faceFillLocked;
  const outlineColor = face.usable ? COLORS.faceOutline : COLORS.faceOutlineLocked;

  const local = face.polygon.map((point): [number, number] => {
    const delta: Vec3 = [
      point[0] - face.origin[0],
      point[1] - face.origin[1],
      point[2] - face.origin[2],
    ];
    return [
      delta[0] * face.xAxis[0] + delta[1] * face.xAxis[1] + delta[2] * face.xAxis[2],
      delta[0] * face.yAxis[0] + delta[1] * face.yAxis[1] + delta[2] * face.yAxis[2],
    ];
  });
  for (const index of triangulate(local)) {
    const point = face.polygon[index];
    fill.push(point[0], point[1], point[2], ...fillColor);
  }

  const count = face.polygon.length;
  for (let index = 0; index < count; index++) {
    const start = face.polygon[index];
    const end = face.polygon[(index + 1) % count];
    outline.push(start[0], start[1], start[2], end[0], end[1], end[2], ...outlineColor);
  }
  return { fill, outline };
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
