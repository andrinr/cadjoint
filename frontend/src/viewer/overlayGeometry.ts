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
import { normalTip, originMarkSize, type PlaneFrame } from "../planes";
import { AXIS_COLORS, gizmoEdges, type AxisIndex } from "./gizmo";
import type { Vec3 } from "./math";
import { triangulate } from "./triangulate";
import {
  CHROME,
  FIELD_VIEWPORT_STATES,
  OVERLAY_TONES,
  VIEWPORT_STATES,
  hexToRgb,
  overlayRgba,
} from "../tokens";

/** Bytes per instance: position pair + rgba (+ an active flag for gizmos). */
export const EDGE_STRIDE = 40;
/** Handle: centre + rgba + emphasis + shape (see HANDLE_SHAPE). */
export const HANDLE_STRIDE = 36;
export const GIZMO_STRIDE = 44;
/** Bytes per *vertex* of the face highlight: position + rgba. */
export const FACE_STRIDE = 28;

export type Rgba = readonly [number, number, number, number];

/**
 * Construction-overlay tones, evaluated from `OVERLAY_TONES` in tokens.ts.
 * The skeleton is the construction hue (C10), and state is its ladder (V1):
 * rest `construction`, hover `construction-strong`, selected `ink`, locked
 * `construction-locked`.
 */
export const COLORS: Record<string, Rgba> = Object.fromEntries(
  Object.keys(OVERLAY_TONES).map((name) => [name, overlayRgba(name)]),
);

const INK = hexToRgb(CHROME.ink);

const opaque = (token: keyof typeof CHROME): Rgba => [...hexToRgb(CHROME[token]), 1];

/**
 * The skeleton's tones while a solved field is on screen: the ink ladder of
 * `FIELD_VIEWPORT_STATES`, so the field stays the only colour (V2).
 */
const FIELD_COLORS = {
  edge: opaque(FIELD_VIEWPORT_STATES.rest),
  edgeHover: opaque(FIELD_VIEWPORT_STATES.hover),
  edgeSelected: opaque(FIELD_VIEWPORT_STATES.selected),
  edgeLocked: opaque(FIELD_VIEWPORT_STATES.locked),
  handleHover: opaque(FIELD_VIEWPORT_STATES.selected),
  handleSelected: opaque(FIELD_VIEWPORT_STATES.selected),
  handleLocked: opaque(FIELD_VIEWPORT_STATES.locked),
  held: opaque(FIELD_VIEWPORT_STATES.rest),
};

/** A held vertex's square at rest: the construction hue, like its edges. */
const HANDLE_HELD: Rgba = [...hexToRgb(CHROME[VIEWPORT_STATES.rest]), 1];

/** A free vertex's ring: the free-parameter blue `free=True` takes in the editor. */
const HANDLE_FREE: Rgba = [...hexToRgb(CHROME.param), 1];

/**
 * Handle shapes, the instance's last float (see `fs_handle`): free is a
 * hollow param-blue ring, held a filled construction-hue square, selected a
 * filled ink circle, each cased in sheet. Shape and hue together survive a
 * colour-blind reading.
 */
export const HANDLE_SHAPE = { free: 0, held: 1, selected: 2 } as const;

/**
 * Whether a sketch vertex is held: written as a constant in the source
 * (`free=False`), or pinned by a `fixed` constraint naming it alone or among
 * others. Everything else is free for a drag or the optimiser to move.
 */
export function vertexHeld(node: Pick<ConstructionNode, "vertices" | "constraints">, index: number): boolean {
  const vertex = node.vertices[index];
  if (vertex && !vertex.free) return true;
  return (node.constraints ?? []).some(
    (constraint) => constraint.kind === "fixed" && constraint.vertices.includes(index),
  );
}

/**
 * Flatten the construction tree into edge and handle instances.
 *
 * The payload ships a ready-made wireframe, so boxes, spheres, and sketches
 * all draw through one path; selection and hover only choose colours, and
 * whether a vertex is held (`vertexHeld`) chooses its shape.
 */
export function packConstructionOverlay(
  profiles: readonly ConstructionNode[],
  selection: Selection | null,
  hover: Selection | null,
  overField = false,
): { edges: number[]; handles: number[] } {
  const edges: number[] = [];
  const handles: number[] = [];
  const tones = overField
    ? FIELD_COLORS
    : {
        edge: COLORS.edge,
        edgeHover: COLORS.edgeHover,
        edgeSelected: COLORS.edgeSelected,
        edgeLocked: COLORS.edgeLocked,
        handleHover: COLORS.handleHover,
        handleSelected: COLORS.handleSelected,
        handleLocked: COLORS.handleLocked,
        held: HANDLE_HELD,
      };

  for (const node of profiles) {
    const selected = selection?.nodeId === node.id;
    // Whole-object hover previews the pick before it is committed.
    const hovered = hover?.nodeId === node.id && hover.vertexIndex === null;
    const edgeColor = selected
      ? tones.edgeSelected
      : hovered
        ? tones.edgeHover
        : node.editable
          ? tones.edge
          : tones.edgeLocked;
    for (const [start, end] of node.edges) {
      edges.push(start[0], start[1], start[2], end[0], end[1], end[2], ...edgeColor);
    }
    // A handle is drawn only where it can be taken: a derived (read-only)
    // node shows its skeleton and no handles, which on a scene built mostly
    // from derived geometry was a scatter of squares over the part.
    if (!node.editable) continue;
    for (let index = 0; index < node.vertices.length; index++) {
      const isSelected = selection?.nodeId === node.id && selection.vertexIndex === index;
      const isHovered = hover?.nodeId === node.id && hover.vertexIndex === index;
      const held = vertexHeld(node, index);
      const shape =
        isSelected
          ? HANDLE_SHAPE.selected
          : held
            ? HANDLE_SHAPE.held
            : HANDLE_SHAPE.free;
      // A free handle keeps its hue on hover and gains weight via the emphasis flag.
      const handleColor = isSelected
        ? tones.handleSelected
        : !held
          ? HANDLE_FREE
          : isHovered
            ? tones.handleHover
            : tones.held;
      const position = node.vertices[index].world;
      handles.push(
        position[0],
        position[1],
        position[2],
        ...handleColor,
        isSelected || isHovered ? 1 : 0,
        shape,
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

function pushSegment(target: number[], start: Vec3, end: Vec3, color: Rgba): void {
  target.push(start[0], start[1], start[2], end[0], end[1], end[2], ...color);
}

/**
 * Flatten the sketch planes into a filled quad each, plus the marks on it.
 *
 * The fill shares the face-highlight pipeline (the only filled thing in the
 * overlay), and the frame, the origin cross and the normal glyph share the
 * edge pipeline, so a plane costs no pipeline of its own. Two triangles per
 * quad; the outline is four segments, the origin cross two, the normal
 * glyph three (a shaft and two barbs), all in the plane's own frame so they
 * turn with it.
 */
export function packPlaneOverlay(
  frames: readonly PlaneFrame[],
  selection: Selection | null,
  hover: Selection | null,
  preview: PlaneFrame | null,
): { fill: number[]; outline: number[] } {
  const fill: number[] = [];
  const outline: number[] = [];
  const draw = (frame: PlaneFrame, fillColor: Rgba, lineColor: Rgba) => {
    const [a, b, c, d] = frame.corners;
    for (const point of [a, b, c, a, c, d]) {
      fill.push(point[0], point[1], point[2], ...fillColor);
    }
    pushSegment(outline, a, b, lineColor);
    pushSegment(outline, b, c, lineColor);
    pushSegment(outline, c, d, lineColor);
    pushSegment(outline, d, a, lineColor);
    // Origin cross, along u and v.
    const mark = originMarkSize(frame);
    const { origin, u, v, normal } = frame;
    const along = (axis: Vec3, k: number): Vec3 => [
      origin[0] + axis[0] * k,
      origin[1] + axis[1] * k,
      origin[2] + axis[2] * k,
    ];
    pushSegment(outline, along(u, -mark), along(u, mark), lineColor);
    pushSegment(outline, along(v, -mark), along(v, mark), lineColor);
    // Normal glyph: shaft plus two barbs leaning back along u and v.
    const tip = normalTip(frame);
    pushSegment(outline, origin, tip, lineColor);
    const barb = mark * 0.8;
    const back: Vec3 = [
      tip[0] - normal[0] * barb,
      tip[1] - normal[1] * barb,
      tip[2] - normal[2] * barb,
    ];
    pushSegment(outline, tip, [back[0] + u[0] * barb, back[1] + u[1] * barb, back[2] + u[2] * barb], lineColor);
    pushSegment(outline, tip, [back[0] - u[0] * barb, back[1] - u[1] * barb, back[2] - u[2] * barb], lineColor);
  };

  for (const frame of frames) {
    const selected = selection?.nodeId === frame.nodeId && selection.vertexIndex === null;
    const onPlane = selected && selection?.part === "plane";
    const hovered =
      hover?.nodeId === frame.nodeId && hover.vertexIndex === null && hover.part === "plane";
    if (frame.derived && !onPlane && !hovered) {
      draw(frame, COLORS.planeFillLocked, COLORS.planeOutlineLocked);
    } else if (onPlane) {
      draw(frame, COLORS.planeFillSelected, COLORS.planeOutlineSelected);
    } else if (hovered || selected) {
      draw(frame, COLORS.planeFillHover, COLORS.planeOutlineHover);
    } else {
      draw(frame, COLORS.planeFill, COLORS.planeOutline);
    }
  }
  if (preview) draw(preview, COLORS.planeFillPreview, COLORS.planeOutlinePreview);
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
    // The dragged or hovered axis keeps its hue and gains weight: pressure
    // darkens on paper, so it steps a third of the way toward ink rather than
    // toward white, and the shader widens it through the emphasis flag.
    const color: Rgba = active
      ? [
          base[0] + (INK[0] - base[0]) * 0.3,
          base[1] + (INK[1] - base[1]) * 0.3,
          base[2] + (INK[2] - base[2]) * 0.3,
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
