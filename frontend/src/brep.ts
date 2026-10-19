/**
 * The B-rep inspector: what the private tier's extraction found, made drawable.
 *
 * The server (`/api/brep`, `cadjoint/viewer/_brep_inspector.py`) serves the
 * derived B-rep as plain data — a tessellation with a face id per triangle,
 * every edge with its kind and polyline, every vertex. This module is the
 * pure half of the inspector: which colour each face gets, how each edge
 * kind is stroked, the GPU instance arrays the renderer uploads, and the
 * screen-space picking behind the hover readout. No device, no DOM, so all
 * of it is unit tested (`test/brep.test.ts`).
 *
 * **Edge kinds are open-ended.** The kind is the private tier's word, passed
 * through: `traced` and `sampled` for the curves the feature-edge overlay
 * draws, the graph's own state (`blend`, `tangent`, …) for the rest. A kind
 * this file has no tone for is drawn in the `other` tone and listed under
 * its own name, so a new state from diff-brep shows up without a change here.
 */

import type {
  BRepEdgePayload,
  BRepInspectionPayload,
  BRepVertexPayload,
} from "./types";
import {
  BREP_EDGE_TONES,
  BREP_FACE_PALETTE,
  BREP_OFF_SURFACE,
  CHROME,
  hexToRgb,
  type Rgb,
} from "./tokens";
import { pointSegmentDistance } from "./viewer/hittest";
import { projectPoint, type View } from "./viewer/math";

export type BrepPayload = BRepInspectionPayload;

/** A drawable B-rep: the payload with the fields a drawing needs present. */
export interface BrepModel {
  positions: number[];
  triangles: number[];
  triangleFace: number[];
  /** Flat x, y, z per triangle corner, or null to shade the facets flat. */
  cornerNormals: number[] | null;
  faces: NonNullable<BRepInspectionPayload["faces"]>;
  edges: BRepEdgePayload[];
  vertices: BRepVertexPayload[];
  diagnostics: NonNullable<BRepInspectionPayload["diagnostics"]>;
}

/** The payload as a model, or null when it carries nothing to draw. */
export function brepModel(payload: BrepPayload | null | undefined): BrepModel | null {
  if (!payload || !payload.available || payload.error) return null;
  const { positions, triangles, triangle_face, faces, edges, vertices, diagnostics } = payload;
  if (!positions || !triangles || !triangle_face || !faces || !edges || !vertices || !diagnostics) {
    return null;
  }
  return {
    positions,
    triangles,
    triangleFace: triangle_face,
    cornerNormals:
      payload.corner_normals && payload.corner_normals.length === triangles.length * 3
        ? payload.corner_normals
        : null,
    faces,
    edges,
    vertices,
    diagnostics,
  };
}

// ── colour ──────────────────────────────────────────────────────────────────

/**
 * Give every face a palette index no neighbour shares, where the palette allows.
 *
 * Faces are neighbours when an edge separates them, and — more weakly — when
 * they meet at a vertex: two faces that touch only at a corner still sit side
 * by side on screen. Greedy colouring in order of decreasing degree
 * (Welsh–Powell) takes the least-used index free of every edge neighbour,
 * preferring one also free of the corner neighbours. The B-rep of a solid
 * is near-planar and the palette has nine entries, so an edge clash is
 * rare; where one is forced, the index that clashes with fewest neighbours
 * wins. Deterministic: the same B-rep gets the same colours every time.
 *
 * @returns face id -> palette index.
 */
export function colourFaces(
  model: Pick<BrepModel, "faces" | "edges" | "vertices">,
  paletteSize = BREP_FACE_PALETTE.length,
): Map<number, number> {
  const strong = new Map<number, Set<number>>();
  const weak = new Map<number, Set<number>>();
  for (const face of model.faces) {
    strong.set(face.id, new Set());
    weak.set(face.id, new Set());
  }
  const link = (table: Map<number, Set<number>>, a: number, b: number) => {
    if (a === b || !table.has(a) || !table.has(b)) return;
    table.get(a)!.add(b);
    table.get(b)!.add(a);
  };
  for (const edge of model.edges) link(strong, edge.faces[0], edge.faces[1]);
  for (const vertex of model.vertices) {
    for (let i = 0; i < vertex.faces.length; i++) {
      for (let j = i + 1; j < vertex.faces.length; j++) link(weak, vertex.faces[i], vertex.faces[j]);
    }
  }
  const order = [...strong.keys()].sort(
    (a, b) => strong.get(b)!.size - strong.get(a)!.size || a - b,
  );
  const colour = new Map<number, number>();
  const used = new Array<number>(paletteSize).fill(0);
  for (const face of order) {
    let best = 0;
    let bestScore = Infinity;
    for (let index = 0; index < paletteSize; index++) {
      let clashes = 0;
      for (const other of strong.get(face)!) if (colour.get(other) === index) clashes += 1;
      let corners = 0;
      for (const other of weak.get(face)!) if (colour.get(other) === index) corners += 1;
      // Edge clashes dominate, corner clashes break ties among those, and
      // use count spreads the palette so no tone carries the whole part.
      const score = clashes * 1e6 + corners * 1e3 + used[index];
      if (score < bestScore) {
        bestScore = score;
        best = index;
      }
    }
    colour.set(face, best);
    used[best] += 1;
  }
  return colour;
}

/** One face's colour, from its palette index. */
export function faceColour(index: number | undefined): Rgb {
  return BREP_FACE_PALETTE[(index ?? 0) % BREP_FACE_PALETTE.length];
}

/** The tone an edge kind is stroked in; unknown kinds fall back to `other`. */
export function edgeTone(kind: string): { rgb: Rgb; label: string; known: boolean } {
  const tone = BREP_EDGE_TONES[kind];
  if (tone) return { rgb: hexToRgb(CHROME[tone.token]), label: tone.label, known: true };
  const other = BREP_EDGE_TONES.other;
  return { rgb: hexToRgb(CHROME[other.token]), label: kind, known: false };
}

/**
 * The edge kinds present, in legend order: the known kinds in the order the
 * tone table lists them, then any unknown ones alphabetically, with counts.
 */
export function edgeLegend(model: Pick<BrepModel, "edges">): {
  kind: string;
  label: string;
  rgb: Rgb;
  count: number;
}[] {
  const counts = new Map<string, number>();
  for (const edge of model.edges) counts.set(edge.kind, (counts.get(edge.kind) ?? 0) + 1);
  const known = Object.keys(BREP_EDGE_TONES).filter((kind) => counts.has(kind));
  const unknown = [...counts.keys()].filter((kind) => !(kind in BREP_EDGE_TONES)).sort();
  return [...known, ...unknown].map((kind) => {
    const tone = edgeTone(kind);
    return { kind, label: tone.label, rgb: tone.rgb, count: counts.get(kind)! };
  });
}

/** Whether an edge lies off the solid by more than the server's tolerance. */
export function offSurface(edge: BRepEdgePayload, model: Pick<BrepModel, "diagnostics">): boolean {
  return edge.off_surface > model.diagnostics.off_surface_tolerance;
}

// ── packing ─────────────────────────────────────────────────────────────────

/** What the user is pointing at. */
export type BrepHover =
  | { kind: "face"; id: number }
  | { kind: "edge"; id: number }
  | { kind: "vertex"; id: number }
  | null;

export interface BrepStyle {
  /** Edge kinds switched off in the legend. */
  hiddenKinds: ReadonlySet<string>;
  /** Draw the off-surface edges at full strength and dim the rest. */
  offSurfaceOnly: boolean;
  /** Outline every triangle of the tessellation, faint, under the edges. */
  showTriangles?: boolean;
  hover: BrepHover;
}

/** A hovered face is lifted this far toward white: a wash, not a new hue. */
const HOVER_LIFT = 0.45;
/** Edges that are not the point of the off-surface view, at this alpha. */
const DIMMED_ALPHA = 0.18;
/** The tessellation's own triangle outlines, at this alpha of the ink. */
const TRIANGLE_ALPHA = 0.3;

/**
 * Every distinct edge of the tessellation's triangles, as segment
 * instances (`EDGE_STRIDE`): the polygons the extraction actually produced,
 * shown on demand so a face's triangulation can be judged.
 */
export function packBrepTriangleOutlines(model: BrepModel): number[] {
  const ink = hexToRgb(CHROME.ink);
  const seen = new Set<number>();
  const count = model.positions.length / 3;
  const out: number[] = [];
  for (let t = 0; t < model.triangles.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = model.triangles[t + k];
      const b = model.triangles[t + ((k + 1) % 3)];
      if (a === b) continue;
      const key = Math.min(a, b) * count + Math.max(a, b);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(
        model.positions[a * 3], model.positions[a * 3 + 1], model.positions[a * 3 + 2],
        model.positions[b * 3], model.positions[b * 3 + 1], model.positions[b * 3 + 2],
        ink[0], ink[1], ink[2], TRIANGLE_ALPHA,
      );
    }
  }
  return out;
}

const lift = (rgb: Rgb, amount: number): Rgb => [
  rgb[0] + (1 - rgb[0]) * amount,
  rgb[1] + (1 - rgb[1]) * amount,
  rgb[2] + (1 - rgb[2]) * amount,
];

/**
 * Triangle-list vertices for the face pass: position + rgba + normal, 10
 * floats each (`BREP_FACE_STRIDE`). Faces do not share vertices — a corner
 * of two faces is two vertices in two colours, each with its own face's
 * normal. Without corner normals the normal is zero and the shader shades
 * the facet flat.
 */
export function packBrepFaces(
  model: BrepModel,
  colours: ReadonlyMap<number, number>,
  hover: BrepHover,
): Float32Array {
  const count = model.triangleFace.length;
  const out = new Float32Array(count * 3 * 10);
  const normals = model.cornerNormals;
  const hoveredFace = hover?.kind === "face" ? hover.id : null;
  let cursor = 0;
  for (let t = 0; t < count; t++) {
    const face = model.triangleFace[t];
    const base = faceColour(colours.get(face));
    const rgb = face === hoveredFace ? lift(base, HOVER_LIFT) : base;
    for (let k = 0; k < 3; k++) {
      const p = model.triangles[t * 3 + k] * 3;
      out[cursor++] = model.positions[p];
      out[cursor++] = model.positions[p + 1];
      out[cursor++] = model.positions[p + 2];
      out[cursor++] = rgb[0];
      out[cursor++] = rgb[1];
      out[cursor++] = rgb[2];
      out[cursor++] = 1;
      const n = (t * 3 + k) * 3;
      out[cursor++] = normals ? normals[n] : 0;
      out[cursor++] = normals ? normals[n + 1] : 0;
      out[cursor++] = normals ? normals[n + 2] : 0;
    }
  }
  return out;
}

/** An edge's rgba under the current style, or null when it is not drawn. */
export function edgeRgba(
  edge: BRepEdgePayload,
  model: Pick<BrepModel, "diagnostics">,
  style: BrepStyle,
): [number, number, number, number] | null {
  if (style.hiddenKinds.has(edge.kind)) return null;
  const hovered = style.hover?.kind === "edge" && style.hover.id === edge.id;
  if (style.offSurfaceOnly) {
    if (offSurface(edge, model)) return [...hexToRgb(CHROME[BREP_OFF_SURFACE]), 1];
    const rgb = edgeTone(edge.kind).rgb;
    return [rgb[0], rgb[1], rgb[2], hovered ? 1 : DIMMED_ALPHA];
  }
  const rgb = edgeTone(edge.kind).rgb;
  return [rgb[0], rgb[1], rgb[2], 1];
}

/**
 * Segment instances for the edge pass: start, end, rgba — 10 floats each
 * (`EDGE_STRIDE`). The hovered edge is drawn twice, the second copy last,
 * so it lands on top of whatever crosses it.
 */
export function packBrepEdges(model: BrepModel, style: BrepStyle): Float32Array {
  // The triangle outlines go first, so every graph edge is drawn over them.
  const segments: number[] = style.showTriangles ? packBrepTriangleOutlines(model) : [];
  const push = (edge: BRepEdgePayload, rgba: readonly number[]) => {
    const points = edge.points;
    const n = Math.floor(points.length / 3);
    const last = edge.closed ? n : n - 1;
    for (let i = 0; i < last; i++) {
      const a = i * 3;
      const b = ((i + 1) % n) * 3;
      segments.push(
        points[a], points[a + 1], points[a + 2],
        points[b], points[b + 1], points[b + 2],
        rgba[0], rgba[1], rgba[2], rgba[3],
      );
    }
  };
  let hovered: BRepEdgePayload | null = null;
  for (const edge of model.edges) {
    const rgba = edgeRgba(edge, model, style);
    if (!rgba) continue;
    if (style.hover?.kind === "edge" && style.hover.id === edge.id) hovered = edge;
    push(edge, rgba);
  }
  if (hovered) push(hovered, [...hexToRgb(CHROME.ink), 1]);
  return new Float32Array(segments);
}

/**
 * Handle instances for the vertex pass: centre, rgba, emphasis, shape — 9
 * floats each (`HANDLE_STRIDE`). A filled disc (shape 2) at under half the
 * sketch handle's radius (emphasis −1.5): a vertex is a mark, not a grip.
 */
export function packBrepVertices(model: BrepModel, hover: BrepHover): Float32Array {
  const out = new Float32Array(model.vertices.length * 9);
  const ink = hexToRgb(CHROME.ink);
  model.vertices.forEach((vertex, index) => {
    const hovered = hover?.kind === "vertex" && hover.id === vertex.id;
    out.set(
      [...vertex.point, ink[0], ink[1], ink[2], 1, hovered ? -0.6 : -1.5, 2],
      index * 9,
    );
  });
  return out;
}

// ── picking ─────────────────────────────────────────────────────────────────

/** Pixels within which a vertex or an edge wins over the face under it. */
export const PICK_RADIUS = { vertex: 8, edge: 6 };

/**
 * What is under pixel (x, y): a vertex, else an edge, else a face.
 *
 * Vertices and edges are picked in screen space, against a radius, and only
 * when they are not behind the nearest face along the pointer ray (`depth`,
 * the view depth of the face hit, with a tolerance of a few per cent so a
 * curve lying on its own face is never occluded by it). Faces are picked by
 * casting the ray against the tessellation in view space: every triangle is
 * projected once, so a ten-thousand-triangle B-rep is a millisecond.
 */
export function pickBrep(
  model: BrepModel,
  x: number,
  y: number,
  view: View,
  hiddenKinds: ReadonlySet<string> = new Set(),
): BrepHover {
  const face = pickFace(model, x, y, view);
  const limit = face ? face.depth * 1.03 + 1e-6 : Infinity;

  let vertex: { id: number; distance: number } | null = null;
  for (const v of model.vertices) {
    const p = projectPoint(v.point as [number, number, number], view);
    if (!p.visible || p.viewDepth > limit) continue;
    const distance = Math.hypot(p.x - x, p.y - y);
    if (distance <= PICK_RADIUS.vertex && (!vertex || distance < vertex.distance)) {
      vertex = { id: v.id, distance };
    }
  }
  if (vertex) return { kind: "vertex", id: vertex.id };

  let edge: { id: number; distance: number } | null = null;
  for (const e of model.edges) {
    if (hiddenKinds.has(e.kind)) continue;
    const n = Math.floor(e.points.length / 3);
    let previous = null as ReturnType<typeof projectPoint> | null;
    const at = (i: number) =>
      projectPoint([e.points[i * 3], e.points[i * 3 + 1], e.points[i * 3 + 2]], view);
    const first = n ? at(0) : null;
    previous = first;
    for (let i = 1; i <= (e.closed ? n : n - 1); i++) {
      const current = i === n ? first! : at(i);
      if (
        previous?.visible &&
        current.visible &&
        Math.min(previous.viewDepth, current.viewDepth) <= limit
      ) {
        const distance = pointSegmentDistance(x, y, previous.x, previous.y, current.x, current.y);
        if (distance <= PICK_RADIUS.edge && (!edge || distance < edge.distance)) {
          edge = { id: e.id, distance };
        }
      }
      previous = current;
    }
  }
  if (edge) return { kind: "edge", id: edge.id };
  return face ? { kind: "face", id: face.id } : null;
}

/** The nearest face under (x, y) and its view depth, by 2D point-in-triangle. */
export function pickFace(
  model: BrepModel,
  x: number,
  y: number,
  view: View,
): { id: number; depth: number } | null {
  const count = model.positions.length / 3;
  const sx = new Float64Array(count);
  const sy = new Float64Array(count);
  const sz = new Float64Array(count);
  const ok = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const p = projectPoint(
      [model.positions[i * 3], model.positions[i * 3 + 1], model.positions[i * 3 + 2]],
      view,
    );
    sx[i] = p.x;
    sy[i] = p.y;
    sz[i] = p.viewDepth;
    ok[i] = p.visible ? 1 : 0;
  }
  let best: { id: number; depth: number } | null = null;
  for (let t = 0; t < model.triangleFace.length; t++) {
    const a = model.triangles[t * 3];
    const b = model.triangles[t * 3 + 1];
    const c = model.triangles[t * 3 + 2];
    if (!ok[a] || !ok[b] || !ok[c]) continue;
    const denominator = (sy[b] - sy[c]) * (sx[a] - sx[c]) + (sx[c] - sx[b]) * (sy[a] - sy[c]);
    if (Math.abs(denominator) < 1e-12) continue;
    const u = ((sy[b] - sy[c]) * (x - sx[c]) + (sx[c] - sx[b]) * (y - sy[c])) / denominator;
    const v = ((sy[c] - sy[a]) * (x - sx[c]) + (sx[a] - sx[c]) * (y - sy[c])) / denominator;
    const w = 1 - u - v;
    if (u < 0 || v < 0 || w < 0) continue;
    // Depth interpolated in screen space: exact under orthographic, and
    // close enough under perspective to order faces a pixel apart.
    const depth = u * sz[a] + v * sz[b] + w * sz[c];
    if (!best || depth < best.depth) best = { id: model.triangleFace[t], depth };
  }
  return best;
}
