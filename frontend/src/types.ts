/** Shapes exchanged with the playground server. */

export interface SessionResponse {
  ok: boolean;
  token: string;
  example: string;
}

/** A sketch vertex, in both sketch-plane and world coordinates. */
export interface ConstructionVertex {
  name: string | null;
  free: boolean;
  uv: [number, number];
  world: [number, number, number];
  /** Character span of this vertex's literal, or null when not editable. */
  span: [number, number] | null;
}

export interface ConstructionPlane {
  origin: [number, number, number];
  u: [number, number, number];
  v: [number, number, number];
  normal: [number, number, number];
}

export type ConstructionKind = "profile" | "box" | "sphere" | "cylinder";

/** Placement and size of a construction primitive. */
export interface ConstructionTransform {
  position: [number, number, number];
  /** Intrinsic X, Y, Z angles in radians. */
  rotation: [number, number, number];
  dimensions: Record<string, number | number[]>;
}

/**
 * One construction object from the executed program.
 *
 * `edges` is a ready-made world-space wireframe, so the viewer draws sketches,
 * boxes, and spheres through one path without knowing their topology.
 */
export interface ConstructionNode {
  id: string;
  kind: ConstructionKind;
  name: string | null;
  /** 1-based line of the constructor call, null if unknown. */
  line: number | null;
  /** False when the literals cannot be safely rewritten. */
  editable: boolean;
  edges: [number, number, number][][];
  /** Sketch profiles only. */
  plane: ConstructionPlane | null;
  /** Sketch profiles only; primitives carry no per-vertex handles. */
  vertices: ConstructionVertex[];
  /** Primitives only. */
  transform: ConstructionTransform | null;
  /** Source spans of the primitive's keyword arguments. */
  spans: Record<string, [number, number]>;
}

export interface CompileResponse {
  ok: boolean;
  error?: string;
  sdf: string;
  preview_shader: string;
  path_shader: string;
  present_shader: string;
  construction: ConstructionNode[];
  output: string;
}

export type PatchOperation =
  | "set_vertex"
  | "insert_vertex"
  | "delete_vertex"
  | "set_value"
  | "add_primitive";

export interface PatchResponse {
  ok: boolean;
  source?: string;
  error?: string;
}

/** What the user has selected: a whole object, or one of its vertices. */
export interface Selection {
  nodeId: string;
  /** Null when the selection is the object itself, as for a primitive. */
  vertexIndex: number | null;
}

export type ToolMode = "select" | "polygon" | "box" | "sphere" | "cylinder";

/** How a gizmo drag transforms the selected primitive. */
export type GizmoMode = "translate" | "rotate";
