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

/** One `PolygonProfile` from the executed program. */
export interface ConstructionProfile {
  id: string;
  name: string | null;
  /** 1-based line of the `PolygonProfile(...)` call, null if unknown. */
  line: number | null;
  /** False for sketches built in a loop or from variables. */
  editable: boolean;
  plane: ConstructionPlane;
  vertices: ConstructionVertex[];
}

export interface CompileResponse {
  ok: boolean;
  error?: string;
  sdf: string;
  preview_shader: string;
  path_shader: string;
  present_shader: string;
  construction: ConstructionProfile[];
  output: string;
}

export type PatchOperation = "set_vertex" | "insert_vertex" | "delete_vertex";

export interface PatchRequest {
  source: string;
  op: PatchOperation;
  line: number;
  index: number;
  xy?: [number, number];
}

export interface PatchResponse {
  ok: boolean;
  source?: string;
  error?: string;
}

/** Which sketch vertex the user has selected, if any. */
export interface Selection {
  profileId: string;
  vertexIndex: number;
}

export type ToolMode = "select" | "add";
