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

export interface ConstructionConstraint {
  kind: "fixed" | "distance";
  vertices: number[];
  value: number | number[];
}

/** A constraint relating whole construction objects rather than sketch points. */
export interface ConstructionRelation {
  kind: "fixed" | "distance";
  nodes: string[];
  value: number | number[];
}

export type ConstraintSolverMethod = "newton" | "adam" | "sgd";

/** Diagnostics captured from one source-level satisfy_constraints call. */
export interface ConstraintSolverRun {
  node: string | null;
  method: ConstraintSolverMethod;
  iterations: number;
  losses: number[];
}

/** A source-computed proof that gradients traverse the active CAD pipeline. */
export interface DifferentiabilityDemo {
  pipeline: string;
  metric: string;
  value: number;
  parameter_count: number;
  sensitivities: {
    parameter: string;
    value: number;
  }[];
}

export interface ConstructionOperator {
  kind: "extrude" | "revolve";
  line: number;
}

/** A named Python Material definition shown in the material browser. */
export interface MaterialDefinition {
  id: string;
  /** Stable Python variable used when assigning the material to an object. */
  name: string;
  line: number;
  editable: boolean;
  color: [number, number, number];
  roughness: number;
  metallic: number;
  opacity: number;
  ior: number;
  reflectivity: number;
  spans: Record<string, [number, number]>;
}

export type ConstructionKind = "profile" | "box" | "sphere" | "cylinder";

/** Placement and size of a construction primitive. */
export interface ConstructionTransform {
  position: [number, number, number];
  /** Intrinsic X, Y, Z angles in radians. */
  rotation: [number, number, number];
  dimensions: Record<string, number | number[]>;
  /** Line of the call that owns the placement — a plane, for a sketch. */
  line: number;
  /** That call's name, e.g. `box` or `SketchPlane`. */
  call: string;
  /** Keyword holding the position: `position`, or `origin` for a plane. */
  positionArgument: string;
  /** False for sketches, whose orientation is a normal rather than angles. */
  canRotate: boolean;
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
  /** Constraints attached to sketch vertices. */
  constraints: ConstructionConstraint[];
  /** Construction-history operations consuming this sketch. */
  operators: ConstructionOperator[];
  /** Named material assigned to this primitive or the profile's extrusion. */
  material: string | null;
}

export interface CompileResponse {
  ok: boolean;
  error?: string;
  sdf: string;
  preview_shader: string;
  path_shader: string;
  present_shader: string;
  construction: ConstructionNode[];
  relations: ConstructionRelation[];
  solver_runs: ConstraintSolverRun[];
  materials: MaterialDefinition[];
  differentiability: DifferentiabilityDemo | null;
  output: string;
}

export type PatchOperation =
  | "set_vertex"
  | "insert_vertex"
  | "delete_vertex"
  | "set_value"
  | "add_primitive"
  | "add_material"
  | "assign_material"
  | "add_sketch"
  | "add_extrusion"
  | "add_constraint"
  | "solve_sketch"
  | "delete_object";

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

export type ToolMode =
  | "select"
  | "sketch"
  | "polygon"
  | "distance"
  | "box"
  | "sphere"
  | "cylinder";

/** How a gizmo drag transforms the selected construction object. */
export type GizmoMode = "translate" | "rotate" | "scale";

/** What a click in the viewport picks. */
export type SelectionMode = "object" | "vertex";
