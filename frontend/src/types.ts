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

/**
 * How a sketch's plane is written in the source.
 *
 * `constructor` is the `SketchPlane` classmethod used (`on`, `tangent`,
 * `offset`, or `plain` for a stated origin/normal); `owner` is the variable
 * the face was read off, and `accessor`/`argument` the call that named it —
 * `body.cap('+')` reads back as `("body", "cap", "'+'")`.
 */
export interface PlaneReference {
  constructor: string | null;
  owner: string | null;
  accessor: string | null;
  argument: string | null;
}

export interface ConstructionPlane {
  origin: [number, number, number];
  u: [number, number, number];
  v: [number, number, number];
  normal: [number, number, number];
  /** Sketch profiles only: the plane expression already in the source. */
  reference?: PlaneReference | null;
}

/**
 * One analytic face of a feature — a reference, not stored geometry.
 *
 * The construction tree knows a feature's flat faces exactly (an extrusion's
 * caps sit at `plane.origin ± depth/2 · normal`, each polygon edge sweeps a
 * planar wall), and knows them *parametrically*: re-dimension the parent and
 * the face moves. That is what makes a sketch placed on one follow its
 * parent, and it is why the viewer picks against these rather than against
 * the render mesh, which is only a picture of the surface.
 */
export interface ConstructionFace {
  /** `<nodeId>:<key>`, stable across rebuilds. */
  id: string;
  /** Identity within the owner: `cap+`, `side3`, `+x`. */
  key: string;
  kind: "cap" | "side" | "planar";
  origin: [number, number, number];
  /** Outward unit normal. */
  normal: [number, number, number];
  xAxis: [number, number, number];
  yAxis: [number, number, number];
  /** World-space boundary loop; both the extent test and the highlight. */
  polygon: [number, number, number][];
  /** Distance slack for the hit test, scaled to this face's own size. */
  tolerance: number;
  /** The accessor call that reproduces the face: `cap("+")`, `side(3)`. */
  reference: { call: string; args: (string | number)[] };
  /** The feature that declared it, and the variable naming it in source. */
  owner: { kind: string; line: number; variable: string | null } | null;
  /** False when the owner has no variable, so the face cannot be written. */
  usable: boolean;
}

export type ConstraintKind =
  | "fixed"
  | "distance"
  | "horizontal"
  | "vertical"
  | "coincident"
  | "equal_length"
  | "parallel"
  | "perpendicular";

export interface ConstructionConstraint {
  kind: ConstraintKind;
  vertices: number[];
  /** Target for fixed/distance constraints; null for purely relational kinds. */
  value: number | number[] | null;
  /**
   * Position in the profile's serialized constraint list — the stable identity
   * used to delete or edit this constraint at the source level.
   */
  index?: number;
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

export interface ConstructionOperator {
  kind: "extrude" | "revolve" | "loft";
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
  /**
   * Stated physical properties in SI, or null for each one the Material does
   * not declare. Absent entirely on a server older than this payload.
   */
  physical?: Record<string, number | null>;
  /** SI unit string per physical key, e.g. `density` → `kg/m^3`. */
  units?: Record<string, string>;
  /** Which parameters are free — the ones an optimization may drive. */
  free?: Record<string, boolean>;
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
  /** Analytic faces this node's features declare; empty for curved ones. */
  faces?: ConstructionFace[];
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

/**
 * World-space line segments of the extracted dual-contour mesh.
 *
 * `sharp` edges sit across a significant dihedral angle (creases, corners,
 * CSG seams); `wire` is the rest of the wireframe.
 */
export interface MeshEdgePayload {
  wire: [number, number, number][][];
  sharp: [number, number, number][][];
  resolution: number;
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
  mesh_edges: MeshEdgePayload | null;
  studies?: StudyPayload[];
  sim_meshes?: SimMeshPayload[];
  optimizations?: OptimizationPayload[];
  output: string;
}

/** A serialized node selection, mirroring cadjoint.fem.selection describe(). */
export type StudySelection =
  | { kind: "box"; min_corner: number[]; max_corner: number[] }
  | { kind: "sphere"; center: number[]; radius: number }
  | { kind: "halfspace"; point: number[]; normal: number[] }
  | { kind: "side"; side: string; tol: number | null }
  | { kind: "predicate"; name: string }
  | { kind: "and" | "or"; operands: StudySelection[] }
  | { kind: "not"; operand: StudySelection };

export type StudyBcType = "dirichlet" | "heat_flux" | "fixed" | "traction";

/** One boundary condition of a declared study. */
export interface StudyBc {
  type: StudyBcType;
  nodes: StudySelection;
  value?: number;
  flux?: number;
  vector?: [number, number, number];
  /** False only for predicate selections, which the viewer cannot edit. */
  serializable: boolean;
  span: [number, number] | null;
}

/** The domain object a mesh or study discretizes, reported by name. */
export interface DomainEntry {
  name: string | null;
  type: string;
}

/** One ThermalStudy/ElasticStudy declared in the scene program. */
export interface StudyPayload {
  index: number;
  name: string;
  kind: "thermal" | "elastic";
  /** Null when the study solves on a declared SimMesh. */
  resolution: number | [number, number, number] | null;
  bounds: [number, number, number] | null;
  size: [number, number, number] | null;
  /** Declared SimMesh this study solves on, by name; null for implicit. */
  mesh: string | null;
  domain: DomainEntry | null;
  material: Record<string, number>;
  source?: number;
  line: number | null;
  span: [number, number] | null;
  /** False when the declaration cannot be aligned to source (loops etc.). */
  editable: boolean;
  /** Character span of the `mesh=` argument, when present in source. */
  mesh_span?: [number, number] | null;
  domain_span?: [number, number] | null;
  bcs: StudyBc[];
}

/** Element type a SimMesh extracts. */
export type MeshMethod = "hex" | "tet4" | "tet10";

/** One SimMesh declared in the scene program. */
export interface SimMeshPayload {
  kind: "mesh";
  index: number;
  name: string;
  resolution: number | [number, number, number];
  bounds: [number, number, number] | null;
  size: [number, number, number] | null;
  padding: number;
  method?: MeshMethod;
  domain: DomainEntry | null;
  line: number | null;
  span: [number, number] | null;
  editable: boolean;
}

/** min/mean/max summary of a per-element quality metric. */
export interface QualitySummary {
  min: number;
  mean: number;
  max: number;
}

/** JSON inspection report of a built mesh (SimMesh.inspect()). */
export interface MeshInspectInfo {
  name: string;
  nodes: number;
  elements: number;
  method?: MeshMethod;
  bounds: { min: number[]; max: number[] };
  grid: { origin: number[]; spacing: number[]; cells: number[] } | null;
  quality: Record<string, QualitySummary>;
}

/** POST /api/mesh_inspect: build a declared mesh and report its quality. */
export interface MeshInspectResponse {
  ok: boolean;
  kind?: "mesh_inspect";
  name?: string;
  /** The scalar field carried by `mesh.scalars` (scaled_jacobian). */
  field?: string;
  info?: MeshInspectInfo;
  mesh?: SimulationMeshPayload;
  /** Per-vertex min scaled Jacobian, same order as `mesh.positions`. */
  quality_scalars?: number[];
  error?: string;
  output?: string;
}

/** One Optimization(...) declared in the scene program. */
export interface OptimizationPayload {
  kind: "optimization";
  index: number;
  name: string;
  steps: number;
  learning_rate: number;
  method: string;
  /** Names of the free parameters the run drives. */
  parameters: string[];
  /** Name of the objective the declaration minimizes. */
  objective: string;
  /** Study-backed objectives: the study driven each step, and its metric. */
  study?: string | null;
  metric?: string | null;
  remesh_every?: number | null;
  line: number | null;
  span: [number, number] | null;
  editable: boolean;
}

/** One recorded optimizer step. */
export interface OptimizeHistoryEntry {
  step: number;
  objective: number;
  grad_norm: number;
}

/** One replayable trajectory frame (step 0 is the initial state). */
export interface OptimizeTrajectoryEntry {
  step: number;
  objective: number;
  parameters: Record<string, number | number[]>;
}

export interface OptimizeRequest {
  source: string;
  name: string;
  steps?: number;
}

/** The final design's solved field, attached to study-backed optimize runs. */
export interface OptimizeSimulateBlock {
  field?: string | null;
  mesh?: SimulationMeshPayload;
  result?: SimulationResultSummary;
  mesh_info?: MeshInspectInfo | null;
}

/** POST /api/optimize: run a declared optimization to completion. */
export interface OptimizeResponse {
  ok: boolean;
  kind?: "optimize";
  name?: string;
  /** The program with the optimized parameter literals written back. */
  source?: string;
  history?: OptimizeHistoryEntry[];
  /** Parameter snapshots along the run, for the replay player (≤100). */
  trajectory?: OptimizeTrajectoryEntry[];
  parameters?: Record<string, number | number[]>;
  initial?: Record<string, number | number[]>;
  /** Study-backed runs: the optimized design's solved field, ready to show. */
  simulate?: OptimizeSimulateBlock;
  error?: string;
  output?: string;
}

/** A viewport-picked BC region, pre-filling the add-BC builder. */
export type BcProposal =
  | { kind: "sphere"; center: [number, number, number]; radius: number }
  | { kind: "box"; min: [number, number, number]; max: [number, number, number] };

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
  | "add_revolution"
  | "add_loft"
  | "add_constraint"
  | "delete_constraint"
  | "set_constraint_value"
  | "solve_sketch"
  | "delete_object"
  | "add_study"
  | "delete_study"
  | "add_study_bc"
  | "delete_study_bc"
  | "set_study_value"
  | "add_mesh"
  | "delete_mesh"
  | "set_mesh_value"
  | "set_optimization_value"
  | "delete_optimization"
  | "set_sketch_plane";

/**
 * The plane a `set_sketch_plane` patch plants a sketch on.
 *
 * `owner` is the 1-based source line of the feature call that declared the
 * face, which is how the server finds the variable to write. `tangent` is the
 * fallback for a surface with no analytic face — it reads the plane off the
 * solid's own gradient at the picked point.
 */
export type SketchPlaneReference =
  | { kind: "cap"; owner: number; sign: "+" | "-" }
  | { kind: "side"; owner: number; edge: number }
  | { kind: "face"; owner: number; key: string }
  | { kind: "tangent"; owner: number; near: [number, number, number] };

export interface PatchResponse {
  ok: boolean;
  source?: string;
  error?: string;
}

/** Lazy mesh-edge extraction, requested only while a mesh overlay is on. */
export interface MeshResponse {
  ok: boolean;
  mesh_edges?: MeshEdgePayload | null;
  error?: string;
}

export interface SceneListResponse {
  ok: boolean;
  files?: string[];
  error?: string;
}

export interface SceneLoadResponse {
  ok: boolean;
  name?: string;
  source?: string;
  error?: string;
}

export interface SceneSaveResponse {
  ok: boolean;
  name?: string;
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
  | "horizontal"
  | "vertical"
  | "coincident"
  | "parallel"
  | "perpendicular"
  | "box"
  | "sphere"
  | "cylinder"
  | "face";

/** How a gizmo drag transforms the selected construction object. */
export type GizmoMode = "translate" | "rotate" | "scale";

/** One boundary face group of the simulation hex mesh — a BC target. */
export interface SimulationFaceGroup {
  /** Gradient-axis id such as `+x` or `-z`. */
  id: string;
  axis: "x" | "y" | "z";
  side: "+" | "-";
  center: [number, number, number];
  area: number;
  faces: number;
  /** This group's triangle range in the payload's index buffer. */
  start: number;
  count: number;
}

/** Indexed boundary-surface triangles with one scalar per vertex. */
export interface SimulationMeshPayload {
  /** Flat xyz positions, three floats per vertex. */
  positions: number[];
  /** One scalar per vertex (temperature, von Mises, or zero for a probe). */
  scalars: number[];
  /** Flat triangle list into the compacted vertex array. */
  indices: number[];
  /** Unique element boundary-face edges, flat index pairs (no diagonals). */
  edges?: number[];
  groups: SimulationFaceGroup[];
  /** Min and max of the scalar field. */
  range: [number, number];
  vertex_count: number;
  /**
   * Solved payloads carry the full per-vertex field catalog here (the
   * backend's `_result_field_payload` attaches them to the mesh payload):
   * every nodal field for display switching, its `[min, max]` per entry,
   * and — for elastic solves — raw displacement vectors for a warped view.
   */
  fields?: Record<string, number[]>;
  ranges?: Record<string, [number, number]>;
  displacements?: [number, number, number][];
}

/** `probe` only meshes and returns the face-group catalog; the rest solve. */
export type SimulationKind = "probe" | "thermal" | "elastic";

/** One boundary condition, targeting a face group by id. */
export interface SimulationBc {
  group: string;
  type: "dirichlet" | "traction";
  value: number | [number, number, number];
}

export interface SimulateRequest {
  source: string;
  kind: SimulationKind;
  resolution: number;
  bcs: SimulationBc[];
  material: Record<string, number>;
}

/** Run a study declared in the scene program, resolved server-side by name. */
export interface SimulateStudyRequest {
  source: string;
  kind: "study";
  name: string;
}

/** JSON summary of a solved study (SimulationResult.describe()). */
export interface SimulationResultSummary {
  name: string;
  kind: "thermal" | "elastic";
  /** The display field carried by the response mesh scalars. */
  field: string;
  /** SimMesh name the study solved on, or null for an implicit mesh. */
  mesh: string | null;
  nodes: number;
  elements: number;
  range: [number, number];
  fields: Record<string, QualitySummary>;
}

export interface SimulateResponse {
  ok: boolean;
  kind?: SimulationKind | "study";
  /** Which nodal field `mesh.scalars` carries; null for a probe. */
  field?: string | null;
  mesh?: SimulationMeshPayload;
  /** The solved study's description, echoed on `kind: "study"` responses. */
  study?: StudyPayload;
  /** Solved-result summary: field ranges, element counts, source mesh. */
  result?: SimulationResultSummary;
  /** Inspection report of the mesh the study solved on. */
  mesh_info?: MeshInspectInfo | null;
  /**
   * Per-vertex nodal fields, ranges, and displacements historically sat at
   * the top level; the server ships them on `mesh` now (see
   * SimulationMeshPayload) and readers coalesce both homes.
   */
  fields?: Record<string, number[]>;
  ranges?: Record<string, [number, number]>;
  displacements?: [number, number, number][];
  error?: string;
  /** `fem_unavailable` when the jax-fem extra is not installed (HTTP 501). */
  error_kind?: string;
  output?: string;
}

/** What a click in the viewport picks. */
export type SelectionMode = "object" | "vertex";

// ── editor intelligence ─────────────────────────────────────────────────────
//
// Three endpoints read the editor's Python without running it: `/api/lint`
// (ruff, plus the last compile traceback), `/api/complete` and
// `/api/signature` (jedi). They share one coordinate convention, which is
// also CodeMirror's: **lines are 1-based, columns are 0-based**, so a
// position becomes an offset with `doc.line(from_line).from + from_col`.

export type LintSeverity = "error" | "warning" | "info";

/** Where a ruff autofix came from, i.e. how far it may be trusted. */
export type FixApplicability = "safe" | "unsafe" | "display";

/** One replacement of a ruff autofix, in the shared line/column convention. */
export interface LintFixEdit {
  from_line: number;
  from_col: number;
  to_line: number;
  to_col: number;
  content: string;
}

/** A ruff autofix: a label plus the edits that apply it in one go. */
export interface LintFix {
  message: string;
  applicability: FixApplicability;
  edits: LintFixEdit[];
}

/**
 * One diagnostic.
 *
 * `source` is `"ruff"` for static analysis and `"runtime"` for the traceback
 * of the last failed `/compile` of this exact text — the latter names the
 * line that actually blew up, which no static analyser can produce.
 */
export interface LintDiagnostic {
  from_line: number;
  from_col: number;
  to_line: number;
  to_col: number;
  severity: LintSeverity;
  message: string;
  code: string;
  source: "ruff" | "runtime";
  /** Rule documentation, rendered as a "learn more" link. */
  url: string | null;
  fix: LintFix | null;
}

export interface LintResponse {
  ok: boolean;
  /** True when a remembered traceback contributed one of the diagnostics. */
  runtime?: boolean;
  diagnostics?: LintDiagnostic[];
  error?: string;
}

/** One jedi completion, already shaped like CodeMirror's `Completion`. */
export interface CompletionItem {
  label: string;
  /** CodeMirror completion type: `function`, `class`, `property`, … */
  type: string;
  /** Jedi's own kind, shown beside the label. */
  detail: string;
  /** Signature and docstring; present only for the head of the list. */
  info: string | null;
  apply: string;
}

export interface CompleteResponse {
  ok: boolean;
  /** Caret line the completions were computed at (1-based). */
  from_line?: number;
  /** Where the already-typed prefix starts (0-based). */
  from_column?: number;
  truncated?: boolean;
  completions?: CompletionItem[];
  error?: string;
}

export interface SignatureParameter {
  name: string;
  label: string;
}

/** One call signature the caret sits inside. */
export interface SignatureInfo {
  name: string;
  label: string;
  /** Index of the argument being typed, or null between calls. */
  active_parameter: number | null;
  parameters: SignatureParameter[];
  documentation: string | null;
}

export interface SignatureResponse {
  ok: boolean;
  signatures?: SignatureInfo[];
  error?: string;
}
